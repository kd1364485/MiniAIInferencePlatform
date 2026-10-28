"""GPU の使用率・VRAM などを Prometheus 形式で出す、小さな exporter。標準ライブラリだけで動く。

なぜ自作か: NVIDIA 公式の dcgm-exporter は、WSL2 では動かない可能性が高い(DCGM が WSL2 を想定していない)。
  nvidia-smi は WSL2 でも動く(NVML 経由)ので、取得のたびに nvidia-smi を呼んで、その出力を /metrics の形に直す。

仕組み: GET /metrics のたびに  nvidia-smi --query-gpu=... --format=csv,noheader,nounits  を1回実行する(約 0.1 秒)。
  Prometheus が 5 秒ごとに取りに来るなら、負荷は無視できる。

注意:
  - utilization.gpu は「直近のサンプル期間(約 1 秒)に、GPU が何かのカーネルを実行していた時間の割合」。
    取得した瞬間の値なので、5 秒ごとの取得では、その間の変動を取りこぼす(平均ではない)
  - コンテナの中で nvidia-smi が使えるのは、NVIDIA Container Toolkit が起動時に注入するため(utility の権限が要る)
  - 認証なし。Docker のネットワークの内側だけで使う(ポートを外に公開しない)
"""
import http.server
import os
import subprocess
import time

PORT = int(os.environ.get("GPU_EXPORTER_PORT", "9400"))
QUERY = ["index", "name", "utilization.gpu", "memory.used", "memory.total",
         "temperature.gpu", "power.draw", "clocks.sm"]


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def collect():
    """nvidia-smi を1回呼んで、/metrics の本文を作る。失敗したら gpu_exporter_up 0 だけを返す。"""
    t0 = time.time()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(QUERY), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
    except (subprocess.SubprocessError, OSError) as e:
        return ("# HELP gpu_exporter_up 1 if nvidia-smi worked.\n# TYPE gpu_exporter_up gauge\n"
                "gpu_exporter_up 0\n# error: %s\n" % esc(str(e)).replace("\n", " "))

    lines = ["# HELP gpu_exporter_up 1 if nvidia-smi worked.", "# TYPE gpu_exporter_up gauge", "gpu_exporter_up 1"]
    rows = [[c.strip() for c in l.split(",")] for l in out.strip().splitlines() if l.strip()]

    def num(s):
        try:
            return float(s)
        except ValueError:          # "[N/A]" など(電力の取得できない GPU など)
            return None

    metrics = [   # (名前, 種類, 説明, 列の位置, 単位の変換)
        ("gpu_utilization_ratio", "gauge", "GPU utilization (0..1) over the last sample period.", 2, 0.01),
        ("gpu_memory_used_bytes", "gauge", "GPU memory in use, in bytes (includes other processes, e.g. Windows apps).", 3, 1024 * 1024),
        ("gpu_memory_total_bytes", "gauge", "GPU memory total, in bytes.", 4, 1024 * 1024),
        ("gpu_temperature_celsius", "gauge", "GPU temperature.", 5, 1),
        ("gpu_power_watts", "gauge", "GPU power draw.", 6, 1),
        ("gpu_sm_clock_hertz", "gauge", "SM clock frequency in Hz.", 7, 1e6),
    ]
    for name, typ, helptext, col, scale in metrics:
        lines += ["# HELP %s %s" % (name, helptext), "# TYPE %s %s" % (name, typ)]
        for r in rows:
            v = num(r[col])
            if v is not None:
                lines.append('%s{gpu="%s",name="%s"} %s' % (name, esc(r[0]), esc(r[1]), repr(v * scale)))
    lines += ["# HELP gpu_exporter_scrape_seconds Time spent running nvidia-smi.",
              "# TYPE gpu_exporter_scrape_seconds gauge", "gpu_exporter_scrape_seconds %.4f" % (time.time() - t0)]
    return "\n".join(lines) + "\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/metrics":
            body, code, ctype = collect(), 200, "text/plain; version=0.0.4; charset=utf-8"
        elif path == "/healthz":
            body, code, ctype = "ok\n", 200, "text/plain"
        else:
            body, code, ctype = "not found\n", 404, "text/plain"
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):      # 取得のたびのアクセスログは出さない(うるさいので)
        pass


if __name__ == "__main__":
    print("gpu_exporter: listening on :%d" % PORT, flush=True)
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
