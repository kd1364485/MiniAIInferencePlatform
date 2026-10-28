"""Phase 7 M3: infer_server を provider(cpu / cuda / trt)ごとに起動して測り、表にする。

使い方:
  python tools/server_bench.py <infer_server> <infer_bench> <model.onnx> [絞り込み=全部] [秒/条件=5]
    絞り込み: 設定名に含まれる文字列のカンマ区切り(例: trt-fp16,cpu)
  PYTHON 環境変数: CUDA 13 / cuDNN / TensorRT の pip パッケージが入った Python (run_gpu_env.sh が使う)

1 条件ごとにやること:
  1) サーバーを起動し、起動完了のメッセージ(ウォームアップ後。TensorRT はエンジン作成を含む)を待つ
  2) 測定の直前に、GPU の「待機時の使用率」を1秒サンプルする(Windows 側の使用分の目安)
  3) infer_bench(dummy 入力)を回しながら、nvidia-smi を 100ms ごとにサンプルする(使用率・VRAM・SM クロック)
  4) サーバーを SIGTERM で止め、終了時に出る統計(平均バッチ、Run()/件、キュー待ち)を読む

TensorRT のエンジンは作業フォルダ(既定 ~/.cache/infer_server_bench)にキャッシュされ、2回目以降は速い。
"""
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_SH = os.path.join(HERE, "run_gpu_env.sh")

if len(sys.argv) < 4:
    sys.exit(__doc__)
SERVER, BENCH, MODEL = (os.path.abspath(p) for p in sys.argv[1:4])
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else None
SECONDS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
WORK = os.path.expanduser("~/.cache/infer_server_bench")
os.makedirs(WORK, exist_ok=True)

# (名前, provider, I/O worker, 推論 worker, max_batch)   推論 worker=0 はインライン
CONFIGS = [
    ("cpu-inline-w8-b8",     "cpu",      8, 0, 8),
    ("cpu-q-io2-inf6-b32",   "cpu",      2, 6, 32),
    ("cuda-q-io2-inf1-b32",  "cuda",     2, 1, 32),
    ("cuda-q-io2-inf2-b32",  "cuda",     2, 2, 32),
    ("cuda-q-io2-inf4-b32",  "cuda",     2, 4, 32),
    ("cuda-q-io2-inf1-b128", "cuda",     2, 1, 128),
    ("cuda-q-io2-inf2-b128", "cuda",     2, 2, 128),
    ("trt-q-io2-inf1-b128",  "trt",      2, 1, 128),
    ("trt-fp16-q-io2-inf1-b32",  "trt-fp16", 2, 1, 32),
    ("trt-fp16-q-io2-inf2-b32",  "trt-fp16", 2, 2, 32),
    ("trt-fp16-q-io2-inf1-b128", "trt-fp16", 2, 1, 128),
    ("trt-fp16-q-io2-inf2-b128", "trt-fp16", 2, 2, 128),
    # I/O worker を変える(GPU が余っているとき、上限が受け付け側にあるかを確かめる)
    ("trt-fp16-q-io1-inf1-b128", "trt-fp16", 1, 1, 128),
    ("trt-fp16-q-io4-inf1-b128", "trt-fp16", 4, 1, 128),
    ("trt-fp16-q-io8-inf1-b128", "trt-fp16", 8, 1, 128),
]
# 同時接続数。環境変数 SERVER_BENCH_CONNS=100,200 で変えられる
CONNS = [int(c) for c in os.environ.get("SERVER_BENCH_CONNS", "16,100").split(",")]


def start_sampler():
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm", "--format=csv,noheader,nounits",
           "-lms", "100"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def stop_sampler(p):
    p.terminate()
    out = p.communicate(timeout=5)[0]
    rows = []
    for line in out.strip().splitlines():
        try:
            rows.append([float(x) for x in line.split(",")])
        except ValueError:
            pass
    return rows


def mean(v):
    return sum(v) / len(v) if v else None


def stat(text, key):
    """'key=1.23' の値を探す(ログの最後に出るものを優先)。"""
    val = None
    for tok in text.replace("  ", " ").split():
        if tok.startswith(key + "="):
            try:
                val = float(tok.split("=", 1)[1])
            except ValueError:
                pass
    return val


def mem_used_now():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def run_one(name, provider, io, infer, batch, conns):
    log_path = os.path.join(WORK, f"{name}.log")
    env = dict(os.environ)
    base_mem = mem_used_now()                      # VRAM の基準は「サーバーを起動する前」に取る
    with open(log_path, "w") as log:
        srv = subprocess.Popen(
            # bash 経由で起動する: 実行権限のビットは Windows の git 経由で失われやすいので、それに頼らない
            ["bash", ENV_SH, SERVER, MODEL, str(io), "1", "0", str(batch), str(infer), provider],
            stdout=log, stderr=subprocess.STDOUT, cwd=WORK, env=env)
        ready = False
        for _ in range(300):                       # 最大 300 秒(TensorRT の初回エンジン作成を含む)
            time.sleep(1)
            if srv.poll() is not None:
                break
            with open(log_path) as f:
                if any(l.startswith("infer_server:") for l in f):
                    ready = True
                    break
        if not ready:
            srv.kill()
            with open(log_path) as f:
                return {"name": name, "conns": conns, "error": f.read()[-400:]}

        s = start_sampler(); time.sleep(1.0); idle = stop_sampler(s)       # 待機時(サーバー起動後・測定前)
        smp = start_sampler()
        bench = subprocess.run([BENCH, str(conns), str(SECONDS), "dummy"], capture_output=True, text=True)
        rows = stop_sampler(smp)
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
    text = open(log_path).read()
    b = bench.stdout
    r = {"name": name, "conns": conns, "batch": batch, "rps": None}
    # infer_bench の出力: "...req/s: 123.4" と "latency(us): p50=.. p95=.. p99=.. max=.."
    for line in b.splitlines():
        if "req/s:" in line:
            r["rps"] = float(line.split("req/s:")[1].split()[0])
        if line.startswith("latency"):
            for tok in line.replace("latency(us):", "").split():
                k, v = tok.split("=")
                r[k] = float(v)
    r["avg_batch"] = stat(text, "avg_batch")
    r["run_per_req"] = stat(text, "Run()/request")
    r["qwait"] = stat(text, "avg_wait")
    r["util"] = mean([x[0] for x in rows]) if provider != "cpu" else None
    r["util_idle"] = mean([x[0] for x in idle]) if provider != "cpu" else None
    r["clk"] = mean([x[2] for x in rows]) if provider != "cpu" else None
    r["vram"] = (max(x[1] for x in rows) - base_mem) if (rows and provider != "cpu") else None
    return r


def main():
    results = []
    for conns in CONNS:
        for name, provider, io, infer, batch in CONFIGS:
            if ONLY and not any(o in name for o in ONLY):
                continue
            print(f"... {name} conns={conns}", file=sys.stderr, flush=True)
            results.append(run_one(name, provider, io, infer, batch, conns))
            time.sleep(1.0)

    f = lambda v, fmt: ("-" if v is None else format(v, fmt))
    print(f"\n{'conns':>5} {'config':<26}{'req/s':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}"
          f"{'平均バッチ':>9}{'Run/件':>8}{'キュー待ち':>9}{'util%':>7}{'(待機)':>7}{'clk':>6}{'VRAM+':>7}")
    for r in results:
        if "error" in r:
            print(f"{r['conns']:>5} {r['name']:<26} 起動失敗: {r['error']!r}")
            continue
        print(f"{r['conns']:>5} {r['name']:<26}{f(r.get('rps'), '.0f'):>8}{f(r.get('p50'), '.0f'):>8}"
              f"{f(r.get('p95'), '.0f'):>8}{f(r.get('p99'), '.0f'):>8}{f(r.get('max'), '.0f'):>8}"
              f"{f(r['avg_batch'], '.1f'):>9}{f(r['run_per_req'], '.0f'):>8}{f(r['qwait'], '.0f'):>9}"
              f"{f(r['util'], '.0f'):>7}{f(r['util_idle'], '.0f'):>7}{f(r['clk'], '.0f'):>6}{f(r['vram'], '.0f'):>7}")


if __name__ == "__main__":
    main()
