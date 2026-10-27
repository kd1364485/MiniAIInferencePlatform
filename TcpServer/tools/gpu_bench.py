"""Phase 7 M1: GPU 比較基盤。同じ MLP を PyTorch / ONNX Runtime(CPU・CUDA)で測り、表にする。

使い方:  python tools/gpu_bench.py [model.onnx=models/mlp_dyn.onnx] [秒/条件=2]
  ※ models/mlp_dyn.onnx は tools/export_mlp.py で作る。PyTorch 側は同じ seed で同じ重みを作り直す。
  ※ ORT の CUDA プロバイダが CUDA 13 系ライブラリを見つけられるよう、必要な検索パスは
     スクリプトが自動で足して自分自身を再実行する(LD_LIBRARY_PATH は起動時にしか効かないため)。
  ※ 測定の前に、AC 電源に繋ぎ、Windows の電源プランを高パフォーマンスにし、他のアプリを閉じること。

何を測るか(原本の Phase 7: latency / throughput / VRAM / GPU utilization):
  latency    : 1回の呼び出しの p50 / p95 / p99 (us)。「CPU 配列を渡して CPU 配列で受け取る」まで含む
  throughput : batch x 回数 / 経過時間 (samples/s)。1本のストリームを回し続けたときの値
  VRAM       : 測定中の nvidia-smi の memory.used(最大) - その backend を作る直前の値。
               backend ごとに別プロセスで測るので、CUDA コンテキスト(数十〜百数十MB)も含む。
               ※ Windows 側のアプリも同じ GPU を使うので、数十MB の揺れは誤差
  GPU util   : 測定中の utilization.gpu の平均(100ms ごと)。同じ GPU を Windows 側も使っているので、
               測定前に1.5秒サンプルした「待機時の値」を併記する(差が自分の分の目安)
  SM clock   : 測定中の SM クロック平均。アイドル時は 210MHz まで下がるので、低いままなら
               GPU が省電力状態のまま測っている = 結果が不当に悪い可能性がある

公平にするための注意:
  - ORT の CUDA プロバイダは既定で TF32(Ampere 以降の行列積で精度を少し落として速くする)が有効。
    PyTorch の FP32 は既定で TF32 無効。そこで ORT は use_tf32=0 (厳密な FP32) を基準にし、
    TF32 のままのものを別行 (ort-cuda-tf32) で示す。「対torch差」に精度の差が出る。
  - 「要求した」プロバイダと「実際に使われた」プロバイダを確認し、GPU が使われていなければ止める
    (黙って CPU にフォールバックしたまま測らない)。ただし ORT の一覧には CPU が常に末尾に付く。
"""
import os
import site
import subprocess
import sys
import time


def ensure_ld_path():
    """CUDA 13 / cuDNN / TensorRT のライブラリの場所を LD_LIBRARY_PATH に足して再実行する。"""
    dirs = []
    for base in site.getsitepackages():
        for rel in ("nvidia/cu13/lib", "nvidia/cudnn/lib", "tensorrt_libs"):
            d = os.path.join(base, rel)
            if os.path.isdir(d):
                dirs.append(d)
    if os.path.isdir("/usr/lib/wsl/lib"):
        dirs.append("/usr/lib/wsl/lib")          # WSL の libcuda.so.1
    cur = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [d for d in dirs if d not in cur]
    if missing and os.environ.get("_GPU_BENCH_REEXEC") != "1":
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + cur)
        os.environ["_GPU_BENCH_REEXEC"] = "1"
        os.execv(sys.executable, sys.orig_argv)  # orig_argv: -I などのインタプリタ引数も保つ


ensure_ld_path()

import gc  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/mlp_dyn.onnx"
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
BATCHES = [1, 8, 32, 128, 256]


class MLP(nn.Module):  # tools/export_mlp.py と同じ構成
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.net(x)


def torch_model():
    torch.manual_seed(0)     # export_mlp.py と同じ手順 → 同じ重み
    return MLP().eval()


def check_input():
    return np.random.RandomState(0).rand(8, 1, 28, 28).astype(np.float32)


# ---- nvidia-smi のサンプリング (utilization, memory.used, sm clock, power) ----
def start_sampler():
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw",
           "--format=csv,noheader,nounits", "-lms", "100"]
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


def mem_used_now():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def mean(v):
    return sum(v) / len(v) if v else None


# ---- backend の定義: (名前, 作る関数, GPU か) ----
def make_ort(providers, expect):
    def build():
        sess = ort.InferenceSession(MODEL, providers=providers)
        actual = sess.get_providers()
        if actual[0] != expect:
            sys.exit(f"★ {expect} を要求したが実際は {actual}。黙ったフォールバックなので測らない")

        def run(x):
            return sess.run(["output"], {"input": x})[0]
        return run, sess
    return build


def make_torch_cuda():
    def build():
        m = torch_model().cuda()

        def run(x):
            with torch.inference_mode():
                return m(torch.from_numpy(x).cuda()).cpu().numpy()   # .cpu() で同期も済む
        return run, m
    return build


def make_trt(fp16):
    """ネイティブの TensorRT: ONNX からエンジンを作り、直接動かす(M2)。
    エンジンは「バッチ 1〜256 の1つの最適化プロファイル」で作る(最適化は batch=128 を狙う)。
    FP32 は厳密にするため TF32 を外す(TensorRT は Ampere で TF32 が既定で有効)。
    入出力は他の backend と同じく「CPU 配列を渡して CPU 配列で受け取る」(転送を含める)。"""
    def build():
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()           # TensorRT 10 は明示バッチが既定
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(os.path.abspath(MODEL)):   # 外部の重みファイル(.data)も読める
            sys.exit("ONNX の解析に失敗: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 1, 28, 28), (128, 1, 28, 28), (256, 1, 28, 28))   # min / opt / max
        config.add_optimization_profile(profile)
        t0 = time.time()
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            sys.exit("エンジンの作成に失敗")
        print(f"  [trt] エンジン作成 {time.time() - t0:.1f}s fp16={fp16}", file=sys.stderr)
        engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        context = engine.create_execution_context()
        bufs = {}
        current = [None]                              # いま context に設定してある batch

        def run(x):
            b = x.shape[0]
            if b not in bufs:                         # batch ごとに GPU 側の入出力バッファを1回だけ確保
                bufs[b] = (torch.empty((b, 1, 28, 28), dtype=torch.float32, device="cuda"),
                           torch.empty((b, 10), dtype=torch.float32, device="cuda"))
            xin, yout = bufs[b]
            if current[0] != b:                       # 形とアドレスは batch が変わったときだけ設定する
                context.set_input_shape("input", (b, 1, 28, 28))   # (毎回呼ぶと Python の手間が増えて不公平)
                context.set_tensor_address("input", xin.data_ptr())
                context.set_tensor_address("output", yout.data_ptr())
                current[0] = b
            xin.copy_(torch.from_numpy(x))            # CPU → GPU
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            return yout.cpu().numpy()                 # GPU → CPU(ここで同期も済む)
        return run, (engine, context, bufs)
    return build


def make_ort_trt(fp16):
    """ORT の TensorRT 連携(M3 の C++ サーバーが使う経路)。プロファイルでバッチ 1〜256 を1つのエンジンで受ける。
    ORT の TensorRT 連携には TF32 を切る公開オプションが見当たらない(未確認)ので、
    「対torch差」が 5e-5 前後なら TF32 が効いている、と読む。"""
    opts = {
        "trt_fp16_enable": fp16,
        "trt_profile_min_shapes": "input:1x1x28x28",
        "trt_profile_opt_shapes": "input:128x1x28x28",
        "trt_profile_max_shapes": "input:256x1x28x28",
    }
    return make_ort([("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"], "TensorrtExecutionProvider")


BACKENDS = [
    ("pytorch-cuda",   make_torch_cuda(), True),
    ("ort-cuda",       make_ort([("CUDAExecutionProvider", {"use_tf32": "0"})], "CUDAExecutionProvider"), True),
    ("ort-cuda-tf32",  make_ort([("CUDAExecutionProvider", {"use_tf32": "1"})], "CUDAExecutionProvider"), True),
    ("ort-cpu",        make_ort(["CPUExecutionProvider"], "CPUExecutionProvider"), False),
    # ---- M2: TensorRT ----
    ("trt-fp32",       make_trt(False), True),
    ("trt-fp16",       make_trt(True), True),
    ("ort-trt-fp32",   make_ort_trt(False), True),
    ("ort-trt-fp16",   make_ort_trt(True), True),
]
# 開発中に一部だけ測りたいとき:  GPU_BENCH_ONLY=trt-fp32,trt-fp16 python tools/gpu_bench.py
_only = os.environ.get("GPU_BENCH_ONLY")
if _only:
    BACKENDS = [b for b in BACKENDS if b[0] in _only.split(",")]


def pct(sorted_us, p):
    return sorted_us[min(int(len(sorted_us) * p / 100), len(sorted_us) - 1)]


def child(name):
    """1つの backend だけを、このプロセス専用の CUDA コンテキストで測り、結果を ROW 行で出す。"""
    build, is_gpu = {n: (b, g) for n, b, g in BACKENDS}[name]
    x = check_input()
    ref = torch_model()(torch.from_numpy(x)).detach().numpy()

    util_idle = clk_idle = base_mem = None
    if is_gpu:   # 待機時の使用率・クロック(Windows 側の使用分を含む)と、作る前の VRAM
        s = start_sampler()
        time.sleep(1.5)
        idle = stop_sampler(s)
        util_idle, clk_idle = mean([r[0] for r in idle]), mean([r[2] for r in idle])
        base_mem = mem_used_now()

    run, keep = build()
    diff = float(np.abs(run(x) - ref).max())     # この backend が PyTorch(CPU) と同じ答えを返すか
    for b in BATCHES:
        xb = np.random.rand(b, 1, 28, 28).astype(np.float32)
        for _ in range(30):
            run(xb)                              # ウォームアップ(クロックを上げる目的も兼ねる)
        sampler = start_sampler() if is_gpu else None
        lat, t_all = [], time.perf_counter()
        while time.perf_counter() - t_all < SECONDS:
            t0 = time.perf_counter()
            run(xb)
            lat.append((time.perf_counter() - t0) * 1e6)
        elapsed = time.perf_counter() - t_all
        smp = stop_sampler(sampler) if sampler else []
        lat.sort()
        row = {
            "backend": name, "batch": b, "n": len(lat),
            "p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99),
            "sps": b * len(lat) / elapsed,
            "util": mean([r[0] for r in smp]), "util_idle": util_idle,
            "clk": mean([r[2] for r in smp]), "clk_idle": clk_idle,
            "vram": (max(r[1] for r in smp) - base_mem) if smp else None,
            "diff": diff,
        }
        print("ROW " + json.dumps(row), flush=True)
    del run, keep
    gc.collect()


def main():
    if len(sys.argv) > 3:                         # 子プロセス: backend 名が第3引数
        return child(sys.argv[3])

    print(f"model={MODEL}  torch={torch.__version__}  ort={ort.__version__}  秒/条件={SECONDS}")
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,temperature.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    x = check_input()
    ref = torch_model()(torch.from_numpy(x)).detach().numpy()
    got = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"]).run(["output"], {"input": x})[0]
    print(f"PyTorch と ORT(CPU) の出力の最大差: {np.abs(ref - got).max():.2e}")

    rows = []
    for name, _, _ in BACKENDS:
        # backend ごとに別プロセス: CUDA コンテキストが混ざらず、VRAM を公平に比べられる
        cmd = [sys.executable] + (["-I"] if sys.flags.isolated else []) + \
              [os.path.abspath(__file__), MODEL, str(SECONDS), name]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"★ {name} が失敗:\n{r.stdout[-400:]}{r.stderr[-1200:]}")
            continue
        rows += [json.loads(l[4:]) for l in r.stdout.splitlines() if l.startswith("ROW ")]
        time.sleep(1.0)

    print()
    print(f"{'backend':<14}{'batch':>6}{'p50us':>8}{'p95us':>8}{'p99us':>8}{'samples/s':>11}"
          f"{'util%':>7}{'(待機)':>7}{'SMclk':>7}{'(待機)':>7}{'VRAM+MiB':>10}{'対torch差':>10}")
    f = lambda v, fmt: ("-" if v is None else format(v, fmt))
    for r in rows:
        print(f"{r['backend']:<14}{r['batch']:>6}{r['p50']:>8.0f}{r['p95']:>8.0f}{r['p99']:>8.0f}"
              f"{r['sps']:>11.0f}{f(r['util'], '.0f'):>7}{f(r['util_idle'], '.0f'):>7}"
              f"{f(r['clk'], '.0f'):>7}{f(r['clk_idle'], '.0f'):>7}{f(r['vram'], '.0f'):>10}{r['diff']:>10.1e}")


if __name__ == "__main__":
    main()
