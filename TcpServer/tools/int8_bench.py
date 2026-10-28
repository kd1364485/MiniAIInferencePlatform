"""Phase 7 M4: TensorRT の FP32 / FP16 / INT8 を、精度と速度の両方で比べる。

使い方:  python tools/int8_bench.py [model=models/mnist_mlp_big.onnx] [MNIST raw ディレクトリ] [秒/条件=2]
  model は tools/train_mnist_mlp.py で作る(学習済み。精度が測れる)。

比べるもの(エンジンはすべて同じ ONNX から作る):
  fp32 : TF32 を外した厳密な FP32
  fp16 : FP16 を許可
  int8 : INT8 を許可(+ FP16。INT8 にできない層は FP16 に任せる)。後訓練量子化(学習し直さない)

INT8 のキャリブレーション(後訓練量子化の要):
  INT8 は整数 256 段階しか表せない。FP32 の値を INT8 に写す倍率(スケール)を、層ごとに決める必要がある。
  重みの範囲はモデルから分かるが、各層の出力(活性化)の範囲は入力次第 → 代表的な入力(学習データ 512 枚)を
  実際に流して値の分布を調べ、TensorRT が「どこで切り捨てるか」(エントロピーが最小になる位置)を決める。
  結果は .cache ファイルに保存され、2 回目以降は再利用される(モデルを変えたら消すこと)。

測ること: テスト 10,000 枚の正解率 / FP32 との予測の一致率と logits の最大差 /
          バッチ 1・32・256 の p50 と samples/s / エンジンのサイズ / VRAM の増加
  ※ 速度の測り方は gpu_bench.py と同じ「CPU 配列を渡して CPU 配列で受け取る」(転送を含む)。
"""
import json
import os
import site
import subprocess
import sys
import tempfile
import time


def ensure_ld_path():
    """CUDA 13 / cuDNN / TensorRT のライブラリの場所を LD_LIBRARY_PATH に足して再実行する(gpu_bench.py と同じ)。"""
    dirs = []
    for base in site.getsitepackages():
        for rel in ("nvidia/cu13/lib", "nvidia/cudnn/lib", "tensorrt_libs"):
            d = os.path.join(base, rel)
            if os.path.isdir(d):
                dirs.append(d)
    if os.path.isdir("/usr/lib/wsl/lib"):
        dirs.append("/usr/lib/wsl/lib")
    cur = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [d for d in dirs if d not in cur]
    if missing and os.environ.get("_INT8_BENCH_REEXEC") != "1":
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + cur)
        os.environ["_INT8_BENCH_REEXEC"] = "1"
        os.execv(sys.executable, sys.orig_argv)


ensure_ld_path()

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnist_idx import load_images, load_labels, preprocess  # noqa: E402

MODEL = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "models/mnist_mlp_big.onnx")
RAW = sys.argv[2] if len(sys.argv) > 2 else "../OnnxLearn/data/MNIST/raw"
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
PRECS = ["fp32", "fp16", "int8"]
BATCHES = [1, 32, 256]
MAXB = 256
CALIB_IMAGES = 512
WORK = os.path.join(tempfile.gettempdir(), "int8_bench_work")
os.makedirs(WORK, exist_ok=True)


class Calibrator(trt.IInt8EntropyCalibrator2):
    """代表的な入力を TensorRT に渡す係。TensorRT が get_batch を呼び、各層の出力の分布を集める。"""

    def __init__(self, data, batch, cache_path):
        super().__init__()
        self.data, self.batch, self.cache_path, self.i = data, batch, cache_path, 0
        # TensorRT は「GPU 上のアドレス」を受け取るので、GPU に置き場を作っておく(寿命はこのオブジェクトと同じ)
        self.dev = torch.empty((batch, 1, 28, 28), dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.i + self.batch > len(self.data):
            return None                                  # None = もう無い(キャリブレーション終了)
        self.dev.copy_(torch.from_numpy(self.data[self.i:self.i + self.batch]))
        self.i += self.batch
        return [int(self.dev.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            return open(self.cache_path, "rb").read()    # あれば再利用 → キャリブレーションを省略
        return None

    def write_calibration_cache(self, cache):
        open(self.cache_path, "wb").write(cache)


def build_engine(prec, calib_data):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(MODEL):
        sys.exit("ONNX の解析に失敗: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    profile = builder.create_optimization_profile()
    # 最適化の狙い(opt)=最大。キャリブレーションは opt の形で行われるので、バッチ 256 で流す
    profile.set_shape("input", (1, 1, 28, 28), (MAXB, 1, 28, 28), (MAXB, 1, 28, 28))
    config.add_optimization_profile(profile)
    if prec == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    elif prec == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)            # INT8 にできない層の受け皿
        cache = os.path.join(WORK, os.path.basename(MODEL) + ".int8.cache")
        config.int8_calibrator = Calibrator(calib_data, MAXB, cache)
        config.set_calibration_profile(profile)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        sys.exit("エンジンの作成に失敗")
    return trt.Runtime(logger).deserialize_cuda_engine(blob), blob.nbytes, time.time() - t0   # IHostMemory は len() が無く nbytes を使う


def make_runner(engine):
    context = engine.create_execution_context()
    bufs, current = {}, [None]

    def run(x):
        b = x.shape[0]
        if b not in bufs:
            bufs[b] = (torch.empty((b, 1, 28, 28), dtype=torch.float32, device="cuda"),
                       torch.empty((b, 10), dtype=torch.float32, device="cuda"))
        xin, yout = bufs[b]
        if current[0] != b:
            context.set_input_shape("input", (b, 1, 28, 28))
            context.set_tensor_address("input", xin.data_ptr())
            context.set_tensor_address("output", yout.data_ptr())
            current[0] = b
        xin.copy_(torch.from_numpy(x))                              # CPU → GPU
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return yout.cpu().numpy()                                   # GPU → CPU(ここで完了を待つ)
    return run, context


def mem_used_now():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def pct(v, p):
    return v[min(int(len(v) * p / 100), len(v) - 1)]


def child(prec):
    """1 つの精度だけを、このプロセス専用の CUDA コンテキストで測る(VRAM を公平に比べるため別プロセス)。"""
    x_test = preprocess(load_images(RAW, "t10k-images-idx3-ubyte"))
    y_test = load_labels(RAW, "t10k-labels-idx1-ubyte")
    calib = preprocess(load_images(RAW, "train-images-idx3-ubyte")[:CALIB_IMAGES])

    base_mem = mem_used_now()
    engine, blob_bytes, build_s = build_engine(prec, calib)
    run, _ = make_runner(engine)

    # 精度: テスト 10,000 枚を 256 枚ずつ流す
    logits = np.concatenate([run(np.ascontiguousarray(x_test[i:i + MAXB])) for i in range(0, len(x_test), MAXB)])
    acc = float((logits.argmax(1) == y_test).mean())
    np.save(os.path.join(WORK, f"logits_{prec}.npy"), logits)

    # 速度
    speed = []
    for b in BATCHES:
        xb = np.ascontiguousarray(preprocess(load_images(RAW, "t10k-images-idx3-ubyte")[:b]))
        for _ in range(30):
            run(xb)
        lat, t_all = [], time.perf_counter()
        while time.perf_counter() - t_all < SECONDS:
            t0 = time.perf_counter()
            run(xb)
            lat.append((time.perf_counter() - t0) * 1e6)
        el = time.perf_counter() - t_all
        lat.sort()
        speed.append({"batch": b, "p50": pct(lat, 50), "p99": pct(lat, 99), "sps": b * len(lat) / el})

    print("ROW " + json.dumps({"prec": prec, "acc": acc, "engine_mib": blob_bytes / 2**20, "build_s": build_s,
                               "vram": mem_used_now() - base_mem, "speed": speed}), flush=True)


def main():
    if len(sys.argv) > 4:
        return child(sys.argv[4])
    print(f"model={os.path.basename(MODEL)}  tensorrt={trt.__version__}  キャリブレーション={CALIB_IMAGES}枚  秒/条件={SECONDS}")
    cache = os.path.join(WORK, os.path.basename(MODEL) + ".int8.cache")
    if os.path.exists(cache):
        os.remove(cache)                          # 毎回キャリブレーションからやり直す(再現性)
    rows = {}
    for prec in PRECS:
        cmd = [sys.executable] + (["-I"] if sys.flags.isolated else []) + \
              [os.path.abspath(__file__), MODEL, RAW, str(SECONDS), prec]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"★ {prec} が失敗:\n{r.stdout[-300:]}{r.stderr[-1500:]}")
            continue
        for line in r.stdout.splitlines():
            if line.startswith("ROW "):
                rows[prec] = json.loads(line[4:])
        time.sleep(1.0)

    if "fp32" not in rows:
        sys.exit("fp32 の測定に失敗したので、比較できない(上のエラーを確認)")
    ref = np.load(os.path.join(WORK, "logits_fp32.npy"))
    print(f"\n{'精度':<6}{'正解率%':>9}{'FP32と同じ予測%':>16}{'logits最大差':>13}{'エンジンMiB':>12}{'VRAM+MiB':>10}"
          f"{'作成秒':>8}   p50us / samples/s  (batch 1 | 32 | 256)")
    for prec in PRECS:
        if prec not in rows:
            continue
        r = rows[prec]
        got = np.load(os.path.join(WORK, f"logits_{prec}.npy"))
        same = float((got.argmax(1) == ref.argmax(1)).mean()) * 100
        sp = "  |  ".join(f"{s['p50']:.0f} / {s['sps']:.0f}" for s in r["speed"])
        print(f"{prec:<6}{r['acc'] * 100:>9.2f}{same:>16.2f}{np.abs(got - ref).max():>13.1e}{r['engine_mib']:>12.1f}"
              f"{r['vram']:>10.0f}{r['build_s']:>8.1f}   {sp}")


if __name__ == "__main__":
    main()
