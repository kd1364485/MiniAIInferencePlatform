# M2: ONNX Runtimeの CPU vs CUDA 推論速度を比較
# sess.run を繰り返し呼んで、レイテンシ分布とスループットを計測する
# (Phase 2のbenchと同じ「往復時間をためてパーセンタイル」)

import numpy as np
import onnxruntime as ort
import time

import os
MODEL  = "mlp.onnx"
BATCH  = int(os.environ.get("BATCH", "1"))  # バッチサイズ(envで変更可)
WARMUP = 50           # 計測前のウォームアップ回数
ITERS  = 500          # 計測回数

x = np.random.randn(BATCH, 1024).astype(np.float32)

def bench(provider):
    sess = ort.InferenceSession(MODEL, providers=[provider])
    # ウォームアップ(JIT/初期化を計測から除外 — matmulの初回と同じ問題)
    for _ in range(WARMUP):
        sess.run(["output"], {"input": x})

    lat = []
    t0 = time.perf_counter()
    for _ in range(ITERS):
        s = time.perf_counter()
        sess.run(["output"], {"input": x})
        lat.append((time.perf_counter() - s) * 1e6)   # μs
    total_s = time.perf_counter() - t0

    lat = np.sort(np.array(lat))
    p = lambda q: lat[int(len(lat) * q / 100)]
    print(f"{provider:30s}  "
          f"p50={p(50):8.1f}us  p95={p(95):8.1f}us  p99={p(99):8.1f}us  "
          f"max={lat[-1]:8.1f}us  infer/s={ITERS/total_s:8.1f}")

bench("CPUExecutionProvider")
bench("CUDAExecutionProvider")
