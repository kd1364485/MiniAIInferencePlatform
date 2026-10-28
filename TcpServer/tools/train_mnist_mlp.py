"""Phase 7 M4 用: 重い MLP (mlp_dyn と同じ構造) を MNIST で学習して ONNX に書き出す。

  784 -> 2048 -> 2048 -> 10 (約 5.8M パラメータ)。mlp_dyn は乱数重みで精度を測れないため、
  速度の測定に十分重く、かつ精度(正解率)も測れるモデルを作る。FP32 / FP16 / INT8 の精度比較に使う。

使い方:  python tools/train_mnist_mlp.py [MNIST の raw ディレクトリ] [出力パス=models/mnist_mlp_big.onnx]
  入力 "input" (batch,1,28,28)、出力 "output" (batch,10)、バッチ軸は可変。前処理は x/255*2-1。
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I では実行ファイルの場所が path に入らない
from mnist_idx import load_images, load_labels, preprocess        # noqa: E402

RAW = sys.argv[1] if len(sys.argv) > 1 else "../OnnxLearn/data/MNIST/raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/mnist_mlp_big.onnx"
EPOCHS, BATCH, LR = 3, 128, 1e-3

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

x_train = torch.from_numpy(preprocess(load_images(RAW, "train-images-idx3-ubyte"))).to(dev)
y_train = torch.from_numpy(load_labels(RAW, "train-labels-idx1-ubyte").astype(np.int64)).to(dev)
x_test = torch.from_numpy(preprocess(load_images(RAW, "t10k-images-idx3-ubyte"))).to(dev)
y_test = torch.from_numpy(load_labels(RAW, "t10k-labels-idx1-ubyte").astype(np.int64)).to(dev)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 2048), nn.ReLU(),
    nn.Linear(2048, 2048), nn.ReLU(),
    nn.Linear(2048, 10),
).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(x_train), device=dev)
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        # 学習の5ステップ: zero_grad → forward → loss → backward → step
        opt.zero_grad()
        loss = loss_fn(model(x_train[idx]), y_train[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f"epoch {epoch + 1}/{EPOCHS}  loss={loss.item():.4f}  test accuracy={acc * 100:.2f}%  ({time.time() - t0:.0f}s)")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
model.cpu().eval()
torch.onnx.export(
    model, torch.randn(1, 1, 28, 28), OUT,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
)
print(f"exported {OUT}  params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
