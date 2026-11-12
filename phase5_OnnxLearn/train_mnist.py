# M3: MNIST手書き数字認識 (784→256→128→10 MLP)
# 1エポックだけ学習 → ONNXに書き出す → テスト精度を表示

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

tf = transforms.Compose([
    transforms.ToTensor(),#PIL画像 0-255 -> 0.0-1.0
    transforms.Normalize((0.5,), (0.5,)),# 0.0-1.0 -> -1.0-1.0
])
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=256)

model = nn.Sequential(
    nn.Flatten(),                    # (1,28,28) -> 784
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(dev) #モデルごとGPUへ移す（重みをGPUメモリへ）

opt = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# ---- 1エポック学習 ----
model.train()# 学習モードに切り替え
for x, y in train_dl:
    x, y = x.to(dev), y.to(dev)
    opt.zero_grad()# 勾配を0にする
    loss = loss_fn(model(x), y)# 損失を計算
    loss.backward()# 勾配を計算
    opt.step()# 勾配を更新

# ---- テスト精度 ----
model.eval()# 推論モードに切り替え
correct = total = 0
with torch.no_grad():
    for x, y in test_dl:# 推論モードでは勾配計算不要
        x, y = x.to(dev), y.to(dev)
        pred = model(x).argmax(1)#10個の出力の中で最大値のindexを取る = 「どの数字が一番スコア高いか」
        correct += (pred == y).sum().item()#正解数をカウント
        total += y.numel()#総数をカウント
print(f"test accuracy: {correct}/{total} = {correct/total:.3f}")

# ---- ONNX書き出し (Flatten込みで (1,1,28,28) 入力) ----
dummy = torch.randn(1, 1, 28, 28, device=dev)## 入力形: (バッチ, ch, 縦, 横)
torch.onnx.export(
    model, dummy, "mnist.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
)
print("exported mnist.onnx")
