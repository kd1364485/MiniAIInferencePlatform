# M1: 書き出した model.onnx を ONNX Runtime で推論し、
#      手計算(行列積)と照合する

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# --- 入力データ ---
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)   # batch=1, 4次元

# --- ONNX Runtime で推論 ---
sess = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("providers:", sess.get_providers())

out = sess.run(["output"], {"input": x})[0]
print("ONNX output :", out)

# --- 手計算で照合(Phase 4のmatmul: out = x @ W^T + b) ---
# export_onnx.py で作ったのと同じ重みをPyTorchから取り出す
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
    def forward(self, x):
        return self.fc(x)

# 重みはexport時と違う乱数になるので、ONNX側の重みを直接読む
import onnx
m = onnx.load("model.onnx")
W = None; b = None
for init in m.graph.initializer:
    arr = onnx.numpy_helper.to_array(init)
    if arr.shape == (3, 4): W = arr          # fc.weight (out,in)
    elif arr.shape == (3,):  b = arr          # fc.bias
manual = x @ W.T + b                         # 行列積+バイアス
print("manual matmul:", manual)
print("max diff:", np.abs(out - manual).max())
