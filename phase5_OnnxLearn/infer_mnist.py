# M3: mnist.onnx を ONNX Runtime で読み、テスト画像を推論する
# 「本物の画像 → 数字の予測」を実際に動かす

import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

sess = ort.InferenceSession("mnist.onnx", providers=["CPUExecutionProvider"])

# 最初の10枚を推論して「予測 vs 正解」を出す
print("idx  pred  true  confidence")
correct = 0
for i in range(10):
    img, label = test_ds[i]
    x = img.unsqueeze(0).numpy()                 #(1,28,28)に先頭1次元を足して (1,1,28,28)（バッチ次元）
    out = sess.run(["output"], {"input": x})[0]  # (1,10)のlogits
    logits = out[0]
    pred = int(np.argmax(logits))
    conf = float(np.exp(logits[pred]) / np.exp(logits).sum())  # softmax確率
    mark = "OK" if pred == label else "NG"
    correct += (pred == label)
    print(f"{i:3d}  {pred:4d}  {label:4d}  {conf:.3f}  {mark}")
print(f"{correct}/10 correct")
