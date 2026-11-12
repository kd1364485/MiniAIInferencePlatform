# M1: PyTorchモデルを作りONNX形式で書き出す
# モデルは Linear(4->3) の1層だけ = 行列積 + バイアス(Phase 4のmatmulと同じ)

import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)   # 入力4 -> 出力3
    def forward(self, x):
        return self.fc(x)

model = TinyNet()
model.eval()                        # 推論モード(勾配計算しない)

# ONNXへ書き出す: ダミー入力を1個食わせて形をトレースする
dummy = torch.randn(1, 4)
torch.onnx.export(
    model,                  # ① 書き出す対象（PyTorchモデル）
    dummy,                  # ② 形を教えるダミー入力（前の質問のやつ）
    "model.onnx",           # ③ 出力ファイル名
 
    input_names=["input"],  # ④ 入力の名前（infer.pyで使うキー名）
    output_names=["output"],# ⑤ 出力の名前（同上）
 
    dynamic_axes={          # ⑥ 「可変にする次元」の指定
        "input":  {0: "batch"},   # inputの0次元目=batch=可変
        "output": {0: "batch"},
    },
 
    opset_version=17,       # ⑦ ONNXの命令セットバージョン
)
print("exported model.onnx")
print("fc weight:", model.fc.weight.data)
print("fc bias  :", model.fc.bias.data)
