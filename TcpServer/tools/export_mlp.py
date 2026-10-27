# 重いモデル(MLP)を ONNX に書き出す。infer_server の「推論が支配的な領域」を作るためのもの。
#
#   784 -> 2048 -> 2048 -> 10 の3層 (約 5.8M パラメータ、重み約 23MB)
#   入力 "input"  : (batch, 1, 28, 28)  ← infer_server が送るテンソルの形と同じ
#   出力 "output" : (batch, 10)
#   バッチ軸 (batch) は可変。dynamic batching (max_batch >= 2) にはこれが必要。
#
# 重みは乱数(seed 固定)。精度は見ない。測るのはレイテンシとスループットだけ。
# 使い方:  python tools/export_mlp.py [出力パス=models/mlp_dyn.onnx]
#   ※ 重みが別ファイル (<出力パス>.data) に分かれることがある。2つを同じフォルダに置くこと。

import os
import sys

import torch
import torch.nn as nn

out_path = sys.argv[1] if len(sys.argv) > 1 else "models/mlp_dyn.onnx"
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

torch.manual_seed(0)   # 再現性: 誰が作っても同じ重みになる


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                       # (b,1,28,28) -> (b,784)
            nn.Linear(784, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.net(x)


model = MLP().eval()
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy, out_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
)
params = sum(p.numel() for p in model.parameters())
print(f"exported {out_path}  params: {params / 1e6:.2f} M")
