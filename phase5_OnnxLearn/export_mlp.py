# M2: 推論速度比較用のMLPを作ってONNXに書き出す
# 1024 -> 2048 -> 2048 -> 10 の3層 (matmulが支配的 = GPU向きの仕事)

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 10),
        )
    def forward(self, x):
        return self.net(x)

model = MLP().eval()
dummy = torch.randn(1, 1024)
torch.onnx.export(
    model, dummy, "mlp.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
)
print("exported mlp.onnx")
params = sum(p.numel() for p in model.parameters())
print(f"params: {params/1e6:.2f} M")   # 約630万パラメータ
