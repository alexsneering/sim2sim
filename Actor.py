import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            # ✅ 核心修复：235 → 68（和观测维度完全匹配）
            nn.Linear(235, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            # 输出12维动作（四足12关节）
            nn.Linear(128, 12)
        )

    def forward(self, x):
        return self.mlp(x)