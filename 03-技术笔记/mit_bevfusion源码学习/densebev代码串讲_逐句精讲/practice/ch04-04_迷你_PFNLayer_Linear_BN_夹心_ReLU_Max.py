"""densebev代码串讲 逐句精讲 · Ch4 Radar代码实走与远距离切分 · 练习4：迷你_PFNLayer_Linear_BN_夹心_ReLU_Max
跑法：conda activate yolov8 && python ch04-04_迷你_PFNLayer_Linear_BN_夹心_ReLU_Max.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn, torch.nn.functional as F

class MiniPFN(nn.Module):
    def __init__(self, cin=10, cout=64):
        super().__init__()
        self.linear = nn.Linear(cin, cout, bias=False)       # 16个点共享权重
        self.norm = nn.BatchNorm1d(cout, eps=1e-3, momentum=0.01)
    def forward(self, x):                                    # x: N×16×10
        x = self.linear(x)                                   # N×16×64
        x = self.norm(x.permute(0, 2, 1).contiguous()
                      ).permute(0, 2, 1).contiguous()        # BN1d要求通道在dim=1
        x = F.relu(x)
        return torch.max(x, dim=1, keepdim=True)[0]          # 沿"点数维"取Max → N×1×64

pfn = MiniPFN().eval()                                       # eval避免BN在N=5上抖动
feats = torch.randn(5, 16, 10)
mask = (torch.arange(16)[None, :] < torch.tensor([16, 3, 7, 1, 16])[:, None])
feats = feats * mask.unsqueeze(-1).float()                   # 先抹零再进PFN(同视频顺序)
out = pfn(feats)
print(out.shape)                 # 预期: torch.Size([5, 1, 64])
print(out.squeeze(1).shape)      # 预期: torch.Size([5, 64])  ← features.squeeze(1)
# 置换不变性验证: 把第0个voxel的16个点打乱, 输出应完全不变
perm = torch.randperm(16)
out2 = pfn(feats[:, perm, :])
print(torch.allclose(out, out2, atol=1e-6))   # 预期: True
