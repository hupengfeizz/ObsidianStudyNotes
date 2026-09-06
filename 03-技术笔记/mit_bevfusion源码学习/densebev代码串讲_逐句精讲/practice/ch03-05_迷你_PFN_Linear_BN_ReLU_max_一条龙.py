"""densebev代码串讲 逐句精讲 · Ch3 Lidar透传与Radar编码图解 · 练习5：迷你_PFN_Linear_BN_ReLU_max_一条龙
跑法：conda activate yolov8 && python ch03-05_迷你_PFN_Linear_BN_ReLU_max_一条龙.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N, P, C_in, C_out = 2407, 16, 10, 64
feature = torch.randn(N, P, C_in)                 # 已mask消毒的 [N,16,10]
num_points = torch.randint(1, 6, (N,))
mask = (torch.arange(P)[None, :] < num_points[:, None]).unsqueeze(-1).float()
feature = feature * mask                          # 假点归零（衔接练习ch3-4）

linear = nn.Linear(C_in, C_out, bias=False)       # 参数量仅 10*64=640
norm = nn.BatchNorm1d(C_out, eps=1e-3, momentum=0.01)

x = linear(feature)                               # [N,16,64]
x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)     # BN1d按64通道归一
x = F.relu(x)
print(x.shape)                                    # 预期: torch.Size([2407, 16, 64])

pillar_feat = torch.max(x, dim=1)[0]              # 沿16个点取max
print(pillar_feat.shape)                          # 预期: torch.Size([2407, 64])

# 验证max的置换不变性：打乱16个点的顺序，输出不变
perm = torch.randperm(P)
x2 = torch.max(x[:, perm, :], dim=1)[0]
print(torch.allclose(pillar_feat, x2))            # 预期: True
