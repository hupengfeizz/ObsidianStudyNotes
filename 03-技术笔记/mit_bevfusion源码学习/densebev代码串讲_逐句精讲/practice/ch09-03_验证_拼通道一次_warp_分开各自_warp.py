"""densebev代码串讲 逐句精讲 · Ch9 MemoryManager与时序融合 · 练习3：验证_拼通道一次_warp_分开各自_warp
跑法：conda activate yolov8 && python ch09-03_验证_拼通道一次_warp_分开各自_warp.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn.functional as F
torch.manual_seed(0)

bev = torch.randn(1, 64, 16, 8)                 # 模拟64通道BEV   真实:[1,64,224,112]
rl  = torch.randn(1, 16, 16, 8)                 # 模拟16通道RL    真实:[1,16,224,112]
grid = torch.rand(1, 16, 8, 2) * 2 - 1          # 随便一个采样网格(归一化坐标)

# 路线A：各自warp（笨办法，两次kernel）
a_bev = F.grid_sample(bev, grid, align_corners=True)
a_rl  = F.grid_sample(rl,  grid, align_corners=True)

# 路线B：channel拼接后一次warp，再拆（代码里的做法）
cat = torch.cat((bev, rl), dim=1)               # [1,80,16,8]
b = F.grid_sample(cat, grid, align_corners=True)
b_bev, b_rl = b[:, :64], b[:, 64:]

print(torch.allclose(a_bev, b_bev), torch.allclose(a_rl, b_rl))   # 预期: True True
print("拼接后shape:", tuple(cat.shape))                            # (1, 80, 16, 8)
