"""densebev代码串讲 逐句精讲 · Ch9 MemoryManager与时序融合 · 练习4：手搓一个_SE_2_ego_motion_warp_本章核心数学
跑法：conda activate yolov8 && python ch09-04_手搓一个_SE_2_ego_motion_warp_本章核心数学.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn.functional as F

H, W, res = 64, 32, 0.8                      # 真实: 224x112 @0.8m
hist = torch.zeros(1, 1, H, W)               # 历史帧BEV：在第18~22行放一个"静止障碍物"
hist[0, 0, 18:23, 8:13] = 1.0

dy = 8                                       # 自车向前开了 8格×0.8m = 6.4米（无旋转）
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
src_y, src_x = ys + dy, xs                   # 目标查源：当前帧格子(y,x) ← 历史帧(y+dy,x)
grid = torch.stack([src_x / (W - 1) * 2 - 1,          # 归一化到[-1,1]，x在前y在后
                    src_y / (H - 1) * 2 - 1], -1)[None].float()
warped = F.grid_sample(hist, grid, mode="bilinear",
                       padding_mode="zeros", align_corners=True)   # 与源码参数一致

print("历史帧障碍物所在行:", torch.nonzero(hist[0, 0].sum(1)).squeeze(-1).tolist())
print("warp后障碍物所在行:", torch.nonzero(warped[0, 0].sum(1) > .5).squeeze(-1).tolist())
# 预期输出：
# 历史帧障碍物所在行: [18, 19, 20, 21, 22]
# warp后障碍物所在行: [10, 11, 12, 13, 14]   <- 整体前移8格：静止物体在当前自车系下"迎面而来"
# 越界处被 padding_mode='zeros' 补零 —— 对应车尾方向历史帧没覆盖到的区域
