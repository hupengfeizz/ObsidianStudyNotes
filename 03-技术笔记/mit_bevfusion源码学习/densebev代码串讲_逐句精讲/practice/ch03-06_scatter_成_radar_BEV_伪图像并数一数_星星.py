"""densebev代码串讲 逐句精讲 · Ch3 Lidar透传与Radar编码图解 · 练习6：scatter_成_radar_BEV_伪图像并数一数_星星
跑法：conda activate yolov8 && python ch03-06_scatter_成_radar_BEV_伪图像并数一数_星星.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

bs3, C, H, W = 3, 64, 448, 224
N = 2407
feature = torch.randn(N, C)                        # PFN输出 [N,64]
coords_batch = torch.stack([
    torch.randint(0, bs3, (N,)),                   # 帧索引（卡3-14 pad进来的）
    torch.zeros(N, dtype=torch.long),              # z恒0
    torch.randint(0, H, (N,)),                     # y: 前后向448格
    torch.randint(0, W, (N,)),                     # x: 左右224格
], dim=1)

canvas = torch.zeros(bs3, C, H, W)                 # 初始化BEV画布（卡3-27）
f, y, x = coords_batch[:, 0], coords_batch[:, 2], coords_batch[:, 3]
canvas[f, :, y, x] = feature                       # 一行scatter（卡3-28）

print(canvas.shape)                                # 预期: torch.Size([3, 64, 448, 224])
occ = (canvas.abs().sum(1) > 0).float()            # [3,448,224] 非零格
print(occ.sum(dim=(1, 2)))                         # 预期: 每帧约800上下(随机坐标少量撞格)
print(occ.mean().item())                           # 预期: ~0.008 —— 99%以上是零,radar BEV是星空图

# 验证范围换算（卡3-30）
print((95.4 + 83.8) / 0.4)                         # 预期: 448.0
print((83.8 - 45.4) / 0.4)                         # 预期: 96.0 —— 448-352, 全扩在后向
