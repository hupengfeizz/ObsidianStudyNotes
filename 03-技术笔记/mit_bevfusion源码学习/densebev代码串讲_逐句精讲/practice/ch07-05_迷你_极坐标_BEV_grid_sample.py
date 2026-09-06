"""densebev代码串讲 逐句精讲 · Ch7 LSS投影本体 · 练习5：迷你_极坐标_BEV_grid_sample
跑法：conda activate yolov8 && python ch07-05_迷你_极坐标_BEV_grid_sample.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn.functional as F, math

# 源"极坐标"特征: 高=深度D, 宽=方位角A (模拟 (1,C,100,160))
D, A, C = 32, 48, 3
src = torch.zeros(1, C, D, A)
src[:, 0, :, A//2] = 1.0        # 正前方一条径向亮线
src[:, 1, D//2, :] = 1.0        # 等距离一圈亮弧

# 目标 BEV 网格 96x96, 自车在底边中点, 前向 0~20m, 横向±10m, 相机FOV=90°
H, W, max_d, fov = 96, 96, 20.0, math.pi/2
ys = torch.linspace(max_d, 0, H).view(H,1).expand(H,W)      # 前向距离
xs = torch.linspace(-10, 10, W).view(1,W).expand(H,W)       # 横向
r  = (xs**2 + ys**2).sqrt()
az = torch.atan2(xs, ys)
gx = az / (fov/2)               # 方位角 -> 归一化x
gy = r / max_d * 2 - 1          # 距离   -> 归一化y
grid = torch.stack([gx, gy], -1).unsqueeze(0)               # (1,H,W,2)

bev = F.grid_sample(src, grid, align_corners=True)
valid = (grid[..., 0].abs() <= 1).float()
print(bev.shape, valid.shape, f"视野内比例={valid.mean():.2f}")
# 打个字符画: 径向亮线在BEV上应是一条竖线, 等距弧应是一个圆弧, FOV外为0
for row in (bev[0,0] + bev[0,1])[::8]:
    print("".join(".#"[int(v > 0.3)] for v in row[::2]))
# 预期输出: torch.Size([1, 3, 96, 96]) torch.Size([1, 96, 96]) 视野内比例≈0.5
# 字符画中可见楔形(FOV)内的十字/弧线——正是视频可视化里楔形特征的成因
