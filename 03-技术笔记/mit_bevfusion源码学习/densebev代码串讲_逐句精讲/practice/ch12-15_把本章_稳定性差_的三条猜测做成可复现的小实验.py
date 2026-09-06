"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习15：把本章_稳定性差_的三条猜测做成可复现的小实验
跑法：conda activate yolov8 && python ch12-15_把本章_稳定性差_的三条猜测做成可复现的小实验.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn.functional as F

# =========== (a) 无 NMS 导致的重复框 ===========
H, W, K = 448, 224, 256
heat = torch.full((1, 1, H, W), 0.05)
# 造一个"胖峰值"：中心 0.9，四邻 0.85（模拟训练不足、峰值不够尖）
heat[0, 0, 200, 100] = 0.90
for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
    heat[0, 0, 200+dr, 100+dc] = 0.85

sc, idx = torch.topk(heat.view(1, 1, -1), 5)
rows, cols = idx[0,0] // W, idx[0,0] % W
print("=== (a) 无 NMS ===")
print("Top5 分数 :", sc[0,0].tolist())
print("Top5 位置 :", list(zip(rows.tolist(), cols.tolist())))
print("→ 同一个目标被输出了 5 次\n")

# 加上 CenterNet 标准的 3x3 max-pool NMS 再看
hmax = F.max_pool2d(heat, 3, stride=1, padding=1)
heat_nms = heat * (hmax == heat).float()
sc2, idx2 = torch.topk(heat_nms.view(1,1,-1), 5)
print("加 maxpool-NMS 后 Top5 分数 :", sc2[0,0].tolist())
print("→ 只剩 1 个 0.9，其余被压成 0.05（底噪）\n")

# =========== (b) dir_cls 在 0.5 附近抖动造成 180° 翻转 ===========
import math
base_yaw = torch.tensor(0.3)                       # 车身轴线（半圈内）
dir_logit_frames = torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01])   # 帧间微小抖动
print("=== (b) 朝向翻转 ===")
for t, lg in enumerate(dir_logit_frames):
    d = (torch.sigmoid(lg) > 0.5).float()
    yaw = base_yaw + d * math.pi
    print(f"  frame{t}: dir_logit={lg:+.3f} → dir={int(d)} → yaw={yaw:.3f} rad ({math.degrees(yaw):+.1f}°)")
print("→ logit 只抖了 ±0.03，yaw 却在 17° 和 197° 之间反复横跳\n")

# =========== (c) movement 打包解码用错取整 ===========
print("=== (c) movement 解包 ===")
prob = torch.tensor([0.52, 0.68, 0.95, 0.51])
cls  = torch.tensor([0.,   0.,   1.,   1.  ])
packed = prob + cls
print("  packed        :", packed.tolist())
print("  floor 解出类别:", packed.floor().tolist(), " ← 正确", cls.tolist())
print("  round 解出类别:", packed.round().tolist(), " ← 错误")
