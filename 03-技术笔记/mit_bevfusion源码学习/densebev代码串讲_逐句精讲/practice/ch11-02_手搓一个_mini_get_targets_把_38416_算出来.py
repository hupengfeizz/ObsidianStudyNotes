"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习2：手搓一个_mini_get_targets_把_38416_算出来
跑法：conda activate yolov8 && python ch11-02_手搓一个_mini_get_targets_把_38416_算出来.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, math

H, W, NUM_CLS, MAX_OBJ = 448, 224, 5, 256
VOXEL = 0.4
PC_RANGE_X_FRONT = 95.4   # 车前 95.4m 对应 row 0（见正文 ⚠ 推断）

def draw_gaussian(hm, cx, cy, radius):
    """CenterNet 的 draw_umich_gaussian 简化版，峰顶精确 = 1.0"""
    d = 2 * radius + 1
    sigma = d / 6.0
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1),
                          torch.arange(-radius, radius + 1), indexing='ij')
    g = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[radius, radius] = 1.0                       # ← 关键契约：峰顶严格 1.0
    y0, y1 = max(0, cy - radius), min(hm.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(hm.shape[1], cx + radius + 1)
    gy0, gy1 = radius - (cy - y0), radius + (y1 - cy)
    gx0, gx1 = radius - (cx - x0), radius + (x1 - cx)
    hm[y0:y1, x0:x1] = torch.maximum(hm[y0:y1, x0:x1], g[gy0:gy1, gx0:gx1])

# ---- 造一个"正前方 27m、横向 0m 的 car" ----
x_m, y_m, cls_id = 27.0, 0.0, 0          # cls 0 = car
row = int((PC_RANGE_X_FRONT - x_m) / VOXEL)   # 171
col = int(W / 2 + y_m / VOXEL)                # 112
print("row, col =", row, col)

heatmap = torch.zeros(NUM_CLS, H, W)
draw_gaussian(heatmap[cls_id], col, row, radius=4)

inds  = torch.zeros(MAX_OBJ, dtype=torch.long)
masks = torch.zeros(MAX_OBJ, dtype=torch.uint8)
inds[0]  = row * W + col           # ← 展平索引
masks[0] = 1

print("ind      =", inds[0].item())            # 期望 38416
print("反解 row  =", (inds[0] // W).item())     # 期望 171
print("反解 col  =", (inds[0] %  W).item())     # 期望 112
print("num_pos  =", heatmap.eq(1).float().sum().item())   # 期望 1.0
print("heatmap.shape =", tuple(heatmap.unsqueeze(0).shape))  # 期望 (1,5,448,224)
