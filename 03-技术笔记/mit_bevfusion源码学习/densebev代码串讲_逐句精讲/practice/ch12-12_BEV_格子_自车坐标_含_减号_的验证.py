"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习12：BEV_格子_自车坐标_含_减号_的验证
跑法：conda activate yolov8 && python ch12-12_BEV_格子_自车坐标_含_减号_的验证.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

H, W       = 448, 224
RES        = 0.4                                # = out_size_factor * voxel_size
PC_RANGE   = [95.4, 44.8]                       # ⚠ 这份代码里存的是 MAX 不是 MIN

def grid_to_ego(row, col):
    """复现 centerpoint_bbox_coders.py:194-195"""
    x = PC_RANGE[0] - row * RES
    y = PC_RANGE[1] - col * RES
    return x, y

for r, c, tag in [(0, 0, "左前角"), (0, W-1, "右前角"),
                  (H-1, 0, "左后角"), (H-1, W-1, "右后角"),
                  (H//2, W//2, "中心附近")]:
    x, y = grid_to_ego(r, c)
    print(f"row={r:3d} col={c:3d} ({tag}) → x={x:+7.2f} m  y={y:+7.2f} m")

# ★ 注意：格心到格心的跨度是 (H-1)*RES，不是 H*RES ——差一整格 0.4 m
span_x_center = grid_to_ego(0,0)[0] - grid_to_ego(H-1,0)[0]
span_y_center = grid_to_ego(0,0)[1] - grid_to_ego(0,W-1)[1]
print("\n首末格'代表点'跨度 x :", round(span_x_center,1), "m   = (448-1)*0.4 = 178.8")
print("首末格'代表点'跨度 y :", round(span_y_center,1), "m   = (224-1)*0.4 =  89.2")
print("整块 BEV 覆盖    x :", round(H*RES,1), "m   = 448*0.4 = 179.2  ← 与 前95.4+后83.8 对得上")
print("整块 BEV 覆盖    y :", round(W*RES,1), "m   = 224*0.4 =  89.6  ← 与 左右±44.8 对得上")

# ---- 反例：照 mmdet3d 原版的加号写 ----
def grid_to_ego_mmdet3d(row, col, pc_min=(-83.8, -44.8)):
    return pc_min[0] + row * RES, pc_min[1] + col * RES
print("\n若用原版加号(且 pc_range 存 min)：row=0 →", grid_to_ego_mmdet3d(0,0))
print("→ 第0行变成了车尾最远处，整个 BEV 图前后颠倒")

# 预期输出：
# row=  0 col=  0 (左前角) → x= +95.40 m  y= +44.80 m
# row=  0 col=223 (右前角) → x= +95.40 m  y= -44.40 m
# row=447 col=  0 (左后角) → x= -83.40 m  y= +44.80 m
# row=447 col=223 (右后角) → x= -83.40 m  y= -44.40 m
# 首末格'代表点'跨度 x = 178.8 m ；整块覆盖 x = 179.2 m
# 首末格'代表点'跨度 y =  89.2 m ；整块覆盖 y =  89.6 m
