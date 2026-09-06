"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习6：扁平索引_行列_的双向验证_含_x_y_对调
跑法：conda activate yolov8 && python ch12-06_扁平索引_行列_的双向验证_含_x_y_对调.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

H, W = 448, 224                      # DenseBEV 的 BEV 网格
RES  = 0.4                           # 米/格
X_FRONT, X_BACK = 95.4, 83.8         # 前 / 后
Y_HALF = 44.8                        # 左右各

print("纵向覆盖 :", H * RES, "m  (应 =", X_FRONT + X_BACK, ")")   # 179.2 == 179.2 ✓
print("横向覆盖 :", W * RES, "m  (应 =", 2 * Y_HALF, ")")         # 89.6  == 89.6  ✓

# ---- 造几个已知格子 ----
rows = torch.tensor([0, 100, 223, 447])
cols = torch.tensor([0,  60, 112, 223])
flat = rows * W + cols
print("\nflat index :", flat.tolist())     # [0, 22460, 50064, 100351]
#   0*224+0    = 0
# 100*224+60   = 22460
# 223*224+112  = 50064
# 447*224+223  = 100351  ← 正好是最大合法索引 = H*W-1 = 100351

# ---- 按代码的写法还原（注意 xs=行, ys=列）----
topk_xs = (flat.float() / torch.tensor(W, dtype=torch.float)).int().float()   # 行
topk_ys = (flat % W).int().float()                                            # 列
print("还原 xs(行) :", topk_xs.tolist(), " 期望", rows.tolist())
print("还原 ys(列) :", topk_ys.tolist(), " 期望", cols.tolist())
assert torch.equal(topk_xs.long(), rows) and torch.equal(topk_ys.long(), cols)
print("✓ 行列还原正确")

# ---- 反例：用 CenterNet 原版（被注释掉的那两行）会怎样 ----
wrong_ys = (flat.float() / torch.tensor(W, dtype=torch.float)).int().float()
wrong_xs = (flat % W).int().float()
print("\n原版命名下 xs =", wrong_xs.tolist(), "（其实是列！）")
print("原版命名下 ys =", wrong_ys.tolist(), "（其实是行！）")
print("→ 若不改名直接送进后面的 pc_range 换算，x/y 会整体转置 90°")
