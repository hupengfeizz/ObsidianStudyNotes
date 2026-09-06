"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习10：量化_有_reg_无_reg_的定位精度差
跑法：conda activate yolov8 && python ch12-10_量化_有_reg_无_reg_的定位精度差.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
torch.manual_seed(0)

RES = 0.4                          # 米/格
N   = 100000

# 真实中心的连续格子坐标（随机小数）
true_grid = torch.rand(N) * 400

# 方案 A：只用整数格子（无 reg）
pred_A = true_grid.floor()
# 方案 B：整数格子 + 完美 reg
pred_B = true_grid.floor() + (true_grid - true_grid.floor())
# 方案 C：整数格子 + 0.5（else 分支）
pred_C = true_grid.floor() + 0.5
# 方案 D：整数格子 + reg，但 reg 有 0.05 格的回归噪声
pred_D = true_grid + torch.randn(N) * 0.05

for name, p in [("A 无偏移      ", pred_A), ("B 完美reg     ", pred_B),
                ("C 恒加0.5(格心)", pred_C), ("D reg有噪声   ", pred_D)]:
    err_m = (p - true_grid).abs() * RES
    print(f"{name}  平均误差 {err_m.mean():.4f} m   最大误差 {err_m.max():.4f} m")

# 预期输出（约）：
# A 无偏移         平均误差 0.2000 m   最大误差 0.4000 m
# B 完美reg        平均误差 0.0000 m   最大误差 0.0000 m
# C 恒加0.5(格心)  平均误差 0.1000 m   最大误差 0.2000 m
# D reg有噪声      平均误差 0.0160 m   最大误差 0.0900 m
#
# 结论：光是把"取左上角"改成"取格心"就把平均误差砍半；
#      加上一个哪怕不太准的 reg 头（σ=0.05格=2cm），误差再降一个数量级。
