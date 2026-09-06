"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习11：sin_cos_atan2_的完整往返_周期性断点演示
跑法：conda activate yolov8 && python ch12-11_sin_cos_atan2_的完整往返_周期性断点演示.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, math

# ---------- 1. atan2 能恢复完整 4 象限 ----------
angles = torch.tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi,
                       -3*math.pi/4, -math.pi/2, -math.pi/4])
s, c = torch.sin(angles), torch.cos(angles)
rec_atan2 = torch.atan2(s, c)
rec_atan   = torch.atan(s / c)          # 错误做法
print("原始角度(度) :", (angles * 180/math.pi).round().tolist())
print("atan2 恢复   :", (rec_atan2 * 180/math.pi).round().tolist())
print("atan  恢复   :", (rec_atan  * 180/math.pi).round().tolist())
# 实测输出：
#   atan2 恢复 : [0.0, 45.0, 90.0, 135.0, -180.0, -135.0, -90.0, -45.0]
#   atan  恢复 : [0.0, 45.0, -90.0, -45.0,    0.0,   45.0,  90.0, -45.0]
# → atan 把 135° 变成 -45°、把 180° 变成 0°、把 -135° 变成 45° —— 象限信息全丢，一半的角度是错的。
#
# ★ 但请注意 atan2 那一行的第 5 个值：输入是 +180°，输出是 **-180°**，不是 +180°。
#   物理上两者是同一个方向，数值上却差 360°。原因：float32 存不下 π，
#   torch.tensor(math.pi) 实际是 3.14159274（略大于 π），于是 sin 变成一个极小的负数，
#   atan2 就落到了 -π 那一侧。
#   ★★ 这不是"练习里的小 bug"，而是 Part 12-11 说的"朝向不稳定"的一个真实来源：
#      车头几乎正对 ±180°（也就是车在你正后方、朝你开）时，网络的 sin 输出在 0 附近
#      抖动，解出来的 yaw 就在 +179.99° 和 -179.99° 之间来回跳。数值上跳了 360°，
#      任何"用差值判断是否稳定"的下游逻辑（比如 tracker 的朝向平滑）都会被骗到。
#      正确做法是永远用 wrap 到 (-π, π] 之后的角度差：
#          d = torch.atan2(torch.sin(a - b), torch.cos(a - b))
print("\n180° 附近的分支切割 :", torch.atan2(torch.sin(angles[4:5]), torch.cos(angles[4:5])).item())
diff = torch.atan2(torch.sin(rec_atan2 - angles), torch.cos(rec_atan2 - angles))
print("用 wrap 后的角度差 :", (diff * 180/math.pi).round().tolist(), "← 全 0，说明 atan2 其实没错")

# ---------- 2. 模长不影响结果 ----------
k = torch.tensor([0.1, 1.0, 7.3, 100.0]).view(-1, 1)
ang = torch.tensor([[2.0]])                       # 2 rad
out = torch.atan2(k * torch.sin(ang), k * torch.cos(ang))
print("\n不同模长下的 atan2 :", out.flatten().tolist())   # 全是 2.0 —— 无需归一化

# ---------- 3. 为什么不能直接回归 θ：断点处的假梯度 ----------
gt   = torch.tensor(3.10)                          # 177.6°
pred = torch.tensor(-3.10)                         # -177.6°，物理上只差 4.8°
print("\n直接回归θ 的 L1 loss :", (pred - gt).abs().item())        # 6.20 —— 巨大假梯度
loss_sc = (torch.sin(pred)-torch.sin(gt)).abs() + (torch.cos(pred)-torch.cos(gt)).abs()
print("回归 sin/cos 的 L1   :", loss_sc.item())                    # 0.083 —— 正确反映"很接近"
