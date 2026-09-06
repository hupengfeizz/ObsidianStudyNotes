"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习3：复现_movement_的_prob_class_打包_解包
跑法：conda activate yolov8 && python ch12-03_复现_movement_的_prob_class_打包_解包.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn.functional as F

B, H, W = 1, 4, 4
mov_two_stage = torch.randn(B, 2, H, W)             # 2 类：0=静止 1=运动

prob      = F.softmax(mov_two_stage, dim=1)
prob_max  = torch.max(prob, dim=1, keepdim=True).values     # [B,1,H,W] ∈ [0.5,1]
cls       = torch.argmax(prob, dim=1, keepdim=True)         # [B,1,H,W] ∈ {0,1}
packed    = prob_max + cls                                   # ★ 打包

print("prob_max 范围 :", prob_max.min().item(), prob_max.max().item())  # 应 ≥ 0.5
print("packed  范围 :", packed.min().item(),   packed.max().item())

# ---------- 解包 ----------
cls_back  = packed.floor().long()      # 用 floor / int，绝不能用 round！
prob_back = packed - cls_back

print("类别还原正确 :", torch.equal(cls_back, cls))                       # True
print("概率还原误差 :", (prob_back - prob_max).abs().max().item())        # ~0 (1e-7)

# ---------- 反例：用 round 会翻车 ----------
bad = packed.round().long()
print("用 round 的错误率 :", (bad != cls).float().mean().item())
# 预期：1.0（即 100% 全错），不是 50%！算一下就知道为什么：
#   class=0 → packed = prob_max ∈ (0.5, 1.0]  → round → 1  ≠ 0  ✗
#   class=1 → packed = 1 + prob ∈ (1.5, 2.0]  → round → 2  ≠ 1  ✗
# 两类各自都被整体平移了 +1 —— 这正是 round 的"四舍五入到最近整数"和
# floor 的"向下取整"在 [n+0.5, n+1) 这个区间上行为相反造成的。
# ★ 换句话说：用 round 解包不是"有时候错"，而是"永远错、且错得很整齐"，
#   在联调时会表现为"动静判断整体反了/整体偏了一类"，很容易被误判成模型没训好。

# ---------- 再看一个边界：3 分类时 floor 还成立吗？----------
p3 = torch.tensor([[0.34], [0.50], [0.99]])      # 3 类时 prob_max 最小可到 1/3
c3 = torch.tensor([[0.],   [1.],   [2.]])
pk3 = p3 + c3
print("\n3分类 packed  :", pk3.flatten().tolist())   # [0.34, 1.50, 2.99]
print("3分类 floor 还原:", pk3.floor().flatten().tolist(), " 期望", c3.flatten().tolist())
# 预期：[0.0, 1.0, 2.0] 全对 —— 因为 prob_max ∈ (0,1]，floor 天然把整数位切出来。
# ★ 唯一真正会崩的情况是 prob_max 恰好 = 1.0：packed = class + 1，floor 会多算一类。
#   二分类 softmax 在 fp32 下 prob_max 达到 1.0 是可能的（另一路 logit 极小时溢出饱和）。
#   所以最鲁棒的解包是 (packed - 1e-6).floor()，或者干脆别用这种打包。
