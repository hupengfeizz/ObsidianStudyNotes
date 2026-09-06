"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习13：post_center_range_立方体裁剪_分数阈值
跑法：conda activate yolov8 && python ch12-13_post_center_range_立方体裁剪_分数阈值.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

K = 256
torch.manual_seed(1)

# 造 256 个"框"：x,y,z 三列 + 其余 7 列随便
boxes  = torch.zeros(1, K, 10)
boxes[0, :, 0] = torch.empty(K).uniform_(-120, 130)   # x：故意超出 [-83.8, 95.4]
boxes[0, :, 1] = torch.empty(K).uniform_(-60, 60)     # y：故意超出 [-44.8, 44.8]
boxes[0, :, 2] = torch.empty(K).uniform_(-8, 8)       # z
scores = torch.rand(1, K)

post_center_range = [-83.8, -44.8, -5.0, 95.4, 44.8, 3.0]   # [xyz_min, xyz_max]
pcr = torch.tensor(post_center_range, dtype=boxes.dtype)

mask  = (boxes[..., :3] >= pcr[:3]).all(2)
mask &= (boxes[..., :3] <= pcr[3:]).all(2)
print("范围内的框数 :", mask.sum().item(), "/", K)

score_threshold = 0.3
thresh_mask = scores > score_threshold
print("过阈值的框数 :", thresh_mask.sum().item(), "/", K)

cmask = mask[0] & thresh_mask[0]
print("最终保留     :", cmask.sum().item(), "/", K)

kept = boxes[0, cmask]
print("kept.shape   :", tuple(kept.shape))    # (N_valid, 10) —— 动态形状！

# ---- 演示 .all(2) 的作用 ----
one_dim_only = (boxes[..., 0:1] >= pcr[0:1]).all(2)
print("\n只看 x 维保留 :", one_dim_only.sum().item(),
      " vs  xyz 全看保留 :", mask.sum().item())
# 预期：只看 x 会明显多留一些（y 或 z 超界的漏网了）

# ---- 车端痛点演示：动态 shape ----
for seed in range(3):
    torch.manual_seed(seed)
    s = torch.rand(1, K)
    m = (s > 0.3)[0] & mask[0]
    print(f"seed={seed} 输出长度 = {m.sum().item()}")
# 预期：三次长度都不同 —— 这就是为什么最后还要填回固定的 [1,256,15]
