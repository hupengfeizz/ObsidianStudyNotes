"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习4：Gaussian_Focal_Loss_num_pos_归一化_sample_m
跑法：conda activate yolov8 && python ch11-04_Gaussian_Focal_Loss_num_pos_归一化_sample_m.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)

def gaussian_focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """CenterNet 版；pred 已是概率，gt 是高斯软标签"""
    pos_inds = gt.eq(1).float()
    neg_inds = 1.0 - pos_inds
    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos_inds
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg_inds
    return pos_loss.sum(), neg_loss.sum()

B, C, H, W = 2, 5, 64, 32                      # 缩小版 BEV，跑得快
gt = torch.zeros(B, C, H, W)
# 样本0：2 个目标；样本1：1 个目标
for (b, c, y, x) in [(0, 0, 20, 16), (0, 3, 40, 8), (1, 0, 10, 10)]:
    gt[b, c, y, x] = 1.0
    gt[b, c, y-1:y+2, x-1:x+2] = torch.maximum(
        gt[b, c, y-1:y+2, x-1:x+2],
        torch.tensor([[0.6, 0.8, 0.6], [0.8, 1.0, 0.8], [0.6, 0.8, 0.6]]))

pred = clip_sigmoid(torch.zeros(B, C, H, W))   # 初始 logit=0 → p=0.5

                                               # ⚠ 形状必须是 [B,1] 不是 [B]，见正文推导
sample_mask = torch.tensor([[1.0], [0.0]])     # [2,1]：样本1 无效（RL 分支缺失）
gt_masked = gt * sample_mask[..., None, None]  # [2,1,1,1] × [2,5,64,32] ✅

# 反面演示：写成 [B] 会怎样
try:
    _ = gt * torch.tensor([1.0, 0.0])[..., None, None]      # [2,1,1] × [2,5,64,32]
except RuntimeError as e:
    print("sample_mask 写成 [B] 的下场:", str(e)[:60], "...")

for name, g in [("不加 mask", gt), ("加 sample_mask", gt_masked)]:
    num_pos = g.eq(1).float().sum().item()
    pos, neg = gaussian_focal_loss(pred, g)
    print(f"{name:14s} num_pos={num_pos:.0f}  "
          f"未归一化={pos+neg:9.1f}  归一化后={(pos+neg)/max(num_pos,1):7.3f}")

# 对比：如果用像素总数归一化会怎样
num_pix = B * C * H * W
pos, neg = gaussian_focal_loss(pred, gt)
print(f"用像素数归一化: {(pos+neg)/num_pix:.6f}   ← 小 4 个数量级，正样本梯度被淹没")
