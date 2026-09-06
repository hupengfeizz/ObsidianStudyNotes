"""densebev代码串讲 逐句精讲 · Ch2 DepthGT与DepthLoss · 练习5：mini_DDNLoss_全流程_NaN_保护_Focal_mask_归一化
跑法：conda activate yolov8 && python ch02-05_mini_DDNLoss_全流程_NaN_保护_Focal_mask_归一化.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn.functional as F

def focal_loss(logits, target, gamma=2.0):
    """logits:(N,D,H,W)  target:(N,H,W)∈[0,D-1]  → 逐像素loss (N,H,W)"""
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    return -((1 - p_t) ** gamma) * logp_t

torch.manual_seed(0)
N, D, H, W = 21, 100, 88, 160
depth_logits = torch.randn(N, D, H, W)
depth_logits[0, :, 0, 0] = float('nan')          # 模拟上游坏值
depth_target = torch.randint(0, D, (N, H, W))
depth_masks  = (torch.rand(N, H, W) > 0.95).float()

# 1) 数值保护 (L174-176)
_l = depth_logits
_l = torch.where(torch.isnan(_l), torch.full_like(_l, 0), _l)
_l = torch.where(torch.isinf(_l), torch.full_like(_l, 0), _l)

# 2) focal loss + mask + 有效像素归一化 (L181, L189-192)
loss = focal_loss(_l, depth_target)
loss = loss * depth_masks
num_pixels = depth_masks.sum() + 1e-6
loss = loss.sum() / num_pixels
print(loss)          # ≈4.6上下的有限标量(随机logits下每像素≈-log(1/100)≈4.6, focal因子略压低)
assert torch.isfinite(loss)   # 若注释掉步骤1, 这里会因NaN传播而失败
