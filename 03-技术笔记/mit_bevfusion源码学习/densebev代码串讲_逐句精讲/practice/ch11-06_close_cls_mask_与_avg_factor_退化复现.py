"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习6：close_cls_mask_与_avg_factor_退化复现
跑法：conda activate yolov8 && python ch11-06_close_cls_mask_与_avg_factor_退化复现.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, numpy as np

def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)

def gaussian_focal_loss_sum(pred, gt, sample_mask=None, alpha=2.0, beta=4.0):
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    l = (-torch.log(pred) * (1 - pred).pow(alpha) * pos
         - torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg)
    if sample_mask is not None:
        l = l * sample_mask
    return l.sum()

B, C, H, W = 1, 5, 448, 224

# 1) 造一张"近距离区域"mask：只有 row 300~448（车前 20m 内 ⚠ 示意）为 1
close_cls_mask = np.zeros((H, W), dtype=np.float32)
close_cls_mask[300:, :] = 1.0
_cls_close_mask = torch.from_numpy(np.tile(close_cls_mask[None, None, ...], (B, 1, 1, 1)))
print("_cls_close_mask.shape =", tuple(_cls_close_mask.shape))   # 期望 (1,1,448,224)
print("近距离区域占比 = %.1f%%" % (100 * close_cls_mask.mean()))

# 2) GT：造一帧【真实感】的场景——12 个目标全在远处（row < 300），close 区域内一个都没有
#    （上一版我只放 1 个目标，导致 num_pos=1、主 loss 也没被归一化掉，比值反而 <1，
#     那样根本演示不出问题；一帧十几个目标才是实际情况）
gt = torch.zeros(B, C, H, W)
rows = [40, 60, 80, 100, 120, 140, 160, 171, 190, 210, 240, 270]
for i, r in enumerate(rows):
    gt[0, i % 5, r, 60 + i * 12] = 1.0
pred = clip_sigmoid(torch.zeros(B, C, H, W))          # 训练第 0 步，logit=0 → p=0.5

_task_close_heatmap = gt * _cls_close_mask
close_num_pos = _task_close_heatmap.eq(1).float().sum().item()
num_pos       = gt.eq(1).float().sum().item()
print("num_pos =", num_pos, "  close_num_pos =", close_num_pos)   # 期望 12.0 / 0.0

loss_main  = gaussian_focal_loss_sum(pred, gt) / max(num_pos, 1)          # ÷12
loss_close = gaussian_focal_loss_sum(pred, gt, _cls_close_mask) / max(close_num_pos, 1)  # ÷1 !
print("loss_heatmap       = %.1f   ← 分母 = 12 个目标" % loss_main)
print("loss_close_heatmap = %.1f   ← 分母退化成 1" % loss_close)
print("比值 = %.1fx   （close 只覆盖 33%% 的像素，却比主 loss 大 4 倍）" % (loss_close / loss_main))

# 3) 如果训练已收敛（背景概率压到 1e-3），同样的公式会怎样？
pred_conv = torch.full((B, C, H, W), 1e-3)
for i, r in enumerate(rows):
    pred_conv[0, i % 5, r, 60 + i * 12] = 0.9
loss_close2 = gaussian_focal_loss_sum(pred_conv, gt, _cls_close_mask) / max(close_num_pos, 1)
print("收敛后 loss_close_heatmap = %.4f  ← 塌缩，说明 13126 里含大量初始瞬态" % loss_close2)
