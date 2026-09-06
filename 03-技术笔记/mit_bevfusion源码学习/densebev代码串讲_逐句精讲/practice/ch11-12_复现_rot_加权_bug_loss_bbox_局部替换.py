"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习12：复现_rot_加权_bug_loss_bbox_局部替换
跑法：conda activate yolov8 && python ch11-12_复现_rot_加权_bug_loss_bbox_局部替换.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, numpy as np

B, MAX_OBJ = 1, 256
# 4 个目标，朝向分别 5° / 45° / 85° / 5°
angles = [5., 45., 85., 5.]
target_box = torch.zeros(B, MAX_OBJ, 10)
mask_raw   = torch.zeros(B, MAX_OBJ)
for i, a in enumerate(angles):
    th = np.deg2rad(a)
    target_box[0, i, 6] = np.sin(th)     # sin
    target_box[0, i, 7] = np.cos(th)     # cos
    target_box[0, i, 8] = 3.0            # vx
    mask_raw[0, i] = 1
mask = mask_raw.unsqueeze(2).expand_as(target_box).clone().float()

rot_range_weight = {'a': dict(range=[0, 30],   weight=1.0),
                    'b': dict(range=[30, 60],  weight=2.0),
                    'c': dict(range=[60, 90.1],weight=5.0)}

target_rot = target_box[..., -4:-2]
rot = torch.atan(target_rot[..., 0] / (target_rot[..., 1] + 1e-10)) * 180 / np.pi
print("解算出的角度:", [round(v,1) for v in rot[0,:4].tolist()])   # [5.0, 45.0, 85.0, 5.0]

def apply_weight(use_bug: bool):
    w = mask[..., -4:-2].clone()
    for k in rot_range_weight:
        src = w if use_bug else rot.unsqueeze(-1).expand_as(w)   # ← bug 用 w，正确用 rot
        m = (src.abs() >= rot_range_weight[k]['range'][0]) & \
            (src.abs() <  rot_range_weight[k]['range'][1])
        w[m] *= rot_range_weight[k]['weight']
    return w

w_bug = apply_weight(True)
w_ok  = apply_weight(False)
print("BUG 版 前4个目标的 rot 权重:", w_bug[0,:4,0].tolist())
print("正确版 前4个目标的 rot 权重:", w_ok[0,:4,0].tolist())
print("→ BUG 版把所有目标都塞进了 [0,30) 这个桶（因为 mask 值 0/1 都 <30）")

# ---- loss_bbox 局部替换 ----
pred = torch.zeros(B, MAX_OBJ, 10)
code_weights = torch.tensor([1.]*8 + [0.2, 0.2])
bbox_weights = mask * code_weights
num = mask_raw.sum()

def l1_keep_last(p, g, w):
    return (torch.abs(p - g) * w).sum(dim=(0, 1)) / (num + 1e-4)

loss_bbox = l1_keep_last(pred, target_box, bbox_weights)        # [10]
print("替换前 loss_bbox =", [round(v,3) for v in loss_bbox.tolist()])
loss_rot = l1_keep_last(pred[..., -4:-2], target_rot, w_ok)     # [2]
loss_bbox[-4:-2] = loss_rot
print("替换后 loss_bbox =", [round(v,3) for v in loss_bbox.tolist()])
print("注意: bbox_loss_total 只取 loss_bbox[:6]，所以 rot/vel 这 4 维只进 TensorBoard")
print("  loss_bbox[:6].sum() =", round(loss_bbox[:6].sum().item(), 4))
