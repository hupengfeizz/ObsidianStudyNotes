"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习7：4_层_mask_叠乘_car_3_近处VRU_3_角度分桶
跑法：conda activate yolov8 && python ch11-07_4_层_mask_叠乘_car_3_近处VRU_3_角度分桶.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, numpy as np

B, H, W, NUM_CLS = 1, 448, 224, 5
CAR, VRU = 0, 3

# ---- 造场景：3 个目标 ----
#  A: car, 正前 27m (row171,col112), 朝向 5°   （同向）
#  B: VRU, 近处 (row380,col100),     朝向 88°  （横穿）
#  C: truck, 远处 (row60,col60),     朝向 45°
heatmap = torch.zeros(B, NUM_CLS, H, W)
attr_heats = torch.zeros(B, 9, H, W)
attr_heat_masks_0 = torch.zeros(B, H, W)

def put(cls_id, r, c, deg, hr=5, wr=2):
    heatmap[0, cls_id, r-hr:r+hr, c-wr:c+wr] = 0.9
    heatmap[0, cls_id, r, c] = 1.0
    th = np.deg2rad(deg)
    attr_heats[0, 0, r-hr:r+hr, c-wr:c+wr] = np.sin(th)
    attr_heats[0, 1, r-hr:r+hr, c-wr:c+wr] = np.cos(th)
    attr_heat_masks_0[0, r-hr:r+hr, c-wr:c+wr] = 1.0

put(CAR, 171, 112, 5);  put(VRU, 380, 100, 88);  put(1, 60, 60, 45)

close_cls_mask = torch.zeros(1, 1, H, W); close_cls_mask[:, :, 300:, :] = 1.0

# ---- 第1层：attr mask + not_cls_only ----
attr_mask = torch.cat([attr_heat_masks_0.unsqueeze(1)] * 2, dim=1)   # [1,2,448,224]
not_cls_only = torch.tensor([1.0])
attr_mask = attr_mask * not_cls_only[..., None, None]
print("L1 唯一权重值:", sorted(set(attr_mask.unique().tolist())))          # [0.0, 1.0]

# ---- 第2层：car ×3 ----
attr_mask = attr_mask * ((heatmap[:, :1] > 0).float() * 2 + 1)
print("L2 唯一权重值:", sorted(set(attr_mask.unique().tolist())))          # [0.0, 1.0, 3.0]

# ---- 第3层：近处 VRU ×3 ----
vru_pos_mask = (heatmap[:, VRU:VRU+1] > 0) * close_cls_mask
attr_mask = attr_mask * (vru_pos_mask.float() * 2 + 1)
print("L3 唯一权重值:", sorted(set(attr_mask.unique().tolist())))          # [0.0, 1.0, 3.0]

# ---- 第4层：角度分桶 ----
rot_gt = attr_heats[:, :2]
rot = torch.atan(rot_gt[:, 0] / (rot_gt[:, 1] + 1e-10)) * 180 / np.pi     # [1,448,224] 度
rot_range_weight = {'near0': dict(range=[0, 30], weight=1.0),
                    'mid':   dict(range=[30, 60], weight=1.5),
                    'lateral': dict(range=[60, 90.1], weight=3.0)}
rot_attr_mask = attr_mask.clone()
for k in rot_range_weight:
    m = (rot.abs() >= rot_range_weight[k]['range'][0]) & (rot.abs() < rot_range_weight[k]['range'][1])
    m = m.unsqueeze(1).repeat(1, 2, 1, 1)
    rot_attr_mask[m] *= rot_range_weight[k]['weight']

print("最终权重直方图:")
for v in sorted(set(rot_attr_mask.unique().tolist())):
    print(f"   w={v:5.1f}  格数={int((rot_attr_mask==v).sum())}")
print("car(5°) 处权重 =", rot_attr_mask[0,0,171,112].item())     # 1*3*1.0 = 3.0
print("VRU(88°,近) 权重 =", rot_attr_mask[0,0,380,100].item())   # 1*3*3.0 = 9.0
print("truck(45°,远) 权重 =", rot_attr_mask[0,0,60,60].item())   # 1*1*1.5 = 1.5

avg_factor = (rot_attr_mask > 0).float().sum().item()
rot_pred = torch.zeros_like(rot_gt)
rot_dense_loss = (torch.abs(rot_pred - rot_gt) * rot_attr_mask).sum() / max(avg_factor, 1) * 2
print("rot_dense_loss =", round(rot_dense_loss.item(), 4))
