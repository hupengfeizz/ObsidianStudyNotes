"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习10：动静二分类归并_dir_cls_1_based_编码
跑法：conda activate yolov8 && python ch11-10_动静二分类归并_dir_cls_1_based_编码.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn as nn

B, H, W = 1, 64, 32
ATTR_MOV2_CLS, ATTR_MOV2_W, ATTR_DIR = 4, 5, 8

attr_heats = torch.zeros(B, 9, H, W)
attr_heat_masks_0 = torch.zeros(B, H, W)

# 目标A: 静止车 (mov2=1, 权重3), 朝向类别1
attr_heats[0, ATTR_MOV2_CLS, 10:20, 5:10] = 1.0
attr_heats[0, ATTR_MOV2_W,   10:20, 5:10] = 3.0     # ← 注释"把动静的loss调大3倍"的那个3，烧在GT通道5里（此处按读法B演示）
attr_heats[0, ATTR_DIR,      10:20, 5:10] = 1.0
# 目标B: 移动车 (mov2=2, 权重1), 朝向类别2
attr_heats[0, ATTR_MOV2_CLS, 30:40, 15:20] = 2.0
attr_heats[0, ATTR_MOV2_W,   30:40, 15:20] = 1.0
attr_heats[0, ATTR_DIR,      30:40, 15:20] = 2.0
# 目标C: 动静未知 (mov2=0) —— 应被完全屏蔽
attr_heats[0, ATTR_MOV2_CLS, 50:55, 25:28] = 0.0
attr_heats[0, ATTR_MOV2_W,   50:55, 25:28] = 1.0
attr_heats[0, ATTR_DIR,      50:55, 25:28] = 0.0
for sl in [(slice(10,20),slice(5,10)), (slice(30,40),slice(15,20)), (slice(50,55),slice(25,28))]:
    attr_heat_masks_0[0][sl] = 1.0

# ---- 动静二分类（第 969-978 行）----
mov_attr_mask = ((attr_heat_masks_0
                  * (attr_heats[:, ATTR_MOV2_CLS] < 3.0)
                  * (attr_heats[:, ATTR_MOV2_CLS] >= 1.0))
                 * attr_heats[:, ATTR_MOV2_W]).type(torch.float32).unsqueeze(1)
print("mov_attr_mask 唯一值:", sorted(set(mov_attr_mask.unique().tolist())))  # [0.0,1.0,3.0]
print("  静止区权重 =", mov_attr_mask[0,0,15,7].item(),   "  (期望 3.0)")
print("  移动区权重 =", mov_attr_mask[0,0,35,17].item(),  "  (期望 1.0)")
print("  未知区权重 =", mov_attr_mask[0,0,52,26].item(),  "  (期望 0.0 → 被屏蔽)")

target = attr_heats[:, ATTR_MOV2_CLS, None] - 1.0
print("target 唯一值:", sorted(set(target.unique().tolist())))   # [-1.0, 0.0, 1.0]
print("  ↑ -1 是非法 BCE 目标，靠 weight=0 屏蔽（本章的统一手法）")

pred_logit = torch.zeros(B, 1, H, W)          # logit 0 → p=0.5
fnc = nn.BCEWithLogitsLoss(weight=mov_attr_mask, reduction='sum')
valid_num = max(1.0, (mov_attr_mask > 0).float().sum().item())
mov_dense_loss = fnc(pred_logit, target) * 10 / valid_num
print("valid_num =", valid_num, " mov_dense_loss = %.4f" % mov_dense_loss.item())

# ---- dir_cls（第 994-1000 行）----
enable_corner_det, activate_move = False, True
idx = (6 if enable_corner_det else 4) + (4 if activate_move else 0)
dir_cls_gt   = attr_heats[:, idx:idx+1]
attr_mask_ori = attr_heat_masks_0.unsqueeze(1)                 # 干净 mask，无 car/VRU 加权
dir_cls_mask = attr_mask_ori * (dir_cls_gt > 0)
dir_target   = torch.clamp(dir_cls_gt - 1, min=0)
print("dir idx =", idx, " dir_target 唯一值:", sorted(set(dir_target.unique().tolist())))
print("dir_cls_mask 有效格数 =", int((dir_cls_mask > 0).sum()),
      " (未知区被排除:", int((attr_heat_masks_0 > 0).sum()) - int((dir_cls_mask > 0).sum()), "格)")
