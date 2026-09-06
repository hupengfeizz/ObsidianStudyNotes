"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习3：验证_attr_heats_9_通道切片索引
跑法：conda activate yolov8 && python ch11-03_验证_attr_heats_9_通道切片索引.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, H, W = 1, 448, 224
ATTR = dict(ROT_SIN=0, ROT_COS=1, VX=2, VY=3,
            MOV2_CLS=4, MOV2_W=5, MOV4_CLS=6, MOV4_W=7, DIR_CLS=8)

# 造一份 attr_heats：每个通道填自己的编号，方便验证切片
attr_heats = torch.zeros(B, 9, H, W)
for name, idx in ATTR.items():
    attr_heats[:, idx] = idx

# --- 复刻源码里的四处切片（行号来自画面） ---
rot_gt = attr_heats[:, :2]                    # 第 872 行
v_gt   = attr_heats[:, 2:4]                   # 第 875 行
print("rot_gt 通道值:", rot_gt[0, :, 0, 0].tolist())   # 期望 [0.0, 1.0]
print("v_gt   通道值:", v_gt[0, :, 0, 0].tolist())     # 期望 [2.0, 3.0]

# 第 969-970 行：动静二分类 mask
mov2_valid = (attr_heats[:, ATTR['MOV2_CLS']] < 3.0) & (attr_heats[:, ATTR['MOV2_CLS']] >= 1.0)
print("mov2 通道号 =", ATTR['MOV2_CLS'], " 权重通道号 =", ATTR['MOV2_W'])

# 第 995 行：dir_cls 索引公式
enable_corner_det, activate_move = False, True
dir_cls_task_idx = (6 if enable_corner_det else 4) + (4 if activate_move else 0)
print("dir_cls_task_idx =", dir_cls_task_idx)          # 期望 8
dir_cls_gt = attr_heats[:, dir_cls_task_idx:dir_cls_task_idx + 1]
print("dir_cls 通道值 =", dir_cls_gt[0, 0, 0, 0].item())  # 期望 8.0

# attr_heat_masks 是 [B,H,W]，靠 unsqueeze(1)+cat 变成 [B,2,H,W]（第 877-878 行）
attr_heat_masks_0 = torch.ones(B, H, W)
attr_mask = torch.cat([attr_heat_masks_0.unsqueeze(1),
                       attr_heat_masks_0.unsqueeze(1)], dim=1)
print("attr_mask.shape =", tuple(attr_mask.shape))     # 期望 (1, 2, 448, 224)
