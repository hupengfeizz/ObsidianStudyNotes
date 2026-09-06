"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习9：rot_lidar_分区专家_lidar_rot_weight_先验蒸馏
跑法：conda activate yolov8 && python ch11-09_rot_lidar_分区专家_lidar_rot_weight_先验蒸馏.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, numpy as np
torch.manual_seed(0)          # 固定随机数，保证下面的数值可复现

B, H, W = 1, 448, 224

rot_gt        = torch.zeros(B, 2, H, W); rot_gt[0, 1] = 1.0            # 全部朝向 0°(cos=1)
rot_attr_mask = torch.zeros(B, 2, H, W); rot_attr_mask[:, :, 150:200, 100:130] = 1.0

# lidar 前向有效区：车前 40m 内的一个扇形（这里用矩形近似）
lidar_front_mask = torch.zeros(B, H, W); lidar_front_mask[:, 100:220, 60:170] = 1.0

rot_pred       = torch.randn(B, 2, H, W) * 0.1 + torch.tensor([0., 1.])[None, :, None, None]
rot_lidar_pred = torch.randn(B, 2, H, W) * 0.02 + torch.tensor([0., 1.])[None, :, None, None]

def l1_masked(p, g, m, scale=2.0):
    af = (m > 0).float().sum().item()
    return (torch.abs(p - g) * m).sum() / max(af, 1) * scale

rot_attr_mask_2 = rot_attr_mask * lidar_front_mask.unsqueeze(1)
loss_rot       = l1_masked(rot_pred,       rot_gt, rot_attr_mask)
loss_rot_lidar = l1_masked(rot_lidar_pred, rot_gt, rot_attr_mask_2)
print("有效格子: 全图 %d,  lidar前向区 %d"
      % ((rot_attr_mask > 0).sum(), (rot_attr_mask_2 > 0).sum()))
print("loss_rot        = %.4f" % loss_rot)
print("loss_rot_lidar  = %.4f  ← 专家头误差更小" % loss_rot_lidar)
print("rot_dense_loss  = %.4f  ← 两者相加，不单独记录" % (loss_rot + loss_rot_lidar))

# ---- lidar_rot_weight：用常量先验图当 GT ----
close_rot_heatmap = np.zeros((H, W), dtype=np.float32)
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((yy - 171) * 0.4) ** 2 + ((xx - 112) * 0.4) ** 2)
close_rot_heatmap[d < 40] = np.clip(1 - d[d < 40] / 40, 0, 1)          # 越近越大
rot_lidar_w_gt   = torch.from_numpy(np.tile(close_rot_heatmap[None, None], (B, 1, 1, 1)))
rot_lidar_w_mask = (rot_lidar_w_gt > 0).float()
print("rot_lidar_w_gt.shape =", tuple(rot_lidar_w_gt.shape))
print("监督区域格数 =", int(rot_lidar_w_mask.sum()))
print("target 用的是 1-gt，其值域 = %.5f ~ %.5f"
      % ((1 - rot_lidar_w_gt)[rot_lidar_w_mask > 0].min().item(),
         (1 - rot_lidar_w_gt)[rot_lidar_w_mask > 0].max().item()))
pos_num = rot_lidar_w_mask.eq(1).float().sum().item()
print("rot_lidar_w_pos_num =", pos_num, " → avg_factor =", max(pos_num, 1))
