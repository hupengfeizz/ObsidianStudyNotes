"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习13：复现_loss_dict_汇总_loss_p__过滤_数值核对
跑法：conda activate yolov8 && python ch11-13_复现_loss_dict_汇总_loss_p__过滤_数值核对.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

# 完全照抄画面 01:52:52 调试器里的 15 项
loss_dict = {
    'task0.loss_movement':          torch.tensor(0.0204),
    'task0.loss_p_reg_loc':         torch.tensor(1.6824),
    'task0.loss_p_height':          torch.tensor(0.8559),
    'task0.loss_p_box_size':        torch.tensor(2.6234),
    'task0.loss_p_rot':             torch.tensor(134.1376),
    'task0.loss_p_vel':             torch.tensor(3.8619),
    'task0.loss_p_vel_dense':       torch.tensor(7.9563),
    'task0.loss_p_rot_dense':       torch.tensor(30.5692),
    'task0.loss_p_lidar_rot_w':     torch.tensor(1.6938),
    'task0.loss_movement_dense':    torch.tensor(6.3973),
    'task0.loss_dir_cls':           torch.tensor(1.5016),
    'task0.loss_heatmap':           torch.tensor(5.0235),
    'task0.loss_bbox':              torch.tensor(43.8560),
    'task0.loss_close_heatmap':     torch.tensor(13126.7217),
    'task0.loss_p_rl_percentage':   1.0,
}
print("len(loss_dict) =", len(loss_dict))                      # 期望 15

# ---- 1) 反推 bbox_loss_total（画面 01:52:45 第 1127/1131 行）----
d = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
recon = (d['task0.loss_p_reg_loc'] + d['task0.loss_p_height'] + d['task0.loss_p_box_size']
         + d['task0.loss_p_vel_dense'] + d['task0.loss_p_rot_dense']
         + d['task0.loss_p_lidar_rot_w'] * 0.1)
print(f"重建的 bbox_loss_total = {recon:.4f}   实际 = {d['task0.loss_bbox']:.4f}   "
      f"误差 = {abs(recon - d['task0.loss_bbox']):.4f}")     # 误差应 < 0.001

# ---- 2) 复刻 get_loss 的过滤规则（画面 01:53:10 第 1548 行）----
total = sum([v for k, v in loss_dict.items() if '.loss_p_' not in k])
print(f"总 loss = {float(total):.4f}")
print("参与反传的项:")
for k, v in loss_dict.items():
    if '.loss_p_' not in k:
        val = v.item() if torch.is_tensor(v) else v
        print(f"   {k:32s} {val:12.4f}   占比 {100*val/float(total):5.2f}%")

# ---- 3) 如果写成讲者说的 'loss_' 会怎样 ----
wrong = [k for k in loss_dict if 'loss_' not in k]
print(f"\n若按 'loss_' 过滤，剩下 {len(wrong)} 项 → 总 loss = 0，训练直接死掉")

# ---- 4) 如果忘了过滤（重复计入）----
naive = sum([v for k, v in loss_dict.items() if torch.is_tensor(v)])
print(f"不过滤的话 = {float(naive):.4f}，其中 reg_loc/height/box_size 被重复计了一遍")
