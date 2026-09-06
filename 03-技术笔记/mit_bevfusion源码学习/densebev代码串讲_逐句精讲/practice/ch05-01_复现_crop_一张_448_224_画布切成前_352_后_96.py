"""densebev代码串讲 逐句精讲 · Ch5 LidarRadar融合 · 练习1：复现_crop_一张_448_224_画布切成前_352_后_96
跑法：conda activate yolov8 && python ch05-01_复现_crop_一张_448_224_画布切成前_352_后_96.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

bs, T, C = 1, 3, 64                       # batch=1, 3帧折叠, 64通道
canvas = torch.randn(bs * T, C, 448, 224)  # radar scatter 出的完整 BEV 画布

# 假设 行0..351 是"lidar 也覆盖的区域"，行352..447 是"后向远距离带"
radar_feature          = canvas[:, :, :352, :]   # 前向切片
crop_rear_radar_feature = canvas[:, :, 352:, :]  # 后向远距离切片

print(radar_feature.shape)            # torch.Size([3, 64, 352, 224])
print(crop_rear_radar_feature.shape)  # torch.Size([3, 64, 96, 224])

# 验证"切片可反传"（对应帧00:33:31里 grad_fn=SliceBackward）
canvas.requires_grad_(True)
loss = canvas[:, :, :352, :].sum()
loss.backward()
print(canvas.grad[:, :, :352, :].abs().sum() > 0,   # tensor(True)  前352行有梯度
      canvas.grad[:, :, 352:, :].abs().sum() == 0)  # tensor(True)  后96行无梯度
# 换算米数：96*0.4=38.4m 后向远距离带；352*0.4=140.8m 共视区
