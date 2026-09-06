"""densebev代码串讲 逐句精讲 · Ch2 DepthGT与DepthLoss · 练习1：三件套形状与稀疏监督
跑法：conda activate yolov8 && python ch02-01_三件套形状与稀疏监督.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, n_frames, n_cams = 1, 3, 7
n = n_frames * n_cams                       # 21
D, H, W = 100, 88, 160

depth_logits = torch.randn(B * n, D, H, W)        # 网络预测 (21,100,88,160)
depths      = torch.rand(B, n, H, W) * 0.12       # 归一化后的GT, 量程0~0.12(见Part2.2)
depth_masks = (torch.rand(B, n, H, W) > 0.95).float()   # 模拟lidar稀疏投影:约5%像素有效

# 复现 fpn_forward.py L627-629 的整形
_, _, dh_gt, dw_gt = depths.size()
depths      = depths.view(-1, dh_gt, dw_gt)       # (21,88,160)
depth_masks = depth_masks.view(-1, dh_gt, dw_gt)  # (21,88,160)

print(depth_logits.shape, depths.shape, depth_masks.shape)
# torch.Size([21, 100, 88, 160]) torch.Size([21, 88, 160]) torch.Size([21, 88, 160])
print(f"有效监督像素占比: {depth_masks.mean():.3f}")   # ≈0.050 —— 稀疏监督必须mask+按有效数平均
