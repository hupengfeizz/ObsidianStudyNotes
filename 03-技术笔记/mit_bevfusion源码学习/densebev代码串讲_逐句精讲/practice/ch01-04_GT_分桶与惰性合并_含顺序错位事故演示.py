"""densebev代码串讲 逐句精讲 · Ch1 DepthNet网络结构 · 练习4：GT_分桶与惰性合并_含顺序错位事故演示
跑法：conda activate yolov8 && python ch01-04_GT_分桶与惰性合并_含顺序错位事故演示.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

BT, H, W = 3, 88, 160                 # B*T=3
k = 2                                 # 假设主组(前视)2路, endpart 5路
depths_main = torch.full((BT, k,   H, W), 10.0)   # 主组GT深度全10m
depths_end  = torch.full((BT, 7-k, H, W), 50.0)   # 侧后组全50m
mask_main   = torch.ones_like(depths_main)
mask_end    = torch.ones_like(depths_end)

depths = torch.cat([depths_main, depths_end], dim=1)       # 正确顺序
masks  = torch.cat([mask_main, mask_end], dim=1)
print(depths.shape, depths[0, 0, 0, 0].item(), depths[0, -1, 0, 0].item())
# 预期 torch.Size([3, 7, 88, 160]) 10.0 50.0

flat = depths.view(-1, H, W)                                # 与21路logits对齐
print(flat.shape)                     # 预期 torch.Size([21, 88, 160])

# 事故演示：拼接顺序反了,GT整体错位到别的相机上,不报任何错!
wrong = torch.cat([depths_end, depths_main], dim=1).view(-1, H, W)
print("第0路GT本应10m, 错序后:", wrong[0, 0, 0].item())   # 预期 50.0 —— 静默污染
