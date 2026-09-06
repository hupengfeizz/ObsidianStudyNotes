"""densebev代码串讲 逐句精讲 · Ch2 DepthGT与DepthLoss · 练习3：复现_bin_depths_UD_模式_越界清洗
跑法：conda activate yolov8 && python ch02-03_复现_bin_depths_UD_模式_越界清洗.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    else:
        raise NotImplementedError
    if target:                                   # 训练标签路径: 清洗+取整
        indices = torch.nan_to_num(indices)
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices = indices * ~mask + num_bins * mask   # 越界 → dustbin(=num_bins)
        indices = indices.type(torch.int64)
    return indices

c = 0.002                                        # 名义归一化系数(60m→0.12)
gt_m  = torch.tensor([0.0, 0.3, 5.0, 30.0, 59.9, 60.0, 80.0, float('nan')])
idx = bin_depths(gt_m * c, "UD", 0.0, 60 * c, 100, target=True)
print(idx)
# tensor([  0,   0,   8,  50,  99, 100, 100,   0])
# 0m(背景)→bin0 !  0.3m→bin0  5m→bin8  30m→bin50  59.9m→bin99
# 60m/80m→dustbin100   NaN→nan_to_num→0→bin0
# 注意: 背景0和近处0.3m都落在bin0 —— 为什么不出事? 看Part2.4的valid_depth_mask/dustbin流程
