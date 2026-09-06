"""densebev代码串讲 逐句精讲 · Ch9 MemoryManager与时序融合 · 练习5：torch_split_拆帧_64_16_拆通道_全流程复现
跑法：conda activate yolov8 && python ch09-05_torch_split_拆帧_64_16_拆通道_全流程复现.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
from functools import partial

temporal_num = 3
# 模拟合并warp后的结果：两帧拼在batch维（真实:[2,80,224,112]）
warped_feature = torch.randn(2, 80, 16, 8)
featmaps = torch.randn(1, temporal_num, 80, 16, 8)               # 三帧特征(含当前帧)

featmap_list = list(torch.split(warped_feature, warped_feature.shape[0] // (temporal_num - 1)))
featmap_list.append(featmaps[:, temporal_num - 1, :])            # 当前帧append进去
print("三帧list:", [tuple(x.shape) for x in featmap_list])
# [(1, 80, 16, 8), (1, 80, 16, 8), (1, 80, 16, 8)]

lidar_feature_list = [x[:, 64:, ...] for x in featmap_list]      # 后16通道=RL
featmap_list       = [x[:, :64, ...] for x in featmap_list]      # 前64通道=融合BEV

fusion_module = partial(torch.cat, dim=1)   # 帧证：fusion_module=functools.partial(cat,...)
bev_temporal_feat_map = fusion_module(featmap_list)
lidar_2nd_feat_map    = fusion_module(lidar_feature_list)
print(tuple(bev_temporal_feat_map.shape))   # (1, 192, 16, 8)  真实:[1,192,224,112]
print(tuple(lidar_2nd_feat_map.shape))      # (1, 48, 16, 8)   真实:[1, 48,224,112]
