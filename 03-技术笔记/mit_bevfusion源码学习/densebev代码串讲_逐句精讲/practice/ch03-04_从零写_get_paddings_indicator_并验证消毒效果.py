"""densebev代码串讲 逐句精讲 · Ch3 Lidar透传与Radar编码图解 · 练习4：从零写_get_paddings_indicator_并验证消毒效果
跑法：conda activate yolov8 && python ch03-04_从零写_get_paddings_indicator_并验证消毒效果.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)   # [N,1]
    ids = torch.arange(max_num, device=actual_num.device).view(1, -1)  # [1,16]
    return actual_num.int() > ids                        # [N,16] bool

N, P, C = 5, 16, 10
voxels = torch.randn(N, P, C)
num_points = torch.tensor([3, 1, 16, 7, 2])

mask = get_paddings_indicator(num_points, P)             # [5,16]
print(mask[0])   # 预期: 前3个True其余False
print(mask.sum(1))  # 预期: tensor([ 3,  1, 16,  7,  2]) —— 与num_points一致

mask = mask.unsqueeze(-1).float()                        # [5,16,1]
feature = voxels * mask                                  # [5,16,10]
print(feature.shape)                                     # 预期: torch.Size([5, 16, 10])
print(feature[0, 3:].abs().sum())                        # 预期: tensor(0.) 假点全零
print(feature[0, :3].abs().sum() > 0)                    # 预期: tensor(True) 真点保值
