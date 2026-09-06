"""densebev代码串讲 逐句精讲 · Ch4 Radar代码实走与远距离切分 · 练习1：inputs_多形态入口分发
跑法：conda activate yolov8 && python ch04-01_inputs_多形态入口分发.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
# 复现 RadarVoxelGenerator.forward 的输入分发逻辑：同一函数吃两种输入形态
import torch

def radar_forward(*inputs):
    voxelized = None
    if len(inputs) == 2:                    # 训练态：原始点 + 每帧点数
        pts, pts_index = inputs
        mode = "train: raw points"
    else:                                   # 部署态：dataset 已 voxelize 好
        pts, pts_index = None, None
        voxelized = inputs
        mode = "deploy: pre-voxelized"
    return mode, pts, pts_index, voxelized

pts = torch.randn(1000, 7)                  # 1000 个 radar 点(演示用7维)
pts_index = torch.tensor([400, 350, 250])   # 3 帧各自的点数, 和=1000
print(radar_forward(pts, pts_index)[0])     # 预期: train: raw points

voxels = torch.randn(6, 16, 10)
coors  = torch.randint(0, 200, (6, 3))
num_points = torch.randint(1, 17, (6,))
voxel_num  = torch.tensor([2, 1, 3])
print(radar_forward(voxels, coors, num_points, voxel_num)[0])
# 预期: deploy: pre-voxelized   ← len(inputs)==4 走 else 分支
