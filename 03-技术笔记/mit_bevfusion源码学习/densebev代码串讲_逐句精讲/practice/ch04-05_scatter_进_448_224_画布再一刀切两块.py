"""densebev代码串讲 逐句精讲 · Ch4 Radar代码实走与远距离切分 · 练习5：scatter_进_448_224_画布再一刀切两块
跑法：conda activate yolov8 && python ch04-05_scatter_进_448_224_画布再一刀切两块.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

FRAMES, C, H, W = 3, 64, 448, 224                  # bs=1, 3帧, 与视频调试同规格
voxel_feat = torch.randn(6, C)                     # N=6 个非空pillar的64维特征(Part4输出)
coors = torch.tensor([[0, 238, 112],               # (帧号, y, x) ← Part2 pad好的N×4去掉z列
                      [0,  10,   5],
                      [1, 400,  50],               # y=400 > 352 → 落在后向远距离区!
                      [1,   3, 200],
                      [2, 351,   0],               # 常规区最后一行
                      [2, 352, 223]])              # 远距离区第一行
canvas = torch.zeros(FRAMES, C, H, W)
canvas[coors[:, 0], :, coors[:, 1], coors[:, 2]] = voxel_feat   # 高级索引一步scatter
print(canvas.shape)                                # 预期: torch.Size([3, 64, 448, 224])

bev_grid_lw0 = 352                                 # self.bev_grid_lw[0]
crop_rear = canvas[:, :, bev_grid_lw0:, :]         # 后向远距离: 后45.4m~83.8m
front     = canvas[:, :, :bev_grid_lw0, :]         # 常规区: 前95.4m~后45.4m(与lidar对齐)
print(front.shape)                                 # 预期: torch.Size([3, 64, 352, 224])
print(crop_rear.shape)                             # 预期: torch.Size([3, 64, 96, 224])

# 验证voxel落位: 帧1的y=400应出现在crop_rear的第400-352=48行
print(bool(crop_rear[1, :, 48, 50].abs().sum() > 0))    # 预期: True
print(bool(front[1, :, :, 50].abs().sum() == front[1, :, 3, 200].abs().sum() * 0 + front[1,:,3,200].abs().sum()))  # 帧1常规区仅(3,200)有值
print(float(canvas.abs().sum(dim=(1,2,3))[0]) > 0)      # 预期: True 每帧画布都有货
