"""densebev代码串讲 逐句精讲 · Ch4 Radar代码实走与远距离切分 · 练习2：split_pad_帧号_cat_三连
跑法：conda activate yolov8 && python ch04-02_split_pad_帧号_cat_三连.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn.functional as F

voxel_num = torch.tensor([2, 1, 3])            # 每帧voxel个数(bs*3=3帧), 和=N=6
coors = torch.tensor([[0, 10,   5],
                      [0, 20,   7],
                      [0, 238, 112],            # ← 视频调试样本同款坐标
                      [0,  3,   4],
                      [0,  5,   6],
                      [0,  7,   8]], dtype=torch.int32)   # N×3, (z,y,x)

coors_split = torch.split(coors, voxel_num.tolist())
print(len(coors_split))                         # 预期: 3   ← 每帧一个元素(对应视频 len()=3)

coors_batch = []
for i, coor in enumerate(coors_split):
    coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)   # 最后一维左侧pad帧号
    coors_batch.append(coor_pad)
coors_batch = torch.cat(coors_batch, dim=0)

print(coors_batch.shape)                        # 预期: torch.Size([6, 4])  N×3 → N×4
print(coors_batch[:, 0].tolist())               # 预期: [0, 0, 1, 2, 2, 2]  帧号列
print(coors_batch[2].tolist())                  # 预期: [1, 0, 238, 112]   第2帧那个voxel
