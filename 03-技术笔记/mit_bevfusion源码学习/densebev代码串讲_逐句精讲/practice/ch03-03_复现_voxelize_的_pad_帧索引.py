"""densebev代码串讲 逐句精讲 · Ch3 Lidar透传与Radar编码图解 · 练习3：复现_voxelize_的_pad_帧索引
跑法：conda activate yolov8 && python ch03-03_复现_voxelize_的_pad_帧索引.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn.functional as F

voxel_num = torch.tensor([4, 3, 5])            # 3帧，各4/3/5个voxel（缩小版）
N = int(voxel_num.sum())
coords = torch.randint(0, 10, (N, 3))          # [N,3] 网格坐标

# --- self.voxelize 的核心：对coords第0位pad当前帧索引 ---
out, offset = [], 0
for i, n in enumerate(voxel_num.tolist()):
    c = coords[offset:offset + n]
    out.append(F.pad(c, (1, 0), value=i))      # 左侧补一列常数i
    offset += n
coords_batch = torch.cat(out)                  # [N,4]

print(coords_batch.shape)                      # 预期: torch.Size([12, 4])
print(coords_batch[:, 0])                      # 预期: tensor([0,0,0,0, 1,1,1, 2,2,2,2,2])
# 第0列就是帧索引，后3列原样保留 —— 与帧00_16_55注释
# "对coors的第二维索引0的位置pad当前帧的索引"逐字对应
