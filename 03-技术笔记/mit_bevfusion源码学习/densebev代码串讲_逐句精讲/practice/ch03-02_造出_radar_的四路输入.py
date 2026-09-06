"""densebev代码串讲 逐句精讲 · Ch3 Lidar透传与Radar编码图解 · 练习2：造出_radar_的四路输入
跑法：conda activate yolov8 && python ch03-02_造出_radar_的四路输入.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

bs, frames, max_pts, feat_dim = 1, 3, 16, 10
# 每帧非空voxel数（模拟动态N）
voxel_num = torch.tensor([812, 790, 805])       # input[9]: (bs*3,)
N = int(voxel_num.sum())                        # N = 2407

voxels = torch.zeros(N, max_pts, feat_dim)      # input[5]: N*16*10
num_points = torch.randint(1, 6, (N,))          # input[7]: N  (radar稀疏,每格1~5点)
# 前 num_points 个槽位填真点特征，其余保持0（模拟DataLoader定长填充）
for i in range(N):
    voxels[i, :num_points[i]] = torch.randn(num_points[i], feat_dim)

# input[6]: N*3 网格坐标 (z=0, y<224方向, x<448方向) —— pillar场景z恒0
coords = torch.stack([
    torch.zeros(N, dtype=torch.long),
    torch.randint(0, 224, (N,)),
    torch.randint(0, 448, (N,)),
], dim=1)

print(voxels.shape, coords.shape, num_points.shape, voxel_num.shape)
# 预期: torch.Size([2407, 16, 10]) torch.Size([2407, 3]) torch.Size([2407]) torch.Size([3])
print("和=N 校验:", voxel_num.sum().item() == N)   # 预期: True
