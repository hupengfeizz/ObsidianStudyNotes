"""densebev代码串讲 逐句精讲 · Ch6 LSS投影输入准备 · 练习1：grid_map_的_查表投影_一分钟复现
跑法：conda activate yolov8 && python ch06-01_grid_map_的_查表投影_一分钟复现.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn.functional as F

# 造一张"图像特征"：1×1×4×8，像素值=列号，方便肉眼验证采样对不对
feat = torch.arange(8.).repeat(4, 1).view(1, 1, 4, 8)

# 造一张"grid map"：BEV 网格 2×3，每格存归一化采样坐标 (x,y)∈[-1,1]
# 这就是 DataLoader 在 CPU 上用内外参算好的东西的最简化版
grid = torch.tensor([[[[-1., -1.], [0., 0.], [1., -1.]],
                      [[-1.,  1.], [0., 0.], [1.,  1.]]]])  # (1,2,3,2)

bev = F.grid_sample(feat, grid, align_corners=True)
print(bev.squeeze())
# 预期输出：
# tensor([[0.0000, 3.5000, 7.0000],
#         [0.0000, 3.5000, 7.0000]])
# 左列采到图像最左(值0)，中间采到图像中心(3.5)，右列采到最右(7)。
# 体会：网络里只剩 grid_sample 一步，几何早在 grid 里定死了。
