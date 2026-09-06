"""densebev代码串讲 逐句精讲 · Ch10 BEVUNet与CenterPoint检测头 · 练习2：三帧堆叠特征取当前帧_inputs_2_的切片逻辑预演
跑法：conda activate yolov8 && python ch10-02_三帧堆叠特征取当前帧_inputs_2_的切片逻辑预演.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

bs = 1
# parsing_embedding：三帧沿batch维堆叠 (bs*3)x64x448x224（练习缩小到56x28）
parsing_embedding = torch.randn(bs * 3, 64, 56, 28)
lidar_vel_feat    = torch.randn(bs,     64, 56, 28)   # 时序融合后：本来就是单帧

# —— 复刻 det_head.py use_lidar_rt 分支 ——
bst, c, h, w = parsing_embedding.shape
lidar_rt_ft = parsing_embedding.view(bs, bst // bs, c, h, w)  # [1,3,64,56,28]
lidar_rt_feat = lidar_rt_ft[:, -1, :]                         # 取时序最后一帧=当前帧
print('lidar_rt_ft :', lidar_rt_ft.shape)   # torch.Size([1, 3, 64, 56, 28])
print('lidar_rt    :', lidar_rt_feat.shape) # torch.Size([1, 64, 56, 28])

# —— use_lidar_vel 分支：单帧特征不需要再取[-1] ——
print('lidar_vel   :', lidar_vel_feat.shape) # torch.Size([1, 64, 56, 28])

# 验证 [:, -1] 取到的确实是堆叠的最后一帧
assert torch.equal(lidar_rt_feat[0], parsing_embedding[bs*3 - 1])
print('slice check pass: [:, -1] == 堆叠中的最后一帧（当前帧）')
