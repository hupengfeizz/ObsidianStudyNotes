"""densebev代码串讲 逐句精讲 · Ch5 LidarRadar融合 · 练习3：复现_补齐_相加_并抓出后向_2_增强
跑法：conda activate yolov8 && python ch05-03_复现_补齐_相加_并抓出后向_2_增强.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

C = 32
lidar_emb = torch.randn(3, C, 352, 224)   # first_conv 后的 lidar embedding
radar_emb = torch.randn(3, C, 352, 224)   # first_conv 后的 radar 前向 embedding
rear_emb  = torch.randn(3, C,  96, 224)   # first_conv 后的后向远距离 radar embedding

# 对应 236/237 行：两支都沿 H 维(dim=2)拼上后向段
radar_full = torch.cat([radar_emb, rear_emb], dim=2)   # [3,32,448,224]
lidar_full = torch.cat([lidar_emb, rear_emb], dim=2)   # [3,32,448,224] ← lidar 缺口用 radar 补
# 对应 239 行：逐元素加
parsing_embedding = lidar_full + radar_full

print(parsing_embedding.shape)                                   # torch.Size([3, 32, 448, 224])
print(torch.allclose(parsing_embedding[:, :, :352], lidar_emb + radar_emb))  # True 前352行=真两模态融合
print(torch.allclose(parsing_embedding[:, :, 352:], 2 * rear_emb))           # True 后96行=2×radar(被"增强")
# 反面实验：把 lidar 缺口改成补零，观察 352/353 行交界会出现分布阶跃
lidar_zero = torch.cat([lidar_emb, torch.zeros_like(rear_emb)], dim=2)
print((lidar_zero + radar_full)[:, :, 352:].std(), (2 * rear_emb).std())  # 补零版方差减半→人工边缘
