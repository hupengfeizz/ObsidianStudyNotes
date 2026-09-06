"""densebev代码串讲 逐句精讲 · Ch10 BEVUNet与CenterPoint检测头 · 练习6：init_bias_4_5951_的玄机_heatmap_头的_focal_初始
跑法：conda activate yolov8 && python ch10-06_init_bias_4_5951_的玄机_heatmap_头的_focal_初始.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

# yaml: separate_head: {type: SeparateHead, init_bias: -4.5951, final_kernel: 3}
# 问题：为什么heatmap最后一层卷积的bias要初始化成-4.5951？
p = torch.sigmoid(torch.tensor(-4.5951))
print(f'sigmoid(-4.5951) = {p:.4f}')          # 预期输出: 0.0100 —— 初始前景概率≈1%

# 复现：一个heatmap分支，final层bias填-4.5951
heatmap_head = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 5, 3, padding=1))            # 5类, final_kernel=3
nn.init.constant_(heatmap_head[-1].bias, -4.5951)

x = torch.randn(1, 64, 112, 56)
with torch.no_grad():
    hm = torch.sigmoid(heatmap_head(x))
print('heatmap out :', hm.shape)               # torch.Size([1, 5, 112, 56])
print(f'初始平均响应: {hm.mean():.4f}')          # ≈0.01量级（卷积随机权重导致轻微浮动）

# 道理：448x224=10万个格子里目标中心只有几十个，正负比~1:1000。
# 若bias=0，初始所有位置预测0.5，第一步focal loss会被海量负样本的大梯度淹没;
# 把初始概率压到1%（log(0.01/0.99)≈-4.595），负样本初始loss极小，训练从第一步就稳定。
# RetinaNet论文的 prior probability π=0.01 同款技巧，CenterPoint/mmdet3d原样继承。
