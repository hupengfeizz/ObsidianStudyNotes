"""densebev代码串讲 逐句精讲 · Ch10 BEVUNet与CenterPoint检测头 · 练习4：CenterHead_骨架_shared_conv_task_heads_循环
跑法：conda activate yolov8 && python ch10-04_CenterHead_骨架_shared_conv_task_heads_循环.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

class TinyTaskHead(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv = nn.Conv2d(c, 5, 3, padding=1)   # 假装是5类heatmap分支
    def forward(self, x, **kwargs):
        return {'heatmap': self.conv(x),
                'got_lidar_rt': kwargs['lidar_rt_feat'].shape}  # 证明kwargs一路透传

class TinyCenterHead(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.shared_conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1),
                                         nn.BatchNorm2d(c), nn.ReLU())
        self.task_heads = nn.ModuleList([TinyTaskHead(c)])
    def forward_single(self, x, **kwargs):
        x = self.shared_conv(x)                     # “额外的一个卷积”
        return [task(x, **kwargs) for task in self.task_heads]

head = TinyCenterHead()
x = torch.randn(1, 64, 112, 56)                     # 真实为 1x64x448x224
out = head.forward_single(x, lidar_rt_feat=torch.randn(1, 64, 112, 56))
print(out[0]['heatmap'].shape)     # torch.Size([1, 5, 112, 56])
print(out[0]['got_lidar_rt'])      # torch.Size([1, 64, 112, 56]) ← kwargs穿透成功
