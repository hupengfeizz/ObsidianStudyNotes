"""densebev代码串讲 逐句精讲 · Ch1 DepthNet网络结构 · 练习5：五分钟复现整个_DepthNet_含形状全链路验证
跑法：conda activate yolov8 && python ch01-05_五分钟复现整个_DepthNet_含形状全链路验证.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

class TinyDepthNet(nn.Module):
    """按本章证据复原：输入多尺度list只吃[0]，两个卷积，通道128->100"""
    def __init__(self, in_ch=128, mid=128, num_bins=100):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1),
                                   nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(mid, num_bins, 1)      # 通道=深度bin数
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]                                   # 只取下采样8倍特征
        return self.conv2(self.conv1(x))               # logits，不做softmax

B, T, V = 1, 3, 7
depth_input = [torch.randn(B*T*V, 128, 88, 160),       # 8×
               torch.randn(B*T*V, 256, 44, 80),        # 16× (陪跑)
               torch.randn(B*T*V, 256, 22, 40)]        # 32× (陪跑)

net = TinyDepthNet()
depth_logits = net(depth_input)
print(depth_logits.shape)
# 预期: torch.Size([21, 100, 88, 160])   <- 与00:06:26调试台一字不差

prob = depth_logits.softmax(dim=1)
print(prob.sum(1).allclose(torch.ones(21, 88, 160)))   # 预期: True 每像素分布归一
print(f"参数量: {sum(p.numel() for p in net.parameters())/1e3:.1f}K")
# 预期约 160.6K —— 整个深度头不到0.2M参数
