"""densebev代码串讲 逐句精讲 · Ch10 BEVUNet与CenterPoint检测头 · 练习1：迷你_CaddnBEVBackbone_下采样_上采样_留中间尺度
跑法：conda activate yolov8 && python ch10-01_迷你_CaddnBEVBackbone_下采样_上采样_留中间尺度.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

class MiniBEVUNet(nn.Module):
    """结构等比例复刻 CaddnBEVBackbone：2个下采样block + 2个反卷积 + cat + reduce。
    真实工程通道为64/128，这里用16/32省算力，形状规律完全一致。"""
    def __init__(self, c=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c,   c,   3, 2, 1), nn.BatchNorm2d(c),   nn.ReLU(),
                          nn.Conv2d(c,   c,   3, 1, 1), nn.BatchNorm2d(c),   nn.ReLU()),
            nn.Sequential(nn.Conv2d(c,   2*c, 3, 2, 1), nn.BatchNorm2d(2*c), nn.ReLU(),
                          nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.BatchNorm2d(2*c), nn.ReLU()),
        ])
        self.deblocks = nn.ModuleList([
            nn.ConvTranspose2d(c,   c, 2, stride=2),   # 224x112 -> 448x224
            nn.ConvTranspose2d(2*c, c, 4, stride=4),   # 112x56  -> 448x224
        ])
        self.reduce_channel = nn.Conv2d(2*c, c, 1)

    def forward(self, x):
        ups, mid_feat = [], []
        for blk, deblk in zip(self.blocks, self.deblocks):
            x = blk(x)
            mid_feat.append(x)
            ups.append(deblk(x))
        x = torch.cat(ups, dim=1)
        return self.reduce_channel(x), mid_feat[::-1]

net = MiniBEVUNet()
bev_temporal = torch.randn(1, 16, 448, 224)      # 对应真实 1x64x448x224
bev_feat, mid_feat = net(bev_temporal)
print('bev_feat :', bev_feat.shape)              # torch.Size([1, 16, 448, 224]) 尺寸不变
for m in mid_feat:
    print('mid_feat :', m.shape)
# 预期输出（对照真实工程 [1,128,112,56] / [1,64,224,112]）：
# mid_feat : torch.Size([1, 32, 112, 56])   ← 最深尺度在前（[::-1]的效果）
# mid_feat : torch.Size([1, 16, 224, 112])
