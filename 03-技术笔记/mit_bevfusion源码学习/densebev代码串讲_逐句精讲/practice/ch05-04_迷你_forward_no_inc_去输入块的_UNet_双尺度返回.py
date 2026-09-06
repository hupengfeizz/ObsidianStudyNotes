"""densebev代码串讲 逐句精讲 · Ch5 LidarRadar融合 · 练习4：迷你_forward_no_inc_去输入块的_UNet_双尺度返回
跑法：conda activate yolov8 && python ch05-04_迷你_forward_no_inc_去输入块的_UNet_双尺度返回.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

class MiniUNetNoInc(nn.Module):
    """模拟 DenseBEV lidar_backbone：inc 被外部 first_conv 取代，主干双尺度返回"""
    def __init__(self, c=32):
        super().__init__()
        # 注意：没有 inc！输入默认已是 c 通道（对应两支 first_conv + 相加的产物）
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c,   c*2, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c*2, c*4, 3, padding=1), nn.ReLU())
        self.up1   = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec1  = nn.Sequential(nn.Conv2d(c*4, c*2, 3, padding=1), nn.ReLU())
        self.up2   = nn.ConvTranspose2d(c*2, c,   2, stride=2)
        self.dec2  = nn.Sequential(nn.Conv2d(c*2, c,   3, padding=1), nn.ReLU())

    def forward_no_inc(self, x, return_reciprocal_2nd=False):
        d1 = self.down1(x)                                   # 1/2 尺度
        d2 = self.down2(d1)                                  # 1/4 尺度
        u1 = self.dec1(torch.cat([self.up1(d2), d1], dim=1)) # 回到 1/2 尺度(带skip)
        u2 = self.dec2(torch.cat([self.up2(u1), x],  dim=1)) # 回到全尺度(带skip)
        return (u2, u1) if return_reciprocal_2nd else u2     # u1 即 feat_reciprocal_2nd

net = MiniUNetNoInc(c=32)
parsing_embedding = torch.randn(1, 32, 448, 224)             # Part5.3 的融合结果
out, feat_reciprocal_2nd = net.forward_no_inc(parsing_embedding, return_reciprocal_2nd=True)
print(out.shape)                  # torch.Size([1, 32, 448, 224])  ← 全尺度融合特征
print(feat_reciprocal_2nd.shape)  # torch.Size([1, 64, 224, 112])  ← 半尺度(通道翻倍)特征
# 体会：grad_fn 链末端是 ReLU → 对应帧00:35:25里 parsing_embedding.grad_fn=ReluBackward1
print(out.grad_fn)                # <ReluBackward0 ...>
