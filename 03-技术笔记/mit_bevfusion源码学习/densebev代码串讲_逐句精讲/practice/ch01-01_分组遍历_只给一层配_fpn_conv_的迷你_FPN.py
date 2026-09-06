"""densebev代码串讲 逐句精讲 · Ch1 DepthNet网络结构 · 练习1：分组遍历_只给一层配_fpn_conv_的迷你_FPN
跑法：conda activate yolov8 && python ch01-01_分组遍历_只给一层配_fpn_conv_的迷你_FPN.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn, torch.nn.functional as F

class MiniFPN(nn.Module):
    """复刻本段两个要点：lateral+top-down，且只有第0层过fpn_conv(256->128)"""
    def __init__(self, in_channels=(128, 256, 512), mid=256, out0=128):
        super().__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, mid, 1) for c in in_channels])
        self.fpn_conv0 = nn.Conv2d(mid, out0, 3, padding=1)   # 只服务8×层
    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):             # top-down
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
        return [self.fpn_conv0(laterals[0])] + laterals[1:]   # i==0才过卷积

# 造假数据：N=21(3帧x7路针孔), 尺度按真实值 88x160 / 44x80 / 22x40
feats = [torch.randn(21, 128, 88, 160),
         torch.randn(21, 256, 44, 80),
         torch.randn(21, 512, 22, 40)]
fpn = MiniFPN()
outs = fpn(feats)
for idx, grp in enumerate(['', 'fisheye_']):                  # 组遍历的意思
    print(f"grp='{grp}' -> 属性名 {grp}fpn_nets")
print([tuple(o.shape) for o in outs])
# 预期输出（与00:06:11调试台完全一致的三元列表）：
# [(21, 128, 88, 160), (21, 256, 44, 80), (21, 256, 22, 40)]
