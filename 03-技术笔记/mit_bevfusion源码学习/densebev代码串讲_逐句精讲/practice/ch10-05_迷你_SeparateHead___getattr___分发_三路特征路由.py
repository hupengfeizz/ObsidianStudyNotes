"""densebev代码串讲 逐句精讲 · Ch10 BEVUNet与CenterPoint检测头 · 练习5：迷你_SeparateHead___getattr___分发_三路特征路由
跑法：conda activate yolov8 && python ch10-05_迷你_SeparateHead___getattr___分发_三路特征路由.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

class MiniSeparateHead(nn.Module):
    def __init__(self, heads, c=64):
        super().__init__()
        self.heads, self.rot_spec, self.vel_spec = heads, True, True
        self.conv_3x3 = nn.Conv2d(2 * c, c, 3, padding=1)
        for name, (out_c, num_conv) in heads.items():          # [属性维数, 卷积个数]
            layers, cin = [], c
            for _ in range(num_conv - 1):
                layers += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU()]
            layers.append(nn.Conv2d(c, out_c, 3, padding=1))   # final_kernel=3
            self.__setattr__(name, nn.Sequential(*layers))     # 字符串key注册分支

    def forward(self, x, **kw):
        ret = {}
        fus_vel = self.conv_3x3(torch.cat((x, kw['lidar_vel_feat']), 1))
        for head in self.heads:
            if self.rot_spec and head.startswith('rot'):
                feat = kw['lidar_rt_feat'] if head == 'rot_lidar' else x  # 简化simple_fusion
            elif self.vel_spec and (head.startswith('vel') or head.startswith('mov')):
                feat = fus_vel
            else:
                feat = x
            ret[head] = self.__getattr__(head)(feat)           # ← 按key取头
        return ret

heads = {'reg': [2,2], 'height': [1,2], 'dim': [3,2], 'rot': [2,2], 'vel': [2,2],
         'dir_cls': [1,2], 'movement': [1,2], 'rot_lidar': [2,2],
         'close_heatmap': [5,2], 'heatmap': [5,2]}
h, w = 112, 56                                # 真实为448x224，等比缩小省算力
net = MiniSeparateHead(heads)
out = net(torch.randn(1,64,h,w), lidar_rt_feat=torch.randn(1,64,h,w),
          lidar_vel_feat=torch.randn(1,64,h,w))
for k, v in out.items():
    print(f'{k:14s} {tuple(v.shape)}')
# 预期输出（对照视频调试台，仅H W按比例小4倍）：
# reg            (1, 2, 112, 56)
# height         (1, 1, 112, 56)
# dim            (1, 3, 112, 56)
# rot            (1, 2, 112, 56)
# vel            (1, 2, 112, 56)
# dir_cls        (1, 1, 112, 56)
# movement       (1, 1, 112, 56)
# rot_lidar      (1, 2, 112, 56)
# close_heatmap  (1, 5, 112, 56)
# heatmap        (1, 5, 112, 56)
