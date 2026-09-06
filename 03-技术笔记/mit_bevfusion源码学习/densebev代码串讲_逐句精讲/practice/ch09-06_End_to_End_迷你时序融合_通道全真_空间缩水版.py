"""densebev代码串讲 逐句精讲 · Ch9 MemoryManager与时序融合 · 练习6：End_to_End_迷你时序融合_通道全真_空间缩水版
跑法：conda activate yolov8 && python ch09-06_End_to_End_迷你时序融合_通道全真_空间缩水版.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn, torch.nn.functional as F
from functools import partial

class MiniHDTempoFusion(nn.Module):
    """按帧上代码复刻：通道数与真实一致(128/32/64/96/16/80/192/48/64)，空间缩8倍"""
    def __init__(s):
        super().__init__()
        s.inc = nn.Conv2d(128, 32, 3, padding=1)                # 128->32 @0.4m
        s.down_sample_conv = nn.Conv2d(32, 64, 3, 2, 1)         # ->64 @0.8m
        s.conv1x1 = nn.Conv2d(96, 16, 1)                        # RL 96->16
        s.fusion_module = partial(torch.cat, dim=1)             # "gru or Lstm"的真身
        s.output_conv = nn.Conv2d(192, 64, 3, padding=1)        # 3帧BEV 192->64
        s.output_conv_recip = nn.Conv2d(48, 64, 3, padding=1)   # 3帧RL   48->64
        s.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        s.temporal_num = 3

    def _grid(s, n, h, w):     # 单位grid顶替FeatureWarp(真实版由rot/trans+增强阵生成)
        return F.affine_grid(torch.eye(2, 3)[None].repeat(n, 1, 1), (n, 1, h, w),
                             align_corners=True)

    def forward(s, bev_multiview, rl_feat):                     # [3,128,H,W],[3,96,H/2,W/2]
        in_feat = s.down_sample_conv(s.inc(bev_multiview))      # [3,64,H/2,W/2]
        in_feat = torch.cat((in_feat, s.conv1x1(rl_feat)), 1)   # [3,80,...] 拼车
        feats = in_feat.view(1, s.temporal_num, *in_feat.shape[1:])
        feat_cat = torch.cat([feats[:, i] for i in range(2)], 0)          # 两历史帧拼batch
        grid_cat = s._grid(2, *feat_cat.shape[-2:])
        warped = F.grid_sample(feat_cat, grid_cat, mode="bilinear",
                               padding_mode="zeros", align_corners=True)  # 一次warp
        featmap_list = list(torch.split(warped, 1)) + [feats[:, 2]]       # append当前帧
        lidar_list   = [x[:, 64:] for x in featmap_list]                  # 拆RL(16)
        featmap_list = [x[:, :64] for x in featmap_list]                  # 拆BEV(64)
        det_feat = s.up(s.output_conv(s.fusion_module(featmap_list)))     # 192->64,上采样
        rl_out   = s.up(s.output_conv_recip(s.fusion_module(lidar_list))) # 48->64,上采样
        return det_feat, rl_out

m = MiniHDTempoFusion()
det, rl = m(torch.randn(3, 128, 56, 28), torch.randn(3, 96, 28, 14))
print(tuple(det.shape), tuple(rl.shape))
# 预期输出: (1, 64, 56, 28) (1, 64, 56, 28)   真实: [1,64,448,224] 两枚 —— 与01:15:27帧调试台一致
