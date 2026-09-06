"""densebev代码串讲 逐句精讲 · Ch9 MemoryManager与时序融合 · 练习2：通道链_128_32_64_的_shape_走查
跑法：conda activate yolov8 && python ch09-02_通道链_128_32_64_的_shape_走查.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

# 真实shape: inc: [3,128,448,224]->[3,32,448,224]; down_sample_conv: ->[3,64,224,112]
# 为了CPU秒跑，把448x224缩成64x32；通道数与真实完全一致
inc = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
down_sample_conv = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(64), nn.ReLU())

bev_multiview = torch.randn(3, 128, 64, 32)      # 真实: [3,128,448,224] @0.4m
x = inc(bev_multiview)
print("inc(128->32)      :", tuple(x.shape))      # (3, 32, 64, 32)
x = down_sample_conv(x)
print("down_sample(->64) :", tuple(x.shape))      # (3, 64, 32, 16)  真实:(3,64,224,112)
# 数据量对比：warp前瘦身到入口的几分之一？
print("体积压缩比 :", (128 * 64 * 32) / (64 * 32 * 16))   # 8.0 —— 正是卡9-15算的1/8
