"""densebev代码串讲 逐句精讲 · Ch8 多视角与RC与模态融合 · 练习1：收料台盘点_三路输入与_BEV_网格验算
跑法：conda activate yolov8 && python ch08-01_收料台盘点_三路输入与_BEV_网格验算.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

# DenseBEV 车辆配置: 7针孔+4鱼眼, 3帧时序, bs=1
bs, T = 1, 3
# BEV 范围: 前95.4m 后83.8m 左右±44.8m
front, back, side = 95.4, 83.8, 44.8

for res, name in [(0.4, "全分辨率(lidar/radar/输出)"), (0.8, "半分辨率(相机LSS投影)")]:
    H = round((front + back) / res)   # 纵向格子数
    W = round(side * 2 / res)         # 横向格子数
    print(f"{name}: {res}m/格 -> {H} x {W}")
# 预期: 0.4m -> 448 x 224 ;  0.8m -> 224 x 112

inputs = [
    [(torch.randn(bs*T, 7, 32, 224, 112), torch.randn(bs*T, 7, 224, 112)),   # 针孔(feat, mask)
     (torch.randn(bs*T, 4, 32,  16,  16), torch.randn(bs*T, 4,  16,  16))],  # 鱼眼(feat, mask)
    torch.randn(bs*T, 64, 448, 224),   # inputs[1]: RL融合特征(lidar_parsing_embedding)
    torch.randn(bs*T, 64, 448, 224),   # inputs[2]: 纯radar特征
]
for grp, (f, m) in zip(["针孔", "鱼眼"], inputs[0]):
    print(grp, "feat:", tuple(f.shape), "mask:", tuple(m.shape))
print("RL   :", tuple(inputs[1].shape))
print("radar:", tuple(inputs[2].shape))
# 预期:
# 针孔 feat: (3, 7, 32, 224, 112) mask: (3, 7, 224, 112)
# 鱼眼 feat: (3, 4, 32, 16, 16)  mask: (3, 4, 16, 16)
# RL   : (3, 64, 448, 224)
# radar: (3, 64, 448, 224)
