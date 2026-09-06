"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习1：复刻检测头输出_dict_并核对_25_个通道
跑法：conda activate yolov8 && python ch11-01_复刻检测头输出_dict_并核对_25_个通道.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
# 复刻画面 01:31:58 终端里那 11 行打印
B, H, W, NUM_CLS = 1, 448, 224, 5
net_dict = {
    'reg':              torch.randn(B, 2, H, W),   # dx, dy
    'height':           torch.randn(B, 1, H, W),   # z
    'dim':              torch.randn(B, 3, H, W),   # log l, log w, log h
    'lidar_rot_weight': torch.randn(B, 1, H, W),
    'rot':              torch.randn(B, 2, H, W),   # sin, cos
    'vel':              torch.randn(B, 2, H, W),   # vx, vy
    'dir_cls':          torch.randn(B, 1, H, W),
    'movement':         torch.randn(B, 1, H, W),
    'rot_lidar':        torch.randn(B, 2, H, W),
    'close_heatmap':    torch.randn(B, NUM_CLS, H, W),
    'heatmap':          torch.randn(B, NUM_CLS, H, W),
}
total_c = 0
for k, v in net_dict.items():
    print(f"{k:18s} {tuple(v.shape)}")
    total_c += v.shape[1]
print("total channels =", total_c)          # 期望 25
print("BEV cells H*W  =", H * W)            # 期望 100352
print("一帧监督的标量数 =", total_c * H * W) # 期望 2508800（250 万，注意这才是1帧1个batch）
