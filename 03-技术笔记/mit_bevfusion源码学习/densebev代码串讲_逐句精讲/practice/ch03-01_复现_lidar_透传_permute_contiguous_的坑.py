"""densebev代码串讲 逐句精讲 · Ch3 Lidar透传与Radar编码图解 · 练习1：复现_lidar_透传_permute_contiguous_的坑
跑法：conda activate yolov8 && python ch03-01_复现_lidar_透传_permute_contiguous_的坑.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

# 造假数据：bs=1, 3帧折叠, BEV 352x224, 64维特征, channel-last（模拟DataLoader输出）
inputs0 = torch.randn(3, 352, 224, 64)

# --- lidar backbone forward 的全部内容 ---
output = inputs0.permute(0, 3, 1, 2).to(torch.float32).contiguous()

print(output.shape)          # 预期: torch.Size([3, 64, 352, 224])
print(output.is_contiguous())# 预期: True

# 体会 contiguous 的必要性：
x = inputs0.permute(0, 3, 1, 2)   # 不加 contiguous
print(x.is_contiguous())     # 预期: False —— 只是换了stride，数据没动
try:
    x.view(3, -1)            # view 要求连续内存
except RuntimeError as e:
    print("view 报错:", str(e)[:60])   # 预期: 报错，提示 use .reshape(...)

# 验证物理范围换算（卡3-4）
res = 0.4
print((95.4 + 45.4) / res)   # 预期: 352.0
print(44.8 * 2 / res)        # 预期: 224.0
