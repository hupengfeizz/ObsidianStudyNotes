"""densebev代码串讲 逐句精讲 · Ch5 LidarRadar融合 · 练习2：证明_conv_concat_a_b_conv_a_a_conv_b_b_DDR
跑法：conda activate yolov8 && python ch05-02_证明_conv_concat_a_b_conv_a_a_conv_b_b_DDR.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn as nn

torch.manual_seed(0)
a = torch.randn(1, 64, 32, 32)   # 假装是 lidar embedding 输入
b = torch.randn(1, 64, 32, 32)   # 假装是 radar embedding 输入

# 老写法：concat 成 128 通道，过一个大卷积
big = nn.Conv2d(128, 32, 3, padding=1, bias=True)
y_old = big(torch.cat([a, b], dim=1))

# 新写法：把大卷积的权重沿"输入通道"切成两半，各卷各的再相加
conv_a = nn.Conv2d(64, 32, 3, padding=1, bias=True)
conv_b = nn.Conv2d(64, 32, 3, padding=1, bias=False)  # bias 只留一份！
with torch.no_grad():
    conv_a.weight.copy_(big.weight[:, :64])   # 前64个输入通道的权重
    conv_a.bias.copy_(big.bias)
    conv_b.weight.copy_(big.weight[:, 64:])   # 后64个输入通道的权重
y_new = conv_a(a) + conv_b(b)

print(torch.allclose(y_old, y_new, atol=1e-5))   # True  ← 数学上严格等价
print((y_old - y_new).abs().max())               # tensor(≈1e-6) 浮点误差量级
# 结论：DenseBEV 把"concat+第一层conv"拆成"两支first_conv+逐元素加"零精度损失；
# 真正需要重训兜底的是后面被砍掉的两层conv和中间ReLU（容量变化）。
