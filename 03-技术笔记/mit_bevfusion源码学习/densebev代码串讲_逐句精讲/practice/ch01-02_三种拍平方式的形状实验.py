"""densebev代码串讲 逐句精讲 · Ch1 DepthNet网络结构 · 练习2：三种拍平方式的形状实验
跑法：conda activate yolov8 && python ch01-02_三种拍平方式的形状实验.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

x = torch.randn(21, 128, 88, 160)          # 3帧x7路针孔的8×特征

# mode 1: 高度全拍平
m1 = x.mean(dim=2).unsqueeze(2)
# mode 3: 高度折成2段，段内均值
m3 = x.reshape(21, 128, 2, -1, 160).mean(dim=3)
# 稀疏方案的token数对比
tok_full, tok_m1, tok_m3 = 88*160, 1*160, 2*160
print(m1.shape)   # 预期 torch.Size([21, 128, 1, 160])
print(m3.shape)   # 预期 torch.Size([21, 128, 2, 160])
print(f"每路相机token数: 原始{tok_full} -> mode1:{tok_m1}(÷88) -> mode3:{tok_m3}(÷44)")
# 预期: 每路相机token数: 原始14080 -> mode1:160(÷88) -> mode3:320(÷44)

# 顺手复刻 prepare_location 的思想：特征网格归一化坐标
H, W = 88, 160
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
location = torch.stack([(xs + 0.5)/W, (ys + 0.5)/H], -1).view(-1, 2)
print(location.shape, location[0], location[-1])
# 预期 torch.Size([14080, 2]) tensor([0.0031, 0.0057]) tensor([0.9969, 0.9943])
