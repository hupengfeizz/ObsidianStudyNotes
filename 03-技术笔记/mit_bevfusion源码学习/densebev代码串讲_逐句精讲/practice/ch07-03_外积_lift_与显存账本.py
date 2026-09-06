"""densebev代码串讲 逐句精讲 · Ch7 LSS投影本体 · 练习3：外积_lift_与显存账本
跑法：conda activate yolov8 && python ch07-03_外积_lift_与显存账本.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

N, C, D, H, W = 21, 32, 100, 88, 160
img = torch.randn(N, C, H, W)
dep = torch.softmax(torch.randn(N, D, H, W), dim=1)   # 每像素100个bin概率和为1

# --- 整体外积 ---
vol = dep.unsqueeze(1) * img.unsqueeze(2)             # (21,32,100,88,160)
print(vol.shape, f"{vol.numel()*4/1024**3:.2f} GB (fp32)")

# --- save_memory: 拆 21 份逐份 ---
vol_split = [d * i for d, i in zip(torch.split(dep.unsqueeze(1), 1),
                                   torch.split(img.unsqueeze(2), 1))]
print(len(vol_split), vol_split[0].shape)
print("等价:", torch.allclose(vol, torch.cat(vol_split, 0)))

# 软分配语义: 沿深度求和应还原原特征 (概率和=1)
print("sum over D 还原:", torch.allclose(vol.sum(dim=2), img, atol=1e-5))
# 预期输出:
# torch.Size([21, 32, 100, 88, 160]) 3.52 GB (fp32)
# 21 torch.Size([1, 32, 100, 88, 160])
# 等价: True
# sum over D 还原: True
