"""densebev代码串讲 逐句精讲 · Ch7 LSS投影本体 · 练习4：验证_bmm_外积_高度均值
跑法：conda activate yolov8 && python ch07-04_验证_bmm_外积_高度均值.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

N, C, D, H, W = 2, 8, 10, 6, 12
img = torch.randn(N, C, H, W)
dep = torch.softmax(torch.randn(N, D, H, W), dim=1)

# 路径1: 外积 -> mean(H)   (本次视频走的 save_memory 语义)
vol = dep.unsqueeze(1) * img.unsqueeze(2)     # (N,C,D,H,W)
out1 = vol.mean(dim=3)                        # (N,C,D,W)

# 路径2: use_bmm, 照抄 liftsplat_proj.py 第75~85行
b, c, h, w = img.shape
im2 = img.permute(0, 3, 2, 1).reshape(b * w, h, c)     # (N*W, H, C)
b, z, h, w = dep.shape
dp2 = dep.permute(0, 3, 1, 2).reshape(b * w, z, h)     # (N*W, D, H)
out2 = torch.bmm(dp2, im2) / h                          # (N*W, D, C)
out2 = out2.reshape(b, w, z, c).permute(0, 3, 2, 1)     # -> (N,C,D,W)

print(out1.shape, out2.shape, torch.allclose(out1, out2, atol=1e-5))
# 预期输出: torch.Size([2, 8, 10, 12]) torch.Size([2, 8, 10, 12]) True
