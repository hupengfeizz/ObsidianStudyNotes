"""densebev代码串讲 逐句精讲 · Ch7 LSS投影本体 · 练习2：循环_cat_permute_reshape
跑法：conda activate yolov8 && python ch07-02_循环_cat_permute_reshape.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

BT, V, C, H, W = 3, 7, 4, 8, 10          # 缩小版: (B*T)=3, 7 路相机
fv = torch.randn(BT, 1, V, C, H, W)

# 写法A: 视频里的循环 + cat
lst = [fv[:, 0, j, ...] for j in range(V)]
cat_a = torch.cat(lst, 0)                 # (21,C,H,W)

# 写法B: 一行 permute + reshape
cat_b = fv.squeeze(1).permute(1, 0, 2, 3, 4).reshape(V * BT, C, H, W)

print(cat_a.shape, torch.allclose(cat_a, cat_b))
# 还原索引: dim0 第 k 份 = 相机 k//BT, 帧 k%BT
k = 10
print(torch.allclose(cat_a[k], fv[10 % BT, 0, 10 // BT]))
# 预期输出:
# torch.Size([21, 4, 8, 10]) True
# True
