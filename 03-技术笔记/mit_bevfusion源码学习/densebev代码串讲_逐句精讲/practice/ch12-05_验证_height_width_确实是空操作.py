"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习5：验证_height_width_确实是空操作
跑法：conda activate yolov8 && python ch12-05_验证_height_width_确实是空操作.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
B, C, H, W, K = 1, 5, 448, 224, 256
scores = torch.rand(B, C, H, W)

topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), K)
before = topk_inds.clone()
after  = topk_inds % (H * W)

print("H*W               =", H * W)                       # 100352
print("索引最大值        =", before.max().item())          # < 100352
print("取余前后完全相同  :", torch.equal(before, after))    # True  ← 空操作实锤

# 反例：如果第一步写成 view(B, -1)（把类别也拍进去），取余就是必需的
_, inds2 = torch.topk(scores.view(B, -1), K)
print("\nview(B,-1) 时索引最大值 =", inds2.max().item())    # 可达 501759 (=5*100352-1)
print("此时必须取余才能得到格子位置 :", (inds2 % (H*W)).max().item() < H*W)  # True
