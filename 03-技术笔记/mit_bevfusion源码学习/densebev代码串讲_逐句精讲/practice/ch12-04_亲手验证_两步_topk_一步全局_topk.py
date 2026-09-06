"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习4：亲手验证_两步_topk_一步全局_topk
跑法：conda activate yolov8 && python ch12-04_亲手验证_两步_topk_一步全局_topk.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
torch.manual_seed(0)

B, C, H, W, K = 1, 5, 16, 8, 6          # 用小尺寸方便打印；真实是 1,5,448,224,256
scores = torch.rand(B, C, H, W)

# ---------- 方案 A：代码里的两步 topk ----------
topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), K)   # [B,C,K]
print("第一重 topk_scores:", tuple(topk_scores.shape))          # (1, 5, 6)
print("第一重 topk_inds  :", tuple(topk_inds.shape))            # (1, 5, 6)

topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), K)   # [B,K]
cls_A  = (topk_ind / torch.tensor(K, dtype=torch.float)).int()  # 类别 = ind // K
# 用 gather 把类内扁平索引取回来
flat_inds = topk_inds.view(B, -1, 1)                            # [B, C*K, 1]
idx       = topk_ind.unsqueeze(2).expand(B, K, 1)               # [B, K, 1]
pos_A     = flat_inds.gather(1, idx).view(B, K)                 # [B, K]

# ---------- 方案 B：一步全局 topk ----------
gscore, gind = torch.topk(scores.view(B, -1), K)                # 在 C*H*W 上直接选
cls_B = gind // (H * W)
pos_B = gind %  (H * W)

print("\n分数一致 :", torch.allclose(topk_score, gscore))       # True
print("类别一致 :", torch.equal(cls_A.long(), cls_B))           # True
print("位置一致 :", torch.equal(pos_A, pos_B))                  # True
print("\nA: cls", cls_A.tolist(), "pos", pos_A.tolist())
print("B: cls", cls_B.tolist(), "pos", pos_B.tolist())
