"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习7：从零手写__topk_并与_PyTorch_逐值对拍
跑法：conda activate yolov8 && python ch12-07_从零手写__topk_并与_PyTorch_逐值对拍.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
torch.manual_seed(42)

def gather_feat(feats, inds):
    """CenterNet 的 _gather_feat（去掉 mask 分支）"""
    dim  = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    return feats.gather(1, inds)

def topk_centerpoint(scores, K):
    """完整复现 centerpoint_bbox_coders.py:65-103"""
    batch, cat, height, width = scores.size()

    # --- 第一重：类内 topk ---
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)                      # no-op
    topk_xs = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()  # 行
    topk_ys = (topk_inds % width).int().float()                                            # 列

    # --- 第二重：跨类 topk ---
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
    topk_inds  = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys    = gather_feat(topk_ys.view(batch, -1, 1),  topk_ind).view(batch, K)
    topk_xs    = gather_feat(topk_xs.view(batch, -1, 1),  topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


# ---------- 用 DenseBEV 真实尺寸跑一遍 ----------
B, C, H, W, K = 1, 5, 448, 224, 256
heat = torch.rand(B, C, H, W) * 0.02 + 0.49          # 模拟"未训练"的 0.5 附近
# 埋 3 个真目标
heat[0, 0, 250, 112] = 0.95     # 类0，正前方 (row=250, col=112)
heat[0, 2,  60,  40] = 0.90     # 类2
heat[0, 0, 251, 112] = 0.88     # 类0，紧挨着上面那个 —— 模拟"峰值不够尖"

s, inds, cls, ys, xs = topk_centerpoint(heat, K)
print("scores.shape :", tuple(s.shape))     # (1, 256)
print("inds.shape   :", tuple(inds.shape))  # (1, 256)
print("\nTop3 分数    :", s[0, :3].tolist())
print("Top3 类别    :", cls[0, :3].tolist())
print("Top3 行(xs)  :", xs[0, :3].tolist())
print("Top3 列(ys)  :", ys[0, :3].tolist())
print("Top3 扁平idx :", inds[0, :3].tolist())

# 校验：扁平索引 == 行*W + 列
assert torch.equal(inds[0, :3], (xs[0, :3] * W + ys[0, :3]).long())
print("\n✓ inds == row*W + col 成立")
# 预期输出：
#   Top3 分数 [0.95, 0.90, 0.88]
#   Top3 类别 [0, 2, 0]
#   Top3 行   [250.0, 60.0, 251.0]
#   Top3 列   [112.0, 40.0, 112.0]
#   —— 注意第 1 名和第 3 名只差一行！这就是"没有 NMS 时同一目标被重复输出"的真实演示
