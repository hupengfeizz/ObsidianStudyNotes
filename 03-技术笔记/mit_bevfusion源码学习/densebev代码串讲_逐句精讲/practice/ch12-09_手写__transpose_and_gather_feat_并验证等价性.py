"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习9：手写__transpose_and_gather_feat_并验证等价性
跑法：conda activate yolov8 && python ch12-09_手写__transpose_and_gather_feat_并验证等价性.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
torch.manual_seed(7)

def gather_feat(feats, inds):
    dim  = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    return feats.gather(1, inds)

def transpose_and_gather_feat(feat, ind):
    """centerpoint_bbox_coders.py:105 —— CenterNet 万能提货器"""
    feat = feat.permute(0, 2, 3, 1).contiguous()        # [B,C,H,W] → [B,H,W,C]
    feat = feat.view(feat.size(0), -1, feat.size(3))    # → [B,H*W,C]
    return gather_feat(feat, ind)                        # → [B,K,C]

B, C, H, W, K = 1, 8, 448, 224, 256
bev = torch.randn(B, C, H, W)
inds = torch.randint(0, H * W, (B, K))                   # 假装是 topk 出来的位置

out = transpose_and_gather_feat(bev, inds)
print("输入 :", tuple(bev.shape))    # (1, 8, 448, 224)
print("输出 :", tuple(out.shape))    # (1, 256, 8)

# ---- 用最朴素的循环对拍，确认语义正确 ----
ref = torch.empty(B, K, C)
for b in range(B):
    for k in range(K):
        r, c = inds[b, k].item() // W, inds[b, k].item() % W
        ref[b, k] = bev[b, :, r, c]
print("与朴素循环一致 :", torch.allclose(out, ref))       # True

# ---- 演示不加 .contiguous() 会怎样 ----
try:
    bad = bev.permute(0, 2, 3, 1).view(B, -1, C)
except RuntimeError as e:
    print("\n不加 contiguous 的报错 :", str(e)[:80], "...")
# 预期：RuntimeError: view size is not compatible with input tensor's size and stride...

# ---- 数据量对比：为什么要在 decode 里抠 ----
print("\n整张 BEV 特征元素数 :", bev.numel())              # 802816
print("抠出后元素数        :", out.numel())               # 2048
print("压缩比              :", bev.numel() / out.numel())  # 392.0
