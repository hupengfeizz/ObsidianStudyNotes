"""densebev代码串讲 逐句精讲 · Ch8 多视角与RC与模态融合 · 练习3：双_list_view_cat_dim_2_全流程迷你复现
跑法：conda activate yolov8 && python ch08-03_双_list_view_cat_dim_2_全流程迷你复现.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

bs_t = 3            # bs=1, t=3 折叠
frame_num = 1       # 本配置: 时序折在 batch, frame 维恒为 1
C, H, W = 32, 224, 112

# 模拟 8.2 结束时的 feat_and_mask 双 list（鱼眼已 pad 到 224x112）
feat_and_mask = [
    [torch.randn(bs_t, 7, C, H, W), torch.randn(bs_t, 4, C, H, W)],  # [0]: 特征仓
    [torch.ones(bs_t, 7, H, W),     torch.ones(bs_t, 4, H, W)],      # [1]: mask 仓
]

bs, tn, c, h, w = feat_and_mask[0][0].shape
assert tn % frame_num == 0
feats = [x.view(bs, frame_num, -1, c, h, w) for x in feat_and_mask[0]]
masks = [x.view(bs, frame_num, -1, h, w) for x in feat_and_mask[1]]
print([tuple(x.shape) for x in feats])
# 预期: [(3, 1, 7, 32, 224, 112), (3, 1, 4, 32, 224, 112)]  <- 帧00_58_08调试台原文

feats = torch.cat(feats, dim=2)   # 沿相机(view)维拼接
masks = torch.cat(masks, dim=2)
print(feats.shape)                # 预期: torch.Size([3, 1, 11, 32, 224, 112])
print(masks.shape)                # 预期: torch.Size([3, 1, 11, 224, 112])

# 体会 mask 的休眠用途(use_concat=False 时的平均融合):
fused_avg = feats.sum(dim=2) / masks.float().sum(dim=2).unsqueeze(2).clamp(min=1.0)
print(fused_avg.shape)            # 预期: torch.Size([3, 1, 32, 224, 112])
