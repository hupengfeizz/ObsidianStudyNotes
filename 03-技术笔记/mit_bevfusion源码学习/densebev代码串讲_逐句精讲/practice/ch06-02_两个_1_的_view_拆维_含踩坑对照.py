"""densebev代码串讲 逐句精讲 · Ch6 LSS投影输入准备 · 练习2：两个_1_的_view_拆维_含踩坑对照
跑法：conda activate yolov8 && python ch06-02_两个_1_的_view_拆维_含踩坑对照.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

bs = 3                      # = batch(1) × 时序3帧
fv_feat      = torch.randn(21, 128, 88, 160)   # (B*T*N, C, H, W)
depths_logit = torch.randn(21, 100, 88, 160)

_, fc, fh, fw = fv_feat.size()
fv_feat = fv_feat.view(bs, -1, fc, fh, fw)
depths_logit = depths_logit.view(bs, fv_feat.size(1), -1, fh, fw)
print(fv_feat.shape)        # 预期: torch.Size([3, 7, 128, 88, 160])
print(depths_logit.shape)   # 预期: torch.Size([3, 7, 100, 88, 160])
# 注意 bin数100 是被 -1 推断出来的，鱼眼组(32 bin)同样代码可复用

# 踩坑对照：如果当初是按 (N,T,B) 顺序折叠的，view(3,7,...) 会"形状对、数据错"
x = torch.arange(6).view(2, 3)       # 语义: (T=2帧, N=3相机)
wrong = x.view(3, 2)                 # 不报错！但帧和相机已经串台
right = x.permute(1, 0).contiguous() # 换轴必须用 permute 而不是 view
print(wrong.tolist())  # [[0,1],[2,3],[4,5]]  ← 串台
print(right.tolist())  # [[0,3],[1,4],[2,5]]  ← 正确
