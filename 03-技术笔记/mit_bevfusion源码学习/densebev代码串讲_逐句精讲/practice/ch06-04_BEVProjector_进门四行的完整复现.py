"""densebev代码串讲 逐句精讲 · Ch6 LSS投影输入准备 · 练习4：BEVProjector_进门四行的完整复现
跑法：conda activate yolov8 && python ch06-04_BEVProjector_进门四行的完整复现.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def bev_projector_entry(fv_feat, depth_probs, frame_num=1):
    """复现帧00_40_58第62-70行"""
    bs, nt, fc, fh, fw = fv_feat.size()
    assert nt % frame_num == 0, "相机总数必须能被帧数整除"
    num_views = nt // frame_num
    fv_feat = fv_feat.view(bs, frame_num, num_views, fc, fh, fw)
    _, _, dc, dh, dw = depth_probs.size()
    depth_probs = depth_probs.view(bs, frame_num, num_views, dc, dh, dw)
    return fv_feat, depth_probs

fv  = torch.randn(3, 7, 128, 88, 160)     # 本视频配置: 3帧已折入bs
dep = torch.randn(3, 7, 100, 88, 160)
f, d = bev_projector_entry(fv, dep, frame_num=1)
print(f.shape)  # 预期: torch.Size([3, 1, 7, 128, 88, 160])
print(d.shape)  # 预期: torch.Size([3, 1, 7, 100, 88, 160])

# 同一函数服务另一种配置: 不折帧、t=3 直进 (bs=1, nt=21)
f2, d2 = bev_projector_entry(torch.randn(1, 21, 128, 88, 160),
                             torch.randn(1, 21, 100, 88, 160), frame_num=3)
print(f2.shape)  # 预期: torch.Size([1, 3, 7, 128, 88, 160])
# 预告Ch7: 外积 f.unsqueeze(4)*d.unsqueeze(3) → (…,7,128,100,88,160), 自己算算多少GB?
