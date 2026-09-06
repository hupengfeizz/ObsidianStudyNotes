"""densebev代码串讲 逐句精讲 · Ch7 LSS投影本体 · 练习1：占位维与_帧折进_batch
跑法：conda activate yolov8 && python ch07-01_占位维与_帧折进_batch.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, T, V, C, H, W = 1, 3, 7, 128, 88, 160
# DepthNet 阶段: 帧和相机全在 dim0 (21 路)
feat_flat = torch.randn(B * T * V, C, H, W)          # (21,128,88,160)
# 进 projector 前: 拆出相机维,再补 frame_num=1 占位维
fv_feat = feat_flat.reshape(B * T, V, C, H, W)        # (3,7,128,88,160)  <- 调试台看到的
fv_feat = fv_feat.unsqueeze(1)                        # (3,1,7,128,88,160) <- 讲者口中的
print(fv_feat.shape)   # torch.Size([3, 1, 7, 128, 88, 160])

frame_num = fv_feat.shape[1]
for i in range(frame_num):          # 只会跑 i=0
    one = fv_feat[:, i, 2, ...]     # 取第 3 路相机
    print(i, one.shape)             # 0 torch.Size([3, 128, 88, 160])
# 预期输出:
# torch.Size([3, 1, 7, 128, 88, 160])
# 0 torch.Size([3, 128, 88, 160])
