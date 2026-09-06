"""densebev代码串讲 逐句精讲 · Ch8 多视角与RC与模态融合 · 练习5：RC_加法_FeatureConcat_模态融合_含_grad_fn_侦探术
跑法：conda activate yolov8 && python ch08-05_RC_加法_FeatureConcat_模态融合_含_grad_fn_侦探术.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn as nn

B, C, H, W = 3, 64, 448, 224            # 显存紧张可整体除以4: (3,16,112,56)

bev_multiview = torch.randn(B, C, H, W, requires_grad=True)  # 相机BEV(已上采样)
radar_feat    = torch.randn(B, C, H, W)                      # inputs[2]
rl_feat       = torch.randn(B, C, H, W)                      # inputs[1] lidar_parsing_embedding

# --- RC融合: 投影(shape不变) + 逐元素相加 ---
increase_channel = nn.Conv2d(C, C, 1)                 # 64->64, "加法前先投影"
radar_bev_feature = increase_channel(radar_feat)
print(radar_bev_feature.shape)                        # 预期: torch.Size([3, 64, 448, 224]) 不变
x = bev_multiview + radar_bev_feature                 # RC = element-wise add
print(x.shape, "通道数不涨:", x.shape[1] == C)         # 预期: (3,64,448,224) True

# --- 模态融合: FeatureConcat 仅是cat, 无降channel ---
class FeatureConcat(nn.Module):
    def forward(self, point_bev_feat, cam_bev_feat):
        return torch.cat([point_bev_feat, cam_bev_feat], dim=1)

fusion = FeatureConcat()
out = fusion(rl_feat, x)
print(out.shape)              # 预期: torch.Size([3, 128, 448, 224])  128=64+64
print(out.grad_fn)            # 预期: <CatBackward0 ...>  <- 帧01_01_30悬浮窗同款证据:
                              #  最后算子是cat, 证明本步没有卷积降通道
