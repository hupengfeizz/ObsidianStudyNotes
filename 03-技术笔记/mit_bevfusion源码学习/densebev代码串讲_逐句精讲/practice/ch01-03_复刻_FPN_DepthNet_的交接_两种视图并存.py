"""densebev代码串讲 逐句精讲 · Ch1 DepthNet网络结构 · 练习3：复刻_FPN_DepthNet_的交接_两种视图并存
跑法：conda activate yolov8 && python ch01-03_复刻_FPN_DepthNet_的交接_两种视图并存.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, T, V, C, H, W = 1, 3, 7, 128, 88, 160
N = B * T * V                                     # 21

outs = [torch.randn(N, 128, 88, 160),             # 8×  (过了fpn_conv,128ch)
        torch.randn(N, 256, 44, 80),              # 16× (lateral,256ch)
        torch.randn(N, 256, 22, 40)]              # 32× (lateral,256ch)

depth_input = outs                                # 传全家桶
main_feat   = outs[0]                             # 主线只认8×

num_views = V
bst = main_feat.shape[0] // num_views             # 21//7=3 (=B*T)
img_feat_5d = main_feat.view(bst, num_views, *main_feat.shape[1:])

print("depth_input 是", type(depth_input).__name__, "长度", len(depth_input))
try:
    depth_input.shape                             # 复刻讲者的现场翻车
except AttributeError as e:
    print("AttributeError:", e)
print([tuple(i.shape) for i in depth_input])
print("主线5维视图:", tuple(img_feat_5d.shape))
# 预期输出：
# depth_input 是 list 长度 3
# AttributeError: 'list' object has no attribute 'shape'
# [(21, 128, 88, 160), (21, 256, 44, 80), (21, 256, 22, 40)]
# 主线5维视图: (3, 7, 128, 88, 160)
