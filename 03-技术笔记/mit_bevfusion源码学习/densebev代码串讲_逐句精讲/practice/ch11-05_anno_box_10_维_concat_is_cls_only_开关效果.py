"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习5：anno_box_10_维_concat_is_cls_only_开关效果
跑法：conda activate yolov8 && python ch11-05_anno_box_10_维_concat_is_cls_only_开关效果.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, H, W, MAX_OBJ = 2, 448, 224, 256
preds = {
    'reg':    torch.randn(B, 2, H, W),
    'height': torch.randn(B, 1, H, W),
    'dim':    torch.randn(B, 3, H, W),
    'rot':    torch.randn(B, 2, H, W),
    'vel':    torch.randn(B, 2, H, W),
}
# 画面 01:48:46 第 1021-1025 行
anno_box = torch.cat((preds['reg'], preds['height'], preds['dim'],
                      preds['rot'], preds['vel']), dim=1)
print("anno_box.shape =", tuple(anno_box.shape))      # 期望 (2, 10, 448, 224)

LAYOUT = ['dx','dy','z','log_l','log_w','log_h','sin','cos','vx','vy']
print("切片自检:")
print("  pred[..., -4:-2] →", LAYOUT[-4:-2])          # 期望 ['sin','cos']
print("  pred[..., -2:]   →", LAYOUT[-2:])            # 期望 ['vx','vy']
print("  loss_bbox[:6]    →", LAYOUT[:6])             # 期望 dx..log_h

# ---- is_cls_only 开关 ----
# ⚠ 形状必须是 [B,1]：源码里 not_cls_only 要同时乘 [B,256]、[B,256,1]、[B,C,H,W]
#    三种秩，只有 [B,1] 能让三处同时合法（见 §11-4 的四处交叉验证表）
is_cls_only  = torch.tensor([[0.0], [1.0]])           # [2,1]：样本1 只做分类
not_cls_only = 1 - is_cls_only

masks = torch.zeros(B, MAX_OBJ); masks[0, :7] = 1; masks[1, :5] = 1
num = (masks * not_cls_only).float().sum()            # [2,256] × [2,1] ✅
print("参与回归的目标数 num =", num.item())            # 期望 7.0（样本1 的 5 个被废掉）

attr_mask = torch.ones(B, 2, H, W) * not_cls_only[..., None, None]   # [2,1,1,1] ✅
print("样本0 attr_mask 均值 =", attr_mask[0].mean().item())   # 期望 1.0
print("样本1 attr_mask 均值 =", attr_mask[1].mean().item())   # 期望 0.0

# ---- 反面演示：写成 [B] 会静默算错（B 恰好 == 通道数 2 时不报错！）----
bad = torch.ones(B, 2, H, W) * (1 - torch.tensor([0.0, 1.0]))[..., None, None]
print("[B] 版 样本0 均值 =", bad[0].mean().item(), " 样本1 均值 =", bad[1].mean().item())
print("  ↑ 两个都是 0.5：它沿【通道维】广播了，不是沿 batch 维——不报错但全错")

# ---- 帧级 OR 累加（load_object.py 第 272 行）----
frame_objs = [{'cls_only': False}, {'cls_only': False}, {'cls_only': True}]
flag = False
for o in frame_objs:
    flag = flag or o.get('cls_only', False)
print("这一帧 is_cls_only =", flag, " → 3 个目标全部丢失回归监督")  # 期望 True
