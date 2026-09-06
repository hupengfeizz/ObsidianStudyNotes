"""densebev代码串讲 逐句精讲 · Ch2 DepthGT与DepthLoss · 练习4：dustbin_三开关与顺序陷阱
跑法：conda activate yolov8 && python ch02-04_dustbin_三开关与顺序陷阱.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
num_bins = 100
depth_target = torch.tensor([0, 8, 50, 99, 100, 100])   # 含2个dustbin像素
depth_masks  = torch.ones(6)

dustbin_zero, sup_dust, dustbin_additional = True, False, False

if dustbin_zero:                                  # L164: dustbin → 0
    m = (depth_target == num_bins).to(depth_target.dtype)
    depth_target = depth_target * (1 - m)
print(depth_target)      # tensor([ 0,  8, 50, 99,  0,  0])

if not sup_dust:                                  # L169: 本想"垃圾桶不监督"
    m2 = (depth_target == num_bins).to(depth_target.dtype)
    depth_masks = depth_masks * (1 - m2)
print(depth_masks)       # tensor([1., 1., 1., 1., 1., 1.]) ← 全1! 100早已被置0, 此块空转
                         # 顺序陷阱: dustbin_zero 在前, sup_dust 永远失效

if not dustbin_additional:                        # L178: 最后保险
    depth_target = torch.clamp(depth_target, 0, num_bins - 1).to(torch.int64)
print(depth_target.max())  # tensor(99) —— 标签保证 ∈ [0,99], 与100通道logits匹配
