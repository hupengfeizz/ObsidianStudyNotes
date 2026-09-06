"""densebev代码串讲 逐句精讲 · Ch6 LSS投影输入准备 · 练习3：softmax_dustbin_zero_的能量语义
跑法：conda activate yolov8 && python ch06-03_softmax_dustbin_zero_的能量语义.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn.functional as F

bs, n, D, h, w = 3, 7, 100, 2, 3          # 空间维缩小便于打印
depths_logit = torch.randn(bs, n, D, h, w)
# 人为制造一个"天空像素"：bin0 打分极高
depths_logit[0, 0, 0, 0, 0] = 8.0

depth_prob_ = F.softmax(depths_logit, dim=2)
print(depth_prob_.sum(2)[0, 0, 0, 0].item())      # 预期: 1.0000 (softmax后归一)

# dustbin_zero: 保形清零而非切片
depth_prob = depth_prob_.new_zeros(depth_prob_.shape)
depth_prob[:, :, 1:] += depth_prob_[:, :, 1:]

print(depth_prob.shape)                            # 预期: torch.Size([3,7,100,2,3]) 形状不变
print(depth_prob[:, :, 0].abs().max().item())      # 预期: 0.0  (bin0全零)
print(depth_prob.sum(2)[0, 0, 0, 0].item())        # 预期: ≈0.02 (天空像素总能量≈被熄灭)
print(depth_prob.sum(2)[0, 0, 1, 1].item())        # 预期: ≈0.99 (普通像素能量基本保留)
# 结论:丢bin0 = 给每个像素乘了一个(1-p_无效)的软门控,天空自动"隐形"
