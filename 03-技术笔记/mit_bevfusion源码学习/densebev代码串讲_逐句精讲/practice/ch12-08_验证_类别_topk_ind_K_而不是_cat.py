"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习8：验证_类别_topk_ind_K_而不是_cat
跑法：conda activate yolov8 && python ch12-08_验证_类别_topk_ind_K_而不是_cat.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
B, C, K = 1, 5, 256
# 构造一个"第 3 类分数全场最高"的极端情况
topk_scores = torch.zeros(B, C, K)
topk_scores[0, 3, :] = 0.9          # 类 3 的 256 个候选全是 0.9
topk_scores[0, 0, :] = 0.8

topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), K)
cls_right = (topk_ind / torch.tensor(K, dtype=torch.float)).int()   # ✅ 除以 K
cls_wrong = (topk_ind / torch.tensor(C, dtype=torch.float)).int()   # ❌ 除以 cat

print("正确类别(除以K=256) 的取值集合 :", set(cls_right[0].tolist()))   # {3}
print("错误类别(除以C=5  ) 的取值集合 :", sorted(set(cls_wrong[0].tolist()))[:5], "...")
# 预期：正确的全是 3（因为 topk_ind ∈ [768,1024)，//256 = 3）
#      错误的会散成 153~204 这种荒谬的"类别号"
