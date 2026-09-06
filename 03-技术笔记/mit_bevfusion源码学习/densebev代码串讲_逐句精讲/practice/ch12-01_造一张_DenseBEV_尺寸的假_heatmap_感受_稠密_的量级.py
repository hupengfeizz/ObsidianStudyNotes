"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习1：造一张_DenseBEV_尺寸的假_heatmap_感受_稠密_的量级
跑法：conda activate yolov8 && python ch12-01_造一张_DenseBEV_尺寸的假_heatmap_感受_稠密_的量级.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, C, H, W = 1, 5, 448, 224          # DenseBEV 的真实尺寸
logits = torch.randn(B, C, H, W) * 0.01   # 模拟"几乎没训练"的网络：logit≈0
heat   = logits.sigmoid()

print("heat.shape        :", tuple(heat.shape))       # (1, 5, 448, 224)
print("每类候选格子数     :", H * W)                    # 100352
print("全部候选(含类别)   :", C * H * W)                # 501760
print("heat 数值范围      :", heat.min().item(), heat.max().item())
print("heat 均值          :", heat.mean().item())      # ≈0.5，和讲者屏幕上的 0.5024 对得上

# 埋两个"真目标"进去看看差别
heat[0, 0, 100, 60] = 0.93     # 第0类，第100行第60列
heat[0, 3, 300, 150] = 0.88    # 第3类
print("top5 全局分数     :", heat.view(-1).topk(5).values)
# 预期输出：tensor([0.9300, 0.8800, 0.5<xx>, 0.5<xx>, 0.5<xx>])
# ——两个人造峰值远高于噪声底噪，这就是 heatmap 检测能靠 topk 直接选目标的原因
