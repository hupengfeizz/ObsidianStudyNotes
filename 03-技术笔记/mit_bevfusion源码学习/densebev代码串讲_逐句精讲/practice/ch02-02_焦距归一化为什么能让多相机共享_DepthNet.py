"""densebev代码串讲 逐句精讲 · Ch2 DepthGT与DepthLoss · 练习2：焦距归一化为什么能让多相机共享_DepthNet
跑法：conda activate yolov8 && python ch02-02_焦距归一化为什么能让多相机共享_DepthNet.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

# 两台相机拍同一个3米高的杆子, 距离都是30米
f_tele, f_wide = 1000., 500.        # 焦距(像素): 长焦 vs 广角
h_pix_tele = 3.0 * f_tele / 30.0    # 成像高度 = H*f/d = 100 像素
h_pix_wide = 3.0 * f_wide / 30.0    # = 50 像素  → 外观不同、深度相同!

# 用"米"当GT: 外观不同却要求输出相同 → 网络困惑
# 用 d/f 当GT: 标签 = H / h_pix, 只依赖外观(像素高度), 与相机无关
gt_tele = 30.0 / f_tele             # 0.030
gt_wide = 30.0 / f_wide             # 0.060
print(3.0 / h_pix_tele, 3.0 / h_pix_wide)   # 0.03 0.06 —— 恰好等于 d/f, 外观可直接推出标签

# 名义系数0.002(=1/500)下, 60米 → 0.12, 即本视频里 depth_max 的来历
print(60 * 0.002)                   # 0.12
