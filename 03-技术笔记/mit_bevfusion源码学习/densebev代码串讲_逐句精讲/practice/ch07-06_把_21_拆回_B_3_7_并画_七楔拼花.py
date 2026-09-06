"""densebev代码串讲 逐句精讲 · Ch7 LSS投影本体 · 练习6：把_21_拆回_B_3_7_并画_七楔拼花
跑法：conda activate yolov8 && python ch07-06_把_21_拆回_B_3_7_并画_七楔拼花.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch, torch.nn.functional as F, math

BT, V, C, Hb, Wb = 3, 7, 4, 64, 32
# 伪造 grid: 每路相机一个不同朝向的楔形(复用练习ch7-5思路, 此处简化为平移遮罩)
bev21 = torch.zeros(V * BT, C, Hb, Wb)
for cam in range(V):
    r0 = (cam * Hb) // V
    bev21[cam*BT:(cam+1)*BT, :, r0:r0+Hb//V, :] = cam + 1   # 每路占一条横带,值=相机号

# === 视频第105~117行的拆回逻辑 ===
split = torch.split(bev21, bev21.shape[0] // V)      # 7 份, 每份 (3,C,H,W)
out = torch.cat([s.unsqueeze(1) for s in split], 1)  # (3,7,C,H,W)  相机维回到 dim1
print(out.shape)                                     # torch.Size([3, 7, 4, 64, 32])

# 验证没有"相机窜位": 第 j 路的带子值应恒为 j+1
for j in range(V):
    band = out[0, j].amax()
    assert band == j + 1, f"相机{j}窜位!"
print("7 路相机各归各位 ✓")
# 反例: 直接 reshape(BT, V, ...) 会窜位 —— 自己打开注释体会
# wrong = bev21.reshape(BT, V, C, Hb, Wb); print((wrong[0,0]).amax())  # ≠1
# 预期输出:
# torch.Size([3, 7, 4, 64, 32])
# 7 路相机各归各位 ✓
