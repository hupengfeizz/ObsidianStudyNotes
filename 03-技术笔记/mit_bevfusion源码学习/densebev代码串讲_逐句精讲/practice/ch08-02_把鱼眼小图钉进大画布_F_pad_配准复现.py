"""densebev代码串讲 逐句精讲 · Ch8 多视角与RC与模态融合 · 练习2：把鱼眼小图钉进大画布_F_pad_配准复现
跑法：conda activate yolov8 && python ch08-02_把鱼眼小图钉进大画布_F_pad_配准复现.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn.functional as F

# 鱼眼 BEV: (bs*T=3, N=4相机, C=32, 16, 16), 值全填 1 便于观察落点
fisheye = torch.ones(3, 4, 32, 16, 16)
pad = (48, 48, 111, 97)          # 帧 00_55_34 调试悬浮窗实录: (左, 右, 上, 下)
padded = F.pad(fisheye, pad, "constant", 0)
print(padded.shape)               # 预期: torch.Size([3, 4, 32, 224, 112])

# 验证 patch 恰好以自车为中心:
# 0.8m 网格: 自车前方 95.4/0.8≈119 格 -> 自车行号 119; 横向中心列号 56
nonzero = padded[0, 0, 0]         # 取一张 224x112 切片
rows = nonzero.sum(dim=1).nonzero().flatten()
cols = nonzero.sum(dim=0).nonzero().flatten()
print("patch 行范围:", rows.min().item(), "~", rows.max().item())   # 预期: 111 ~ 126
print("patch 列范围:", cols.min().item(), "~", cols.max().item())   # 预期: 48 ~ 63
print("行中心:", (rows.min()+rows.max()).item()/2, "≈ 自车行 118.5~119")
print("列中心:", (cols.min()+cols.max()).item()/2, "≈ 横向中心 55.5~56")

# 对照: 针孔组 pad=(0,0,0,0), havpad==0 直接跳过 F.pad
pin_pad = (0, 0, 0, 0)
print("针孔需要 pad 吗:", sum(pin_pad) != 0)   # 预期: False
