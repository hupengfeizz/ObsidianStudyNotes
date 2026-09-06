"""densebev代码串讲 逐句精讲 · Ch8 多视角与RC与模态融合 · 练习4：352_64_降通道_反卷积回到_0_4m_网格
跑法：conda activate yolov8 && python ch08-04_352_64_降通道_反卷积回到_0_4m_网格.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
import torch.nn as nn

x = torch.randn(3, 1, 11, 32, 224, 112)          # 8.3 结束: (bs*t, frame, cam, C, H, W)
bs, h, w = x.shape[0], x.shape[-2], x.shape[-1]

feat_multiview = x.view(bs, -1, h, w)             # "融合"其实只是 view
print(feat_multiview.shape)                       # 预期: torch.Size([3, 352, 224, 112])
assert feat_multiview.shape[1] == 1 * 11 * 32     # 352 会讲故事: frame*cam*C

increase_channel = nn.Conv2d(352, 64, kernel_size=1)   # 名叫increase实为降: 352->64
feat_multiview = increase_channel(feat_multiview)
print(feat_multiview.shape)                       # 预期: torch.Size([3, 64, 224, 112])

deblock = nn.Sequential(                          # multifusion.py 的 self.deblock
    nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
)
bev_multiview = deblock(feat_multiview)
print(bev_multiview.shape)                        # 预期: torch.Size([3, 64, 448, 224])
print("分辨率: 179.2m/448 =", 179.2/448, "m/格")   # 预期: 0.4 m/格
