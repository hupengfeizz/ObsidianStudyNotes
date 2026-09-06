"""densebev代码串讲 逐句精讲 · Ch6 LSS投影输入准备 · 练习0：用_inputs_约定还原模块入口
跑法：conda activate yolov8 && python ch06-00_用_inputs_约定还原模块入口.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

# ModuleBevProject.forward(self, *inputs, **labels) 的"按位置打包"约定迷你复现
def forward(*inputs, **labels):
    print(f"收到 {len(inputs)} 个位置输入")
    for i, t in enumerate(inputs):
        print(f"  inputs[{i}]: {tuple(t.shape)}")

B3 = 1 * 3  # batch=1, 时序3帧 → 折叠进第0维
forward(
    torch.randn(B3 * 7, 128, 88, 160),   # inputs[0] 针孔图像特征
    torch.randn(B3 * 4, 128, 64, 96),    # inputs[1] 鱼眼图像特征
    torch.randn(B3 * 7, 100, 88, 160),   # inputs[2] 针孔depth logits
    torch.randn(B3 * 4, 32, 64, 96),     # inputs[3] 鱼眼depth logits
    torch.randn(B3, 7, 224, 112, 2),     # inputs[4] 针孔grid map
    torch.randn(B3, 4, 16, 16, 2),       # inputs[5] 鱼眼grid map(尺寸存疑,见6-8)
)
# 预期输出：6行shape。体会"全靠位置索引"的脆弱性：换一个顺序,后面全错。
