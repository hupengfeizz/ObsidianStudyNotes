"""densebev代码串讲 逐句精讲 · Ch4 Radar代码实走与远距离切分 · 练习3：手写_get_paddings_indicator_并抹零
跑法：conda activate yolov8 && python ch04-03_手写_get_paddings_indicator_并抹零.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def get_paddings_indicator(actual_num, max_num, axis=0):
    """actual_num: (N,) 每个voxel真实点数; max_num: 容量上限16 → (N,16) bool"""
    actual_num = torch.unsqueeze(actual_num, axis + 1)          # N → N×1
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    arange = torch.arange(max_num, dtype=torch.int,
                          device=actual_num.device).view(max_num_shape)  # 1×16
    return actual_num.int() > arange                            # 广播比较 → N×16

num_points = torch.tensor([16, 3, 7])            # 视频里是 [16,16,16](mock全满)
mask = get_paddings_indicator(num_points, 16, axis=0)
print(mask.shape)                                # 预期: torch.Size([3, 16])
print(mask.sum(dim=1).tolist())                  # 预期: [16, 3, 7] 每行True数=真实点数

features = torch.randn(3, 16, 10)                # 对应调试里的 [3,16,10]
features = features * mask.unsqueeze(-1).float() # N×16×1 广播到 N×16×10
print(features.shape)                            # 预期: torch.Size([3, 16, 10])
print(bool((features[1, 3:] == 0).all()))        # 预期: True ← 第1个voxel只有3个真点,后13行全0
