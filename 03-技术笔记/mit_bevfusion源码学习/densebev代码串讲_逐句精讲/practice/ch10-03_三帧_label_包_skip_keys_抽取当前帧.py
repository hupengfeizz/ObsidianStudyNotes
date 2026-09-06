"""densebev代码串讲 逐句精讲 · Ch10 BEVUNet与CenterPoint检测头 · 练习3：三帧_label_包_skip_keys_抽取当前帧
跑法：conda activate yolov8 && python ch10-03_三帧_label_包_skip_keys_抽取当前帧.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def label_extract(labels: dict, skip_keys=()):
    """复刻 seq_info_extract.label_extract 的核心语义：
    三帧堆叠的 key 取当前帧（最后一帧），skip_keys 里的单帧 key 原样保留。"""
    out = {}
    for k, v in labels.items():
        if k in skip_keys:
            out[k] = v                    # 天生单帧：不动
        else:
            out[k] = v[-1]                # 三帧结构：取当前帧
    return out

labels = {
    'depth_gt':   torch.randn(3, 21, 88, 160),   # 三帧堆叠（图像模块的额外label）
    'ego_pose':   torch.randn(3, 4, 4),          # 三帧位姿
    'obj_label':  torch.randint(0, 5, (7,)),     # 单帧：当前帧7个目标的类别
    'centerpoint_head_gt_rl': torch.randn(5, 56, 28),  # 单帧：预生成的BEV heatmap GT
}
skip_keys = ['obj_label', 'centerpoint_head_gt_rl']
cur = label_extract(labels, skip_keys)

for k, v in cur.items():
    print(f'{k:24s} {tuple(v.shape)}')
# 预期输出：
# depth_gt                 (21, 88, 160)   ← 3帧 -> 当前帧
# ego_pose                 (4, 4)
# obj_label                (7,)            ← skip：原样
# centerpoint_head_gt_rl   (5, 56, 28)     ← skip：原样
