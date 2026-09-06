"""densebev代码串讲 逐句精讲 · Ch9 MemoryManager与时序融合 · 练习1：十行复现_MemoryManager_滑窗
跑法：conda activate yolov8 && python ch09-01_十行复现_MemoryManager_滑窗.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

class MiniMemoryManager:
    """DenseMemoryManager 迷你版：训练=透传；10Hz推理=存当前帧、取历史两帧"""
    def __init__(self, temporal_num=3, case_10hz_infer=True):
        self.temporal_num, self.case_10hz_infer = temporal_num, case_10hz_infer
        self.memory = []                              # 对应 memory_inputs_dict（简化成单特征）

    def forward(self, cur_feat):
        if not self.case_10hz_infer:                  # 训练：输入是什么输出就是什么
            return cur_feat
        if len(self.memory) == 0:                     # 对应 init_memory（首帧冷启动）
            self.memory = [cur_feat.clone() for _ in range(self.temporal_num - 1)]
        self.memory.append(cur_feat)                  # 对应 saveHistoryMemory
        out = torch.cat(self.memory, dim=0)           # [-2帧,-1帧,当前帧] 折叠进batch维
        self.memory.pop(0)                            # 对应 pop(0)：淘汰最老帧
        return out

mm = MiniMemoryManager()
for t in range(4):
    cur = torch.full((1, 2, 4, 4), float(t))          # 单帧输入，用数值当帧号
    out = mm.forward(cur)
    print(f"t={t}  out={tuple(out.shape)}  各帧帧号={[int(v) for v in out[:, 0, 0, 0]]}")
# 预期输出：
# t=0  out=(3, 2, 4, 4)  各帧帧号=[0, 0, 0]   <- 首帧用自身填满历史
# t=1  out=(3, 2, 4, 4)  各帧帧号=[0, 0, 1]
# t=2  out=(3, 2, 4, 4)  各帧帧号=[0, 1, 2]
# t=3  out=(3, 2, 4, 4)  各帧帧号=[1, 2, 3]   <- 稳态滑窗：永远 [t-2, t-1, t]
