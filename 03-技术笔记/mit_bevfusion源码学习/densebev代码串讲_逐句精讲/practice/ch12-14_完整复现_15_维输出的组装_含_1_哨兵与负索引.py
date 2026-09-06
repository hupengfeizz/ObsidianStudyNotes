"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习14：完整复现_15_维输出的组装_含_1_哨兵与负索引
跑法：conda activate yolov8 && python ch12-14_完整复现_15_维输出的组装_含_1_哨兵与负索引.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch
from enum import Enum

class Obj(Enum):
    x = 0; y = 1; z = 2; w = 3; l = 4; h = 5; ry = 6
    label = 7; tag = 8; conf = 9
    dir_cls = -4; mov = -3; vx = -2; vy = -1

B, MAX_NUM, CODE_DIM = 1, 256, 15

# ---------- 预分配：new_ones * -1 ----------
heatmap = torch.rand(B, 5, 448, 224)                 # 只是为了借它的 device/dtype
pred_bboxes = heatmap.new_ones([B, MAX_NUM, CODE_DIM]) * -1
print("初始化值(前3行前5列):\n", pred_bboxes[0, :3, :5])   # 全 -1

# ---------- 模拟 decode 出来的 12 个有效框 ----------
# ★ center 模式下 final_box_preds 是 9 列（帧 02_01_36 坐实）：
#   x, y, z, w, l, h, ry, vx, vy
N = 12
pred = {
    'bboxes':     torch.randn(N, 9),
    'labels':     torch.randint(0, 5, (N,)).float(),
    'scores':     torch.rand(N),
    'movements':  torch.rand(N, 1) + torch.randint(0, 2, (N, 1)).float(),  # prob+class
    'directions': torch.rand(N, 1),
}
num_pred = pred['bboxes'].shape[0]

# ---------- 逐段填充（完全照抄 centerpoint_head.py:1422-1435） ----------
pred_bboxes[0, :num_pred, :Obj.ry.value + 1] = pred['bboxes'][:, :Obj.ry.value + 1]
pred_bboxes[0, :num_pred,  Obj.label.value]  = pred['labels']
pred_bboxes[0, :num_pred,  Obj.conf.value]   = pred['scores']
pred_bboxes[0, :num_pred, -2:]               = pred['bboxes'][:, -2:]
pred_bboxes[0, :num_pred,  Obj.mov.value]     = pred['movements'][:, 0]
pred_bboxes[0, :num_pred,  Obj.dir_cls.value] = pred['directions'][:, 0]

print("\n第0个框的15维 :\n", pred_bboxes[0, 0])
print("\n第8列(tag)   :", pred_bboxes[0, :3, 8].tolist(), " ← 永远是 -1，当前未使用")
print("第10列(预留)  :", pred_bboxes[0, :3, 10].tolist(), " ← 也是 -1")
print("\n空槽位(第12行):\n", pred_bboxes[0, 12])          # 全 -1

# ---------- 下游怎么判"这一行是不是空槽" ----------
valid = pred_bboxes[0, :, Obj.conf.value] >= 0
print("\n有效框数 :", valid.sum().item(), "(期望 12)")
# ⚠ 注意不能用 x 列判断：x 完全可以是负数（车在自车后方）
bad_valid = pred_bboxes[0, :, Obj.x.value] >= 0
print("用 x 列判断会得到 :", bad_valid.sum().item(), "← 错的（约 6，因为一半的 x 是负数）")

# ---------- 关键验证：换成 corner 模式的 10 列，同一段填充代码依然正确 ----------
pred_bboxes2 = heatmap.new_ones([B, MAX_NUM, CODE_DIM]) * -1
bboxes10 = torch.randn(N, 10)          # x,y,z,w,l,h,ry, rot_short, vx,vy  ← 中间多一列
bboxes10[:, :7] = pred['bboxes'][:, :7]
bboxes10[:, -2:] = pred['bboxes'][:, -2:]
pred_bboxes2[0, :num_pred, :Obj.ry.value + 1] = bboxes10[:, :Obj.ry.value + 1]
pred_bboxes2[0, :num_pred, -2:]               = bboxes10[:, -2:]

print("\n9列 vs 10列 填出来的前7列一致 :",
      torch.allclose(pred_bboxes[0, :num_pred, :7], pred_bboxes2[0, :num_pred, :7]))
print("9列 vs 10列 填出来的 vx/vy 一致 :",
      torch.allclose(pred_bboxes[0, :num_pred, -2:], pred_bboxes2[0, :num_pred, -2:]))
print("→ 两个 True：'前缀切片 + 负索引'的写法对中间列数完全免疫")
print("→ 但也意味着 corner 模式那一列 rot_short 被静默丢弃了，从没进过 15 维输出")
