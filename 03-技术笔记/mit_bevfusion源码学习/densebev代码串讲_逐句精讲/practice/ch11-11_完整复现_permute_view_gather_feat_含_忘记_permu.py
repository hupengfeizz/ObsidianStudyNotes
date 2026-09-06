"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习11：完整复现_permute_view_gather_feat_含_忘记_permu
跑法：conda activate yolov8 && python ch11-11_完整复现_permute_view_gather_feat_含_忘记_permu.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

B, H, W, MAX_OBJ = 1, 448, 224, 256
# 直接造 concat 后的 anno_box，在 (row=171, col=112) 写入可识别的标记值 100..109
anno_box = torch.zeros(B, 10, H, W)
anno_box[0, :, 171, 112] = torch.arange(10, dtype=torch.float32) + 100   # 100..109
anno_box[0, :, 0, 0]     = torch.arange(10, dtype=torch.float32) + 200   # 干扰项

ind = torch.zeros(B, MAX_OBJ, dtype=torch.long)
ind[0, 0] = 171 * W + 112       # 38416
print("ind[0,0] =", ind[0, 0].item())

def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)

# ---------- 正确路径 ----------
pred = anno_box.permute(0, 2, 3, 1).contiguous()
print("permute 后:", tuple(pred.shape))               # (1, 448, 224, 10)
pred = pred.view(pred.size(0), -1, pred.size(3))
print("view 后   :", tuple(pred.shape))               # (1, 100352, 10)
pred = _gather_feat(pred, ind)
print("gather 后 :", tuple(pred.shape))               # (1, 256, 10)
print("目标0 取到 :", pred[0, 0].tolist())            # 期望 [100..109]

# ---------- 真正的错误路径：忘记 permute，直接 view ----------
wrong = _gather_feat(anno_box.reshape(B, -1, 10), ind)
print("忘记permute取到:", wrong[0, 0].tolist())      # 期望全 0 —— 静默错乱，不报错！
print("→ 与正确路径一致:", torch.allclose(pred[0, 0], wrong[0, 0]))   # 期望 False

# ---------- 忘记 contiguous 会怎样？实测：什么都不会发生 ----------
p_nc = anno_box.permute(0, 2, 3, 1)
print("permute 后 is_contiguous =", p_nc.is_contiguous())   # False
print("stride =", p_nc.stride())                            # (1003520, 224, 1, 100352)
v_nc = p_nc.view(B, -1, 10)                                 # ✅ 不报错
print("不加 contiguous 也能 view，结果一致:",
      torch.equal(v_nc, p_nc.contiguous().view(B, -1, 10)))  # 期望 True
print("  ↑ 因为被合并的 H,W 在原 NCHW 里本就 stride 连续，view 合法；"
      "contiguous 在这里是防御/性能，不是正确性必需")

# ---------- 等价写法对照 ----------
r1 = _gather_feat(anno_box.permute(0,2,3,1).contiguous().view(B,-1,10), ind)
r2 = _gather_feat(anno_box.permute(0,2,3,1).reshape(B,-1,10), ind)
r3 = _gather_feat(anno_box.permute(0,2,3,1).view(B,-1,10), ind)
print("三种写法一致:", torch.equal(r1, r2) and torch.equal(r2, r3))   # 期望 True

# ---------- code_weights + keep_last_dim ----------
target_box = torch.zeros(B, MAX_OBJ, 10); target_box[0, 0] = torch.arange(10.) + 100
masks = torch.zeros(B, MAX_OBJ); masks[0, 0] = 1
# ⚠ 注意 .clone()：expand_as 出来的是 view，多个位置共享同一块内存，
#    直接对它做 *= 会报 "more than one element ... refers to a single memory location"。
#    源码第 1009-1011 行之所以能原地乘，是因为 masks 的 dtype 不是 float32，
#    那里的 .float() 触发了一次真实的类型转换（=复制），顺带把 view 变成了独立张量。
#    → 这是一条隐式依赖：哪天有人把 masks 存成 float32，第 1011 行就会当场报错。
mask = masks.unsqueeze(2).expand_as(target_box).float().clone()
mask *= (~torch.isnan(target_box)).float()
code_weights = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,0.2,0.2])
bbox_weights = mask * code_weights
num = masks.sum()
loss_vec = (torch.abs(pred - target_box) * bbox_weights).sum(dim=(0,1)) / (num + 1e-4)
print("loss_bbox 向量 (keep_last_dim=True) 长度 =", loss_vec.numel())   # 10
name_and_dim = [('reg_loc',2), ('height',1), ('box_size',3), ('rot',2), ('vel',2)]
cur = 0
for name, d in name_and_dim:
    print(f"  loss_p_{name:9s} = {loss_vec[cur:cur+d].sum().item():.4f}")
    cur += d
