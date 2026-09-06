"""densebev代码串讲 逐句精讲 · Ch11 Loss全解 · 练习8：速度权重公式复现_含非对称欠估惩罚
跑法：conda activate yolov8 && python ch11-08_速度权重公式复现_含非对称欠估惩罚.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

def vel_weight(v_gt_norm, v_pred_norm, neg_vel_loss=0.0):
    """复刻画面 01:45:05 第 936-941 行"""
    w = (v_gt_norm > 0.05) * (v_gt_norm < 1.0) * 1.0 \
      + (v_gt_norm < 0.05) * 0.5
    if neg_vel_loss > 0.0:
        w = w + ((v_gt_norm > 0.2) * (v_pred_norm - v_gt_norm < 0.0)).float() \
              * (torch.abs(v_pred_norm - v_gt_norm) * neg_vel_loss / (v_gt_norm + 1e-6))
    return (w + 1).float()

cases = [
    ("完全静止 v=0.00", 0.00, 0.00),
    ("蠕行     v=0.50", 0.50, 0.50),
    ("正常     v=5.00", 5.00, 5.00),
    ("欠估10%  v=5.00", 5.00, 4.50),
    ("欠估50%  v=5.00", 5.00, 2.50),
    ("高估50%  v=5.00", 5.00, 7.50),
    ("欠估但v小 v=0.10", 0.10, 0.00),
]
print(f"{'场景':<20}{'w(neg=0)':>10}{'w(neg=1)':>10}")
for name, g, p in cases:
    g_t, p_t = torch.tensor(g), torch.tensor(p)
    print(f"{name:<20}{vel_weight(g_t,p_t,0.0).item():>10.3f}{vel_weight(g_t,p_t,1.0).item():>10.3f}")

# ---- 完整 dense vel loss ----
B, H, W = 1, 64, 32
v_gt = torch.zeros(B, 2, H, W); v_gt[0, 0, 20:30, 10:15] = 5.0     # 一辆 5m/s 的车
v_pred = torch.zeros(B, 2, H, W); v_pred[0, 0, 20:30, 10:15] = 2.5 # 严重欠估
attr_mask = torch.zeros(B, 2, H, W); attr_mask[0, :, 20:30, 10:15] = 1.0

INVALID_VELOCITY = 50.0
v_attr_mask = attr_mask * (v_gt[:, 0] < INVALID_VELOCITY).float().unsqueeze(1)
g_n = torch.norm(v_gt, p=2, dim=1); p_n = torch.norm(v_pred, p=2, dim=1)
for neg in (0.0, 1.0):
    m = v_attr_mask * vel_weight(g_n, p_n, neg).unsqueeze(1)
    af = (m > 0).float().sum().item()
    loss = (torch.abs(v_pred - v_gt) * m).sum() / max(af, 1) * 5
    print(f"neg_vel_loss={neg}  vel_dense_loss = {loss.item():.4f}")

# ---- 哨兵值污染演示 ----
v_gt_bad = v_gt.clone(); v_gt_bad[0, 0, 40, 20] = 999.0
attr_mask2 = attr_mask.clone(); attr_mask2[0, :, 40, 20] = 1.0
raw   = (torch.abs(v_pred - v_gt_bad) * attr_mask2).sum().item()
filt_mask = attr_mask2 * (v_gt_bad[:, 0] < INVALID_VELOCITY).float().unsqueeze(1)
filt  = (torch.abs(v_pred - v_gt_bad) * filt_mask).sum().item()
print(f"不过滤哨兵值 sum={raw:.1f}   过滤后 sum={filt:.1f}   ← 差 {raw-filt:.0f}")
