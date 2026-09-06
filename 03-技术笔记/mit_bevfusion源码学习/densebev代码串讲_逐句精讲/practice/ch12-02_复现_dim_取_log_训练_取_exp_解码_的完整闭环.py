"""densebev代码串讲 逐句精讲 · Ch12 Box解码与收尾QA · 练习2：复现_dim_取_log_训练_取_exp_解码_的完整闭环
跑法：conda activate yolov8 && python ch12-02_复现_dim_取_log_训练_取_exp_解码_的完整闭环.py
只依赖 torch，CPU 即可，不占 GPU。预期输出见代码内注释。
"""
import torch

# ---------- 训练侧：GT 取 log ----------
gt_dim = torch.tensor([[4.6, 1.9, 1.6],     # 轿车 l,w,h
                       [12.0, 2.6, 3.4],    # 公交
                       [0.6, 0.6, 1.7]])    # 行人
gt_log = torch.log(gt_dim)
print("GT 取 log 后 :\n", gt_log)
# 预期：tensor([[1.5261, 0.6419, 0.4700],
#               [2.4849, 0.9555, 1.2238],
#               [-0.5108, -0.5108, 0.5306]])
# 注意动态范围：线性空间 0.6~12（20倍），log 空间 -0.51~2.48（区间宽度仅 3）

# ---------- 网络输出（模拟"每个都差 0.1 的 log 误差"）----------
pred_log = gt_log + 0.1
pred_dim = torch.exp(pred_log)                      # 解码：exp 还原
print("\n解码后的尺寸 :\n", pred_dim)
print("相对误差     :\n", (pred_dim - gt_dim) / gt_dim)
# 预期：三行相对误差全是 0.1052（=e^0.1-1），与目标大小无关
#      —— 这就是"log 空间 L1 = 惩罚相对误差"的直接证据

# ---------- 反例：若在线性空间同样差 0.1 米 ----------
pred_lin = gt_dim + 0.1
print("\n线性空间同样差0.1米时的相对误差 :\n", (pred_lin - gt_dim) / gt_dim)
# 预期：轿车长 0.0217、公交长 0.0083、行人长 0.1667
#      —— 小目标被严重"欺负"，网络会学得偏向大目标
