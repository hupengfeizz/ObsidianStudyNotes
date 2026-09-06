# 阶段 0 基准测量结果

数据: 全量 nuScenes trainval (28130 帧)   GPU: RTX 4060 Ti 16G   2026-08-22

| 模型 | batch | s/iter | s/帧 | data_time | 显存 MB | 用 spconv | 状态 |
|---|---|---|---|---|---|---|---|
| **pointpillars** | 1 | **0.218** | 0.218 | 0.041 | **1113** | 否 | ✅ 修复配置后可用 |
| **voxelnet_0p075** | 1 | **0.241** | 0.241 | 0.005 | **2048** | 是 | ⚠️ spconv 间歇崩溃 |
| voxelnet_0p075 | 2 | 0.465 | 0.233 | 0.006 | 3691 | 是 | ⚠️ 同上 |
| voxelnet_0p075 | 4 | 0.983 | 0.246 | **0.209** | 5310 | 是 | ❌ 崩溃 |
| **camera_only** | 1 | **0.289** | 0.289 | 0.004 | **4886** | 否 | ✅ |
| **convfuser (融合)** | 1 | **0.477** | 0.477 | 0.004 | **5147** | 是 | ⚠️ spconv 间歇崩溃 |
| convfuser | 2 | — | — | — | — | 是 | ❌ spconv |
| convfuser | 4 | — | — | — | — | 是 | ❌ 真 OOM (需 2.38 GiB) |

## 结论

### 1. 加大 batch 没有吞吐收益 —— 原假设被推翻

voxelnet 每帧耗时：bs1 = 0.241 / bs2 = 0.233 / bs4 = 0.246 秒。

显存虽然富余（bs=1 仅用 2048 MB / 16380 MB），但 **GPU 计算单元在 bs=1 时已经跑满**，
加 batch 只是让每步处理更多帧、每步耗时同比例增加，总吞吐几乎不变。

bs=4 时 `data_time` 从 0.005 飙到 **0.209**（占单步 21%），dataloader 反而成了瓶颈。

> **决策：所有实验一律用 bs=1**，不必为调 batch size 花时间。
> 这条推翻了计划里"batch 开到 4 能把总时长砍半"的判断——这正是阶段 0 的价值。

### 2. PointPillars 不用 spconv，是当前最稳最快的选择

0.218 s/iter、1113 MB，比 voxelnet 快 10%、省一半显存，且**免疫 spconv 崩溃**。

PointPillars 把点云压成柱子后用 2D 卷积处理，不依赖 3D 稀疏算子——这正是它当年的卖点。

### 3. 全量数据单卡训练时长（bs=1，关 CBGS，28130 iter/epoch）

| 模型 | 1 epoch |
|---|---|
| pointpillars | **1.7 h** |
| voxelnet | 1.9 h |
| camera_only | 2.3 h |
| convfuser | 3.7 h |

开 CBGS 的话 epoch 长度变成 123580 iter，pointpillars 也要 7.5 h/epoch。

## 遗留问题：spconv 间歇性崩溃

报错：`mmdet3d/ops/spconv/include/tensorview/helper_launch.h` 第 17 行，断言 `N > 0` 失败

- **影响** voxelnet / convfuser；**不影响** pointpillars / camera_only
- **表现**：同配置同数据，有时跑 400+ 步正常，有时 50 步内就崩；voxelnet bs=4 那次跑过了第 50 步才崩，说明是训练途中遇到特定样本触发
- **推测**：GPU 是 sm_89 (Ada Lovelace)，而 torch 1.10.2+cu113 只编译到 sm_86 且未嵌入 PTX。
  spconv 用哈希表生成体素索引，若该 CUDA kernel 在不受支持的架构上静默失败会返回 0 个活跃体素，正好触发断言。哈希行为与具体体素坐标相关，因此表现为数据相关的间歇失败。
- **待试方案 A**：用 `TORCH_CUDA_ARCH_LIST="8.6+PTX"` 重编译 `mmdet3d/ops` 扩展，
  让驱动在运行时把 PTX 中间码 JIT 成 sm_89。只动扩展，不动 torch 和其他包，可备份还原。
- **方案 B**：升级到 cu118+ 原生支持 sm_89。彻底但要重新匹配 torch/mmcv/mmdet3d 版本，风险大，暂不考虑。

## 已修复：PointPillars 配置 bug（仓库自带，非环境问题）

`configs/nuscenes/det/transfusion/secfpn/lidar/pointpillars.yaml` 只覆盖了
`train_cfg.grid_size = [512, 512, 1]`，**漏了 `test_cfg.grid_size`**，
于是继承上级配置的 `[1024, 1024, 1]`。

而 `mmdet3d/models/heads/bbox/transfusion.py` 第 166 行的位置编码网格取自 **test_cfg**：

```python
x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"]
y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"]
self.bev_pos = self.create_2D_grid(x_size, y_size)
```

导致位置编码 256×256 = 65536 与实际特征图 128×128 = 16384 不匹配，
报 `The size of tensor a (16384) must match the size of tensor b (65536)`。

**修复**：在 `test_cfg` 下补一行 `grid_size: [512, 512, 1]`。
原文件备份在 `pointpillars.yaml.bak`，改动可用 `git diff` 查看。
