---
tags: [BEV, BEVFusion, 训练, nuScenes, 4060]
创建: 2026-08-22
状态: 阶段2已完成，待执行阶段0
---

# BEVFusion 训练计划（4060 Ti 工作站）

> 目标不是复现 SOTA，是**把训练流程练熟 + 跑出一批能对比的实验数据**。
> 关联笔记：[[BEVFusion数据流_源码地图与pipeline流水表]]、[[BEVFusion两个版本对照表]]

---

## 0. 先说清楚：这台机器能做什么，不能做什么

### 硬件与现状盘点（2026-08-22 实测）

| 项 | 实际情况 |
|---|---|
| GPU | RTX 4060 Ti，**16 GB** 显存，驱动 595.58.03 |
| CPU / 内存 | 16 核 / 31 GB（可用 21 GB） |
| 磁盘 | 3.7 T，已用 360 G，**可用 3.2 T** |
| 代码 | `/home/hpf/project/bevfusion-main`（MIT 版，ICRA 2023） |
| conda 环境 | **`bev`**（torch 1.10.2+cu113）。⚠️ `bevfusion` 环境是坏的，别用 |
| 数据 | `data/nuscenes/` **全量 trainval 398 G 已就位**（2026-08-22）+ `data/nuscenes-mini/` |
| 预训练权重 | `swint-nuimages-pretrained.pth`(106M) / `lidar-only-det.pth`(32M) / `bevfusion-det.pth`(157M) |
| 历史训练 | `runs/run-9eaf327b`：mini 上 fusion 训了 6 epoch，有 `epoch_6.pth` |

### 实测性能基准（来自 run-9eaf327b 日志）

- 模型：camera+lidar convfuser（完整 BEVFusion）
- `samples_per_gpu: 1`，**0.481 s/iter**，显存占用 **5161 MB**
- mini train 323 帧，CBGS 重采样后 **1630 iter/epoch** ≈ 13 分钟/epoch

### ❌ 做不到的事

1. **没有 AMP / fp16 训练**。仓库里搜不到 `autocast`/`GradScaler`，训练全程 fp32。
2. **全量数据 + CBGS 不可行**：28130 帧 × CBGS(≈4.5x) ≈ 127k iter × 0.481s ≈ **17 h/epoch**。
3. **复现论文指标不可行**：MIT 是 8×A100 训 20+ epoch。

### ✅ 做得到的事

1. mini 上把 train / eval / visualize / tensorboard 全流程练到闭眼能敲。
2. 全量数据**关掉 CBGS**：28130 iter × 0.481s ≈ **3.8 h/epoch**，一夜能训 6 epoch。
3. 在**固定子集**上跑一整套受控消融实验 → 这才是本计划的核心产出。

### ⚠️ 一个必须纠正的认知

`run-9eaf327b` 的 **NDS 0.4967 / mAP 0.4367 不是有效结论**：

- mini val 只有 **81 帧 / 2 个场景**，统计量太小
- 该 run `load_from: pretrained/lidar-only-det.pth`，那是 MIT 在**全量数据**上训好的
- 也就是说这个分数主要来自别人的预训练权重，不是你这 6 个 epoch 训出来的

**任何在 mini 上得到的精度数字，都只能用来验证"流程没崩"，不能用来下结论。**

---

## 阶段 0：基准测量（半天）

**目的**：把后面所有时间估算建立在实测上，而不是我的估计上。

### 0.1 进环境

```bash
ssh 4060 -t 'cd ~/project/bevfusion-main && source ~/miniconda3/etc/profile.d/conda.sh && conda activate bev && exec zsh'
```

### 0.2 前置：把 dataset_root 切到全量数据（已完成 2026-08-22）

`configs/nuscenes/default.yaml` 第 2 行原本是 `data/nuscenes-mini/`，
**不改的话所有基准测量都测的是 mini 的速度，白测**。已改为：

```yaml
dataset_root: data/nuscenes/          # 全量 trainval。要用 mini 加 --dataset_root data/nuscenes-mini/
```

原文件备份在 `configs/nuscenes/default.yaml.bak`。

改完验证解析结果（三个都应指向 `data/nuscenes/`）：

```bash
python -c "
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval
configs.load('configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml', recursive=True)
cfg = Config(recursive_eval(configs))
for k in ['train','val','test']:
    d = cfg.data[k]; d = d.get('dataset', d)
    print(k, d.get('ann_file'))
"
```

> 补充：`configs/nuscenes/default.yaml` 第 289 行，`data.test` 的 `ann_file` 指向的是
> **`nuscenes_infos_val.pkl`**，不是 test 集。所以没拷 nuScenes test 数据不影响 `tools/test.py`。

### 0.3 用脚本自动测（推荐）

**脚本位置**：

| 机器 | 完整路径 |
|---|---|
| 工作站（执行） | `/home/hpf/project/bevfusion-main/tools/bench_stage0.sh` |
| Mac（源码备份） | `/Users/apple/Downloads/36/bench_stage0.sh` |

在工作站上执行：

```bash
cd ~/project/bevfusion-main && nohup ./tools/bench_stage0.sh > runs/bench_stage0.log 2>&1 &
```

**它做的事**：8 个「模型 × batch」组合**串行**跑，每个跑到第 150 iter（日志间隔 50，取第 3 条避开预热），
从日志抓 `time:` 和 `memory:`，然后杀掉整个进程组、等 8 秒释放显存、进入下一个。

自动处理 OOM / 异常退出 / 超时三种失败，单个组合失败不影响后续。

**测试组合**：

| 模型 | batch |
|---|---|
| pointpillars | 1 |
| voxelnet_0p075 | 1, 2, 4 |
| camera_only (swint 256×704) | 1 |
| convfuser (camera+lidar) | 1, 2, 4 |

预计 25–35 分钟。结果写入 `runs/bench_stage0_result.md`，各组合的原始日志在 `runs/bench_<模型>_bs<N>.log`。

看进度：

```bash
tail -20 ~/project/bevfusion-main/runs/bench_stage0.log
```

### 0.4 手动方式（备用，脚本挂了才用）

```bash
torchpack dist-run -np 1 python tools/train.py <config> --data.samples_per_gpu 2 --run-dir runs/bench-tmp
```

跑到日志出现第 3 条 `time:` 记录就 Ctrl-C，人工抄下 `time:` 和 `memory:`。

> ⚠️ batch size 改了，学习率要按**线性缩放**跟着调（`--optimizer.lr`），否则收敛会变差。
> 基准测量阶段只测速度不看精度，可以不调；**正式训练时必须调**。

### 0.5 验收标准

| 模型 | bs | s/iter | 显存 MB | 最大可用 bs |
|---|---|---|---|---|
| pointpillars | 1 | | | |
| voxelnet_0p075 | 1 / 2 / 4 | | | |
| camera_only | 1 | | | |
| convfuser (fusion) | 1 / 2 / 4 | | 5161 (mini 实测) | |

**产出**：这张表填满，后面 8 组实验的耗时才能准确排期。

**重点关注**：fusion 在 bs=1 只占 5.1 G / 16 G，**显存空了 2/3**。
如果 batch 能开到 4，阶段 3 的总时长可能直接砍半——从两三天变成一个晚上。

---

## 阶段 1：mini 上的流程演练（1 天）

**目的**：熟练度。这一阶段**不追求任何精度结论**。

### 1.1 完整训练

```bash
torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml --run-dir runs/drill-01
```

### 1.2 单独评测

```bash
torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml runs/drill-01/latest.pth --eval bbox
```

### 1.3 可视化（这一步最能建立直觉）

```bash
torchpack dist-run -np 1 python tools/visualize.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml --checkpoint runs/drill-01/latest.pth --mode pred --split val --out-dir viz/drill-01 --bbox-score 0.3
```

先跑 `--mode gt` 看真值长什么样，再跑 `--mode pred` 对比。

### 1.4 看曲线

```bash
tensorboard --logdir runs/ --port 6006 --bind_all
```

本地开隧道：

```bash
ssh -N -L 6006:localhost:6006 4060
```

### 1.5 故意训崩一次

把学习率放大 50 倍，观察 loss 变 NaN 的过程和日志长相：

```bash
torchpack dist-run -np 1 python tools/train.py <config> --optimizer.lr 0.01 --run-dir runs/drill-fail
```

> 面试常问"训练不收敛怎么排查"。**亲手崩过一次**，答案就有细节了。

### 验收标准

- [ ] 能不查文档敲出 train / test / visualize 三条命令
- [ ] 能看懂日志每一列（`loss_heatmap` / `matched_ious` / `grad_norm` 各是什么）
- [ ] 能解释 nuScenes 的 NDS 由哪几项加权组成（mAP + mATE/mASE/mAOE/mAVE/mAAE）
- [ ] 见过 loss 变 NaN 的样子

---

## 阶段 2：全量数据落地（1–3 天，取决于网速）

### 2.1 ⚠️ 需要你本人操作的部分

我不能代替你做这两件事：

1. 去 <https://www.nuscenes.org/nuscenes> **注册账号**
2. **同意数据使用条款**（Terms of Use）后才会出现下载链接

### 2.2 需要下载的文件

| 文件 | 大小 | 必需？ |
|---|---|---|
| `v1.0-trainval_meta.tgz` | ~400 MB | ✅ 必需 |
| `v1.0-trainval01_blobs.tgz` … `10_blobs.tgz` | 每个约 30 GB，共 ~300 GB | ✅ 必需 |
| `nuScenes-map-expansion-v1.3.zip` | ~500 MB | ✅ **必需**，`LoadBEVSegmentation` 要用 |
| `v1.0-test_blobs.tgz` | ~30 GB | ❌ 不需要（test 无标注，只能提交榜单） |

峰值磁盘占用约 700 GB（压缩包 + 解压），**3.2 T 完全够**。解压完可删 tgz。

### 2.3 下载建议

拿到带 token 的链接后用 `aria2c` 断点续传，别用浏览器：

```bash
aria2c -x 8 -s 8 -c -d ~/project/bevfusion-main/data/nuscenes_download '<带token的链接>'
```

> 链接 token 有时效（通常几小时），过期了回网页重新点一次。

### 2.4 生成 pkl

解压到 `data/nuscenes/` 后：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

⚠️ **不要加 `--workers`**：`nuscenes_data_prep()` 的签名里根本没有 workers 参数，
命令行传了也不会往下传，整个过程是**单线程**的。该参数只对 KITTI 分支有效。

**实测耗时（2026-08-22）**：
- 第一步 `create_nuscenes_infos`（查表拼路径，不读图像点云）：**778 秒**，峰值内存 9.2 GB
- 第二步 `create_groundtruth_database`（逐帧读 11 个点云抠物体）：**约 26 分钟**
- 合计约 **40 分钟**，比原估的 3–5 小时快得多

### 验收标准 —— ✅ 全部通过（2026-08-22 完成）

- [x] `nuscenes_infos_train.pkl` 473 MB，**28130 帧** ✅
- [x] `nuscenes_infos_val.pkl` 98 MB，**6019 帧** ✅
- [x] `nuscenes_dbinfos_train.pkl` 200 MB + `nuscenes_gt_database/` 7.9 GB，**823476 个物体 / 19 类** ✅
- [x] `maps/expansion/` 4 个城市 json，MD5 与 mini 副本一致 ✅
- [x] `debug_pipeline.py` 在全量数据上 17 个 transform 全通，`LoadBEVSegmentation` 产出 `gt_masks_bev(6,200,200)` ✅
- [x] `create_nuscenes_infos` 报告 `exist scene num: 850`、`train scene: 700, val scene: 150` ✅

**数据实际来源**：不是从官网下载，而是从移动硬盘 `T7 Shield` 拷贝已解压的 `v1.0-trainval`（398 GB）。
源盘与工作站 24 个传感器目录**逐个比对文件数全部一致**，总体积均为 398 G。

**已知的无害报错**：`create_data.py` 最后会去处理 `v1.0-test` 并因找不到数据而 `AssertionError` 退出。
test 集无标注、只用于提交官方榜单，**我们故意没拷**，这个报错可以忽略——所有需要的产出在它之前就已全部生成。

---

## 阶段 3：受控实验矩阵（2026-08-23 修订版）

> 本节于 2026-08-23 重写。原版按「总共 20 小时」的预算倒推方案，把 epoch 砍到 4、
> 数据砍到 25%，是**先定时间再定方案**，本末倒置。现按实验有效性重新设计。

### 3.0 命名约定（此前未定义，补上）

| 简写 | 含义 | 帧数 |
|---|---|---|
| **25% 子集**（文件名 `q25`） | 175 场景 | 7,026 |
| **50% 子集**（文件名 `q50`） | 350 场景 | 14,052 |
| **全量** | 700 场景 | 28,130 |

`q25`/`q50` 是本项目自定的文件名后缀，**不是行业术语**。
论文里通常写 "25% of the train split" 或直接给场景数。

### 3.0b run-dir 命名规范（2026-08-23 补定）

`--run-dir` 的值是**自由字符串，程序不解析它**，只当目录名用。
但 20 个 run 之后没有规范就分不清了，所以定死格式：

```
<阶段>-<模型>-<数据>-<关键变量>
```

| 字段 | 取值 |
|---|---|
| 阶段 | `A`（可行性验证）/ `E1`~`E9`（消融）/ `sweep`（参数扫描）/ `bench`（基准测量） |
| 模型 | `pp`=PointPillars / `vox`=voxelnet / `cam`=camera_only / `fus`=convfuser |
| 数据 | `q25` / `q50` / `full` |
| 关键变量 | `lr3e5` / `nogtpaste` / `sweeps1` / `vox015` / `cbgs` … |

示例：

| run-dir | 含义 |
|---|---|
| `A-pp-q25-lr3e5` | 阶段 A，PointPillars，25% 数据，lr=3e-5 |
| `E4-pp-q50-nogtpaste` | E4，PointPillars，50% 数据，关 GT-Paste |
| `E9-pp-q25-cbgs` | E9，开 CBGS 的那一组 |

**核心规则：run-dir 必须体现「这组实验与其他组的差异」。**

run 目录下自动生成的内容：

```
runs/<run-dir>/
├── configs.yaml       ← 本次实验实际生效的完整参数（排查第一站）
├── <时间戳>.log       ← mmdet3d 的 INFO 日志（不含 Python 异常）
├── epoch_N.pth        ← checkpoint
└── tf_logs/           ← TensorBoard 数据
```

### 3.0c 各阶段为什么用不同的数据规模

| 阶段 | 数据 | 理由 |
|---|---|---|
| **参数扫描** | 25% | 只比相对优劣，2 epoch 就能分辨，越快越好 |
| **阶段 A 可行性验证** | **25%** | 回答的是「能不能产出非零 mAP」这个**是/否问题**。8.6h vs 17h，用一半时间得到同样的答案；若 25% 都不行，也省了 8.4 小时 |
| **E7 数据量曲线** | 25% / 50% / 100% | 它本身就是在测数据量的影响 |
| **E1~E6, E8** | **待 E7 结果决定** | 此前我说「用 50%」是**未经验证的猜测**，已撤回。由 E7 的曲线决定 |
| **阶段 C 最终数字** | 全量 + CBGS | 对标论文用 |

### 3.1 单组耗时基准（实测 PointPillars 0.22 s/iter）

| 配置 | 每 epoch 迭代 | 20 epoch 耗时 |
|---|---|---|
| 25% 子集，关 CBGS | 7,026 | **8.6 h** |
| 50% 子集，关 CBGS | 14,052 | **17 h** |
| 全量，关 CBGS | 28,130 | **34 h** |
| 全量 + CBGS（**MIT 官方设置**） | 123,580 | **151 h = 6.3 天** |

其他模型按 s/iter 折算：voxelnet ×1.10，camera_only ×1.31，convfuser ×2.17。

### 3.2 执行顺序（有依赖关系，不能乱序）

```
阶段 A  可行性验证：胜出 lr + 25% 子集 + 20 epoch，确认 mAP 非零      8.6 h
   ↓
诊断    同一 checkpoint 分别在训练集/验证集评测，定性判断瓶颈          0.5 h
   ↓
E7      数据量曲线 25% / 50% / 100%                                60 h
   ↓
        ★ 用 E7 的结果决定其余实验的数据规模（不再靠猜）
   ↓
E1~E6, E8, E9   消融矩阵
```

**为什么 E7 必须先跑**：其余实验用 25% 还是 50% 还是全量，取决于
「多少数据能让基线充分收敛」——而这正是 E7 要测的。
先前直接假定「50% 够用」是未经验证的猜测，不能拿它去锁定 200+ 小时的实验。

### 3.3 「训练集 vs 验证集」诊断（阶段 A 之后立刻做）

拿同一个 checkpoint 分别评测，成本仅一次评测（约 13 分钟）：

| 现象 | 诊断 | 该加什么 |
|---|---|---|
| 训练 mAP **远高于**验证 mAP | 过拟合 → **数据不足** | 加数据 |
| 两者**都低** | 欠拟合 → **训练量/学习率不足** | 加 epoch 或调 lr |
| 两者都高且接近 | 收敛良好 | 可以扩规模 |

### 3.4 实验矩阵

**统一设置**：20 epoch、bs=1、**关 CBGS**、固定种子 0、
验证集用**全量 6019 帧**（nuScenes 官方评测器不接受验证集子集，见实验记录 [07]）。

| 组 | 变量 | 模型 | 数据 | 耗时 |
|---|---|---|---|---|
| **E7** | **数据量 25/50/100%** | PointPillars | 三档 | **60 h** |
| E1 | 纯激光基线 | PointPillars | 待 E7 定 | 17 h |
| E2 | 纯视觉 | camera_only | 同上 | 22 h |
| E3 | 相机+激光融合 | convfuser | 同上 | 37 h |
| E4 | 关 `ObjectPaste`（GT-Paste） | PointPillars | 同上 | 17 h |
| E5 | `LoadPointsFromMultiSweeps` 10 帧 → 1 帧 | PointPillars | 同上 | 17 h |
| E6 | 关 `GlobalRotScaleTrans` + `RandomFlip3D` | PointPillars | 同上 | 17 h |
| E8 | 体素尺寸 0.2 / 0.15 / 0.1 | PointPillars | 同上 | 51 h |
| **E9** | **CBGS 开 / 关** | PointPillars | 25% | **47 h** |

合计约 **285 小时 ≈ 12 天**连续训练。

### 3.5 关于 CBGS 的决策（2026-08-23 确定）

**8 组消融全部关闭 CBGS，把 CBGS 本身作为独立实验 E9 单独测量。**

理由：开 CBGS 会让每组耗时 ×4.4（8 组要 25 天）。
而「CBGS 值多少分」本身是个未经测量的问题——
**不应该用一个未经验证的假设去锁死所有实验的设置**。

E9 的设计：同配置跑两次，唯一差别是 `--use_cbgs True/False`，用 25% 子集降成本。

- 25% 关 CBGS：8.6 h
- 25% 开 CBGS：38 h

**E9 的看点不是总 mAP，而是分类别对比**——
bicycle(8185)、motorcycle(8846)、construction_vehicle(11050) 这些稀有类的 AP 涨多少。
若提升明显，说明正式训练该开；若几乎不变，后续彻底不用管。

### 3.6 已知的结论边界（写报告时必须注明）

1. **单卡 batch=1，而官方是 8 卡 batch=8**，学习率按实测重新标定，
   **绝对指标不可与论文直接对比**
2. **关闭 CBGS**，稀有类别（bicycle / motorcycle / construction_vehicle）
   样本不足，**这些类的 AP 不可靠**，结论应聚焦 car / pedestrian / truck / barrier / traffic_cone
3. **spconv 存在上游随机崩溃 bug**（issue #82、#297），
   voxelnet / convfuser 的实验可能需要重跑；PointPillars / camera_only 不受影响

### 3.7 学习率 —— 已由扫描确定为 **3e-5**（2026-08-23）

**脚本**：`tools/lr_sweep.sh`　**结果**：`runs/lr_sweep_result.md`

3 个候选各训 2 epoch（14052 步），不评测，单一变量只改 `--optimizer.lr`：

| base lr      | cyclic 峰值 | 结果              | 末期 matched_ious | 崩溃时 lr | grad_norm       |
| ------------ | --------- | --------------- | --------------- | ------ | --------------- |
| **3e-5**     | 3.0e-4    | ✅ **完整跑完，持续上升** | **0.2063**      | —      | 稳定个位数           |
| 6e-5         | 6.0e-4    | ❌ ep1 步5150 NaN | 0.0144          | 5.9e-4 | **258.4（梯度爆炸）** |
| 1e-4（MIT 默认） | 1.0e-3    | ❌ ep1 步5000 NaN | 0.0621          | 9.7e-4 | 15.2            |

两个失败组的报错完全相同：
`ValueError: matrix contains invalid numeric entries`（匈牙利匹配器拿到 NaN 代价矩阵）。

**3e-5 的轨迹**（2 epoch 内一路上升，无回落）：

```
ep1 步4200  0.110      ep2 步2000  0.164
ep1 步6000  0.150      ep2 步4000  0.191
                       ep2 步6000  0.205
                       ep2 步7000  0.206   ← 跑完仍在涨
```

**PointPillars 的发散临界点：3e-4 ~ 5.9e-4 之间**（voxelnet 实测为 2.3e-4，两者接近）。

### 3.8 关于学习率的三次判断修正（教训）

| 时间 | 判断 | 实际 | 错在哪 |
|---|---|---|---|
| 08-23 早 | 「训练量不足是 mAP=0 的主因」 | **主因是学习率** | 未做对照就归因 |
| 08-23 早 | 用 voxelnet 的临界点 2.3e-4 推出 base=1.5e-5 | 太保守，4 epoch 只到 0.018 | **把临界点当成了 base lr，实际它对应的是 cyclic 峰值** |
| 08-23 早 | 「4 epoch 够看出趋势」 | mAP=0，连检测能力都没有 | 低估了收敛所需步数 |

**核心教训**：cyclic 策略下 `峰值 = base × 10`，
**判断安全性要看峰值，不是看 base**。这个换算关系此前一直没理清。

### 3.9 阶段 A：可行性验证（下一步）

**目的**：用胜出的 3e-5 拿到第一个**非零 mAP**，验证整条链路能产出有意义的结果。

```bash
cd ~/project/bevfusion-main && nohup torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/pointpillars.yaml --use_cbgs False --optimizer.lr 3e-5 --data.train.dataset.ann_file data/nuscenes/nuscenes_infos_train_q25.pkl --max_epochs 20 --evaluation.interval 20 --checkpoint_config.max_keep_ckpts 20 --run-dir runs/A-pp-q25-lr3e5 > /tmp/A_pp.log 2>&1 &
```

| 参数 | 值 | 说明 |
|---|---|---|
| `--optimizer.lr` | **3e-5** | 扫描胜出值 |
| `--max_epochs` | 20 | 官方默认轮数 |
| `--evaluation.interval` | 20 | 只在最后一轮评测（省 13 分钟 × 19） |
| `--checkpoint_config.max_keep_ckpts` | 20 | **保留每轮 checkpoint**，供后续诊断与中间轮次评测 |
| 数据 | 25% 子集 7026 帧 | 验证阶段先用小规模 |
| 验证集 | **不覆盖，用全量 6019 帧** | nuScenes 评测器不接受验证集子集 |

**预计 8.6 小时训练 + 13 分钟评测。**

**验收标准**：mAP 明显非零（> 0.05）。达标才继续走 E7 数据量曲线。

### 3.10 待验证的优化（有余力再做）

3e-5 的代价是**起步慢**——cyclic 从 3e-5 爬到 3e-4，第 1 个 epoch 大半时间在低学习率区。

解法：把 `lr_config.target_ratio` 从默认的 `(10, 1e-4)` 改成 `(3, 1e-4)`，
则 base 可开到 1e-4 而峰值仍是 3e-4（安全区内），**起点高 3.3 倍**。

**暂不采用**：这是第二处偏离官方配置的改动，而 3e-5 已证明可行。
先拿基线，之后作为独立实验验证。

## 阶段 4：汇总与结论（半天）

### 4.1 产出物

1. **一张总表**：8 组实验 × (mAP / NDS / 5 项误差 / 耗时)
2. **四张图**：
   - 模态对比柱状图（E1/E2/E3，**按类别拆开**）
   - 数据量–精度曲线（E7）
   - 增强消融瀑布图（E1/E4/E6）
   - 精度–延迟散点（E8 + `tools/benchmark.py` 测的 FPS）
3. **每个结论一句话**，例如："关掉 GT-Paste 后 mAP 掉 X 个点，其中 bicycle / motorcycle 这类小样本掉得最多"

### 4.2 预期能得到的直观结论

- 融合相对纯激光的增益**主要来自哪几类**（大概率是小目标和远距离目标）
- 纯视觉在**测距误差 mATE** 上的短板有多大（这是 BEV 视觉方案的核心痛点）
- 数据增强对**长尾类别**的作用远大于对 car/pedestrian 的作用
- 多帧叠加对**速度估计 mAVE** 的影响

> 这几条都是面试里能直接讲的东西，而且是**你自己跑出来的数**，不是背的论文结论。

---

## 附录 A：命令速查

```bash
# 进环境
ssh 4060 -t 'cd ~/project/bevfusion-main && source ~/miniconda3/etc/profile.d/conda.sh && conda activate bev && exec zsh'
```

```bash
# 训练（后台跑，断开 ssh 不中断）
nohup torchpack dist-run -np 1 python tools/train.py <config> --run-dir runs/<name> > runs/<name>.out 2>&1 &
```

```bash
# 评测
torchpack dist-run -np 1 python tools/test.py <config> <ckpt> --eval bbox
```

```bash
# 推理速度
python tools/benchmark.py <config> <ckpt> --samples 500
```

```bash
# 看数据 pipeline（阶段 0 之前先跑这个建立直觉）
python tools/debug_pipeline.py <config>
```

## 附录 B：已知的坑

| 坑 | 表现 | 解法 |
|---|---|---|
| 用错 conda 环境 | `numpy.core._multiarray_umath` 报错 | 用 `bev`，不是 `bevfusion` |
| 非交互 ssh 找不到 conda | `command not found: conda` | 先 `source ~/miniconda3/etc/profile.d/conda.sh` |
| pipeline 里断点不停 | 断点被跳过 | `--data.workers_per_gpu 0` |
| 缺 map expansion | `LoadBEVSegmentation` 报错 | 下载 `nuScenes-map-expansion-v1.3.zip` |
| 改 batch size 忘了改 lr | 收敛变差 | lr 按 batch 线性缩放 |
| 按帧随机切子集 | 精度虚高 | 必须按 `scene_token` 切 |

---

## 执行进度

- [ ] 阶段 0：基准测量
- [ ] 阶段 1：mini 流程演练
- [x] 阶段 2：全量数据落地 ✅ 2026-08-22 完成
- [ ] 阶段 3：实验矩阵（E1–E8）
- [ ] 阶段 4：汇总结论

---

## 附录 B：学习率数值速查（2026-08-28，用户要求常规小数对照）

**读法**：`NeM` = N×10^M，`e-5` 即小数点后挪 5 位；**指数差 1 = 差 10 倍**。
看科学计数法先看 e 后面的指数，再看系数（`1e-4` 比 `1.5e-5` 大 6.7 倍，别被 1 开头骗了）。

| 写法 | 常规小数 | 角色 |
|---|---|---|
| 1e-5 | 0.000 01 | 单帧扫描暂定胜者 |
| 1.5e-5 | 0.000 015 | 曾误用的保守值（10帧下太低→mAP=0）|
| 3e-5 | 0.000 03 | 10帧扫描胜者（T1-B）|
| 1e-4 | 0.000 1 | MIT 默认（8卡设置）|
| 2e-4 | 0.000 2 | 单帧发散阈值（实测）|
| 2.3e-4 | 0.000 23 | voxelnet 10帧发散拐点（实测）|
| 1e-3 | 0.001 | MIT 默认的 cyclic 峰值 |

**三配置窗口下移链条**（面试可讲的完整故事）：

```
MIT 原厂:  8卡×batch1 → 1e-4   (0.0001)
单卡10帧:  批量小8倍   → 3e-5   (0.00003)   降到 1/3.3
单卡单帧:  点数少9倍   → 1e-5   (0.00001)   再降 1/3
```

规律：**有效梯度信噪比每降一档（卡少→批小→点少），学习率窗口整体下移一档。**
