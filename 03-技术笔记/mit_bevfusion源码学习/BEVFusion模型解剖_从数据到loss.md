---
tags: [BEV, BEVFusion, 模型解剖, PointPillars, TransFusion]
创建: 2026-08-28
状态: 第1站进行中
---

# BEVFusion 模型解剖：从 Dataset 吐出数据到 loss 算出来

> 目的：补上实验之外的那段黑盒——网络内部。每站 30–60 分钟，塞在训练等待期。
> 方式：读代码（带行号路标）→ 跑形状打印脚本亲眼看数据变形 → 自己画一张图。
> 解剖对象：正在训的 lidar-only 路径 `pointpillars.yaml`（PointPillars + TransFusion head）。
> 关联：[[BEVFusion专题实验_叠多帧与运动补偿]]（第4站回答 mAVE 从哪来）

## 全链路地图 v2（2026-08-28 重制为双支路。v1 只画了点云一条腿——受"叠多帧"专题
干扰所致，据第2次语音讨论纠正；BEVFusion 的 fusion 二字正来自两条支路各出一张 BEV 后融合）

```
            共同起点: Dataset / DataLoader【已学】
            一帧 = 点云(N,5) + 6路环视图像 + 标定/ego_pose + 标注
                          │
        ┌─────────────────┴─────────────────┐
        ▼ 点云支路                           ▼ 相机支路（全部未学）
   体素化: 点→柱子(M,20,5)【已学】        图像backbone: 6张图→特征图
        ▼                                  ▼
   柱子变特征+散射→伪图像(64,512,512)      depth net: 每像素出深度分布
        ▼                                  ▼
   点云 BEV 图                        视锥→投到BEV→相机BEV图
        └─────────────────┬─────────────────┘
                          ▼
                   融合(Concat+卷积)   ← lidar-only 配置无此站
                          ▼
                 骨干+颈部 → BEV特征(384,128,128)
                          ▼
                 检测头: heatmap→选query→出框+速度(mAVE 的落点)
                          ▼
                 真值配对(匈牙利匹配, matched_ious 出处)+各项loss → 回传
```

**学法三原则**（语音讨论定）：
1. **两条支路平行走**，不是先点云后相机——模型结构本身平行，先走完一条，
   到融合站会因半边输入没见过而卡住（"概念悬着串不起来"正是要避免的）
2. 粒度 = **跑通 demo、入门程度**：抓骨干不逐行精读；每站必答
   "数据形状怎么变、为什么这么变"（检验方式：假设在面试，讲给面试官听）
3. 每站要能回答"另一边支路此刻在干什么"，维持两条支路的时间对齐感

**命名改白话**（旧编号仅留档/run-dir 用）：体素化(原①)/柱子变特征(原②)/
骨干颈部(原③)/检测头(原④)/真值与loss(原⑤)；相机支路新增：图像backbone/
depth net/视锥投影；实验组：单帧组(T1-A)/十帧组(T1-B)/关补偿组(T1-C)/四帧组(T1-D)。

注：目前训练的 pointpillars.yaml 是 lidar-only，只走点云支路；学相机支路时
解剖对象换 convfuser（camera+lidar）配置，代码同一套仓库。

---

## 第 1 站：体素化——点云怎么变成网络能吃的形状（2026-08-28）

### 目的与预期

- **回答的问题**：无序、变长的点云，怎么变成卷积网络要求的规则网格？
  第 5 维时间戳（叠多帧的标记）从哪里进网络？
- **预期现象**：单帧和 10 帧输入，**网格总尺寸恒定 512×512**，变的只是
  非空柱子数 M 和柱内点数——这是"点少≠网络变小"的第一手证据。
- **判读标准**：跑完脚本能亲口说出 voxels/coords/num_pts 三个张量各是什么。

### 概念：voxel 与 pillar 两条路

点云是集合（无序、数量不定），卷积要网格（有序、尺寸固定）。两种切法：

| | voxel（体素） | pillar（柱子） |
|---|---|---|
| 切法 | x/y/z 三个方向都切 → 3D 小方块 | 只切 x/y，z 方向整根不切 → 2D 网格上的柱子 |
| 后续卷积 | 3D 稀疏卷积（**spconv**，就是随机崩的那个库） | 普通 2D 卷积 |
| 我们的选择 | voxelnet 用 | **pointpillars 用 ← 免疫 spconv bug 的根本原因** |

### 配置数字逐项定义（pointpillars.yaml + default.yaml:10 实况）

| 配置 | 值 | 含义 |
|---|---|---|
| point_cloud_range | [-51.2,-51.2,-5.0, 51.2,51.2,3.0] | 只保留自车周围 ±51.2m（xy）、-5~3m（z）的点，出界丢弃 |
| voxel_size | [0.2, 0.2, 8] | 每根柱子底面 0.2m×0.2m；z 方向 8m=整个高度范围，即"不切" |
| → 网格数 | 102.4/0.2 = **512×512** | 与 head 的 grid_size 512 呼应（就是之前修 bug 补的那行）|
| max_num_points | 20 | 每柱最多 20 个点，多了**随机丢**，少了补零（hard voxelization）|
| max_voxels | [30000, 60000] | 非空柱子上限：训练 3 万 / 测试 6 万，超了整柱丢弃 |

### 代码路标

| 位置 | 内容 |
|---|---|
| `mmdet3d/models/fusion_models/bevfusion.py:244` | forward 里 lidar 分支入口 `extract_lidar_features(points)` |
| `bevfusion.py:137` `def voxelize` | 逐样本调 Voxelization，coords 前面 pad 一列样本号 k（区分 batch 内哪帧）|
| `mmdet3d/ops/voxel/voxelize.py:77` | Voxelization 层本体（CUDA 算子包装）|
| `bevfusion.py:157` voxelize_reduce | pointpillars.yaml 设 false：**保留柱内每个点**交给下一站，不提前平均 |

### 动手脚本 tools/anatomy_s1.py（工作站）

见实验记录正文/脚本文件。跑两次对比：

```bash
python tools/anatomy_s1.py --sweeps 0
python tools/anatomy_s1.py --sweeps 9
```

### 实测结果（2026-08-28，--idx 0，⚠️ 退化样本见下）

| 指标 | sweeps=0 | sweeps=9 |
|---|---|---|
| 输入点数 N | 33 686 | 252 458 |
| 非空柱子 M / 占比 | 7 324 / 2.8% | **7 324 / 2.8%（与单帧完全相同！）** |
| 每柱平均点数 / 满载率 | 3.4 / 1.9% | 16.1 / **61.3%** |
| 进不了柱子的点 | 9 116 (27.1%) | 134 370 (53.2%) |

**符号定义**：N = 点的总数（shape 第一个数字）；M = 非空柱子根数（voxels shape 第一个数字）。

**发现：idx=0 是叠帧退化样本**。两次 M 精确相等，真历史帧几乎不可能——
诊断：val infos 按 timestamp 排序，idx=0 是首个场景的**第一个关键帧，没有历史 sweep**。
`LoadPointsFromMultiSweeps` 的 `pad_empty_sweeps` 机制会用**当前帧自我复制**凑数
（复制前过 `remove_close` 删自车近点）。算术验证：(252458−33686)÷8≈27 346 ≈
单帧删近点后的规模 → "10帧"实为同一帧的多份拷贝 → 落进完全相同的格子 → M 不变、
满载率飙到 61.3%、超半数点被截断丢弃。
**教训：解剖/评测选样本要避开场景首帧**（和 Part1 可视化选帧条件"自车位移>4m"同理）。

**已确认的结论（不受退化影响）**：
1. 网格恒定 512×512，输入点数×7.5 画布尺寸不变 ✅（本站核心）
2. 单帧占用率仅 2.8% —— 97% 的 BEV 画布是空的，"点云稀疏性"的第一手数字
3. M=7324 ≪ 训练配额 30000 → 单帧训练无柱子截断

### 复测结果（--idx 20，场景中段，pkl 确认 idx0 sweeps数=0 / idx20 sweeps数=10 → 退化诊断坐实）

| 指标 | 单帧(sweeps=0) | 叠10帧(sweeps=9) | 对比 |
|---|---|---|---|
| N | 32 120 | 229 846 | ×7.2 |
| M / 画布占比 | 7 689 / 2.9% | **27 186 / 10.4%** | **×3.5** |
| 每柱均点 / 满载率 | 2.9 / 1.0% | 5.9 / 9.7% | 浓度×2 |
| 进不了柱子的点 | 31.4% | 29.9% | 基本持平（主因均为出range）|

**复测结论**：
1. 真历史帧下 M 增长 3.5 倍——Part1 亲眼看过的"盲区填补/环间隙稠密化"的量化版
   （对照退化样本 idx0：M 纹丝不动，满载率 61.3% 全是自我复制的假浓度）
2. **新发现：M=27186 已达训练配额 30000 的 91%**。本样本未越界，但更稠密的
   场景（市区、多车）很可能超 3 万 → 训练中部分帧的柱子被静默丢弃。
   结论从"无此问题"修正为"**贴线运行，个别帧可能越界**"
3. 叠帧丢点率与单帧持平(~30%)，主因都是出 range（>51.2m 的远点），
   idx0 那个 53.2% 是自我复制撞满柱子的假象

### 胶水 a：Dataset / DataLoader 分工 + batch 怎么走到 voxelize（2026-08-28 补全）

**厨房类比**——三个角色，泾渭分明：

| 角色 | 类比 | 干什么 | 代码位置 |
|---|---|---|---|
| pkl (infos) | 菜单/索引卡 | 只存路径和元信息，**不存数据本身** | data/nuscenes/*.pkl |
| Dataset | 厨师 | 接到号 idx 做**一帧**：查索引卡 → 过 pipeline 加工 → 吐一个字典 | custom_3d.py:284 `__getitem__` |
| DataLoader | 传菜系统 | 发号(shuffle)、雇工人(workers)、凑单(batch)、**摆盘(collate)** | mmdet3d/apis/train.py:33 借 mmdet 的 build_dataloader |

**Dataset 内部**（custom_3d.py）：
- `:284 __getitem__`：test_mode 走 prepare_test_data；训练走 `:292 while True` —
  `:293 prepare_train_data` 若返回 None（该帧无有效 GT），`:295 _rand_another` 随机换一帧重做，
  **保证 batch 里不出现空目标样本**
- `:151 prepare_train_data`：`:160 get_data_info` 查 pkl 拿路径/标定 → `self.pipeline(input_dict)`
  过流水线——**debug_pipeline.py 当年单步过的 17 个 transform 就是这一段**，即"厨师的做菜工序"
- CBGS 是在**发号环节**做手脚的包装层（改某类样本被抽中的频率），厨师本身不知情
  ——这就是 train.py 里那个三行开关能干净剥离它的原因

**DataLoader 摆盘规则（collate，来自 mmcv.parallel）**：
- 尺寸相同的堆成大张量（图像 → (B,3,H,W)）
- **点云每帧数量不同堆不了** → 保持"每帧一个张量的列表"；DataContainer 就是
  "别帮我堆叠"的标签
- 我们的配置 default.yaml:262-263：`samples_per_gpu: 1, workers_per_gpu: 1`
  —— 一次凑 1 帧、1 个工人厨师

**进模型后**：collate 产物 → train_step → `bevfusion.py:207 forward_single` 按模态分发 →
`:244 extract_lidar_features` → `:137 voxelize` 逐帧体素化，`:149` 给柱子坐标补样本号 k
→ 全 batch 柱子拼一起也不串帧，第 2 站 scatter 靠 k 撒回各自画布。

**和动手脚本的关系**：anatomy_s1.py 里 `dataset[args.idx]` 是**亲手替 DataLoader 叫号**
——绕过传菜系统直接找厨师，所以输出没有 batch 维；脚本里剥 DataContainer 壳，
就是手动做了摆盘的逆操作。

### 面试题（第1站验收）

1. PointPillars 为什么快？（z 不切 → 2D 卷积 → 免 spconv）
2. hard voxelization 丢点吗？丢在哪两个环节？（柱内超 20 截断 + 柱数超上限整柱丢 + 出 range）
3. 叠多帧后点数×9，网络第一层计算量×9 吗？（不：网格恒定，变的是非空柱子数，且有 max_voxels 封顶）
