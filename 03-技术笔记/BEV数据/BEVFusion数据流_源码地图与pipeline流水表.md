---
title: BEVFusion 数据流 · 源码地图与 pipeline 流水表
type: 技术笔记
project: 项目一
tags: [BEV, BEVFusion, 数据流, dataset, pipeline, 源码]
updated: 2026-08-21
---

# BEVFusion 数据流 · 源码地图与 pipeline 流水表

> **这篇解决什么**：「BEV 数据太抽象、不知道怎么用」——把「数据从 pkl 到模型 forward」这条路
> 拆成**看得见的 17 步**，并给出**每一步对应哪个源码文件的哪一行**。
>
> **前置**：[[BEV数据学习规划]] 的第 1~3 步（磁盘盘点 / token 走查 / 投影可视化）已完成。本篇是第 4 步「进模型」。
> **关联**：[[BEVFusion数据流_问题清单]]（16 题，本篇直接答掉 B2/C1/C3）

---

## 〇、五分钟上手（不想读文档就看这段）

**核心认知：看 dataloader 不需要跑训练。** `dataset[0]` 就把整条 pipeline 跑完了，两秒钟。

工作站上跑这一条，得到本文档第三节那张流水表：

```bash
/home/hpf/miniconda3/envs/bev/bin/python /home/hpf/project/bev_data_lab/tools/pipeline_dump.py
```

| 项 | 完整路径 |
|---|---|
| 工具 | `/home/hpf/project/bev_data_lab/tools/pipeline_dump.py` |
| 输出 | `/home/hpf/project/bev_data_lab/out/pipeline_流水表.txt` |
| 代码仓 | `/home/hpf/project/bevfusion-main/` |
| 配置 | `/home/hpf/project/bevfusion-main/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml` |
| 数据 | `/home/hpf/project/bevfusion-main/data/nuscenes-mini/` |

可选参数：`pipeline_dump.py [配置文件] [sample序号]`，换帧看差异用 `pipeline_dump.py "" 5`。

---

## 一、源码地图：Dataset 侧 5 个文件 + DataLoader 侧 3 个位置

全部位于 `/home/hpf/project/bevfusion-main/mmdet3d/datasets/`

| #   | 文件                           | 行数   | 职责                                                    | 什么时候看它                 |
| --- | ---------------------------- | ---- | ----------------------------------------------------- | ---------------------- |
| 1   | `nuscenes_dataset.py`        | 647  | **Dataset 主类**：读 pkl、`get_data_info()` 拼初始 dict、评测    | 想知道那 24 个初始 key 哪来的    |
| 2   | `custom_3d.py`               | 306  | Dataset 基类：`__getitem__` / `__len__` / pipeline 怎么被调用 | 想知道 dataset[0] 内部发生了什么 |
| 3   | `pipelines/loading.py`       | 558  | **5 个「读数据」transform**                                 | 第 1~4 步                |
| 4   | `pipelines/transforms_3d.py` | 1002 | **17 个「加工数据」transform**                               | 第 5~15 步               |
| 5   | `pipelines/formating.py`     | 197  | 打包：`DefaultFormatBundle3D` + `Collect3D`              | 第 16~17 步              |

**三个配角**

| 文件 | 行数 | 说明 |
|---|---|---|
| `builder.py` | 44 | 只有 `build_dataset`（**没有 build_dataloader**，见第一节末尾） |
| `dataset_wrappers.py` | 76 | ★ **CBGS 类别重采样**（简历第 4 条优化点的实现） |
| `pipelines/dbsampler.py` | 323 | `ObjectPaste` 用的 GT 数据库 |

### 类 → 行号速查

**`pipelines/loading.py`**

| 类 | 行 |
|---|---|
| `LoadMultiViewImageFromFiles` | 19 |
| `LoadPointsFromMultiSweeps` | 84 |
| `LoadBEVSegmentation` | 239 |
| `LoadPointsFromFile` | 312 |
| `LoadAnnotations3D` | 433 |

**`pipelines/transforms_3d.py`**

| 类 | 行 | 备注 |
|---|---|---|
| `ImageAug3D` | **26** | ★★ 图像变换 + `img_aug_matrix` |
| `GlobalRotScaleTrans` | **124** | ★★ 点云增强 + GT 同步 + `lidar_aug_matrix` |
| `GridMask` | 159 | |
| `RandomFlip3D` | 247 | |
| `ObjectPaste` | 276 | GT-Paste 增强 |
| `PointShuffle` | 447 | |
| `ObjectRangeFilter` | 454 | |
| `PointsRangeFilter` | 504 | |
| `ObjectNameFilter` | 529 | |
| `ImageNormalize` | 903 | |

**`pipelines/formating.py`**：`DefaultFormatBundle3D`(14)、`Collect3D`(128)

### ⚠ DataLoader 在哪？——不在这个仓里

**BEVFusion / mmdet3d 没有自己写 DataLoader**，用的就是 PyTorch 的 `torch.utils.data.DataLoader`。
本仓 `builder.py` 里**只有 `build_dataset`，没有 `build_dataloader`**（44 行全文可自行确认）。

真正的调用链是这样的，跨了三个包：

| 层 | 位置 | 干什么 |
|---|---|---|
| 调用点 | `/home/hpf/project/bevfusion-main/mmdet3d/apis/train.py:33` | `build_dataloader(ds, samples_per_gpu, workers_per_gpu, ...)` |
| 包装函数 | `.../site-packages/mmdet/datasets/builder.py` 的 `build_dataloader()` | 选 sampler、设 batch_size、绑 `collate_fn`、`worker_init_fn`，最后 `return DataLoader(...)` |
| **拼 batch 的函数** ★ | `.../site-packages/mmcv/parallel/collate.py` 的 `collate()` | **不定长数据怎么打包成 batch —— B4 的答案** |

（`.../site-packages/` 完整前缀：`/home/hpf/miniconda3/envs/bev/lib/python3.8/`）

#### B4 答案：collate 怎么处理不定长

关键在 `DataContainer`（DC）这个包装类。第 16 步 `DefaultFormatBundle3D` 把数据包进 DC 时，
就给每样东西打了标签，`collate()` 按标签分三种处理：

| 情况 | 标签 | 例子 | 怎么合批 |
|---|---|---|---|
| 1 | `cpu_only=True` | `metas`（元信息 dict） | 原样堆成 list，不转 tensor、不上 GPU |
| 2 | `cpu_only=False, stack=True` | `img` (6,3,256,704) | **真的 stack**：每帧 shape 相同 → (B,6,3,256,704) |
| 3 | `cpu_only=False, stack=False` ★ | `points`、`gt_bboxes_3d` | **不 stack，保持 list**：每帧点数/框数不同，堆成 `[帧0的点, 帧1的点, ...]` |

**所以「不定长」的处理方式是：干脆不合并。** 模型 forward 里拿到的 `points` 是一个 list，
逐帧处理（体素化时才各自转成稀疏张量）。这也是为什么 `pipeline_dump` 里看到
`points DC(242504, 5)` —— DC 包着的是**单帧**，合批后外面套一层 list。

> 对照记忆：这正是 pytorch-3 课上老师说的「分类可以直接 cat 合并，目标检测做不到」——
> 检测/BEV 里每帧框数点数都不同，所以走的是第 3 种。

### 建议读法：别从头到尾读，只读两个函数

1. `nuscenes_dataset.py:209` 的 **`get_data_info()`** —— 看它怎么把 pkl 一条 info 变成初始 dict（`lidar2image` 等矩阵在这算出来，答问题清单 C2）
2. `transforms_3d.py:26` 的 **`ImageAug3D.__call__()`** —— 全仓最值钱的 60 行，图像变换与内参同步的耦合都在这

其余按第三节的流水表顺序跳读，**每个 transform 只看 `__call__`**（`__init__` 都是存参数，跳过）。

---

## 二、pipeline 逐步 dump 是什么

**dump** = 把内存里的数据结构原样倒出来（字段名、形状、数值）。
**逐步 dump** = 每执行完一个 transform 就倒一次，形成流水账。

为什么需要：`dataset[0]` 只给最终结果，中间 17 步是黑盒——`img_aug_matrix` 第几步冒出来的、`gt_bboxes_3d` 在哪一步被改、点数怎么从 3 万变 27 万，全看不见。逐步 dump 在每步之间插检查点：

```python
for i, t in enumerate(pipeline.transforms):
    keys_before = set(data.keys())
    fp_before   = {k: fingerprint(v) for k, v in data.items()}
    data = t(data)                       # 执行这一步
    print(f"{i+1}. {type(t).__name__}")
    print(f"   新增: {set(data.keys()) - keys_before}")     # 多了什么字段
    print(f"   修改: {[k for k in ... if 指纹变了]}")        # 哪些字段的值变了
```

比喻：不是只看终点照片，而是给数据装**行车记录仪**。

---

## 三、TRAIN PIPELINE 完整 17 步（实测，sample #0）

> ★ = 关键字段。数据：nuscenes-mini，配置 convfuser.yaml。

### 第 0 步 · 初始 dict（`get_data_info()` 的产物，24 个 key）

来自 pkl 的一条 info，此时**磁盘文件还没读**，只有路径和矩阵：

```
★ camera2ego          list[6] 每项(4,4)      六路相机装在车上哪
★ camera_intrinsics   list[6] 每项(4,4)      内参 K
★ lidar2image         list[6] 每项(4,4)      ★ 点云→图像的合成矩阵(C2的答案)
  camera2lidar / lidar2camera / lidar2ego / ego2global   其余变换矩阵
  image_paths         list[6]                六张 jpg 的路径(还没读)
  lidar_path          str                    点云 bin 路径(还没读)
  ann_info            dict(3键)              标注(还没解析)
  sweeps              [](空)                 pkl 里的 sweeps 字段为空
  token / sample_idx / timestamp / location  身份信息
```

| 步 | Transform | dict 发生了什么 | 关键点 |
|---|---|---|---|
| **1** | `LoadMultiViewImageFromFiles` | ＋`img`(list[6] JpegImage)、`filename`、`img_shape`、`ori_shape`、`pad_shape`、`scale_factor` | **磁盘 jpg 此刻才真正读进内存** |
| **2** | `LoadPointsFromFile` | ＋`points` **(34,688, 5)** | 单帧点云 x,y,z,intensity,ring |
| **3** | `LoadPointsFromMultiSweeps` | ~`points` 34,688 → **272,414** | ★ **涨 8 倍**。pkl 的 sweeps 字段虽空，它仍直接从 `sweeps/` 目录读多帧 |
| **4** | `LoadAnnotations3D` | ＋`gt_bboxes_3d` **LiDARInstance3DBoxes(66框, 9维)**、`gt_labels_3d`(66,) | ★ **GT 首次出现，已经是雷达系** → 说明 global→lidar 转换发生在更早的 `create_data`，不在 pipeline 里（**C1 答案**） |
| **5** | `ObjectPaste` | dict 无变化 | GT-Paste 增强；本帧没触发（概率性） |
| **6** | `ImageAug3D` | ＋`img_aug_matrix` **list[6] (4,4)**；~`img` 被 resize/crop | ★★ **图像增强与内参同步的现场**（**C3 答案上半**） |
| **7** | `GlobalRotScaleTrans` | ＋`lidar_aug_matrix` **(4,4)**；~`points`、~`gt_bboxes_3d`、~`ann_info` | ★★ **点云增强的同时 GT 框数值一起被改** —— 这就是"增强同步作用到 GT"（**C3 答案下半**） |
| **8** | `LoadBEVSegmentation` | ＋`gt_masks_bev` (6, 200, 200) | BEV 分割真值（地图要素） |
| **9** | `RandomFlip3D` | ~`gt_bboxes_3d`、~`ann_info` | 翻转，GT 同步 |
| **10** | `PointsRangeFilter` | ~`points` 272,414 → **242,504** | 砍掉感知范围外的点 |
| **11** | `ObjectRangeFilter` | ~`gt_bboxes_3d` 66 → **52 框** | 砍掉范围外的 14 个 GT |
| **12** | `ObjectNameFilter` | ~`gt_bboxes_3d` 52 → **51 框** | 砍掉 1 个无关类别 |
| **13** | `ImageNormalize` | ＋`img_norm_cfg`；~`img` → list[6] 每项 **(3, 256, 704)** | HWC→CHW + 归一化，尺寸定型 |
| **14** | `GridMask` | dict 无变化 | 图像随机遮挡（概率性） |
| **15** | `PointShuffle` | dict 无变化 | 点云打散顺序 |
| **16** | `DefaultFormatBundle3D` | ~`img`/`points`/`gt_*` 全部包成 **DataContainer(DC)** | numpy → torch.Tensor + DC 封装 |
| **17** | `Collect3D` | ＋`metas`；~8 个矩阵转 DC；**－删除 24 个 key** | ★ **只留 14 个 key 交给模型，其余全丢**（`token`、`lidar_path`、`ann_info`… 都没了） |

### 终点：模型 `forward()` 实际收到的 14 个 key

```
★ img                  DC(6, 3, 256, 704) float32     六路图像
★ points               DC(242504, 5)      float32     点云
★ gt_bboxes_3d         DC[LiDARInstance3DBoxes 51框]  GT 框
★ gt_labels_3d         DC(51,)            int64       GT 类别
  gt_masks_bev         (6, 200, 200)      int64       BEV 分割真值
★ camera_intrinsics    DC(6,4,4)   ★ camera2ego     DC(6,4,4)
★ lidar2image          DC(6,4,4)     camera2lidar   DC(6,4,4)
  lidar2camera         DC(6,4,4)     lidar2ego      DC(4,4)
★ img_aug_matrix       DC(6,4,4)   ★ lidar_aug_matrix DC(4,4)
  metas                DC[dict]                       元信息
```

**这 14 个 key = 问题清单 D1 的答案。**

### 数量变化一览（只有 dump 看得见）

```
点数:  34,688 ──(第3步拼sweeps)──> 272,414 ──(第10步范围过滤)──> 242,504
GT框:      66 ──(第7/9步只改数值)──>     66 ──(第11步范围)──> 52 ──(第12步类别)──> 51
图像:  900×1600 ──(第6步Aug)──> ... ──(第13步定型)──> 256×704
```

---

## 四、TEST PIPELINE 11 步 + 与 TRAIN 的真实差异

| 步 | Transform | 与 train 的区别 |
|---|---|---|
| 1 | `LoadMultiViewImageFromFiles` | 相同 |
| 2 | `LoadPointsFromFile` | 相同 |
| 3 | `LoadPointsFromMultiSweeps` | 相同 |
| 4 | `LoadAnnotations3D` | 相同（**test 也加载 GT**，因为要在线评测） |
| 5 | `ImageAug3D` | ★ `is_train=False`：**固定参数**，不随机 |
| 6 | `GlobalRotScaleTrans` | ★ **只新增 `lidar_aug_matrix`，不改 points 和 GT**（恒等变换） |
| 7 | `LoadBEVSegmentation` | 相同 |
| 8 | `PointsRangeFilter` | 相同 |
| 9 | `ImageNormalize` | 相同 |
| 10 | `DefaultFormatBundle3D` | 相同 |
| 11 | `Collect3D` | 相同 |

**train 独有的 6 步**：`ObjectPaste`、`RandomFlip3D`、`ObjectRangeFilter`、`ObjectNameFilter`、`GridMask`、`PointShuffle`

### ⚠ 一个反直觉的实测结论

```
train 终点 key: 14 个
test  终点 key: 14 个
仅 train 有: (无)     仅 test 有: (无)
```

**两条 pipeline 交给模型的 key 完全一样**，差别只在**数值**：
- test 的 `img_aug_matrix` / `lidar_aug_matrix` 是恒等变换
- test 的 GT 不做范围/类别过滤（本帧 23 框 vs train 51 框，注意两者 sample 不同，不可直接比数量）

**教训：看配置文件的 transform 名字会猜错（我第一次就猜错了，以为 test 不带 GT），dump 看数据不会。**

---

## 五、这一步直接答掉的问题

| 题号 | 问题 | 答案 |
|---|---|---|
| **B2** | pipeline 每一步进出，dict 里多了/变了哪些 key | ✅ 第三节整张表 |
| **C1** | GT 框此刻在哪个坐标系？在哪一步变的？ | ✅ 第 4 步一出现就是雷达系 → 转换在 `create_data`（`nuscenes_converter.py`），不在 pipeline |
| **C3** | 图像增强后矩阵怎么跟着改？ | ✅ 第 6 步生 `img_aug_matrix`、第 7 步生 `lidar_aug_matrix` 且同步改 GT |
| **D1** | `model.forward` 收到的 dict 有哪些 key、各什么 shape | ✅ 第三节「终点」14 个 key |
| **B4** | collate 怎么处理不定长（每帧点数/框数不同） | ✅ 第一节「DataLoader 在哪」：DC 的三种标签，不定长的那类**不 stack、保持 list** |

---

## 六、附带发现：CBGS 就在这条链路上

跑 dump 时撞见的：配置里 `data.train` 外面包了一层 **`CBGSDataset`** —— 正是简历第 4 条那个"类别重采样"。日志能看到效果：

```
load 4082 car   database infos  →  After filter: 2287 car
load  147 bicycle                →              108 bicycle
```

实现只有 76 行：`/home/hpf/project/bevfusion-main/mmdet3d/datasets/dataset_wrappers.py`
（`pipeline_dump.py` 里已自动剥掉这层 wrapper 拿内层 dataset，代码里有注释说明。）

---

## 七、下一步

- [ ] 换几帧跑（`pipeline_dump.py "" 5`），看 `ObjectPaste` / `GridMask` 触发时 dict 怎么变
- [ ] 读 `transforms_3d.py:26` 的 `ImageAug3D.__call__`，验证 `img_aug_matrix` 是怎么算出来的
- [ ] 读 `nuscenes_dataset.py:209` 的 `get_data_info`，找 `lidar2image` 的合成公式（问题清单 C2）
- [ ] 增强开关对拍：同一帧关掉增强再 dump，对比 GT 数值差异

---

## 变更记录

- **2026-08-21** 建档。工具 `pipeline_dump.py` 写成并跑通（踩坑：`configs.reset()` 不存在、train 被 CBGSDataset 包裹、`np.ndarray` 的 `.data` 是 memoryview 会误判成 DataContainer）。合并了原「如果我自己去看 bevfusion 的话…」那篇的源码地图内容。
- **2026-08-21（补）** 补第一节「DataLoader 在哪」：更正 builder.py 里没有 build_dataloader 的错误，
  补齐跨 mmdet3d/mmdet/mmcv 三包的调用链，并写入 B4（collate 不定长）的答案。
