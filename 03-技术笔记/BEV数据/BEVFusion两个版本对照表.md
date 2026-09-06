---
title: BEVFusion 两个版本对照表（MIT vs 北大&阿里）
type: 技术笔记
project: 项目一
tags: [BEV, BEVFusion, MIT, ADLab, 源码对照, LSS, BEVPool]
updated: 2026-08-22
---

# BEVFusion 两个版本对照表

> **为什么需要这张表**：《自动驾驶BEV感知算法指南》**第7章用的是北大&阿里版**（书 P111 原文：
> "本章所讨论的 BEVFusion 特指由北京大学与阿里巴巴集团共同提出的版本"），
> 而**你简历上的技术栈属于 MIT 版那条线**。看书时对着这张表，就知道"这段代码在我的版本里对应哪里"。
>
> 关联：[[BEVFusion数据流_源码地图与pipeline流水表]]

---

## 〇、两份代码都在你工作站上

| 版本 | 完整路径 | 论文 | 配置风格 |
|---|---|---|---|
| **MIT-han-lab** ★ 你的主线 | `/home/hpf/project/bevfusion-main/` | ICRA 2023 · arXiv 2205.13542 | torchpack + **yaml** |
| **ADLab 北大&阿里** ← 书用这个 | `/home/hpf/project/bevfusion_study/BEVFusion/` | NeurIPS 2022 · arXiv 2205.13790 | mmdet3d + **py** |

> 两篇论文差一天投出，同名撞车。**看书报路径找不到，多半是在 MIT 仓里找 ADLab 的文件。**

**书里出现的路径，一律加前缀** `/home/hpf/project/bevfusion_study/BEVFusion/`

---

## 一、核心文件对照

### 视图变换（LSS）—— 差异最大的地方

| 功能 | **MIT 版** | **ADLab 版** |
|---|---|---|
| 主文件 | `mmdet3d/models/vtransforms/base.py` | `mmdet3d/models/detectors/cam_stream_lss.py` |
| 类 | `BaseTransform`(21) / `BaseDepthTransform`(208) | `LiftSplatShoot`(149) |
| 建视锥 | `create_frustum()` **:53** | `create_frustum()` **:216** |
| 投自车系 | `get_geometry()` **:79** | `get_geometry()` **:228** |
| 池化 | `bev_pool()` **:128** ← 调自研 CUDA | `voxel_pooling()` **:279** |
| cumsum | （在 CUDA kernel 里） | `cumsum_trick()` **:85** ／ `QuickCumsum`(96) ★ |
| 具体实现 | `lss.py:14 LSSTransform`<br>`depth_lss.py:15 DepthLSSTransform` | `CamEncode`(125) / `BevEncode`(39) |

★ **ADLab 的 `cumsum_trick` 和 `QuickCumsum` 是纯 PyTorch 的，可读性远高于 MIT 的 CUDA 版**——
想看懂 BEVPool 到底在算什么，读 ADLab 这段最快（它就是我们 LSS notebook 里手写那段的工业版）。

### 融合模块

| | MIT 版 | ADLab 版 |
|---|---|---|
| 位置 | `mmdet3d/models/fusers/` | `mmdet3d/models/detectors/bevf_faster_rcnn.py` |
| 实现 | `conv.py:12 ConvFuser`（concat+卷积）<br>`add.py:13 AddFuser` | `extract_feat()`**:100** 里做<br>用 `SE_Block`(:58) 做通道注意力 |
| 设计取向 | **精度优先**，单一路线做到最好 | **鲁棒性优先**，双流可切换（配置里有多种策略） |

### 数据准备（两版几乎一样 ★）

| | MIT 版 | ADLab 版 |
|---|---|---|
| 入口 | `tools/create_data.py` | `tools/create_data.py` |
| 转换器 | `tools/data_converter/nuscenes_converter.py` | 同名同位置 |

两版 `create_data.py` 差异仅 359 行（主要是数据集支持范围），**核心的 pkl 生成逻辑同源**——
都继承自 mmdet3d。所以：

> **问题清单 A2/C1（pkl 字段、GT 从 global 转 lidar 在哪一步），读哪个版本都行。**

---

## 二、书上代码清单 → 你的实际路径

| 书 | 清单 | 书上写的路径 | 实际完整路径 |
|---|---|---|---|
| P117 | 7-1 附近 | `tools/create_data.py` | `/home/hpf/project/bevfusion_study/BEVFusion/tools/create_data.py` |
| P131 | 7-19 | `nuscenes_converter.export_2d_annotation` | `.../BEVFusion/tools/data_converter/nuscenes_converter.py` |
| **P131** | **7-20** | `configs/bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.py` | `/home/hpf/project/bevfusion_study/BEVFusion/configs/bevfusion/cam_stream/bevf_pp_4x8_2x_nusc_cam.py` ✅ 存在 |

`configs/bevfusion/` 下的完整清单（书后面还会用到）：

```
bevf_cp_4x8_6e_nusc.py          bevf_pp_2x8_1x_nusc.py       cam_stream/     ← 图像支路
bevf_nse_tf_4x8_6e_nusc.py      bevf_pp_2x8_1x_waymo.py      lidar_stream/   ← 点云支路
bevf_tf_4x8_6e_nusc.py          bevf_tf_2x8_6e_waymo.py      drop_bbox/ drop_fov/  ← 鲁棒性实验
bevf_tf_4x8_10e_nusc_aug.py
```

命名规律：`bevf_{骨干}_{GPU数}x{batch}_{epoch}_{数据集}[_cam|_lidar]`
- `pp` = PointPillars，`cp` = CenterPoint，`tf` = TransFusion

---

## 三、第7章对我的正确用法

**❌ 不要**：跟着第7章逐行读 ADLab 代码当主线学 —— 会分散精力，且和简历技术栈不一致
**✅ 要**：当**参考书**，只在三种情况翻开

| 场景 | 为什么这时候看书有用 |
|---|---|
| **概念卡住时** | 中文详解 + 图，比啃 MIT 英文源码快。概念（融合层次、图像/点云支路）**两版是通的** |
| **看 create_data / pkl 时** ★ | 两版同源，读哪个都行。直接答问题清单 A2/C1 |
| **想对比两种设计时** ★ | 面试被问"为什么这么设计"，能说出另一种做法及取舍 = 加分 |

### 面试可用的一段话

> "BEVFusion 有两个同名工作，同期投出。MIT 那版把视图变换的 BEVPool 做了 CUDA 优化，
> 走的是精度和速度优先；北大&阿里那版是双流设计，强调单模态失效时的鲁棒性，
> 视图变换用纯 PyTorch 的 LSS。我们项目的技术栈更接近 MIT 那条线（LSS→BEVDet→BEVPoolv2），
> 但两版的数据管线是同源的，都基于 mmdet3d。"

---

## 四、一个反直觉的建议

**读 BEVPool 的实现，先读 ADLab 版。**

原因：MIT 的 `bev_pool()` 只是个壳，真正的逻辑在 CUDA kernel 里（`.cu` 文件），初学者读不动；
而 ADLab 的 `cam_stream_lss.py:85 cumsum_trick` 是**纯 PyTorch 二十行**，
和我们在 [[06_LSS原理与代码]] notebook 里手写的那段几乎一样。

**读懂 ADLab 版 → 再回头看 MIT 的 CUDA 版在优化什么 → 才能讲清简历第2条"BEVPoolv2 索引池化替代原 Splat"。**

```
理解链条：
  手写 cumsum_trick（LSS notebook，已做 ✅）
    → ADLab cam_stream_lss.py:85（纯 PyTorch 工业版）
      → MIT bev_pool + CUDA kernel（v1）
        → BEVPoolv2（预计算索引 + 融合 kernel，避免大中间张量）← 简历第2条
```

---

## 变更记录

- **2026-08-22** 建档。起因：书第7章用 ADLab 版，书上路径在 MIT 仓里找不到。
  已实测确认两仓都在工作站上，且书上 P131 的配置文件路径在 ADLab 仓里存在。
