给你完整的地图：

## 核心 5 个文件，按阅读顺序

| #   | 文件（完整路径前缀 `/home/hpf/project/bevfusion-main/`） | 行数   | 干什么                                                               |
| --- | ---------------------------------------------- | ---- | ----------------------------------------------------------------- |
| 1   | `mmdet3d/datasets/nuscenes_dataset.py`         | 647  | **Dataset 主类**：读 pkl、`get_data_info` 拼 info dict、`__getitem__`、评测 |
| 2   | `mmdet3d/datasets/custom_3d.py`                | 306  | Dataset 基类：`__getitem__` / `__len__` / pipeline 怎么被调用             |
| 3   | `mmdet3d/datasets/pipelines/loading.py`        | 558  | **5 个"读数据"的 transform**                                           |
| 4   | `mmdet3d/datasets/pipelines/transforms_3d.py`  | 1002 | **17 个"加工数据"的 transform**（增强、过滤、归一化）                              |
| 5   | `mmdet3d/datasets/pipelines/formating.py`      | 197  | 打包：`DefaultFormatBundle3D` + `Collect3D`                          |

## 配置里那 17 步，逐个对到行号

**loading.py（把磁盘数据读进内存）**

|类|行|
|---|---|
|`LoadMultiViewImageFromFiles`|19|
|`LoadPointsFromMultiSweeps`|84|
|`LoadBEVSegmentation`|239|
|`LoadPointsFromFile`|312|
|`LoadAnnotations3D`|433|

**transforms_3d.py（加工）**

|类|行|备注|
|---|---|---|
|`ImageAug3D`|**26**|★★ 图像 resize/crop + `img_aug_matrix`，C3 的答案|
|`GlobalRotScaleTrans`|**124**|★★ 点云增强 + GT 同步 + `lidar_aug_matrix`|
|`GridMask`|159||
|`RandomFlip3D`|247||
|`ObjectPaste`|276|GT-Paste 增强|
|`PointShuffle`|447||
|`ObjectRangeFilter`|454||
|`PointsRangeFilter`|504||
|`ObjectNameFilter`|529||
|`ImageNormalize`|903||

**formating.py**：`DefaultFormatBundle3D`(14)、`Collect3D`(128)

## 另外三个配角

- `mmdet3d/datasets/builder.py`(44) — `build_dataset` / `build_dataloader` 在这
- `mmdet3d/datasets/pipelines/dbsampler.py`(323) — `ObjectPaste` 用的 GT 数据库
- `mmdet3d/datasets/dataset_wrappers.py`(76) — **CBGS 类别重采样的 wrapper 在这**（你简历第 4 条优化点）

## 建议的读法（别从头到尾读）

**只读两个函数就能打通主线**：

1. `nuscenes_dataset.py` 里的 **`get_data_info()`** —— 看它怎么把 pkl 的一条 info 变成初始 dict（`lidar2image` 那几个矩阵就是在这算的，C2 的答案）
2. `transforms_3d.py` 第 26 行的 **`ImageAug3D.__call__()`** —— 全文件最值钱的 60 行，图像变换和内参同步的耦合就在这

然后按配置里的 17 步顺序，用上面的行号表跳着读，**每个 transform 只看它的 `__call__`**（`__init__` 都是存参数，跳过）。

**顺序建议**：先跑一次 `pipeline_dump`（我上条提的那个工具，两秒），拿到"每步新增/改动了什么"的流水表，再对着流水表去翻源码——**先知道结果再看实现**，比硬啃 1002 行快得多。要我现在把那个工具写了吗？