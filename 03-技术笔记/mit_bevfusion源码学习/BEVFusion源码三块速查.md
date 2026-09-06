# BEVFusion 源码速查 · 推理链路 / 数据加载 / 损失计算

> 基于官方仓库 `mit-han-lab/bevfusion`（Apache 2.0），路径均已在实际目录中核对。
> 所有路径都是**相对仓库根目录**的，直接在你工作站上照着找。

---

## ⚠️ 先读这段：官方版 ≠ 你项目里那套

这是最容易踩的坑。你读官方代码是为了补训练期的认知，但**有三处结构性差异**，混了会在面试里说错自己的项目。

| | 官方 BEVFusion | 你项目里那套 |
|---|---|---|
| **模态融合方式** | `ConvFuser`：先 concat 再过 3×3 卷积 | 激光/毫米波各过 1×1 降到 32 通道后**逐元素相加** |
| **时序模块** | **完全没有**，单帧检测 | 有，当前帧+历史2帧，按位姿 warp 对齐 |
| **毫米波** | 有 camera+radar 配置，但只融一次 | **融两次**，第二次叠原始特征防稀释 |
| **相机路数** | 6 路针孔 | 7 针孔 + 4 鱼眼（鱼眼走 Transformer） |
| **激光强度图支路** | 无 | 有，投影成 640×256 距离图，只注入近场 |

**能从官方代码学到的**：LSS 完整实现、外积那一行、深度损失怎么算、多帧点云聚合、CBGS 重采样、CenterPoint 头的标签构造。这些和你简历上的四个优化点直接对得上。

**学不到的**：时序模块、毫米波融两次、鱼眼支路、强度图支路。这四块官方没有，只能靠你自己项目的理解。

---

## 第一块 · 推理链路

### 入口

```
tools/visualize.py          # 你跑可视化用的就是这个
tools/test.py               # 跑评测
```

`visualize.py` 的主流程很短，看 `main()` 就够：

```python
configs.load(args.config, recursive=True)          # 读配置
cfg = Config(recursive_eval(configs), filename=args.config)
dataset = build_dataset(cfg.data[args.split])      # 建数据集
model = build_model(cfg.model)                     # 建模型
load_checkpoint(model, args.checkpoint, map_location="cpu")
model.eval()
...
outputs = model(**data)                            # 推理
```

支持 `--mode gt` 和 `--mode pred` 两种模式——**只看数据不跑模型时用 gt 模式**，这是你摸数据结构最省事的入口，不用加载权重。

### 主干

```
mmdet3d/models/fusion_models/bevfusion.py    ← 整套的骨架
```

核心是 `forward_single()`（约 275 行起）。执行顺序一目了然：

```python
for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
    if sensor == "camera":
        feature = self.extract_camera_features(...)      # 相机支路
        if self.use_depth_loss:
            feature, auxiliary_losses['depth'] = feature[0], feature[-1]
    elif sensor == "lidar":
        feature = self.extract_features(points, sensor)  # 激光支路
    elif sensor == "radar":
        feature = self.extract_features(radar, sensor)   # 毫米波支路
    features.append(feature)

if not self.training:
    features = features[::-1]                # 推理时倒序，避免 OOM

x = self.fuser(features)                     # 模态融合
x = self.decoder["backbone"](x)              # BEV backbone
x = self.decoder["neck"](x)                  # BEV neck
# → 接任务头
```

**值得注意的两个细节**（面试可以当"我读过源码"的证据）：

1. 训练和推理**编码器遍历顺序是反的**，推理时倒序再翻回来，纯粹为了省显存峰值。
2. `if self.fuser is not None` 分支——单模态时不走融合，直接取唯一那份特征。

### 相机支路展开

`extract_camera_features()`（约 110 行起）：

```python
B, N, C, H, W = x.size()
x = x.view(B * N, C, H, W)                       # 多路相机压进 batch 维
x = self.encoders["camera"]["backbone"](x)
x = self.encoders["camera"]["neck"](x)
x = x.view(B, int(BN / B), C, H, W)              # 再拆回来
x = self.encoders["camera"]["vtransform"](...)   # ← LSS 在这里
```

### 视图变换（LSS 核心）

```
mmdet3d/models/vtransforms/base.py           # BaseTransform：视锥、几何、bev_pool
mmdet3d/models/vtransforms/depth_lss.py      # DepthLSSTransform：点云深度先验版
mmdet3d/models/vtransforms/aware_bevdepth.py # 带深度监督的版本
```

**`create_frustum()`** — 建视锥网格，出来是 `D×fH×fW×3`：

```python
ds = torch.arange(*self.dbound).view(-1,1,1).expand(-1, fH, fW)   # 深度轴
xs = torch.linspace(0, iW-1, fW).view(1,1,fW).expand(D, fH, fW)
ys = torch.linspace(0, iH-1, fH).view(1,fH,1).expand(D, fH, fW)
frustum = torch.stack((xs, ys, ds), -1)
```

**`get_geometry()`** — 把视锥点从图像系变到自车系。注意这一段：

```python
points = torch.cat((points[..., :2] * points[..., 2:3], points[..., 2:3]), 5)
combine = camera2lidar_rots.matmul(torch.inverse(intrins))
```

先乘深度再过内参逆，是把归一化平面坐标还原成相机系三维点的标准写法。前面还有一步 `undo post-transformation`——**撤销图像增强**，因为增强是在图像上做的，几何计算必须回到原始图像坐标。这个点面试常问："做了图像增强，投影关系怎么办？"

**⭐ 外积那一行**（`depth_lss.py` 第 82 行起，你之前问过的就是这里）：

```python
def get_cam_feats(self, x, d):
    d = self.dtransform(d)
    x = torch.cat([d, x], dim=1)          # 点云深度先验拼进图像特征
    x = self.depthnet(x)                  # ← 你简历里"加深 DepthNet"改的就是它

    depth = x[:, : self.D].softmax(dim=1)                              # D 通道 → 深度分布
    x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)  # ← 外积

    x = x.view(B, N, self.C, self.D, fH, fW)
    x = x.permute(0, 1, 3, 4, 5, 2)
    return x
```

`unsqueeze(1)` 和 `unsqueeze(2)` 是外积的全部机关：`depth` 变成 `[BN, 1, D, H, W]`，语义那半变成 `[BN, C, 1, H, W]`，广播相乘得 `[BN, C, D, H, W]`。**一条语义向量沿视线复制 D 份，按各深度的概率缩放**——和你之前理解的一致。

注意 `depthnet` 输出的前 D 个通道是深度、紧接着 C 个通道是语义，**同一个卷积头一次出两支**。这正是"加深 DepthNet 能同时改善深度和语义"的结构原因。

---

## 第二块 · 数据加载与标签结构

```
mmdet3d/datasets/pipelines/loading.py        ← 读点云、读图、读标注
mmdet3d/datasets/pipelines/transforms_3d.py  ← 增强与坐标变换
mmdet3d/datasets/pipelines/formating.py      ← 打包成 batch
mmdet3d/datasets/nuscenes_dataset.py         ← 数据集本体
mmdet3d/datasets/dataset_wrappers.py         ← CBGS 在这
tools/create_data.py                         ← 生成标注 pkl
```

### ⭐ 多帧点云聚合（对应你简历第 3 点）

`loading.py` 第 87 行 `LoadPointsFromMultiSweeps`，`__call__` 里这几行是全部关键：

```python
for idx in choices:
    sweep = results["sweeps"][idx]
    points_sweep = self._load_points(sweep["data_path"])
    ...
    sweep_ts = sweep["timestamp"] / 1e6
    points_sweep[:, :3] = points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
    points_sweep[:, :3] += sweep["sensor2lidar_translation"]
    points_sweep[:, 4] = ts - sweep_ts          # ← 时间差写进第 5 通道
    sweep_points_list.append(points_sweep)

points = points.cat(sweep_points_list)
```

**三个必须讲清的点**：

1. **运动补偿就是这两行矩阵乘加**——用 `sensor2lidar` 外参把历史帧点变到当前帧激光系。
2. **时间差不是丢掉，是当成一个输入通道**（第 5 维）。网络因此知道每个点"有多旧"。这是"多帧叠加为什么不会把动态目标拖成一条线"的答案的一半——另一半是网络能学到按时间通道加权。
3. **训练时是随机采样历史帧**（`np.random.choice`），测试时取固定前几帧。这本身就是一种增强。

> 你知识库里标红的那条——"多帧叠加**不解决**时间对齐，它**依赖**时间对齐"——在这里看得最清楚：`sensor2lidar` 外参和 timestamp 都必须先准，聚合才有意义。因果不能说反。

### 标签长什么样

`loading.py` 第 438 行 `LoadAnnotations3D`，往 `results` 里塞：

- `gt_bboxes_3d` — `LiDARInstance3DBoxes`，每个框 9 维：`x, y, z, dx, dy, dz, yaw, vx, vy`
- `gt_labels_3d` — 类别索引
- `gt_masks_bev` — BEV 分割图（`LoadBEVSegmentation`，第 244 行）

**注意第 8、9 维就是速度真值 `vx, vy`。** 它在标注里是直接给的，网络回归它——和你之前确认过的一致：速度是回归输出，不是推理时差分出来的。

### 一个 batch 里到底有什么

`formating.py` 第 131 行 `Collect3D`，`meta_lis_keys` 列表就是元信息全集：

```python
meta_keys = ("camera_intrinsics", "camera2ego", "img_aug_matrix", "lidar_aug_matrix")
meta_lis_keys = ("filename", "timestamp", "lidar2image", "depth2img", "cam2img",
                 "pcd_horizontal_flip", "pcd_rotation", "lidar_path", ...)
```

**想快速摸清 batch 结构，最省事的办法**：不用训练，直接建 dataloader 取一个 batch 打印。或者跑 `visualize.py --mode gt`，它走完整 pipeline 但不加载模型。

### ⭐ CBGS 类别重采样（对应你简历第 4 点）

`dataset_wrappers.py` 第 32 行 `_get_sample_indices()`，逻辑只有十几行：

```python
class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
for idx in range(len(self.dataset)):
    for cat_id in self.dataset.get_cat_ids(idx):
        class_sample_idxs[cat_id].append(idx)         # 每类出现在哪些帧里

duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
class_distribution = {k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()}

frac = 1.0 / len(self.CLASSES)                        # 理想均匀占比
ratios = [frac / v for v in class_distribution.values()]   # 稀有类 ratio > 1
for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
    sample_indices += np.random.choice(cls_inds, int(len(cls_inds) * ratio)).tolist()
```

**一句话说清 CBGS**：统计每类的帧占比，稀有类按倍数重复采样，让每类在一个 epoch 里出现的次数趋近均匀。

**面试官必追的两个反问，先准备好**：
- *"重采样会不会过拟合稀有类？"* → 会有风险，重复的是**帧**不是**样本**，同一帧里其他类也跟着被重复，所以实际是软化的；配合增强能缓解。
- *"为什么不用 loss 加权？"* → 加权只改梯度幅度，不改 BN 统计量和数据分布；重采样改的是网络实际见到的分布。两者可以叠加。

---

## 第三块 · 损失计算

损失**不在单独文件里**，分散在三处。

### 汇总处

回到 `bevfusion.py` 的 `forward_single()`（约 341 行起）：

```python
if self.training:
    outputs = {}
    for type, head in self.heads.items():
        if type == "object":
            pred_dict = head(x, metas)
            losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
        elif type == "map":
            losses = head(x, gt_masks_bev)
        for name, val in losses.items():
            if val.requires_grad:
                outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
            else:
                outputs[f"stats/{type}/{name}"] = val
    if self.use_depth_loss:
        outputs["loss/depth"] = auxiliary_losses['depth']
    return outputs
```

多任务头各自算损失，按 `loss_scale` 加权后汇总。**不带梯度的项自动归到 `stats/` 而不是 `loss/`**——这个小设计挺讲究，监控指标和优化目标分开。

### ⭐ 深度损失（对应你简历第 1、3 点）

```
mmdet3d/models/vtransforms/aware_bevdepth.py    第 423 行 get_depth_loss
```

```python
def get_depth_loss(self, depth_labels, depth_preds):
    if len(depth_labels.shape) == 5:
        depth_labels = depth_labels[:, 0, ...]      # 只用关键帧算深度损失

    depth_labels = self.get_downsampled_gt_depth(depth_labels)
    depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
    fg_mask = torch.max(depth_labels, dim=1).values > 0.0    # ← 只在有点的像素上算

    with autocast(enabled=False):
        depth_loss = (F.binary_cross_entropy(
            depth_preds[fg_mask], depth_labels[fg_mask], reduction='none',
        ).sum() / max(1.0, fg_mask.sum()))

    return self.depth_loss_factor * depth_loss
```

**四个必须能解释的设计**：

1. **`fg_mask`**——点云投到图上是稀疏的，绝大多数像素没有深度真值。只在有点的像素上算损失。**这正是"多帧聚合造稠密深度真值"的价值所在：`fg_mask` 覆盖率越高，监督越充分。** 你简历那个点的收益就是从这来的。
2. **用 BCE 不用回归损失**——深度被离散成 D 个 bin，当分类问题做。one-hot 真值配 BCE。
3. **`autocast(enabled=False)`**——混合精度下 BCE 数值不稳，强制走 fp32。
4. **`depth_loss_factor`（默认 3.0）**——深度是辅助任务，权重要单独调。

配置里 `depth_input: one-hot` / `scalar` 两种模式，决定深度先验以什么形式喂进 DepthNet，见 `base.py` 的 `BaseDepthTransform.forward`。

### CenterPoint 头的损失

```
mmdet3d/models/heads/bbox/centerpoint.py    第 585 行 loss / 第 432 行 get_targets_single
```

```python
heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)

# 分类：高斯 focal loss 打在 heatmap 上
loss_heatmap = self.loss_cls(preds_dict[0]["heatmap"], heatmaps[task_id],
                             avg_factor=max(num_pos, 1))

# 回归：把各回归头拼成一个 anno_box
preds_dict[0]["anno_box"] = torch.cat(
    (preds_dict[0]["reg"], preds_dict[0]["height"], preds_dict[0]["dim"],
     preds_dict[0]["rot"], preds_dict[0]["vel"]), dim=1)

code_weights = self.train_cfg.get("code_weights", None)
bbox_weights = mask * mask.new_tensor(code_weights)
loss_bbox = self.loss_bbox(pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
```

**标签构造**在 `get_targets_single()`，两个关键动作：

```python
# 1. 中心点打高斯
radius = gaussian_radius((length, width), min_overlap=self.train_cfg["gaussian_overlap"])
draw_gaussian(heatmap[cls_id], center_int[[1, 0]], radius)

# 2. 回归真值拼装
vx, vy = task_boxes[idx][k][7:]
anno_box[new_idx] = torch.cat([
    center - torch.tensor([x, y]),      # 中心亚像素偏移
    z.unsqueeze(0),                     # 高度
    box_dim,                            # 尺寸（norm_bbox 时取 log）
    torch.sin(rot).unsqueeze(0),        # 角度用 sin/cos 两路，避免 ±π 跳变
    torch.cos(rot).unsqueeze(0),
    vx.unsqueeze(0), vy.unsqueeze(0),   # ← 速度真值直接来自标注
])
```

**三个高频考点**：

- **为什么只回归中心偏移不回归中心？** 因为 heatmap 已经给了整数格位置，回归头只补格内小数部分，值域小、好学。
- **角度为什么拆 sin/cos？** 直接回归 yaw 在 ±π 处不连续，梯度会炸。
- **`code_weights` 干嘛的？** 9 个回归维度量纲不同（米、对数尺寸、无量纲三角值、米每秒），必须分别加权。**速度那两维的权重通常单独调小**——这是"速度回归为什么难"的一个具体抓手。

⭐ **再确认一次这个区分**：`preds_dict[0]["vel"]` 是**网络回归输出**；真值 `vx, vy` 来自**标注框**（nuScenes 里由相邻帧标注框差分预计算好）。时序特征只是给网络提供推断速度所需的信息。这两层不能混。

---

## 建议的读码顺序

不用从头训，按这个顺序读，每步都能独立验证：

1. **跑 `visualize.py --mode gt`** — 不加载模型，只走数据 pipeline。确认能出图。
2. **在 `Collect3D.__call__` 里下断点**，打印一个 batch 的所有 key 和 shape。**这一步做完，"标签长什么样"就通了。**
3. **读 `LoadPointsFromMultiSweeps.__call__`** — 对应你简历第 3 点，十几行看完。
4. **读 `depth_lss.py::get_cam_feats`** — 外积那三行，对应第 1 点。
5. **读 `get_depth_loss`** — 理解 `fg_mask`，把第 1 点和第 3 点串起来。
6. **读 `dataset_wrappers.py::_get_sample_indices`** — 对应第 4 点，十几行。
7. **读 `centerpoint.py::get_targets_single`** — 标签怎么从框变成 heatmap + anno_box。

前六步都不需要 GPU 训练，纯读加打印就够。第七步建议配合断点看实际张量。

---

## 一句话提醒

读官方代码是为了**补训练期的认知**，不是为了照搬。面试里讲你自己的项目时，凡是涉及时序、毫米波融两次、鱼眼、强度图的部分，官方代码里**没有对应实现**，别顺口把官方的说法搬过去。
