> [[00_总览与脉络|📖 总览]] · [[Ch02_DepthGT与DepthLoss|Ch2 →]]

# Ch1 DepthNet 网络结构（00:00:00–00:06:42）

> **你在地图的哪一站**：整个视频的流水线是
> `[未覆盖：Dataset/DataLoader、图像backbone] → 【FPN收尾 → DepthNet】→ Depth Loss → Lidar Backbone → Radar Backbone → RL融合 → LSS投影 → 多视角融合 → RC融合 → 模态融合 → MemoryManager → 时序融合 → BEV UNet → CenterPoint头 → Loss → Box解码`。
> 本章是全片第一章，讲者从 **FPN 的收尾**讲起（backbone 已经跑完、特征已经进来了），一路把图像特征送进 **DepthNet**，终点是拿到深度分布 logits `21×100×88×160`。下一章（Ch2, 00:06:42 起）才讲这个 logits 怎么算 Loss。

---

## 0. 开讲前必读：画面环境 + 术语档案 + 帧证清单

### 0.1 讲者的屏幕上是什么

整段视频是讲者在**远程开发机的 IDE 里边调试边讲**（画面上不时出现"正在加载…xxKB/s"的远程桌面加载提示，右上角有 `pilei p00804189` 的屏幕水印）。本章涉及两个核心文件，路径从 IDE 面包屑上可以完整读出：

| 文件 | 完整路径（IDE 面包屑） | 本章相关类 |
|---|---|---|
| `resnet_forward.py` | `workspace/00_DEBUG_develop/e2e/tasks/bev_task/uvp_module/models/streampetr/streampetr_backbone/resnet_forward.py` | `BackBoneForward`、`NeckForward`（约 79 行 / 181 行起） |
| `fpn_forward.py` | `workspace/00_DEBUG_develop/e2e/tasks/bev_task/uvp_module/models/streampetr/streampetr_neck/fpn_forward.py` | `FPNForward`（24 行起）、**`DepthNet`（528 行起）** |

注意目录名是 **`streampetr`**——这套 DenseBEV 工程是在 StreamPETR 系（稀疏查询路线，讲者口中的"SparseBEV/SpaS BV"）的老代码基座上改出来的稠密 BEV 方案，所以代码里到处是"以前稀疏方案用、现在不用"的遗留分支——这正是本章前半段反复出现的主题。

IDE 顶部还开着一排标签页：`weighter.py`、`multi_fusion_dataset.py`、`dataset_bevaug.py`、`bev_dataset_aug.py`、`dataset_multiframebev.py`、`module_container.py`、`resnet_forward.py`、`grid_mask.py`、`fpn_forward.py`、`ddn_loss.py`——这排标签页本身就是后面几章的"目录预告"（`ddn_loss.py` 是 Ch2 的主角）。

底部调试控制台是一个**真实训练任务暂停在断点上**的现场，日志可读出（时间戳 `2025-09-20 14:08`）：

```
Task: bev_dataset, Training batch numbers 1839 / epoch
Rank 0: group current network parts.
Rank 0: gather all network groups.
Rank 0: initialize all process groups.
Rank 0: transfer network to GPU and convert BatchNormalization to SyncBatchNorm,
Rank 0: init saver, build solver, using torch built-in solvers,
Enable discarding step when abnormal gradients detected,
Rank 0: multi-task work start.
StreamDevEffectiveV2Base_gv_img mean=True data0.dtype=torch.uint8 Mean:28.699 Min:0 Max:255
```

三个可挖的信息：①一个 epoch 有 1839 个 batch；② BN 全部转 **SyncBatchNorm**（多卡训练标配，BEVFusion 官方配置里也这么干）；③"discarding step when abnormal gradients detected"——梯度异常就丢弃该步，工程上防炸 loss 的保险丝，这在你自己训 BEVFusion 出现 loss 突刺时可以直接借鉴。

### 0.2 术语档案（转写稿噪声 → 真实代码名）

| 转写稿写法 | 真实所指 | 证据 |
|---|---|---|
| SpaS / SpaS BV | SparseBEV/StreamPETR 系的**旧稀疏方案** | 代码目录就叫 `streampetr` |
| Dons BV | **DenseBEV**（当前方案） | 视频标题 |
| FPNFW / "FPNFW的模块" | `fpn_forward.py` 里的 `FPNForward` 类 | 00:01:43 帧面包屑 + 类名 |
| "Blat model" | **`flatten_mode`**（沿高度拍平的模式开关） | 00:02:18 帧代码 `if self.flatten_mode == 1:` |
| "DepthNetload" | DepthNet 的 **dataload**（数据加载时对 depth GT 的分组处理） | 上下文 |
| "取Label" | 取 `label_obj`/图像原始尺寸等元信息 | 00:00:08 帧 `self._img_ori_sizes` |
| "12号相机" | ⚠ 推断为"**1、2号（前视主）相机**"与侧/后相机的分组（代码里叫主组 `''` 与 `endpart_` 组） | 00:04:55 帧 `grp_endpart = 'endpart_'` |
| "只缺了下采样8倍" | 应为"只**取**了下采样8倍" | 00:06:29 调试输出 |

### 0.3 本章帧证清单（本章解释里所有代码行、形状数字的出处）

精读单帧 13 张：`00_00_08`、`00_00_38`、`00_01_43`、`00_02_18`、`00_02_45`、`00_03_23`、`00_04_10`、`00_04_31`、`00_04_37`、`00_04_55`、`00_05_01`、`00_05_19`、`00_05_33`、`00_05_52`、`00_06_11`（+ `00_06_11/00_06_29` 底部控制台放大裁剪）。

其中**全章最关键的一块屏幕证据**，是 00:06:11–00:06:29 调试控制台的三次交互（我逐字抄录）：

```python
> depth_input.shape
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AttributeError: 'list' object has no attribute 'shape'

> [i.shape for i in depth_input]
[torch.Size([21, 128, 88, 160]), torch.Size([21, 256, 44, 80]), torch.Size([21, 256, 22, 40])]

> depth_logits.shape
torch.Size([21, 100, 88, 160])
```

这三行就是本章的"标准答案"：DepthNet 的输入是**3个尺度的特征列表**（8×/16×/32×），输出是 `21×100×88×160` 的深度 logits。讲者连自己敲错（对 list 取 `.shape` 报 AttributeError）都被录了下来——这恰好教了我们一件事：**这里的 depth_input 不是张量，是多尺度列表**。

---

## Part 1　FPN 收尾：NeckForward 里的针孔/鱼眼双分支遍历（00:00:00–00:01:43）

**导读**：本段的输入是图像 backbone（ResNet 系）已经算好的多尺度特征（针孔一套、鱼眼一套），输出是经过各自 FPN 融合后的特征列表。位置上这是"图像分支的最后一公里"：backbone 和 Dataset 视频没讲，讲者直接从 `NeckForward` 这个"脖子"模块开讲。要抓住的主线只有一条：**代码按"相机分组（grp）"组织——针孔组 `''` 和鱼眼组 `'fisheye_'` 各自有自己的 FPN 网络，用 `getattr(self, f'{grp}fpn_nets')` 动态取出来跑**。

---

#### 卡 1 ｜ [00:00:00]（重点句，5角度）

> **原话**：然后接下来是我们的 DepthNet。FPN 主要的输入的话就是我们的针孔的 backbone 和鱼眼的 backbone 的输入过来的几个不同的分辨率的特征。

【直译】开场定调：本章目标是 DepthNet，但得先把它的上游 FPN 交代完。FPN 吃的是两类相机 backbone 的输出——7 路**针孔**相机一套、4 路**鱼眼**相机一套——每套都是多个分辨率（多尺度）的特征图。

【代码】00:00:08 帧显示 `NeckForward.__init__`（`resnet_forward.py` 181 行起）里分组逻辑的真实写法：

```python
self.split_pinhole_in_fpn = cfg.get('split_pinhole_in_fpn', False)
cams = set(cfg.data.camids)
self._img_ori_sizes = [self.cfg.data.input_size, self.cfg.data.fisheye_input_size]
if self.split_pinhole_in_fpn:
    self._img_ori_sizes.insert(1, self.cfg.data.endpart_input_size)

grps_list = list(cam_grp(x, self.split_pinhole_in_fpn) for x in cams)
self._fv_grps = sorted(set(grps_list))              # 例如 ['', 'fisheye_']
self._num_views = [grps_list.count(grp) for grp in self._fv_grps]   # 例如 [7, 4]
assert len(self._fv_grps) >= 1

# fv networks (fv_backbone with depthnet if specified)
for idx, grp in enumerate(self._fv_grps):
    # todo:current only pinhole,need to add fisheye
    fpn_nets = build_fpn_nets(cfg, grp=grp)
```

`cam_grp()` 把每个相机 id 映射到组名（针孔→空串 `''`，鱼眼→`'fisheye_'`），`_num_views` 数出每组几路相机。**每组各建一个 FPN**（`build_fpn_nets(cfg, grp=grp)`）。注意 `# fv_nets = build_fv_backbone(...)` 被注释掉了——backbone 不在这个模块里建，`NeckForward` 收到的 `*input` 就已经是特征了。类头上还挂着 `@frozen_modules` 装饰器，暗示这套图像分支支持整体冻结（大模型多模态训练常用手段）。

【形状】backbone 输出四个尺度（00:04:10 帧的旧注释）：`C1~C4 = (N,64,176,320)/(N,128,88,160)/(N,256,44,80)/(N,512,22,40)`，即下采样 4/8/16/32 倍、通道 64/128/256/512（典型 ResNet18/34 宽度）。反推**针孔输入图像是 704×1280**（88×8=704、160×8=1280，四个尺度全部对得上）。鱼眼输入尺寸独立（`fisheye_input_size`），本章没展示具体值。

【为什么】为什么针孔/鱼眼要拆两套 FPN 而不是共享？因为两类相机的成像模型（针孔投影 vs 等距鱼眼畸变）、分辨率、视场角都不同，特征分布差异大，共享权重会互相拖累；拆组还让"针孔先跑通、鱼眼后接入"的渐进开发成为可能——代码里那句 `todo: current only pinhole, need to add fisheye` 就是这种开发节奏的化石。

【连接】对照你熟的 BEVFusion：它在 nuScenes 上只有 6 路针孔（`mmdet3d` 里 `imgs` 直接 `B*6` 展平进 SwinT+FPN），没有鱼眼分组问题；这里是量产车 7 针孔+4 鱼眼的双组版，"组"这个维度是 BEVFusion 没有的新东西。而"多尺度特征进 FPN"这件事本身，和你在智谷 YOLO 课里看的 FPN/PAN 完全同源——都是 top-down 上采样融合。

---

#### 卡 2 ｜ [00:00:28] + [00:00:39]（合并：[00:00:39] 只有一个词"FPN"，是上句拖尾）

> **原话**：然后的话在这里会去分别遍历针孔和针孔的 FPN 以及鱼眼的 FPN。（FPN。）

【直译】forward 的时候不是一把梭，而是**循环两个组**：第一轮跑针孔的 FPN，第二轮跑鱼眼的 FPN。（口误重复了"针孔和针孔的"，实际是"针孔的 FPN 和鱼眼的 FPN"。）

【代码】00:04:10 帧的 `NeckForward.forward`（241 行起）就是这个循环：

```python
for idx, grp in enumerate(self._fv_grps):     # grp ∈ {'', 'fisheye_'}
    img_fpn = getattr(self, f'{grp}fpn_nets')  # 动态取本组的FPN
    num_outs = len(img_fpn.in_channels)        # =3（8×/16×/32×三个尺度）
    img_feat = input[(idx * num_outs):(idx * num_outs) + num_outs]  # 切出本组的3个尺度
```

`getattr(self, f'{grp}fpn_nets')` 这种 **f-string 前缀 + getattr** 的动态属性访问，是这套代码库贯穿全片的组织习惯（后面 depthnet、ddn_loss 全是同款），针孔组前缀是空串所以属性名就叫 `fpn_nets`，鱼眼组叫 `fisheye_fpn_nets`。

【形状】`input` 是一个拍平的特征列表：每组 3 个尺度张量，按组顺序排。两组时 `len(input)≥6`，`idx=0` 取 `input[0:3]`（针孔），`idx=1` 取 `input[3:6]`（鱼眼）。

【为什么】用"列表切片 + 组循环"而不是字典传参，是为了兼容老框架 `forward(self, *input)` 的位置参数约定（也方便 ONNX 导出时输入是纯张量序列）；代价是可读性差——你必须知道 `num_outs` 才能算出谁是谁，这也是讲者要花 6 分钟带大家"对切片"的原因。

【连接】BEVFusion 里对应物是 `img_neck(img_feats)` 一次调用完事；这里因为两组相机异构，等价于把 BEVFusion 的 neck 复制两份、循环调用。

---

#### 卡 3 ｜ [00:00:47]

> **原话**：然后调用的是这里的模块。

【直译】讲者用鼠标指了指循环体里被调用的对象——即上面 `getattr` 取出来的那个 FPN 模块实例。

【代码】被调用处是（00:04:10 帧 267 行附近）：

```python
img_feat, location, depth_loss, depth_probs, depth_input, fpn_feat = \
    img_fpn(img_feat, img_ori_shape, grp, kwargs)
```

一次 FPN 调用居然返回 6 个东西——除了融合后特征 `img_feat`，还捎带了 `location`（位置编码，旧方案遗留）、`depth_loss/depth_probs`（FPN 内嵌深度头的产物，当前配置为空转）、`depth_input`（给 DepthNet 的特征）、`fpn_feat`（原始多尺度 outs）。这暴露了一个工程事实：**FPN 模块被历史包袱撑肥了**，它既是 neck 又半个 depth 头。

【形状】进：3 个尺度的列表 `[(21,128,88,160),(21,256,44,80),(21,512,22,40)]`（针孔组，通道为 backbone 的 C2~C4）。出：`img_feat` 变成 `(bst,num_view,C,H,W)` 的 5 维视图 + 上述 5 个附属输出。

【为什么】返回值这么多而不重构成 dict，是老代码"只加不删"演化的典型形态——每加一个功能就在 return 尾巴上挂一个变量，下游按位置解包。看工作代码时要习惯这种"考古地层"。

【连接】你在 BEVFusion 里看到的 `mmcv/mmdet` 风格是 neck 返回 tuple of tensors、loss 归 head 管，职责干净；对比之下这里是"车企自研框架"风格，读代码时更依赖调试器而不是文档——这正是本视频通篇用断点讲课的原因。

---

#### 卡 4 ｜ [00:00:56]

> **原话**：然后 ImageFPN。ImageFPN 的话是调用的 FPNFW 的里面的 FPNFW 的模块。

【直译】这一层套娃关系：配置里叫 `ImageFPN` 的组件，实现体是 `fpn_forward.py`（讲者口中的"FPNFW"）文件里的 `FPNForward` 类。也就是说 `build_fpn_nets` 建出来的实例，类型就是 `FPNForward`。

【代码】00:01:43 帧证实：`fpn_forward.py` 24 行 `class FPNForward(BaseModule):`，223 行 `def forward(self, img_feats, img_ori_shape, grp, kwargs):`。文件面包屑在 `streampetr_neck` 目录下。

【形状】不涉及新形状；这句是"配置名 → 文件名 → 类名"的三级映射说明。

【为什么】为什么要强调这个映射？因为这套框架靠注册器/builder 从配置字符串建网络（和 mmdet 的 Registry 同思路），**看配置猜不出代码在哪个文件**，必须记住 `ImageFPN → fpn_forward.py → FPNForward` 这条线，以后自己跳转才不迷路。

【连接】等价于 mmdet 里 `type='FPN'` 映射到 `mmdet/models/necks/fpn.py::FPN`。你调 BEVFusion 配置时改 `img_neck=dict(type=...)` 就是在动同一类映射。

---

#### 卡 5 ｜ [00:01:10]

> **原话**：然后前面的话这些都是……因为这里为什么要取 Label 呢？是因为以前我们 SpaS 里面会去需要取图像的一个原始的一个尺寸。然后当前的话这些是没有用到的。对。

【直译】`FPNForward.forward` 开头有一段"取 Label/取图像原始尺寸"的代码，是给**以前的 SparseBEV（稀疏）方案**用的——稀疏方案要把 3D 查询点投回原图，必须知道图像原始宽高；**当前 DenseBEV 用不上，属于死代码**。

【代码】与之呼应的是 `NeckForward` 里给 FPN 拼原始尺寸的那行（00:04:10 帧 264 行附近）：

```python
bst = img_feat[0].shape[0] // self._num_views[idx]      # 21//7 = 3
img_ori_shape = bst, self._num_views[idx], 3, \
    self._img_ori_sizes[idx][1], self._img_ori_sizes[idx][0]   # (3, 7, 3, H_ori, W_ori)
```

注意 `sizes[idx][1], sizes[idx][0]` 的**反序取法**——配置里存的是 `(W, H)`，拼元组时换成 `(H, W)`。这类小细节是以后自己接相机时最容易踩的坑。

【形状】`img_ori_shape = (bst=3, num_view=7, 3, H_ori, W_ori)`，纯元数据元组，不是张量。

【为什么】稀疏方案（StreamPETR/SparseBEV）的核心操作是"3D 参考点 → 投影到各相机原图 → 采样特征"，投影必须用原图尺寸做归一化；DenseBEV 走 LSS 路线（特征图逐像素抬升），只关心特征图尺寸，所以原图尺寸线索就闲置了。**但代码没删**——量产代码库里稀疏/稠密两套方案共存，靠配置切换，删了怕别的任务还要用。

【连接】记一个通用判断力：读大厂工程代码，先问"这段是活的还是死的"，判断依据不是注释而是**当前配置 + 调试器**。这也是讲者选择"边调试边讲"的根本原因。你以后读华为内部 BEV 代码，第一步同样是拿一份真实配置跑通断点。

---

#### 卡 6 ｜ [00:01:36]

> **原话**：然后这里的话就是 FPN 的一些具体的一些实现的一些细节。这些都不用讲。

【直译】FPN 内部（lateral 卷积、top-down 上采样相加）是教科书标准件，讲者跳过。**但我们不跳**——00:01:43 帧恰好把这段代码整屏拍清楚了，而且里面藏了一个非标准细节。

【代码】00:01:43 帧（`fpn_forward.py` 231–256 行，讲者当时还选中了一大段代码高亮着）：

```python
assert len(img_feats) == len(self.in_channels)
# build laterals
laterals = [lateral_conv(img_feats[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)]
# build top-down path
used_backbone_levels = len(laterals)
for i in range(used_backbone_levels - 1, 0, -1):
    if 'scale_factor' in self.upsample_cfg:
        laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
    else:
        prev_shape = laterals[i - 1].shape[2:]
        laterals[i - 1] = laterals[i - 1] + F.interpolate(
            laterals[i], size=prev_shape, **self.upsample_cfg)
# part 1: from original levels
outs = [
    self.fpn_convs[i](laterals[i]) if i == 0 else laterals[i]
    for i in range(used_backbone_levels)
]
```

前半段与 mmdet 官方 `FPN.forward` 逐行同源（连注释都一样，后面还有 `add_extra_convs == 'on_input'`、`F.max_pool2d(outs[-1], 1, stride=2)` 这些 RetinaNet 式加层代码）。**非标准点在 `outs` 那行**：mmdet 是对每层都过 `fpn_convs[i]` 的 3×3 卷积，这里改成了 **只有 i==0（8×那层）过卷积，16×/32× 直接把 lateral 原样输出**。

【形状】laterals 全部被 lateral 1×1 卷积对齐到 FPN 通道数（本工程约 256）；`outs[0]` 再过 3×3 fpn conv。结合 00:06:11 的实测 `depth_input = [(21,128,88,160),(21,256,44,80),(21,256,22,40)]` 可以反推：**`fpn_convs[0]` 把 8× 那层从 256 压到了 128 通道**，16×/32× 保持 lateral 的 256 通道——三个尺度通道数不一致的"怪相"在这里找到了出处。

【为什么】只给 8× 层配 3×3 卷积是算力取舍：后续所有正主（DepthNet、LSS 投影）只吃 8× 特征，16×/32× 只是陪跑传下去"备用"（见卡 12），给它们配卷积纯属浪费；8× 层过一个 3×3 还能平滑 top-down 相加后的混叠伪影（aliasing），这正是 FPN 论文里 3×3 conv 的原始动机。

【连接】智谷 YOLO 课里 FPN/PAN 的每个输出层都要接检测头，所以每层都得有输出卷积；这里"单消费层"的 FPN 就可以砍掉冗余——**网络结构跟着消费者裁剪**，是从课程 demo 到量产代码的典型思维升级。

---

### 🔨 动手练习 ch1-1：分组遍历 + 只给一层配 fpn_conv 的迷你 FPN

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MiniFPN(nn.Module):
    """复刻本段两个要点：lateral+top-down，且只有第0层过fpn_conv(256->128)"""
    def __init__(self, in_channels=(128, 256, 512), mid=256, out0=128):
        super().__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, mid, 1) for c in in_channels])
        self.fpn_conv0 = nn.Conv2d(mid, out0, 3, padding=1)   # 只服务8×层
    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):             # top-down
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
        return [self.fpn_conv0(laterals[0])] + laterals[1:]   # i==0才过卷积

# 造假数据：N=21(3帧x7路针孔), 尺度按真实值 88x160 / 44x80 / 22x40
feats = [torch.randn(21, 128, 88, 160),
         torch.randn(21, 256, 44, 80),
         torch.randn(21, 512, 22, 40)]
fpn = MiniFPN()
outs = fpn(feats)
for idx, grp in enumerate(['', 'fisheye_']):                  # 组遍历的意思
    print(f"grp='{grp}' -> 属性名 {grp}fpn_nets")
print([tuple(o.shape) for o in outs])
# 预期输出（与00:06:11调试台完全一致的三元列表）：
# [(21, 128, 88, 160), (21, 256, 44, 80), (21, 256, 22, 40)]
```

**【小结】** ① 图像分支按相机组（针孔 `''`/鱼眼 `'fisheye_'`）组织，各组独享一套 FPN，用 `getattr(self, f'{grp}fpn_nets')` 循环调用；② FPN 主体与 mmdet 逐行同源，但只有 8× 层过 3×3 输出卷积（256→128），16×/32× 直接输出 lateral；③ forward 开头取原图尺寸/Label 的代码是稀疏方案遗留，当前 DenseBEV 不用。

---

## Part 2　SparseBEV 遗留分支：flatten_mode 拍平与位置编码（00:01:48–00:02:32）

**导读**：本段没有任何"活代码"——讲者快进式地掠过 `FPNForward` 里两块 SparseBEV 时代的遗产：①沿高度把侧向相机特征"拍平"的 `flatten_mode` 分支；②给每个特征位置算位置编码的 `prepare_location`。它们的输入输出在当前配置下都不发生，但**读懂它们能反向理解稀疏方案为什么被换掉**（图像 token 太多、依赖位置编码做隐式几何）。这也是本章"考古"色彩最重的一段。

---

#### 卡 7 ｜ [00:01:48]（重点句，5角度）

> **原话**：然后这里的话是有一个 Blat model（→ `flatten_mode`）。是因为我们以前 SpaS BV 在这里的话会因为图像的 Token 太多，所以说会把侧向的针孔相机沿着高度做一个拍平的操作。

【直译】旧稀疏方案里，图像特征要被展成 token 序列送进注意力模块；11 路相机的特征图全展开 token 数爆炸，于是对侧向针孔相机的特征**沿高度维求平均/降维（"拍平"）**，牺牲垂直分辨率换 token 数。控制这套行为的开关叫 `flatten_mode`（转写稿的"Blat model"就是它）。

【代码】00:02:18 帧把 `flatten_mode` 的 4 档分支拍得很全（`fpn_forward.py` 286–323 行）：

```python
bst, num_view, c, pad_h, pad_w = img_ori_shape
bstn, C, H, W = img_feat.shape
assert bst * num_view == bstn                       # 3*7 == 21，前后自洽校验

if self.flatten_mode == 1:                          # 全图高度均值
    img_feat = img_feat.mean(dim=2).unsqueeze(2)    # H -> 1
elif self.flatten_mode == 2:                        # 学习式拍平（专用小网络）
    feat_flatten = getattr(self, f'{grp}feat_flatten')
    img_feat = feat_flatten(img_feat)
elif self.flatten_mode == 3:                        # 高度二分后组内均值
    bts, c, h, w = img_feat.shape
    img_feat = img_feat.reshape(bts, c, 2, -1, w)
    img_feat = img_feat.mean(dim=3)                 # H -> 2
elif self.flatten_mode == 4:
    if self.part_flatten and grp == '':             # 只拍部分相机
        front_img_feat = img_feat[:, self.front_pin, :, :, :].view(-1, C, H, W)
        rear_img_feat  = img_feat[:, self.rear_pin,  :, :, :].view(-1, C, H, W)
        if self.main_pin_flatten:
            front_img_feat = front_img_feat.reshape(n, C, -1, 2, W).mean(dim=3)
            ...
        new_img_feat = {'front_': front_img_feat, 'rear_': rear_img_feat}
```

`front_pin/rear_pin` 是相机索引张量（首次用会 `torch.tensor(...).to(device)` 搬上卡），mode 4 甚至把特征拆成 front/rear 两个 dict 分别拍——可见当年为压 token 数迭代过至少 4 版方案。

【形状】mode1：`(21,C,H,W)→(21,C,1,W)`（token 数 ÷H）；mode3：`(21,C,H,W)→(21,C,2,W)`（把 H 折成 2×H/2 再对 H/2 求均值，保留"上/下半图"两行）；mode4 更细，按相机子集拍。以 8× 特征 H=88 计，mode1 直接把每路相机 token 从 88×160=14080 压到 160。

【为什么】稀疏方案（PETR/StreamPETR 系）要做全局 cross-attention，复杂度正比 token 数；侧向相机对远距离纵向定位贡献有限，沿高度拍平是"信息换算力"的定向阉割。而 DenseBEV 走 LSS：每个像素独立抬升、无全局注意力，token 爆炸问题不存在，这套开关便整体废弃——**方案换代的真正原因往往写在被废弃的代码里**。

【连接】"沿高度拍平"你其实见过近亲：BEV 里把 Z 轴 collapse 掉得到 2D BEV 特征（BEVFusion 的 `bev_pool` 后 flatten Z）就是同一哲学——**在信息量最低的轴上做压缩**。图像的 H 轴之于侧向相机，近似 BEV 的 Z 轴之于地面场景。⚠ 校正稿头部提示"拍平用均值还是最大值建议对代码核实"：本帧代码明确是 **`.mean()` 均值**（mode1/3/4 都是），最大值池化在此分支未出现。

---

#### 卡 8 ｜ [00:02:06]

> **原话**：然后对于当前我们 Dons BV（→DenseBEV）的话，这些实现其实都没有涉及到。

【直译】再次确认：上面整套 flatten 分支在 DenseBEV 配置下一行都不会执行（`flatten_mode` 取的值走不进任何拍平分支，或 `part_flatten=False`）。

【代码】等价于配置里 `flatten_mode=0`（或不在 1~4），`if/elif` 全部落空，`img_feat` 原样通过。验证方法就是断点：在 mode 分支里下断点，跑一个 step 不命中即死代码。

【形状】特征保持 `(21,C,88,160)` 不变——这正是后面 LSS 需要的完整 2D 分辨率。

【为什么】DenseBEV 恰恰**需要**保留高度维：LSS 的每个像素都要沿深度 bin 抬升成视锥点，拍平了高度等于把视锥压成一条线，几何信息全毁。所以这不是"懒得删"，而是两种范式对特征形态的要求根本冲突。

【连接】面试高频对比题素材：稀疏 query 方案（PETR 系）压 token 保注意力，稠密 LSS 方案保分辨率压深度 bin 数——两条路线的算力都花在刀刃上，只是刀刃不同。

---

#### 卡 9 ｜ [00:02:18] + [00:02:32]（合并：两句同讲位置编码遗留）

> **原话**：然后这里的话也是以前 SpaS 会用到，去算每一个特征它的一个位置编码。……然后在这里的话其实对于当前这一套也不会用。

【直译】FPN forward 里还有一处给每个特征像素算"位置/坐标编码"的调用，同样是稀疏方案专用，DenseBEV 不用（虽然函数还在跑，见下）。

【代码】00:02:45 帧 369 行：

```python
if isinstance(img_feat, dict):
    ...
else:
    location = self.prepare_location(img_ori_shape, img_feat)
```

`prepare_location` 是 StreamPETR 谱系的标志性函数：按特征图网格生成每个位置的归一化 2D 坐标（配合相机参数可进一步变 3D 位置编码），供 PETR 式"3D position embedding + attention"用。注意它**没有被 if 掉**，每个 step 都会算并作为 `location` 一路 return 出去（`return img_feat, location, ...`）——只是下游没人消费，属于"活着的死代码"，白花一点算力。

【形状】典型实现下 `location ≈ (bst*num_view, H*W, 2)` 的归一化网格坐标（本段画面未展示其内部，标 ⚠ 推断）。

【为什么】PETR 系为什么离不开位置编码？因为它把图像特征当无序 token 集合送进 attention，几何关系全靠 position embedding 注入；而 LSS 用显式的相机内外参投影建立几何，坐标关系是**算出来**的不是**学出来**的，位置编码自然失业。

【连接】这与你在 Transformer 通识里学的"attention 无位置感、必须加 PE"是同一条原理在 3D 感知的投影。以后读 StreamPETR 论文源码，`prepare_location`/`position_embedding` 就是核心函数，这里算提前打了照面。

---

### 🔨 动手练习 ch1-2：三种拍平方式的形状实验

```python
import torch

x = torch.randn(21, 128, 88, 160)          # 3帧x7路针孔的8×特征

# mode 1: 高度全拍平
m1 = x.mean(dim=2).unsqueeze(2)
# mode 3: 高度折成2段，段内均值
m3 = x.reshape(21, 128, 2, -1, 160).mean(dim=3)
# 稀疏方案的token数对比
tok_full, tok_m1, tok_m3 = 88*160, 1*160, 2*160
print(m1.shape)   # 预期 torch.Size([21, 128, 1, 160])
print(m3.shape)   # 预期 torch.Size([21, 128, 2, 160])
print(f"每路相机token数: 原始{tok_full} -> mode1:{tok_m1}(÷88) -> mode3:{tok_m3}(÷44)")
# 预期: 每路相机token数: 原始14080 -> mode1:160(÷88) -> mode3:320(÷44)

# 顺手复刻 prepare_location 的思想：特征网格归一化坐标
H, W = 88, 160
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
location = torch.stack([(xs + 0.5)/W, (ys + 0.5)/H], -1).view(-1, 2)
print(location.shape, location[0], location[-1])
# 预期 torch.Size([14080, 2]) tensor([0.0031, 0.0057]) tensor([0.9969, 0.9943])
```

**【小结】** ① `flatten_mode`（转写"Blat model"）是稀疏方案压图像 token 的 4 档拍平开关，全部用 `.mean()` 沿高度压缩，DenseBEV 一档不用；② `prepare_location` 位置编码是 PETR 系 attention 的几何注入手段，LSS 用显式投影替代了它，但函数仍在空转白算；③ 这两块遗产合起来回答了"为什么换稠密方案"：token 爆炸 + 隐式几何，换成 LSS 的逐像素显式抬升。

---

## Part 3　取下采样 8 倍特征：FPN 的输出打包（00:02:39–00:04:01）

**导读**：本段是 FPN 与 DepthNet 之间的"交接仪式"。输入是 FPN 融合好的 3 尺度特征，输出是两样东西：①提前抽出的**下采样 8 倍特征**（DepthNet 的正餐）；②整个多尺度列表（历史上 DepthNet 吃过多尺度，现在只是陪跑传下去）。在流水线上，这一步决定了后续深度预测、LSS 投影全部发生在 8× 分辨率（88×160）上。

---

#### 卡 10 ｜ [00:02:39] + [00:02:50]（合并：一句话被断成两半；重点句，5角度）

> **原话**：然后在这里是会取我们下采样 8 倍的一个特征，作为我们在后续模块 DepthNet——就是预测 Depth 的时候——作为它的一个输入。然后在这里就提前从 FPN 里面取出了下采样 8 倍的一个图像的一个特征。

【直译】在 FPN 的 forward 末尾，把 `outs` 里的第 0 个元素（8× 特征）单独拎出来，起名 `depth_input`，专门作为 DepthNet 的输入提前准备好。

【代码】00:02:45 帧 375 行（讲者当时把 `depth_input` 高亮了）：

```python
depth_input = outs[0]   # 14,256,88,160   <- 旧注释，形状已过时！
depth_loss = 0.0
depth_logits = None
if self.with_depthnet:
    ...                 # FPN内嵌的深度头分支（当前配置未启用，见下）
return img_feat, location, depth_loss, depth_logits, depth_input, outs
```

`outs[0]` 就是 Part1 卡 6 里唯一过了 3×3 fpn conv 的 8× 层。它和完整的 `outs` 一起被 return，在 `NeckForward` 里分别 append 进 `depth_inputs` 和 `fpn_feats` 两个列表。

【形状】**旧注释说 `14,256,88,160`，实测（00:06:11 调试台）是 `21,128,88,160`**。两处差异都值得抠：`14→21` 是帧数配置从 2 帧×7 路升到 3 帧×7 路；`256→128` 是 fpn_convs[0] 后来加了通道压缩。⚠ 结论：**这行注释是老版本的化石，以调试器为准**——讲者全片反复用断点验形状，就是对"注释会说谎"的最好示范。

【为什么】为什么深度预测选 8× 而不是 4×（更精细）或 16×（更省）？8×=88×160 是精度/算力的甜点位：LSS 要为**每个像素×每个深度 bin** 生成一个视锥点，点数 = N×D×H×W = 21×100×88×160 ≈ **2960 万**点；若用 4× 特征点数翻 4 倍直接爆显存，用 16× 则 BEV 网格（0.4m 分辨率）会因图像投影点过稀出现空洞。88×160 也和 BEV 侧"在 224×112 上投影"的设计（总配置：448×224 网格、投影在半分辨率上）算力量级匹配。

【连接】BEVDepth/BEVFusion 同款选择：BEVFusion 的 `DepthLSSTransform` 也吃 stride=8 的 FPN 输出（nuScenes 上 256×704 图 → 32×88 特征）。你训 BEVFusion 时改 `image_size` 会连锁改这里的 H×W——量级概念完全可迁移。

---

#### 卡 11 ｜ [00:03:00]

> **原话**：然后在这里也把原始的、我们在这里的 FPN 所出来的几种不同尺度的一个特征也传出去了。

【直译】除了单抽的 8× 特征，`return` 的最后一项把整个 `outs`（8×/16×/32× 三个尺度的完整列表）也原样传了出去。

【代码】`return img_feat, location, depth_loss, depth_logits, depth_input, outs` 的最后一位；上游接住后 `fpn_feats.append(fpn_feat)`（00:04:10 帧 272 行，讲者高亮了 `fpn_feats`）。

【形状】`outs = [(21,128,88,160), (21,256,44,80), (21,256,22,40)]`——注意第三个尺度实测是 **256 通道、22×40**（32× 下采样）。⚠ 22×40 与 backbone C4 的空间尺寸一致，通道 256 说明它出自 lateral（512→256 的 1×1），而非 extra-level 池化；此推断与卡 6 的"只有第 0 层过 fpn_conv"互相咬合。

【为什么】多尺度全量透传是一种**接口冗余设计**：下游模块想吃哪层自己挑，上游不做假设。代价是显存里多躺两个没人吃的张量（16×/32× 合计约 21×256×(44×80+22×40)×4B ≈ 94MB，FP32、不含梯度），量产裁剪时这就是第一批优化点。

【连接】YOLO 的 neck 输出 P3/P4/P5 三层是因为三个头都要用；这里传三层却只用一层，是"历史接口"而非"当前需求"——同样的结构，动机可以完全不同，读代码要问动机。

---

#### 卡 12 ｜ [00:03:14] + [00:03:25] + [00:03:35]（合并：三句是同一件事的推进与收束）

> **原话**：也是我们之前就是在预测 DepthNet 的时候用了一个多尺度的一个特征。所以说在这里也会额外传出去。但是当前的话应该也没有在 DepthNet 的时候用到多尺度的特征，也只是用到了一个下采样 8 倍的一个图像的一个特征。

【直译】历史剧情补完：早期版本的 DepthNet 真的吃过多尺度（大概率类似 BEVDepth 的多层融合深度头），所以接口传全量；**当前版本退化为只吃 8×**，多尺度就成了摆设。注意讲者说"应该也没有用到"——他自己也是靠记忆+调试确认，不是背文档。

【代码】证据链闭环在 00:06:11 调试台：`depth_input` 是 3 元素 list（多尺度确实传进来了），而 `depth_logits` 的空间尺寸 88×160 与 8× 层完全一致（16×/32× 若参与，输出要么变尺寸要么需上采样融合，画面中 depthnet 调用无任何融合迹象）→ **传了三个，只吃第一个**。

【形状】进 DepthNet：`[(21,128,88,160),(21,256,44,80),(21,256,22,40)]`；实际消费：仅 `[0]`；出：`(21,100,88,160)`。

【为什么】多尺度深度头为什么被砍？深度估计的难点在远距离小目标，理论上 16×/32× 的大感受野有帮助；但实测收益若撑不起额外的融合算力+调参成本，量产工程会毫不留情砍成单尺度——**结构选择是 ROI 决策不是论文美学**。留接口不删是为将来翻案留门。

【连接】BEVDepth 论文的 DepthNet 用了 camera-aware 的 SE 模块和多层卷积栈也仍是单尺度输入；真正多尺度深度融合的是更晚的一些工作。这套代码的演化轨迹（多尺度→单尺度）反着走，说明他们实测多尺度不划算。

---

#### 卡 13 ｜ [00:03:40]

> **原话**：然后这些其实在后续模块都没有用到。

【直译】指着 return 里的若干项（`location`、16×/32× 特征等）总结：后续没人消费。这是本段第三次"宣判死刑"，讲者在用重复强调帮听众建立"这个 return 里只有两样是活的"的心智模型：**`img_feat`（给 LSS/融合）和 `depth_input[0]`（给 DepthNet）**。

【代码】活性标注版 return：

```python
return (img_feats,        # ✅ 活：主特征，去往LSS投影
        locations,        # ❌ 死：PETR位置编码遗留
        depth_loss_total, # ❌ 半死：FPN内嵌深度头未启用，恒0
        depth_probses[0], depth_probses[1],  # ❌ 当前为None占位
        depth_inputs,     # ✅ 活：DepthNet的输入（只吃[0]）
        fpn_feats)        # ❌ 死：多尺度陪跑
```

（此为 `NeckForward` 的 return，00:04:10 帧 275 行原样可见。）

【形状】无新形状；本卡价值在"活/死标注表"。

【为什么】为什么讲者肯花口舌反复讲死代码？因为**新人读这套代码最大的坑就是把死分支当主线去理解**，一旦顺着 `location` 或 flatten 分支追下去会白费几天。导师串讲的核心增值正是"剪枝"。

【连接】你给自己定的 BEV 学习纪律里有"记录哪些分支没用"一条——本章的活/死表可以直接抄进笔记当模板。

---

#### 卡 14 ｜ [00:03:52] + [00:04:01]（合并：总结句+衔接句；重点句，5角度）

> **原话**：然后从 FPN 出来的话，主要是下采样 8 倍的一个特征，以及对应的下采样 8 倍、16 倍和也是 32 倍的一个图像的特征。然后传出去给到后续的 DepthNet。

【直译】FPN 阶段正式收尾。两路输出：①单抽的 8× 特征；②8×/16×/32× 全家桶。两者一起流向 DepthNet。

【代码】`NeckForward` 循环体内的完整交接（00:04:10 帧 261–272 行）：

```python
num_outs = len(img_fpn.in_channels)                              # 3
img_feat = input[(idx * num_outs):(idx * num_outs) + num_outs]   # 本组3尺度
bst = img_feat[0].shape[0] // self._num_views[idx]               # 21//7=3
img_ori_shape = bst, self._num_views[idx], 3, sizes[1], sizes[0]
img_feat, location, depth_loss, depth_probs, depth_input, fpn_feat = \
    img_fpn(img_feat, img_ori_shape, grp, kwargs)
img_feats.append(img_feat);      locations.append(location)
depth_losses.append(depth_loss); depth_probses.append(depth_probs)
depth_inputs.append(depth_input); fpn_feats.append(fpn_feat)
...
depth_loss_total = sum(depth_losses)
```

每个变量都是 **per-group 列表**：`depth_inputs[0]` 是针孔组的（多尺度列表），`depth_inputs[1]` 是鱼眼组的。这个"列表套列表"结构就是 00:06:11 那次 `AttributeError` 的伏笔。

【形状】交接清单（针孔组）：`img_feat (3,7,C,88,160)`（5 维视图，B 与相机分开）；`depth_input [(21,128,88,160),(21,256,44,80),(21,256,22,40)]`（N=B×T×V=1×3×7 拍平）。同一份特征，**主线用 5 维分视角，深度线用 4 维拍平**——两种视图并存，后续模块按需 reshape。

【为什么】为什么 `img_feat` 要 view 成 5 维而 `depth_input` 保持 4 维？深度预测是纯 per-image 任务（每张图独立出一张深度图），4 维直接喂 Conv2d 最高效；而 LSS 投影/多视角融合需要知道"哪 7 张图属于同一时刻"，必须显式分出 view 维。**维度组织方式泄露模块的语义需求**。

【连接】BEVFusion 里同样的手法：`imgs.view(B*N, C, H, W)` 进 backbone，出来再 `view(B, N, ...)` 进 view-transform。`B*N` 拍平跑卷积是多视角感知的通用惯例，因为 Conv2d 只认 4 维。

---

### 🔨 动手练习 ch1-3：复刻 FPN→DepthNet 的交接（两种视图并存）

```python
import torch

B, T, V, C, H, W = 1, 3, 7, 128, 88, 160
N = B * T * V                                     # 21

outs = [torch.randn(N, 128, 88, 160),             # 8×  (过了fpn_conv,128ch)
        torch.randn(N, 256, 44, 80),              # 16× (lateral,256ch)
        torch.randn(N, 256, 22, 40)]              # 32× (lateral,256ch)

depth_input = outs                                # 传全家桶
main_feat   = outs[0]                             # 主线只认8×

num_views = V
bst = main_feat.shape[0] // num_views             # 21//7=3 (=B*T)
img_feat_5d = main_feat.view(bst, num_views, *main_feat.shape[1:])

print("depth_input 是", type(depth_input).__name__, "长度", len(depth_input))
try:
    depth_input.shape                             # 复刻讲者的现场翻车
except AttributeError as e:
    print("AttributeError:", e)
print([tuple(i.shape) for i in depth_input])
print("主线5维视图:", tuple(img_feat_5d.shape))
# 预期输出：
# depth_input 是 list 长度 3
# AttributeError: 'list' object has no attribute 'shape'
# [(21, 128, 88, 160), (21, 256, 44, 80), (21, 256, 22, 40)]
# 主线5维视图: (3, 7, 128, 88, 160)
```

**【小结】** ① FPN 收尾时单抽 `outs[0]`（8×，实测 21×128×88×160）为 `depth_input`，同时把三尺度全家桶陪跑传出；② 历史上 DepthNet 吃过多尺度，现只吃 8×，多尺度接口是遗留冗余；③ 旧注释 `14,256,88,160` 与实测 `21,128,88,160` 双双打脸（帧数 2→3、通道 256→128），**形状问题永远信调试器不信注释**。

---

## Part 4　DepthNet.forward：深度 GT 与 Mask 的分组 concat（00:04:05–00:05:15）

**导读**：镜头切进 `fpn_forward.py` 528 行的 `class DepthNet(BaseModule)`。它的 forward 干三件事：循环针孔/鱼眼两组；把 dataload 阶段按"主相机组/endpart 组"拆开的 depth GT 和 mask **concat 回一份**；再把 8× 特征喂给真正的卷积小网络。本段先处理前两件（GT 侧的整备），输入是 `kwargs['labels']` 里的 GT 字典 + Part3 传来的 `depth_inputs`，输出是对齐好的 `depths / depth_masks` 和 `depth_input`。

---

#### 卡 15 ｜ [00:04:05] + [00:04:10]（合并：跳转衔接句）

> **原话**：然后在 DepthNet 的时候。对，这里 DepthNet 的 FORWARD 的话，在这里传过来的这个就是下采样 8 倍的一个……

【直译】讲者从 `NeckForward` 跳转到 `DepthNet.forward`（`fpn_forward.py` 570 行），指认入参：就是刚才打包的那份 depth_inputs。话说到一半他发现不对，下一句立刻自我纠正（见卡 16）。

【代码】00:05:19 帧，`DepthNet` 的类头与 forward 头：

```python
class DepthNet(BaseModule):            # fpn_forward.py:528
    def __init__(self, cfg): ...       # 528~569，画面中始终折叠 ⚠
    def forward(self, *input, **kwargs):   # :570
        depth_inputs = input[0]        # 各组的多尺度特征列表
        depth_losses = []
        depth_probses = []
        for idx, grp in enumerate(self._fv_grps):
            if not self.convertD:      # 非模型转换(部署导出)模式才算GT
```

注意 `DepthNet` 与 `FPNForward` 同文件——deep 头被视为 neck 的附属品。另外 `__init__` 里能看到的一角（00:04:31 帧 559–568 行）是 loss 相关装配：

```python
for idx, grp in enumerate(self._fv_grps):
    setattr(self, f'{grp}ddn_loss', DDNLoss(cfg, grp=grp))
    if cfg.depth_loss.enabled or cfg.ddn.prj_gt:
        self.ddn_supervision = not cfg.ddn.prj_gt
        self.ddn_weight = cfg.depth_loss.weight * (1 / len(self._fv_grps))
    else:
        self.ddn_supervision = False
        self.ddn_weight = 0.
```

`DDNLoss`（Depth Distribution Network Loss，名字直承 CaDDN 论文）按组各建一个；**深度 loss 权重被除以组数**（两组各拿 1/2），保证针孔+鱼眼加起来的深度监督总强度与单组时一致。

【形状】`input[0] = [针孔的多尺度列表, 鱼眼的多尺度列表]`。

【为什么】`if not self.convertD:` 把整段 GT 处理围起来——导出 ONNX/部署模型时没有 label，这段必须能整体跳过。部署友好性从训练代码阶段就要设计进去，这是量产代码与学术代码的分水岭。

【连接】`DDNLoss` 的血统：CaDDN（CVPR2021）首创把深度离散成分布并用投影 GT 监督，BEVDepth 将其发扬光大；Ch2 讲的 Depth Loss 就是这个 `ddn_loss` 的内部。

---

#### 卡 16 ｜ [00:04:24] + [00:04:27]（合并：现场自我纠正）

> **原话**：这个应该是多尺度的一个特征。分别是下采样 8 倍、然后下采样 16 倍以及 32 倍的一个特征。

【直译】纠正上一句：传进来的不是"8 倍特征"这一个张量，而是**8×/16×/32× 三个尺度的列表**。这句口头修正与 00:06:11 的调试输出（3 元素 list）严丝合缝。

【代码】对应变量 `depth_input = depth_inputs[idx]`（每组一份，本身是 3 元素 list）。讲者的认知路径值得学：**先按印象说 → 觉得不对 → 用调试器验证 →修正表述**，全片他遇到含糊处一律现场断点验证。

【形状】`depth_inputs[0]（针孔组） = [(21,128,88,160), (21,256,44,80), (21,256,22,40)]`。

【为什么】接口传多尺度、实际只用 8× 的原因已在卡 12 展开；这里的增量信息是**组维度和尺度维度是两层嵌套**：外层 list 按组（针孔/鱼眼），内层 list 按尺度。搞混这两层是新人调这段代码的高发事故。

【连接】nuScenes 链路里你熟悉的 `mlvl_feats`（multi-level features）就是内层这个东西；外层的"组"是这套量产代码特有的，读任何函数签名先问"这是 per-group 还是 per-level 的列表"。

---

#### 卡 17 ｜ [00:04:31]（重点句，5角度）

> **原话**：然后在具体的 DepthNet 里面的时候，其实只缺了（⚠应为"只取了"）下采样 8 倍的一个特征。

【直译】虽然三尺度都传了进来，真正做深度预测的小网络只消费 8× 那个。转写的"只缺了"是同音误写，按上下文和调试证据应为"只**取**了"。

【代码】消费点在 00:05:33 帧 601–603 行：

```python
depth_input = depth_inputs[idx]   # 14,256,88,160  <- 又一处过时注释
depthnet = getattr(self, f'{grp}depthnet')
depth_logits = depthnet(depth_input)
```

`depthnet(depth_input)` 直接把 list 传了进去，说明**取 [0] 的动作发生在 depthnet 模块内部**（其 forward 大概率第一行就是 `x = x[0]` 之类，画面未展示，标 ⚠ 推断）。又见同款过时注释 `14,256,88,160`。

【形状】名义输入：3 元素 list；有效输入：`(21,128,88,160)`；输出：`(21,100,88,160)`——空间尺寸原封不动，只有通道从 128 换成 100，这是"只用 8×、无跨尺度融合"的形状铁证。

【为什么】深度头为什么能这么薄（见下卡 23）还工作？因为它不是从零估深度：输入的 128 维特征已经被整个 backbone+FPN 加工过，深度线索（纹理梯度、物体大小、地面接触点位置）已隐含其中，深度头只需做"特征→分布"的线性读出；深度质量的重担实际由 Ch2 的投影 GT 监督来扛。

【连接】BEVFusion（mit 版）的 `dtransform`/depth 分支同样是几层卷积的轻量头；对照 BEVDepth 论文的 camera-aware DepthNet（含相机内参 SE 调制、残差块），这套代码选择了更工程化的极简版——**深度头的复杂度是各家 LSS 系方案的主要分歧点之一**，面试可以拿这个对比展开。

---

#### 卡 18 ｜ [00:04:39]

> **原话**：然后在这里也是去分别去循环我们针孔和鱼眼。

【直译】DepthNet.forward 里又出现同款组循环：`for idx, grp in enumerate(self._fv_grps)`，针孔一轮、鱼眼一轮，各算各的深度和 loss。

【代码】00:05:19 帧 574 行。循环体内所有资源都按 `grp` 前缀取：`getattr(self, f'{grp}depthnet')`、`getattr(self, f'{grp}ddn_loss')`、`label_obj_ori.get(f'{grp}depths')`——**同一套代码，两组数据，参数不共享**。鱼眼组还有专属特判（00:05:33 帧 597/606 行）：`if grp == 'fisheye_' and self.use_fisheye_depth_proj_mask ...` 用 `fisheye_mask` 把鱼眼图像无效区（圆形视场外的黑角）的 GT 置 0，以及 `depth_logits.view(-1, 4, ...)` 里那个 **4 = 4 路鱼眼相机**的 reshape。

【形状】针孔轮：N=21（3帧×7路）；鱼眼轮：N=3帧×4路=12（本段画面未直接打印，由 `view(-1, 4, ...)` 与车辆配置推得，标 ⚠）。

【为什么】针孔/鱼眼深度头不共享权重，原因同 FPN 不共享（成像模型不同，深度-像素关系完全不同：鱼眼的等距投影下同一物理深度对应的像素尺度随视场角剧烈变化）；mask 特判则因为鱼眼成像圈外是纯黑填充，投影 GT 落在那里是脏数据，不 mask 会教坏网络。

【连接】nuScenes 没有鱼眼，所以 BEVFusion/BEVDepth 都没这套逻辑；量产近距感知（泊车、加塞检测）离不开鱼眼，这段是你从"学术 BEV"跨到"量产 BEV"要补的独门课。

---

#### 卡 19 ｜ [00:04:46]

> **原话**：然后这里的话会去取我们 Depth 的 GT 以及我们 Depth 的一个 Mask。

【直译】从 label 字典里取两样监督材料：`depths`（每个像素的真值深度，来自 lidar 点云投影到图像）和 `depth_masks`（哪些像素有有效真值的 0/1 掩码——lidar 点稀疏，大部分像素投不到点）。

【代码】00:05:19 帧 576–578 行：

```python
label_obj_ori = kwargs['labels'][0]
depths = label_obj_ori.get(f'{grp}depths')
depth_masks = label_obj_ori.get(f'{grp}depth_masks')
```

用 `.get()` 而非 `[]`：某组没有深度 GT 时返回 None 而不崩，后面配套 `depths is not None` 判断。另一个细节（00:02:45 帧 383–386 行，FPN 内嵌分支里的同款代码）：取出后立刻 `depths.clone()`——因为后面要做切片赋值/拼接等原地操作，clone 防止污染原始 label 字典（同一 label 可能被多个头消费）。

【形状】`depths ≈ (B, N_cam, H', W')`、`depth_masks` 同形状（H'×W' 与 8× 特征同为 88×160——GT 在 dataload 时就已经按特征分辨率栅格化，见 Ch2 的 `(B, n, H', W')` 注释）。

【为什么】深度监督为什么需要 mask？32/64 线 lidar 投到 88×160 的栅格上，有效像素通常只占百分之几；没有 mask 的话空像素会被当成"深度=0"的错误监督。mask 机制 + Ch2 里的 loss 归一化（只对有效像素平均）是投影式深度监督的标配组合。

【连接】BEVDepth 生成深度 GT 的 `get_downsampled_gt_depth` 做的就是同一件事（点云投影→按 stride 下采样取最近深度→生成 valid mask）；你在 4060 上跑 BEVFusion 时 `depth_loss` 相关 tensor 里的 NaN 防护也是同源问题。

---

#### 卡 20 ｜ [00:04:53] + [00:05:05]（合并：一句话+它的收尾半句）

> **原话**：然后在这里也是因为我们在处理 DepthNet 的 dataload 的时候，把 Depth 和 Depth Mask 也把它拆分成 1、2 号相机（⚠转写"12号相机"）和侧向和后相机，然后把它分成了两组。

【直译】数据加载阶段，7 路针孔相机的深度 GT 不是一整块，而是被拆成两组存放：主组（前视 1、2 号相机，label 键 `depths`）和 **endpart 组**（侧向+后相机，label 键 `endpart_depths`）。为什么拆？因为前视相机与侧后相机的输入分辨率不同（`__init__` 里的 `endpart_input_size` 就是给后者的），dataload 按分辨率分桶处理。

【代码】00:04:55 帧（讲者鼠标高亮了这几行）579–584 行：

```python
if self.merge_img and grp == '' and depths is not None:
    grp_endpart = 'endpart_'
    depths_endpart = label_obj_ori.get(f'{grp_endpart}depths')
    depth_masks_endpart = label_obj_ori.get(f'{grp_endpart}depth_masks')
```

只对针孔组（`grp == ''`）做，鱼眼组没有 endpart 拆分。开关叫 `merge_img`——"把（dataload 拆开的）图重新合并"。⚠ "1、2 号相机"为我对转写"12号相机"的推断：结合 `NeckForward.forward` 里 `zip(input[:3], input[3:6])` 的 startpart/endpart 特征拼接（00:00:38 帧）和 `endpart_input_size` 的存在，主组/endpart 组的划分确凿，但主组到底含几路相机（2 路还是 3 路）画面未给出实数。

【形状】设主组 k 路、endpart 组 7−k 路：`depths (B*T, k, H', W')` 与 `depths_endpart (B*T, 7−k, H', W')`。

【为什么】不同朝向相机用不同输入分辨率，是量产算力分配的常见操作：前视要看 150m 外的小目标，分辨率给高；侧后主要管近距离，分辨率可以低。代价就是数据管道全程要维护"两桶"，直到某个需要统一视角维的模块前再合并。

【连接】nuScenes 六路相机分辨率统一（1600×900），所以你在 BEVFusion 里从没见过这种分桶；这是学术数据集与真车传感器套件的又一差异点。

---

#### 卡 21 ｜ [00:05:07] + [00:05:12] + [00:05:13] + [00:05:15]（合并：讲者车轱辘四连——"需要把它 concat 起来 / 对，concat / 把 Depth GT concat 起来 / 然后 Depth 的 Mask 也 concat 起来"，实为一个操作）

> **原话**（顺滑合并）：在这里需要把它们 concat 起来——把 Depth GT concat 起来，Depth 的 Mask 也 concat 起来。

【直译】把两桶 GT 沿**相机维**拼回一整块，让 GT 的相机排布和特征张量的 21 路排布重新对齐。

【代码】00:04:55 帧 583–584 行：

```python
depths = torch.cat([depths, depths_endpart], dim=1)
depth_masks = torch.cat([depth_masks, depth_masks_endpart], dim=1)
```

`dim=1` 是相机维（dim0 是 B*T）。**拼接顺序 = 主组在前、endpart 在后**，必须与特征侧 `NeckForward` 里 `concat_feat = torch.cat([startpart_feat, endtpart_feat], dim=1)`（00:00:38 帧）的顺序一致，否则第 i 路相机的特征会对上第 j 路的 GT——这类"顺序耦合"没有任何类型检查兜底，纯靠约定，是这套代码最脆的地方之一。

【形状】`(3, k, 88, 160) ⊕ (3, 7−k, 88, 160) → (3, 7, 88, 160)`；随后（Ch2 开头,00:02:45 帧 395–397 行已预演）`depths.view(-1, dh_gt, dw_gt) → (21, 88, 160)`，与 `depth_logits (21,100,88,160)` 逐像素对齐。

【为什么】为什么在模型里合而不是 dataload 里就存成一块？因为 dataload 的分桶服务于"不同分辨率的图像增广/深度栅格化"，那时合不了；而 loss 计算需要与网络输出（21 路统一排布）对齐，此处是第一个"两边都齐了"的时机。**数据形态的每次转换都发生在需求出现的最晚时刻**——工程上这叫惰性对齐，减少无效搬运。

【连接】等价于你在 BEVFusion dataset 里看到的 `torch.stack(imgs)` 把 6 路相机堆成一维的时刻；只是这里因为分桶多了一步 cat。练习见下。

---

### 🔨 动手练习 ch1-4：GT 分桶与惰性合并（含顺序错位事故演示）

```python
import torch

BT, H, W = 3, 88, 160                 # B*T=3
k = 2                                 # 假设主组(前视)2路, endpart 5路
depths_main = torch.full((BT, k,   H, W), 10.0)   # 主组GT深度全10m
depths_end  = torch.full((BT, 7-k, H, W), 50.0)   # 侧后组全50m
mask_main   = torch.ones_like(depths_main)
mask_end    = torch.ones_like(depths_end)

depths = torch.cat([depths_main, depths_end], dim=1)       # 正确顺序
masks  = torch.cat([mask_main, mask_end], dim=1)
print(depths.shape, depths[0, 0, 0, 0].item(), depths[0, -1, 0, 0].item())
# 预期 torch.Size([3, 7, 88, 160]) 10.0 50.0

flat = depths.view(-1, H, W)                                # 与21路logits对齐
print(flat.shape)                     # 预期 torch.Size([21, 88, 160])

# 事故演示：拼接顺序反了,GT整体错位到别的相机上,不报任何错!
wrong = torch.cat([depths_end, depths_main], dim=1).view(-1, H, W)
print("第0路GT本应10m, 错序后:", wrong[0, 0, 0].item())   # 预期 50.0 —— 静默污染
```

**【小结】** ① `DepthNet`（`fpn_forward.py:528`）forward 先循环针孔/鱼眼两组，所有子模块与 GT 键都按 `grp` 前缀动态取用，两组参数互不共享；② dataload 把针孔深度 GT 按前视/侧后分辨率拆成主组与 `endpart_` 两桶，模型内用 `torch.cat(dim=1)` 沿相机维惰性合并，顺序必须与特征侧拼接一致；③ 深度监督 = 稀疏投影 GT + 有效 mask，鱼眼组另有视场 mask 特判，部署模式（convertD）下整段 GT 逻辑可跳过。

---

## Part 5　DepthNet 本体：两个卷积 + 21×100×88×160（00:05:18–00:06:42）

**导读**：终于到网络本体。本段输入是 8× 特征 `(21,128,88,160)`，经过一个**只有两个卷积**的小网络，输出深度分布 logits `(21,100,88,160)`。讲者在断点上现场打印形状，把 21（3 帧×7 路）和 100（深度 bin 数）两个数字掰开讲清。这是全章的收束，也是通往 Ch2（Depth Loss）和 LSS 投影的接口定义。

---

#### 卡 22 ｜ [00:05:18] + [00:05:25]（合并：一句话两段）

> **原话**：然后在这里的话会用我们就是 FPN 出来的图像的特征，然后经过 DepthNet，然后预测我们的一个 Depth。

【直译】流程总述：FPN 特征 → depthnet 模块 → 深度预测。对应的正是三行核心代码。

【代码】00:05:33/00:05:52 帧 601–604 行（调试器黄条恰好停在这附近，行号旁有断点标记）：

```python
depth_input = depth_inputs[idx]                # 本组多尺度列表
depthnet = getattr(self, f'{grp}depthnet')     # 本组的深度小网络
depth_logits = depthnet(depth_input)           # -> (21,100,88,160)
ddn_loss = getattr(self, f'{grp}ddn_loss')     # 下一章的主角在此登场
```

变量名值得咬文嚼字：输出叫 `depth_logits` 而不是 `depth`——它是**未归一化的分布打分**，不是深度值本身。真正变成概率要等 softmax（在 loss 内部/LSS 投影前做），真正变成"深度图"则永远不会发生——LSS 用的就是整个分布。

【形状】`(21,128,88,160) → (21,100,88,160)`。

【为什么】为什么预测"分布"而不是回归一个深度值？①单目深度本质多解，分布能表达不确定性（一个像素可以"40% 在 10m、30% 在 12m"）；②LSS 的外积操作天然需要每个 bin 一个权重；③分类式训练比 L1 回归对离群 lidar 点更鲁棒。这是 LSS/CaDDN 路线区别于单目深度估计（回归稠密深度图）的根本设计。

【连接】LSS 论文式子 `c_d = α_d · c`（深度分布 α 与图像特征 c 做外积）里的 α 就从这个 `depth_logits` softmax 而来；本片 LSS 投影章会回收这个输出。你复现时可先在此处打印 `depth_logits.softmax(1).sum(1)` 验证每像素分布归一化为 1。

---

#### 卡 23 ｜ [00:05:29] + [00:05:34]（合并；重点句，5角度）

> **原话**：然后 DepthNet 主要是其实网络结构比较简单，就只是两个卷积。

【直译】深度小网络（`{grp}depthnet` 属性指向的模块）结构极简：**两层卷积，没了**。没有注意力、没有残差栈、没有相机参数调制。

【代码】⚠ 两个卷积的具体定义画面从未展示（`DepthNet.__init__` 528–569 行在所有帧里都处于折叠状态，只露出 ddn_loss 装配段）。按"两个卷积 + 输入128通道 + 输出100通道 + 空间尺寸不变"四个硬约束，最合理的复原是：

```python
depthnet = nn.Sequential(
    nn.Conv2d(128, mid, 3, padding=1),   # 卷积1：3×3 特征变换(可能带BN+ReLU)
    nn.ReLU(inplace=True),
    nn.Conv2d(mid, 100, 1),              # 卷积2：1×1 读出100个深度bin的logits
)
```

（mid 常见取 128 或 256；也可能两层都是 3×3。结构细节标 ⚠，"两个卷积、通道到 100、stride=1"三点由口述+形状锁死。）

【形状】参数量粗算（按 mid=128）：3×3 卷积 128→128 约 14.7 万参数，1×1 卷积 128→100 约 1.3 万，合计 **~16 万参数**——对比 backbone 动辄千万级，深度头几乎免费；计算量约 21×88×160×(128×128×9+128×100) ≈ 39 GFLOPs·(累加计一次)，在整网里同样是零头。

【为什么】敢用这么薄的头，是三个条件共同成立：①上游 FPN 特征足够肥（有全局上下文）；②下游有强监督（lidar 投影 GT 逐像素教）；③ LSS 对深度分布的误差有一定容忍（BEV pooling 会做空间聚合平均掉部分噪声）。反过来说，如果你发现深度分支欠拟合（深度 loss 降不动），第一刀就应该砍向这里——加深头、加相机参数调制（BEVDepth 的做法）。

【连接】和智谷 YOLO 课挂钩：YOLO 检测头也是"backbone 肥、头薄"的哲学，head 用几层卷积把特征读出成 `(num_anchors×(5+类别数))` 通道——**"通道数=预测语义"的读出式设计**在 2D 检测和深度分布头上一模一样，见下一卡。

---

#### 卡 24 ｜ [00:05:36] + [00:05:42]（合并："把它的 Channel 变成……深度"；重点句，5角度）

> **原话**：然后把它的 Channel 变成我们所预测的一个 Depth 的……深度（bin 数）。

【直译】第二个卷积的唯一使命：把通道数从特征维（128）改写成深度 bin 数（100）。此后张量的通道维不再是"特征语义"，而是"**每个候选深度档位的打分**"。

【代码】等价一行：`nn.Conv2d(128, D, 1)`，D=100。读出后每个像素 `(u,v)` 得到 100 维向量 `logits[:, :, u, v]`，`softmax` 后即该像素的深度概率质量函数（pmf）。

【形状】通道维语义转变全景：`64/128/256/512（backbone 特征）→ 128（FPN 统一）→ 100（深度 bin 打分）`。空间 88×160 全程不动。

【为什么】100 个 bin 覆盖什么范围？本段只说了"100 就是我们所预测的一个深度范围"（卡 28），离散化细节（起止距离、均匀 UD 还是 LID/SID 间隔）画面未给 ⚠——按整车 BEV 前向 95.4m 的配置推断，若均匀划分则约 1m/bin 上下。bin 数是精度/算力/显存的三方博弈：LSS 原文 D=41（4~45m）、BEVDepth 用 112（2~58m），这里 100 与前向 95m 的量产需求匹配。bin 太少→远处深度分辨率差、BEV 上目标纵向糊；bin 太多→视锥点数线性涨（点数=N·D·H·W，D=100 时已 2960 万）。

【连接】与 YOLO 头的"通道=预测"对照记忆：YOLO 是 `C→3×(5+80)`（anchor×框+类别），这里是 `C→100`（深度 bin）；甚至 YOLOv8 的 DFL 把框回归也变成"离散 bin 分布+期望"，与深度分布离散化是同一思想的两个应用场——**回归问题分类化**。这个梗概你可以直接写进转岗面试的自我总结。

---

#### 卡 25 ｜ [00:05:45] + [00:05:49] + [00:06:05 前的静默]（合并：两句都是"看一下输入"的操作性过场，其间讲者在调试台敲命令，画面 00:05:52/00:06:05 帧可见黄色断点条与控制台）

> **原话**：可以看一下当前对于我们……这个模块的输入的话就是我们……（现场在调试控制台操作）

【直译】讲者停止讲解、切到断点调试台，实际敲了三条命令（完整输出见 0.3 节）：先 `depth_input.shape` 吃了个 `AttributeError`，再用列表推导打出三尺度形状，最后打 `depth_logits.shape`。

【代码】这次"翻车"本身信息量最大：`AttributeError: 'list' object has no attribute 'shape'` 直接证明 `depth_input` 是 list——比任何文档都硬的证据。随后的 `[i.shape for i in depth_input]` 是排查未知容器的标准动作。

【形状】（见 0.3 引文）`[(21,128,88,160), (21,256,44,80), (21,256,22,40)]`。

【为什么】为什么值得把导师的调试手法单独立卡？因为这是本视频隐藏的第二条教学线：**形状不确定就断点打印，容器类型不确定就先试 `.shape` 再退列表推导**。你之后独立扒 DenseBEV 其他模块时，复用这套动作比记住任何结论都重要。

【连接】与你的 BEV 记录纪律呼应：把"每个模块进出形状"记进笔记的最快方法，就是像这样在每个模块 return 前挂断点抄形状——一次训练 step 就能抄完全网。

---

#### 卡 26 ｜ [00:06:11] + [00:06:16] + [00:06:20]（合并：三句连贯解释同一个数字；重点句，5角度）

> **原话**：我们这里的 21 其实是我们因为正式（版本）它是需要三帧的，所以说它是历史三帧以及七路针孔相机 concat 到一起的，所以说是 21。

【直译】解密 batch 维的 21：当前训练配置带**时序 3 帧**（当前帧+历史 2 帧），每帧 7 路针孔相机，3×7=21 张图在 batch 维拍平一起过网络（batch size=1）。

【代码】21 的生成链条（结合 `NeckForward` 里的换算）：

```python
N = B * T * V = 1 * 3 * 7 = 21          # 拍平进backbone/FPN/DepthNet
bst = N // num_views = 21 // 7 = 3      # B*T，恢复"图组"维
# FPNForward 里的自洽校验：
assert bst * num_view == bstn            # 3 * 7 == 21 ✓
```

【形状】`(21, ·, 88, 160)` 中的 21 会一路贯穿到 LSS 投影（每张图各自抬升视锥），到时序融合章才把 T=3 拆出来 warp 对齐。**深度是对三帧全部预测的**——历史帧的深度供其视锥投影用，不是只算当前帧。

【为什么】为什么历史帧也要过深度网络而不是缓存上次的结果？本片后面有 MemoryManager（10Hz 缓存）章节处理特征级缓存；在训练阶段三帧联合前传能让深度网络吃到时序一致性的梯度（同一物体在三帧中的深度应连续变化）。而 14→21 的旧注释差异（卡 10）说明帧数是从 2 帧演进到 3 帧的——时序帧数本身就是他们迭代过的超参。

【连接】BEVFusion（你复现的版本）是单帧方案，没有 T 维；BEVDet4D/StreamPETR 系是 2~8 帧。看到任何 BEV 代码里 batch 维出现"不像 batch size 的数"，第一反应就该是 `B×T×V` 分解——这是读多视角时序代码的通用钥匙。

---

#### 卡 27 ｜ [00:06:22] + [00:06:26]（合并：宣布输出形状；重点句，5角度）

> **原话**：然后它预测出来的一个 Depth 的 shape，Depth 的 shape 就是 21 乘以 100 乘 88 和 160。

【直译】调试台白纸黑字：`depth_logits.shape == torch.Size([21, 100, 88, 160])`。本章标题里的那个数字正式落地。

【代码】四个维度逐位读法：

```python
depth_logits: (21, 100, 88, 160)
#              │    │    │    └ W: 1280/8，特征宽
#              │    │    └ H: 704/8，特征高
#              │    └ D: 100个深度bin的logits(未softmax)
#              └ N: 1批×3帧×7路针孔
```

【形状】内存量 sanity check：21×100×88×160 = 29,568,000 元素 ≈ **113MB**（FP32）——单单一个深度 logits 就上百兆，这解释了为什么 D、H、W 每个数字都要抠着算力预算选（也解释了你 4060 上跑 BEV 模型显存吃紧的日常）。

【为什么】这个形状是三个下游的契约：①Ch2 的 DDNLoss 要求它与 `(21,88,160)` 的 GT 逐像素对齐；②LSS 章拿它 softmax 后与 `(21,128,88,160)` 特征做外积得 `(21,128,100,88,160)` 级别的视锥（随后 bev_pool 拍扁）；③鱼眼组会产出自己的一份（12×D'×H'×W'）。**记接口形状=记住模块间的全部契约**。

【连接】对齐 BEVFusion 源码：`DepthLSSTransform` 里 `x = depth.unsqueeze(1) * feat.unsqueeze(2)` 的两个操作数，就是本卡的 logits(softmax 后) 与卡 10 的 8× 特征——你已经在自己机器上跑过这行代码的 nuScenes 版了。

---

#### 卡 28 ｜ [00:06:30]

> **原话**：100 就是我们所预测的一个深度范围。

【直译】口语压缩说法，展开是：100 是把可预测深度范围离散化之后的**档位（bin）数**，每个通道对应一个候选深度值。

【代码】离散化通式（以均匀划分 UD 为例）：

```python
d_bins = torch.linspace(d_min, d_max, 100)          # 每bin一个代表深度
depth_prob = depth_logits.softmax(dim=1)            # (21,100,88,160)
expected_depth = (depth_prob * d_bins.view(1, -1, 1, 1)).sum(1)  # 可选:期望深度图
```

【形状】`(21,100,88,160)` 沿 dim=1 softmax 后每像素和为 1。

【为什么】⚠ `d_min/d_max` 与划分方式（UD 均匀 / LID 线性增 / SID 对数）本段未讲。CaDDN 论文实验 LID 最优（远处 bin 稀、近处密，符合像素-深度的透视非线性）；本工程用哪种要看 Ch2 的 `ddn_loss.py` 里 GT 深度→bin 索引的转换公式，此处按下不表，Ch2 见分晓。

【连接】把 100 与整车 BEV 范围（前 95.4m）放在一起记：BEV 网格 0.4m×448 格 ≈ 前后 179m，深度 bin 若均匀 1m/bin 只到 ~100m——**图像深度范围小于 BEV 范围**是常态，超出深度范围的 BEV 远区靠 lidar/radar 分支补，这正是多模态融合的分工逻辑。

---

#### 卡 29 ｜ [00:06:35]

> **原话**：然后这个是 DepthNet 过了。

【直译】章节收束宣言：DepthNet 讲完。下一句（[00:06:42]，属 Ch2）无缝转入"计算 Depth 的一个 Loss"。

【代码】本章最终接口（交给 Ch2 与 LSS 章的全部资产）：`depth_logits (21,100,88,160)`、`depths (21,88,160)`、`depth_masks (21,88,160)`、`ddn_loss` 模块句柄、以及主线特征 `img_feat (3,7,128,88,160)`。

【形状】见上。

【为什么】讲者的章节切分（网络结构 → loss）也值得学：**先固定前向契约，再谈优化目标**——你写自己的模块笔记时按同样顺序组织（forward 形状表 + loss 输入表），可以直接复用本章的卡片骨架。

【连接】Ch2 预告：`ddn_loss(depth_logits, depths, depth_masks, fg_masks)`（00:02:45 帧 398 行已剧透签名，`fg_masks=None` 说明还预留了前景加权的口子）——深度 loss 如何做 one-hot/软标签、如何用 mask 归一化，下一章逐句拆。

---

### 🔨 动手练习 ch1-5：五分钟复现整个 DepthNet（含形状全链路验证）

```python
import torch, torch.nn as nn

class TinyDepthNet(nn.Module):
    """按本章证据复原：输入多尺度list只吃[0]，两个卷积，通道128->100"""
    def __init__(self, in_ch=128, mid=128, num_bins=100):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, padding=1),
                                   nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(mid, num_bins, 1)      # 通道=深度bin数
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]                                   # 只取下采样8倍特征
        return self.conv2(self.conv1(x))               # logits，不做softmax

B, T, V = 1, 3, 7
depth_input = [torch.randn(B*T*V, 128, 88, 160),       # 8×
               torch.randn(B*T*V, 256, 44, 80),        # 16× (陪跑)
               torch.randn(B*T*V, 256, 22, 40)]        # 32× (陪跑)

net = TinyDepthNet()
depth_logits = net(depth_input)
print(depth_logits.shape)
# 预期: torch.Size([21, 100, 88, 160])   <- 与00:06:26调试台一字不差

prob = depth_logits.softmax(dim=1)
print(prob.sum(1).allclose(torch.ones(21, 88, 160)))   # 预期: True 每像素分布归一
print(f"参数量: {sum(p.numel() for p in net.parameters())/1e3:.1f}K")
# 预期约 160.6K —— 整个深度头不到0.2M参数
```

**【小结】** ① DepthNet 本体只有两个卷积（3×3 变换 + 1×1 读出），把 8× 特征的 128 通道改写成 100 个深度 bin 的 logits，空间尺寸 88×160 不变；② 输出 `(21,100,88,160)` 的 21=1批×3帧×7路针孔、100=深度 bin 数，这份"分布而非深度值"的输出是 Ch2 loss 与 LSS 外积的共同输入；③ 讲者现场断点三连（含一次 AttributeError）示范了多尺度 list 的排查手法，也坐实了"传三尺度、只吃 8×"的结论。

---

## 全章总结

**一张图记住 Ch1 的数据流**（针孔组，B=1、T=3、V=7）：

```
backbone C2/C3/C4: (21,128,88,160) (21,256,44,80) (21,512,22,40)
        │  lateral 1×1 (统一256) + top-down 上采样相加
        ▼
FPN outs: [ fpn_conv3×3→(21,128,88,160) , (21,256,44,80) , (21,256,22,40) ]
        │            └────────── 只有8×层过输出卷积
        ├── depth_input = 整个列表 ──► DepthNet（只吃[0]）
        │                              conv3×3 ─ conv1×1(→100ch)
        │                              ► depth_logits (21,100,88,160) ─► Ch2 Loss / LSS外积
        │       GT侧: depths/masks 主组⊕endpart组 cat(dim=1) → view(-1,88,160) → (21,88,160)
        └── img_feat.view(3,7,128,88,160) ──► LSS投影/多视角融合（后续章）
死代码陪跑: flatten_mode拍平、prepare_location位置编码、16×/32×特征、location返回值
```

三句话版本：**FPN 按针孔/鱼眼分组收尾，只给 8× 层配输出卷积并单抽它给深度分支；DepthNet 用两个卷积把 128 维特征读出成 100 个深度 bin 的分布 logits，21=3帧×7路；本章一半篇幅在剪 SparseBEV 时代的死代码——活的只有 8× 特征和深度 logits 两条线。**

## 存疑清单（⚠汇总）

1. ⚠ **depthnet 两个卷积的具体结构未上屏**：`DepthNet.__init__`（fpn_forward.py 528–569）全程折叠，仅露出 ddn_loss 装配段。"3×3+1×1、128→100、stride1"为口述+输出形状反推，卷积核尺寸/是否带 BN 需对源码核实。
2. ⚠ **注释与实测双重打架**：代码内两处 `# 14,256,88,160`（FPNForward:375、DepthNet:601）vs 调试实测 `21,128,88,160`。推断 14=2帧×7路的旧配置、256=fpn_conv 压缩前的旧通道；结论已按实测写，但"14 的确切来源"未获画面证实。
3. ⚠ **深度 100 bin 的离散化参数**（d_min/d_max、UD/LID/SID）本章未出现，须待 Ch2 `ddn_loss.py` 确认；"约 1m/bin"为按前向 95.4m 的推算。
4. ⚠ **"12号相机"**：按 `endpart_input_size`、`grp_endpart='endpart_'`、`zip(input[:3], input[3:6])` 证据链推断为"1、2 号前视主相机组 vs 侧/后相机组"，但主组具体含几路（2 或 3）画面未给实数。
5. ⚠ **32× 特征 `(21,256,22,40)` 的来源**：按"只有 i==0 过 fpn_conv"推断为 C4 的 lateral 输出（512→256），非 extra-level 池化；与 FPN 配置的 `num_outs/add_extra_convs` 有关，未 100% 锁死。
6. ⚠ **`prepare_location` 的输出形状与内部实现**未上屏，`(N, H*W, 2)` 为按 StreamPETR 惯例的推断；其"每步空转白算"的判断基于 return 后无消费者，若有隐藏消费者（如 GOD 任务分支）则需修正。
7. ⚠ **鱼眼组 N=12（3帧×4路）**由 `view(-1, 4, ...)` 与车辆配置推得，本章未打印鱼眼张量形状。
8. ⚠ 转写校正确认两处：「Blat model」=`flatten_mode`（帧证坐实）；「只缺了」=「只取了」（语义+调试坐实）。校正稿头部存疑的「拍平用均值还是最大值」：本章帧证明确为 `.mean()` 均值。


---
> [[00_总览与脉络|📖 总览]] · [[Ch02_DepthGT与DepthLoss|Ch2 →]]

