> [[Ch02_DepthGT与DepthLoss|← Ch2]] · [[00_总览与脉络|📖 总览]] · [[Ch04_Radar代码实走与远距离切分|Ch4 →]]

# Ch3 Lidar 透传 与 Radar Pillar 编码(图解)（00:13:04–00:24:49）

> **本章在全视频地图中的位置**：
> `FPN收尾 → DepthNet → Depth Loss → 【本章：Lidar Backbone(透传) → Radar Backbone(pillar编码)】 → RL融合(UNet) → LSS投影 → …`
>
> 上一章（Ch2）讲完了 DepthNet：图像分支拿到了下采样 8 倍的特征并预测出 21×100×88×160 的深度分布。本章视角切换到**另外两个传感器分支**——lidar 和 radar。这两个分支在 DenseBEV 里地位很不对称：lidar 分支在网络里**一行参数都没有**（体素化和拍平全部在 DataLoader 离线做完，网络里只剩一个 permute 透传）；radar 分支则是一个**完整的迷你 PointPillars**：voxelize 补帧索引 → get_paddings_indicator 生成有效点 mask → Linear+Norm+ReLU 升维 → 沿 16 个点取 max → scatter 填充回 BEV 画布。讲者先在 draw.io 上画了一张 `RadarVoxelGenerator` 框图逐块讲解（本章主体），00:25 之后才回到代码（属于下一章开头）。
>
> **本章转写的特殊情况**：讲者在 00:19:43 处明显"回炉重讲"了一遍 radar 框图——00:19:43–00:23:18 与 00:15:27–00:19:08 内容几乎逐句重复（连"我先因为感觉radar这一块会比较复杂，我就先按照框架先讲一下"这句开场白都原样说了第二遍）。推测是讲者第一遍讲到一半去翻代码卡住了（00:19:11–00:19:37 有一段"我看一下……好吧"的停顿），于是索性从头再顺一遍。本章把两遍**合并成一组逐句卡**，每张卡都注明合并了哪些时间戳，两遍表述有差异处单独指出。

---

## 帧证总说明（本章精读的 10 张关键单帧）

| 帧 | 画面内容 | 提取到的关键信息 |
|---|---|---|
| `00_14_00.jpg` | VSCode：`lidar_backbone/voxel_generator.py`，`class VoxelGenerator(BaseModule)`，断点停在 178 行 | `if len(inputs) == 1:` 高亮；注释原文 `#* 在dataloader里做好了lidar voxlize,直接输出`；`output = inputs[0].permute(0, 3, 1, 2).to(torch.float32).contiguous()`；调试台 `inputs[0].shape → torch.Size([3, 352, 224, 64])` |
| `00_13_49.jpg` | 同上文件稍早时刻 | 调试台残留上一章内容：`[i.shape for i in depth_input]` → `[21,128,88,160],[21,256,44,80],[21,256,22,40]`，`depth_logits.shape → [21,100,88,160]`；下方还有 `rl_feature_crop_area`、`use_grid_mask`、`use_mlp_scatter_feat` 分支代码 |
| `00_14_41.jpg` | VSCode：`radar_backbone/voxel_generator.py`，`class RadarVoxelGenerator(BaseModule)`，断点停在 42 行 | `voxelized = None` 高亮；`if len(inputs)==2: pts, pts_index = inputs`；`radar_feature = self.radar_encoder(pts, pts_index, voxelized)`；还有 `self.convertD`、`self.voxel_pad`、`use_multiframe_rl`、`rear_far_crop_radar_feature` 等分支 |
| `00_15_57.jpg` | draw.io 全景：`RadarVoxelGenerator` 框图 + 左侧 4 个绿色输入块 | 绿块文字：`input[5] N*16*10 每一个voxel，16个点，每一个点10维特征`；`input[6] N*3 每一个voxel的中心点在三维空间中的位置`；`input[7] N 每一个voxel内radar点的个数`；`input[9] (bs*3) 每一帧的voxel的个数，和=N` |
| `00_16_55.jpg` | 同框图选中态 | `self.voxelize` 块注释：`对coors的第二维索引0的位置pad当前帧的索引`；输出三口：`voxels N*16*10`、`num_points N`、`coors_batch N*4` |
| `00_17_36.jpg` | `self.pts_voxel_encoder` 放大 | `get_paddings_indicator 计算每一个voexl内有效点的mask` → `mask N*16*1`；`voxels N*16*10` 与 mask 汇合处标 `*`（逐元素乘）→ `feature N*16*10` → 进 `self.pfn_layers` |
| `00_18_52.jpg` | `self.pfn_layers` 放大 | `linear → norm → relu`，出口箭头标 `N*16*64`，接 `totch.max(x,dim=1)`（图上把 torch 拼成了 totch）→ `feature N*64` |
| `00_19_45.jpg` | draw.io 缩小全景，radar 图上方露出 lidar 图 | lidar 分支：绿块 `input[4] (bs*3)*352*224*64` → `permute` → 蓝块 `output (bs*3)*64*352*224`——**lidar 透传的框图实锤** |
| `00_24_05.jpg` / `00_24_43.jpg` | `self.pts_middle_encoder` 放大 | 输入 `batch_size bs*3` 与 `feature N*64`；块内注释 `根据coors，把特征填充到初始的bev feature上`；出口边标 `(bs*3)*64*448*224`，后接一个黄色小块 `crop` |

下面所有卡片中的【帧证】都引用这张表。

---

## Part 1｜Lidar Backbone：一个没有参数的"假"骨干网（00:13:04–00:14:32）

**导读**：本段输入是 DataLoader 直接给出的 lidar BEV 特征张量 `[bs*3, 352, 224, 64]`（channel-last），输出是 permute 成 PyTorch 卷积惯例 channel-first 的 `[bs*3, 64, 352, 224]`。在流水线里它与图像分支、radar 分支并列，是三条模态支路中最短的一条——短到整个 "backbone" 里没有一个可学习参数。这个设计的本质是：**把 lidar 的体素化+特征提取全部搬进了 CPU 侧的 DataLoader 离线流程**，GPU 网络里只保留内存布局调整。理解这一点，就理解了这套工程代码"训练侧算力往数据管线挪"的取舍风格。

---

### 卡 3-1 ｜从 DepthNet 转场到 Lidar Backbone

> **原话** `[00:13:04]` "DepthNet之后，然后就是我们的一个lidar的一个Backbone。" `[00:13:20]` "然后lidar的Backbone。"（两句为转场+翻代码停顿，合并为一卡）

- 【直译】图像分支的深度预测讲完了，现在切到第二个传感器分支：lidar 的骨干网络。讲者花了十几秒在 VSCode 里把文件切到 `lidar_backbone/voxel_generator.py`。
- 【代码】对应工程目录 `e2e > tasks > bev_task > uvp_module > models > lidar_backbone > voxel_generator.py`（帧证 `00_14_00.jpg` 顶部面包屑导航原样可见），类名是 `VoxelGenerator(BaseModule)`。注意：**radar 分支的文件也叫 `voxel_generator.py`**，只是在 `radar_backbone` 目录下、类名为 `RadarVoxelGenerator`——看帧时靠面包屑路径区分两个同名文件。
- 【连接】在 BEVFusion（用户跑过的 mmdet3d 版）里，对应位置是 `pts_voxel_layer + pts_voxel_encoder + pts_middle_encoder + pts_backbone` 这一整串；DenseBEV 这里把前三步全塞进了 DataLoader，"backbone" 只剩个壳。这也是为什么讲者只用了 80 秒就讲完了 lidar 分支。
- 【为什么】模块虽空也要保留：一是保持"每个模态一个 backbone"的接口对称性，方便配置文件按统一 schema 组装网络；二是给未来在 GPU 侧做 lidar 特征提取（比如换成稀疏卷积）留位置。

---

### 卡 3-2 ⭐重点 ｜Lidar 特征在 DataLoader 里就已拍平成 BEV

> **原话** `[00:13:32]` "然后我们直接从DataLoader输入的输出过来的," `[00:13:37]` "其实就已经是把lidar已经拍平到BV上的一个特征了。"

- 【直译】进入网络的 lidar 数据**不是原始点云**，而是 DataLoader 已经处理好的一张"BEV 特征图"——点云→体素化→沿高度压扁（拍平）这些活儿全在数据加载阶段干完了。
- 【代码】帧证 `00_14_00.jpg` 里 178 行断点上方的注释就是原文实锤：`#* 在dataloader里做好了lidar voxlize,直接输出`（代码注释里 voxelize 拼成了 voxlize）。forward 的判断逻辑是 `if len(inputs) == 1:` ——只传进来一个张量，说明是"已拍平特征"模式，直接 permute 输出；否则走 `pts, pts_index = inputs` 的原始点云路径（`use_mlp_scatter_feat` 分支里还能看到 `self.lidar_encoder(pts, pts_index, frame_ids=["0"], positions_batch=[[[0.0, 0.0, 0.0]]])`，说明代码保留了在线编码的能力，只是当前配置不走）。
- 【形状】进网络时是 `[bs*3, 352, 224, 64]`（H、W 在前，特征 64 维在最后，channel-last），这是 DataLoader 用 numpy 组装的自然布局。
- 【为什么】lidar 一帧几万到十几万个点，体素化是典型的"逐点散乱访存"操作，放 GPU 上要么写 CUDA 算子要么用 scatter，都麻烦；放 DataLoader 里用 CPU 多进程 worker 做，可以和 GPU 训练流水线重叠，几乎白赚。代价是**拍平方式被固定死在数据侧**（网络学不了"怎么拍平"），这正是 DenseBEV 与"把 VFE 放网络里端到端学"的 PointPillars/BEVFusion 路线的差别。
- 【连接】用户在 4060 上跑的 BEVFusion 中，`voxelize` 是 forward 里的第一步（`@force_fp32` 修饰的那个 `voxelize()`），每个 iter 都在 GPU/CPU 上现算；DenseBEV 相当于把这一步挪到了 `Dataset.__getitem__`。类比智谷课程里"预处理 vs 网络层"的边界讨论：任何不需要梯度的确定性变换，理论上都可以往数据管线里搬。

---

### 卡 3-3 ｜Lidar BEV 特征的 shape：⚠转写数字有误，帧证还原为 [3, 352, 224, 64]

> **原话** `[00:13:42]` "它也是它的尺度就是6乘以6乘352乘20乘64。" `[00:13:55]` "64就是我们lidar的一个特征的一个维度。"

- 【直译】讲者报了 lidar 输入张量的形状，并解释最后的 64 是特征维（channel 数）。
- ⚠【转写勘误】"6乘以6乘352乘20乘64"是 Whisper 误听，一个张量不可能有五个维度还带个"20"。帧证 `00_14_00.jpg` 调试台里讲者现场敲了 `inputs[0].shape`，输出 **`torch.Size([3, 352, 224, 64])`**；帧证 `00_19_45.jpg` 的 draw.io lidar 框图上也写着 `input[4] (bs*3)*352*224*64`。所以原话应为"3 乘以 352 乘 224 乘 64"——转写把"3、352、224、64"听劈了。此处以帧证为准。
- 【形状】`[3, 352, 224, 64]`：第 0 维 3 = **bs×3**（batch_size=1 调试 × 3 帧时序，与上一章图像分支 21=3帧×7相机 的折叠逻辑同源）；352 = 前后向格子数；224 = 左右向格子数；64 = 每个 BEV 格子的特征维度。
- 【为什么】把"帧"折进 batch 维是这套代码的统一约定：单帧处理阶段（backbone/编码）对每帧完全独立，folded-batch 可以让所有 Conv/BN 天然并行处理 3 帧，到时序融合模块再 reshape 回 `[bs, 3, ...]`。radar 分支出口 `(bs*3)*64*448*224`、图像分支的 21 路都是同一逻辑。
- 【连接】64 维特征具体怎么来的视频没讲（DataLoader 不在串讲范围）。⚠推断：大概率是沿高度方向切 bin 的占据/强度统计（类似 PointPillars 之前的"手工 BEV 特征"或 BEVFusion 中 sparse conv 输出后 `flatten(高度维×通道)` 的做法——BEVFusion 是 128 通道 = 高度2×64 拍平；这里 64 维与之量级一致）。待回代码核实。

---

### 卡 3-4 ｜Lidar BEV 的物理范围：前 95.4 / 后 45.4 / 左右 ±44.8

> **原话** `[00:14:00]` "352和224对应的是我们前向95.4到后向45.4," `[00:14:08]` "然后左右是4.8的一个BV的一个范围。"

- 【直译】BEV 网格的 352×224 不是随便定的，对应真实物理范围：车前 95.4 米、车后 45.4 米，左右各 44.8 米。
- ⚠【转写勘误】"左右是4.8"缺了个"4"，应为 **44.8 米**（全局配置"左右±44.8m"，且只有 44.8 才能对上 224 这个格子数，见下）。
- 【形状】用 0.4 m 分辨率验算，三组数字严丝合缝：
  - 前后：(95.4 + 45.4) m = 140.8 m ÷ 0.4 m = **352** ✓
  - 左右：44.8 × 2 = 89.6 m ÷ 0.4 m = **224** ✓
  - （预告：radar 的 448 = (95.4 + 83.8) ÷ 0.4 = 179.2 ÷ 0.4 = **448** ✓，见卡 3-30）
- 【为什么】前向 95.4 远大于后向 45.4：前向是主要行驶方向，需要早看见远处慢车/静止物；后向 45 米对 lidar 够用（后方来车主要靠 radar 补远距，这正是本章末尾 radar 扩到 83.8 的伏笔）。0.4 m/格 是工程折中：格子太细 BEV 图太大（显存、耗时都翻倍），太粗小目标（VRU）占不满一个格。
- 【连接】nuScenes 上的 BEVFusion 用的是 [-54, 54] 对称正方形、0.075→0.6 m 多档分辨率；DenseBEV 这种"前长后短"的非对称矩形是量产车常见配置——车头方向多给格子，寸土寸金的显存花在刀刃上。用户以后在 BEVFusion 里改 `point_cloud_range` 时可以对照这套换算：`格子数 = (range_max - range_min) / voxel_size`，必须整除。

---

### 卡 3-5 ⭐重点 ｜permute 一下，直接透传

> **原话** `[00:14:14]` "然后在这里它是直接就是一个," `[00:14:16]` "会做一个perpermute，就是把我们的Channel的维度放到BV去," `[00:14:23]` "然后再直接透传出去。"

- 【直译】lidar backbone 的 forward 只做一件事：把 channel 维从最后一位挪到第 1 位，然后原样输出。"放到BV去"口语上有点绕，意思是把 channel 维放到 BEV 空间维（H、W）**前面**，即 NHWC → NCHW。
- 【代码】帧证 `00_14_00.jpg` 180 行原文：
  ```python
  output = inputs[0].permute(0, 3, 1, 2).to(torch.float32).contiguous()
  ```
  三个链式调用各有用意：`permute(0,3,1,2)` 把 `[B,H,W,C]` 转成 `[B,C,H,W]`；`.to(torch.float32)` 把 DataLoader 可能给的半精度/整型统一成 fp32（这套代码没开 AMP，与用户 BEVFusion 训练计划里"无 AMP"的处境一致）；`.contiguous()` 让转置后的张量在内存里真实重排——permute 只改 stride 不搬数据，后续卷积要求连续内存，不加这句后面 `view/reshape` 会报错。
- 【形状】`[3, 352, 224, 64] → [3, 64, 352, 224]`。数据一个没动，只是"看数据的顺序"变了，再由 contiguous 落成物理重排。
- 【为什么】PyTorch 的 Conv2d 约定 NCHW；DataLoader 用 numpy 组装时 channel-last 更自然（逐格子写 64 维特征）。两边各按自己舒服的布局来，边界上一次 permute 解决。
- 【连接】帧证 `00_19_45.jpg` 的 draw.io 图把这一步画成了完整一条链：`input[4] (bs*3)*352*224*64 → permute → output (bs*3)*64*352*224`——整个 lidar backbone 的"网络结构图"就这三个框，和 radar 那一大串形成强烈反差。另外帧 `00_14_00.jpg` 里还能看到透传前的两个可选后处理：`rl_feature_crop_area`（四元组时对 BEV 做切片裁剪）和 `use_grid_mask`（对 BEV 特征做 GridMask 数据增强），当前配置都没启用，但说明这个"空壳"其实是个可配置的预处理站。

---

### 卡 3-6 ｜Lidar Backbone 没有参数

> **原话** `[00:14:27]` "在lidar的Backbone是没有参数的。"

- 【直译】这个模块里没有任何可学习权重——没有卷积、没有 BN、没有 Linear，`state_dict` 是空的。
- 【代码】等价于 `nn.Identity()` 加一个 permute。用 `sum(p.numel() for p in model.parameters())` 数一下就是 0。
- 【为什么】参数为 0 意味着：反向传播不经过它更新任何东西、加载 checkpoint 时它没有 key、剪枝/量化都跳过它。lidar 特征的"表达能力"完全由 DataLoader 的拍平规则决定，网络对 lidar 原始信息的加工要等到后面 RL 融合 UNet 才开始。
- 【连接】对比：radar backbone（本章下半场）有 Linear+Norm 的真参数；图像 backbone 有几千万参数。三条支路"参数投入"的悬殊排序（图像 >> radar > lidar=0），侧面反映了这套系统的定位——图像是主力，lidar 提供几何底座（已离线编码），radar 是轻量补充。

---

### 🔨 动手练习 ch3-1：复现 lidar 透传（permute + contiguous 的坑）

```python
import torch

# 造假数据：bs=1, 3帧折叠, BEV 352x224, 64维特征, channel-last（模拟DataLoader输出）
inputs0 = torch.randn(3, 352, 224, 64)

# --- lidar backbone forward 的全部内容 ---
output = inputs0.permute(0, 3, 1, 2).to(torch.float32).contiguous()

print(output.shape)          # 预期: torch.Size([3, 64, 352, 224])
print(output.is_contiguous())# 预期: True

# 体会 contiguous 的必要性：
x = inputs0.permute(0, 3, 1, 2)   # 不加 contiguous
print(x.is_contiguous())     # 预期: False —— 只是换了stride，数据没动
try:
    x.view(3, -1)            # view 要求连续内存
except RuntimeError as e:
    print("view 报错:", str(e)[:60])   # 预期: 报错，提示 use .reshape(...)

# 验证物理范围换算（卡3-4）
res = 0.4
print((95.4 + 45.4) / res)   # 预期: 352.0
print(44.8 * 2 / res)        # 预期: 224.0
```

**【小结】** ① lidar 分支的体素化+拍平全部在 DataLoader 离线完成，进网络的已是 `[bs*3, 352, 224, 64]` 的 BEV 特征；② 网络内只做 `permute(0,3,1,2)+contiguous` 的布局转换，模块零参数；③ 352×224 网格由"前95.4/后45.4/左右±44.8、0.4m分辨率"严格换算而来，这套换算式贯穿全网络。

---

## Part 2｜Radar Backbone 登场：四路输入与总框图（00:14:35–00:16:07，合并重讲段 00:19:43–00:20:21）

**导读**：radar 分支的输入不是一张现成特征图，而是**四个配套张量**：`voxels[N,16,10]`（每个 voxel 内 16 个点、每点 10 维特征）、`coords[N,3]`（每个 voxel 的网格坐标）、`num_points[N]`（每个 voxel 里真实点数）、`voxel_num[bs*3]`（每帧的 voxel 个数，加起来等于 N）。输出是 scatter 回 BEV 画布的 `(bs*3)×64×448×224`。本段讲者觉得"radar 这块比较复杂"，于是打开 draw.io 画的 `RadarVoxelGenerator` 框图（帧证 `00_15_57.jpg`），先图解、后看码。这个结构就是 **PointPillars 的 PillarFeatureNet（PFN）+ PointPillarsScatter** 的工程变体——用户可以把本段当作"手撕一遍 PointPillars 编码器"。

---

### 卡 3-7 ｜转场：radar backbone 的输入来自 DataLoader 的"这几维"

> **原话（合并 6 句转场/停顿）** `[00:14:35]` "然后的话就是我们RIDA的一个Backbone。" `[00:14:38]` "RIDA的Backbone刚刚刚说了," `[00:14:43]` "这里说了，会传这几个从DataLoader处理的这几个围的输入。" `[00:14:53]`/`[00:15:01]`/`[00:15:03]` "这几个围的输入……DataLoader……" `[00:15:12]` "我还是接一下这个围。"
> （"RIDA"=radar 的音译残留；"围"=维。这 6 句是讲者切文件+决定画图的过渡，无独立信息，合并成一卡。）

- 【直译】接下来讲 radar backbone。它的输入同样来自 DataLoader，但不是一个张量而是"好几维（个）输入"，讲者觉得口头说不清，决定"接一下这个维"——去 draw.io 上把输入画出来。
- 【代码】帧证 `00_14_41.jpg`：radar 分支入口在 `radar_backbone/voxel_generator.py` 的 `RadarVoxelGenerator.forward(*inputs, **kwargs)`，断点行 42 `voxelized = None` 高亮，紧接着 `if len(inputs) == 2: pts, pts_index = inputs; else: ...; voxelized = inputs`——同样用 `len(inputs)` 区分"原始点/已体素化"两种喂法，和 lidar 的 `len(inputs)==1` 判断是同一套接口设计。主干调用是 `radar_feature = self.radar_encoder(pts, pts_index, voxelized)`；代码里还躺着 `self.convertD`（走 `radar_encoder.forward_infer`，⚠疑为部署/ONNX导出专用路径）和 `self.voxel_pad`（额外返回 `[location, indices]`）两个分支，当前都不走。
- 【为什么】radar 点太稀疏（一帧几百个点），不值得像 lidar 那样离线拍成 448×224×C 的稠密图存盘——存储和 IO 都亏。所以 DataLoader 只做到"体素化+定长填充"这一步，把**升维和 scatter 留给 GPU 网络在线做**，顺便让 radar 编码器的 Linear 参数可学习。
- 【连接】这正是 BEVFusion 里 lidar 的标准处理流程（voxelize→VFE→scatter），只不过 DenseBEV 把它用在了 radar 上、而把 lidar 反而离线化了——两个模态的处理策略互换，核心考量就是点数密度差两个数量级。

---

### 卡 3-8 ｜"radar 比较复杂，我先按框图讲"（两遍开场白合并）

> **原话** `[00:15:27]` "我先因为感觉RIDA这一块会比较复杂," `[00:15:31]` "我就先按照网络画的一个框架先讲一下。"
> **重讲版** `[00:19:43]` "我先因为感觉radar这一块会比较复杂," `[00:19:46]` "我就先按照这个画的一个框架先讲一下。"（两遍逐字近似，合并）

- 【直译】讲者宣布讲解策略：先离开代码，用自己画的 draw.io 框图把数据流走一遍，再回代码对照。00:19:43 这句原话重现，就是前文说的"回炉重讲"的起点。
- 【帧证】`00_15_57.jpg`/`00_16_55.jpg` 就是这张图：浏览器开着 draw.io（地址栏 `…ams.net/?lang=zh#`，即 app.diagrams.net），画布上一个大粉框 `RadarVoxelGenerator`，内部三个子块 `self.voxelize`、`self.pts_voxel_encoder`（蓝底）、`self.pfn_layers`（绿底，嵌在 encoder 里），左侧悬着 4 个绿色输入块。
- 【为什么】radar 分支涉及"变长→定长→mask→池化→scatter"五次数据形态切换，纯看代码容易迷路；框图先建立"N 个 pillar 流水加工"的心智模型，代码就只是填细节。本精讲也沿用这个顺序。
- 【连接】子块命名 `pts_voxel_encoder`/`pfn_layers`/`pts_middle_encoder` 全部沿用 mmdet3d 的 PointPillars 命名（`PillarFeatureNet`/`PFNLayer`/`PointPillarsScatter`），说明这套 radar 编码器大概率是从 mmdet3d 抄改的——用户读过 BEVFusion 配置文件的话对这三个名字应该眼熟。

---

### 卡 3-9 ⭐重点 ｜输入一：voxels [N,16,10] —— 每个 voxel 16 个点、每点 10 维

> **原话** `[00:15:36]` "RIDA的输入主要是有这四维输入。" `[00:15:42]` "刚刚说了分别是voxel," `[00:15:46]` "就是对应的每一个voxel," `[00:15:47]` "然后16个点," `[00:15:49]` "然后每一个点十为特征。"
> **重讲版合并**：`[00:19:55]`–`[00:20:05]`（"radar的输入主要是有这四为输入…每一个voxel…16个点…每一个点十为特征"，逐字重复）

- 【直译】radar 有四路输入，第一路是 voxels 张量：N 个 voxel，每个 voxel 固定装 16 个点的槽位，每个点用 10 维向量描述。
- 【帧证】`00_15_57.jpg` 绿块原文：**`input[5] N*16*10 每一个voxel，16个点，每一个点10维特征`**。旁边画布边缘还标着变量名 `voxels`。注意输入编号是 input[5][6][7][9]——不连号，说明网络总输入是个大 tuple，图像占了前几号，radar 拿到的是 5、6、7、9 号位。
- 【形状】`[N, 16, 10]`。N 是**整个 folded-batch（bs×3 帧）所有非空 voxel 的总数**，每次迭代都不同（radar 点落在哪些格子是动态的）——这是本章第一个"动态形状"张量，与 lidar 那种固定 `[3,352,224,64]` 形成对照。
- 【为什么 16】radar 点极稀，一个 0.4m 格子里通常只有 1~几个点，16 是"最大容量"上限：超过 16 截断，不足 16 补零。定长化之后才能堆成规则张量上 GPU；补的零怎么排除，就是后面 get_paddings_indicator 的事（卡 3-18）。
- 【为什么 10 维】视频没有展开 10 维具体是什么。⚠推断：radar 点原始特征一般是 x、y、z、RCS（雷达散射截面）、vx/vy（径向速度分解，含自车运动补偿版），再加上 PointPillars 惯例的增强项（点到 voxel 质心的偏移 xc,yc,zc 或到 pillar 中心的 xp,yp）凑到 10 维。nuScenes 的 radar 点本身有 18 个字段，量产 radar 通常取其子集。具体组合需回 DataLoader 代码核实。
- 【连接】对比 BEVFusion lidar 分支的 voxels `[M, 10, 5]`（nuScenes：每 voxel 最多 10 点、每点 5 维 x,y,z,intensity,timestamp）——结构完全同构，只是"每格容量"和"每点维数"按传感器特性换了数字。理解了这一个，两边全通。

---

### 卡 3-10 ｜输入二：coords [N,3] —— voxel 中心的三维位置

> **原话** `[00:15:52]` "然后或者每一个voxel中心点在三维空间中的一个位置。"（"或者"应为"coords"的误听）
> **重讲版合并**：`[00:20:08]` "然后Core就是每一个voxel中心点," `[00:20:11]` "在三维空间中的一个位置。"（重讲版明确说了变量名 Core=coords）

- 【直译】第二路输入 coords：N 个 voxel 各自"在三维空间中的位置"，每个 3 个数。
- 【帧证】`00_15_57.jpg` 绿块：**`input[6] N*3 每一个voxel的中心点在三维空间中的位置`**，画布边标注变量名 `coors`（mmdet3d 系代码惯用的缩写拼法，比标准 coords 少个 d）。
- 【形状】`[N, 3]`，与 voxels 的第 0 维 N 一一对应：`voxels[i]` 这 16 个点都属于 `coords[i]` 指定的那个格子。
- ⚠【表述存疑】图上写"三维空间中的位置"，但按 mmdet3d 惯例，voxelize 输出的 coords 是**整数网格索引**（z_idx, y_idx, x_idx），不是米制物理坐标——后面 scatter 时要拿它当 BEV 图的行列下标用（卡 3-28），整数索引才说得通。推断这里 3 列是网格坐标 (z, y, x)，pillar 场景 z 恒为 0。"三维空间中的位置"是口语化说法，勿按字面理解为浮点坐标。
- 【为什么】coords 是稀疏表示的"地址簿"：特征加工阶段（PFN）完全不用它，最后 scatter 阶段全靠它把 N 条特征放回 448×224 画布的正确格子。稀疏计算的通用范式就是"数据张量+坐标张量"成对流动，用户以后看任何 sparse conv 代码（BEVFusion 的 SparseEncoder）都会再见到这对搭档。

---

### 卡 3-11 ｜输入三：num_points [N] —— 每个 voxel 里真实的 radar 点数

> **原话** `[00:15:57]` "然后还有的话就是每个voxel类RIDA点的一个个数。"（"类"=里）
> **重讲版合并**：`[00:20:13]` "然后还有的话就是每个voxel内radar点的一个个数,"

- 【直译】第三路输入：每个 voxel 里**实际**装了几个 radar 点（1~16 之间的整数）。
- 【帧证】`00_15_57.jpg` 绿块：**`input[7] N 每一个voxel内radar点的个数`**，变量名 `num_points`。
- 【形状】`[N]`，一维整型。`voxels[i]` 的 16 个槽位里，前 `num_points[i]` 个是真点，后面 `16 - num_points[i]` 个是补零的假点。
- 【为什么】定长填充丢失了"哪些是真点"的信息，num_points 就是把这个信息单独存一份。它是后面 get_paddings_indicator 的唯一原料——没有它，补零点会混进 max 池化：零特征经过 Linear+BN 后**不再是零**，会污染每个 pillar 的聚合特征。
- 【连接】这就是 NLP 里 padding + attention_mask 的完全同构物：句子补 `<pad>` 到定长、mask 记录真实长度；这里 pillar 补零点到 16、num_points 记录真实点数。一套思想两个领域。

---

### 卡 3-12 ｜输入四：voxel_num [bs*3] —— 每帧多少个 voxel，总和等于 N

> **原话** `[00:16:00]` "然后还有的话是每一帧voxel的个数。" `[00:16:06]` "这个是对应它的一个shape。"
> **重讲版合并**：`[00:20:16]` "然后还有的话是每一帧voxel的个数," `[00:20:21]` "这个是对应它的一个shape,"

- 【直译】第四路输入：把 folded-batch 里每一帧各自贡献了多少个 voxel 记下来。"这个是对应它的shape"指讲者用鼠标指着图上各绿块标的形状。
- 【帧证】`00_15_57.jpg` 绿块：**`input[9] (bs*3) 每一帧的voxel的个数，和=N`**，变量名 `voxel_num`。注释"和=N"信息量最大：`voxel_num.sum() == N`，即四路输入的 N 维是把 bs×3 帧的 pillar **按帧顺序首尾拼接**出来的。
- 【形状】`[bs*3]`，调试配置下就是 `[3]`，例如 `[812, 790, 805]` 之类（每帧约几百上千个非空格子），三个数加起来才是 voxels 的 N。
- 【为什么】不同帧的 voxel 数天然不同，无法堆成规则 batch 维，工程上便"拼长条+记分段"。voxel_num 提供分段信息，配合 coords 补上的帧索引（下一卡）实现两种用途：切分归属 + scatter 时选对画布。
- 【连接】与 PyTorch 官方 `torchvision` 处理变长 box 的 `batch_idx`、以及 point cloud 库里的 `batch offset`（如 spconv 的 indices 第 0 列）是同一招。BEVFusion 里 voxelize 后干脆直接把 batch_idx `F.pad` 进 coords——DenseBEV 下一步做的就是这件事。

---

### 🔨 动手练习 ch3-2：造出 radar 的四路输入

```python
import torch

bs, frames, max_pts, feat_dim = 1, 3, 16, 10
# 每帧非空voxel数（模拟动态N）
voxel_num = torch.tensor([812, 790, 805])       # input[9]: (bs*3,)
N = int(voxel_num.sum())                        # N = 2407

voxels = torch.zeros(N, max_pts, feat_dim)      # input[5]: N*16*10
num_points = torch.randint(1, 6, (N,))          # input[7]: N  (radar稀疏,每格1~5点)
# 前 num_points 个槽位填真点特征，其余保持0（模拟DataLoader定长填充）
for i in range(N):
    voxels[i, :num_points[i]] = torch.randn(num_points[i], feat_dim)

# input[6]: N*3 网格坐标 (z=0, y<224方向, x<448方向) —— pillar场景z恒0
coords = torch.stack([
    torch.zeros(N, dtype=torch.long),
    torch.randint(0, 224, (N,)),
    torch.randint(0, 448, (N,)),
], dim=1)

print(voxels.shape, coords.shape, num_points.shape, voxel_num.shape)
# 预期: torch.Size([2407, 16, 10]) torch.Size([2407, 3]) torch.Size([2407]) torch.Size([3])
print("和=N 校验:", voxel_num.sum().item() == N)   # 预期: True
```

**【小结】** ① radar 输入是四件套：voxels[N,16,10]、coords[N,3]、num_points[N]、voxel_num[bs*3]，N 为三帧 pillar 总数、每 iter 动态变化；② 定长 16 槽位+num_points 记真实长度，是"变长数据上 GPU"的标准解法；③ 四路张量按帧拼长条，voxel_num"和=N"负责记住分段——这套稀疏表示范式与 BEVFusion/mmdet3d 完全同源。

---

## Part 3｜self.voxelize：给 coords 补上帧索引（00:16:07–00:17:20，合并重讲段 00:20:23–00:21:38）

**导读**：四路输入进 `RadarVoxelGenerator` 后的第一站是 `self.voxelize` 子块。名字叫 voxelize，实际**并不做体素化**（体素化在 DataLoader 已完成）——它只干一件小事：把 coords 从 `[N,3]` 变成 `[N,4]`，在最前面 pad 一列"这个 voxel 属于第几帧"的索引，得到 `coords_batch`；voxels 和 num_points 原样透传。这一列帧索引是后面 scatter 能把三帧 pillar 分别填进三张 BEV 画布的关键。

---

### 卡 3-13 ｜voxelize 子块登场：处理对象是 N×3 的 coords

> **原话** `[00:16:07]` "然后首先会经过一个voxel的一个操作。" `[00:16:12]` "其实会把我们的coords会对它," `[00:16:16]` "因为它是N乘3," `[00:16:18]` "N就是有对应的就是有N个voxel," `[00:16:23]` "然后3就是对应的中心点的XYZ。"
> **重讲版合并**：`[00:20:23]`–`[00:20:39]`（"首先会经过一个voxel的一个操作…把我们的Core…N乘3…N个voxel…3就是对应的中心点的XYZ"，逐字重复）

- 【直译】第一步操作只针对 coords：它是 N×3，N 个 voxel、每个 3 个坐标分量（讲者口头说 XYZ）。
- 【帧证】`00_16_55.jpg`：`self.voxelize` 块（浅蓝）左接四个绿块，右出三口 `voxels N*16*10`、`num_points N`、`coors_batch N*4`。可见它是四进三出：voxel_num 被"消化"掉了（它的信息转移进了帧索引列）。
- 【代码】mmdet3d 同款操作（对每帧 i 的 coords 前面 pad 常数 i 再 concat）：
  ```python
  coords_batch = []
  offset = 0
  for i, n in enumerate(voxel_num):          # 遍历 bs*3 帧
      c = coords[offset:offset + n]          # 该帧的 [n,3]
      c = F.pad(c, (1, 0), value=i)          # 左侧pad一列帧索引 → [n,4]
      coords_batch.append(c); offset += n
  coords_batch = torch.cat(coords_batch)     # [N,4]
  ```
- ⚠【表述存疑】"3就是中心点的XYZ"同卡 3-10 的疑点：更可能是网格索引 (z_idx,y_idx,x_idx) 而非物理 XYZ；讲者两遍都这么说，属于口语惯性，不影响后续逻辑。
- 【连接】mmdet3d 的 `Base3DDetector.voxelize()` 里一模一样的循环：`coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)`。用户翻 BEVFusion 源码 `mmdet3d/models/detectors/` 就能找到原型。

---

### 卡 3-14 ⭐重点 ｜在第 0 位 pad 当前帧的索引

> **原话** `[00:16:28]` "然后会额外在3的定名为会给它Pine的一个,"（"3的定名为"=第3维的第0位；"Pine"=pad） `[00:16:33]` "它是属于当前的一帧的一个索引。"
> **重讲版合并**：`[00:20:43]` "然后会额外在3的D0为," `[00:20:47]` "会给它Tine的一个,"（"Tine"仍是pad的误听） `[00:20:49]` "它是属于当前雷帧的一个索引,"

- 【直译】在 coords 那个"3"的维度上、**索引 0 的位置**额外补一个数：该 voxel 属于当前 batch 里第几帧。补完每行变 4 个数。
- 【帧证】`00_16_55.jpg` 的 `self.voxelize` 块内注释原文：**"对coors的第二维索引0的位置pad当前帧的索引"**——图上白纸黑字，把转写里"Pine/Tine"的谜团解掉了：就是 `F.pad(..., (1,0), value=frame_idx)`。
- 【形状】`[N,3] → [N,4]`，新列取值范围 `0 … bs*3-1`（调试时 0/1/2）。每行含义变为 `(帧号, z_idx, y_idx, x_idx)`。
- 【为什么】四路输入把三帧 pillar 拼成了一根长条（卡 3-12），拼完之后"谁属于哪帧"只剩 voxel_num 的分段信息——分段信息在后续逐 voxel 的并行操作里很难用（要不停做前缀和切片）。把帧号**逐行钉在数据上**之后，任何时刻拿到一行 coords_batch 都能自描述归属，scatter 时直接 `coords_batch[:,0]==i` 选出第 i 帧的 pillar。空间换易用性。
- 【为什么 5 角度补充——不这么做会怎样】如果不 pad 帧索引，scatter 阶段三帧的 pillar 会全部砸进同一张 BEV 画布：不同帧、相同格子的特征互相覆盖，时序信息直接毁掉。这一列 4 字节整数是三张画布互不串门的唯一保险。
- 【连接】BEVFusion 里同位置 pad 的是 **batch 样本索引**；DenseBEV 因为把帧折进了 batch，这里 pad 的自然就是"帧索引"（本质还是 folded-batch 的样本索引）。名字不同，机制完全一样。

---

### 卡 3-15 ｜voxels 和 num_points 原样透传

> **原话** `[00:16:38]` "然后在这里会直接," `[00:16:42]` "会用，会把这里的voxel会直接透传过来。" `[00:16:45]` "然后以及num_points也会透传过来。"
> **重讲版合并**：`[00:20:53]`–`[00:21:01]`（"会把这里的voxel会直接透传过来，然后以及Number Input也会透传过来"——⚠"Number Input"为 num_points 误听，第一遍转写已校正）

- 【直译】voxelize 子块对 voxels 和 num_points 什么都不做，进来什么出去什么。
- 【帧证】`00_16_55.jpg`：voxelize 右侧三个出口小白框 `voxels N*16*10`、`num_points N` 的形状与输入侧完全一致，只有 `coors_batch N*4` 是新的。
- 【为什么】voxelize 子块的职责被刻意收窄成"只补帧索引"——单一职责让框图/代码一一对应，调试时形状哪一步变了一目了然。
- 【连接】"透传"这个词本章出现两次：lidar backbone 整体透传（卡 3-5）、这里 voxels/num_points 局部透传。可以体会这套代码的风格：**能不动数据就不动**，每个模块只做名字承诺的最小变换。

---

### 卡 3-16 ｜coords_batch：N×4

> **原话** `[00:16:50]` "然后对应处理的coords会变成这里的coords_batch," `[00:16:55]` "就变成了N乘4的一个数值。"
> **重讲版合并**：`[00:21:06]` "然后对应处理的Core会变成这里的coords_batch," `[00:21:11]` "就变成N乘4的一个数值,"

- 【直译】处理后的坐标改名叫 coords_batch，形状 N×4。
- 【形状】`[N,4]`：`(frame_idx, z, y, x)`。变量名后缀 `_batch` 直说了新列的含义——带 batch（帧）归属的坐标。
- 【代码】此后 coords_batch 将沉睡整个 Part 4/5（特征加工用不到坐标），直到 Part 6 scatter 才被唤醒。数据流上它和 voxels 是"平行铁轨"：一根轨道运特征，一根轨道运地址。
- 【连接】mmdet3d 里这个变量叫 `coors_batch`（帧证 `00_16_55.jpg` 图上拼写正是 `coors_batch N*4`），又一次坐实"抄改自 mmdet3d"的判断（卡 3-8）。

---

### 卡 3-17 ｜预告：后面会把 voxel 变成 N×64（第一遍讲述在此断流）

> **原话（合并 4 句）** `[00:16:58]` "然后会在这里呢，会有一个," `[00:17:03]` "在这里会提取一个RIDA的一个特征。" `[00:17:07]` "就是会经过，把我们的voxel变成了N乘64。" `[00:17:20]` "对,……"
> **另合并第一遍尾部的卡壳段** `[00:19:11]`–`[00:19:37]`："然后会在这里呢会有一个，在这里会提取一个radar的一个特征，就是会经过，把我们的voxel我看一下，voxel变成了N乘64，好吧。"（讲者翻代码确认形状，随后从 00:19:43 起整段重讲）
> **及重讲版对应句** `[00:21:13]`–`[00:21:32]`："然后会在这里呢会有一个……在这里会提取一个radar的一个特征……把我们的voxel变……我看一下……voxel变成了N乘64,"

- 【直译】这几句是同一个意思说了三遍：接下来 pts_voxel_encoder 要把 voxels 从 `[N,16,10]` 加工成 `[N,64]` 的 pillar 特征。中间"我看一下""好吧"是讲者切窗口核对代码的停顿。
- 【形状】预告了本章最核心的形状变化链：`[N,16,10] →(mask)→ [N,16,10] →(Linear+Norm+ReLU)→ [N,16,64] →(max over 16)→ [N,64]`——16 个点的集合被压成 pillar 的一个 64 维向量。
- 【为什么】把它单独立卡，是因为这是讲者两次断流的"接缝处"：第一遍讲到这里发现自己没记清细节，回代码查证后决定整段重讲。学习者在此处最容易被转写绕晕——认清 00:19:43–00:23:18 是重播，按本精讲的合并卡读即可。
- 【连接】"集合→向量"的压缩正是 PointNet 思想（对无序点集用共享 MLP+对称函数池化），PointPillars 的 PFN 就是单层 PointNet。下一 Part 逐步拆开。

---

### 🔨 动手练习 ch3-3：复现 voxelize 的 pad 帧索引

```python
import torch
import torch.nn.functional as F

voxel_num = torch.tensor([4, 3, 5])            # 3帧，各4/3/5个voxel（缩小版）
N = int(voxel_num.sum())
coords = torch.randint(0, 10, (N, 3))          # [N,3] 网格坐标

# --- self.voxelize 的核心：对coords第0位pad当前帧索引 ---
out, offset = [], 0
for i, n in enumerate(voxel_num.tolist()):
    c = coords[offset:offset + n]
    out.append(F.pad(c, (1, 0), value=i))      # 左侧补一列常数i
    offset += n
coords_batch = torch.cat(out)                  # [N,4]

print(coords_batch.shape)                      # 预期: torch.Size([12, 4])
print(coords_batch[:, 0])                      # 预期: tensor([0,0,0,0, 1,1,1, 2,2,2,2,2])
# 第0列就是帧索引，后3列原样保留 —— 与帧00_16_55注释
# "对coors的第二维索引0的位置pad当前帧的索引"逐字对应
```

**【小结】** ① `self.voxelize` 名不副实，不做体素化，只把 coords 从 [N,3] pad 成 [N,4] 的 coords_batch，新增第 0 列=帧索引；② voxels/num_points 原样透传，voxel_num 的分段信息被"逐行钉进"坐标里；③ 这列帧索引是后续三帧 pillar 各回各家（各自 BEV 画布）的唯一凭据，丢了它时序就废了。

---

## Part 4｜get_paddings_indicator：把补零的假点 mask 掉（00:17:20–00:18:34，合并重讲段 00:21:39–00:22:49）

**导读**：进入蓝底大框 `self.pts_voxel_encoder`。第一步是解决定长填充留下的隐患：每个 voxel 的 16 个槽位里混着补零假点。`get_paddings_indicator` 拿 num_points 生成 `[N,16,1]` 的布尔 mask，与 voxels `[N,16,10]` 逐元素相乘，让假点特征归零、真点保留，得到干净的 `feature [N,16,10]`。这是"变长数据定长化"的标准收尾动作，输入输出形状不变、内容被"消毒"。

---

### 卡 3-18 ｜get_paddings_indicator 登场，输入是 num_points

> **原话** `[00:17:20]` "对，然后在这里有的get_paddings_indicator," `[00:17:26]` "get_paddings_indicatorInDictor,"（口吃复读） `[00:17:28]` "然后它会输入我们的LamboInPost," `[00:17:32]` "然后以及对应的这个," `[00:17:35]` "会根据这个去算我们这个每个voxel类16个点哪些是有效的。"
> **重讲版合并**：`[00:21:39]` "然后在这里有一个Gate Pending," `[00:21:42]` "Gate Pending in Dictor,"（=get_paddings_indicator 音译） `[00:21:44]` "然后它会输入我们的num_points," `[00:21:48]`–`[00:21:59]` "……会根据这个去算我们每个voxel类16个点哪些是有效的,"

- 【直译】这里调用一个叫 get_paddings_indicator 的函数，输入 num_points（⚠第一遍转写"LamboInPost"是 num_points 的严重误听，重讲版 00:21:44 讲者咬字清楚，已交叉确证），算出每个 voxel 的 16 个槽位中哪些是有效真点。
- 【帧证】`00_17_36.jpg`：米黄色块 `get_paddings_indicator`，块内注释 **"计算每一个voexl内有效点的mask"**（图上 voxel 拼成 voexl），入边来自 `num_points N`，出边指向 `mask N*16*1`。
- 【代码】mmdet3d 原版实现（本工程同款）：
  ```python
  def get_paddings_indicator(actual_num, max_num, axis=0):
      actual_num = torch.unsqueeze(actual_num, axis + 1)       # [N] -> [N,1]
      max_num = torch.arange(max_num).view(1, -1)              # [1,16]: 0..15
      return actual_num.int() > max_num                        # [N,16] bool
  ```
  一行广播比较：槽位号 `0..15` 小于该 voxel 真实点数的位置为 True。
- 【形状】`num_points [N] → mask [N,16]`，再 `unsqueeze(-1)` 成 `[N,16,1]` 以便和 `[N,16,10]` 广播相乘。
- 【连接】与 Transformer 的 `attention_mask = (ids != pad_id)` 完全同构（卡 3-11 已铺垫）；也和用户在 BEVFusion 里见过的 `mask = get_paddings_indicator(num_points, voxels.size(1), axis=0)` 一字不差——搜 `pillar_encoder.py` 即得。

---

### 卡 3-19 ｜为什么需要 mask：每个 voxel 的点数不一样多

> **原话** `[00:17:43]` "因为每每个voxel类它的那个点的个数实际上是不一样," `[00:17:48]` "不一样多的," `[00:17:49]` "是不一样多的。"（同义复读，合并）
> **重讲版合并**：`[00:21:59]`–`[00:22:07]` "因为每个voxel类它的那个点的个数实际上是不一样，不一样多的,"

- 【直译】根本原因：radar 点在空间里散布不均，每个格子网到的点数天然不同——有的 1 个、有的 5 个、偶尔挤满 16 个。
- 【形状】num_points 的取值分布决定了 mask 的稀疏度。radar 场景 num_points 大多是个位数，意味着 16 个槽位常常 **一大半是补零假点**——mask 不做，假点占多数票，池化结果基本被零污染。
- 【为什么】GPU 张量必须规则（矩形），现实数据必须参差（变长）——mask 是两者之间的通用桥。这句话虽朴素，却是所有稀疏/变长深度学习代码的第一性原理。
- 【连接】lidar 同样有此问题，但 DenseBEV 的 lidar 在 DataLoader 离线拍平时已经处理掉了；radar 在线编码，所以 mask 出现在网络里。同一问题、两种解法的空间分布，再次呼应本章开头"两分支策略互换"的观察（卡 3-7）。

---

### 卡 3-20 ｜16 是 DataLoader 定的每格最大点数上限

> **原话** `[00:17:51]` "所以说会在DataLoader的时候," `[00:17:53]` "它是统一设置了一个最大的," `[00:17:56]` "每个voxel类最大的一个点数设置为16个。"
> **重讲版合并**：`[00:22:09]`–`[00:22:12]` "所以说会在num_points的时候，它是统一设置了一个最大的，每个voxel类最大的一个点数设置为16个,"（⚠重讲版口误说成"在num_points的时候"，第一遍"在DataLoader的时候"才是对的——上限16是 DataLoader 体素化时截断/补零用的超参）

- 【直译】"16"这个数是 DataLoader 侧的配置：体素化时每格最多留 16 个点，多了截断、少了补零。
- 【代码】对应 mmdet3d `Voxelization(max_num_points=16, ...)` 的参数。补零发生在 DataLoader，**去零（mask）发生在网络**——一对跨进程的配套操作，16 是它们之间的契约数字。
- 【为什么 16 而不是 10 或 32】上限取"覆盖绝大多数格子的真实点数分布 + 对齐友好"的折中。radar 每格点数通常个位数，16 已是宽裕上限；截断丢点的概率极低。定 32 则 voxels 张量直接翻倍，全是零。
- 【连接】BEVFusion 的 nuScenes lidar 配置是 `max_num_points=10`（点密所以格小、每格点少）；radar 格大点稀，16 反而更宽松。参数逻辑：**每格容量 ≈ 传感器点密度 × voxel 底面积的高分位数**。

---

### 卡 3-21 ｜逐 voxel 计算 16 槽位中的有效位

> **原话** `[00:18:00]` "然后对，在这里的话会去算一下," `[00:18:03]` "会去计算一下我每一个voxel类16个点哪些点是真实有效的。"
> **重讲版合并**：`[00:22:16]`–`[00:22:19]` "然后对在这里的话会去算一下，会去计算一下我每一个voxel类16个点哪些点是真实有效的,"

- 【直译】具体执行卡 3-18 的函数：对全部 N 个 voxel 并行地判断各自 16 槽位的真假。
- 【代码】`torch.arange(16) < num_points[:, None]` 一次广播完成 N×16 个比较，没有任何循环——GPU 上是一个 elementwise kernel 的事。
- 【形状】输出布尔 `[N,16]`：第 i 行前 `num_points[i]` 个 True、其余 False（DataLoader 是顺序填点，真点必在前段——这是"前缀有效"假设成立的前提）。
- 【为什么】强调"真实有效"是因为假点的特征值恰好是 0.0，而 0.0 在数值上是合法输入——网络自己分不出真假，必须靠元数据（num_points）外部标注。任何"用魔法值填充"的方案都有这个宿命。

---

### 卡 3-22 ｜生成 mask：N×16×1

> **原话** `[00:18:13]` "然后在这里会生成这样一个Mask,"（后半句见下卡）

- 【直译】上一步的布尔结果落成名为 mask 的张量。
- 【帧证】`00_17_36.jpg`：白色小框 **`mask N*16*1`**，上游是 get_paddings_indicator，下游与 `voxels N*16*10` 在标着 `*` 的节点汇合。
- 【形状】`[N,16] → unsqueeze(-1) → [N,16,1]`，再转 float。最后补的 1 维是为了与 10 维特征广播：`[N,16,1] * [N,16,10] → [N,16,10]`。
- 【代码】`mask = torch.unsqueeze(mask, -1).type_as(voxels)`——mmdet3d 原句。`type_as` 顺带把 bool 转成和特征同 dtype（fp32），乘法才合法。
- 【连接】"unsqueeze 到可广播"是用户在智谷课程 numpy/张量广播一节的直接应用：尾部维度 1 对 10，规则允许拉伸。

---

### 卡 3-23 ⭐重点 ｜mask 与 voxels 相乘：有效点保值、无效点归零

> **原话** `[00:18:16]` "这个Mask可能会和生成的voxel会做一个相成,"（"相成"=相乘；"可能"是讲者对细节不确定的口头缓冲） `[00:18:22]` "然后出来的值呢就是变成N乘16还是乘10," `[00:18:27]` "然后里面具体有效的radar点," `[00:18:30]` "然后才是有数值的," `[00:18:31]` "无效的话就是0。"
> **重讲版合并**：`[00:22:29]`–`[00:22:47]` "然后在这里会生成这样一个Mask，这个Mask可能会和生成的voxel会做一个相成，然后出来的值就是变成N16还是成10，然后里面具体有效的radar点，然后才是有数值的，无效的话就是0,"

- 【直译】mask 与 voxels 逐元素相乘，输出形状仍是 N×16×10（"还是乘10"＝形状没变），但内容被清洗：有效槽位保留原特征，无效槽位强制为 0。
- 【帧证】`00_17_36.jpg`：voxels 和 mask 两条箭头汇进一个标 `*` 的节点，出边 **`feature N*16*10`** ——图上的 `*` 就是逐元素乘。
- 【代码】`features = voxels * mask` 一行。⚠讲者说"可能会……做一个相乘"带着不确定语气（他还没翻到代码），但框图和 mmdet3d 原版都确证就是乘法，本精讲按确定处理。
- 【形状】`[N,16,10] * [N,16,1] → [N,16,10]`（广播）。信息量变化：形状不变、熵减——假点从"值恰为0的合法数据"变成"被显式置0的哨兵"。
- 【为什么此处乘一次还不够、后面还要靠 max】置 0 后假点仍会过 Linear：`W·0+b = b`，BN 还会再加均值偏移——假点在 `[N,16,64]` 里**不是 0 而是某个常数向量**。真正把它们排除出局的是下一步 max 池化：只要任一真点在某通道上超过该常数，假点就选不上。⚠但若某通道所有真点激活都低于偏置值，max 仍可能取到假点的 b——这是 PointPillars 系实现的知名小瑕疵（mmdet3d 靠 ReLU 后特征非负 + 经验上无碍容忍它）。此处标注供用户日后排查数值问题时参考。
- 【连接】BEVFusion `PillarFeatureNet.forward` 里同样是先 `features *= mask` 再进 PFNLayer——两边代码可逐行对照读。

---

### 🔨 动手练习 ch3-4：从零写 get_paddings_indicator 并验证消毒效果

```python
import torch

def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)   # [N,1]
    ids = torch.arange(max_num, device=actual_num.device).view(1, -1)  # [1,16]
    return actual_num.int() > ids                        # [N,16] bool

N, P, C = 5, 16, 10
voxels = torch.randn(N, P, C)
num_points = torch.tensor([3, 1, 16, 7, 2])

mask = get_paddings_indicator(num_points, P)             # [5,16]
print(mask[0])   # 预期: 前3个True其余False
print(mask.sum(1))  # 预期: tensor([ 3,  1, 16,  7,  2]) —— 与num_points一致

mask = mask.unsqueeze(-1).float()                        # [5,16,1]
feature = voxels * mask                                  # [5,16,10]
print(feature.shape)                                     # 预期: torch.Size([5, 16, 10])
print(feature[0, 3:].abs().sum())                        # 预期: tensor(0.) 假点全零
print(feature[0, :3].abs().sum() > 0)                    # 预期: tensor(True) 真点保值
```

**【小结】** ① get_paddings_indicator 用一次广播比较把 num_points[N] 变成 mask[N,16,1]，标出每 voxel 16 槽位中的真点；② `feature = voxels * mask` 形状不变（N×16×10）、内容消毒——有效点保值、补零假点显式归零；③ mask 的彻底生效要等下一步 max 池化配合，且存在"偏置项可能漏进 max"的已知小瑕疵，工程上被容忍。

---

## Part 5｜self.pfn_layers：Linear+Norm+ReLU 升到 64 维，沿 16 点取 max（00:18:34–00:19:37，合并重讲段 00:22:49–00:23:18）

**导读**：清洗后的 `feature [N,16,10]` 进入绿底块 `self.pfn_layers`——PointPillars 的 PFNLayer。共享的 Linear 把每点 10 维升到 64 维，Norm+ReLU 之后得到 `[N,16,64]`，再沿"16 个点"这个维度取 max，把点集压成每 pillar 一个 64 维向量 `[N,64]`。这是全章唯一有可学习参数的地方，也是 PointNet 思想（共享 MLP + 对称池化）的最小实现。

---

### 卡 3-24 ⭐重点 ｜Linear + Norm + ReLU：每个点 10 维 → 64 维

> **原话** `[00:18:34]` "然后在这里呢会有," `[00:18:36]` "会经过Linear," `[00:18:38]` "Norm," `[00:18:39]` "然后以及RELU," `[00:18:40]` "然后把我们的voxel的一个feature," `[00:18:43]` "然后在这里处理," `[00:18:45]` "会把它变成N乘16乘以64,"
> **重讲版合并**：`[00:22:49]`–`[00:23:01]` "然后在这里会经过Linear，Norm，以及ReLU，然后把我们voxel的一个feature，然后在这里处理，会把它变成N16乘以64,"

- 【直译】feature 依次过线性层、归一化层、ReLU 激活，输出 N×16×64——每个点的 10 维特征被独立映射成 64 维。
- 【帧证】`00_18_52.jpg`：绿底 `self.pfn_layers` 内三个黄块 **`linear → norm → relu`**，relu 出边上标注 **`N*16*64`**。
- 【代码】mmdet3d PFNLayer 同款：
  ```python
  self.linear = nn.Linear(10, 64, bias=False)
  self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
  x = self.linear(features)                        # [N,16,10] -> [N,16,64]
  x = self.norm(x.permute(0,2,1)).permute(0,2,1)   # BN1d按通道归一，需先转[N,64,16]
  x = F.relu(x)
  ```
  ⚠Norm 的具体类型（BatchNorm1d vs LayerNorm）视频未明说，按 mmdet3d 血统推断为 BN1d（linear 配 bias=False 也因 BN 会吸收偏置）；建议回代码核实。
- 【形状】`[N,16,10] → [N,16,64]`。Linear 只作用在最后一维，对 N×16 个点**共享同一组 10×64 权重**——参数量 640 个，radar 编码器轻得可以忽略。
- 【为什么共享权重】点集无序且数量不定，给每个槽位配独立权重既不满足置换不变性也不能泛化；共享 MLP 让"点的编码规则"与点的位置/数量解耦，这正是 PointNet 的第一原理。
- 【连接】升到 64 维是为了对齐 lidar 分支的 64 维 BEV 特征（卡 3-3）——后面 RL 融合要把两者拼/加在一起，同维度是前提。用户在智谷课程学过 1×1 卷积升维：`nn.Linear(10,64)` 作用于点维，数学上等价于对 `[N,10,16]` 做 1×1 Conv1d。

---

### 卡 3-25 ⭐重点 ｜沿 16 个点取 max：N×16×64 → N×64（⚠转写"取Mask"实为"取max"）

> **原话** `[00:18:52]` "然后会再沿着16会取一个Mask," `[00:18:59]` "然后会变成我们真正的一个radar的,"
> **重讲版合并**：`[00:23:07]` "然后会再沿着16会取一个Mask," `[00:23:13]` "然后会变成我们真正的一个radar的一个feature," `[00:23:18]` "就是N164,"（=N×64）

- ⚠【转写勘误·本章最重要的一处】两遍转写都写"沿着16取一个 **Mask**"，но帧证 `00_18_52.jpg` 上 relu 之后的白框写的是 **`totch.max(x,dim=1)`**（图上 torch 手误拼成 totch），出边接 **`feature N*64`**。结合 dim=1 正是 16 所在维，讲者说的必然是"沿着16取一个 **max**"——Whisper 把 max 听成了 Mask（前文 mask 出现频繁，语言模型带偏）。文件头术语说明也提示过"均值/最大值建议对代码核实"，帧证在此拍板：**是 max，最大值池化**。
- 【直译】在 16 个点的维度上取每通道最大值，把 pillar 内的点集压成单个 64 维向量——这才是"真正的 radar feature"。
- 【代码】`x_max = torch.max(x, dim=1)[0]`（`[0]` 取 values 丢 indices）。
- 【形状】`[N,16,64] → [N,64]`。每个输出通道回答"这个 pillar 里最强的那个响应是多少"，与点的顺序、数量无关。
- 【为什么用 max 不用 mean】① max 是置换不变的对称函数且梯度稀疏（只回传给最强点），抗补零假点干扰的能力也强于 mean（mean 会被大量零/常数假点拉低，除非按 num_points 归一）；② PointNet 论文实验证明 max 在点集特征聚合上优于 mean/sum；③ radar 点少而珍贵，"最显著响应"比"平均响应"更能保住强散射目标（如金属车体）的信号。
- 【连接】PointPillars 原文 PFN 用的就是 max；BEVFusion 里 `PFNLayer.forward` 的 `torch.max(x, dim=1, keepdim=True)[0]` 同款。至此 pillar 编码完成，与 PointPillars 唯一的差别是这里只有一层 PFN（PointPillars 也常只用一层）。

---

### 卡 3-26 ｜一处口误：Core 变成 "N乘16"（应为 coords_batch N×4 的回指）

> **原话** `[00:19:02]` "然后对应处理的这个Core," `[00:19:06]` "会变成这里的Core," `[00:19:08]` "就变成N乘16的一个数值,"

- 【直译】讲者第一遍讲到 PFN 输出时顺嘴回指了 coords 的处理，说"Core 变成 N乘16"。
- ⚠【存疑标注】"N乘16"与任何一路 coords 张量都对不上：coords 是 N×3、coords_batch 是 N×4。结合上下文（前一句刚讲完 mask/max、后面紧接翻代码卡壳），推断这是把"coords_batch 变成 N×4"或"feature 变成 N×16×64"两句话搅在一起的口误。重讲版对应位置（00:21:06–00:21:11）说的是"coords_batch 变成 N乘4"，正确。本卡不提供技术结论，仅提示读者跳过此句的字面意思。
- 【为什么保留此卡】按逐句纪律，含糊句也要交代去向，避免读者拿转写原文对照时怀疑自己漏了一个"N×16 的 coords"。没有这个张量，放心。

---

### 🔨 动手练习 ch3-5：迷你 PFN——Linear+BN+ReLU+max 一条龙

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

N, P, C_in, C_out = 2407, 16, 10, 64
feature = torch.randn(N, P, C_in)                 # 已mask消毒的 [N,16,10]
num_points = torch.randint(1, 6, (N,))
mask = (torch.arange(P)[None, :] < num_points[:, None]).unsqueeze(-1).float()
feature = feature * mask                          # 假点归零（衔接练习ch3-4）

linear = nn.Linear(C_in, C_out, bias=False)       # 参数量仅 10*64=640
norm = nn.BatchNorm1d(C_out, eps=1e-3, momentum=0.01)

x = linear(feature)                               # [N,16,64]
x = norm(x.permute(0, 2, 1)).permute(0, 2, 1)     # BN1d按64通道归一
x = F.relu(x)
print(x.shape)                                    # 预期: torch.Size([2407, 16, 64])

pillar_feat = torch.max(x, dim=1)[0]              # 沿16个点取max
print(pillar_feat.shape)                          # 预期: torch.Size([2407, 64])

# 验证max的置换不变性：打乱16个点的顺序，输出不变
perm = torch.randperm(P)
x2 = torch.max(x[:, perm, :], dim=1)[0]
print(torch.allclose(pillar_feat, x2))            # 预期: True
```

**【小结】** ① PFN 用共享 Linear(10→64)+Norm+ReLU 对 N×16 个点逐点升维到 [N,16,64]，是本章唯一有参数的模块（仅 640 权重+BN）；② 沿 16 点维 `torch.max(dim=1)` 压成 [N,64]——转写的"取Mask"实为"取max"，帧上 `totch.max(x,dim=1)` 为铁证；③ 共享MLP+max池化=最小号 PointNet，保证对点序/点数的不变性，与 PointPillars/BEVFusion 的 PFNLayer 同宗同款。

---

## Part 6｜self.pts_middle_encoder：scatter 回 BEV 画布，radar 独享的 448×224（00:23:24–00:24:49）

**导读**：radar 流水线最后一站（本段只讲了一遍，无重复）。拿着 `feature [N,64]` 和沉睡已久的 `coords_batch [N,4]`，`pts_middle_encoder` 先初始化 bs*3 张全零 BEV 画布，再按坐标把每个 pillar 的 64 维特征填进对应格子，输出 `(bs*3)×64×448×224` 的稠密 radar BEV。注意 448≠lidar 的 352：radar 把后向范围从 45.4 m 扩到 83.8 m——为了看清后方远处快速接近的来车。图上出口还挂着一个 `crop` 小块，负责后续对齐裁剪。

---

### 卡 3-27 ⭐重点 ｜根据 coords 初始化一张 BEV 特征画布

> **原话** `[00:23:24]` "然后根据我们生成的coords," `[00:23:30]` "然后会我们会在这里生成," `[00:23:34]` "会初始化一个BV的一个特征,"

- 【直译】进入 pts_middle_encoder：先凭空造一张（一批）BEV 特征图，初值全零，作为待填充的画布。
- 【帧证】`00_24_05.jpg`/`00_24_43.jpg`：`self.pts_middle_encoder` 块内注释原文 **"根据coors，把特征填充到初始的bev feature上"**；上方入边接 **`batch_size bs*3`**——画布张数由折叠后的帧数决定，每帧一张。
- 【代码】等价写法：
  ```python
  canvas = torch.zeros(bs*3, 64, 448, 224, device=feature.device)
  ```
  或 mmdet3d PointPillarsScatter 的逐帧版本：先建 `[64, 448*224]` 的展平画布，算 `index = y * 224 + x`（⚠行列先后按实际代码定），`canvas[:, index] = feature_t`，最后 reshape 回 `[64,448,224]`。
- 【为什么】前面所有步骤都在"稀疏世界"（N 条记录）里省算力；但后续 RL 融合的 UNet 是标准稠密卷积网络，只吃规则网格。scatter 是稀疏→稠密的官方换乘站。N 个 pillar 只占 448×224=100352 格中的百分之几，画布绝大部分保持零——radar BEV 是一张"星空图"。
- 【连接】BEVFusion 的 `PointPillarsScatter` 类干的就是这件事（用户可搜 `pts_middle_encoder` 配置项——连注册名都一样）。CenterPoint/PointPillars 论文里叫 "scatter back to a 2D pseudo-image"，伪图像一词很传神。

---

### 卡 3-28 ⭐重点 ｜按 voxel 的 BEV 位置逐条填充特征

> **原话** `[00:23:38]` "然后根据每一个voxel它对应是在BV上的哪个位置," `[00:23:43]` "把我们生成的radar," `[00:23:46]` "每个voxel的radar feature把它对应的给它填充到我们BV上的一个," `[00:23:52]` "出示好的一个BV的feature上去,"（"出示好"=初始化好）

- 【直译】对每个 pillar：查 coords_batch 得知它属于哪帧、落在画布哪行哪列，把它的 64 维特征写进那一格。N 条记录写完，一批稀疏点就"显影"成 bs*3 张稠密特征图。
- 【代码】向量化实现（帧索引列在此兑现价值，呼应卡 3-14）：
  ```python
  frame_idx, y_idx, x_idx = coords_batch[:,0], coords_batch[:,2], coords_batch[:,3]
  canvas[frame_idx, :, y_idx, x_idx] = feature   # [N,64] 一次性散射
  ```
  高级索引在 GPU 上是单个 scatter kernel，无循环。⚠列序 (z,y,x) 取第 2、3 列为 y、x 是按 mmdet3d 惯例推断，实际下标映射（含 y*W+x 还是 x*H+y）需回代码核对。
- 【形状】`[N,64] + [N,4] → [bs*3, 64, 448, 224]`。信息守恒：N 个格子有值、其余为零；若两 pillar 坐标相同则后写覆盖前写（体素化保证同帧坐标唯一，不会发生）。
- 【为什么梯度能通】scatter 的高级索引赋值是可微操作（反向时把画布上对应格子的梯度 gather 回 feature），所以 Linear 的 640 个参数能收到来自检测 loss 的梯度——整条 radar 支路端到端可训。
- 【连接】这一步与 Ch7 将讲的 LSS "拍平" 异曲同工：都是把带坐标的稀疏特征砸进 BEV 网格，LSS 用 cumsum/bev_pool 处理多点同格的求和，这里 radar 一格一 pillar 直接赋值——先在简单版上建立直觉，后面 LSS 就不难了。

---

### 卡 3-29 ｜输出形状：(bs*3)×64×448×224（⚠转写"3Z""48"勘误）

> **原话** `[00:23:56]` "然后就变成了batch size乘3," `[00:24:00]` "3是我们要用到3Z,"（⚠"3Z"=三帧的误听） `[00:24:03]` "然后64," `[00:24:04]` "48以及224的一个radar的一个feature,"（⚠"48"=448 掉字）

- 【直译】radar backbone 最终输出：batch_size×3 帧、64 通道、448×224 的 BEV 特征。
- 【帧证】`00_24_43.jpg` 出口边标注原文 **`(bs*3)*64*448*224`** ——转写的"3Z"与"48"由此帧一锤定音为"三帧"与"448"。
- 【形状】`[bs*3, 64, 448, 224]`，调试时 `[3, 64, 448, 224]`。与 lidar 的 `[3, 64, 352, 224]` 恰好通道同、宽同、只有前后向长度不同——为下一章 RL 融合的空间对齐埋下伏笔。
- 【为什么 64 通道】再次强调对齐设计：lidar 64 维（DataLoader 拍平决定）、radar 64 维（PFN Linear 输出决定），两条支路在通道维上说同一种语言，融合时 concat/add 都顺手。
- 【连接】把帧折在 batch 维的输出布局与图像分支 21 路、lidar 3 路一致（卡 3-3），三条支路在"时序折叠约定"上完全统一——后面 MemoryManager/时序融合模块 reshape 回 `[bs,3,…]` 时才不会乱。

---

### 卡 3-30 ⭐重点 ｜radar 为什么是 448：后向从 45.4 m 扩到 83.8 m

> **原话** `[00:24:12]` "然后我们现在48和224,"（=448和224） `[00:24:15]` "Ladar所用到的35224,"（⚠"35224"=352和224 连读误听；Ladar=lidar） `[00:24:19]` "其实是不太一样的," `[00:24:22]` "48其实前向范围是一样的," `[00:24:24]` "主要是我们后向扩充了一下后向的远距离," `[00:24:29]` "就是从后向的45.4扩充到了后向的过83.8,"

- 【直译】radar 的 448×224 与 lidar 的 352×224 不同：前向都是 95.4 m 没变，差别全在后向——radar 把后向感知距离从 45.4 m 拉到了 83.8 m。
- 【形状】验算（0.4 m/格）：(95.4 + 83.8) = 179.2 m ÷ 0.4 = **448** ✓；多出来的 448−352=96 格 = 38.4 m = 83.8−45.4 ✓，全部严丝合缝——扩的每一米都长在后向。
- 【为什么后向要 83.8】高速变道场景：后方快车以 30+ m/s 的相对速度接近，45 m 的后向感知只给你 1.5 秒反应窗，不够做安全变道决策；~84 m 能把窗口拉到近 3 秒。而这个任务恰好只有 radar 能便宜地干——radar 测距远（200 m+）、直接出径向速度、对金属车体敏感；lidar 后向远距点稀疏且贵（要靠后置高线束）。**传感器各按其长处定制感知范围**，这是量产 BEV 系统与学术数据集"统一正方形范围"的典型差别。
- 【连接】nuScenes 上的 BEVFusion 用对称 [-54,54]，从不给单模态开小灶；DenseBEV 这种"radar 独享加长后厨"的设计意味着后续 RC 融合/检测头要处理**不同支路视野不一致**的问题——出口那个 `crop` 块（帧 `00_24_43.jpg`，pts_middle_encoder 之后的黄色小框）就是干这个的：⚠推断它把 448 裁回与 lidar/RL 特征对齐的窗口（或按 `rear_far_crop_radar_feature` 配置裁出远后向单独用，代码帧 `00_14_41.jpg` 末行恰好露出这个配置名），视频本段未讲，待后续章节/代码核实。

---

### 卡 3-31 ｜左右不变：224 收尾

> **原话** `[00:24:34]` "所以说BV的范围就是从352变成了48,"（=448） `[00:24:38]` "然后左右还是一样的," `[00:24:41]` "就是变成了还是224的一个BV的一个范围。" `[00:24:49]` "这个是整个radar的一个处理特性的一个过程,"（本章收束句；其后 00:25:00 起"再结合代码讲一下"属下一章）

- 【直译】总结形状差异：前后向 352→448，左右维持 224。至此 radar 的框图流程讲完，讲者宣布接下来回代码逐行对照（Ch4 内容）。
- 【形状】左右 ±44.8 m 不动的原因：横向感知需求由车道决定（左右各 3~4 条车道 ≈ 15 m 已足够，44.8 m 已很富余），无论哪个传感器都不需要更宽；扩宽只会稀释显存。
- 【为什么此处收束】本章至此完成两条支路：lidar `[3,64,352,224]`、radar `[3,64,448,224]`，双双站在 BEV 网格上、64 通道对齐，只差一个 crop 就能与图像分支的 BEV 特征会师——下一章 RL 融合 UNet 就吃这两个张量。
- 【连接】用户可对照自己 BEVFusion 训练配置里的 `point_cloud_range`/`grid_size` 做一遍同样的三行验算（练习 ch3-1 尾部），这是检查任何 BEV 配置自洽性的最快手段。

---

### 🔨 动手练习 ch3-6：scatter 成 radar BEV 伪图像并数一数"星星"

```python
import torch

bs3, C, H, W = 3, 64, 448, 224
N = 2407
feature = torch.randn(N, C)                        # PFN输出 [N,64]
coords_batch = torch.stack([
    torch.randint(0, bs3, (N,)),                   # 帧索引（卡3-14 pad进来的）
    torch.zeros(N, dtype=torch.long),              # z恒0
    torch.randint(0, H, (N,)),                     # y: 前后向448格
    torch.randint(0, W, (N,)),                     # x: 左右224格
], dim=1)

canvas = torch.zeros(bs3, C, H, W)                 # 初始化BEV画布（卡3-27）
f, y, x = coords_batch[:, 0], coords_batch[:, 2], coords_batch[:, 3]
canvas[f, :, y, x] = feature                       # 一行scatter（卡3-28）

print(canvas.shape)                                # 预期: torch.Size([3, 64, 448, 224])
occ = (canvas.abs().sum(1) > 0).float()            # [3,448,224] 非零格
print(occ.sum(dim=(1, 2)))                         # 预期: 每帧约800上下(随机坐标少量撞格)
print(occ.mean().item())                           # 预期: ~0.008 —— 99%以上是零,radar BEV是星空图

# 验证范围换算（卡3-30）
print((95.4 + 83.8) / 0.4)                         # 预期: 448.0
print((83.8 - 45.4) / 0.4)                         # 预期: 96.0 —— 448-352, 全扩在后向
```

**【小结】** ① pts_middle_encoder 先零初始化 bs*3 张 64×448×224 画布，再按 coords_batch 的（帧,y,x）把 N 条 64 维 pillar 特征散射进格子，稀疏点集就地"显影"为稠密 BEV 伪图像；② radar 独享 448：前向 95.4 与 lidar 相同，后向 45.4→83.8 专为后方远距来车加长，左右 224 不变，所有数字被 0.4 m 分辨率严格锁定；③ 出口 crop 块负责把加长版画布与其余支路对齐（细节待后续章节），至此 lidar/radar 双支路 64 通道 BEV 特征就位，等待 RL 融合。

---

## 本章总收束

**一张图记住本章**（对应帧 `00_19_45.jpg` 的 draw.io 全景）：

```
lidar:  input (bs*3)×352×224×64 ──permute──▶ (bs*3)×64×352×224      【零参数】
radar:  voxels N×16×10 ─┐
        coords N×3 ──pad帧索引──▶ coords_batch N×4 ─────────────┐
        num_points N ──get_paddings_indicator──▶ mask N×16×1    │
                         voxels*mask ▶ feature N×16×10           │
                         Linear(10→64)+Norm+ReLU ▶ N×16×64       │
                         torch.max(x,dim=1) ▶ feature N×64 ──────┤
        voxel_num (bs*3) ─(分段信息已并入帧索引)                 ▼
                    pts_middle_encoder: scatter ▶ (bs*3)×64×448×224 ▶ crop
```

三个必须带走的认知：**其一**，DenseBEV 把 lidar 编码离线化（DataLoader 拍平、网络零参数透传）、radar 编码在线化（迷你 PointPillars），策略由点云密度决定——与 BEVFusion 的 lidar 在线 voxelize 恰成镜像；**其二**，"变长→定长(16槽)→mask消毒→共享MLP→max池化→scatter显影"是稀疏点云特征提取的完整闭环范式，radar 这条 640 参数的小支路是理解 PointPillars/BEVFusion lidar 支路的最佳沙盘；**其三**，所有 BEV 形状都被"物理范围÷0.4 m"锁死（352=140.8/0.4，448=179.2/0.4，224=89.6/0.4），radar 后向独扩到 83.8 m 是为高速后方来车定制——形状对不上时先查范围配置，这是排查 BEV 代码的第一反射。


---
> [[Ch02_DepthGT与DepthLoss|← Ch2]] · [[00_总览与脉络|📖 总览]] · [[Ch04_Radar代码实走与远距离切分|Ch4 →]]

