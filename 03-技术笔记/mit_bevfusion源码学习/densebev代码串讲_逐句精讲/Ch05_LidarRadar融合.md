> [[Ch04_Radar代码实走与远距离切分|← Ch4]] · [[00_总览与脉络|📖 总览]] · [[Ch06_LSS投影输入准备|Ch6 →]]

# Ch5 Lidar-Radar(RL) 融合（00:31:13–00:36:01）

> **本章在全局地图的位置**
>
> ```
> FPN收尾 → DepthNet → Depth Loss → Lidar Backbone(透传) → Radar Backbone(pillar编码)
> → ★你在这里：RL融合(UNet)★
> → LSS投影 → 多视角融合 → RC融合 → 模态融合 → MemoryManager → 时序融合
> → BEV UNet backbone → CenterPoint检测头 → Loss → Box解码
> ```
>
> 上一章（Ch4）里，lidar 分支把点云 scatter 成了稠密 BEV 画布、radar 分支把 pillar 编码后的特征也 scatter 成了一张 **448×224 的完整 BEV 画布**，并在 00:30:30 附近用一个 crop 操作把它切成"前向 352 行 + 后向远距离 96 行"两块。本章讲的就是这三路特征（lidar、radar、后向远距离 radar）如何在 `LidarNetsSingle.forward_dense()` 里汇成一条 BEV 特征流：先各过一层 64→32 的"第一卷积"（这是 DDR 优化后的写法，优化前是 concat 后过三层卷积），再用"沿高度维 concat 补齐 + 逐元素加"的方式融合，最后送进一个 UNet backbone，吐出 448×224 全尺度融合特征和 224×112 半尺度特征两路输出。
>
> **本章帧证清单**（精读单帧 9 张 + 总览 sheet 3 张）：
> `00_31_26.jpg`（forward_dense 下半段代码+调试台 shape）、`00_31_44.jpg`／`00_31_55.jpg`（draw.io 数据流图：crop→radar_feature/crop_rear_radar_feature→LidarNetsSingle）、`00_32_07.jpg`（forward_dense 全函数，216 行高亮）、`00_32_22.jpg`（讲者用鼠标框选优化后 else 分支 228–237 行）、`00_33_31.jpg`（radar_feature 悬停检查：cuda、float32、SliceBackward）、`00_35_25.jpg`（parsing_embedding 悬停检查：ReluBackward1、ndim=4）、`00_35_46.jpg`／`00_36_00.jpg`（242 行 return 高亮；标签页显示 unet.py 属于 lidar_backbone）。

---

## 本章代码全景（从帧 00:32:07 / 00:32:22 / 00:35:46 逐行转录）

文件路径（帧 00:36:00 悬停可见）：`.../e2e/tasks/bev_task/uvp_module/models/lidar_nets/lidar_nets.py`，类 `LidarNetsSingle(BaseModule)`。调试是在 `e2e/uvpsandbox/solver.py` 的 Solver 里打断点单步进来的。

```python
# lidar_nets.py — class LidarNetsSingle(BaseModule)（行号照抄画面）
215  def forward_dense(self, *inputs, **kwargs):
216      lidar_feature = inputs[0]
217      radar_feature = inputs[1]
218      rear_far_radar_feature = inputs[2]
219
220      if not self.dense_extractfeat_rm_relu_merge_conv:      # ← 优化前的老路径
221          if rear_far_radar_feature is not None:
222              lidar_feature = torch.cat([lidar_feature, rear_far_radar_feature], dim=2)
223              radar_feature = torch.cat([radar_feature, rear_far_radar_feature], dim=2)
224
225          parsing_embedding = torch.cat([lidar_feature, radar_feature], dim=1)
226          parsing_embedding, feat_reciprocal_2nd = self.lidar_backbone(
                 parsing_embedding, return_reciprocal_2nd=True)
227      else:                                                   # ← DDR 优化后的新路径（实际走这里）
228          radar_feature_embedding = self.lidar_backbone.inc.forward_radar_first_conv(radar_feature)
229          lidar_feature_embedding = self.lidar_backbone.inc.forward_img_first_conv(lidar_feature)
230
231          if rear_far_radar_feature is not None:
232              radar_feature = torch.cat([radar_feature, rear_far_radar_feature], dim=2)
233
234              rear_far_radar_feature_embedding = \
                     self.lidar_backbone.inc.forward_radar_first_conv(rear_far_radar_feature)
235
236              radar_feature_embedding = torch.cat(
                     [radar_feature_embedding, rear_far_radar_feature_embedding], dim=2)
237              lidar_feature_embedding = torch.cat(
                     [lidar_feature_embedding, rear_far_radar_feature_embedding], dim=2)
238
239          parsing_embedding = lidar_feature_embedding + radar_feature_embedding
240          parsing_embedding, feat_reciprocal_2nd = self.lidar_backbone.forward_no_inc(
                 parsing_embedding, return_reciprocal_2nd=True)
241
242      return parsing_embedding, radar_feature, feat_reciprocal_2nd

244  def forward(self, *inputs, **kwargs):
246      if self.forward_type == 'default':
247          return self.forward_default(*inputs, **kwargs)
248
249      if self.forward_type == 'dense':
250          return self.forward_dense(*inputs, **kwargs)
```

调试控制台（帧 00:32:07 左下角，讲者手敲的三条 shape 查询）：

```
> inputs[0].shape
torch.Size([3, 352, 224, 64])        # lidar 特征（注意：通道在最后一维！见⚠5.1）
> radar_feature.shape
torch.Size([3, 64, 352, 224])        # radar 前向特征（NCHW）
> crop_rear_radar_feature.shape
torch.Size([3, 64, 96, 224])         # 后向远距离 radar（NCHW）
```

draw.io 数据流图（帧 00:31:44 / 00:31:55）：

```
self.pts_middle_encoder ──(bs*3)*64*448*224──▶ [crop] ─┬─▶ radar_feature           (bs*3)*64*352*224 ─┐
                                                       └─▶ crop_rear_radar_feature (bs*3)*64*96*224  ─┴─▶ LidarNetsSingle
```

这张图坐实了两件事：① radar scatter 出来的原始画布就是完整 BEV 网格 448×224，前 352 行与后 96 行是**同一张画布切开的**；② 图上明确标注 batch 维是 `bs*3`——3 帧时序被折叠进 batch 维一起前向（和 Ch2 里 DepthNet 的 21=3帧×7相机 是同一套"折 batch"手法）。

---

## Part 5.1 模块入口与三路输入（00:31:13–00:32:07）

**导读**：本段是 RL 融合模块的"开场点名"。输入是三个张量——lidar 稠密 BEV 特征（64 通道、352×224）、radar 前向 BEV 特征（(bs·3)×64×352×224）、后向远距离 radar 特征（(bs·3)×64×96×224）；它们分别以 `inputs[0]/[1]/[2]` 的顺序传进 `forward_dense`。输出要到本章末尾才出现：448×224 融合特征 + 224×112 半尺度特征 + 448×224 纯 radar 特征三元组。在流水线上，它承接 Ch4 的两个模态 backbone，为后面 LSS 投影完成后的"模态融合"备好点云侧的 BEV 底料。

---

### 卡 5.1-1 进入 RL 融合模块

> **原话** `[00:31:13][00:31:16][00:31:23]`（三行合并，讲者原话重复了一遍"lidar和radar融合的模块"）：
> "然后接下来的话就是我们 lidar 和 radar 融合的一个模块。"

- 【直译】上一个模块（radar backbone 的 crop 收尾）讲完了，现在镜头切到 lidar 与 radar 两种点云类模态在 BEV 平面上合流的模块。
- 【代码】入口是 `LidarNetsSingle.forward()` 的类型分发：`forward_type == 'dense'` 时走 `forward_dense(*inputs)`（帧 00:31:26 中 249 行正高亮着 `if self.forward_type == 'dense':`）。也就是说这个类同时保有老的稀疏路径 `forward_default` 和 DenseBEV 用的稠密路径 `forward_dense`，用一个字符串开关切换——这是"一份代码养两代模型"的典型工程写法。
- 【为什么】为什么 lidar 和 radar 先融、而不是三模态一起融？因为这两者天然都在自车坐标系的 BEV 平面上（scatter 完就是 448/352×224 网格），空间上直接对齐，融合只需要对齐 shape；而相机特征还要等 LSS 投影才能到 BEV。先把"便宜"的融合做掉，是按坐标系归属安排融合顺序。
- 【连接】对照你熟的 BEVFusion：它是 camera BEV 与 lidar BEV 两路 concat 后过 ConvFuser；DenseBEV 多了 radar，且把 radar 塞到 lidar 支路里先融（RL），camera 稍后在 RC 融合、模态融合两站再进来。可以把 `LidarNetsSingle` 理解成 BEVFusion 里 "fuser + 部分 bev backbone" 的点云侧前半段。
- 【类名彩蛋】类名叫 **LidarNets**Single，却干着 lidar+radar 融合的活——命名滞后于功能演进，读工作代码时要认变量流向、别认名字（下文 `forward_img_first_conv` 会再撞见一次同款彩蛋）。

---

### 卡 5.1-2 输入有三个

> **原话** `[00:31:26][00:31:27][00:31:31]`（合并，"它的输入呢"重复两次）：
> "它的输入呢是有这三个。"

- 【直译】这个模块吃三份数据，不是常见的"一对儿"融合。
- 【代码】216–218 行逐行解包：`lidar_feature = inputs[0]`、`radar_feature = inputs[1]`、`rear_far_radar_feature = inputs[2]`。用 `*inputs` 位置解包而不是关键字参数，意味着调用方（solver 里的上层 forward）必须严格按 0/1/2 的约定顺序传——顺序即接口。
- 【形状】三者 shape 见调试台实录：`[3,352,224,64]`、`[3,64,352,224]`、`[3,64,96,224]`（bs=1，3 帧折叠在 batch 维）。
- 【为什么】radar 为什么被拆成两个输入而不是一整张 448×224？因为 lidar 只覆盖前向 352 行，融合主干只能在 352×224 上做对齐运算；后向 96 行是"radar 独有区"，需要单独一条支路走特殊处理（Part 5.3 的沿高度 concat）。拆开传，等于把"共有区/独有区"的边界在接口层就划清了。

---

### 卡 5.1-3 输入一：lidar 特征 64×352×224 ⚠（重点句，5 角度）

> **原话** `[00:31:32][00:31:36][00:31:39]`：
> "分别是我们 lidar Backbone 出来的 lidar 的一个特征，就是 64 乘以 352 乘以 224。"
> （⚠ 校正：原转写此处"24"均为"224"之略，本章统一还原成 224。）

- 【直译】第一路输入是 Ch4 讲过的 lidar 稠密 backbone（实际是"透传"式 scatter）输出的 BEV 特征图：64 个通道，BEV 网格前向 352 行 × 横向 224 列。
- 【代码】`lidar_feature = inputs[0]`。它的上游是 Ch4 的 voxelize→pillar/voxel 编码→scatter 到稠密画布，无重卷积（所谓"透传"），所以 64 通道还是点特征编码器给的原始通道数。
- 【形状】讲者口述 `64×352×224`（CHW 顺序）；但调试台实打实显示 `inputs[0].shape = torch.Size([3, 352, 224, 64])`——**通道在最后一维（NHWC）**，且 batch 维是 3（=1 个 batch × 3 帧时序）。⚠ 存疑：229 行紧接着就把它喂给 `forward_img_first_conv`（内部是 Conv2d，要求 NCHW），所以要么该封装函数内部先 `permute(0,3,1,2)`，要么调试打印的时刻在 permute 之前。讲者按逻辑通道序口述、调试台按物理内存序显示，两者都"对"，但读代码时必须分清。scatter 类操作天然产出 NHWC（按 (y,x) 索引往画布上撒 C 维向量），这也旁证了"透传"的说法。
- 【为什么】为什么 lidar 只有 352 行而不是完整 448 行？按 0.4 m 分辨率换算：BEV 纵向全长 95.4+83.8=179.2 m ÷ 0.4 = 448 行；352 行 = 140.8 m，等于"前向 95.4 m + 后向 45.4 m"。lidar 对后向超过约 45 m 的目标点数稀少、置信度低，索性不铺画布，省下 96 行 × 224 列的内存和算力；这 38.4 m 的后向远距离带完全交给 radar（radar 测距远、对金属目标稳定）。
- 【连接】BEVFusion 里 lidar BEV 特征是全范围一张图（nuScenes ±54 m 正方形），不存在"前后覆盖不对称"；DenseBEV 这种"前长后短 + 后向补 radar"是量产车传感器配置（前向主 lidar）逼出来的真实工程设计，这正是工作代码和论文代码的差别所在。

---

### 卡 5.1-4 输入二：radar 前向特征 (bs·3)×64×352×224（重点句，5 角度）

> **原话** `[00:31:41][00:31:42][00:31:44]`：
> "然后以及 radar 过来的一个特征，然后是 batch size×3 乘以 64 乘以 352 乘以 224。"

- 【直译】第二路输入是 radar backbone（pillar 编码 + scatter）输出的前向 BEV 特征，batch 维明确写成 batch_size×3。
- 【代码】`radar_feature = inputs[1]`；上游是 Ch4 末尾的 crop：从完整画布 `(bs*3)×64×448×224` 里切出前 352 行（draw.io 图上 crop 节点的上分支）。帧 00:33:31 悬停显示它 `grad_fn=<SliceBackward>`——**切片反传**！坐实了它是切片而来、且梯度能穿过 crop 流回 radar backbone。
- 【形状】`[bs*3, 64, 352, 224]`，调试时 bs=1 故为 `[3,64,352,224]`。这里的 3 是**时序 3 帧折叠进 batch**：radar 和 lidar 一样按 3 帧各自独立前向，直到时序融合那站才 warp 对齐相加。与 DepthNet 的 21=3×7 同源。
- 【为什么】把 3 帧折进 batch 而不是循环 3 次：GPU/NPU 上大 batch 一次卷积远比 3 次小卷积高效（核启动开销、并行度），而且权重共享天然成立——同一个 backbone 处理任意时刻的帧。
- 【连接】radar pillar 编码是 PointPillars 思路（nuScenes 上 radar 点只有百级数量，pillar 化后 scatter 极稀疏）；BEVFusion 官方版没用 radar，你可以对照 RCBEVDet/CRN 这类 radar-camera 工作理解 radar BEV 特征的稀疏性——这也是后面 RL 融合敢用"逐元素加"的底气之一：radar 画布大部分格子是 0，加法几乎不污染 lidar 特征。

---

### 卡 5.1-5 输入三：后向远距离 radar (bs·3)×64×96×224（重点句，5 角度）

> **原话** `[00:31:48][00:31:50]`：
> "然后以及后向远距离的 radar，就 batch size×3 乘以 64 乘以 96 乘以 224。"

- 【直译】第三路输入是同一张 radar 画布上切出来的**后向最远 96 行**，专门表示 lidar 覆盖不到的车尾远区。
- 【代码】`rear_far_radar_feature = inputs[2]`；draw.io 图上它叫 `crop_rear_radar_feature`（crop 节点的下分支），调试台变量名也是这个——进入 `forward_dense` 后改名为 `rear_far_radar_feature`。同一个张量在不同作用域两个名字，读调试记录时要能对上号。
- 【形状】`[3, 64, 96, 224]`。96 行 × 0.4 m = **38.4 m** 的后向远距离带（自车后方约 45.4 m 到 83.8 m 之间）。352+96=448 正好拼回完整 BEV 网格。
- 【为什么】为什么切出来单独传而不是让 lidar 也铺满 448 行（后向补零）？两个原因：① lidar 画布少铺 96 行，scatter 和后续第一层卷积都省 21% 的行数；② 更关键的是语义诚实——lidar 在那个区域不是"值为 0 的观测"，而是"没有观测"，硬补零会让网络把"无观测"学成"无目标"。DenseBEV 的做法（Part 5.3）是用 radar 自己的 embedding 去填那 96 行，让填充值携带真实观测。
- 【连接】这是"传感器视场（FOV）不对称"问题的 BEV 处理范式，你在华为车 BU 见过的前向主 lidar + 角雷达布局就是它的物理来源；论文复现（nuScenes 32 线全向 lidar）里遇不到，面试聊量产落地时这是很好的谈资。

---

### 卡 5.1-6 按索引 0/1/2 取输入

> **原话** `[00:31:59][00:32:02][00:32:04]`（合并）：
> "然后在这里会分别取出索引为 0、1、2 对应的 lidar、radar、后向远距离 radar 的一个 feature。"

- 【直译】函数开头三行就是把 `*inputs` 元组按位置拆成三个具名变量。
- 【代码】216–218 行（帧 00:32:07 中 216 行 `lidar_feature = inputs[0]` 正被黄条高亮——调试器停在这里，说明讲者是**边单步边讲**的，后面每个 shape 都是现场査出来的，可信度高）。
- 【形状】拆包不改 shape，只是给三个张量起名。
- 【连接】`*inputs` + 手工索引，而不是 `forward(self, lidar_feature, radar_feature, rear_far_radar_feature)`——mmdet/mmcv 系工作代码常这么写，因为上层用统一的 `net(*feats)` 调不同模块；代价是 IDE 无法提示参数含义，只能靠调试或读上游拼装代码。你以后接手这类代码，第一件事就是像讲者一样在入口打断点把 `inputs[i].shape` 全打一遍。

---

### 🔨 动手练习 ch5-1：复现 crop——一张 448×224 画布切成前 352 / 后 96

```python
import torch

bs, T, C = 1, 3, 64                       # batch=1, 3帧折叠, 64通道
canvas = torch.randn(bs * T, C, 448, 224)  # radar scatter 出的完整 BEV 画布

# 假设 行0..351 是"lidar 也覆盖的区域"，行352..447 是"后向远距离带"
radar_feature          = canvas[:, :, :352, :]   # 前向切片
crop_rear_radar_feature = canvas[:, :, 352:, :]  # 后向远距离切片

print(radar_feature.shape)            # torch.Size([3, 64, 352, 224])
print(crop_rear_radar_feature.shape)  # torch.Size([3, 64, 96, 224])

# 验证"切片可反传"（对应帧00:33:31里 grad_fn=SliceBackward）
canvas.requires_grad_(True)
loss = canvas[:, :, :352, :].sum()
loss.backward()
print(canvas.grad[:, :, :352, :].abs().sum() > 0,   # tensor(True)  前352行有梯度
      canvas.grad[:, :, 352:, :].abs().sum() == 0)  # tensor(True)  后96行无梯度
# 换算米数：96*0.4=38.4m 后向远距离带；352*0.4=140.8m 共视区
```

**【小结】** ① RL 融合模块入口 `forward_dense` 按 0/1/2 位置约定接收 lidar(64×352×224)、radar((bs·3)×64×352×224)、后向远距离 radar((bs·3)×64×96×224) 三路 BEV 特征，3 帧时序折叠在 batch 维。② radar 的两块是同一张 448×224 画布 crop 出来的，切片保梯度（SliceBackward）；352/96 的分界来自 lidar 前向覆盖 140.8 m、后向 38.4 m 远区只有 radar。③ 调试台显示 lidar 特征物理布局是 NHWC（[3,352,224,64]），与口述的 CHW 逻辑序不同，读 shape 时要分清（⚠详见存疑清单）。

---

## Part 5.2 DDR 优化：老"concat+三层卷积" vs 新"单层卷积各自过"（00:32:07–00:33:15）

**导读**：本段是全章工程含金量最高的部分。代码里留着一个开关 `dense_extractfeat_rm_relu_merge_conv`（rm_relu=去掉 ReLU，merge_conv=合并卷积）：关掉时走老路径——lidar/radar 沿通道 concat 成 128 通道再进带三层卷积的 UNet 输入块（inc），DDR 占用大；打开时走新路径——lidar、radar **各自**过一层 64→32 的卷积再逐元素加，从三层卷积+若干 ReLU 精简成一层卷积。输入是 Part 5.1 的三路特征，输出是两路 32 通道的 embedding，供 Part 5.3 拼接融合。理解这段的钥匙是一条数学恒等式：`conv(concat(a,b)) ≡ conv_a(a) + conv_b(b)`。

---

### 卡 5.2-1 优化前这里 DDR 会比较大

> **原话** `[00:32:07][00:32:11][00:32:12][00:32:14][00:32:17]`（合并，含口误"Rio"）：
> "然后这个呢是之前没有做那个 RL backbone 优化的，就是这里在 DDR 上会比较大。"
> （⚠ 校正：转写"Rio/没有做Rio"应为 **RL**（Radar-Lidar）backbone 优化。）

- 【直译】屏幕上 220–226 行的 if 分支是优化前的老实现，它在 DDR（内存带宽/占用）上开销大。
- 【代码】老路径核心两行：`parsing_embedding = torch.cat([lidar_feature, radar_feature], dim=1)`（225 行，64+64=128 通道）→ `self.lidar_backbone(parsing_embedding, return_reciprocal_2nd=True)`（226 行，**完整** UNet 前向，含三层卷积的 inc 输入块）。
- 【形状】concat 后的中间张量 `[3, 128, 448, 224]`：128×448×224×3 ≈ 3850 万元素，fp32 下约 **154 MB**、fp16 下约 77 MB——这一坨要先写入 DDR，再被 inc 的第一层卷积读回来，之后每多一层卷积/ReLU 又是一轮"写出去、读回来"。
- 【为什么】DDR 是车载 NPU（MDC/昇腾类平台）的命门：算力（TOPS）常常富余，瓶颈在片外内存带宽。conv→ReLU→conv 这种链条如果编译器融合不掉，每个中间特征图都要走一遍 DDR 往返；在 448×224 这么大的 BEV 平面上，多一层就是上百 MB 的搬运。所以部署优化的第一刀往往不是砍 FLOPs，而是砍**中间特征图的字节数与层数**。
- 【连接】你 4060 上跑 BEVFusion 感受到的 0.481 s/iter 主要是算力瓶颈；换到车端 NPU，同一个网络的耗时排序会重排——这就是"训练侧 profile"和"部署侧 profile"的区别。面试谈部署优化，能把"DDR bound vs compute bound"讲清楚是加分项。

---

### 卡 5.2-2 现在看到的是优化之后的 backbone

> **原话** `[00:32:21][00:32:24]`：
> "然后这个是我们优化之后的一个 lidar 的一个 backbone。"（⚠ 转写"修化"=优化。）

- 【直译】接下来讲的 else 分支（227–240 行）才是当前实际在跑的版本。
- 【代码】开关是 `self.dense_extractfeat_rm_relu_merge_conv`（220 行取反判断）。帧 00:32:22 里讲者正用鼠标把 228–237 行整块蓝选，肢体语言等于说"看这一段"。
- 【为什么】优化不删老代码而是留 if/else 双路径：① 老 checkpoint / 老配置还要能复现；② A/B 对比精度时可一键切换。量产仓库里这种"开关化重构"极常见，代价是读者必须先搞清哪个分支是活的——讲者特意强调"实际走优化后的"，就是帮你排雷。
- 【连接】flag 命名即变更日志：`rm_relu`（去 ReLU）+ `merge_conv`（并卷积）+ `dense_extractfeat`（稠密版特征提取），一个变量名浓缩了整场优化的三个动作。

---

### 卡 5.2-3 优化前：三层卷积，64→128→…→32（重点句，5 角度）⚠

> **原话** `[00:32:27]–[00:32:47]`（合并 436–444 共 9 行，讲者围绕"三层Cover"车轱辘了三遍）：
> "在优化之前呢，是直接把我们的 lidar feature——就是 lidar 的一个 backbone——其实它在这里会有三层卷积。没有优化之前会有三层卷积，会把对应的从 64 变成 128，然后后面还会再把 128 变成我们的一个 32 的 Channel。"
> （⚠ 转写"Cover"=卷积 conv。）

- 【直译】老版本里 UNet 的输入块（inc）由三层卷积组成，通道走 64→128→（128）→32 的胖中间路线。
- 【代码】对应 220–226 行老路径：concat 出 128 通道后进 `self.lidar_backbone(...)` 的完整前向，其 `inc` 成员就是那三层卷积（inc 这个名字来自经典 UNet 开源实现 milesial/Pytorch-UNet 里的 `self.inc = DoubleConv(...)`，"input conv"之意；帧 00:36:00 标签页可见 `unet.py .../lidar_backbone`，坐实 backbone 就是 UNet）。
- 【形状】按讲者口径的中间通道：`[3,128,448,224]`(concat) → conv1 → `[3,128,...]` 量级的胖中间层 → … → `[3,32,448,224]`。⚠ 存疑：讲者说"从 64 变成 128"，但代码里进 inc 之前 concat 已是 128 通道，口述的通道序列（64→128→128→32？）与代码事实（128 进、32 出、共三层）在细节上对不拢；确切序列要看 `unet.py` 里老版 inc 的定义。可确认的硬事实是：**老路径三层卷积 + 中间若干 ReLU，新路径一层卷积、零中间 ReLU**。
- 【为什么】老写法是学术代码的标准范式（BEVFusion 的 ConvFuser 同款：concat 两模态 → 卷积压通道），先胖后瘦的通道设计给融合层足够容量。它在 GPU 训练时无感，但上车后三层大分辨率卷积的 DDR 往返成了热点——于是有了下一卡的合并。
- 【连接】把它跟你在智谷课程学的"1×1 卷积做通道变换"连起来：inc 干的本质就是"跨模态通道混合"，三层和一层的差别是非线性容量，而非空间感受野（下一卡会证明第一层的合并在数学上是无损的）。

---

### 卡 5.2-4 优化后：一层卷积直接 64→32，少了两层

> **原话** `[00:32:48][00:32:53]`：
> "然后这里优化之后呢，会是直接从 64 变成 32，中间就少了两层卷积。"

- 【直译】新版本每个模态只过一层卷积，通道直接 64→32，砍掉了两层。
- 【代码】228–229 行：`radar_feature_embedding = self.lidar_backbone.inc.forward_radar_first_conv(radar_feature)`；`lidar_feature_embedding = self.lidar_backbone.inc.forward_img_first_conv(lidar_feature)`。注意不再 concat！两个模态**各自**过各自的第一卷积，融合动作推迟到 239 行的加法。
- 【形状】`[3,64,352,224] → [3,32,352,224]`（两路各一份）。中间张量从老路径的 128 通道 154 MB 降到 32 通道 ×2 份 ≈ 77 MB，且少了两轮层间 DDR 往返。
- 【为什么】关键数学：对 concat 后的第一层卷积，权重可按输入通道切两半，`conv([a;b]) = conv_a(a) + conv_b(b)`（卷积对输入通道是线性求和的，bias 归并到任一支）——所以"每模态一个 64→32 卷积 + 相加"与"concat 128 → 一个 128→32 卷积"**严格等价**（练习 ch5-2 会数值验证）。真正有损的是砍掉后两层卷积和中间 ReLU，那部分是容量删减，需要重训并用精度回归来兜底。
- 【连接】`forward_img_first_conv` 处理的是 **lidar**，名字里却是 img——又一个命名化石（这套 UNet 早年多半吃过图像 BEV 特征）。工作代码考古学第二课：函数名会说谎，调用处不会。

---

### 卡 5.2-5 还去掉了中间的几个 ReLU（重点句，5 角度）

> **原话** `[00:32:55]`：
> "然后以及中间的几个 ReLU。"（⚠ 转写"Rero"=ReLU。）

- 【直译】优化不只砍卷积层，连层间的 ReLU 激活也一并去掉了。
- 【代码】flag 名里的 `rm_relu` 说的就是这个。新路径里 228/229/234 行的 first_conv 之后直接 236/237 concat、239 相加，**中间无任何激活函数**；第一个非线性被推迟到 `forward_no_inc` 内部（帧 00:35:25 悬停 `parsing_embedding` 显示 `grad_fn=<ReluBackward1>`，说明 UNet 主干里当然还有 ReLU，被删的只是 inc 内部那几个）。
- 【形状】ReLU 不改 shape，但每个独立 ReLU 层在不支持算子融合的 NPU 上就是一次全特征图的 DDR 读+写：`[3,32,448,224]` fp16 也有 ~19 MB/次，几个 ReLU 就是几十 MB 白搬。
- 【为什么】去 ReLU 有双重收益：① 省 DDR 往返；② **让合并有数学依据**——若 conv 之间夹着 ReLU，`conv2(relu(conv1(x)))` 无法折叠成单层；把 ReLU 拿掉后相邻线性层可以合并（两个线性卷积的复合仍是线性卷积），这正是 `rm_relu` 与 `merge_conv` 成对出现的原因：先去非线性，才有资格并层。
- 【连接】同族技巧你以后部署会天天见：Conv+BN 折叠、RepVGG 的多分支重参数化（训练时多支、部署时并成单 3×3）。DenseBEV 这里更激进——直接改训练图（重训），而不是训后重参数化；因为它同时还想省训练侧显存/耗时。

---

### 卡 5.2-6 下面只讲优化后的：先把 64 变成 32

> **原话** `[00:32:59][00:33:04][00:33:09]`（合并）：
> "这里就直接只讲一下优化后的。在这里优化后的呢，就是首先会直接只经过——直接就把 64 变成 32 的一个 Channel。"

- 【直译】老分支跳过不细讲；新分支第一步：lidar、radar 各自 64→32。
- 【代码】即 228–229 行两个 first_conv。它们挂在 `self.lidar_backbone.inc` 上——优化后的 inc 从"一个吃 128 通道的三层块"重构成"两个各吃 64 通道的单层卷积 + 外部加法"，UNet 其余部分（down/up 路径）原封不动，所以才有 240 行 `forward_no_inc`（跳过 inc 的前向）这个奇怪接口。
- 【形状】lidar：`[3,64,352,224]→[3,32,352,224]`（若输入确为 NHWC，则封装内先 permute，见⚠）；radar 同。
- 【为什么】把新第一卷积仍然挂在 `inc` 名下而不是新建模块，好处是老 checkpoint 的参数命名空间（`lidar_backbone.inc.*`）部分可复用、配置文件改动最小——工程重构讲究"手术切口最小化"。

---

### 🔨 动手练习 ch5-2：证明 conv(concat(a,b)) ≡ conv_a(a)+conv_b(b)（DDR 优化的数学执照）

```python
import torch, torch.nn as nn

torch.manual_seed(0)
a = torch.randn(1, 64, 32, 32)   # 假装是 lidar embedding 输入
b = torch.randn(1, 64, 32, 32)   # 假装是 radar embedding 输入

# 老写法：concat 成 128 通道，过一个大卷积
big = nn.Conv2d(128, 32, 3, padding=1, bias=True)
y_old = big(torch.cat([a, b], dim=1))

# 新写法：把大卷积的权重沿"输入通道"切成两半，各卷各的再相加
conv_a = nn.Conv2d(64, 32, 3, padding=1, bias=True)
conv_b = nn.Conv2d(64, 32, 3, padding=1, bias=False)  # bias 只留一份！
with torch.no_grad():
    conv_a.weight.copy_(big.weight[:, :64])   # 前64个输入通道的权重
    conv_a.bias.copy_(big.bias)
    conv_b.weight.copy_(big.weight[:, 64:])   # 后64个输入通道的权重
y_new = conv_a(a) + conv_b(b)

print(torch.allclose(y_old, y_new, atol=1e-5))   # True  ← 数学上严格等价
print((y_old - y_new).abs().max())               # tensor(≈1e-6) 浮点误差量级
# 结论：DenseBEV 把"concat+第一层conv"拆成"两支first_conv+逐元素加"零精度损失；
# 真正需要重训兜底的是后面被砍掉的两层conv和中间ReLU（容量变化）。
```

**【小结】** ① 老路径 `cat(dim=1)→128通道→三层conv+ReLU` 在 448×224 的 BEV 大平面上造出 154 MB 级中间张量和多轮 DDR 往返，是车端瓶颈。② 新路径由 flag `dense_extractfeat_rm_relu_merge_conv` 启用：每模态一层 64→32 的 first_conv、去掉中间 ReLU、融合改为逐元素加；其中"拆第一层卷积"有严格数学等价性，"砍后两层+ReLU"是重训过的容量取舍。③ `inc`/`forward_no_inc`/`forward_img_first_conv` 这些名字共同暴露了这套 UNet 的演化史：先有完整 UNet，后为 DDR 把输入块动了手术。

---

## Part 5.3 后向远距离 radar 补齐 + 逐元素加（00:33:15–00:34:56）

**导读**：本段解决"lidar 352 行、radar 448 行，shape 对不上怎么加"的问题。做法分三步：① 原始 radar 特征沿高度维（dim=2）concat 回 448×224，留给后面 RC 融合用；② 后向远距离 radar 也过同一个 first_conv 得到 32 通道 embedding，分别拼到 radar embedding 和 lidar embedding 的尾部——lidar 缺的 96 行用 radar embedding 补；③ 两支 448×224×32 的 embedding 逐元素相加。副作用是后向 96 行里 radar 信号被加了两次（×2 增强），讲者认为这是合理的。输入是 Part 5.2 的两路 32 通道 embedding + 原始后向 radar，输出是一张 `[3,32,448,224]` 的 parsing_embedding 和一张 `[3,64,448,224]` 的纯 radar 特征。

---

### 卡 5.3-1 原始 radar 前后两段 concat 成 448×224 纯 radar 特征（重点句，5 角度）

> **原话** `[00:33:15][00:33:20][00:33:24][00:33:26][00:33:31]`（合并）：
> "然后在这里会 concat，会把我们那个远距离的 radar——就是正常范围的 352×224 的一个 radar feature——和后向远距离的一个 radar feature concat 起来，这里会是一个 448×224 的纯 radar 的一个 feature。"
> （⚠ 转写"3524"=352×224、"48成24"=448×224、"存radar"=纯radar。）

- 【直译】把 64 通道的**原始**（未过 first_conv 的）radar 前向段和后向段沿空间高度方向接回一整张 448×224。
- 【代码】232 行：`radar_feature = torch.cat([radar_feature, rear_far_radar_feature], dim=2)`。注意是**变量重绑定**：radar_feature 这个名字从此指向 448 行的完整版，242 行 return 出去的就是它。dim=2 在 NCHW 里是 H 维——BEV 的纵向（车前后方向）。
- 【形状】`[3,64,352,224] ⊕H [3,64,96,224] → [3,64,448,224]`。此刻调试器黄条正停在 232 行（帧 00:33:31），讲者悬停查看了 radar_feature：`device cuda:0, dtype float32, grad_fn=SliceBackward`——cat 执行前它还是那张 crop 切片。
- 【为什么】既然 Ch4 里它本来就是一张 448 的画布、被 crop 成两半，这里为什么又拼回去？因为 crop 的目的只是让"lidar 共视区/radar 独有区"分开走不同的融合逻辑；等融合逻辑安排完，下游消费方（RC 融合、检测头）要的是完整 BEV 范围的 radar 观测，自然要复原。一拆一合之间，中间那段"分而治之"就是本模块的全部价值。
- 【连接】这张 448×224 的纯 radar 特征会原样穿过本模块（不参与 UNet），在流水线"RC 融合"站与相机 BEV 特征会师——见下一卡。这种"一份输入多路输出、各走各的命"的写法在多模态框架里极常见，画数据流图（像讲者的 draw.io 一样）是唯一不迷路的办法。

---

### 卡 5.3-2 纯 radar 特征留给后面的 RC（radar-camera）融合

> **原话** `[00:33:35][00:33:39][00:33:46]`（合并）：
> "这个我们后续在做那个模态融合的时候，会有一个 RC 融合的一个操作，就是 radar 和 Camera 融合。"
> （⚠ 转写"Cameral"此处=Camera；00:33:57 处的"Cameral"则=卷积，两处含义不同。）

- 【直译】刚拼好的 448×224 纯 radar 特征不是给本模块用的，是预留给后面"radar×相机"融合环节的原料。
- 【代码】它作为 `forward_dense` 返回三元组的第 2 个元素 `radar_feature` 传出（242 行），上层 solver 会把它接力传到 RC 融合模块。
- 【为什么】radar 要"入两次伙"：和 lidar 融一次（本章，几何对齐容易、都是点云系），再和 camera 融一次（RC 站，radar 的测速/远距对相机特征是强互补）。用**原始 64 通道**而非过完 first_conv 的 32 通道版本去做 RC，是为了不让 RL 融合的压缩损失污染 RC 支路——两条融合支路各自从最原始的观测出发。
- 【连接】全局地图里 "LSS投影 → 多视角融合 → RC融合 → 模态融合" 的 RC 站到时会回收这个张量；BEVFusion 没有这一站，这是 DenseBEV 针对 radar 的增量设计，听到后面章节时记得回来对账。

---

### 卡 5.3-3 后向远距离 radar 也只过一个卷积：64→32

> **原话** `[00:33:53][00:33:57][00:34:02]`（合并）：
> "然后呢这里也会同样的把我们远距离的 radar feature，也会在这里只经过一个卷积提取，就是把 Channel 从 64 变成 32。"

- 【直译】后向 96 行的 radar 独有区，同样走一层 64→32 卷积得到它的 embedding。
- 【代码】234 行：`rear_far_radar_feature_embedding = self.lidar_backbone.inc.forward_radar_first_conv(rear_far_radar_feature)`。注意用的是**同一个** `forward_radar_first_conv`（和 228 行处理前向 radar 的是同一份权重）——同一模态共享同一套第一卷积，前向/后向只是空间位置不同，物理语义相同，权重当然共享。
- 【形状】`[3,64,96,224] → [3,32,96,224]`。
- 【为什么】为什么不先把 radar 拼成 448 再一次性过卷积（还能省一次核启动）？因为 228 行执行时还走在"共视区对齐"的主线上，352 版 embedding 要先出来；后向段单独过卷积后得到的 96 行 embedding 有**双重用途**（下两卡：既补 radar 支路、又补 lidar 支路），拆开算反而复用最大化。此外 3×3 卷积在拼接缝（352/353 行交界）两侧各看 1 行邻域，分开卷会让缝上一两行感受野略有差异——工程上可忽略，但读者应知道这个细节存在。

---

### 卡 5.3-4 radar embedding 沿高度维拼回 448

> **原话** `[00:34:05][00:34:09][00:34:12][00:34:17]`（合并）：
> "然后对应的会把 352 的 radar feature 和后向 96 的沿着高度这一维给它 concat 起来，就变成了正常的 448×224 的 radar 的 feature。"
> （⚠ 转写"颜色高度这一围"=沿着高度这一维。）

- 【直译】32 通道的 radar embedding 也做和 232 行同款的高度维拼接，恢复完整 448 行。
- 【代码】236 行：`radar_feature_embedding = torch.cat([radar_feature_embedding, rear_far_radar_feature_embedding], dim=2)`。至此 radar 支路在 embedding 空间也是完整画布了。
- 【形状】`[3,32,352,224] ⊕H [3,32,96,224] → [3,32,448,224]`。
- 【连接】"高度维"这个说法容易误会成 3D 的 z 轴——在 BEV 特征图语境里 H 维是**图像意义的高度**，物理上对应自车前后方向（x 轴），千万别和 pillar 编码时被压掉的真实高度 z 混淆。BEV 新人栽在"哪个维是哪个方向"上的概率极高，建议你把 `[B, C, X前后448, Y左右224]` 写在便签上对着代码核。

---

### 卡 5.3-5 lidar 只有前向 352，为了能和 radar 融合……（动机句）

> **原话** `[00:34:20][00:34:23][00:34:26][00:34:29]`（合并）：
> "然后在这里呢，因为我们 lidar 的 feature 其实只有前向的一个 352，但是为了和 radar 能够进行沿着 Channel 维——就是能够进行融合……"

- 【直译】铺垫问题：lidar embedding 是 352 行，radar embedding 已是 448 行，行数不齐没法逐元素运算。
- 【代码】这是 237 行出场前的"痛点陈述"。讲者口中"沿着 Channel 维融合"略有口误——后面实际做的是**逐元素加**（不是 channel concat）；老路径才是 channel 维 concat。听串讲要习惯讲者在新老方案间来回横跳的表述。
- 【为什么】逐元素加的前提是两张量 shape 完全一致（PyTorch 广播也救不了 352 vs 448 的 H 维）；于是只剩两个选项：把 radar 裁短，或把 lidar 补长。裁短等于扔掉 radar 独有观测，不可接受；补长则要回答"用什么补"——下一卡给出 DenseBEV 的答案。
- 【连接】BEVFusion 不存在这个问题（两模态同画布尺寸），这是传感器 FOV 不对称在代码里砸出的真实坑；nuScenes 复现里你永远遇不到，但量产项目里"对齐两个不同覆盖范围的特征图"是日经问题。

---

### 卡 5.3-6 lidar 缺的 96 行用 radar embedding 补齐，然后逐元素加（重点句，5 角度）

> **原话** `[00:34:31][00:34:36][00:34:39][00:34:42][00:34:46]`（合并，含结巴"把就是他也把他也把他"）：
> "……在这里是直接做的一个相加。为了保证它的那个 shape 是一致的，在这里也把后向远距离的 radar feature 也沿着高度进行了一个 concat。"
> （⚠ 转写"不向"=后向。）

- 【直译】两步棋：先把 lidar embedding 的尾部拼上那 96 行 **radar** embedding 凑成 448 行，再和 radar embedding 逐元素相加完成融合。
- 【代码】237 行：`lidar_feature_embedding = torch.cat([lidar_feature_embedding, rear_far_radar_feature_embedding], dim=2)`——**拼进 lidar 支路的是 radar 的 embedding**，这是全函数最精妙的一行；239 行：`parsing_embedding = lidar_feature_embedding + radar_feature_embedding`。
- 【形状】237 行：`[3,32,352,224] ⊕H [3,32,96,224] → [3,32,448,224]`；239 行：`[3,32,448,224] + [3,32,448,224] → [3,32,448,224]`，零参数、零 FLOPs 级的融合动作。
- 【为什么】补零 vs 补 radar embedding：补零会在 352/353 行交界制造一条人工阶跃边缘（卷积最爱在这种边上产生伪响应），且向网络谎报"后向无目标"；用 radar 自己的 embedding 补，填充值携带真实观测、和相邻行分布连续，还顺手实现了"该区域没有 lidar 就全信 radar"的贝叶斯直觉。加法融合（而非 concat+conv）则是 Part 5.2 等价变换的延续——第一层卷积已经拆到两支里，加法就是那层卷积的求和项本身。
- 【连接】"缺失模态用另一模态特征占位"与 MAE 的 mask token、多模态里的 modality dropout 填充是一个思想族谱；你给面试官讲 DenseBEV 时，这行代码值得单独拎出来讲 3 分钟。

---

### 卡 5.3-7 后向 96 行里 radar 特征被增强了（×2）（重点句，5 角度）

> **原话** `[00:34:50][00:34:54]`：
> "相当于在后向 96 的一个 BEV 范围内，这里其实 radar 的 feature 是被增强了的。"

- 【直译】讲者点破副作用：后向那 96 行，lidar 支路填的是 radar embedding，radar 支路本来也是它，相加后该区域 = 2×radar embedding。
- 【代码】把 239 行按区域展开：前 352 行 = `lidar_emb + radar_emb`（真融合）；后 96 行 = `rear_radar_emb + rear_radar_emb = 2 × rear_radar_emb`（自我放大）。练习 ch5-3 用 `allclose(fused[:,:,352:], 2*rear)` 一行即可验证。
- 【形状】不变，仍 `[3,32,448,224]`；变的是后 96 行的数值幅度（×2）。
- 【为什么】这个 ×2 是 bug 还是 feature？讲者语气是当 feature 讲的："增强"。合理性：该区域只有 radar 一票，把它的响应抬高相当于告诉后端"这里信号弱但唯一，请认真对待"；而且后续紧跟 UNet 卷积 + BN 类归一化，尺度差异可被网络自适应吸收。但严格说它让前后两区的特征分布不一致（前区是两模态均衡和，后区是单模态双倍），若后端没有归一化兜底就可能学出区域偏置——这属于"重训验证过没问题"级别的工程妥协。
- 【连接】等价视角：`2×radar_emb` = 给后向区隐式乘了个区域权重图，可以类比 attention 里对稀缺信息源加权；如果哪天要精细化，把"×2"换成可学习的逐区域标量就是最小改造方案。这也是你读工作代码该养成的嗅觉——每个"顺手为之"的数值效应，都值得问一句"训练时网络怎么消化它"。

---

### 🔨 动手练习 ch5-3：复现"补齐+相加"，并抓出后向 ×2 增强

```python
import torch

C = 32
lidar_emb = torch.randn(3, C, 352, 224)   # first_conv 后的 lidar embedding
radar_emb = torch.randn(3, C, 352, 224)   # first_conv 后的 radar 前向 embedding
rear_emb  = torch.randn(3, C,  96, 224)   # first_conv 后的后向远距离 radar embedding

# 对应 236/237 行：两支都沿 H 维(dim=2)拼上后向段
radar_full = torch.cat([radar_emb, rear_emb], dim=2)   # [3,32,448,224]
lidar_full = torch.cat([lidar_emb, rear_emb], dim=2)   # [3,32,448,224] ← lidar 缺口用 radar 补
# 对应 239 行：逐元素加
parsing_embedding = lidar_full + radar_full

print(parsing_embedding.shape)                                   # torch.Size([3, 32, 448, 224])
print(torch.allclose(parsing_embedding[:, :, :352], lidar_emb + radar_emb))  # True 前352行=真两模态融合
print(torch.allclose(parsing_embedding[:, :, 352:], 2 * rear_emb))           # True 后96行=2×radar(被"增强")
# 反面实验：把 lidar 缺口改成补零，观察 352/353 行交界会出现分布阶跃
lidar_zero = torch.cat([lidar_emb, torch.zeros_like(rear_emb)], dim=2)
print((lidar_zero + radar_full)[:, :, 352:].std(), (2 * rear_emb).std())  # 补零版方差减半→人工边缘
```

**【小结】** ① 原始 radar 沿 H 维(dim=2)拼回 `[3,64,448,224]` 并作为返回值之一，预留给下游 RC(radar-camera)融合，让 RC 支路吃到未压缩的原始观测。② embedding 层面：后向 96 行 radar 过共享的 first_conv 后，同时拼进 radar 支路和 lidar 支路的尾部，把两支都凑成 `[3,32,448,224]`，再用零参数的逐元素加完成 RL 融合。③ 副作用是后向 96 行 = 2×radar embedding——讲者定性为"增强"，本质是单模态区的隐式加权，靠后续 UNet+重训消化。

---

## Part 5.4 UNet backbone 与双尺度输出（00:34:56–00:36:01）

**导读**：融合完的 `parsing_embedding`（讲者称 BEV embedding）进入本模块最后一站：`self.lidar_backbone.forward_no_inc(...)`——一个**去掉了输入块的 UNet**（输入块的活已在 Part 5.2 被两支 first_conv 干完了）。UNet 输出两路：全尺度 448×224 融合特征，和编码-解码途中的 1/2 尺度 224×112 特征（接口参数 `return_reciprocal_2nd=True` 就是在索要它）。前者继续沿模态融合主线走，后者按讲者说法供检测头结合原始 radar 特征优化目标 yaw。最后 242 行把三元组 `(parsing_embedding, radar_feature, feat_reciprocal_2nd)` 返回，本模块谢幕，下一站 LSS 投影。

---

### 卡 5.4-1 BEV embedding = 融合后的 lidar-radar 特征

> **原话** `[00:35:03][00:35:08]`（合并）：
> "然后得到这个 BEV embedding，就是我们融合后的一个 lidar 的、radar 的一个 feature。"

- 【直译】给 239 行的加法结果起正式名字：BEV embedding——它已经同时含有 lidar 和 radar 的信息，是点云侧统一的 BEV 表征。
- 【代码】变量名实为 `parsing_embedding`（"parsing"疑为该团队对 BEV 特征解析主干的历史称呼）。帧 00:35:25 悬停显示它：`ndim=4, dtype=torch.float32, device=cuda:0, layout=torch.strided, grad_fn=<ReluBackward1>`——注意此时悬停的是 240 行执行**后**的版本，grad_fn 是 ReluBackward1，说明 UNet 主干末端有 ReLU；239 行加法刚出炉时它的 grad_fn 应是 AddBackward。
- 【形状】`[3, 32, 448, 224]`（3=bs×3帧，通道 32 为 first_conv 输出通道推断值，⚠见存疑清单）。
- 【连接】到这一步，DenseBEV 完成了 BEVFusion 语境里 "lidar BEV feature" 的角色构建——只不过它是 L+R 双模态的。后面 LSS 把相机特征也拍到 BEV 后，会和它在模态融合站再 concat/加，套路同源。

---

### 卡 5.4-2 经过一个 UNet 的 backbone（重点句，5 角度）

> **原话** `[00:35:08][00:35:11][00:35:16]`（合并）：
> "然后会经过后续的一个——这里其实就是一个 UNet 的一个操作，UNet 的一个 backbone。"

- 【直译】融合特征要过一个 UNet 结构的卷积主干，做真正的深层特征提取（前面的 first_conv 只是通道压缩，谈不上"提特征"）。
- 【代码】240 行：`parsing_embedding, feat_reciprocal_2nd = self.lidar_backbone.forward_no_inc(parsing_embedding, return_reciprocal_2nd=True)`。铁证在帧 00:36:00 的标签页：`unet.py .../lidar_backbone`——backbone 类就定义在 unet.py。`forward_no_inc` = 跳过 `inc` 输入块、从第一个下采样开始跑的定制前向；这个接口的存在本身就是 Part 5.2 手术的疤痕。
- 【形状】UNet 内部（推断，具体以 unet.py 为准）：448×224 → 下采样若干级（224×112 → 112×56 → …）→ 逐级上采样 + skip connection 拼接 → 回到 448×224。进出同分辨率、中途多尺度，这正是 UNet 的标志。
- 【为什么】BEV 特征提取为什么爱用 UNet？① 检测要在原分辨率出热图，UNet"下去再上来"保住了 448×224 的输出分辨率；② 下采样路径给了大感受野（BEV 上一辆 truck 长十几个格子，得看得够远）；③ skip connection 保细节（小目标 VRU 只占两三个格子）。相比之下 SECOND/PointPillars 用的"两支下采样再 upsample-concat"的 SECOND-FPN 只回到 1/2 尺度，UNet 是它的全分辨率加强版。
- 【连接】全局地图后面还有一站"BEV UNet backbone"（模态+时序融合后的主干）——DenseBEV 前后用了两个 UNet，本章这个只服务 RL 支路。别搞混：本章 UNet 在 unet.py、挂名 `lidar_backbone`；后面那个要到 Ch10 前后才登场。

---

### 卡 5.4-3 输出一：全尺度 448×224 融合特征

> **原话** `[00:35:18][00:35:20][00:35:22]`（合并）：
> "然后在这里会得到正常的一个融合后的 radar 的 feature，是 448×224。"
> （⚠ 讲者口误说"radar 的 feature"，实为 lidar+radar **融合**特征 parsing_embedding。）

- 【直译】UNet 的主输出：和 BEV 网格同分辨率的 448×224 融合特征图。
- 【代码】240 行左值第一项 `parsing_embedding`（被重绑定为 UNet 输出），242 行作为返回三元组第 1 项交出去。
- 【形状】`[3, ~32, 448, 224]`（通道数未在帧中确认⚠）。448×224 覆盖前 95.4 m/后 83.8 m/左右 ±44.8 m @0.4 m。
- 【连接】它就是后续"模态融合"站里点云侧的代表，届时与相机 BEV（LSS 产物）对齐相融；再之后 MemoryManager 缓存的、时序融合 warp 的，都是它的后代。记住它的身份：**点云侧 BEV 主特征**。

---

### 卡 5.4-4 输出二：下采样一倍的 224×112 特征

> **原话** `[00:35:26][00:35:31][00:35:33]`（合并）：
> "然后这个会得到一个下采样一倍的一个 radar 的一个 feature，这个呢就是 224×112。"
> （⚠ 转写"4424×112"应为 224×112；"radar feature"同上，实为融合特征的半尺度版。）

- 【直译】UNet 顺手多吐一份 1/2 分辨率（224×112）的特征。
- 【代码】240 行左值第二项 `feat_reciprocal_2nd`——名字直译"倒数第二分辨率特征"（reciprocal_2nd ≈ 1/2 尺度）；开关参数 `return_reciprocal_2nd=True` 表示"除了终点，把中途 1/2 尺度那站的特征也给我"。它极可能取自 UNet 解码路径上倒数第二级上采样后的特征（该级恰为 1/2 尺度且已融合了深层语义与浅层细节，见练习 ch5-4 的 u1）。
- 【形状】`[3, ?, 224, 112]`，通道数未在帧中出现（半尺度特征惯例通道翻倍，推测 64⚠）。224×112 也正是全局地图里 LSS 投影的工作分辨率——"下采样一倍 224×112 上做投影"。
- 【为什么】要半尺度输出的通用理由：下游有的消费者不需要全分辨率（省算力），有的需要更大感受野/更浓语义。与其让下游自己再 pool 一次（又一轮 DDR），不如 UNet 路过时顺手留一份——"顺路带货"式接口设计。
- 【连接】BEVFusion/CenterPoint 的检测头也常吃 1/2 尺度 BEV 特征（nuScenes 上 head 在 128×128@0.6m 之类的缩尺网格上出热图），DenseBEV 这里同款思路。

---

### 卡 5.4-5 224×112 给检测头：用原始 radar 特征优化检出的 yaw（重点句，5 角度）⚠

> **原话** `[00:35:35][00:35:39][00:35:44][00:35:48][00:35:49]`（合并，句尾"在这里会使用以…会用到"未说完）：
> "这个主要是为了在后续检测头里面，为了增强——我们使用原始的一个 radar 的一个特征，去优化我们检出的一个 yaw，在这里会用到。"
> （⚠ 转写"剪除的yaw"=检出的yaw。）

- 【直译】这份半尺度特征的服务对象是检测头：配合原始 radar 特征，去精修检出目标的朝向角 yaw。
- 【代码】它是返回三元组第 3 项 `feat_reciprocal_2nd`；具体怎么进检测头（哪个分支、和 radar_feature 怎么配对）本章看不到，要等 CenterPoint 检测头章（10 个分支里应有 yaw/rot 相关分支）揭晓。
- 【为什么 radar 能修 yaw】radar 有径向多普勒速度：对运动目标，速度矢量方向与车头朝向强相关（车通常朝速度方向开），所以拿含 radar 信息的特征去约束 yaw 回归，能救活"lidar 点稀疏时朝向估不准"的远距目标——尤其后向远区那 96 行本来就只有 radar。
- 【⚠ 两种解读】讲者把"224×112"与"原始 radar 特征优化 yaw"连讲，但全局地图又标注 LSS 投影发生在 224×112 尺度。合理的拼图是：224×112 半尺度特征既可能作为与相机 BEV 同尺度的融合原料，也可能直通检测头 yaw 分支——两者不互斥；准确接线图待检测头/模态融合章核实，此处按讲者原话记录并存疑。
- 【连接】CenterPoint 的 rot 分支回归 (sin yaw, cos yaw)；DenseBEV 在其上叠"radar 辅助 yaw 优化"，这是对 nuScenes 式标准头的量产化增强。听到 Ch12（检测头）时，记得回来把这条线焊死。

---

### 卡 5.4-6 收尾与过渡：RL 融合结束，下一站 LSS 投影

> **原话** `[00:36:01]`：
> "然后 RL——把我们（这块）结束之后，（接下来）应该就是我们的做 LSS 投影的（模块）。"

- 【直译】本模块讲完，242 行 `return parsing_embedding, radar_feature, feat_reciprocal_2nd` 把三件货交给上层，镜头转向相机支路的 LSS 投影。
- 【代码】三元组去向备忘：① `parsing_embedding` `[3,~32,448,224]` → 模态融合站等相机；② `radar_feature` `[3,64,448,224]`（原始、cat 后完整版）→ RC 融合站；③ `feat_reciprocal_2nd` `[3,?,224,112]` → 检测头 yaw 优化（⚠）。一个模块三个输出各奔前程，这正是画数据流图的意义。
- 【连接】下一章（Ch6）的 LSS：图像特征 × depth 分布外积、拍平、grid_sample 到 BEV——相机侧特征将在 224×112 网格上落地，与本章的点云侧 BEV 特征在模态融合站会师。

---

### 🔨 动手练习 ch5-4：迷你 forward_no_inc——去输入块的 UNet + 双尺度返回

```python
import torch, torch.nn as nn

class MiniUNetNoInc(nn.Module):
    """模拟 DenseBEV lidar_backbone：inc 被外部 first_conv 取代，主干双尺度返回"""
    def __init__(self, c=32):
        super().__init__()
        # 注意：没有 inc！输入默认已是 c 通道（对应两支 first_conv + 相加的产物）
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c,   c*2, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c*2, c*4, 3, padding=1), nn.ReLU())
        self.up1   = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec1  = nn.Sequential(nn.Conv2d(c*4, c*2, 3, padding=1), nn.ReLU())
        self.up2   = nn.ConvTranspose2d(c*2, c,   2, stride=2)
        self.dec2  = nn.Sequential(nn.Conv2d(c*2, c,   3, padding=1), nn.ReLU())

    def forward_no_inc(self, x, return_reciprocal_2nd=False):
        d1 = self.down1(x)                                   # 1/2 尺度
        d2 = self.down2(d1)                                  # 1/4 尺度
        u1 = self.dec1(torch.cat([self.up1(d2), d1], dim=1)) # 回到 1/2 尺度(带skip)
        u2 = self.dec2(torch.cat([self.up2(u1), x],  dim=1)) # 回到全尺度(带skip)
        return (u2, u1) if return_reciprocal_2nd else u2     # u1 即 feat_reciprocal_2nd

net = MiniUNetNoInc(c=32)
parsing_embedding = torch.randn(1, 32, 448, 224)             # Part5.3 的融合结果
out, feat_reciprocal_2nd = net.forward_no_inc(parsing_embedding, return_reciprocal_2nd=True)
print(out.shape)                  # torch.Size([1, 32, 448, 224])  ← 全尺度融合特征
print(feat_reciprocal_2nd.shape)  # torch.Size([1, 64, 224, 112])  ← 半尺度(通道翻倍)特征
# 体会：grad_fn 链末端是 ReLU → 对应帧00:35:25里 parsing_embedding.grad_fn=ReluBackward1
print(out.grad_fn)                # <ReluBackward0 ...>
```

**【小结】** ① 融合后的 `parsing_embedding` 走 `forward_no_inc`——一个输入块被 first_conv 手术替换掉的 UNet（定义在 unet.py，挂名 lidar_backbone），全分辨率进出、skip connection 保小目标细节。② `return_reciprocal_2nd=True` 让 UNet 额外交出 1/2 尺度 224×112 特征，讲者说它配合原始 radar 特征在检测头里优化检出目标的 yaw（radar 径向速度⇒朝向先验）。③ 模块最终返回三元组：全尺度融合特征（→模态融合）、原始 448×224 纯 radar 特征（→RC 融合）、半尺度特征（→检测头/⚠），RL 站到此收工，下一站 LSS 投影。

---

## 本章总备忘

### 一图流：forward_dense 数据流（优化后路径）

```
lidar_feature [3,352,224,64⚠NHWC]            radar_feature [3,64,352,224]      rear_far_radar_feature [3,64,96,224]
      │                                             │                                   │
forward_img_first_conv(64→32)          forward_radar_first_conv(64→32)     forward_radar_first_conv(64→32,共享权重)
      │                                             │                                   │
lidar_emb [3,32,352,224]                 radar_emb [3,32,352,224]              rear_emb [3,32,96,224]
      │                                             │                            ┌──────┴──────┐
      └── cat(dim=2) ◀── rear_emb ──────────────────┼── cat(dim=2) ◀────────────┘   (原始radar也 cat(dim=2)
                    │                               │                                → radar_feature [3,64,448,224] → RC融合)
      lidar_full [3,32,448,224]          radar_full [3,32,448,224]
                    └────────────  +  逐元素加 ─────────┘
                        parsing_embedding [3,32,448,224]   (后96行 = 2×rear_emb，"增强")
                                    │
                    lidar_backbone.forward_no_inc (UNet, unet.py)
                          │                        │
        parsing_embedding [3,~32,448,224]   feat_reciprocal_2nd [3,?,224,112]
              → 模态融合                        → 检测头 yaw 优化⚠ / LSS同尺度
```

### 本章术语校正对照（转写噪声 → 本义）
| 转写原文 | 本义 | 出现处 |
|---|---|---|
| Rio / 没有做Rio | RL（Radar-Lidar） | 00:32:11 |
| Cover / 三层Cover | conv 卷积 | 00:32:35 等 |
| Rero | ReLU | 00:32:55 |
| Cameral（第一处） | Camera（RC=radar-camera） | 00:33:46 |
| Cameral（第二处） | 卷积（"只经过一个卷积"） | 00:33:57 |
| 存radar | 纯 radar | 00:33:31 |
| 颜色高度这一围 | 沿着高度这一维（H, dim=2） | 00:34:12 |
| 不向远距离 | 后向远距离 | 00:34:42 |
| 剪除的yaw | 检出的 yaw | 00:35:44 |
| 3524 / 48成24 / 4824 / 4424×112 | 352×224 / 448×224 / 448×224 / 224×112 | 多处 |
| 修化 | 优化 | 00:32:21 |

### ⚠ 存疑清单（本章）
1. **lidar 输入内存布局**：调试台 `inputs[0].shape = [3,352,224,64]`（NHWC）与口述 64×352×224（CHW）不一致；`forward_img_first_conv` 内部应有 permute，或打印时机在 permute 前。待 unet.py/封装函数核实。
2. **优化前 inc 的确切通道序列**：讲者说"三层卷积，64→128、再 128→32"，但代码老路径 concat 后已是 128 通道进 backbone，口述与代码细节对不拢；仅"三层conv+中间ReLU→一层conv"的结论可靠。
3. **parsing_embedding 与 feat_reciprocal_2nd 的通道数**：32 / 64 均为推断，帧中调试台未打印这两个 shape。
4. **224×112 特征的确切去向**：讲者绑定"检测头用原始 radar 特征优化 yaw"，全局地图又显示 LSS 投影/模态融合也工作在 224×112 尺度——可能一份特征多处消费，待检测头章与模态融合章对账。
5. **"radar feature 被增强(×2)"的训练影响**：前 352 行与后 96 行特征分布不一致（双模态和 vs 单模态×2），推测由后续 UNet 中归一化与重训消化，讲者未展开。
6. **口误**：00:35:22/00:35:26 两处"融合后的 radar 的 feature"实指 lidar+radar 融合特征（parsing_embedding），非纯 radar。


---
> [[Ch04_Radar代码实走与远距离切分|← Ch4]] · [[00_总览与脉络|📖 总览]] · [[Ch06_LSS投影输入准备|Ch6 →]]

