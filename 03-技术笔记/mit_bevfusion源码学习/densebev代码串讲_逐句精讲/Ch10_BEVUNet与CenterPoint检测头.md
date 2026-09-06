> [[Ch09_MemoryManager与时序融合|← Ch9]] · [[00_总览与脉络|📖 总览]] · [[Ch11_Loss全解|Ch11 →]]

# Ch10 BEV UNet backbone 与 CenterPoint 检测头（01:15:39–01:31:16）

> **本章在全局地图的位置**：
> `…… → 模态融合 → MemoryManager(10Hz缓存) → 时序融合(warp历史帧) → 【★你在这里：BEV UNet backbone → CenterPoint检测头(10个分支)】 → Loss → Box解码`
>
> 前一章（时序融合）结束时，我们手里有一张融合了图像、RL（radar+lidar）、且已对齐历史帧的 BEV 特征图 `bev_temporal`，形状 **1×64×448×224**（448 对应纵向 95.4+83.8=179.2m ÷ 0.4m/格，224 对应横向 ±44.8m=89.6m ÷ 0.4m/格）。本章讲两件事：
> ① 这张特征图过一个 **UNet 式的 BEV backbone（`CaddnBEVBackbone`）**——下采样再上采样，尺寸不变、语义变深，同时把中间两个尺度（112×56、224×112）保下来给端到端（E2E 规划）用；
> ② 然后进入 **CenterPoint 风格检测头**——一个共享卷积 + 10 个并行小分支（reg/height/dim/rot/vel/dir_cls/movement/rot_lidar/close_heatmap/heatmap），每个分支就是 2 层卷积，最后输出一整套 `1×C×448×224` 的稠密预测图。
>
> **本章帧证来源**（精读单帧）：`01_15_56.jpg`（BevBackbone draw.io 框图）、`01_16_47.jpg`（caddn_bev.py UNet 前向）、`01_18_25.jpg`（DetHead 框图四路输入）、`01_20_05.jpg`（det_head.py forward 与 skip_keys）、`01_21_58.jpg`（data_dict 组装）、`01_22_45.jpg`/`01_23_16.jpg`（lidar_rt/lidar_vel 切片）、`01_24_51.jpg`（CenterHead.forward_single）、`01_25_13.jpg`（yaml common_heads 配置 + SeparateHead）、`01_26_44.jpg`（SeparateHead.forward 全文）、`01_29_51.jpg`/`01_30_13.jpg`（调试台 shape 清单，另做局部放大裁图核字）。

---

## Part 1｜BEV UNet backbone：下采样再上采样，中间尺度留给端到端（01:15:39–01:17:45）

**导读**：本段的输入是时序融合模块吐出来的那一张 BEV 特征 `bev_temporal`（帧图中叫 `det_feat`，bs×64×448×224），输出有两个：① 尺寸不变的 `bev_feat`（bs×64×448×224），给检测头；② 一个中间特征列表 `mid_feat`，含 bs×128×112×56 和 bs×64×224×112 两个尺度，专门留给端到端网络当 instance embedding 的底料。它在流水线里的角色，等价于 BEVFusion 里 LSS 之后的那个 SECOND/SECONDFPN "BEV encoder"——多模态特征拼完之后，还要在 BEV 平面上再卷几层，把不同模态、不同时间贴上来的特征"揉匀"。讲者只用了两分钟讲这个模块，因为它结构上确实就是个标准 UNet/BaseBEVBackbone，但**输出接口（bev_feat + mid_feat 双输出）是理解后面检测头 5 路输入的钥匙**，所以我们展开讲透。

---

#### 卡10-1｜[01:15:39] "这个是时序融合模块……之后的话就是 BEV feature 的一个 backbone，就是这个 CaddnBEVBackbone"
（合并 [01:15:39][01:15:41][01:15:43][01:15:52]，讲者此处有重复报幕）

- 【直译】上一章的时序融合讲完了，现在进入下一个模块：BEV 特征的骨干网络，代码里类名叫 `CaddnBEVBackbone`。
- 【代码】帧证 `01_15_56.jpg`：draw.io 框图里写的是 `CaddnBEVBackbone(self.bev_backbone)`，即在模型容器里以 `self.bev_backbone` 挂载；帧证 `01_16_47.jpg` 显示源文件路径为 `e2e > tasks > bev_task > uvp_module > models > bev_backbone > caddn_bev.py`，第 126 行 `class CaddnBEVBackbone(nn.Module):`。调用形式等价于 `bev_feat, mid_feat = self.bev_backbone(bev_temporal)`。
- 【为什么】类名里的 "Caddn" 泄露了血统：CaDDN（Categorical Depth Distribution Network, CVPR 2021）是单目 BEV 检测的经典工作，它的 BEV backbone 直接搬的 OpenPCDet 的 `BaseBEVBackbone`。DenseBEV 团队大概率是从 CaDDN/OpenPCDet 抄了这个骨干过来改，所以类名一直没换——工业代码里这种"名不副实的历史遗留命名"非常常见，读代码时认结构不认名字。
- 【连接】你在 BEVFusion（mmdet3d 版）里对应的是 `pts_backbone: SECOND` + `pts_neck: SECONDFPN`：也是 BEV 平面上多尺度卷积再融合。DenseBEV 把"backbone+neck"合在一个类里，并且额外把中间层特征导出去，这是它和标准 BEVFusion 结构上最大的差异点。

---

#### 卡10-2｜[01:15:56] "它的输入呢，其实就只有这一个，就是时序融合完之后的图像以及 RL 的一个 feature"
（合并 [01:15:56][01:15:58][01:16:03][01:16:05][01:16:09]；并呼应上一章末尾 [01:15:08]–[01:15:27] 对该 feature 的铺垫）

- 【直译】这个 backbone 是单输入的：只吃时序融合之后那张"图像+RL 已经融在一起"的 BEV 特征，不再需要任何别的东西。
- 【代码】`forward(self, spatial_features)` 单参数。帧证 `01_15_56.jpg` 框图左侧虽然画了 5 个候选输入框（`det_feat` bs\*64\*448\*224、`lite_feat` None、`seg_feat` None、`for_pnc_in_feat` (bs\*3)\*32\*448\*224、`lidar_feat_reciprocal_2nd` bs\*64\*448\*224），但真正连线进 `BevBackbone` 大框的只有 `det_feat` 这一条；`lidar_feat_reciprocal_2nd` 的线是**绕过** backbone 直接连向后面 DetHead 的。
- 【形状】输入 `det_feat`: bs×64×448×224（训练 mini 配置 bs=1，即 1×64×448×224）。
- 【为什么】单输入意味着"模态融合、时序融合的所有信息都已经沉淀在这一张图里"，backbone 只负责空间上下文的再加工。而 `lite_feat`/`seg_feat` 为 None 说明这套代码还预留了轻量分支和分割分支的接口但当前没启用——读工业代码要习惯这种"占位输入"。
- 【连接】对比 BEVFusion：`fuser` 输出单张 BEV 特征 → `pts_backbone` 也是单输入。逻辑位置完全同构，你可以把这两行代码在脑子里画等号。

---

#### 卡10-3｜[01:16:13] "在这里它主要用的就是一个 UNet"
（合并 [01:16:13][01:16:15][01:16:18][01:16:19][01:16:21]，讲者重复"就是一个unit啊/在这里/在这儿"）

- 【直译】这个 backbone 的结构范式就是 UNet：先压小（编码），再放大（解码），同尺度之间有信息汇合。
- 【代码】帧证 `01_16_47.jpg` 可见其真实实现（后述卡10-5），本质是 OpenPCDet `BaseBEVBackbone` 风格：`self.blocks`（每个 block 第一层 stride=2 下采样 + 若干 3×3 卷积）+ `self.deblocks`（`ConvTranspose2d` 上采样回原尺度）+ `torch.cat` 汇合。严格说它不是经典 UNet 的"逐级 skip-connection 解码"，而是"**各级并行上采样到同一尺度后 concat**"——但讲者用 UNet 来概括"下采样-上采样-保留多尺度"的精神，是合理的口头简化。
- 【为什么】为什么 BEV 特征还要再下采样？因为 448×224、0.4m/格 的分辨率下，一辆 5m 长的车只占 12 格左右，而 3×3 卷积感受野一次只扩 1 格；下采样到 112×56 后一格=1.6m，同样的卷积就能看到整车甚至车与车的关系，**用小图换大感受野**。再上采样回来，是因为 CenterPoint 检测头要在高分辨率图上回归中心点，分辨率丢了定位精度就丢了。
- 【连接】①智谷课程里讲过的"感受野=堆叠卷积的等效视野"在这里是活教材；②你跑的 BEVFusion 中 SECOND 的 `layer_strides=[1,2]` + SECONDFPN `upsample_strides=[1,2]` 是一模一样的"压-放-拼"三段式。

---

#### 卡10-4｜[01:16:41] "对，这个就是输入的特征，1×64×448×224" ⚠
（合并 [01:16:41][01:16:44]）

- 【直译】讲者在调试台里指认了输入张量的实际形状：batch 1、通道 64、BEV 网格 448×224。
- 【形状】⚠ 转写原文是"一乘以64乘48乘24"，Whisper 吞掉了数字：帧证 `01_16_47.jpg` 调试台明确打印 `det_feat_stage: torch.Size([1, 64, 448, 224])`、`lidar_feat_reciprocal_2nd_shape: torch.Size([1, 64, 448, 224])`，应为 **1×64×448×224**。
- 【代码】`assert spatial_features.shape == (1, 64, 448, 224)`。另外帧里 `forward_backup` 的 docstring 写的是 `spatial_features: [N, 128, 352, 224] → bev_feature: [N, 384, 352, 224]`——那是**旧配置的注释没更新**（352×224、128 通道），又一个"注释会说谎、调试台不会"的例证。⚠ 顺带标注：帧里看到的这段前向函数名叫 `forward_backup`，正式 `forward` 应是同逻辑版本或在别处，讲者未提，存疑但不影响理解。
- 【为什么】448=（前95.4m+后83.8m）/0.4，224=左右各44.8m/0.4。通道 64 是全链路统一的 BEV 特征宽度（yaml 里 `in_channels: 64`、`share_conv_channel: 64` 都咬合这个数）。
- 【连接】nuScenes 版 CenterPoint 常用 128×128 或 180×180 的正方形网格；这里 448×224 是"前后不对称+左右窄"的**量产前视为主**设定——跟你在华为车BU见过的实车 ROI 直觉一致：向前看得远（95.4m），向后短（83.8m），侧向 44.8m 够覆盖三车道。

---

#### 卡10-5｜[01:16:47] "在这里会下采样，然后再做一个上采样，然后会把中间的一个特征给保存下来"【重点句，5角度】
（合并 [01:16:47][01:16:49][01:16:52][01:16:54]）

- 【直译】前向流程三步：逐级下采样提特征；再上采样回原尺寸；下采样过程中的中间结果不丢，存进一个列表带出去。
- 【代码】帧证 `01_16_47.jpg` 逐行可读（caddn_bev.py 228–248 行附近）：
  ```python
  x = spatial_features[0] if self.fuse_sd_map else spatial_features[0]
  if self.fuse_sd_map:                      # SD地图融合分支，当前配置未走
      sd_map_in = spatial_features[1]
      sd_map_feature, _, _ = sd_map_in
      map_feature = self.map_conv(sd_map_feature)
      x = x + map_feature
  ups = []
  mid_feat = []
  for i in range(len(self.blocks)):
      x = self.blocks[i](x)                 # 下采样block：stride2卷积×1+3×3卷积×n
      mid_feat.append(x)                    # ← 中间特征保存
      ups.append(self.deblocks[i](x))       # 反卷积上采样回448×224
  x = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
  if len(self.deblocks) > len(self.blocks):
      x = self.deblocks[-1](x)
  bev_feature = self.reduce_channel(x)      # 1×1或3×3卷积压回64通道
  return bev_feature, mid_feat[::-1]
  ```
- 【形状】`x`: 1×64×448×224 → block0 → 1×64×224×112（存入 mid_feat）→ block1 → 1×128×112×56（存入 mid_feat）；两路 deblock 都上采样回 448×224，cat 后 1×(64+128)×448×224=1×192×448×224，`reduce_channel` → **1×64×448×224**。（mid_feat 具体数字由帧证 `01_20_05.jpg` 调试台反推：`[i.shape for i in mid_feat[::-1]]` = `[torch.Size([1,128,112,56]), torch.Size([1,64,224,112])]`。）
- 【为什么】三个设计点：① `mid_feat.append` 放在 block 之后、deblock 之前——保存的是**纯下采样路径上的原生多尺度特征**，不掺上采样的插值痕迹，语义最干净；② `mid_feat[::-1]` 反转，把最深的（112×56）排在最前，符合"coarse-to-fine"的下游消费习惯；③ `fuse_sd_map` 分支（`nn.ZeroPad2d(1)+Conv2d(32,64,k=3,s=2)+BN+ReLU` 后逐元素相加）说明这套框架预留了 SD 导航地图作为 BEV 先验的入口，当前没开。
- 【连接】和 OpenPCDet `BaseBEVBackbone.forward` 几乎逐行同构（`ups`/`x = torch.cat(ups, dim=1)` 连变量名都一样），唯二的差别就是多了 `mid_feat` 导出和 `reduce_channel`。你以后读 BEVFusion 的 `SECOND.forward` 会有强烈既视感——这就是"一套骨干走天下"的证据。

---

#### 卡10-6｜[01:17:02] "最后输出的话，这个 feature 的尺度还是没有变的，还是 64、448 和 224" ⚠
（合并 [01:17:02][01:17:05][01:17:09][01:17:11]；⚠转写"64 48和224"缺位，按帧证补全为 64×448×224）

- 【直译】主输出 `bev_feature` 的空间尺寸与输入完全一致：过了一趟 UNet，分辨率没变、通道数也没变，变的只是特征的"熟度"。
- 【形状】in 1×64×448×224 → out 1×64×448×224。帧证 `01_30_13.jpg` 调试台：`bev_feature.shape → torch.Size([1, 64, 448, 224])`。
- 【代码】`assert bev_feature.shape == spatial_features.shape`——尺寸不变但感受野从几格扩到了全图级别（经过 112×56 尺度的特征等效看到 ~4 倍范围）。
- 【为什么】检测头要求输入输出同网格：CenterPoint 的 heatmap 峰值坐标直接乘 `voxel_size×out_size_factor` 还原成米制坐标，本工程 yaml 里 `out_size_factor: 1`（帧证 `01_25_13.jpg`），即**检测头就工作在 448×224 全分辨率上**，一格误差=0.4m。这跟 nuScenes CenterPoint 常见的 `out_size_factor=8` 差别巨大——量产近距离检测对定位精度的要求，逼着他们把头开在全分辨率上。
- 【连接】BEVFusion 里 SECONDFPN 输出通常是 256~512 通道再给头；这里始终压在 64 通道，是嵌入式算力（Orin 级别）下的瘦身选择——通道砍 4 倍，头上的每个 3×3 卷积计算量同比例砍。

---

#### 卡10-7｜[01:17:12] "保留中间两个尺度的 feature：112×56，以及 224×112 这两个尺度" ⚠【重点句，5角度】
（合并 [01:17:12][01:17:18]；⚠转写"224和124和112"含混，按帧证定为 112×56 与 224×112）

- 【直译】除了主输出，还把下采样路径上两档中间特征原样带出去：一档是 1/4 分辨率（112×56），一档是 1/2 分辨率（224×112）。
- 【代码】即上文 `return bev_feature, mid_feat[::-1]` 的第二个返回值；draw.io 框图（帧证 `01_15_56.jpg`）右侧明确画了两个输出口：`bev_feat: bs*64*448*224` 和 `mid_feat: [bs*128*112*56, bs*64*224*112]`。
- 【形状】`mid_feat[::-1] = [1×128×112×56, 1×64×224×112]`。注意规律：分辨率每砍半、通道翻倍（64→128），是分类网络时代传下来的"保持每层信息量近似恒定"的惯例。
- 【为什么】为什么端到端要的是**中间特征**而不是最终 `bev_feat`？① 最终特征是为检测任务特化过的（后面还要过检测专用 shared_conv），任务偏置太重；中间特征更"通用"，适合规划网络自己再加工；② 端到端网络（决策规划）关心的是可通行空间、交互关系这类**中低频大尺度**信息，112×56 这种粗尺度反而信息密度更合适、算力更省。
- 【连接】这是"感知-规划一体化"（UniAD 式端到端）的接口设计雏形：感知骨干当共享 encoder，规划分支从中间层抽 embedding。你转岗后如果做 BEV 特征给下游共享，这种 `return main, aux_list` 的双出口写法可以直接抄。

---

#### 卡10-8｜[01:17:25] "主要是为了把这里 224×112 的 feature 传到端到端他们要使用，在这里会记录下来"
（合并 [01:17:25(见1161)][01:17:33区间][01:17:40][01:17:41(记录)]）

- 【直译】动机点名：保存中间尺度这件事，不是检测自己要用，而是"端到端"那个团队/分支点名要 224×112 这档特征；代码里在这个位置把它记录下来往外传。
- 【代码】下游对接处在 det_head.py（帧证 `01_23_16.jpg`）：
  ```python
  bev_mid_feat_list = []
  if self.export_instance_embeddings:      # yaml: export_instance_embeddings: True
      bev_mid_feat_list = inputs[4]        # mid_feat 列表从第4号输入进来
  ```
- 【为什么】"端到端他们"这个措辞说明这是**跨团队接口**：BEV 感知组产特征，E2E 组消费。工程上把接口做成"列表透传+开关控制"（`export_instance_embeddings` 开关），检测主链路完全不依赖它，关掉也不影响检测——好的解耦示范。
- 【连接】华为车BU的组织结构你熟：感知组和预测决策组之间的特征接口谈判，就是这行代码的现实来源。面试时讲"我读过一份量产代码，感知给规划留特征口子是这么留的"，是很加分的细节。

---

#### 卡10-9｜[01:17:45] "这个就是所有的检测头之前的所有的模块都讲完了！"
（过渡句）

- 【直译】里程碑宣告：从图像 backbone、DepthNet、LSS、模态融合到时序融合、BEV backbone，检测头上游的全部计算图闭环了。
- 【连接】对照全局地图：`FPN → DepthNet → Lidar/Radar backbone → RL融合 → LSS投影 → 多视角融合 → RC融合 → 模态融合 → MemoryManager → 时序融合 → BEV UNet` 至此全部走完，手里的资产是三样东西：时序融合特征（经UNet）、单帧 RL 特征、时序 RL 特征——正好就是下一 Part 检测头的三大主料。
- 【为什么】讲者在这里分段，也是提示你复习节奏：检测头是消费端，前面全是生产端；生产端任何一处 shape 变了，下面检测头的 5 路输入就要跟着对齐。

---

### 🔨 动手练习 ch10-1：迷你 CaddnBEVBackbone（下采样-上采样-留中间尺度）

```python
import torch, torch.nn as nn

class MiniBEVUNet(nn.Module):
    """结构等比例复刻 CaddnBEVBackbone：2个下采样block + 2个反卷积 + cat + reduce。
    真实工程通道为64/128，这里用16/32省算力，形状规律完全一致。"""
    def __init__(self, c=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c,   c,   3, 2, 1), nn.BatchNorm2d(c),   nn.ReLU(),
                          nn.Conv2d(c,   c,   3, 1, 1), nn.BatchNorm2d(c),   nn.ReLU()),
            nn.Sequential(nn.Conv2d(c,   2*c, 3, 2, 1), nn.BatchNorm2d(2*c), nn.ReLU(),
                          nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.BatchNorm2d(2*c), nn.ReLU()),
        ])
        self.deblocks = nn.ModuleList([
            nn.ConvTranspose2d(c,   c, 2, stride=2),   # 224x112 -> 448x224
            nn.ConvTranspose2d(2*c, c, 4, stride=4),   # 112x56  -> 448x224
        ])
        self.reduce_channel = nn.Conv2d(2*c, c, 1)

    def forward(self, x):
        ups, mid_feat = [], []
        for blk, deblk in zip(self.blocks, self.deblocks):
            x = blk(x)
            mid_feat.append(x)
            ups.append(deblk(x))
        x = torch.cat(ups, dim=1)
        return self.reduce_channel(x), mid_feat[::-1]

net = MiniBEVUNet()
bev_temporal = torch.randn(1, 16, 448, 224)      # 对应真实 1x64x448x224
bev_feat, mid_feat = net(bev_temporal)
print('bev_feat :', bev_feat.shape)              # torch.Size([1, 16, 448, 224]) 尺寸不变
for m in mid_feat:
    print('mid_feat :', m.shape)
# 预期输出（对照真实工程 [1,128,112,56] / [1,64,224,112]）：
# mid_feat : torch.Size([1, 32, 112, 56])   ← 最深尺度在前（[::-1]的效果）
# mid_feat : torch.Size([1, 16, 224, 112])
```

**【小结】** ① BEV backbone = CaddnBEVBackbone，单输入（时序融合特征 1×64×448×224）双输出（同尺寸 bev_feat + 两档中间尺度 mid_feat）。② 结构是 OpenPCDet BaseBEVBackbone 式的"stride2 blocks → 反卷积 ups → cat → reduce_channel"，讲者口头称之为 UNet。③ mid_feat 的 112×56/224×112 两档不是检测要用，而是留给端到端网络取 instance embedding 的跨团队接口。

---

## Part 2｜DetHead 的五路输入：谁供给、谁消费、各自为了什么（01:18:05–01:19:35）

**导读**：本段讲检测头 wrapper 类 `DetHead(BaseModule)` 的输入清单。帧证 `01_18_25.jpg` 的 draw.io 框图画得清清楚楚，5 个输入框连向 DetHead：`bev_multiview (bs*3)*128*448*224`（只做可视化）、`bev_feat bs*64*448*224`（主特征）、`parsing_embedding (bs*3)*64*448*224`（三帧未时序融合的 RL 特征→优化 yaw）、`lidar_feat_reciprocal_2nd bs*64*448*224`（时序融合后的 RL 特征→优化速度）、`mid_feat`（给端到端）。这段是**理解整个检测头设计哲学的核心**：不是所有属性都该用同一张特征图预测——yaw 要用无时序拖影的单帧点云特征，速度要用带时序运动信息的特征。听懂这段，后面 SeparateHead 里按 head 分发特征的 if-else 就全都顺理成章。

---

#### 卡10-10｜[01:18:05] "后面的话就是我们检测的一个 head。它的输入……这个只是为了做一个可视化用的，这个是当前没有用到的"
（合并 [01:18:05][01:18:11]）

- 【直译】进入检测头。输入清单里第一个东西只服务于可视化调试，正常前向里没人消费它。
- 【代码】帧证 `01_20_05.jpg`：`data_dict = dict(dense_ogm=inputs[0], ...)`——第 0 号输入名叫 `dense_ogm`（dense occupancy grid map 之类的稠密可视化底图），对应框图里 `bev_multiview: (bs*3)*128*448*224`。它被塞进 data_dict 但检测 loss 不用。
- 【形状】(bs\*3)×128×448×224：注意第一维是 bs×3（三帧摊平），128 通道——正是**多视角融合后、未过模态融合**的原始图像 BEV 特征，拿它可视化能看到"相机部分到底投了什么上来"。
- 【为什么】量产开发里可视化输入常年挂在接口上不摘，因为排查 badcase 时要随时打开（后面 `if self.vis_heatmap: self.det_head.vis_heatmap(...)` 就是消费口）。代价是接口噪声——所以讲者特意提醒"这个当前没有用到"，帮你排除阅读干扰。
- 【连接】你在 BEVFusion 里 debug 投影对不对，也是先把 camera-only 的 BEV 特征 heatmap 画出来看——同一套排障思路，这里直接做进了接口。

---

#### 卡10-11｜[01:18:17] "其他输入的4个特征呢，就是我们时序融合完之后、又经过 UNet 之后的一个图像以及 RL 的 feature"【重点句，5角度】
（合并 [01:18:17][01:18:21][01:18:27][01:18:28]）

- 【直译】除可视化输入外还有 4 路。第一路主特征：时序融合 → UNet backbone 之后的图像+RL 融合特征，即上一 Part 的 `bev_feat`。
- 【代码】`bev_feat_map=inputs[1]`（帧证 `01_20_05.jpg` 第 81 行）。后面 CenterHead 里它就是主 `x`，过 shared_conv 后供 reg/height/dim/heatmap 等大多数分支使用。
- 【形状】1×64×448×224。
- 【为什么】它是"信息最全"的一张图：相机语义+雷达几何+激光几何+三帧时序全都揉进去了，所以默认所有属性都从它出——**除非**某属性对这张图里的某种"污染"敏感（yaw 对时序拖影敏感、速度反而需要时序），才另开小灶，这就是后三路输入存在的理由。
- 【连接】BEVFusion 检测头只有这一路输入；DenseBEV 的多路输入是它对 CenterPoint 框架的最大魔改。记住这个差异，看下一张卡。

---

#### 卡10-12｜[01:18:39] "然后这个是……它是 embedding，看一下，这个是没有做时序融合的一个 RL 的 feature"
（合并 [01:18:39][01:18:42][01:18:45][01:18:46][01:18:49]，讲者现场翻代码确认）

- 【直译】第二路输入：一个 embedding——确认之后说明它是**没做过时序融合**的 RL（radar+lidar）特征。
- 【代码】框图名 `parsing_embedding: (bs*3)*64*448*224`。消费处（帧证 `01_22_45.jpg`）：`inputs[2]`，进 `use_lidar_rt` 分支。第一维 bs\*3 说明它保留着三帧堆叠、彼此独立，没被 warp/融合过。
- 【形状】(1\*3)×64×448×224，稍后会被切成单帧 1×64×448×224。
- 【为什么】"embedding"这个随口的称呼+名字里的 parsing，暗示这路特征原本可能服务于解析/分割类任务，被检测头借用。工程里特征复用比新算一路便宜得多。
- 【连接】RL 融合章节（前面章）产出的就是这路特征的源头；它绕过了时序融合模块直接送到头上——在框图上就是那条"从左侧一路平行画到 DetHead"的长线。

---

#### 卡10-13｜[01:18:56] "这个主要是为了优化我们 yaw 的一个精度，就是用单帧的一个 RL 的 feature" ⚠【重点句，5角度】
（⚠转写原文"优化我们yaw的一个减速"，"减速"应为"精度/角速度"之误听；结合 [01:19:17] 同样句式"优化速度的一个减速"，推断讲者口癖把"预测/精度"说含糊了，按语义取"精度"）

- 【直译】这路单帧 RL 特征的用途：专门提升航向角（yaw）预测的精度。
- 【代码】它最终喂给两个分支（帧证 `01_26_44.jpg`，SeparateHead.forward）：
  ```python
  if head == 'rot_lidar':
      feat_in = kwargs['lidar_rt_feat']          # 纯单帧RL特征直接回归yaw
  elif head == 'rot':
      feat_in, lidar_weight = self.simple_fusion(x, kwargs['lidar_rt_feat'])
      ret_dict['lidar_rot_weight'] = lidar_weight  # 融合权重也输出，1通道
  ```
- 【形状】lidar_rt_feat: 1×64×448×224；产出 `rot_lidar`: 1×2×448×224、`lidar_rot_weight`: 1×1×448×224。
- 【为什么】yaw 为什么怕时序特征？时序融合把历史帧 warp 到当前帧，warp 用的是**自车运动**补偿，补不掉**他车自身的运动**——动目标在融合特征里会留下"拖影"，车头方向的边缘信息被抹糊，而 yaw 恰恰靠车辆轮廓的朝向边缘来定。激光/毫米波单帧点云的轮廓是瞬时快照，无拖影，所以单帧 RL 特征回归 yaw 更准。团队还不放心到做了两手：`rot` 头用可学习权重把主特征和单帧 RL 特征 simple_fusion 起来，`rot_lidar` 头干脆只用单帧 RL 特征，两个都出，loss 里各自监督。
- 【连接】nuScenes 评测指标 mAOE（朝向误差）是出了名难降的一项；DenseBEV 这套"专用特征喂专用头"是工业界对付 mAOE 的实招。你以后改 BEVFusion 提朝向精度，这是可以直接借鉴的思路。

---

#### 卡10-14｜[01:19:06] "这个是在这里做完时序融合之后的一个 RL 的 feature，这个是为了优化速度的" ⚠
（合并 [01:19:06][01:19:11][01:19:17]；⚠"速度的一个减速"同上，取"速度的预测"）

- 【直译】第三路输入：时序融合**之后**的 RL 特征，专门用来提升速度（velocity）预测。
- 【代码】框图名 `lidar_feat_reciprocal_2nd: bs*64*448*224`；消费处 `inputs[3]` → `lidar_vel_feat`；在 SeparateHead 里：`feat_fus_lidar_vel = self.conv_3x3(torch.cat((x, kwargs['lidar_vel_feat']), 1))`，供 `vel` 和 `movement` 分支使用。
- 【形状】1×64×448×224（时序融合已把三帧并成一帧，所以第一维是 bs 不是 bs×3）。
- 【为什么】速度和 yaw 的需求正好相反：**速度本质上就是"多帧位置差"**，单帧点云原理上不含速度信息（毫米波的径向多普勒除外），只有把 t-2/t-1/t 三帧对齐后叠一起，网络才能从"同一目标在三帧里的位移模式"读出 vx、vy。所以速度头必须吃时序特征——同一个模型里，一个属性嫌弃时序拖影、另一个属性靠时序拖影吃饭，这就是"按属性分特征"的完整逻辑闭环。
- 【连接】CenterPoint 在 nuScenes 上预测 velocity 也是靠输入 10 帧点云叠加（sweep 机制）；DenseBEV 用显式时序融合模块替代了 sweep 叠帧。名字里 `reciprocal_2nd` 直译"第二路互补"，呼应 RL 融合章节里 radar/lidar 互补融合的产物。

---

#### 卡10-15｜[01:19:20] "这个就是刚刚说的，取的中间两种不同尺度的 BEV feature，给到端到端要使用。主要输入就是这几个"
（合并 [01:19:20][01:19:28][01:19:30]）

- 【直译】第四路：Part 1 存下来的 mid_feat 两档中间尺度，透传给端到端。输入清点完毕。
- 【代码】`inputs[4]` → `bev_mid_feat_list`（见卡10-8）。至此 DetHead 的 `*inputs` 元组全貌：
  ```
  inputs[0] dense_ogm        (1*3)x128x448x224  可视化
  inputs[1] bev_feat_map      1x64x448x224      主特征（时序融合+UNet）
  inputs[2] parsing_embedding (1*3)x64x448x224  三帧未融合RL → 单帧 → 优化yaw
  inputs[3] lidar_vel_feat    1x64x448x224      时序融合RL → 优化速度
  inputs[4] mid_feat list     [1x128x112x56, 1x64x224x112] → 端到端
  ```
- 【为什么】用位置参数 `*inputs` 而不是命名参数传 5 路特征，是这份代码的一个可读性坑（读者必须去上游数第几个是什么）；讲者花一整段口头点名每一路，恰恰说明连内部人都需要"接口讲解"——你自己写多输入模块时，尽量用 dict/dataclass 传。
- 【连接】对照 BEVFusion：`bbox_head(feats)` 单路输入。DenseBEV 5 路输入的每一路都对应一个明确的量产诉求（可视化排障/主检测/yaw精度/速度精度/下游共享），这是论文代码和量产代码的气质差异。

---

### 🔨 动手练习 ch10-2：三帧堆叠特征取当前帧（inputs[2] 的切片逻辑预演）

```python
import torch

bs = 1
# parsing_embedding：三帧沿batch维堆叠 (bs*3)x64x448x224（练习缩小到56x28）
parsing_embedding = torch.randn(bs * 3, 64, 56, 28)
lidar_vel_feat    = torch.randn(bs,     64, 56, 28)   # 时序融合后：本来就是单帧

# —— 复刻 det_head.py use_lidar_rt 分支 ——
bst, c, h, w = parsing_embedding.shape
lidar_rt_ft = parsing_embedding.view(bs, bst // bs, c, h, w)  # [1,3,64,56,28]
lidar_rt_feat = lidar_rt_ft[:, -1, :]                         # 取时序最后一帧=当前帧
print('lidar_rt_ft :', lidar_rt_ft.shape)   # torch.Size([1, 3, 64, 56, 28])
print('lidar_rt    :', lidar_rt_feat.shape) # torch.Size([1, 64, 56, 28])

# —— use_lidar_vel 分支：单帧特征不需要再取[-1] ——
print('lidar_vel   :', lidar_vel_feat.shape) # torch.Size([1, 64, 56, 28])

# 验证 [:, -1] 取到的确实是堆叠的最后一帧
assert torch.equal(lidar_rt_feat[0], parsing_embedding[bs*3 - 1])
print('slice check pass: [:, -1] == 堆叠中的最后一帧（当前帧）')
```

**【小结】** ① DetHead 收 5 路输入：可视化底图、主 BEV 特征、三帧未融合 RL、时序融合 RL、mid_feat 列表，各司其职。② 核心设计哲学是"属性适配特征"：yaw 用无拖影的单帧 RL 特征，速度用含运动信息的时序 RL 特征。③ 三帧堆叠特征以 (bs\*3) 摊平在 batch 维传输，用 `view(bs,3,c,h,w)[:, -1]` 还原并取当前帧。

---

## Part 3｜forward 入口：kwargs 取 input、label 只取当前帧、data_dict 组装（01:19:35–01:22:32）

**导读**：本段进入 `DetHead.forward(self, *inputs, **kwargs)` 的开头 30 行（det_head.py 62–99 行，帧证 `01_20_05.jpg`/`01_21_58.jpg`）。输入是上面 5 路特征 + kwargs 里 DataLoader 塞进来的全部 label；输出是一个大 `data_dict`——loss 计算和网络前向需要的所有 key 的集散地。中间的关键动作只有一个：**DataLoader 给的是三帧的 label，检测只认当前帧**，所以要经过一次 `label_extract`，其中"本来就是单帧的 key"用 skip_keys 列表跳过。这段还埋了一个重要的性能优化故事：GT 到 BEV Map 的放置从"训练时现算"改成了"DataLoader 预生成"。

---

#### 卡10-16｜[01:19:35] "这里是去从 kwargs 里面把我们 DataLoader 里面所有的 input 给取出来"
（合并 [01:19:35(1183)][01:19:40(1184)]）

- 【直译】forward 第一步：从关键字参数 kwargs 里取出 DataLoader 打包的全部标签/输入。
- 【代码】帧证 `01_20_05.jpg` 第 67 行（调试黄条高亮行）：
  ```python
  def forward(self, *inputs, **kwargs):
      # Todo, support multi-frame input
      labels = kwargs['labels'][0]
  ```
  `[0]` 是取 batch 里第 0 个样本的标签包（bs=1 时即全部）。上方注释 `# Todo, support multi-frame input` 说明多帧标签监督还在 roadmap 上。
- 【为什么】特征走 `*inputs`（位置参数、有顺序契约），标签走 `**kwargs`（字典、按名取用）——因为标签的种类会随任务开关（track/corner/close_heatmap…）动态增减，用 dict 传天然可扩展。
- 【连接】mmdet3d 的写法是 `loss(preds, gt_bboxes_3d, gt_labels_3d)` 显式传参；这里的 kwargs 大字典风格更像量产框架"万物皆 data_dict"的约定，你在 BEVFusion 的 `data` dict 里见过同款。

---

#### 卡10-17｜[01:19:49] "因为我们 DataLoader 里面是处理的三帧，但是在我们的 load object 里面只是去取了当前帧的相关生成的 GT" ⚠
（合并 [01:19:49(1185)][01:19:54(1186)][01:20:01(1187)][01:20:06(1188)]；⚠转写"机器"应为"GT/标签"）

- 【直译】数据侧背景：DataLoader 为了时序融合一次吃三帧数据；但检测的目标级标注（load object）只为当前帧生成。
- 【代码】数据管线里 dataset 会输出 `labels` 大包，其中检测 GT 只在当前帧上有；后续 `self.seq_info_extract.label_extract(labels, skip_keys=...)` 就是做"三帧→当前帧"的抽取。
- 【为什么】三帧都做目标级标注贵且无用——检测监督只发生在当前帧（历史帧的作用只是提供特征），标当前帧即可。这是"输入多帧、监督单帧"的标准时序检测范式。
- 【连接】你的 BEVFusion+nuScenes 链路里同理：10 个 sweep 只提供点云，box 标注只在 keyframe（sample）上。听懂这句，你就能解释为什么 nuScenes 的 sweep 没有标注也能用。

---

#### 卡10-18｜[01:20:09] "lidar、radar 以及其他的图像的模块里面，其实还会有一些额外的 label 放到了这里面去，所以我们在这里只是取当前帧"
（合并 [01:20:09(1189)][01:20:19(1190)][01:20:23(1191)]）

- 【直译】kwargs['labels'] 这个包不是检测专用的：深度监督（图像模块）、点云监督（lidar/radar 模块）等标签也混在同一个包里；检测头在这里只抽自己要的当前帧部分。
- 【代码】所以才需要显式的抽取函数而不是直接 `labels['xxx']`：`labels = self.seq_info_extract.label_extract(labels, skip_keys=skip_keys)`。
- 【为什么】单一大标签包的好处是 DataLoader 接口稳定（加任务不改签名），坏处是每个头都要"从垃圾堆里捡自己的信"。skip_keys 机制（下一卡）就是这个架构决策的补丁。
- 【连接】回想 Ch3 的 Depth Loss：depth GT/mask 也是从这个大包里取的——同一个包，多个模块各取所需，这就是全链路的标签总线。

---

#### 卡10-19｜[01:20:23] "这些 key 呢，相当于取当前帧的时候，这些 key 它本来就是单帧的，所以这些就不用再去取了；对于其他的 key 它是三帧的，所以在这里会取出当前帧的 label"【重点句，5角度】
（合并 [01:20:23(1192)][01:20:26(1193,1194)][01:20:33(1195)][01:20:36(1196)][01:20:38(1197)]）

- 【直译】label_extract 的规则：包里的 key 分两类——天生单帧的（生成时就只在当前帧上做的）直接跳过不处理；三帧堆叠的才做"抽当前帧"的切片。跳过名单就是 skip_keys。
- 【代码】帧证 `01_20_05.jpg` 第 68–78 行逐字可读：
  ```python
  if self.seq_len > 1:
      # 单帧的一些key需要去掉        ← 源码中文注释原文
      if not self.load_all_label:
          skip_keys = ['obj_label', 'obj_state', 'obj_polygon', 'obj_polygon_list',
              'points3ds', 'is_cls_only', 'origin_bbox3ds', 'obj_label_source',
              'obj_dir_cls_label', 'stop_state', 'stop_source_is_manual',
              'anomaly_tag', 'anomaly_length', 'is_child', 'filter_backward',
              'single_object_annotation', 'errdet_area_...',
              'centerpoint_head_gt_fish', 'centerpoint_head_gt_rl',
              'obj_label_corner', 'centerpoint_head_gt_fish_corner',
              'centerpoint_head_gt_rl_corner', 'obj_dense_label_corner',
              'obj_dense_state_corner', 'obj_dense_dir_cls_label_cor...']
      else:
          skip_keys = []
      labels = self.seq_info_extract.label_extract(labels, skip_keys=skip_keys)
  ```
- 【形状】对非 skip 的 key：`labels[k]` 从"3帧结构"抽成"1帧结构"（如 3×N×… → N×…）；skip 的 key 原样保留。
- 【为什么】看 skip_keys 名单能反读出这家的标注体系有多复杂：多边形标注（obj_polygon）、点级标注（points3ds）、异常目标（anomaly_tag）、儿童（is_child）、停止状态（stop_state）、角点监督（*_corner 系列）、鱼眼专用 GT（centerpoint_head_gt_fish）……这些全是当前帧上直接生成的检测/属性 GT，天生单帧。`load_all_label` 开关（yaml `load_all_label: *LOAD_ALL_LABEL`）为 True 时 skip_keys 置空——即数据侧已把所有 label 做成统一格式，无需区分。
- 【连接】这就是量产数据闭环的冰山一角：nuScenes 只有 box+attribute，这里一个 key 名单就有 25+ 种标签维度。你转岗做量产 BEV，"标签 schema 管理"会是日常工作的真实组成部分。

---

#### 卡10-20｜[01:20:47] "这里是把检测头计算 loss 和前向 forward 所需要的一些 key 或者说输入，放到 data_dict 里面去"
（合并 [01:20:47(1198)][01:20:58(1199)]）

- 【直译】接下来一大段代码只干一件事：把特征和标签统一装进一个 `data_dict` 字典，后面 loss 和 forward 都从这个字典取货。
- 【代码】帧证 `01_21_58.jpg` 第 80–94 行（黄条高亮在 80 行）：
  ```python
  data_dict = dict(dense_ogm=inputs[0],
                   bev_feat_map=inputs[1],
                   obj_label=labels['obj_label'],
                   velocity_flow=labels.get('velocity_flow'),
                   bev_roi_mask=labels.get('bev_roi_mask'),
                   sample_index=labels['sample_index'],
                   rl_sample_mask=labels.get('rl_sample_mask'),
                   lidar_front_mask=labels.get('lidar_front_mask'),
                   is_cls_only=labels.get('is_cls_only'),
                   obj_state=labels['obj_state'],
                   obj_label_source=labels['obj_label_source'],
                   obj_dir_cls_label=labels['obj_dir_cls_label'],
                   centerpoint_head_gt=labels.get('centerpoint_head_gt_rl', None),
                   obj_label_corner=labels.get('obj_label_corner', None))
  if self.det_head.enable_corner_det:
      data_dict.update(dict(obj_dense_label_corner=labels['obj_dense_label_corner'],
                            obj_dense_state_corner=labels['obj_dense_state_corner'],
                            obj_dense_dir_cls_label_corner=labels['obj_dense_dir_cls_label_corner'],
                            centerpoint_head_gt_corner=labels.get('centerpoint_head_gt_rl_corner', None)))
  ```
- 【为什么】注意 `labels[...]`（硬取，缺了就 KeyError）和 `labels.get(...)`（软取，缺了给 None）的混用是有信息量的：`obj_label/obj_state/sample_index` 是硬依赖，任何数据包必须有；`velocity_flow/bev_roi_mask` 等是可选监督，没有就跳过对应 loss。角点检测标签只在 `enable_corner_det` 开启时才要求存在。
- 【连接】`velocity_flow`（速度光流图）、`bev_roi_mask`（BEV 感兴趣区掩码）、`obj_label_source`（标注来源，大概率区分人工标注/自动标注——数据闭环里 lidar 自动标注产生的 GT 要降权）这些 key 会在下一章 Loss 里逐个登场，先混个脸熟。

---

#### 卡10-21｜[01:21:02] "这个是做可视化用的；这个是时序融合完之后的那个特征，就是这个"
（合并 [01:21:02(1200)][01:21:04(1201)][01:21:11(1202)][01:21:12(1203)][01:21:13(1204)]，讲者鼠标指认 dense_ogm 与 bev_feat_map 两个字段）

- 【直译】对 data_dict 前两个字段的现场指认：`dense_ogm=inputs[0]` 可视化用；`bev_feat_map=inputs[1]` 是时序融合（+UNet）后的主特征。
- 【代码】与卡10-10/10-11 一一对应，此处是从 data_dict 视角的复述。
- 【为什么】讲者反复指认同一批张量，因为它们在框图（draw.io）、DetHead 形参、data_dict 字段名三个层面各有一个名字（bev_multiview↔inputs[0]↔dense_ogm；bev_feat↔inputs[1]↔bev_feat_map）——**同一个张量三个名字**是读这份代码最大的认知负担，建议你自己列一张三列对照表。

---

#### 卡10-22｜[01:21:16] "这个是我们的 label。这个 label 其实是原始……出来的结果其实还是一个 BEV 的形式，所以说这个呢，是把我们的 GT 给它放到了 BEV 的 Map 上去" ⚠【重点句，5角度】
（合并 [01:21:16(1205)][01:21:28(1206)][01:22:02(1207)][01:22:05(1208)][01:22:06(1209)]；[01:21:28]–[01:22:02] 之间有约 34 秒讲者翻代码的静默；⚠"机械"为"GT"误听）

- 【直译】重点字段 `centerpoint_head_gt`：它不是一列框参数，而是已经"画"在 BEV 网格上的稠密标签图——把每个 GT 框转换成 heatmap 高斯峰 + 各属性图的形式。
- 【代码】CenterPoint 的标准 target 生成（mmdet3d 里叫 `get_targets`）：对每个 GT box，在其中心 (cx,cy) 落到的网格处画一个二维高斯 `heatmap[cls, y, x] = exp(-d²/2σ²)`（σ 由框尺寸定），同时在中心格记录 reg 偏移（中心在格内的小数部分）、height、dim、sinθ/cosθ、vx/vy 等回归目标。产出的 GT 张量组和网络输出的 10 路 `1×C×448×224` 一一同形。
- 【形状】以 heatmap GT 为例：5×448×224（5 类各一张）；各属性 GT 图与预测同形或以（最大目标数 N×属性维）的稀疏形式+index 存储。
- 【为什么】CenterPoint 的本质就是"检测=在 BEV 图上画热点+在热点处查表读属性"，所以 GT 必须先"光栅化"成图才能和预测逐像素算 loss。这一步在谁那儿做（GPU 训练时现做 or DataLoader 的 CPU worker 预做）就是下一卡的优化故事。
- 【连接】你精读过的 CenterPoint 论文 Sec.3 的 target assignment 就是这段的理论出处；BEVFusion 的 `CenterHead.get_targets` 是它的 mmdet3d 实现。DenseBEV 的差别只在于把这步挪进了数据管线。

---

#### 卡10-23｜[01:22:12] "这个是在之前做完优化之后，就不用在网络计算的时候，再去把我们的 GT 放到 BEV 的 Map 上去。现在用到的就是这个 CenterPoint Head 的 GT" ⚠
（合并 [01:22:12(1210)][01:22:15(1211)][01:22:19(1212)][01:22:27(1213)]；⚠"机械"同前为"GT"）

- 【直译】历史优化：以前 GT→BEV Map 的光栅化在网络训练循环里现算，后来挪到了数据侧预生成；现在训练时直接取 `labels['centerpoint_head_gt_rl']` 这个现成结果。
- 【代码】`centerpoint_head_gt=labels.get('centerpoint_head_gt_rl', None)`——key 名里的 `_rl` 后缀说明这份 GT 是为 RL 主检测分支准备的（skip_keys 名单里还有 `centerpoint_head_gt_fish`，鱼眼分支专用 GT）。
- 【为什么】画高斯 heatmap 是纯 CPU 逻辑（循环每个框、算σ、画核），放在 GPU 训练步里会卡主线程；挪到 DataLoader 的多进程 worker 里，与 GPU 前反向流水线并行，训练吞吐直接受益。代价是数据包体积变大（每帧多存几张 448×224 的图）。
- 【连接】你在 4060 上跑 BEVFusion 实测 0.481s/iter——如果 profile 发现 dataloader 时间占比高或 GPU 等待，这类"target 预计算"就是同款优化方向。mmdet3d 默认是在 loss 里现算 target 的（`get_targets` 每步调用），量产代码选了另一边，这个 trade-off 值得写进你的面试素材。

---

#### 卡10-24｜[01:22:32] "然后这些变量都是算网络需要用到的"
（对应 1214）

- 【直译】data_dict 余下字段（obj_state、obj_label_source、obj_dir_cls_label……）都是 loss 计算的原料，此处不逐个展开。
- 【为什么】讲者在这里收口不展开，是因为这些 key 的真正含义要到 Loss 章（Ch11）才兑现；本章只需记住"检测头 forward 消费特征，loss 消费标签，两者都从 data_dict 拿"。
- 【连接】obj_dir_cls_label（朝向分类标签）对应本章后面的 dir_cls 头——朝向 360° 连续回归容易在 ±π 处跳变，配一个粗分类头兜底，是 SECOND 时代（direction classifier）传下来的老手艺。

---

### 🔨 动手练习 ch10-3：三帧 label 包 + skip_keys 抽取当前帧

```python
import torch

def label_extract(labels: dict, skip_keys=()):
    """复刻 seq_info_extract.label_extract 的核心语义：
    三帧堆叠的 key 取当前帧（最后一帧），skip_keys 里的单帧 key 原样保留。"""
    out = {}
    for k, v in labels.items():
        if k in skip_keys:
            out[k] = v                    # 天生单帧：不动
        else:
            out[k] = v[-1]                # 三帧结构：取当前帧
    return out

labels = {
    'depth_gt':   torch.randn(3, 21, 88, 160),   # 三帧堆叠（图像模块的额外label）
    'ego_pose':   torch.randn(3, 4, 4),          # 三帧位姿
    'obj_label':  torch.randint(0, 5, (7,)),     # 单帧：当前帧7个目标的类别
    'centerpoint_head_gt_rl': torch.randn(5, 56, 28),  # 单帧：预生成的BEV heatmap GT
}
skip_keys = ['obj_label', 'centerpoint_head_gt_rl']
cur = label_extract(labels, skip_keys)

for k, v in cur.items():
    print(f'{k:24s} {tuple(v.shape)}')
# 预期输出：
# depth_gt                 (21, 88, 160)   ← 3帧 -> 当前帧
# ego_pose                 (4, 4)
# obj_label                (7,)            ← skip：原样
# centerpoint_head_gt_rl   (5, 56, 28)     ← skip：原样
```

**【小结】** ① forward 入口用 `labels = kwargs['labels'][0]` 接住 DataLoader 的全任务标签总线，再用 skip_keys+label_extract 把三帧包抽成当前帧包。② data_dict 是检测头内部的统一集散地，硬依赖用 `[]`、可选监督用 `.get()`，同一张量存在"框图名/inputs序号/data_dict名"三套命名。③ GT→BEV Map 的光栅化已从训练时现算优化为 DataLoader 预生成（`centerpoint_head_gt_rl`），训练主循环只管取现成的。

---

## Part 4｜lidar_rt / lidar_vel 切片与 CenterHead 入口（01:22:46–01:24:51）

**导读**：本段是 det_head.py 100–128 行（帧证 `01_22_45.jpg`/`01_23_16.jpg`）到 centerpoint 头文件 `CenterHead.forward_single`（帧证 `01_24_51.jpg`）的衔接段。输入是 Part 3 组好的 data_dict 和 5 路 inputs，动作有三个：把三帧 RL 特征切出当前帧（`[:, -1]`）、把时序 RL 特征直接挂进 data_dict、把 mid_feat 透传；然后 `det_output = self.det_head(data_dict)` 一脚踢进 CenterPoint 头，主特征先过一个共享卷积 shared_conv，再进 task_heads 循环。这段代码短，但"哪路特征在哪一行被挂上哪个 key"决定了你能否读懂 Part 5 的分支分发。

---

#### 卡10-25｜[01:22:46] "这个 lidar_rot 主要就是刚刚说的用了这个 BEV embedding……它是取出来的还是三帧的 RL 的 feature，然后在这里会只取出最后一维，就是时序的最后一维，就是一个单帧的 feature" ⚠【重点句，5角度】
（合并 [01:22:46(1215)][01:22:53(1216)][01:22:59(1217)][01:23:02(1218)][01:23:03(1219)]；⚠转写"起初数的"疑为"取出来的"，按上下文取后者）

- 【直译】处理 `lidar_rt`（rt=rot/rotation 相关）特征：inputs[2] 还是三帧堆叠的 RL 特征，这里只切出时序维的最后一帧，得到单帧特征。
- 【代码】帧证 `01_23_16.jpg` 第 101–106 行（黄条高亮 101 行）：
  ```python
  if self.use_lidar_rt:                      # yaml: use_lidar_rt: True
      assert len(inputs) >= 3
      bs = inputs[1].shape[0]                # 用主特征反推真实batch=1
      bst, c, h, w = inputs[2].shape         # bst = bs*3 = 3
      lidar_rt_ft = inputs[2].view(bs, bst // bs, c, h, w)   # [1,3,64,448,224]
      data_dict['lidar_rt_feat'] = lidar_rt_ft[:, -1, :]     # [1,64,448,224]
  ```
- 【形状】(1\*3)×64×448×224 → view → 1×3×64×448×224 → `[:, -1]` → **1×64×448×224**。
- 【为什么】① 为什么 bs 要从 `inputs[1]` 反推而不是写死？因为 batch 可变而帧数固定 3，`bst // bs` 自动算出帧数，兼容任何 batch；② 为什么取 `-1`？数据约定时序升序排列（t-2, t-1, t），最后一维就是当前帧——yaw 优化要的正是**当前时刻**的无拖影快照，历史两帧直接丢弃。
- 【连接】`view + 切片` 而不是 `torch.split`，零拷贝零开销；这与练习 ch10-2 完全对应。另外注意：转写里讲者说的"lidar_rot"、代码 key 是 `lidar_rt_feat`、yaml 开关叫 `use_lidar_rt`——三个名字一个东西，继续印证卡10-21 的"命名三胞胎"现象。

---

#### 卡10-26｜[01:23:06] "这个 Input3 呢，对应的就是这个做完时序融合之后的 feature。它其实就只有一帧了，所以说这里就不会再像上面取负 1 了"
（合并 [01:23:06(1220)][01:23:08(1221)][01:23:09(1222)][01:23:13(1223)][01:23:14(1224)][01:23:16(1225)][01:23:19(1226)]）

- 【直译】inputs[3]（时序融合后的 RL 特征）本来就是单帧形态，直接用，不需要再做 `[:, -1]` 切片。
- 【代码】帧证同上，第 107–112 行：
  ```python
  if self.use_lidar_vel:                     # yaml: use_lidar_vel: True
      assert len(inputs) >= 4
      bs = inputs[3].shape[0]
      bst, c, h, w = inputs[3].shape
      lidar_vel_feat = inputs[3]             # 已是 [1,64,448,224]，无需切
      data_dict['lidar_vel_feat'] = lidar_vel_feat
  ```
- 【形状】1×64×448×224 原样挂进 data_dict['lidar_vel_feat']。
- 【为什么】时序融合模块的输出天然把三帧并成一帧（warp+融合），所以第一维是 bs 而非 bs×3——**特征是否需要切片，取决于它在流水线里是否已经过时序融合**，这是判断这类代码的通用法则。（细看代码 `bst, c, h, w = inputs[3].shape` 这行解包了却没用 bst，是上面分支复制粘贴留下的冗余——工业代码常态。）
- 【连接】BEVFusion 没有这种"生/熟特征并行送头"的设计——它所有时序信息都在特征里混死了。DenseBEV 把"生特征（单帧）"和"熟特征（时序）"同时保留到头上，相当于给不同属性留了选择权。

---

#### 卡10-27｜[01:23:22] "这个没有……这个是要传给端到端要用的"
（合并 [01:23:22(1227)][01:23:38(1228)][01:23:39(1229前半)]，讲者短暂找代码）

- 【直译】接下来一段代码（mid_feat 透传、fusion_feat 相关）与检测本体无关：mid_feat 是给端到端用的。
- 【代码】帧证 `01_23_16.jpg` 第 114–122 行：
  ```python
  bev_mid_feat_list = []
  if self.export_instance_embeddings:        # yaml: True
      bev_mid_feat_list = inputs[4]
  fusion_feat = inputs[1].new_zeros(*inputs[1].shape)
  if self.use_fusion_instance_embeddings:    # 当前未启用
      vision_feat = inputs[5]
      fusion_feat = torch.cat([inputs[1], vision_feat, data_dict['lidar_rt_feat']], dim=1)
      fusion_feat = self.fusion_conv(fusion_feat)
  ```
- 【为什么】`new_zeros` 造一个同形占位的 fusion_feat，保证下游接口即便功能未启用也拿得到形状正确的张量——接口稳定性优先于省一次显存分配，这是量产代码防御式编程的典型样本。讲者没展开 `use_fusion_instance_embeddings` 分支（inputs[5]、vision_feat），说明当前配置未启用。
- 【连接】与卡10-8 呼应：`export_instance_embeddings: True`（帧证 `01_25_13.jpg` yaml 可见）在当前版本是开着的，mid_feat 真的在往端到端送。

---

#### 卡10-28｜[01:23:39] "然后是我们检测头，这里会进入到这个 CenterPoint 的 Head，把刚刚的那个 data_dict 放到我们这个 Head 里面去"
（合并 [01:23:39(1229)][01:23:46(1230)][01:23:50(1231)][01:23:54(1232)]）

- 【直译】wrapper 的组装工作完毕，一行调用进入真正的检测头：CenterPoint 风格的 CenterHead，整个 data_dict 作为参数传入。
- 【代码】帧证 `01_23_16.jpg` 第 125–128 行：
  ```python
  # 前向网络                                ← 源码中文注释原文
  det_output = self.det_head(data_dict)
  if self.vis_heatmap:
      self.det_head.vis_heatmap(self.vis_base_dir, self.vis_at_train, ego_loc=self.ego_loc)
  ```
- 【为什么】DetHead（wrapper：管输入适配、标签抽取）与 det_head/CenterHead（算法本体：管卷积和预测）分层，可视化钩子挂在 wrapper 层。yaml 里 `det_head: arch: centerpoint` 表明本体是注册表按 arch 字段实例化的——换检测范式只需改一行配置。
- 【连接】等价于 mmdet3d 里 `TransFusionHead/CenterHead` 被 detector 类（如 BEVFusion 的 `bevfusion.py`）包一层调用的关系。

---

#### 卡10-29｜[01:23:57] "这里的 CenterPoint 把我们的 BEV feature Map……对应的就是这个 feature；X 说的 feature 对应的就是单帧的 RL feature，以及时序融合完之后的一个 RL feature。主要是输入这三个再去做网络预测"
（合并 [01:23:57(1233)][01:24:13(1234)][01:24:15(1235)][01:24:24(1236)][01:24:27(1237)]）

- 【直译】CenterHead 真正消费三路张量：主 BEV 特征（bev_feat_map）+ 单帧 RL 特征（lidar_rt_feat）+ 时序 RL 特征（lidar_vel_feat）。
- 【代码】CenterHead 内部从 data_dict 取 `x = data_dict['bev_feat_map']`，而 `lidar_rt_feat/lidar_vel_feat` 以 kwargs 形式一路下传到 SeparateHead（Part 5 可见 `kwargs['lidar_rt_feat']`、`kwargs['lidar_vel_feat']` 的消费点）。
- 【形状】三路都是 1×64×448×224。
- 【为什么】到这里可以画出最终的**特征分工表**：

  | 特征 | 来源 | 消费的头 |
  |---|---|---|
  | bev_feat_map（主） | 时序融合+UNet | reg/height/dim/dir_cls/heatmap/close_heatmap + rot(加权融合的一半) |
  | lidar_rt_feat（单帧RL） | RL融合直出，取[-1]帧 | rot_lidar（独享）、rot（simple_fusion 的另一半） |
  | lidar_vel_feat（时序RL） | RL特征过时序融合 | vel、movement（经 conv_3x3 与主特征拼接融合） |

- 【连接】这张表就是本章的"一图流"。CenterPoint 原版只有第一行；后两行是 DenseBEV 面向 yaw 精度与速度精度的增量创新。

---

#### 卡10-30｜[01:24:32] "在这里把这里的 X 呢就是对应的 BEV feature。在这里会有额外的一个卷积"【重点句，5角度】
（合并 [01:24:32(1238)][01:24:44(1239)]）

- 【直译】进入 CenterHead 前向：主特征 x 先过一个"额外的卷积"——即所有任务分支共享的 shared_conv，然后才分发给各个 task head。
- 【代码】帧证 `01_24_51.jpg`（centerpoint 文件，`class CenterHead(nn.Module)` 246 行起）逐字：
  ```python
  def forward_single(self, x, **kwargs):
      """Forward function for CenterPoint.
      Args:
          x (torch.Tensor): Input feature map with the shape of [B, 512, 128, 128].
      Returns:
          list[dict]: Output results for tasks. """
      ret_dicts = []
      # todo(pilei)
      if self.enhance_vision:
          x = self.vision_network(x)
      x = self.shared_conv(x)
      for task in self.task_heads:
          ret_dicts.append(task(x, **kwargs))
      return ret_dicts

  def forward(self, feats, **kwargs):
      if isinstance(feats, torch.Tensor):
          feats = [feats]
      return multi_apply(self.forward_single, feats, **kwargs)
  ```
- 【形状】x: 1×64×448×224 → shared_conv（yaml `share_conv_channel: 64`，惯例是 3×3 Conv+BN+ReLU）→ 1×64×448×224，尺寸通道都不变。
- 【为什么】① shared_conv 的作用：把 backbone 特征先"翻译"成检测友好的表示，10 个分支共享这次翻译，比每个分支各自翻译省 10 倍算力；② docstring 里 `[B, 512, 128, 128]` 又是 mmdet3d 原版注释的化石（nuScenes 配置），实际是 [1,64,448,224]——本章第三处"注释化石"；③ `multi_apply` + feats 列表化：保留了 mmdet3d 多尺度多 task 的接口形态，但这里只有一个尺度一个 task，循环只跑一圈；④ `# todo(pilei)` 和 `enhance_vision` 说明作者 pilei（屏幕水印同名，即讲者本人或同组）还想在头前加视觉增强网络。
- 【连接】打开你机器上 mmdet3d 的 `centerpoint_head.py` 对比：`forward_single` 里 `x = self.shared_conv(x); for task in self.task_heads: ...` 逐行同源——DenseBEV 的 CenterHead 是 mmdet3d 的 fork 改造，你读熟 BEVFusion 的头，这份代码就是熟人。

---

### 🔨 动手练习 ch10-4：CenterHead 骨架（shared_conv + task_heads 循环）

```python
import torch, torch.nn as nn

class TinyTaskHead(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv = nn.Conv2d(c, 5, 3, padding=1)   # 假装是5类heatmap分支
    def forward(self, x, **kwargs):
        return {'heatmap': self.conv(x),
                'got_lidar_rt': kwargs['lidar_rt_feat'].shape}  # 证明kwargs一路透传

class TinyCenterHead(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.shared_conv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1),
                                         nn.BatchNorm2d(c), nn.ReLU())
        self.task_heads = nn.ModuleList([TinyTaskHead(c)])
    def forward_single(self, x, **kwargs):
        x = self.shared_conv(x)                     # “额外的一个卷积”
        return [task(x, **kwargs) for task in self.task_heads]

head = TinyCenterHead()
x = torch.randn(1, 64, 112, 56)                     # 真实为 1x64x448x224
out = head.forward_single(x, lidar_rt_feat=torch.randn(1, 64, 112, 56))
print(out[0]['heatmap'].shape)     # torch.Size([1, 5, 112, 56])
print(out[0]['got_lidar_rt'])      # torch.Size([1, 64, 112, 56]) ← kwargs穿透成功
```

**【小结】** ① inputs[2]（三帧 RL）经 `view(bs,3,c,h,w)[:, -1]` 切出当前帧挂到 `lidar_rt_feat`；inputs[3]（时序 RL）已是单帧直接挂 `lidar_vel_feat`。② `det_output = self.det_head(data_dict)` 进入 mmdet3d 血统的 CenterHead，主特征先过 shared_conv 再进 task_heads 循环，辅助特征以 kwargs 穿透到最里层。③ 特征分工表：主特征管大多数属性，单帧 RL 管 yaw（rot_lidar/rot），时序 RL 管速度与动静（vel/movement）。

---

## Part 5｜heads 字典逐个拆解与 SeparateHead 的 __getattr__ 分发（01:24:53–01:28:48）

**导读**：本段是全章信息密度最高的部分，对应两块屏幕内容：左边 yaml 配置 `common_heads`（帧证 `01_25_13.jpg`，文件 `20250523_v47.1_ai_dense.yml`），右边 `SeparateHead.forward`（帧证 `01_26_44.jpg`）。输入是 shared_conv 后的 x 和 kwargs 里的两路 RL 特征，输出是 `ret_dict`——10 个 key 各挂一张 `1×C×448×224` 预测图。要抓住的主线只有两条：① heads 字典 `{名字: [输出通道数, 卷积层数]}` 用 `setattr/__getattr__` 动态生成/取用分支；② forward 循环里三组 if-else 按 head 名字把不同特征分发给不同分支——正是 Part 2 铺垫的"属性适配特征"落地成代码的地方。

---

#### 卡10-31｜[01:24:53] "在这里就是去预测我们每个目标它的一些相关的一些属性，对应的属性呢，有这些"
（合并 [01:24:53(1240)][01:24:57(1241)][01:25:04(1242)]）

- 【直译】SeparateHead 的职责：对每个（潜在）目标预测一组属性；属性清单就在配置里。
- 【代码】帧证 `01_25_13.jpg`，yaml 原文（`det_head:` 小节）：
  ```yaml
  det_head:
    arch: centerpoint
    with_track_task: True
    vel_spec: True          # 速度头走专用特征
    rot_spec: True          # 朝向头走专用特征
    close_rot_spec: True
    close_heatmap_cls: True
    dense_attr: True
    neg_vel_loss: 1.0
    in_channels: 64
    tasks:
      - {num_class: *DET_CLASS_NUM, class_names: ...}
    common_heads:
      reg: [2, 2]
      height: [1, 2]
      dim: [3, 2]
      rot: [2, 2]
      vel: [2, 2]
      dir_cls: [1, 2]
      movement: [1, 2]
      rot_lidar: [2, 2]
      close_heatmap: [*DET_CLASS_NUM, 2]
    share_conv_channel: 64
    bbox_coder:
      pc_range: *RL_PC_RANGE
      post_center_range: *POST_CENTER_RANGE
      max_num: *MAX_NUM
      out_size_factor: 1
      voxel_size: *VOXEL_SIZE
      code_size: 9
    separate_head:
      type: SeparateHead
      init_bias: -4.5951
      final_kernel: 3
  ```
- 【为什么】"稠密属性预测"（`dense_attr: True`）是 CenterPoint 范式的精髓：不是先出框再分类属性，而是**每个 BEV 网格位置并行预测全部属性**，解码时只在 heatmap 峰值处"读表"。属性清单可配置化（common_heads 字典），加一个属性=加一行 yaml。
- 【连接】对照 mmdet3d nuScenes CenterPoint 配置 `common_heads=dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2))`——前 5 项一字不差，后 4 项（dir_cls/movement/rot_lidar/close_heatmap）是 DenseBEV 的量产增量。`code_size: 9` 对应解码后 box 的 9 维 [x,y,z,l,w,h,yaw,vx,vy]。

---

#### 卡10-32｜[01:25:09] "对应的属性有 reg，reg 对应的就是长宽——哦，这个是中心点的位置 XY；然后这个是它高的位置；然后这个是它的长宽高" ⚠【重点句，5角度】
（合并 [01:25:09(1243)][01:25:11(1244)][01:25:14(1245)][01:25:16(1246)][01:25:19(1247)][01:25:22(1248)][01:25:24(1249)]；⚠讲者先说"reg对应长宽"随即自纠为"中心点位置XY"，以自纠后为准）

- 【直译】逐项拆前三个头：`reg`=中心点 XY 的亚像素偏移（2 通道）；`height`=目标中心的 Z 高度（1 通道）；`dim`=长宽高三个尺寸（3 通道）。
- 【代码】`reg: [2, 2]`、`height: [1, 2]`、`dim: [3, 2]`。解码时（下一章）：`x = (peak_x + reg[0]) * voxel_size * out_size_factor + pc_range[0]`，`z = height`，`l,w,h = exp(dim)`（dim 一般回归 log 尺寸保证正数）。
- 【形状】reg → 1×2×448×224；height → 1×1×448×224；dim → 1×3×448×224（与卡10-40 调试台实测一致）。
- 【为什么】① reg 为什么必要：heatmap 峰值只能定位到 0.4m 的格心，reg 补上格内小数偏移，把定位精度从 ±0.2m 提到厘米级；② height 为什么单列：BEV 表示把 Z 压扁了，高度信息只能靠回归找回来——这正是 BEV 范式"绕过 Z 轴再补 Z 轴"的代表性补丁；③ 讲者口误"reg是长宽"又自纠，说明 reg/dim 极易混淆——记法：**reg**ression offset 管"在哪"，**dim**ension 管"多大"。
- 【连接】CenterPoint 论文 Eq.(1)-(3) 的 o（offset）、s（size）、z（height）三件套；你在 BEVFusion 输出解码 `bbox_coder.decode` 里见过完全相同的算式。

---

#### 卡10-33｜[01:25:26] "然后这个是 rot，就是会预测它的 yaw，就是 sinθ 和 cosθ"
（合并 [01:25:26(1250)][01:25:31(1251)]）

- 【直译】`rot` 头预测航向角 yaw，但不直接回归角度值，而是回归它的正弦和余弦两个分量。
- 【代码】`rot: [2, 2]` → 输出 2 通道 `[sinθ, cosθ]`；解码 `yaw = atan2(sinθ, cosθ)`。
- 【形状】1×2×448×224。
- 【为什么】直接回归 θ 有 **2π 周期跳变**问题：θ=179° 和 θ=-179° 物理上只差 2°，数值上差 358°，L1 loss 会疯狂惩罚一个几乎正确的预测。映射到 (sinθ, cosθ) 后，角度空间的邻近性=向量空间的邻近性，回归目标连续光滑。代价是两自由度冗余（‖(sin,cos)‖ 未必=1），由 atan2 自然消化。
- 【连接】SECOND 用 sin(θ_pred−θ_gt) 编码，CenterPoint 换成 (sin,cos) 双通道——你看过的谢博原理课里"朝向角编码"一节讲的就是这条演化线。

---

#### 卡10-34｜[01:25:36] "然后这个是我们的一个朝向的分类，然后是动静的分类，然后还有优化 yaw 的一个回归头，然后这个是近距离的一个回归头"
（合并 [01:25:36(1252)][01:25:39(1253)][01:25:41(1254)][01:25:45(1255)]）

- 【直译】剩下四个非标准头：`dir_cls`=朝向分类（1 通道）；`movement`=动/静二分类（1 通道）；`rot_lidar`=用单帧 RL 特征再回归一次 yaw 的增强头（2 通道 sinθ/cosθ）；`close_heatmap`=近距离目标的加强分类头（5 通道）。
- 【代码】`dir_cls: [1, 2]`、`movement: [1, 2]`、`rot_lidar: [2, 2]`、`close_heatmap: [*DET_CLASS_NUM, 2]`（DET_CLASS_NUM=5）。
- 【形状】1×1、1×1、1×2、1×5 ×448×224。
- 【为什么】① dir_cls：sin/cos 回归偶尔会差个 180°（车头车尾对称性导致），单通道二分类（正向/反向）专门纠这个 flip 错误——1 通道+sigmoid 即二分类；② movement：动静标志直接输出给下游 tracker/规划（静止目标可走更强的位置滤波），也与 vel 头互为校验；③ rot_lidar：与 rot 双保险，loss 各自监督，推理可加权融合；④ close_heatmap：近距离（如 AEB 责任区）目标漏检代价极大，单独再出一张 5 类 heatmap 做加强监督——同一个目标在近处会同时出现在 heatmap 和 close_heatmap 里。
- 【连接】mmdet3d 的 SECOND/PointPillars 有 `dir_offset/dir_limit_offset` 的方向分类器，dir_cls 是同一思想在 CenterPoint 上的移植。close_heatmap 则没有开源对应物，是量产安全需求的定制产物（呼应 yaml `close_heatmap_cls: True`、`close_rot_spec: True`）。

---

#### 卡10-35｜[01:25:49] "所有的这些回归头，前面的话就是代表它的预测在 BEV feature 上的一个 Channel，其实就是这个属性它的一个维度"【重点句，5角度】
（合并 [01:25:49(1256)][01:26:02(1257)]）

- 【直译】配置 `[a, b]` 里第一个数 a 的含义：该头输出张量的通道数=该属性的自由度。
- 【代码】构建端（mmdet3d SeparateHead.__init__ 同款逻辑）：
  ```python
  for head, (out_c, num_conv) in self.heads.items():
      layers = []
      c_in = in_channels                       # 64
      for i in range(num_conv - 1):
          layers += [nn.Conv2d(c_in, head_conv, 3, padding=1, bias=False),
                     nn.BatchNorm2d(head_conv), nn.ReLU()]
          c_in = head_conv
      layers.append(nn.Conv2d(c_in, out_c, final_kernel,      # final_kernel: 3
                              padding=final_kernel // 2, bias=True))
      self.__setattr__(head, nn.Sequential(*layers))
  ```
- 【形状】通道数即语义维度：位置 2、高度 1、尺寸 3、角度 2（sin/cos）、速度 2（vx/vy）、二分类 1、多分类 5。
- 【为什么】把"属性维度"暴露成配置第一位，本质是把**输出协议**写进 yaml：下游解码器按同一份配置切通道，网络与解码永不失配。加一个属性（比如再加个"拖挂角"）只需 yaml 加一行，代码零改动——配置驱动架构的教科书示范。
- 【连接】智谷代码课里"Conv2d 的 out_channels 决定输出特征图数"在这里的物理意义被具体化了：**每一张输出特征图就是一张属性地图**——第 0 张图上 (i,j) 的值 = "如果 (i,j) 处有目标，它的 x 偏移是多少"。

---

#### 卡10-36｜[01:26:05] "然后 2 呢……后面这个列表里面最后一位呢，只是代表有多少个卷积操作。在这里所有的都是两个卷积"
（合并 [01:26:05(1258)][01:26:07(1259)][01:26:10(1260)][01:26:12(1261)][01:26:20(1262)]，讲者此处两次口头绕圈，合并为一卡）

- 【直译】配置第二个数 b=该分支的卷积层数；本工程所有头统一为 2。
- 【代码】即上一卡代码中 `num_conv=2` 的情形：`Conv3×3(64→64)+BN+ReLU → Conv3×3(64→out_c)`。每个头就这么浅。
- 【形状】64ch → 64ch → out_c，空间 448×224 全程不变（stride=1, padding=1）。
- 【为什么】头浅的两个理由：① shared_conv 已做共同表征加工，分支只需"最后一公里"的属性特化；② 10 个分支×448×224 全分辨率，每加一层是 10 倍的代价——`2 层` 是精度/算力平衡点。而 final 层带 bias（前层 BN 所以 free bias），正是为了 heatmap 的 init_bias 技巧（见练习 ch10-6）。
- 【连接】mmdet3d SeparateHead 默认 `num_conv=2`、`final_kernel=3`，与此完全一致；量产版没有为了省算力砍到 1 层，说明属性回归质量确实需要这一层缓冲。

---

#### 卡10-37｜[01:26:25] "在这里，我们预测速度和动静的时候要结合时序的 RL feature。所以在这里会把原始的模态时序融合之后的图像和 RL 的 feature，额外再和时序融合的 RL feature，用这个去做一个增强"【重点句，5角度】
（合并 [01:26:25(1263)][01:26:42(1264)][01:26:51(1265)][01:26:59(1266)]）

- 【直译】速度/动静分支的输入不是裸 x，而是"主特征 concat 时序 RL 特征，再过一个 3×3 卷积融合"的增强特征。
- 【代码】帧证 `01_26_44.jpg` 第 208 行（调试黄条所在行）：
  ```python
  feat_fus_lidar_vel = self.conv_3x3(torch.cat((x, kwargs['lidar_vel_feat']), 1)) \
                       if self.vel_spec else None
  ```
  注意它在 for 循环**之前**只算一次，vel 和 movement 两个分支共享。
- 【形状】cat: 1×(64+64)×448×224=1×128×448×224 → conv_3x3 → 1×64×448×224（压回统一宽度）。
- 【为什么】① 为什么 concat 而不是相加？加法要求两路语义对齐，concat+卷积让网络自己学习怎么混合——两路特征一"熟"（主特征）一"专"（时序 RL），语义差异大，concat 更稳；② 为什么提前算好？两个分支要用同一份，算一次省一次；③ `if self.vel_spec else None`：vel_spec 关掉时速度头退回用裸 x，功能开关粒度精确到特征选择。
- 【连接】这行代码就是 Part 2 卡10-14 那句话的落点。BEVFusion 的 fuser（ConvFuser：cat+conv）在模态间干的事，这里在"任务特征定制"层面又干了一次——cat+conv 是 BEV 系代码出现频率最高的融合原语，值得你写进笔记的"惯用招式"清单。

---

#### 卡10-38｜[01:27:08] "这个 key 其实就是一个 Key。在这里会通过这个 key……__getattr__，然后这个 Key 去获得对应的每一个模块它的卷积、它的回归头"【重点句，5角度】
（合并 [01:27:08(1267)][01:27:12(1268)][01:27:15(1269)][01:27:16(1270)][01:27:17(1271)][01:27:23(1272)][01:27:25(1273)]）

- 【直译】forward 里遍历 heads 字典的 key（字符串"reg"、"dim"……），用 `self.__getattr__(key)` 把同名的卷积分支模块取出来调用。
- 【代码】帧证 `01_26_44.jpg` 第 209–227 行全文：
  ```python
  for head in self.heads:
      if head == 'mov_two_stage':
          continue                                   # 二阶段动静头最后单独算
      if self.rot_spec and head.startswith('rot'):
          if head == 'rot_lidar':
              feat_in = kwargs['lidar_rt_feat']      # 纯单帧RL
          elif head == 'rot':
              feat_in, lidar_weight = self.simple_fusion(x, kwargs['lidar_rt_feat'])
              ret_dict['lidar_rot_weight'] = lidar_weight
          else:
              raise NotImplementedError
          ret_dict[head] = self.__getattr__(head)(feat_in)
      elif self.vel_spec and (head.startswith('vel') or head.startswith('mov')):
          feat_in = feat_fus_lidar_vel               # 时序增强特征
          ret_dict[head] = self.__getattr__(head)(feat_in)
      else:
          ret_dict[head] = self.__getattr__(head)(x) # 其余用主特征
  ```
- 【形状】每个分支输出 1×out_c×448×224，按 key 存入 ret_dict。
- 【为什么】`__setattr__(name, module)` 注册 + `__getattr__(name)` 取用，等价于"用字符串索引子网络"。nn.Module 重载过 `__getattr__`：属性查找会落到 `self._modules` 字典里，所以字符串 key 能找到对应 nn.Sequential，且参数注册、`.cuda()`、`state_dict()` 都自动生效。用 `nn.ModuleDict` 语义相同、写法更现代；mmdet3d 历史代码选了 `__setattr__` 风格，DenseBEV 照单全收。
- 【连接】校正稿文件头特别注明"__getattr__（帧证校正）"——原始转写把这个词听成了乱码，是本视频的高频误听点；你以后搜转写原稿时注意。这段 if-else 也是本章"特征分工表"（卡10-29）的最终代码兑现。

---

#### 卡10-39｜[01:27:27] "在遍历的时候，对于位置、高，以及长宽高，还有 close_heatmap、heatmap——heatmap 的话就是分类的；close_heatmap 主要是对于近距离一定范围内的目标作为一个加强分类的回归头"
（合并 [01:27:27(1274)][01:27:40(1275,1276)][01:27:42(1277)][01:27:46(1278)][01:27:47(1279)][01:27:48(1280)][01:27:49(1281)][01:27:51(1282)][01:27:54(1283)]）

- 【直译】走 else 分支（用主特征 x）的头：reg（位置）、height（高）、dim（长宽高）、dir_cls、heatmap、close_heatmap。heatmap 是主分类头；close_heatmap 是近距离目标的加强版分类头。
- 【代码】它们都命中 `ret_dict[head] = self.__getattr__(head)(x)`。heatmap 不在 yaml `common_heads` 里，是 CenterHead 构建 task 时按 `num_class` 自动追加的（调试台 `self.heads` 里能看到它：`'heatmap': [5, 2]`，帧证裁图 `crops/heads5.png`）。
- 【形状】heatmap/close_heatmap 均 1×5×448×224。
- 【为什么】heatmap 与 close_heatmap 的分工：前者全图 179m×90m 范围统一监督，远处目标像素少、响应弱，训练信号被海量远处负样本稀释；后者只在近距离范围内计算 loss（范围外 mask 掉），等于给"责任区"开小灶，提升近距离召回。推理时近处可用 close_heatmap 的分数增强置信度。⚠ 近距离的具体米数阈值本段未给出，待 Loss 章或配置核实。
- 【连接】这是"距离分层监督"的思想，与鱼眼/针孔分工（近处鱼眼、远处针孔，Ch6）同构——量产感知永远在对"近处保安全、远处保预见"做资源倾斜。

---

#### 卡10-40｜[01:28:08] "对于朝向的回归头的话，它是走这个模块；然后对于动静和速度的话会走这里。因为他们每一个其实输入的特征是不一样的"
（合并 [01:28:06(1284)][01:28:08(1285)][01:28:10(1286)][01:28:14(1287)][01:28:21(1288)]）

- 【直译】复述分发规则收口：rot 系走 simple_fusion/lidar_rt_feat 那个分支，vel/movement 走 feat_fus_lidar_vel 分支——**分支存在的唯一原因就是各头输入特征不同**。
- 【代码】即卡10-38 的三组 if-else；`head.startswith('rot')`/`startswith('vel')/startswith('mov')` 的字符串前缀路由意味着**头名字就是路由协议**：新加一个 `rot_xxx` 头会自动进 rot 通道——好处是零配置，坏处是名字起错=特征喂错，且 `raise NotImplementedError` 只兜住了 rot 前缀的意外。
- 【为什么】把"特征选择"写在 forward 的 if-else 而不是配置里，是可读性与灵活性的折中：特征路由种类少（3 种）且稳定，硬编码反而一目了然。
- 【连接】`simple_fusion(x, lidar_rt_feat)` 返回 `(feat_in, lidar_weight)` 双值，说明它是**可学习加权融合**（类似 attention gate：weight=sigmoid(conv(cat))，feat=x·(1-w)+rl·w），并且把权重图 `lidar_rot_weight`（1×1×448×224）也作为输出监督/可视化——你可以在调试时把这张权重图画出来，看网络在哪些区域更信任激光单帧特征（预期：动目标区域权重高）。

---

#### 卡10-41｜[01:28:29] "这个模块其实所有的都是一些卷积。然后完之后会放到 dict 里面"
（合并 [01:28:29(1289)][01:28:40(1290)]）

- 【直译】不管分发多花哨，每个分支本体都只是几层卷积；所有输出统一收进 ret_dict 字典返回。
- 【代码】`ret_dict[head] = ...` 逐个填充；补充一个讲者**没有讲**的细节（帧证 `01_26_44.jpg` 229–241 行）：
  ```python
  if 'mov_two_stage' in self.heads:            # 二阶段动静头（当前heads里未配置）
      if self.vel_spec:
          x = feat_fus_lidar_vel
          x = torch.cat((x, ret_dict['movement'], ret_dict['heatmap']), dim=1)
          ret_dict['mov_two_stage'] = self.mov_two_stage_head(x)
  # 保存使用过的bev_feat，用于后续取instance_embeddings   ← 源码中文注释原文
  used_bev_feat = x.clone()
  ret_dict['used_bev_feat'] = torch.nan_to_num(used_bev_feat, nan=0.0, posinf=0.0, neginf=0.0)
  return ret_dict
  ```
- 【为什么】① mov_two_stage：把一阶段的 movement/heatmap **输出**再 concat 回特征做二阶段动静预测（级联细化思想），本次配置未启用，讲者跳过，我们标注存档；② `torch.nan_to_num`：量产训练防御——上游任何数值事故（除零、log(0)）产生的 NaN/Inf 在这里被拍成 0，防止一颗 NaN 传染整个 loss。这两处都是"代码比讲解多"的部分，读代码时留意。
- 【连接】"一切皆卷积+字典路由"正是 CenterPoint 头优雅之处：没有 anchor 匹配、没有 RoIAlign，与你在二维检测里见过的 YOLO head（也是纯卷积输出属性图）精神同源——这也解释了为什么导师让你先吃透 YOLO 再来看 BEV 检测头。

---

### 🔨 动手练习 ch10-5：迷你 SeparateHead——__getattr__ 分发 + 三路特征路由

```python
import torch, torch.nn as nn

class MiniSeparateHead(nn.Module):
    def __init__(self, heads, c=64):
        super().__init__()
        self.heads, self.rot_spec, self.vel_spec = heads, True, True
        self.conv_3x3 = nn.Conv2d(2 * c, c, 3, padding=1)
        for name, (out_c, num_conv) in heads.items():          # [属性维数, 卷积个数]
            layers, cin = [], c
            for _ in range(num_conv - 1):
                layers += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU()]
            layers.append(nn.Conv2d(c, out_c, 3, padding=1))   # final_kernel=3
            self.__setattr__(name, nn.Sequential(*layers))     # 字符串key注册分支

    def forward(self, x, **kw):
        ret = {}
        fus_vel = self.conv_3x3(torch.cat((x, kw['lidar_vel_feat']), 1))
        for head in self.heads:
            if self.rot_spec and head.startswith('rot'):
                feat = kw['lidar_rt_feat'] if head == 'rot_lidar' else x  # 简化simple_fusion
            elif self.vel_spec and (head.startswith('vel') or head.startswith('mov')):
                feat = fus_vel
            else:
                feat = x
            ret[head] = self.__getattr__(head)(feat)           # ← 按key取头
        return ret

heads = {'reg': [2,2], 'height': [1,2], 'dim': [3,2], 'rot': [2,2], 'vel': [2,2],
         'dir_cls': [1,2], 'movement': [1,2], 'rot_lidar': [2,2],
         'close_heatmap': [5,2], 'heatmap': [5,2]}
h, w = 112, 56                                # 真实为448x224，等比缩小省算力
net = MiniSeparateHead(heads)
out = net(torch.randn(1,64,h,w), lidar_rt_feat=torch.randn(1,64,h,w),
          lidar_vel_feat=torch.randn(1,64,h,w))
for k, v in out.items():
    print(f'{k:14s} {tuple(v.shape)}')
# 预期输出（对照视频调试台，仅H W按比例小4倍）：
# reg            (1, 2, 112, 56)
# height         (1, 1, 112, 56)
# dim            (1, 3, 112, 56)
# rot            (1, 2, 112, 56)
# vel            (1, 2, 112, 56)
# dir_cls        (1, 1, 112, 56)
# movement       (1, 1, 112, 56)
# rot_lidar      (1, 2, 112, 56)
# close_heatmap  (1, 5, 112, 56)
# heatmap        (1, 5, 112, 56)
```

**【小结】** ① heads 字典 `{key: [属性维数, 卷积数]}` 驱动 `__setattr__` 建头、`__getattr__` 用头，每个头=1 层 3×3 特化卷积+1 层 3×3 输出卷积。② forward 用 head 名字前缀路由特征：rot 系→单帧 RL（rot_lidar 独享 / rot 经 simple_fusion 加权），vel/mov 系→主特征⊕时序 RL 的 conv 融合，其余→主特征。③ 代码里还有讲者未提的 mov_two_stage 二阶段头（未启用）和 nan_to_num 数值防御，读源码时不要漏。

---

## Part 6｜输出 shape 清单实测与 5 类定义（01:28:44–01:31:16）

**导读**：本段讲者在调试台现场打印验证：先看 `self.heads` 字典确认配置，再 `for k in ret_dict: print(k, ret_dict[k].shape)` 把 10 路输出+1 路权重图逐个过目。输入是上一 Part 的 ret_dict，"输出"是你脑子里的一张完整 shape 清单——这是全章可验证性最强的一段，也是你复现练习的对照答案。最后点了 5 个检测类别（heatmap 通道数=5 的来源）并把 used_bev_feat 存档，检测头收官。

---

#### 卡10-42｜[01:28:44] "为的第一个的……可以在这里打一眼线下（打个断点线下看一下）" ⚠
（合并 [01:28:44(1291)][01:28:46(1292)][01:28:48(1293)][01:29:21(1294)]；[01:29:21]–[01:29:51] 约 30 秒为讲者设断点、跑调试的静默操作；⚠"打一眼线下"为口误/误听，按操作实况取"打断点线下看"）

- 【直译】讲者话说一半改主意：与其口述每维含义，不如直接断点看真值。
- 【代码】调试台里执行的两条语句（帧证裁图 `crops/shapes.png` 逐字核对）：
  ```
  >>> self.heads
  {'reg': [2, 2], 'height': [1, 2], 'dim': [3, 2], 'rot': [2, 2], 'vel': [2, 2],
   'dir_cls': [1, 2], 'movement': [1, 2], 'rot_lidar': [2, 2],
   'close_heatmap': [5, 2], 'heatmap': [5, 2]}
  >>> for k in ret_dict:
  ...     print(k, ret_dict[k].shape)
  ```
- 【为什么】注意 `self.heads` 的运行时值比 yaml `common_heads` 多了 `'heatmap': [5, 2]`——证实 heatmap 是 CenterHead 构建时按 task 的 num_class 动态追加的（卡10-39 推断被运行时证据坐实）。**"配置是愿望，运行时是事实"**——学讲者这招：讲不清就 print。
- 【连接】你在 4060 上调 BEVFusion 时同款技巧：断点在 `head.forward` 出口，一句 dict comprehension 打全 shape，比读十分钟代码快。

---

#### 卡10-43｜[01:29:51] "对，然后这里就是说的我们中心点的 xy，它所回归出来的 BEV feature 的话就是 1×2×448×224；然后高的话 1×1——因为第二维就是预测的属性的个数" ⚠【重点句，5角度】
（合并 [01:29:51(1295)][01:29:57(1296)][01:30:01(1297)][01:30:03(1298)][01:30:04(1299)][01:30:08(1300)]；⚠转写"1乘244824"实为 1×2×448×224）

- 【直译】清单开念：reg（中心 xy）输出 1×2×448×224；height 输出 1×1×448×224。规律：第 0 维 batch，第 1 维=属性自由度，后两维=BEV 网格。
- 【代码】调试台实测全清单（帧证裁图逐行核对，这就是**本章的标准答案**）：
  ```
  reg              torch.Size([1, 2, 448, 224])   中心偏移 Δx Δy
  height           torch.Size([1, 1, 448, 224])   中心高度 z
  dim              torch.Size([1, 3, 448, 224])   尺寸 l w h
  lidar_rot_weight torch.Size([1, 1, 448, 224])   rot融合权重图
  rot              torch.Size([1, 2, 448, 224])   sinθ cosθ
  vel              torch.Size([1, 2, 448, 224])   vx vy
  dir_cls          torch.Size([1, 1, 448, 224])   朝向翻转分类
  movement         torch.Size([1, 1, 448, 224])   动/静
  rot_lidar        torch.Size([1, 2, 448, 224])   单帧RL版 sinθ cosθ
  close_heatmap    torch.Size([1, 5, 448, 224])   近距离5类加强
  heatmap          torch.Size([1, 5, 448, 224])   主5类分类
  ```
- 【形状】11 项相加通道数 2+1+3+1+2+2+1+1+2+5+5=25：全套预测就是一张 1×25×448×224 的"属性大图"按语义切片。100,352 个网格位置每处并行给出 25 维预测。
- 【为什么】遍历顺序有一个反直觉点：`lidar_rot_weight` 排在 rot 前面——因为 dict 按**插入顺序**，rot 分支执行时先 `ret_dict['lidar_rot_weight'] = lidar_weight` 后 `ret_dict['rot'] = ...`（见卡10-38 代码行序），打印顺序就是代码执行轨迹。用输出顺序反推执行顺序，是无源码调试时的实用技能。
- 【连接】对照 nuScenes CenterPoint：输出在 128×128 网格（out_size_factor=8）且无 lidar_rot_weight/dir_cls/movement/rot_lidar/close_heatmap 五项。一张清单看尽开源与量产的 delta。

---

#### 卡10-44｜[01:30:12] "然后 DIM 的话就是长宽高 3；然后 lidar_rot_weight 这个是优化 yaw 的，只有一个值"
（合并 [01:30:12(1301)][01:30:15(1302)][01:30:17(1303)][01:30:19(1304)]）

- 【直译】继续念清单：dim 3 通道=长宽高；lidar_rot_weight 1 通道=每个位置一个标量权重。
- 【代码】`lidar_rot_weight` 不在 heads 字典里建头，它是 `simple_fusion` 的副产物直接塞进 ret_dict 的（卡10-38），所以调试台里它有 shape 但 `self.heads` 里没有它——两张打印对不上的地方恰是理解代码结构的线索。
- 【形状】1×1×448×224，值域推测 [0,1]（sigmoid 门控）。
- 【为什么】把内部权重图也放进输出 dict：一为可视化排障（哪里在信任 lidar），二为可能的正则监督（约束权重分布）。多输出一张图的成本几乎为零，可解释性收益很大——值得抄的习惯。
- 【连接】与 BEVFusion 论文里"camera/lidar 特征逐元素自适应融合"的可视化图同一思想，只不过这里粒度是"任务内特征融合"。

---

#### 卡10-45｜[01:30:20] "然后这个是朝向角的 xy，就是 sinθ 和 cosθ；速度的话就是 vx、vy；然后朝向（分类）的话就只有一个；然后动静也是一个；然后这个是优化朝向的，也是 sinθ 和 cosθ"
（合并 [01:30:20(1305)][01:30:22(1306)][01:30:24(1307)][01:30:27(1308)][01:30:29(1309)][01:30:31(1310)][01:30:33(1311)]）

- 【直译】念完剩余回归项：rot=[sinθ,cosθ]、vel=[vx,vy]、dir_cls=1、movement=1、rot_lidar=[sinθ,cosθ]。
- 【形状】vel 的 2 通道是**米/秒制的 BEV 平面速度矢量**（vx 纵向、vy 横向），不是像素速度；解码后直接给 tracker 当运动先验。
- 【为什么】rot 与 rot_lidar 通道定义完全相同（都是 sin/cos），差异只在输入特征——这是**同构冗余头**设计：同一个量、两条证据链，loss 各自监督（下一章会看到 rot_lidar 的专门 loss），推理端可选融合或择优。冗余是安全域设计的常规武器。
- 【连接】nuScenes 的 velocity GT 由相邻帧 box 位置差分得到，本工程的 vel GT 同理来自时序标注差分（数据链在 skip_keys 的 velocity_flow 等 key 里露过面）。

---

#### 卡10-46｜[01:30:35] "因为我们现在 BEV 的话是检测 5 类，就是 truck、car……这 5 类，所以说这个（heatmap 通道）只能是 5" ⚠【重点句，5角度】
（合并 [01:30:35(1312)][01:30:38(1313)][01:30:41(1314)][01:30:43(1315)][01:30:44(1316)][01:30:46(1317)][01:30:47(1318)][01:30:49(1319)]；⚠转写"truck cut head 所以sinθ以及head这5类"严重损坏，仅 truck/car 可辨）

- 【直译】heatmap 和 close_heatmap 的通道数 5 由检测类别数决定：BEV 当前检测 5 个类别。
- 【代码】yaml：`tasks: - {num_class: *DET_CLASS_NUM, ...}`、`close_heatmap: [*DET_CLASS_NUM, 2]`——类别数用锚点 `*DET_CLASS_NUM`(=5) 全局引用，改类别数一处生效。运行时 `heatmap: [5, 2]`。
- 【形状】1×5×448×224：第 c 张切片是"类别 c 的目标中心概率图"，同一位置 5 类分数独立（sigmoid 而非 softmax，允许模糊目标多类响应）。
- 【为什么】⚠ 5 类具体名单转写无法还原，仅能确认含 truck、car；结合总编笔记与行业惯例推断为 **car / truck / bus / VRU（行人+骑行者合并）/ 其他（锥桶类或特种车）**，置信度中等，待与代码 `class_names` 核实。类别少（对比 nuScenes 10 类）是量产取舍：检测类别收敛到"对规控行为有区别意义"的粗粒度，细分类交给下游属性头（is_child、anomaly_tag 这些 label key 就是佐证）。
- 【连接】CenterPoint 在 nuScenes 用 6 个 task 分组（car/truck+trailer/bus/…每组一个 SeparateHead）；这里 `tasks` 列表只有 1 个 task、5 类同组同头——head 数从 6 套砍到 1 套，嵌入式算力again。

---

#### 卡10-47｜[01:31:01] "然后在这里会把我们所用到的一个 BEV 的 feature，会也保存到我们的这个 dict 里面去。这个就是检测的（头）——其实很简单"
（合并 [01:31:01(1320)][01:31:04(1321)][01:31:08(1322)][01:31:16(1323)]，本章终点句）

- 【直译】最后一步：把本次前向"用过的" BEV 特征（used_bev_feat）也存进输出 dict；检测头讲解完毕——结构上其实很简单。
- 【代码】即卡10-41 已给出的 `ret_dict['used_bev_feat'] = torch.nan_to_num(used_bev_feat, ...)`，源码注释写明用途："保存使用过的 bev_feat，用于后续取 instance_embeddings"——box 解码出 topk 目标后，回到这张特征图上按目标位置抠出 embedding 向量，给跟踪（with_track_task: True 的 track 分支）和端到端用。
- 【为什么】"其实很简单"是真话：检测头没有任何黑魔法——shared_conv + 10 组两层卷积 + 3 条特征路由 if-else。**复杂度全部在外围**：5 路输入的来源、三帧标签的抽取、GT 预生成、类别与监督的配置化。这也是本套课程反复出现的模式：量产代码的难点从来不在网络结构，而在数据流和接口。
- 【连接】下一章预告（对应 [01:31:17] "接下来的话就是去计算 Loss"）：这 11 张输出图如何与 centerpoint_head_gt_rl 逐像素对账——heatmap 用 GaussianFocalLoss（yaml 已剧透 `loss_cls: GaussianFocalLoss`、`init_bias: -4.5951`），回归项在目标位置用 L1。练习 ch10-6 先把 init_bias 的数字玄机拆给你看。

---

### 🔨 动手练习 ch10-6：init_bias=-4.5951 的玄机——heatmap 头的 focal 初始化

```python
import torch, torch.nn as nn

# yaml: separate_head: {type: SeparateHead, init_bias: -4.5951, final_kernel: 3}
# 问题：为什么heatmap最后一层卷积的bias要初始化成-4.5951？
p = torch.sigmoid(torch.tensor(-4.5951))
print(f'sigmoid(-4.5951) = {p:.4f}')          # 预期输出: 0.0100 —— 初始前景概率≈1%

# 复现：一个heatmap分支，final层bias填-4.5951
heatmap_head = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 5, 3, padding=1))            # 5类, final_kernel=3
nn.init.constant_(heatmap_head[-1].bias, -4.5951)

x = torch.randn(1, 64, 112, 56)
with torch.no_grad():
    hm = torch.sigmoid(heatmap_head(x))
print('heatmap out :', hm.shape)               # torch.Size([1, 5, 112, 56])
print(f'初始平均响应: {hm.mean():.4f}')          # ≈0.01量级（卷积随机权重导致轻微浮动）

# 道理：448x224=10万个格子里目标中心只有几十个，正负比~1:1000。
# 若bias=0，初始所有位置预测0.5，第一步focal loss会被海量负样本的大梯度淹没;
# 把初始概率压到1%（log(0.01/0.99)≈-4.595），负样本初始loss极小，训练从第一步就稳定。
# RetinaNet论文的 prior probability π=0.01 同款技巧，CenterPoint/mmdet3d原样继承。
```

**【小结】** ① 调试台实测锁死全部输出：9 个回归/分类头 1×{2,1,3,2,2,1,1,2}×448×224 + 两张 5 类 heatmap 1×5×448×224 + 附赠 lidar_rot_weight 与 used_bev_feat。② heatmap 通道数=DET_CLASS_NUM=5，类名仅能确认 car/truck（⚠余下推断为 bus/VRU/其他），单 task 单 SeparateHead 是相对 nuScenes CenterPoint 六 task 的算力瘦身。③ used_bev_feat 存档供解码后抠 instance embedding 给跟踪与端到端——检测头本体"其实很简单"，复杂度都在接口与数据流。

---

## 本章总收束

把 01:15:39–01:31:16 的 16 分钟压成一张因果链：

```
bev_temporal 1×64×448×224
   └─ CaddnBEVBackbone（UNet式：blocks下采样→deblocks上采样→cat→reduce_channel）
        ├─ bev_feat 1×64×448×224 ──────────────┐
        └─ mid_feat [1×128×112×56, 1×64×224×112] ──→ 端到端(export_instance_embeddings)
parsing_embedding (1*3)×64×448×224 ─ view+[:, -1] → lidar_rt_feat（单帧RL，救yaw）
lidar_feat_reciprocal_2nd 1×64×448×224 ────────→ lidar_vel_feat（时序RL，救速度）
labels(3帧总线) ─ skip_keys+label_extract → 当前帧label + 预生成centerpoint_head_gt_rl
   └─→ data_dict → CenterHead: shared_conv → SeparateHead(__getattr__路由10头×2卷积)
          → ret_dict: reg2/height1/dim3/rot2/vel2/dir_cls1/movement1/rot_lidar2/
                      close_heatmap5/heatmap5 (+lidar_rot_weight1, used_bev_feat)
          → 全部 1×C×448×224 → 下一章：Loss
```

三个值得写进你求职素材库的观点：**属性适配特征**（yaw 用单帧、速度用时序，if-else 三行写清）、**监督分层**（heatmap 全图 + close_heatmap 近场加强）、**接口留白**（mid_feat/used_bev_feat 给跟踪与端到端）——这三点全都超出开源 CenterPoint，是量产 BEV 的真实增量。

## 本章存疑清单（⚠汇总）

1. ⚠ [01:16:44] 转写"1乘64乘48乘24"→ 按帧证（调试台 torch.Size）应为 1×64×448×224；[01:17:09]"64 48和224"、[01:30:01]"1乘244824" 同类数字吞音，均按帧证修正。
2. ⚠ [01:17:18] "224和124和112"→ mid_feat 两尺度按帧证为 112×56 与 224×112（通道 128 与 64）。
3. ⚠ [01:18:56]/[01:19:17] "优化 yaw/速度的一个减速"→"减速"疑为"精度/预测"误听，按语义取"精度"。
4. ⚠ [01:22:53] "它是起初数的还是三帧的 RL feature"→"起初数"疑为"取出来的"，含糊。
5. ⚠ [01:30:43–49] 5 类类别名转写损坏（"truck cut head sinθ head"），仅 truck/car 可确认；bus/VRU/其他为推断，需对代码 class_names 核实。
6. ⚠ 帧 `01_16_47.jpg` 中 UNet 前向函数名为 `forward_backup`，docstring 形状 [N,128,352,224] 为旧配置遗留；正式 forward 与其关系未在视频中说明。
7. ⚠ close_heatmap 的"近距离"具体米数范围本段未给出，待 Loss 章/配置核实。
8. ⚠ 5 类检测的 task 分组（单 task 5 类同头）为帧证 yaml 推断，`class_names` 锚点展开内容未在画面中完整可见。
9. （非转写内容补充）`mov_two_stage` 二阶段动静头、`use_fusion_instance_embeddings`/inputs[5]、`enhance_vision` 分支在代码帧中可见但讲者未讲，当前配置均未启用。


---
> [[Ch09_MemoryManager与时序融合|← Ch9]] · [[00_总览与脉络|📖 总览]] · [[Ch11_Loss全解|Ch11 →]]

