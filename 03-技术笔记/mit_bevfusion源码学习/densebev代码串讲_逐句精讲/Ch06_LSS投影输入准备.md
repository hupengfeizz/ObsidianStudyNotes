> [[Ch05_LidarRadar融合|← Ch5]] · [[00_总览与脉络|📖 总览]] · [[Ch07_LSS投影本体|Ch7 →]]

# Ch6 LSS 投影输入准备（00:36:01–00:40:59）

> **本章在全局地图的位置**：
> `… → RL融合(UNet) → ★【LSS投影：输入准备（本章）】→ LSS投影内核(外积+拍平+grid_sample, Ch7) → 多视角融合 → RC融合 → …`
>
> 上一章讲者刚讲完 radar/lidar 特征的 UNet 融合（RL 融合），得到 448×224 与下采样一倍的 224×112 两路 BEV 特征。本章视频画面从 `bev_projector.py` 的 `class ModuleBevProject(BaseModule)` 开始（画面路径栏：`e2e > tasks > bev_task > uvp_module > models > fv2bev > bev_projector.py`——注意目录名 **fv2bev**，"front-view 到 BEV"，这个命名本身就把本模块的职责说完了）。本章只covers"投影前的粮草清点"：四路输入是什么、shape 怎么读、softmax 怎么把 logits 变成深度分布、bin 0 这个"垃圾桶"为什么要丢。真正的外积+splat 在下一章。
>
> **本章使用的关键帧证据**：00_36_10（ModuleBevProject 的 `__init__`+`forward` 开头，配置默认值）、00_36_26 / 00_37_32（draw.io 框图：depth_inputs / depth_probses / grid 三组 shape）、00_38_11 / 00_38_22（forward 的输入整理与 for 循环）、00_38_53（调试台打印 fv_feat / depths_logit 的真实 shape）、00_39_26 / 00_39_46（softmax + dustbin 三分支代码、SoftmaxBackward 调试悬浮窗）、00_40_25（prj(...) 调用行）、00_40_58（BEVProjector.forward 的 docstring 与 view 拆分）。

---

## Part 1（00:36:01–00:36:17）从 RL 融合切换到 LSS 投影模块

**导读**：这一小段是模块切换的"过场白"。输入侧：上一章刚结束的 RL 融合（lidar+radar 的 BEV 特征）已经拿到；讲者现在把调试断点从 radar 分支挪到图像分支的投影入口。输出侧：本段没有任何张量变换，只是宣告接下来 debug 的是 LSS 投影。位置上，这是整个图像通路（backbone→FPN→DepthNet）憋了 36 分钟之后，图像特征第一次要"落地"到 BEV 平面的入口。

### 卡片 6-1 ｜进入 LSS 投影（合并 [00:36:01]–[00:36:17] 五句口头过渡）

> **原话**（合并 [00:36:01][00:36:03][00:36:05][00:36:10][00:36:17]，讲者此处多次口头重复）：
> "然后……把我们（RL 融合）结束之后，应该就是我们做 LSS 投影的（模块）。对，投影的话——"

- 【直译】RL 融合讲完了，下一站是 LSS 投影。讲者在 IDE 里切到 `bev_projector.py`，断点停在 `ModuleBevProject.forward` 第一行（00_36_10 帧里第 301 行 `imgs_grp_num = self.imgs_grp_num` 被黄色高亮，行号左侧有断点箭头）。
- 【代码】帧 00_36_10 显示这个类的骨架：`class ModuleBevProject(BaseModule):` → `def forward(self, *inputs, **labels):`。注意入参是 `*inputs` 可变参数——这是老框架风格：所有输入（图像特征、depth、grid map）按**位置约定**打包成一个 tuple 传进来，后面全靠索引 `inputs[i]` 取，可读性差但方便 ONNX 导出时对齐输入顺序。
- 【为什么】为什么图像通路拖到第 36 分钟才投影？因为 LSS 投影必须先集齐三样东西：图像特征（FPN，Ch1）、逐像素深度分布（DepthNet，Ch2-3）、几何映射表（DataLoader 预计算）。前两者是网络算的，第三个是 CPU 端标定+外参算好的——本章就是清点这三样。
- 【连接】对照你熟的 BEVFusion：这一步等价于 `mmdet3d` 里 `LSSTransform.forward` 被调用前的时刻——`get_cam_feats` 已跑完、`get_geometry` 的输入（相机内外参）已就绪。DenseBEV 的差别在于几何部分**不在网络里现算**，而是 DataLoader 直接给成品 grid map（见卡片 6-8）。
- 【形状】此刻挂在调试台底部的还是上一章的旧打印（帧 00_36_10 下方可见 `torch.Size([3, 64, 352, 224])`、`crop_near_radar_feature.shape → torch.Size([3, 64, 96, 224])` 等 radar 遗留输出），本章新 shape 要到 00:38:53 才刷出来。

### 🔨 动手练习 ch6-0：用 `*inputs` 约定还原模块入口

```python
import torch

# ModuleBevProject.forward(self, *inputs, **labels) 的"按位置打包"约定迷你复现
def forward(*inputs, **labels):
    print(f"收到 {len(inputs)} 个位置输入")
    for i, t in enumerate(inputs):
        print(f"  inputs[{i}]: {tuple(t.shape)}")

B3 = 1 * 3  # batch=1, 时序3帧 → 折叠进第0维
forward(
    torch.randn(B3 * 7, 128, 88, 160),   # inputs[0] 针孔图像特征
    torch.randn(B3 * 4, 128, 64, 96),    # inputs[1] 鱼眼图像特征
    torch.randn(B3 * 7, 100, 88, 160),   # inputs[2] 针孔depth logits
    torch.randn(B3 * 4, 32, 64, 96),     # inputs[3] 鱼眼depth logits
    torch.randn(B3, 7, 224, 112, 2),     # inputs[4] 针孔grid map
    torch.randn(B3, 4, 16, 16, 2),       # inputs[5] 鱼眼grid map(尺寸存疑,见6-8)
)
# 预期输出：6行shape。体会"全靠位置索引"的脆弱性：换一个顺序,后面全错。
```

**【小结】** ①本段是纯过场：断点从 RL 融合挪进 `bev_projector.py` 的 `ModuleBevProject.forward`；②该模块入参用 `*inputs` 位置约定打包，是为 ONNX 导出对齐输入准备的老式写法；③接下来 40 秒讲者按 1234 逐个清点输入。

---

## Part 2（00:36:17–00:37:57）四路输入盘点：图像特征、depth、两张 grid map

**导读**：本段是全章信息密度最高的"清单页"。输入侧有四组东西：①FPN 出来的下采样 8 倍图像特征（针孔+鱼眼两份）、②DepthNet 预测的深度 logits（针孔 100 bin、鱼眼 32 bin）、③④DataLoader 预生成的针孔/鱼眼 grid map（相机系→自车 BEV 的采样映射表）。输出侧：本段不变换任何张量，但把每个输入的 shape 一次报全。画面此时切到 draw.io 框图（00_36_26/00_37_32 帧），框图上白纸黑字写着这些 shape，是全视频少有的"讲者自己画的数据流原图"。

### 卡片 6-2 ｜[00:36:18] 输入总述

> **原话** [00:36:18]："（LSS）投影的话，它的主要输入是有这几个。"

- 【直译】LSS 投影模块不是只吃一个张量，是吃一组：图像特征、深度、几何映射表。
- 【代码】对应 `forward(self, *inputs)` 收到的 tuple。结合帧 00_38_11 第 310–311 行可反推出**原始 inputs 有 7 个元素**：`[针孔特征, 鱼眼特征, 针孔depth, 鱼眼depth, 前视grid(2相机), 侧后视grid(5相机), 鱼眼grid(4相机)]`——因为代码里有 `inputs_4 = torch.cat([inputs[4], inputs[5]], dim=1); inputs = [*inputs[:4], inputs_4, inputs[6]]`，把前视 2 路和侧后视 5 路的 grid 在相机维 concat 成 7 路，整理成 6 元素列表。这正好对上框图（00_37_32）绿色框：`input[1] (bs*3)*2*224*112*2` + `input[10] (bs*3)*5*224*112*2` → `grid (bs*3)*7*224*112*2`。
- 【为什么】前视和侧后视的 grid 为什么在 DataLoader 里分开给？因为该车 12/侧/后相机在数据预处理时就按"前视组/环视组"分组处理（Ch·DepthNet 段讲过 depth GT 也是分成 12 号相机和侧后相机两组再 concat 的，[00:04:53] 一带）；到投影这里针孔 7 路统一处理，所以先 cat 回来。
- 【连接】BEVFusion 里对应物是 `camera2lidar_rots/trans, intrins, post_rots/post_trans` 这堆标定参数——BEVFusion 传"原料"（参数）进网络现算几何，DenseBEV 传"熟饭"（grid map）进网络直接采样。

### 卡片 6-3 ｜[00:36:20]–[00:36:33] 输入一：FPN 的下采样 8 倍图像特征（重点句，5 角度）

> **原话**（合并 [00:36:20][00:36:22][00:36:24][00:36:25][00:36:26][00:36:28][00:36:33]，中间多处结巴）：
> "一个是我们在 FPN……出来所用到的、生成的我们图像的一个特征，是下采样八倍的。"

- 【直译】第一路输入=FPN 金字塔里 stride=8 那一层特征图。原图分辨率 704×1280（88×8=704、160×8=1280，由后文 shape 反推⚠——另一个可能是先降采样过一次的输入分辨率），过 backbone+FPN 后取 1/8 尺度。
- 【代码】Ch1 里讲过：`FPN` 出来多尺度特征（8/16/32 倍），但 DepthNet 和这里都只消费 8 倍那层（[00:02:39][00:03:35] "只用到了下采样8倍的特征"）。等价伪代码：`fv_feat = fpn_outs["stride8"]  # (B*T*N, 128, H/8, W/8)`。
- 【形状】框图 00_36_26 顶部蓝框 `depth_inputs：[(bs*3*7)*128*88*160, (bs*3*4)*128*64*96]`——注意这个框叫 depth_inputs，因为**同一份 8 倍特征既喂 DepthNet 也喂 LSS 投影**，一份特征两处消费。
- 【为什么】为什么选 8 倍不选 4 倍或 16 倍？①LSS 要对每个像素外积出 D 份深度加权特征，显存与像素数成正比，88×160=14080 像素已是 4 倍尺度（352×640=225280）的 1/16，显存友好；②8 倍特征感受野足够大，深度估计需要上下文（一辆车占几十个像素才能判断远近）；③BEV 网格 0.4~0.8m 的分辨率也用不着更细的图像特征。不这么做（用 4 倍）：显存爆、投影点数×4、速度掉一半以上。
- 【连接】BEVFusion 同样在 1/8 特征（256 通道）上做 LSS；LSS 原论文用 EfficientNet 的 1/16。你在智谷课程里学的"stride 与感受野"在这里的工程含义就是：**投影层的 stride 决定了 BEV 特征的信息粒度上限**。

### 卡片 6-4 ｜[00:36:36]–[00:36:41] 针孔特征 shape：B×3×7×128×88×160（重点句，5 角度）

> **原话**（合并 [00:36:36][00:36:37][00:36:41]）："对于针孔的话就是 batch×3 乘七，然后乘上 128；然后对应的 feature（原转写"飞雪"⚠，应为 feature 或"分辨率"的误听）的尺度是 88×160 的。"

- 【直译】针孔特征逻辑 shape 是 B×3×7×128×88×160：B=batch、3=时序 3 帧、7=七路针孔相机、128=通道、88×160=特征图高宽。物理存储时前三维折叠在一起：`(B*3*7, 128, 88, 160)`，B=1 时就是 (21,128,88,160)——这正是 Ch2 里 DepthNet 输入的那个"21"（[00:06:11] "21 是因为历史三帧×七路针孔 concat 到一起"）。
- 【代码】折叠/展开的惯用写法：`x = x.view(B*T*N, C, H, W)` 进 2D 卷积（卷积对每路相机权重共享），用完再 `view(B, T, N, C, H, W)` 拆回。本章后半段（卡片 6-13、6-23）就是两次拆回。
- 【形状】数一下元素量：21×128×88×160 ≈ 3784 万 float ≈ 144MB（fp32）。这只是特征，后面外积×100 深度 bin 会膨胀 100 倍——这就是为什么 LSS 的中间量必须精打细算（Ch7 的"拍平"就是为了不实体化这个巨物）。
- 【为什么】为什么把 T=3 帧也折进 batch 维？因为 backbone/FPN/DepthNet 对"哪一帧"完全无感——历史帧和当前帧共享同一套图像网络权重，折叠成 21 路一起算，一次 forward 全出，GPU 利用率最高。时序信息要等到后面 MemoryManager/时序融合章节才被区分。
- 【连接】BEVFusion（nuScenes 版）对应 shape 是 (B, 6, 256, 32, 88)：6 相机无时序、1/8 特征。DenseBEV 是 7 针孔+4 鱼眼+3 帧时序，量级大得多；这也是它把几何算到 CPU、投影用 grid_sample 的动机之一。

### 卡片 6-5 ｜[00:36:46]–[00:36:50] 鱼眼特征 shape：B×3×4×128×64×96

> **原话**（合并 [00:36:46][00:36:49][00:36:50]）："然后（同样地）我们鱼眼的是 batch×3 乘四，然后乘以 128×64×96 的。"

- 【直译】鱼眼四路（前后左右各一颗，看近场），同样 3 帧时序、128 通道，但特征图更小：64×96（原图 512×768 的 1/8⚠反推）。
- 【形状】折叠后 (B*3*4, 128, 64, 96)，B=1 时 (12,128,64,96)。框图 depth_inputs 第二项 `(bs*3*4)*128*64*96` 完全吻合。
- 【为什么】鱼眼分辨率低于针孔是合理取舍：鱼眼负责 10m 内近场（泊车、加塞），近处目标本来就大，用不着高分辨率；且鱼眼畸变大，边缘像素信息密度低。
- 【连接】nuScenes/BEVFusion 没有鱼眼这路，这是量产项目（环视泊车域）特有的配置——也是本代码里处处 `for i, prj in enumerate(prjs)` 针孔/鱼眼双分支的根源。

### 卡片 6-6 ｜[00:36:54]–[00:37:02] 输入二：预测 depth，针孔 100 bin（重点句，5 角度）

> **原话**（合并 [00:36:54][00:36:58][00:36:59][00:37:02]）："然后也会用到我们所预测的一个 depth，就是 batch×3 乘七，然后针孔是预测 100 个 bin，（所以是）100×88×160。"

- 【直译】第二路输入=DepthNet 的输出：对针孔每路相机的每个特征像素，预测一个 100 维向量——把可见深度范围切成 100 个离散档位（bin），每维是"该像素深度落在这个档位"的打分（此刻还是 logits，未归一化，见卡片 6-15）。
- 【代码】Ch2 讲过 DepthNet 就两个卷积（[00:05:29]"网络结构比较简单，就只是两个卷积"），最后一层 `nn.Conv2d(C, 100, 1)` 把通道数变成 bin 数。逻辑 shape B×3×7×100×88×160，物理 (21,100,88,160)——帧 00_36_10 调试台残留的 `torch.Size([21, 100, 88, 160])` 就是它（Ch2 时打的）。
- 【形状】框图蓝框 `depth_probses：[(bs*3*7)*100*88*160, (bs*3*4)*32*64*96]`。注意框图把它命名为 depth_prob**ses**（复数的复数），因为是 [针孔的, 鱼眼的] 两个张量组成的 list。
- 【为什么】深度为什么用分类（100 bin）不用回归（1 个数）？①单目深度天然多峰模糊（一辆车的边缘像素，深度可能是车也可能是背景），分布比点估计表达力强；②LSS 的精髓恰恰是要用**整条分布**去加权散布特征——深度不确定时特征就摊到多个 BEV 格子上，让后续网络自己消化不确定性；③分类可以配 one-hot 交叉熵监督（Ch3 的 Depth Loss/ddn_loss 就是这么做的）。回归做不到这三点。
- 【连接】这正是 LSS 论文（Lift 步骤）与 CaDDN 的做法；BEVFusion 的 `DepthLSSTransform` 里 depth 也是 D=118 bin 的 softmax 分布。帧 00_40_58 里 `BEVProjector.prj_with_depth` 写着 `return self.proj_type in ('caddn', 'liftsplat')`——代码作者自己就把这两个流派并列成可切换选项，是很好的"流派地图"证据。

### 卡片 6-7 ｜[00:37:04]–[00:37:11] 鱼眼 depth：32 bin

> **原话**（合并 [00:37:04][00:37:07][00:37:09][00:37:11]）："然后鱼眼的话是 batch×3×4×32——对于鱼眼我们是预测的 32 个 bin——然后×64×96。"

- 【直译】鱼眼每像素只切 32 个深度档。逻辑 shape B×3×4×32×64×96，物理 (12,32,64,96)。
- 【为什么】鱼眼只管近场（比如 0.5~12m⚠具体范围视频未说），量程短，32 档就够密；针孔要覆盖到几十米上百米，才需要 100 档。bin 数=量程/精度的直接体现。
- 【形状】外积后的 frustum 体积对比：针孔 21×128×100×88×160 vs 鱼眼 12×128×32×64×96——鱼眼那路只有针孔的约 1/45，几乎白送。
- 【连接】BEVFusion 单一相机类型只有一套 D；DenseBEV 的"每种相机各配一套 bin 数、特征尺寸、grid map"是它的多相机异构设计，后面所有代码的双分支循环都由此而来。

### 卡片 6-8 ｜[00:37:14]–[00:37:44] 输入三/四：DataLoader 生成的 grid map（重点句，5 角度）

> **原话**（合并 [00:37:14][00:37:17][00:37:21][00:37:30][00:37:41]）："然后以及我们在 DataLoader 里面所处理生成的 GridMap——就是如何把我们图像的特征给它对应、给它映射到**自车坐标系下的 BEV 的一个 feature 上**。这个就是 DataLoader 生成的一个 Grid。"

- 【直译】第三、四路输入是两张"查表"：针孔 grid map 和鱼眼 grid map。表里存的是——对自车坐标系 BEV 平面上的每一个格子，应该去哪个相机图像的哪个位置取特征。**几何（内外参、畸变、坐标变换）全部在 DataLoader（CPU 侧）算完**，网络在 GPU 上只做一次 `grid_sample` 查表采样。
- 【代码】帧 00_40_58 的 docstring 是最权威解释：`grid_map (Tensor): the grid map used to project front view to BEV; vconv_grid_map: (B, n, bev_h, bev_w, 2); oft_grid_map: (B, n, bev_h, bev_w, bev_c, 4)`。即最常用形态是每个 BEV 格子存 2 个数（归一化采样坐标 u,v），正是 `F.grid_sample(input, grid)` 要求的 grid 格式；还有一种 OFT 风格的 4 数版本（带 bev_c 高度柱），本次没走⚠。
- 【形状】框图绿色框：前视 `input[1] (bs*3)*2*224*112*2` + 侧后 `input[10] (bs*3)*5*224*112*2` → cat 成 `grid (bs*3)*7*224*112*2`；鱼眼 `input[11] (bs*3)*4*16*16*2` → `grid (bs*3)*4*16*16*2`。针孔 BEV 网格 224×112 正是全图 448×224 的**下采样一倍**（0.8m/格）——印证了"投影在半分辨率 BEV 上做"的全局设定。鱼眼 16×16 ⚠低置信（缩略图小字），但与 00_36_10 帧配置默认值 `fisheye_bev_tag_dict: {'crop_coords_bev_area': [286, 96, 318, 128]}`（318−286=32、128−96=32，全分辨率 32×32 → 半分辨率 16×16）自洽：鱼眼只投影**自车周边约 12.8m×12.8m 的近场小方块**。
- 【为什么】为什么预计算而不是像 BEVFusion 那样在网络里现算 frustum 几何？①相机内外参在一个 clip 内固定，逐帧现算是纯浪费；②量产部署要过 ONNX/芯片工具链，`grid_sample` 是标准算子，而"创建 frustum→坐标变换→voxel pooling"这套动态几何在很多推理引擎上难落地；③CPU 预计算把畸变去畸变（尤其鱼眼的复杂畸变模型）挡在训练图之外，GPU 图里干干净净。代价：grid map 是**以 BEV 格子为出发点的"拉取式"映射**，方向与 LSS 原版"以像素为出发点的推送式 splat"相反——两者的等价与差异正是 Ch7 的主戏。
- 【连接】①BEVFusion 的对应物：`get_geometry()` 算出的 (B,N,D,H,W,3) frustum 点云坐标——那是 push 方向；DenseBEV grid map 是 pull 方向，更像 BEVDet 系的 `bev_pool` 预计算索引或 Fast-BEV 的 LUT（查找表）方案。②你在 4060 上跑的 BEVFusion mini 训练里 `geom_feats` 每个 iter 重算一遍，正是这里被优化掉的那部分开销。

### 卡片 6-9 ｜[00:37:44]–[00:37:56] 输入清单收口

> **原话**（合并 [00:37:44][00:37:49][00:37:52][00:37:56]）："它主要输入的话就主要是这 1、2、3、4 个：对应的就是图像的特征、预测的 Depth，以及针孔和鱼眼（各自）的 GridMap。"

- 【直译】清单收口：①图像特征（针孔+鱼眼两份）②depth logits（两份）③针孔 grid map ④鱼眼 grid map。讲者数的"4 个"是按**类别**数的；按张量个数是 6 个（每类针孔/鱼眼各一）。
- 【形状】汇总表（B=1，T=3）：

| 输入 | 针孔 | 鱼眼 |
|---|---|---|
| 图像特征 | (21,128,88,160) | (12,128,64,96) |
| depth logits | (21,100,88,160) | (12,32,64,96) |
| grid map | (3,7,224,112,2) | (3,4,16,16,2)⚠ |

- 【为什么】注意 grid map 的第 0 维是 3（=B×T）而不是 21——几何映射按"帧"给，同一帧 7 路相机的映射叠在第 1 维；而特征/深度按"相机张数"折叠。两种折叠约定并存，是后面一堆 `view` 的根源。
- 【连接】和 BEVFusion 的 `forward(img, points, camera2ego, lidar2ego, ...)` 十几个参数比，这里 6 个张量已经算克制——代价是全靠位置索引，见下一 Part 的 `inputs[2*imgs_grp_num + i]` 这类"魔法下标"。

### 🔨 动手练习 ch6-1：grid map 的"查表投影"一分钟复现

```python
import torch
import torch.nn.functional as F

# 造一张"图像特征"：1×1×4×8，像素值=列号，方便肉眼验证采样对不对
feat = torch.arange(8.).repeat(4, 1).view(1, 1, 4, 8)

# 造一张"grid map"：BEV 网格 2×3，每格存归一化采样坐标 (x,y)∈[-1,1]
# 这就是 DataLoader 在 CPU 上用内外参算好的东西的最简化版
grid = torch.tensor([[[[-1., -1.], [0., 0.], [1., -1.]],
                      [[-1.,  1.], [0., 0.], [1.,  1.]]]])  # (1,2,3,2)

bev = F.grid_sample(feat, grid, align_corners=True)
print(bev.squeeze())
# 预期输出：
# tensor([[0.0000, 3.5000, 7.0000],
#         [0.0000, 3.5000, 7.0000]])
# 左列采到图像最左(值0)，中间采到图像中心(3.5)，右列采到最右(7)。
# 体会：网络里只剩 grid_sample 一步，几何早在 grid 里定死了。
```

**【小结】** ①LSS 投影吃 4 类 6 个张量：针孔/鱼眼的特征、depth logits、grid map；②所有 shape 都能在讲者的 draw.io 框图上找到原文（depth_inputs / depth_probses / grid 三组蓝绿框）；③grid map 是 DataLoader 预计算的"相机→自车 BEV"查表，把几何从 GPU 图里挪到了 CPU，这是与 BEVFusion 最大的工程分歧点。

---

## Part 3（00:37:57–00:38:53）forward 前置整理：分组、bs 推断与 view 拆维

**导读**：进入 `ModuleBevProject.forward` 的正文。输入侧是上一 Part 清点的 6 个张量（相机维折叠状态）；输出侧是整理好的 `fv_feat (3,7,128,88,160)` 与 `depths_logit (3,7,100,88,160)`，即"把 21 拆成 3×7"。中间还有三件杂事：把前视/侧后视 grid cat 成 7 路、推断 batch size、以及针孔/鱼眼两组各跑一遍的 for 循环骨架。这一段讲者语速快、跳着讲，我们用帧 00_38_11/00_38_22/00_38_53 的完整代码把他略过的细节补齐。

### 卡片 6-10 ｜[00:37:57]–[00:38:08] "前面这些只是对输入数据做整理"

> **原话**（合并 [00:37:57][00:38:04][00:38:08]）："前面这些其实都只是（对）输入的数据进行一个整理。"

- 【直译】forward 开头十几行不做任何计算，只是把 `*inputs` 捋顺：拆列表、cat grid、算 batch size。
- 【代码】帧 00_38_11/00_38_22 完整还原（行号为屏幕真实行号）：

```python
# bev_projector.py  ModuleBevProject.forward  (帧00_36_10/00_38_11/00_38_22 逐行抄录)
301  imgs_grp_num = self.imgs_grp_num            # =2：针孔、鱼眼两组
302  if isinstance(inputs[0], list):
303      imgs_grp_num = len(inputs[0])
304      inputs = [*inputs[0], *inputs[1:]]
306  prjs = [self.prj_pinhole, self.prj_fisheye]  # 两个投影器
307  if self.split_pinhole:
308      prjs.insert(0, self.prj_pinhole)         # 前视单拆一组时插第三个
309  else:
310      inputs_4 = torch.cat([inputs[4], inputs[5]], dim=1)   # 2路+5路grid → 7路
311      inputs = [*inputs[:4], inputs_4, inputs[6]]
312  if onnx.is_in_onnx_export():
313      bs = 1
314  else:
315      if self.god_use_hist_seq_len > 0:        # ⚠"god"前缀含义未明,疑为qat/god配置开关
316          B = len(labels['labels'][0]['sample_index'])
317          bs = B * self.god_use_hist_seq_len
318      else:
319          bs = inputs[2*imgs_grp_num].size(0)  # 用grid的第0维当bs → 1×3帧=3
```

- 【形状】关键行 319：`inputs[2*imgs_grp_num]`=inputs[4]=针孔 grid，其第 0 维是 B×T=3。**所以整段代码里的 `bs` 不是 1 而是 3**——"batch"在此语境下=样本×时序帧。讲者 [00:41:44] 之后（Ch7 开头）也会亲口确认"batch size 乘上 3 帧"。
- 【为什么】为什么 ONNX 分支强行 `bs=1`？部署时固定 batch，静态 shape 让编译器可劲优化；训练时才需要从数据里动态推断。这类 `if onnx.is_in_onnx_export()` 双轨写法贯穿全文件（softmax 那里还有一处），是量产代码的典型指纹。
- 【连接】`split_pinhole` 开关暗示还有第三种配置：前视广角/长焦单独成组（`use_front_wide_narrow_cam`、`lss_cam_list` 这些 `__init__` 里的字段同源，见帧 00_36_10 第 283 行 `pinhole_num = len(self.lss_cam_list) if self.use_front_wide_narrow_cam else 7`）。本次运行走 else 分支。

### 卡片 6-11 ｜[00:38:08]–[00:38:20] 分两组：针孔一遍、鱼眼一遍（重点句，5 角度）

> **原话**（合并 [00:38:08]"在这里呢"[00:38:12][00:38:15][00:38:20]）："主要在这里就是也是把我们（的输入）也是分成了两组，就是去分别去对针孔做 LSS 投影，以及对鱼眼做 LSS 投影。"

- 【直译】接下来是一个 `for i, prj in enumerate(prjs)` 循环（帧 00_38_22 第 323 行高亮），i=0 处理针孔组、i=1 处理鱼眼组，各自用各自的投影器、grid、bin 数走完全套 LSS。
- 【代码】循环体取数全靠下标算术（帧 00_38_22 第 324–332 行）：

```python
323  for i, prj in enumerate(prjs):
324      if self.grid_w_c:                        # 本次为False,走else ⚠grid_w_c含义:两张grid(w/c)成对给
326          grid = (inputs[2*imgs_grp_num + 2*i], inputs[2*imgs_grp_num + 2*i + 1])
327      else:
328          grid = inputs[2*imgs_grp_num + i]    # i=0→inputs[4]针孔grid; i=1→inputs[5]鱼眼grid
329      if self.god_use_hist_seq_len > 0:
330          grid = self.seq_info_extract.extract_images_by_slice(grid, B, slice(-self.god_use_hist_seq_len, None))
331      fv_feat = inputs[i]                      # i=0→(21,128,88,160); i=1→(12,128,64,96)
332      depths_logit = inputs[imgs_grp_num + i]  # i=0→inputs[2]; i=1→inputs[3]
```

- 【形状】i=0 时三件套：(21,128,88,160)/(21,100,88,160)/(3,7,224,112,2)；i=1 时：(12,128,64,96)/(12,32,64,96)/(3,4,16,16,2)⚠。
- 【为什么】针孔/鱼眼为什么不能合成一组投一次？①bin 数不同（100 vs 32），depth 张量拼不到一起；②特征分辨率不同（88×160 vs 64×96）；③目标 BEV 区域不同（全图 224×112 vs 近场 16×16）。三个维度全不兼容，只能循环两遍再在后续"多视角融合"章节里拼。
- 【连接】BEVFusion 单相机类型没有这层循环；但如果你以后接触"前视 8M+环视 2M"的多分辨率量产方案，几乎都长这样——**异构相机=按组循环+组内共享**，这个 pattern 值得记住。

### 卡片 6-12 ｜[00:38:22]–[00:38:36] "把 FV 的 feature 做一个 view（reshape）操作" ⚠转写勘误

> **原话**（合并 [00:38:22][00:38:24][00:38:30][00:38:32][00:38:33][00:38:36]）："然后在这里……会把我们的 BV 的 feature（⚠应为 **FV** 的 feature，front-view，Whisper 误听）做一个 Shift 的……就做一个 View 的一个 Shift 的操作（⚠应为 **view 的 reshape/变形** 操作）。"

- 【直译】把折叠的 (21,C,H,W) 用 `.view()` 拆成 (3,7,C,H,W)。转写稿此处有两处误听需要勘误：①"BV 的 feature"应为"FV 的 feature"——此刻还没投影，手上只有 front-view 特征，代码变量名就叫 `fv_feat`；②"Shift 的操作"应为"shape/reshape 的操作"——紧接着讲者报出的正是 reshape 后的 shape，且代码里根本没有 shift 语义的算子。推断依据：帧 00_38_22 第 334–336 行就是三行 view，无其他操作。
- 【代码】（帧 00_38_22 逐行）：

```python
334  _, fc, fh, fw = fv_feat.size()                              # fc=128, fh=88, fw=160
335  fv_feat      = fv_feat.view(bs, -1, fc, fh, fw)             # (21,128,88,160)→(3,7,128,88,160)
336  depths_logit = depths_logit.view(bs, fv_feat.size(1), -1, fh, fw)
                                                                # (21,100,88,160)→(3,7,100,88,160)
```

- 【形状】妙处在两个 `-1`：335 行用 `-1` 推出相机数 7（=21/3）；336 行相机数取 `fv_feat.size(1)`、用 `-1` 推出 bin 数 100。**bin 数在这段代码里从未硬编码**——换成鱼眼组（32 bin）同样三行照跑，这就是双分支循环能共享代码体的原因。
- 【为什么】为什么必须在投影前拆出相机维？因为 grid map 是按 (bs, n_cam, bev_h, bev_w, 2) 组织的——投影是**逐相机**的几何操作，每路相机有自己的映射表，folded 状态对不上号。
- 【连接】`view` 只是改 stride 解释、零拷贝，但要求内存连续且**拆分顺序与折叠顺序一致**——当初是 (B,T,N)→B*T*N 折的，现在 (3=B*T, 7=N) 拆回，顺序严格对得上。你在 BEVFusion 里见过的 `rearrange(x, '(b n) c h w -> b n c h w', b=B)` 是同一件事的 einops 写法。

### 卡片 6-13 ｜[00:38:36]–[00:38:42] 图像特征拆完：3×7×128×88×160

> **原话**（合并 [00:38:36][00:38:38][00:38:42]）："然后这里出来的，这个是图像的特征：3、7、128、88、160。"（转写原文"三层七层一二八层八层一八层一百六"为数字连读误听）

- 【直译】view 完的针孔特征是 (3,7,128,88,160)：3=B×T（1×3帧）、7 路相机、128 通道、88×160。
- 【形状】铁证：帧 00_38_53 底部调试台 `fv_feat.shape → torch.Size([3, 7, 128, 88, 160])`——讲者是对着这行念的。
- 【为什么】留意维度语义的变化：第 0 维从"21 张图"变成"3 个时刻"，第 1 维成为"相机"。后续 BEVProjector 还会再拆一次（卡片 6-23），把时序显式分出来——**同一份数据，随流水线阶段不同，维度语义被逐层"翻译"**，读这类代码时刻盯住每一维当前的含义，是不迷路的唯一办法。

### 卡片 6-14 ｜[00:38:46]–[00:38:49] depth 拆完：3×7×100×88×160

> **原话**（合并 [00:38:46][00:38:49]）："以及它所对应的一个预测的 depth：3、7、100、88、160。"

- 【直译】depth logits 同步拆成 (3,7,100,88,160)，与特征逐相机、逐像素一一对齐。
- 【形状】铁证同上：调试台 `depths_logit.shape → torch.Size([3, 7, 100, 88, 160])`（帧 00_38_53）。第 2 维 100 从此成为"深度分布维"，下一 Part 的 softmax 就沿它做。
- 【连接】特征 (…,128,h,w) 与 depth (…,100,h,w) 在 Ch7 会做外积 → (…,128,100,h,w)：每像素 128 维特征 × 100 档深度权重。两张量此刻的像素级对齐（同 h、同 w、同相机、同帧）是外积成立的前提，这也是为什么两者必须用同一个 `bs`、同一套 view 逻辑整理。

### 🔨 动手练习 ch6-2：两个 `-1` 的 view 拆维（含踩坑对照）

```python
import torch

bs = 3                      # = batch(1) × 时序3帧
fv_feat      = torch.randn(21, 128, 88, 160)   # (B*T*N, C, H, W)
depths_logit = torch.randn(21, 100, 88, 160)

_, fc, fh, fw = fv_feat.size()
fv_feat = fv_feat.view(bs, -1, fc, fh, fw)
depths_logit = depths_logit.view(bs, fv_feat.size(1), -1, fh, fw)
print(fv_feat.shape)        # 预期: torch.Size([3, 7, 128, 88, 160])
print(depths_logit.shape)   # 预期: torch.Size([3, 7, 100, 88, 160])
# 注意 bin数100 是被 -1 推断出来的，鱼眼组(32 bin)同样代码可复用

# 踩坑对照：如果当初是按 (N,T,B) 顺序折叠的，view(3,7,...) 会"形状对、数据错"
x = torch.arange(6).view(2, 3)       # 语义: (T=2帧, N=3相机)
wrong = x.view(3, 2)                 # 不报错！但帧和相机已经串台
right = x.permute(1, 0).contiguous() # 换轴必须用 permute 而不是 view
print(wrong.tolist())  # [[0,1],[2,3],[4,5]]  ← 串台
print(right.tolist())  # [[0,3],[1,4],[2,5]]  ← 正确
```

**【小结】** ①forward 前置段只做三件杂事：cat 前视/侧后 grid 成 7 路、从 grid 第 0 维推 `bs=3`（=B×T）、双分支循环骨架；②核心变形是两行 view：(21,C,H,W)→(3,7,C,H,W)、(21,100,H,W)→(3,7,100,H,W)，bin 数靠 `-1` 推断实现针孔/鱼眼代码共享；③转写稿两处误听已勘误：BV→FV、Shift→shape（view 变形）。

---

## Part 4（00:38:53–00:40:24）沿 100 维 softmax 与 dustbin：bin 0 为什么必须丢

**导读**：这是本章的灵魂段落。输入：`depths_logit (3,7,100,88,160)`（未归一化打分）。输出：`depth_prob (3,7,100,88,160)`——沿 bin 维 softmax 成概率分布，并且把 bin 0（超远/无效深度的"垃圾桶"）的概率整体清零。位置上，它是 LSS"Lift"步骤的前半：拿到干净的深度分布，Ch7 才能拿它与图像特征做外积。讲者在这里顺带回放了 DepthNet 章节埋的伏笔——GT 制作时超量程深度被折叠进 bin 0——两章在此闭环。

### 卡片 6-15 ｜[00:38:53]–[00:39:06] 预测的 depth 其实是 logits ⚠

> **原话**（合并 [00:38:53][00:39:02]）："然后在这里，因为我们做预测的 depth，其实是、其实可以说是一个'内'（⚠原转写残句。按上下文与代码推断，讲者想说的是 **logit / 未归一化的打分**——因为下一句立刻说'所以要取 softmax'，且代码变量名就叫 `depths_logit`）。"

- 【直译】DepthNet 输出的 100 维向量不是概率，是任意实数打分（logits）——可正可负、加起来不等于 1，不能直接当权重用。
- 【代码】证据链：变量名 `depths_logit`（帧 00_38_22 第 332 行）→ 下一行 `F.softmax(depths_logit, 2)`（帧 00_39_26 第 358 行）。网络输出 logits、用时再 softmax，而不是在 DepthNet 里就 softmax 掉，是因为训练时的交叉熵损失（Ch3 的 ddn_loss）要吃 logits（`F.cross_entropy` 内置 log_softmax，数值更稳）。
- 【为什么】如果 DepthNet 出口就 softmax，会发生什么？①loss 侧要改用 `NLLLoss(log(p))`，log(softmax(softmax已做)) 数值不稳；②ONNX 导出侧想换 `custom_softmax`（下一卡片）就没机会了。**"网络出 logits、消费端各自归一化"是解耦的好习惯**。
- 【连接】CenterPoint 的 heatmap 同理：网络出 logits，训练用 focal loss（内部 sigmoid），推理才 sigmoid——你看后面检测头章节会再遇到一模一样的 pattern。

### 卡片 6-16 ｜[00:39:06]–[00:39:09] 沿 100 这一维取 softmax（重点句，5 角度）

> **原话** [00:39:06][00:39:09]："对，然后在这里其实我们会取一个 softmax，会沿着就是 100 这一维度会取一个 softmax。"

- 【直译】对 (3,7,**100**,88,160) 的第 2 维做 softmax：每路相机每个像素的 100 个打分被指数归一化成一条离散概率分布，∑=1。
- 【代码】帧 00_39_26 完整呈现了**双轨三分支**结构（逐行抄录）：

```python
345  if onnx.is_in_onnx_export():
346      depth_prob_ = custom_softmax(depths_logit, 2)   # 部署轨:自定义softmax(算子替换,便于芯片工具链)
347      if self.dustbin_zero: ...                        # 三种垃圾桶策略,同下
358  else:
359      depth_prob_ = F.softmax(depths_logit, 2)        # 训练轨:标准softmax,dim=2即100那一维
360      if self.dustbin_zero:                            # ← 本次运行走这支(高亮行)
361          depth_prob = depth_prob_.new_zeros(depth_prob_.shape)
362          depth_prob[:, :, 1:] += depth_prob_[:, :, 1:]
363      elif self.dustbin_additional:                    # 策略2:额外附加bin,用完切掉 depth_prob_[:, :, :-1]
373      elif self.dustbin_last:                          # 策略3:最后一个bin当垃圾桶
376      else: depth_prob = depth_prob_
```

- 【形状】shape 不变，仍是 (3,7,100,88,160)，但语义从"打分"变"概率"。帧 00_39_46 的调试悬浮窗是现场铁证：`grad_fn = <SoftmaxBackward object at 0x7fcd22999490>`，`dtype=torch.float32, device=cuda:0, ndim=5`，展开的数值 `7.0706e-03, 6.1290e-03, 5.0801e-03, ...`——量级都在 1/100 上下，正是一条尚未训练锐化/较平缓的百维分布该有的样子。
- 【为什么】为什么 softmax 沿 dim=2 而不是别的维？dim=2 是 bin 维——归一化的物理含义是"这个像素的深度必落在 100 档之一"，概率沿深度档竞争。如果错沿 dim=3/4（空间维）做，含义就成了"整行像素里谁最深"，完全错误。**沿哪一维 softmax=在哪一维上建立竞争**，这句话值得刻进肌肉记忆。
- 【连接】LSS 论文式 (1)：`c_d = softmax(a_d)`，外积 `f_{u,v,d} = c_d · f_{u,v}`；BEVFusion `DepthLSSTransform` 里同款 `depth.softmax(dim=1)`（它的 bin 维在 dim=1）。此外注意 345 行：部署轨用 `custom_softmax`——量产代码常为芯片不支持/低精度的算子准备等价替身，这是你转岗后会天天见的东西。

### 卡片 6-17 ｜[00:39:13]–[00:39:21] softmax 的"意义"：看最大概率落在哪个 bin ⚠有口误成分

> **原话**（合并 [00:39:13][00:39:21]）："然后把它——就意义上呢，就是看我们最大概率上会是在哪个 bin 上。"

- 【直译】讲者的通俗解释：softmax 后能看出该像素最可能处于哪个深度档。
- 【为什么·纠偏⚠】严格说这句只说对了一半：LSS **不取 argmax**。它保留整条分布，外积时把特征按概率摊到所有深度档上——某像素若"7m 处 0.5、9m 处 0.4"，特征就一半落 7m 格、四成落 9m 格，不确定性被原样带进 BEV 让后续网络消化。"看最大概率在哪个 bin"是帮助直觉的说法，不是代码行为；真要 argmax，深度错一档特征就全错位，且 argmax 不可导、深度头训不动。
- 【代码】反证：后续代码只有 `feat.unsqueeze(2) * prob.unsqueeze(3)` 这类外积（Ch7），全程无 `argmax` / `topk`。
- 【连接】这正是 LSS 与"伪点云"流派（Pseudo-LiDAR：取深度点估计后反投影成点）的分水岭——前者软分配、端到端可导；后者硬决策、误差不可恢复。面试聊 LSS 时把这一点讲清楚很加分。

### 卡片 6-18 ｜[00:39:26]–[00:39:48] 伏笔回收：GT 制作时超量程深度进"最后一个 bin"（重点句，5 角度）

> **原话**（合并 [00:39:26][00:39:30][00:39:35][00:39:40][00:39:45][00:39:48]，此段讲者多次结巴重复）："然后在那个 DepthNet 的时候，其实我们在处理 depth GT（原转写'depth机器'，'机器'为 GT 的误听⚠）的时候，会把——就是 bin 的索引为一百的时候——会把超过（量程的），就是超过我们所预测的最大区间的一个 depth，会把它分到最后一个 bin。"

- 【直译】做深度真值时：激光投影得到的每像素真实深度要离散成 bin 索引；深度超出可预测最大距离（量程外）的像素，索引会越界（"索引为一百"，合法档只有 0~99），这些统统被折进"最后一个 bin"当收容所。
- 【代码】GT 离散化的典型写法（DepthNet 章节的 load/预处理侧，本段为回放）：`bin_idx = ((d - d_min) / bin_size).long()`，d 超量程时 `bin_idx >= D`，代码 clamp 或直接归入溢出档。
- 【为什么】为什么必须给量程外深度留收容所，而不是直接把这些像素扔掉？①天空、远处建筑占画面很大比例，全 mask 掉会浪费大量"这里没有近处东西"的负信息；②给它们一个专属档，网络就能显式学会说"这个像素很远/无效"——这个信号在投影时反过来变成极有用的过滤器（下两张卡片）。
- 【形状】对 100 bin 的针孔：合法深度档其实是 1~99 共 99 档（见卡片 6-21），bin 0 是收容所——量程被 99 档瓜分，而不是 100。算深度分辨率时别用错分母。
- 【连接】"dustbin（垃圾桶）"是通用技巧：SuperGlue 给无匹配特征点留 dustbin 行列、分类任务的 background 类、CenterPoint heatmap 的"无目标"背景，本质都是**给'不属于任何正常档位'的样本一个显式归宿**。

### 卡片 6-19 ｜[00:39:50]–[00:39:53] 同时把"最后一个 bin"的 GT 映射成 0

> **原话** [00:39:50][00:39:53]："然后同时呢，会把最后一个 bin 它的一个 GT（原转写'机器'⚠同前）呢，会把它映射成 0。"

- 【直译】溢出收容所不占用索引 99 或 100，而是被**重映射到索引 0**：量程外深度的 GT 标签=bin 0。于是编号系统定型：**bin 0=超远/无效档，bin 1~99=有效深度档**。
- 【代码】等价一行：`bin_idx[bin_idx >= D] = 0`（或先离散到 1~99、越界置 0）。配置名 `self.dustbin_zero` 完美对应——"垃圾桶在 0 号位"；而 `dustbin_last`/`dustbin_additional` 是另两种编号方案（垃圾桶在末位/额外附加一位），同一份代码用开关兼容三种 GT 编号，说明团队历史上换过方案⚠（帧 00_39_26 三个 elif 是活化石）。
- 【为什么】垃圾桶放 0 号而不是末位，有一个实打实的工程甜头：丢弃时切片 `[:, :, 1:]` 拿到的是**连续内存段**、且剩余档的索引 i 与物理深度的换算关系不用平移（第 i 档就是第 i 档）；若放末位则切 `[:, :, :-1]`，两者能力等价，纯属约定——但约定必须与 GT 制作端严格一致，否则训练目标就串位了。
- 【连接】回想 Ch3 Depth Loss：交叉熵的 target 里就有大量 0 类像素（天空/远景），网络被明确教导"这些像素把概率押给 bin 0"——正因为教过，下一张卡片的"丢弃"才有意义。

### 卡片 6-20 ｜[00:39:54]–[00:40:02] 所以 bin 0 的预测"不准"

> **原话** [00:39:54][00:39:57][00:40:02]："所以说我们预测的 depth 为 0——为索引为 0 的这个 depth，其实是不准的。"

- 【直译】bin 0 的概率值不代表任何具体距离——它是"天空/超远/无效"的混合收容所，拿它当一个深度档去投影毫无意义。
- 【为什么】更精确地说不是"预测不准"，而是 **bin 0 没有可用的几何语义**：LSS 外积时每个 bin 的概率会把特征放到"该 bin 对应距离"的 BEV 位置上，而 bin 0 对应的距离是"量程外的任何地方"，根本没有一个合法的 BEV 格子可放。若强行给 bin 0 指派一个距离（比如 0 米或最大距离），天空像素的特征就会大量污染自车脚下或 BEV 边缘的格子。
- 【形状】危害有多大可以估算：路面场景里天空+远景常占画面 1/3 以上，即 88×160≈14080 像素中数千个像素的特征×它们押在 bin 0 上的高概率——不丢弃的话，这是一股不小的垃圾流。
- 【连接】BEVFusion/LSS 原版没有这个问题的对偶处理：它们的 frustum 只建在量程内（D 个档全部有物理距离），量程外信息表现为"整条分布都很平"——代价是天空特征仍被摊薄写进 BEV。DenseBEV 的 dustbin 方案更干脆：显式学出无效概率并整体丢掉，BEV 更干净。这是个值得写进你面试素材库的对比点。

### 卡片 6-21 ｜[00:40:02]–[00:40:10] 丢弃 bin 0，只取 1~99（重点句，5 角度）

> **原话**（合并 [00:40:02][00:40:04][00:40:06][00:40:10]）："所以说我们在这里会把——会只取出——会把为 0 的这个预测的 depth 会给它舍弃掉，只取出 1 到 99 这些 bin 所预测的（概率）。"

- 【直译】投影用的概率里，bin 0 那一层被整体清零；只有 bin 1~99（真正对应物理距离的 99 档）参与后续外积与 splat。
- 【代码】实现方式很讲究（帧 00_39_46 高亮行 360–362）：

```python
360  depth_prob = depth_prob_.new_zeros(depth_prob_.shape)   # 全零同形张量(同dtype同device)
362  depth_prob[:, :, 1:] += depth_prob_[:, :, 1:]            # 只拷1~99层, bin0保持0
```

  不用 `depth_prob_[:, :, 1:]` 直接切片砍成 99 层，而是**保持 100 层、把第 0 层置零**。好处：下游所有 shape（外积、grid map、splat 索引）不用因为 99≠100 而全部改写；ONNX 图里也只是 zeros+slice+add 三个标准算子。
- 【形状】depth_prob 仍为 (3,7,100,88,160)，但每像素沿 bin 维的和变成 `1 − p(bin0)` ≤ 1——**分布被有意打破归一化**。这不是 bug：p(bin0) 越大（越可能是天空/超远），该像素注入 BEV 的总能量越小，趋近于"隐形"。丢弃动作同时兼任了 soft 的有效性掩码。
- 【为什么】为什么不清零后再重新归一化（除以 1−p(bin0)）？重归一化会把"90% 概率是天空"的像素剩余 10% 概率重新放大成 100%，垃圾像素又变成满功率发射器——恰恰毁掉了 dustbin 的过滤作用。保留缩水后的能量，正是设计意图。
- 【连接】同思想在你熟悉的地方反复出现：radar 点的 RCS 置信度加权、CenterPoint 解码时低分 box 直接砍、注意力里的 masking——**"用学出来的置信度当乘性门控"**。另外注意帧 00_39_26 第 363–371 行：鱼眼分支（dustbin_additional 时）还会乘一张 `fisheye_mask`（来自 labels，`F.interpolate(scale_factor=0.5, mode='nearest')` 对齐半分辨率）把车身自遮挡区域也清零——同一门控思想的又一应用，本次运行未走该支⚠。

### 🔨 动手练习 ch6-3：softmax + dustbin_zero 的能量语义

```python
import torch
import torch.nn.functional as F

bs, n, D, h, w = 3, 7, 100, 2, 3          # 空间维缩小便于打印
depths_logit = torch.randn(bs, n, D, h, w)
# 人为制造一个"天空像素"：bin0 打分极高
depths_logit[0, 0, 0, 0, 0] = 8.0

depth_prob_ = F.softmax(depths_logit, dim=2)
print(depth_prob_.sum(2)[0, 0, 0, 0].item())      # 预期: 1.0000 (softmax后归一)

# dustbin_zero: 保形清零而非切片
depth_prob = depth_prob_.new_zeros(depth_prob_.shape)
depth_prob[:, :, 1:] += depth_prob_[:, :, 1:]

print(depth_prob.shape)                            # 预期: torch.Size([3,7,100,2,3]) 形状不变
print(depth_prob[:, :, 0].abs().max().item())      # 预期: 0.0  (bin0全零)
print(depth_prob.sum(2)[0, 0, 0, 0].item())        # 预期: ≈0.02 (天空像素总能量≈被熄灭)
print(depth_prob.sum(2)[0, 0, 1, 1].item())        # 预期: ≈0.99 (普通像素能量基本保留)
# 结论:丢bin0 = 给每个像素乘了一个(1-p_无效)的软门控,天空自动"隐形"
```

**【小结】** ①`F.softmax(depths_logit, dim=2)` 把 100 维打分变成深度概率分布（部署轨用 `custom_softmax` 替身）；②bin 0 是 GT 制作时约定的"超远/无效垃圾桶"（溢出深度→最后一个 bin→重映射为 0），因此其预测无几何语义；③`new_zeros + [:, :, 1:] +=` 保形清零 bin 0，让分布总和=1−p(无效)，兼任软有效性门控——只有 bin 1~99 携带能量进入 Ch7 的外积。

---

## Part 5（00:40:24–00:40:59）迈进 BEVProjector：投影器视角的再一次拆维

**导读**：输入整理完毕，`prj(fv_feat, grid, depth_prob, pinhole=is_pinhole)`（帧 00_40_25 第 384 行）把三件套交给真正的投影器。画面跳进 `class BEVProjector(nn.Module)` 的 `forward`（帧 00_40_58），这是一个带完整英文 docstring 的通用投影器——它先做本章最后一次维度翻译：把 (3,7,…) 再拆出显式的时序维得到 (3,1,7,…)。外积、拍平、grid_sample 都在它肚子里，留给 Ch7。

### 卡片 6-22 ｜[00:40:24] "这个就是 project——我们做 LSS 投影的模块了"

> **原话** [00:40:24]："然后这个就是 project，就是我们做 LSS 投影的一个模块了。"（其后 00:40:24–00:40:54 约 30 秒讲者静默翻代码，无口播）

- 【直译】跨过一道模块边界：从"管家" `ModuleBevProject`（管分组、管输入整理）进入"工人" `BEVProjector`（真投影）。调用现场（帧 00_40_25 第 378–386 行）：

```python
378  if self.grid_w_c:
379      grids_w, grids_c = grid
381      bev_prj_rslt = prj(fv_feat, grids_w, grids_c, depth_prob, pinhole=is_pinhole)
382  else:                                                   # ← 本次走这支
383      is_pinhole = i < 2 if self.split_pinhole else i == 0  # i==0 → 针孔
384      bev_prj_rslt = prj(fv_feat, grid, depth_prob, pinhole=is_pinhole)
385  outs.append(bev_prj_rslt)
386  depth_probs.append(depth_prob)
```

- 【代码】`BEVProjector` 的自我介绍（帧 00_40_58 docstring 逐行抄录，全视频最完整的接口文档）：

```python
 37  class BEVProjector(nn.Module):
 48      def prj_with_depth(self):
 49          return self.proj_type in ('caddn', 'liftsplat')   # 两种带深度分布的投影流派
 51      def forward(self, fv_feat, grid_map, depth_probs=None, flip_lr=None, pinhole=True):
 52          """ Args:
 53              fv_feat (Tensor): (B, t * n, C, H, W), front view feature maps,
 55                  where t is the number of tempo frames, n is the number of views
 56              grid_map (Tensor): the grid map used to project front view to BEV
 57                  vconv_grid_map: (B, n, bev_h, bev_w, 2)
 58                  oft_grid_map:   (B, n, bev_h, bev_w, bev_c, 4)
 59              depth_probs (Tensor): (B, t * n, D, H, W), depth distributions used by
 60                  liftsplat projection to project front view to BEV
 61          """
```

- 【为什么】这层"管家/工人"分离让 `BEVProjector` 保持相机类型无关：它不知道什么叫针孔/鱼眼，只认 (B, t*n, C, H, W)+grid+depth 的抽象接口，`pinhole=` 只是个布尔提示。管家把异构性（分组、bin 数、grid 尺寸）全消化掉了——这是典型的**策略模式**分层，你以后自己写多相机投影模块时照抄这个结构不会错。
- 【连接】`proj_type in ('caddn','liftsplat')` 再次点名两大深度分布流派（LSS 系 / CaDDN 系）；`flip_lr` 参数暗示 BEV 级左右翻转增广的支持（本次 None）。docstring 里 `vconv_grid_map` 与 `oft_grid_map` 并列，说明这套投影器还兼容 OFT（Orthographic Feature Transform，无深度、按高度柱采样）方案⚠——一个 forward 接口背着三代技术路线的历史。

### 卡片 6-23 ｜[00:40:54]–[00:40:59] 最后一变：(3,7,…) → 3×1×7×128×88×160（重点句，5 角度）

> **原话** [00:40:54][00:40:59]："再变一下就变成了 3×7×1——（更正）3×1×7×128×88×160。"（讲者先口误报成 3×7×1，随即自己更正维度顺序⚠）

- 【直译】投影器进门先把 (3, 7, 128, 88, 160) 再 view 成 (3, **1**, 7, 128, 88, 160)：在 bs 与相机之间插入显式的时序帧维 t=1。
- 【代码】帧 00_40_58 第 62–70 行（高亮行 62）：

```python
 62  bs, nt, fc, fh, fw = fv_feat.size()      # bs=3, nt=7
 63  assert nt % self.frame_num == 0          # 7 % 1 == 0 ✓
 65  num_views = nt // self.frame_num         # 7//1 = 7
 66  fv_feat = fv_feat.view(bs, self.frame_num, num_views, fc, fh, fw)   # → (3,1,7,128,88,160)
 68  if self.prj_with_depth:
 69      _, _, dc, dh, dw = depth_probs.size()
 70      depth_probs = depth_probs.view(bs, self.frame_num, num_views, dc, dh, dw)  # → (3,1,7,100,88,160)
```

- 【形状】为什么 t=1 而不是 3？因为本工程早把 3 帧折进了 bs（bs=3=B×T），到投影器眼里"每个样本"就是单帧的 7 路相机，所以 `self.frame_num=1`、num_views=7。docstring 的 (B, t*n, …) 是接口的通用承诺，本配置下退化为 t=1——讲者下一章开头（[00:41:12]）那句"这个是 1 可以现在不用管"说的正是这个维度。depth 同步变成 (3,1,7,100,88,160)（[00:41:04] 转写"3×7×7×100×…"为误听⚠，中间应为 1）。
- 【为什么】既然 t=1，这次 view 不是多此一举吗？不是：①接口兼容——同一 BEVProjector 也服务于"不折帧、t=3 直进"的配置，`assert nt % frame_num == 0` 就是这份兼容的守门员；②紧接着的 `for i in range(self.frame_num)` 循环（帧 00_40_58 第 75 行起）按帧逐个投影，维度显式化后循环体才写得干净。
- 【连接】至此本章完成了同一份数据的第三次维度翻译：`(21,C,H,W)`（卷积视角：一堆图）→ `(3,7,C,H,W)`（分组视角：每时刻 7 路）→ `(3,1,7,C,H,W)`（投影器视角：样本×帧×相机）。Ch7 将从这里接手，做外积 (3,1,7,128,**100**,88,160)、拍平与 grid_sample。你在 BEVFusion 里看到的 (B,N,D,H,W,C) frustum 即将有一个"拉取式"的对偶版本。

### 🔨 动手练习 ch6-4：BEVProjector 进门四行的完整复现

```python
import torch

def bev_projector_entry(fv_feat, depth_probs, frame_num=1):
    """复现帧00_40_58第62-70行"""
    bs, nt, fc, fh, fw = fv_feat.size()
    assert nt % frame_num == 0, "相机总数必须能被帧数整除"
    num_views = nt // frame_num
    fv_feat = fv_feat.view(bs, frame_num, num_views, fc, fh, fw)
    _, _, dc, dh, dw = depth_probs.size()
    depth_probs = depth_probs.view(bs, frame_num, num_views, dc, dh, dw)
    return fv_feat, depth_probs

fv  = torch.randn(3, 7, 128, 88, 160)     # 本视频配置: 3帧已折入bs
dep = torch.randn(3, 7, 100, 88, 160)
f, d = bev_projector_entry(fv, dep, frame_num=1)
print(f.shape)  # 预期: torch.Size([3, 1, 7, 128, 88, 160])
print(d.shape)  # 预期: torch.Size([3, 1, 7, 100, 88, 160])

# 同一函数服务另一种配置: 不折帧、t=3 直进 (bs=1, nt=21)
f2, d2 = bev_projector_entry(torch.randn(1, 21, 128, 88, 160),
                             torch.randn(1, 21, 100, 88, 160), frame_num=3)
print(f2.shape)  # 预期: torch.Size([1, 3, 7, 128, 88, 160])
# 预告Ch7: 外积 f.unsqueeze(4)*d.unsqueeze(3) → (…,7,128,100,88,160), 自己算算多少GB?
```

**【小结】** ①`prj(fv_feat, grid, depth_prob, pinhole=is_pinhole)` 把三件套交给通用投影器 `BEVProjector`，其 docstring 明确了 (B, t*n, C, H, W)/(B, n, bev_h, bev_w, 2)/(B, t*n, D, H, W) 的接口契约；②进门先按 `frame_num` 拆出显式时序维，本配置 t=1，得到 (3,1,7,128,88,160) 与 (3,1,7,100,88,160)；③本章结束时，外积所需的两个操作数与查表所需的 grid map 已全部就位——Ch7 正式开演"Lift-Splat"。

---

## 本章总小结

1. **输入=4 类 6 张量**：针孔/鱼眼的 8 倍下采样图像特征（(21,128,88,160)/(12,128,64,96)）、depth logits（100 bin/32 bin）、DataLoader 预计算的 grid map（(3,7,224,112,2)/(3,4,16,16,2)⚠）——几何在 CPU、采样在 GPU，是与 BEVFusion 现算 frustum 最大的工程分歧。
2. **整理=三次维度翻译**：(21,C,H,W)→(3,7,C,H,W)→(3,1,7,C,H,W)，bs=3 恒等于 batch×3 帧；bin 数全程靠 `-1` 推断，针孔/鱼眼共享同一循环体。
3. **深度分布=softmax(dim=2) + dustbin_zero**：bin 0 是 GT 约定的超远/无效垃圾桶，投影前用 `new_zeros+[:, :, 1:]+=` 保形清零，使每像素注入 BEV 的能量=1−p(无效)，天空/远景自动"隐形"；只有 bin 1~99 进入 Ch7 的外积。

### ⚠ 本章存疑清单（汇总）

| 编号 | 位置 | 疑点 | 我的推断与依据 |
|---|---|---|---|
| ⚠1 | 卡片 6-4 [00:36:41] | 转写"飞雪的尺度是88×160" | 应为"feature 的尺度"（或"分辨率"）之误听；88×160 与所有帧证一致 |
| ⚠2 | 卡片 6-8 框图 | 鱼眼 grid `(bs*3)*4*16*16*2` 的 16×16 | 缩略图字小难辨；但与 `fisheye_bev_tag_dict crop_coords [286,96,318,128]`（32×32 全分辨率→16×16 半分辨率）自洽，倾向为真；建议对代码核实 |
| ⚠3 | 卡片 6-10 代码 | `self.god_use_hist_seq_len` 的 `god` 前缀含义 | 屏幕清晰可读为 god，疑为项目内部配置命名（如 qat/god 平台开关），语义=历史帧序列长度 |
| ⚠4 | 卡片 6-12 [00:38:24-33] | 转写"BV的feature做Shift操作" | 勘误为"**FV** 的 feature 做 **view（reshape）**操作"；依据：代码变量 fv_feat、该处仅有三行 view、随后报出的正是 reshape 后 shape |
| ⚠5 | 卡片 6-15 [00:39:02] | 转写残句"其实是一个内" | 推断讲者想说"是一个 logit/未归一化打分"；依据：变量名 depths_logit + 下一句"所以要取 softmax" |
| ⚠6 | 卡片 6-17 [00:39:21] | "看最大概率在哪个 bin"表述 | 为直觉化口误：代码不做 argmax，用整条分布加权外积（LSS 标准做法） |
| ⚠7 | 卡片 6-18/6-19 | "分到最后一个 bin"与"映射成 0"的先后细节 | 讲者叙述结巴；综合理解为：溢出深度先归入溢出档、该档 GT 重映射为索引 0，最终 bin0=无效档、1~99=有效档，与 `dustbin_zero` 开关及 `[:, :, 1:]` 切片互证 |
| ⚠8 | 卡片 6-21 | 鱼眼 `fisheye_mask` 分支（dustbin_additional 内） | 本次运行走 dustbin_zero 未执行；乘 mask 清车身自遮挡为从代码直读，未经讲者口述确认 |
| ⚠9 | 卡片 6-22 | `oft_grid_map (B,n,bev_h,bev_w,bev_c,4)` 的 OFT 兼容路线 | 仅见于 docstring，本次未走；判断为历史技术路线遗留 |
| ⚠10 | 卡片 6-23 [00:40:54] | 讲者先报"3×7×1"后改"3×1×7×…" | 以更正后 + 帧 00_40_58 代码 `view(bs, frame_num, num_views, …)` 为准；[00:41:04] 的"3×7×7×100"同理应为 3×1×7×100 |
| ⚠11 | 卡片 6-3/6-5 | 原图分辨率 704×1280（针孔）/512×768（鱼眼）为 8 倍反推 | 视频未口述原图尺寸，仅由 88×160/64×96×8 反推，若前端另有 resize/crop 则数值不同 |
| ⚠12 | 卡片 6-10 | `grid_w_c` 开关的 w/c 含义 | 本次为 False 未走；从 `grids_w, grids_c = grid` 只能看出是两张成对 grid，具体含义（如 warp/coord？）未核 |


---
> [[Ch05_LidarRadar融合|← Ch5]] · [[00_总览与脉络|📖 总览]] · [[Ch07_LSS投影本体|Ch7 →]]

