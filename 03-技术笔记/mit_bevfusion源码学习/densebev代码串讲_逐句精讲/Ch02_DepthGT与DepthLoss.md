> [[Ch01_DepthNet网络结构|← Ch1]] · [[00_总览与脉络|📖 总览]] · [[Ch03_Lidar透传与Radar编码图解|Ch3 →]]

# Ch2 Depth GT 与 Depth Loss（00:06:42–00:13:04）

> **本章在全局地图的位置**：`[图像backbone→FPN收尾→DepthNet] → 【你在这里：Depth GT 生成 + Depth Loss】 → Lidar Backbone → … → LSS投影`。
> 上一章结尾，DepthNet 已经用两个卷积把下采样 8 倍的图像特征变成了 `depth_logits`，debug 台实测 shape 为 `torch.Size([21, 100, 88, 160])`（21 = 3 帧 × 7 路针孔，100 = 深度 bin 数）。本章讲的是：**这 100 个通道的分类 logits，拿什么当标签、怎么算损失**。
>
> **涉及两个文件**（帧内路径栏可见）：
> - `e2e/tasks/bev_task/uvp_module/models/streampetr/streampetr_neck/fpn_forward.py` —— `class DepthNet(BaseModule)` 的 `forward`，在第 631 行调用 `ddn_loss(...)`；
> - `e2e/tasks/bev_task/uvp_module/models/loss/ddn_loss.py` —— `class DDNLoss(nn.Module)`，`forward` 在第 122 行，工具函数 `bin_depths` 在第 234 行。
>
> **本章帧证据**（精读单帧）：`00_06_53 / 00_07_24 / 00_07_44 / 00_08_41 / 00_08_49 / 00_09_01 / 00_09_30 / 00_09_58 / 00_10_11 / 00_10_54 / 00_11_44 / 00_12_02 / 00_12_28`，共 13 张；总览 sheet_04~07。
>
> **读前术语提醒**：本段转写里反复出现的「机械 / 机器」都是讲者说的 **GT**（ground truth）被听写错了（校正稿文件头未收录此条，本章统一按 GT 理解）；「分辨的操作」=「分 bin 的操作」；「collapse」按帧证应为 **clamp**。

---

## Part 2.1 DDNLoss 的入口：预测、GT、Mask 三件套（00:06:42–00:07:01）

**本段导读**：DepthNet 前向刚结束（上一章），讲者从 `fpn_forward.py` 第 631 行的 `depth_loss = ddn_loss(depth_logits, depths, depth_masks, fg_masks)` 跳进 `ddn_loss.py` 的 `DDNLoss.forward(self, depth_logits, depth_maps, depth_masks, fg_masks=None)`。本段就是把这三个（外加一个默认 None 的第四个）输入认清楚：**预测 logits 是 4 维带 bin 通道的，GT 和 Mask 是 3 维不带通道的**——这个维度差正是"深度当分类做"的直接体现。输出是一个标量 loss。

---

### 句卡 2.1.1 [00:06:42 + 00:06:48]（合并 2 句过渡语）
**原话**：「然后在这里的话会讲一下我们在计算 Depth 的一个 Loss，然后再看一下这块是怎么处理的。」

- 【直译】前面讲完了 DepthNet 网络本体（两个卷积出 100 通道），现在切换话题：这 100 通道的输出怎么被监督、损失怎么算。讲者随即从调用点跳进了损失类内部。
- 【代码】调用点在 `fpn_forward.py` 第 631 行（帧 00_06_53 黄色高亮行）：
  ```python
  fg_masks = None                                                    # L630
  depth_loss = ddn_loss(depth_logits, depths, depth_masks, fg_masks) # L631
  depth_loss *= self.ddn_weight                                      # L632
  ```
  `ddn_loss` 是 `DDNLoss` 的实例，`nn.Module` 直接被调用即触发 `forward`。注意还有一个总权重 `self.ddn_weight`，说明 depth loss 在多任务总 loss 里是加权项。
- 【为什么】"DDN" = Deep/Categorical **D**epth **D**istribution **N**etwork 的缩写习惯，出自 CaDDN（CVPR 2021）一系的做法：深度不回归一个数，而是预测离散分布再监督分布。DenseBEV 沿用了这个命名和代码骨架。
- 【连接】你训练过的 BEVFusion（mmdet3d 版）里 LSS 的深度分布是**无显式监督**、纯靠检测 loss 反传学出来的；而 DenseBEV 这里是 **BEVDepth 路线**——用 lidar 投影得到的稀疏深度真值显式监督深度分布。这是两条技术路线最重要的分叉点之一，面试常问"BEVDepth 相对 LSS/BEVFusion 改了什么"，答案的核心就是本章。

---

### 句卡 2.1.2 [00:06:50]
**原话**：「然后它的输入就是我们预测的一个 Depth。」

- 【直译】第一个入参是网络预测的深度，即上一章 DepthNet 两个卷积吐出来的 logits。
- 【代码】对应形参 `depth_logits`。帧 00_07_24 里 `forward` 的 docstring 写得很清楚：
  ```python
  def forward(self, depth_logits, depth_maps, depth_masks, fg_masks=None):
      """
      Args:
          depth_logits (Tensor): (B * n, D, H, W), where n is the number of views
          depth_maps   (Tensor): (B * n, H, W)
          depth_masks  (Tensor): (B * n, H, W)
      """
  ```
- 【形状】针孔分支实测 `(21, 100, 88, 160)`：B=1，n=21（3 帧×7 相机），D=100 个深度 bin，88×160 是 1/8 分辨率特征图（原图 704×1280）。注意它是 **raw logits**，还没过 softmax——focal loss 内部才做归一化。
- 【为什么】把 view 维折进 batch 维（B*n）而不是单独留一维，是因为逐像素深度监督对"每张图"是完全独立的，拍平成大 batch 能直接复用 2D 的 loss 写法，不用写循环。
- 【连接】和 CenterPoint 检测头输出 heatmap logits 再算 focal loss 是同一个套路：网络出 logits，loss 函数内部做概率化。

---

### 句卡 2.1.3 [00:06:54]
**原话**：「然后以及这个是 Depth 的一个 GT。」（转写作"机械"，实为 GT）

- 【直译】第二个入参是深度真值图 `depth_maps`：每个像素位置放一个"这个像素到底多深"的数。
- 【代码】`depth_maps (Tensor): (B*n, H, W)`。它来自 Depth Loader（数据侧）：把 lidar 点云投影到各相机像平面，落在哪个 1/8 网格就把该点深度填进去；没有点投到的像素填 0。这在 forward 之前的 `fpn_forward.py` 623~629 行还做过整形（见句卡 2.5.4）。
- 【形状】`(21, 88, 160)`，比预测少了 D 那一维——GT 是"每像素一个连续值"，预测是"每像素 100 类打分"，本章的主要工作正是把前者变成后者能比的标签（bin 索引）。
- 【为什么】GT 分辨率直接做成和特征图一样的 1/8（88×160），而不是全分辨率再下采样，省显存也避免上/下采样引入的深度混叠。
- 【连接】BEVDepth 论文里对应 `get_downsampled_gt_depth`：把 lidar 投影深度在每个 16×16 patch 里取 min 得到低分辨率 GT。DenseBEV 的 loader 里怎么池化（min/均值）视频没讲，⚠ 建议对代码核实（校正稿头也提示了这一点）。

---

### 句卡 2.1.4 [00:06:58 + 00:07:01]（合并 2 句）
**原话**：「然后这个是我们 Depth 的 Mask，就是哪些值是有效的一个 Depth。」

- 【直译】第三个入参是有效位掩码：lidar 点是稀疏的，投到图像上只有一小撮像素有真值，其余像素不该算 loss。Mask=1 的地方才监督。
- 【代码】`depth_masks (Tensor): (B*n, H, W)`，浮点 0/1 图。最终用法在 forward 结尾（帧 00_12_02，L189-191）：
  ```python
  loss *= depth_masks
  num_pixels = depth_masks.sum() + 1e-6
  loss = loss.sum() / num_pixels
  ```
  即"逐像素 loss × mask，再除以有效像素数"——按有效像素做平均，而不是按全图 88×160×21 平均。
- 【形状】`(21, 88, 160)`，与 GT 逐像素对齐。
- 【为什么】不加 mask 的话，占绝对多数的无 lidar 像素（深度=0）会把梯度淹没：网络会学成"到处预测 0 深度"。`+1e-6` 是防止某个 batch 恰好一个有效像素都没有时除零。
- 【连接】和 CenterPoint 回归分支只在正样本中心点位置算 L1（用 ind/mask gather）是同一个思想：**稀疏监督必须显式圈出有效位置并按有效数归一化**。

---

### 🔨 动手练习 ch2-1：三件套形状与稀疏监督
```python
import torch

B, n_frames, n_cams = 1, 3, 7
n = n_frames * n_cams                       # 21
D, H, W = 100, 88, 160

depth_logits = torch.randn(B * n, D, H, W)        # 网络预测 (21,100,88,160)
depths      = torch.rand(B, n, H, W) * 0.12       # 归一化后的GT, 量程0~0.12(见Part2.2)
depth_masks = (torch.rand(B, n, H, W) > 0.95).float()   # 模拟lidar稀疏投影:约5%像素有效

# 复现 fpn_forward.py L627-629 的整形
_, _, dh_gt, dw_gt = depths.size()
depths      = depths.view(-1, dh_gt, dw_gt)       # (21,88,160)
depth_masks = depth_masks.view(-1, dh_gt, dw_gt)  # (21,88,160)

print(depth_logits.shape, depths.shape, depth_masks.shape)
# torch.Size([21, 100, 88, 160]) torch.Size([21, 88, 160]) torch.Size([21, 88, 160])
print(f"有效监督像素占比: {depth_masks.mean():.3f}")   # ≈0.050 —— 稀疏监督必须mask+按有效数平均
```

**【小结】** DDNLoss 吃三个张量：4 维预测 logits `(B*n,100,H,W)`、3 维连续 GT `(B*n,H,W)`、3 维有效掩码 `(B*n,H,W)`。维度差一个 D=100，宣告了"深度是 100 类分类问题"。lidar 稀疏投影决定了必须 mask 加权、按有效像素归一化。

---

## Part 2.2 GT 预处理：有效掩码与"除焦距平方"归一化（00:07:07–00:08:17）

**本段导读**：进 forward 后第一件事是造 `valid_depth_mask`（GT > 最小深度才算有效）；紧接着讲者补了一个**藏在数据侧的大前提**：Depth Loader 里已经把以米为单位的原始深度**除以了各相机焦距的平方**做归一化——所以 forward 里见到的 `depth_maps` 根本不是米。这个前提不搞清楚，后面"60 变 0.12"会完全看不懂。本段输入是原始 GT 图，输出是"归一化域上的 GT + 有效掩码"。

---

### 句卡 2.2.1 [00:07:07 + 00:07:22 + 00:07:24]（合并 3 句）
**原话**：「在这里会把我们的 Depth GT 按照自己设定的一个最小值去 Mask。当前设定的最小值是 0，所以说这个应该和原始值是一样的。」

- 【直译】用一个阈值 `real_min_depth` 筛 GT：比它小的深度视为无效。当前配置阈值为 0，讲者说"跟不筛差不多"。
- 【代码】帧 00_07_24 黄色高亮的 L130：
  ```python
  valid_depth_mask = (depth_maps > self.real_min_depth)   # L130
  ```
  注意是严格大于 `>`。后续两个用途（帧 00_08_41，L157-162）：
  ```python
  if self.use_projected_only:
      # depth in [0, self.real_min_depth] => none supervise
      depth_masks *= valid_depth_mask
  else:
      # depth in [0, self.real_min_depth] => supervised as background
      depth_target[~valid_depth_mask] = self.num_bins
  ```
- 【形状】`(21, 88, 160)` 的 bool 图。
- 【为什么】⚠ 讲者说"和原始值一样"其实**不严格**：`real_min_depth=0` 时 `depth_maps > 0` 恰好把"没有 lidar 点投到、填 0"的背景像素全部筛成 False——这正是区分"有真值/无真值"的关键一步，绝不是空操作。代码注释也写明了两种策略：无真值像素要么完全不监督（`use_projected_only`），要么被当成"背景类"（塞进第 num_bins 个 dustbin，见 Part 2.4）监督。
- 【连接】和 CenterPoint heatmap 的"负样本也参与监督（背景=0）vs 忽略区域不监督"是同一类设计选择；也对应 nuScenes 数据链里"深度图 0 = 无投影"的约定。

---

### 句卡 2.2.2 [00:07:28]
**原话**：「然后在这里会把我们预测的 Depth Target 的一个 GT 会做一个分 bin 的一个操作。」（转写"分辨"=分 bin）

- 【直译】接下来把连续值 GT 离散化成 bin 索引——这是本章的核心动作，讲者先预告，细节在 Part 2.3 展开。
- 【代码】帧 00_07_44 黄色高亮的 L148（`use_dorn` 为 False 走 else 分支）：
  ```python
  depth_target = bin_depths(depth_maps, self.discretiziation_mode,
                            self.depth_min, self.depth_max,
                            self.num_bins, target=True)     # L148
  ```
  ⚠ 帧内确认代码里的属性名真的拼成 `discretiziation_mode`（多了个 i，正确拼写是 discretization），工作代码里的历史拼写错误，抄配置时要照抄错的才对得上。
- 【形状】进 `(21,88,160)` float，出 `(21,88,160)` int64（每个像素一个 0~100 的类别号）。
- 【为什么】分类式深度（离散分布）而非回归一个标量，好处有三：① LSS 投影天然需要"每个 bin 一个概率"来做外积；② 分类 + focal loss 对稀疏噪声 GT 比 L1 回归稳；③ 分布能表达多峰不确定性（物体边缘一半前景一半背景）。
- 【连接】上面还有一个没走的 `if self.use_dorn:` 分支（帧 00_07_44 可见 L131-146）：把标签做成 DORN 式**有序回归**的 `(ord_c0, ord_c1)` 两段式编码（`torch.linspace` 造 label_mask、`torch.cat((ord_c0, ord_c1), dim=1)`）。DenseBEV 保留了这条路但没启用——读代码时认出它是 DORN（CVPR 2018 序数深度回归）即可跳过。

---

### 句卡 2.2.3 [00:07:43 → 00:08:17]（合并 8 句：00:07:43/48/53/57、00:08:04/08/14/17，讲者此处有车轱辘重复）
**原话**：「对这里的 Depth GT，它在 Depth Loader 的时候，我们从原始的 OBS 上取的一个数据，它的数值其实还是以米为单位的。然后在 Depth Loader 的时候，因为每个相机它的焦距不一样，所以说会对它做一个归一化的操作——就是会分别除上它对应相机的焦距的一个平方。所以说对于每一路针孔相机的 Depth GT，其实它都不会再是以米为一个单位的数值了。」

**（重点句，5 角度）**

- 【直译】数据集里存的深度是米；但喂进网络前，loader 按"该像素属于哪台相机"把深度除以了那台相机焦距的平方。于是 forward 里见到的 GT 是个无量纲(或奇怪量纲)的小数，不同相机的同一米数对应不同的数值。
- 【代码】等价伪代码（Depth Loader 侧，视频未展示源码）：
  ```python
  # depth_m: 某相机的投影深度图(米); f: 该相机焦距(像素)
  depth_norm = depth_m / (f ** 2)        # ⚠ 讲者原话:除焦距的平方
  # 更常见的做法(BEVDepth虚拟深度)是 depth_m * (f_ref / f)，一次方
  ```
- 【形状】不变，`(H,W)` 逐像素标量变换；但**数值域**从 [0,60] 米变成了 0~0.12 左右的小数（下一张卡给出 0.12 的来历）。
- 【为什么】这是"焦距感知深度/虚拟深度"技巧：同一个物体，焦距长的相机拍出来占的像素多，网络从**图像外观**只能推断 `d/f` 这个组合量，推不出绝对米数。若 7 路针孔焦距不同而标签都用米，等于逼网络给同样的外观回归不同答案，学不动。除以焦距（的函数）后，标签变成外观可推断的量，各相机共享同一个 DepthNet 才成立。反过来，LSS 投影前必须**乘回去**还原成米（后续章节的投影模块会体现）。
- 【连接+⚠】BEVDepth 论文的 virtual depth 与 DD3D 的做法都是**一次方**（d·f_ref/f 或 d/f）；讲者两次明确说"平方"。平方在几何上对应"面积/尺度平方"类的归一化，也有工作用 f² 配合像素面积推导。**数值旁证**：下一卡里 60 m × 0.002 = 0.12，0.002=1/500——若是 d/f 且名义焦距≈500 像素（1/8 特征图尺度上 4000/8=500 这种量级也说得通），数值刚好自洽；若是 d/f²，0.002 就得是 `f_nom²·k` 拼出来的复合系数。**结论存疑，标 ⚠：归一化到底是 f 还是 f²，建议直接翻 Depth Loader 源码确认**。不影响本章主线：GT 与 bin 边界用同一系数变换，bin 索引不变。

---

### 🔨 动手练习 ch2-2：焦距归一化为什么能让多相机共享 DepthNet
```python
import torch

# 两台相机拍同一个3米高的杆子, 距离都是30米
f_tele, f_wide = 1000., 500.        # 焦距(像素): 长焦 vs 广角
h_pix_tele = 3.0 * f_tele / 30.0    # 成像高度 = H*f/d = 100 像素
h_pix_wide = 3.0 * f_wide / 30.0    # = 50 像素  → 外观不同、深度相同!

# 用"米"当GT: 外观不同却要求输出相同 → 网络困惑
# 用 d/f 当GT: 标签 = H / h_pix, 只依赖外观(像素高度), 与相机无关
gt_tele = 30.0 / f_tele             # 0.030
gt_wide = 30.0 / f_wide             # 0.060
print(3.0 / h_pix_tele, 3.0 / h_pix_wide)   # 0.03 0.06 —— 恰好等于 d/f, 外观可直接推出标签

# 名义系数0.002(=1/500)下, 60米 → 0.12, 即本视频里 depth_max 的来历
print(60 * 0.002)                   # 0.12
```

**【小结】** forward 开头用 `depth_maps > real_min_depth`（阈值 0）造出有效掩码，顺手把"无 lidar 投影=0"的背景像素分了出去。更重要的是数据侧前提：GT 已按各相机焦距（讲者称平方，⚠待核）归一化，不再以米为单位——这是多相机共享一个 DepthNet 的前提，也是接下来 depth_max=60 要乘 0.002 变 0.12 的原因。

---

## Part 2.3 bin_depths：0~60 米（归一化后 0~0.12）切成 100 个 bin（00:08:26–00:10:15）

**本段导读**：镜头进入 `ddn_loss.py` L234 的工具函数 `bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False)`。它的 docstring 明说功能是 "Converts depth map into bin indices"，并给出三种离散化模式（UD/LID/SID，引用 arxiv 2005.13423）。DenseBEV 走 **UD 均匀离散化**：量程 `(depth_min=0, depth_max=0.12)` 均分 100 份，`(GT−min)/bin_size` 后取整。输入连续 GT 图，输出整数 bin 索引图——这就是 focal loss 需要的类别标签。

---

### 句卡 2.3.1 [00:08:26 + 00:08:36 + 00:08:40]（合并 3 句，其中"然后都是一个操作"为口误/垫话）
**原话**：「然后在这里会去做一个分 bin 的一个操作……它传入的是我们 Depth 的一个 GT。」

- 【直译】正式跳进 `bin_depths` 函数，第一个实参就是刚才那张归一化 GT 图。
- 【代码】函数签名与 docstring（帧 00_08_49，L234-251）：
  ```python
  def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
      """
      Converts depth map into bin indices
      Args:
          depth_map [torch.Tensor(H, W)]: Depth Map
          mode [string]: Discretiziation mode
              (See https://arxiv.org/pdf/2005.13423.pdf for more details)
              UD:  Uniform discretiziation
              LID: Linear increasing discretiziation
              SID: Spacing increasing discretiziation
          depth_min [float]: Minimum depth value
          depth_max [float]: Maximum depth value
          num_bins [int]: Number of depth bins
          target [bool]: Whether the depth bins indices will be used
                         for a target tensor in loss comparison
      Returns:
          indices [torch.Tensor(H, W)]: Depth bin indices
      """
  ```
- 【为什么】`target` 开关很讲究：同一个函数既服务"造训练标签"（target=True，要做越界清洗+取整），也服务"推理/投影时算 bin 位置"（target=False，保留 float）。一份离散化逻辑两处用，保证训练与投影的 bin 边界永远一致。
- 【连接】这段函数（含 docstring 措辞）与开源 CaDDN 的 `bin_depths` 几乎逐字相同——DenseBEV 的深度监督模块是从 CaDDN 系代码移植改造的；docstring 引用的 arxiv 2005.13423 即讨论 UD/SID/LID 三种深度离散化的论文。

---

### 句卡 2.3.2 [00:08:44 → 00:08:53]（合并 4 句）
**原话**：「然后还有我们预测 Depth 的一个最小值、预测 Depth 的一个最大值。最小值和最大值——对针孔是 0，然后最大值是 60。」

- 【直译】另外两个关键实参：量程下界 `self.depth_min` 和上界 `self.depth_max`。针孔相机配置的物理量程是 0~60 米。
- 【代码】调用点 L148 传的是 `self.depth_min, self.depth_max`，即 DDNLoss 构造时从配置读入的成员。
- 【形状】两个标量，决定后面 100 个 bin 覆盖的区间。
- 【为什么】60 米是针孔相机深度监督的截断距离：更远的 lidar 点稀疏且深度分布长尾，硬塞进量程会稀释近处 bin 的分辨率。60 m / 100 bin = 0.6 m/bin，对 10~40 m 主工作区间的障碍物定位够用。鱼眼分支的量程会更短（鱼眼看近处，讲者在别处提过针孔/鱼眼配置不同——本句只给了针孔的数）。
- 【连接】对比你熟的 nuScenes-BEVDepth 常用 2~58 m 切 0.5 m/bin（112 bin）；DenseBEV 是 0~60 m、100 bin、0.6 m/bin，量级相同，思路一致。

---

### 句卡 2.3.3 [00:08:55 → 00:09:03]（合并 4 句）**（重点句，5 角度）**
**原话**：「然后这里的话它其实也会乘了一个归一化的一个系数——乘了一个 0.002，就从 60 变成 0.12 了。」

- 【直译】配置里的 60（米）不会直接用：会乘上归一化系数 0.002，得到 0.12——因为 GT 已经被焦距归一化了，bin 的边界必须做**同样的变换**才能对得上。
- 【代码】帧 00_09_01 有实锤：讲者鼠标悬停在 `bin_depths` 形参 `depth_max` 上，调试器 tooltip 弹出 **`0.12`**。即运行时 `self.depth_max == 0.12`，等价于：
  ```python
  self.depth_max = 60.0 * norm_coeff      # norm_coeff = 0.002
  self.depth_min = 0.0  * norm_coeff      # = 0.0
  ```
- 【形状】标量变换，但决定了整个分类问题的值域：bin 宽 = (0.12−0)/100 = **0.0012**（归一化域）≙ 0.6 米（物理域，名义相机下）。
- 【为什么】离散化只关心 `(GT − min)/bin_size` 这个**比值**。给 GT 和 (min, max) 同乘一个系数，比值不变、bin 索引不变——所以理论上"全用米"或"全用归一化值"结果一样。真正的收益在于**跨相机**：GT 按各自相机焦距归一化，而 (0, 0.12) 是全局统一的量程；于是长焦相机的 60 m 归一化后 < 0.12，等效于"同样 100 个 bin，长焦相机可编码比 60 m 更远的深度"——量程自动随焦距伸缩，这正是焦距归一化在离散化环节的红利。
- 【连接+⚠】0.002 = 1/500 是**名义系数**；若各相机焦距不同，各自的实际归一化系数不同，则"0.12 恰好对应 60 m"只对名义焦距的相机严格成立，其他相机的物理截断距离会偏移（这不是 bug，是 feature，见上一条【为什么】）。另外这个 0.002 应该写在配置/Loader 里，视频没展示出处，⚠ 记入存疑清单。

---

### 句卡 2.3.4 [00:09:05]
**原话**：「然后和我们的 Depth 的 GT 做了一个相同的操作。」

- 【直译】强调一致性：GT 在 loader 里乘过的系数，量程边界在这里也乘——两边同域，比较才有意义。
- 【代码】不变式：`bin_index(d_m/f², 0, 60·c, 100) == bin_index(d_m, 0, 60·(f²/c)⁻¹…)`——写成检查就是"GT 与边界必须同乘同除"。工程上这类"两处必须同步改"的耦合最容易埋雷（改了 loader 的系数忘了改 loss 配置，深度直接全错）。
- 【为什么】这句是讲者的防坑提示：日后你若调整归一化方式（比如 f²→f），`depth_max=0.12` 必须同步重算，否则 100 个 bin 会整体错位，症状是"depth loss 正常下降但 LSS 投影出来的 BEV 特征位置系统性偏移"——非常难查。
- 【连接】和 BEVFusion 里"点云 voxel 化的 point_cloud_range 必须与检测头 post_center_range 对齐"是同款耦合。

---

### 句卡 2.3.5 [00:09:10 → 00:09:25]（合并 4 句：0~60 划 100 份 / 算每一份 / 分成 100 个 bin / 就是 100 份）
**原话**：「然后在这里是把我们的 0 到 60 米划分成了我们的 100 份……分成了 100 个 bin，就是 100 份。」

- 【直译】UD（均匀离散化）第一步：算每个 bin 的宽度。量程除以 bin 数。
- 【代码】帧 00_09_30/00_09_58 黄色高亮 L252 起的 UD 分支：
  ```python
  if mode == "UD":
      bin_size = (depth_max - depth_min) / num_bins   # (0.12-0)/100 = 0.0012
      indices  = (depth_map - depth_min) / bin_size
  ```
- 【形状】`bin_size` 标量 0.0012；`num_bins=100` 正是 depth_logits 的通道数 100 的来历——**网络输出通道数 = 离散化份数**，两处配置必须一致。
- 【为什么】UD 每个 bin 等宽 0.6 m。缺点：远处物体像素少、深度误差本来就大，等宽 bin 对远处"过于苛刻"；LID/SID 让 bin 随距离变宽来适配误差分布。DenseBEV 选 UD 大概率是因为后端 LSS 投影用等间隔深度平面最方便（外积后 grid_sample 的深度轴是均匀的），并且 0~60 m 量程不算长，UD 够用。
- 【连接】视频里能看到但没启用的另外三个分支（帧 00_09_30，L255-263）值得记下：
  ```python
  elif mode == "LID":
      bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
      indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
  elif mode == "LID_FIX":   # y = depth_min + bin_size * x * (x+1) 的正确解
      bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
      indices = -0.5 + 0.5 * torch.sqrt(1 + 4 * (depth_map - depth_min) / bin_size)
  elif mode == "SID":
      indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                (math.log(1 + depth_max) - math.log(1 + depth_min))
  ```
  `LID_FIX` 的中文注释是团队自己加的：CaDDN 原版 LID 的反解系数 8 对应边界公式 `y = min + bin_size/2·x(x+1)`；若把边界定义为 `y = min + bin_size·x(x+1)`，正确反解系数是 4（解一元二次 `bin_size·x² + bin_size·x − (y−min) = 0` 得 `x = −0.5 + 0.5·√(1+4(y−min)/bin_size)`）。团队专门修过这个坑并留档——读工作代码时看到 `_FIX` 后缀，多半就是这种"上游开源实现有争议，我们改了并保留原版对照"。

---

### 句卡 2.3.6 [00:09:33 → 00:09:50]（合并 5 句）**（重点句，5 角度）**
**原话**：「然后在这里呢，会对应的把我们的 GT 减去 Depth 的 min，然后再除上这个 bin 的一个 size，就能够算到我们距离的每一个深度它对应是在哪一个深度的一个区间。这个就是我们当前计算 Depth Loss 的一个 GT 的一个处理过程。」

- 【直译】离散化核心公式：`(GT − depth_min) / bin_size`，得到"这个像素的深度落在第几个 bin"的（暂时还是小数的）索引。
- 【代码】`indices = (depth_map - depth_min) / bin_size`。逐像素、纯逐元素运算，无卷积无邻域。
- 【形状】`(21, 88, 160)` float 进、`(21, 88, 160)` float 出，数值域从 [0, 0.12+] 变到 [0, 100+]（越界值 >100，背景 0 值→索引 0）。
- 【为什么】这就是"把回归问题改写成分类问题"的那一下：连续量 → 类别号。配合等宽 bin，索引公式退化为一次线性映射；若是 LID/SID，公式就是上一卡里的开方/对数反函数。**注意背景像素**：GT=0 → 索引 0，看起来落在"0~0.6 m"这个合法 bin——如果不处理，背景会被当成"极近深度"的正样本教坏网络；所以才有 Part 2.2 的 `valid_depth_mask` 和 Part 2.4 的 dustbin 流程兜底。
- 【连接】完全同构于你在 BEVFusion 里见过的 voxel 化：`(x − x_min)/voxel_size` 算格子号——那是空间轴的离散化，这是深度轴的离散化；也同构于 CenterPoint 把物体中心量化到 heatmap 格子。**离散化三件套（减 min、除步长、取整+越界处理）在 3D 感知代码里无处不在，认熟这个模式，读任何 BEV 代码都快一倍。**

---

### 句卡 2.3.7 [00:10:04 → 00:10:15]（合并 5 句）
**原话**：「然后在这里算的应该还是一个小数值，还是一个 float 的值。然后在这里会对它取一个整，就变成了具体的——它是属于 0 到 100 米里面具体是哪一个深度区间。」

- 【直译】除完 bin_size 得到的是 37.6 这种小数，要向下取整成 37 才是类别号。
- 【代码】取整并不在公式那一行，而在 target 分支的最后（帧 00_10_11，L278-280）：
  ```python
  # Convert to integer
  indices = indices.type(torch.int64)
  return indices
  ```
  `.type(torch.int64)` 对正数即截断取整（floor），且 int64 正是 `F.cross_entropy`/focal loss 对标签 dtype 的要求。
- 【形状】`(21,88,160)` float → `(21,88,160)` int64，取值 0~100（含 dustbin 100，共 101 个可能值——比网络的 100 通道多 1，这个"多出来的 1"就是 Part 2.4 的主角）。
- 【为什么】先做完越界清洗（下一 Part 的 L274-276）**再**取整，顺序是安全的：NaN/负数/超界都先被替换成合法值，取整不会遇到未定义行为。
- 【连接+⚠】讲者口误说"0 到 100 米"——量程是 0~60 米（归一化 0~0.12），"100"是 bin 的个数不是米数。听视频时注意别被带偏。

---

### 🔨 动手练习 ch2-3：复现 bin_depths（UD 模式 + 越界清洗）
```python
import torch

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    else:
        raise NotImplementedError
    if target:                                   # 训练标签路径: 清洗+取整
        indices = torch.nan_to_num(indices)
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices = indices * ~mask + num_bins * mask   # 越界 → dustbin(=num_bins)
        indices = indices.type(torch.int64)
    return indices

c = 0.002                                        # 名义归一化系数(60m→0.12)
gt_m  = torch.tensor([0.0, 0.3, 5.0, 30.0, 59.9, 60.0, 80.0, float('nan')])
idx = bin_depths(gt_m * c, "UD", 0.0, 60 * c, 100, target=True)
print(idx)
# tensor([  0,   0,   8,  50,  99, 100, 100,   0])
# 0m(背景)→bin0 !  0.3m→bin0  5m→bin8  30m→bin50  59.9m→bin99
# 60m/80m→dustbin100   NaN→nan_to_num→0→bin0
# 注意: 背景0和近处0.3m都落在bin0 —— 为什么不出事? 看Part2.4的valid_depth_mask/dustbin流程
```

**【小结】** `bin_depths` 用 UD 均匀离散化把归一化 GT 映射成 bin 索引：`bin_size=(0.12−0)/100=0.0012`（≙0.6 m），`indices=(GT−min)/bin_size` 后 int64 截断取整。函数里还留着 LID/LID_FIX/SID 三种非均匀方案（LID_FIX 是团队修正过反解系数的版本）。100 个 bin 与 depth_logits 的 100 通道一一对应，`target=True` 才走"清洗+取整"的标签路径。

---

## Part 2.4 越界与 dustbin：clamp 到最后一个 bin，再把它置 0（00:10:21–00:11:31）

**本段导读**：离散化产出的索引可能是 NaN（除出来的）、负数、或超过 100（深度 > 60 m）。代码的处理链条是三段式：① `bin_depths` 内部把一切非法索引统一送进**第 100 号 dustbin**（垃圾桶 bin）；② 回到 forward，背景像素也被塞进 dustbin；③ `dustbin_zero` 开关把"等于 100"的标签**整体改写成 0**，配合最后的 `clamp(0, 99)` 保证标签落在网络的 100 个通道内。讲者这一段说得比较绕（多次自我修正），本段把口播和帧上代码逐行对齐。

---

### 句卡 2.4.1 [00:10:21 → 00:10:48]（合并 5 句：00:10:21/31/37/45/48，讲者边说边修正）**（重点句，5 角度）**
**原话**：「然后在这里的话会有一个操作，就是把我们 Depth 的 GT 等于 100——就是最远处的最远的一个深度——会把它置为 0。就是把超过我们预测的深度（量程）的，会把它置为 0。」

- 【直译】标签里等于 100 的那些像素（= 被扔进 dustbin 的越界/背景像素），最终会被改写成 0。
- 【代码】两段配合。先看 `bin_depths` 里的"进桶"（帧 00_10_11，L267-276，含团队原注释）：
  ```python
  if target:
      # Remove indicies outside of bounds
      # NOTE: since background depth is set to 0 (no lidar projection) and
      #       depth_min = 0 by default, put background prediction to dustbin by writing
      #       indices < 4 instead of indices < 0. When modify depth_min, remember
      #       to modify here accordingly.
      # NOTE: now we just put all invalid pixel to background
      indices = torch.nan_to_num(indices)
      mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
      indices = indices * ~mask + num_bins * mask
      # indices[mask] = num_bins
  ```
  再看 forward 里的"倒桶"（帧 00_11_44，L164-167）：
  ```python
  if self.dustbin_zero:
      mask_depth_target = (depth_target == int(self.num_bins)).to(depth_target.dtype)
      depth_target = depth_target * (1 - mask_depth_target)
      # depth_target[depth_target == self.num_bins] = 0
  ```
- 【形状】全程 `(21,88,160)` 逐像素；`indices * ~mask + num_bins * mask` 是无 in-place 的条件赋值写法（注释掉的 `indices[mask] = num_bins` 是等价的 in-place 版，可能为导出 ONNX/避免版本问题而改写；`(1 - mask_depth_target)` 乘法同理）。
- 【为什么】dustbin（垃圾桶 bin）是"额外的第 101 类"：所有"没法要"的像素（>60 m、负值、NaN、无 lidar 投影的背景）先统一挂到 100 号，**把"非法"显式化**，后面才好按策略批量处理，而不是散落在各处特判。代码注释还留了一个历史方案：曾经想用 `indices < 4`（约 2.4 m 内也当背景）把近处噪声也扔进桶，现已简化为"所有 invalid → background"。
- 【连接】"dustbin"一词沿自 SuperGlue/分类任务里的 garbage class 传统。CenterPoint 里没有这一层，因为它的 heatmap 用高斯软标签天然容纳"背景=0"；分类式深度必须自己造一个垃圾类。

---

### 句卡 2.4.2 [00:10:51 + 00:10:54]
**原话**：「就是最后在真正做 LSS 投影的时候，又会把这一个最远的、预测为 0 的深度会把它给去掉。」

- 【直译】为什么敢把 dustbin 改写成 0？因为后面 LSS 投影阶段会把（对应这一类的）深度分量丢掉，它不参与建 BEV 特征。
- 【代码】伏笔——具体实现在后续 LSS 投影章节。可以预期的形式是：投影时对 100 个深度平面做外积前，跳过/置零第 0 个 bin 的概率分量，或投影网格从 bin 1 开始铺。⚠ 本章看不到那段代码，此处只按讲者口径记录，待 LSS 章节验证"去掉的到底是 bin 0 还是别的"。
- 【为什么】设计闭环是：背景/超界 → dustbin(100) → 置 0 → 投影时丢弃 bin 0。这样"无效深度"在监督端被当作一个可学习的类别（网络学会对背景输出 bin 0），在几何端又不会把背景特征泼进 BEV 空间——监督有着落、投影不受污染，一箭双雕。
- 【连接】BEVDepth/LSS 原版没有这层"0 号 bin=弃用"约定，它们的深度分布 softmax 后全量参与外积；DenseBEV 用 bin 0 兼职"无效类"，是工程上的自定义约定，读投影代码时务必带着这个前提。

---

### 句卡 2.4.3 [00:11:01 → 00:11:27]（合并 6 句：00:11:01/09/13/15/19/23/27，讲者复述前两卡内容）
**原话**：「相当于就是我们刚刚在处理 Depth 的时候会做一个 clamp 的一个操作（转写作 collapse）：把超过设定的最大深度的，会把它分到最后一个 bin 里面去；然后再把最后一个 bin 所对应的深度的 GT，会把它又映射成 0。」

*（此卡合并了讲者对 2.4.1/2.4.2 的车轱辘复述，只补充新信息）*

- 【直译】总结两步走：超界 → 最后一个 bin（100 号）；100 号 → 0。
- 【代码】新信息是帧 00_11_44 L178-179 还有真正的 `clamp` 收尾：
  ```python
  if not self.dustbin_additional:
      depth_target = torch.clamp(depth_target, 0, self.num_bins - 1).to(torch.int64)
  ```
  `dustbin_additional=False`（未给网络额外加第 101 个输出通道）时，把标签硬夹到 [0, 99]——这是最后一道保险：即便 `dustbin_zero` 没开、标签里还残留 100，也会被夹成 99 而不是让 focal loss 拿 100 去索引 100 通道的 logits 导致越界。
- 【为什么】三个开关组合出一张策略矩阵：`dustbin_additional=True` → 网络出 101 通道，dustbin 作为真类别参与分类；`dustbin_zero=True` → dustbin 归并进 bin 0；都不开 → clamp 到 99（超远深度并进最远 bin）。讲者口中的当前配置是 `dustbin_zero=True` 路线。另外注意 L169-172 还有一个 `if not self.sup_dust:` 块（把 `depth_target == num_bins` 处的 `depth_masks` 置 0，即"垃圾桶像素不监督"）——⚠ **代码顺序细节**：它排在 `dustbin_zero` **之后**，若 `dustbin_zero=True`，执行到这里时标签里已经没有 100 了，`sup_dust` 块实际是空转（no-op）。这两个开关事实上互斥，属于"读代码才能发现、听讲发现不了"的坑。
- 【连接】与 CenterPoint 解码时 `post_center_range` 夹合法框、BEVFusion voxel 化时丢弃 range 外点是同族操作：**离散化系统的边界必须显式定义归属**，否则要么越界崩溃、要么静默错标。

---

### 🔨 动手练习 ch2-4：dustbin 三开关与顺序陷阱
```python
import torch
num_bins = 100
depth_target = torch.tensor([0, 8, 50, 99, 100, 100])   # 含2个dustbin像素
depth_masks  = torch.ones(6)

dustbin_zero, sup_dust, dustbin_additional = True, False, False

if dustbin_zero:                                  # L164: dustbin → 0
    m = (depth_target == num_bins).to(depth_target.dtype)
    depth_target = depth_target * (1 - m)
print(depth_target)      # tensor([ 0,  8, 50, 99,  0,  0])

if not sup_dust:                                  # L169: 本想"垃圾桶不监督"
    m2 = (depth_target == num_bins).to(depth_target.dtype)
    depth_masks = depth_masks * (1 - m2)
print(depth_masks)       # tensor([1., 1., 1., 1., 1., 1.]) ← 全1! 100早已被置0, 此块空转
                         # 顺序陷阱: dustbin_zero 在前, sup_dust 永远失效

if not dustbin_additional:                        # L178: 最后保险
    depth_target = torch.clamp(depth_target, 0, num_bins - 1).to(torch.int64)
print(depth_target.max())  # tensor(99) —— 标签保证 ∈ [0,99], 与100通道logits匹配
```

**【小结】** 越界处理链 = 进桶（NaN/负/超界 → dustbin 100）→ 倒桶（`dustbin_zero`: 100 → 0）→ 保险（`clamp(0, 99)`）；背景像素经 `valid_depth_mask` 也走同一条桶链。bin 0 由此兼任"无效/背景类"，LSS 投影时会把它丢弃。注意 `sup_dust` 块排在 `dustbin_zero` 之后而形同虚设——工作代码里开关堆叠的顺序 bug/冗余，靠听讲发现不了，靠逐行读能发现。

---

## Part 2.5 数值保护、Focal Loss 与针孔/鱼眼双分支收尾（00:11:35–00:12:58）

**本段导读**：标签侧全部就绪，回到预测侧做最后两件事：把 logits 里的 NaN/Inf 冲干净（数值保护），然后 `loss_func`（Focal Loss）逐像素算损失、乘 mask、按有效像素数归一化出标量。镜头最后拉回 `fpn_forward.py`：这一整套对 `_fv_grps` 里的**针孔组和鱼眼组各跑一遍**，两个 loss 求和、两组 depth logits 一并传给下游 LSS 投影。本章闭环。

---

### 句卡 2.5.1 [00:11:35 → 00:11:54]（合并 4 句：00:11:35/46/50/54）
**原话**：「然后这里是做了一个数值的一个保护，就是把我们预测的 Depth 做的——NaN 时刻还有无穷大这些——做了一个数值的一个保护。」

- 【直译】算 loss 前，把预测 logits 里可能出现的 NaN 和 ±Inf 全部替换成 0。
- 【代码】帧 00_11_44，L174-176：
  ```python
  _l = depth_logits
  _l = torch.where(torch.isnan(_l), torch.full_like(_l, 0), _l)
  _l = torch.where(torch.isinf(_l), torch.full_like(_l, 0), _l)
  ```
  （变量名帧上显示为 `_1`/`_l` 难以分辨，按 Python 习惯推断是 `_l`。）注意保护的是**预测**；GT 侧的 NaN 早在 `bin_depths` 里被 `torch.nan_to_num` 处理过了——两侧各有一道闸。
- 【形状】`(21,100,88,160)` 原位语义替换，shape 不变。
- 【为什么】混合精度/大模型长训中，上游一次数值溢出（fp16 上 6.5e4 就 Inf）就会让 loss 变 NaN，而 NaN 会沿反传污染**全部**参数，一夜训练作废。与其祈祷不出，不如在 loss 入口设卡。0 是安全值：softmax 后变成均匀概率的一票，不主导梯度。另一个具体来源是本模型自己的 `fisheye_logits_mask`（帧 00_06_53，L611-614：鱼眼无效成像区把 logits 置 0 的 `torch.where`）上游各种 mask/除法都可能造出坏值。
- 【连接】你在 4060 上训 BEVFusion 没开 AMP，可能没吃过 NaN 的亏；但转岗后跑量产大模型（多机+AMP）时，这种"入口数值保护"几乎是标配。另一常见写法是一行 `torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`，此处拆成两个 `torch.where` 可能是为了部署算子兼容。

---

### 句卡 2.5.2 [00:11:56 + 00:12:05 + 00:12:11]（合并 3 句）**（重点句，5 角度）**
**原话**：「然后在这里就用我们预测的 Depth 和对应生成的一个 GT 去算 Loss，用了一个 Focal Loss。这个是我们当前预测 Depth 的一个方法。」

- 【直译】清洗后的 logits + 刚造好的 bin 标签，送进 focal loss，得到逐像素损失，再 mask、归一化成标量。
- 【代码】帧 00_11_44 / 00_12_02，L181-192 完整收尾：
  ```python
  loss = self.loss_func(_l, depth_target)          # focal loss, 逐像素 (21,88,160)

  if fg_masks is not None:                         # 本次调用 fg_masks=None, 不走
      fg_masks = fg_masks.bool()
      bg_masks = ~fg_masks
      fg_bg_weights = self.fg_weight * fg_masks + self.bg_weight * bg_masks
      loss *= fg_bg_weights.to(loss.get_device())

  loss *= depth_masks                              # 只留lidar有效像素
  num_pixels = depth_masks.sum() + 1e-6
  loss = loss.sum() / num_pixels                   # 按有效像素平均 → 标量
  return loss
  ```
- 【形状】`loss_func((21,100,88,160), (21,88,160)) → (21,88,160)`（reduction='none' 的逐像素 focal），乘 `(21,88,160)` mask，sum/除 → `()` 标量。
- 【为什么】选 focal loss 而非普通 cross entropy 的原因是**类别失衡**：100 个 bin 里，一张图的深度大量集中在少数 bin（路面连续渐变+车辆集中在 10~40 m），且"预测对的容易样本"占绝大多数。focal 的 `(1−p_t)^γ` 因子把已学会的像素梯度压低，把火力集中到难例（物体边缘、远处小目标）。为什么不用回归 L1？分类分布是 LSS 外积的刚需（要 100 个概率），且分类对稀疏噪声 GT 更鲁棒。
- 【连接】① CaDDN 原文对 depth bin 用的就是 focal loss，DenseBEV 照单全收；② 和 CenterPoint heatmap 的 Gaussian Focal Loss 同源但不同款：那边是二值热图变体（正负样本按高斯软权重），这边是标准多类 focal；③ 预留的 `fg_masks` 前景加权（本次传 None）思路同 BEVDepth 的讨论——lidar 打在路面上的点远多于打在物体上的点，若只想让深度网络服务检测，可以给前景像素更大权重。DenseBEV 留了接口没启用。

---

### 句卡 2.5.3 [00:12:20 + 00:12:24]（合并 2 句）
**原话**：「然后在这里分别遍历完针孔和鱼眼，然后会求一个 Depth 的一个 Loss。」

- 【直译】镜头拉回 `fpn_forward.py`：以上整套（DepthNet 前向 + DDNLoss）是在一个 for 循环里的，循环变量是"相机组"——针孔一组、鱼眼一组，各算各的 loss。
- 【代码】帧 00_06_53/00_12_28，`DepthNet.forward` 骨架：
  ```python
  for idx, grp in enumerate(self._fv_grps):        # grp ∈ {针孔组前缀, 'fisheye_'}
      ...
      depthnet = getattr(self, f'{grp}depthnet')   # 每组各自独立的DepthNet权重!
      ...
      depth_loss = ddn_loss(depth_logits, depths, depth_masks, fg_masks)
      depth_loss *= self.ddn_weight
      depth_losses.append(depth_loss)
      depth_probses.append(depth_logits)

  depth_loss_total = sum(depth_losses)             # L641
  return depth_loss_total, *depth_probses          # L643
  ```
- 【形状】针孔组 logits `(21,100,88,160)`（3 帧×7 相机）；鱼眼组则是 3 帧×4 鱼眼=12 路，分辨率与 bin 配置可不同（鱼眼组还会走 `downscale_img`：`F.interpolate(depths, scale_factor=0.5, mode='bilinear')` 先把 GT 和 mask 减半再对齐，见帧 00_12_28 L624-626；⚠ 对 0/1 mask 用 bilinear 插值会产生 0~1 之间的软值，等效于给边缘像素降权，是有意为之还是将就，可核实）。
- 【为什么】针孔和鱼眼**不共享 DepthNet**（`getattr(self, f'{grp}depthnet')` 按前缀取不同子模块）：鱼眼的成像模型（等距投影、大畸变）和外观-深度关系与针孔完全不同，焦距归一化也救不了畸变差异，分开建模是必然选择。循环+前缀 getattr 的写法让两组共享同一套 loss/离散化代码，只差参数。
- 【连接】这是全视频"针孔/鱼眼双轨制"的第一次完整亮相，后面 LSS 投影、多视角融合章节会反复出现同款 `for grp in self._fv_grps` 循环——记住这个模式，后面章节读起来会非常顺。另外帧 00_06_53 顶部还有鱼眼专属的 `fisheye_logits_mask` 处理（无效成像区 logits 置 0），是鱼眼圆形视场外黑边的特判。

---

### 句卡 2.5.4 [00:12:31 → 00:12:41]（合并 3 句）
**原话**：「然后会传出去我们针孔所预测的一个 Depth，以及鱼眼所预测的一个 Depth，然后会传到我们后续做 LSS 投影的时候会用到。」

- 【直译】本模块的两个产出：一个总 depth loss（进训练总损失），两组 depth logits（进 LSS 投影做几何）。
- 【代码】`return depth_loss_total, *depth_probses`——星号解包把列表元素平铺进返回元组，调用方按位置接收针孔 logits、鱼眼 logits。debug 台旁证（帧 00_06_53 底部）：
  ```
  >>> [i.shape for i in depth_input]
  [torch.Size([21, 128, 88, 160]), torch.Size([21, 256, 44, 80]), torch.Size([21, 256, 22, 40])]
  >>> depth_logits.shape
  torch.Size([21, 100, 88, 160])
  ```
  （多尺度特征只用了 1/8 那层，与 Ch1 呼应；⚠ L635 旁注释 `# 14,256,88,160` 与实测 21 不符，应是旧配置——2 帧×7 相机时代——的陈旧注释，读代码别被它骗。）
- 【为什么】depth logits 传出去而不是 softmax 后的概率传出去，把归一化时机留给投影模块自己定（后续章节 LSS 那边会做 softmax/丢 bin 0 等后处理）；loss 与特征双输出，训练图和推理图共用一条 forward，是量产代码常见形态（部署时 depths=None walk `depth_loss = 0.0` 分支，见帧 00_12_28 L620-621）。
- 【连接】到这里，Ch1 的"FPN 出 1/8 特征"→ 本章"100 bin 分布 + 监督"→ 下下章"LSS 外积投影"这条相机支线就串起来了：**depth_probs (21,100,88,160) ⊗ image_feat (21,C,88,160) → 视锥特征 → BEV**，正是 LSS 论文的 lift 步骤。

---

### 句卡 2.5.5 [00:12:52 + 00:12:58]（合并 2 句过渡语）
**原话**：「这个是 DepthNet。我再看一下。」

- 【直译】DepthNet + Depth Loss 整体讲完，讲者翻页找下一个模块（下一章 00:13:04 起：Lidar Backbone，DataLoader 直接给出已拍平到 BEV 的 6×352×20×64 特征，透传为主）。
- 【为什么】在全局地图上，本章结束意味着**相机支线的"深度"半边**完结；相机支线剩下的"几何投影"半边要等 LSS 章。中间插播 lidar/radar 支线。
- 【连接】听课策略：Ch3 lidar 是透传、较轻，Ch4 radar pillar 编码较重——可以把本章的离散化三件套（减 min/除步长/取整+越界）带过去看 radar pillar 的坐标量化，模式完全复用。

---

### 🔨 动手练习 ch2-5：mini-DDNLoss 全流程（NaN 保护 + Focal + mask 归一化）
```python
import torch
import torch.nn.functional as F

def focal_loss(logits, target, gamma=2.0):
    """logits:(N,D,H,W)  target:(N,H,W)∈[0,D-1]  → 逐像素loss (N,H,W)"""
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    return -((1 - p_t) ** gamma) * logp_t

torch.manual_seed(0)
N, D, H, W = 21, 100, 88, 160
depth_logits = torch.randn(N, D, H, W)
depth_logits[0, :, 0, 0] = float('nan')          # 模拟上游坏值
depth_target = torch.randint(0, D, (N, H, W))
depth_masks  = (torch.rand(N, H, W) > 0.95).float()

# 1) 数值保护 (L174-176)
_l = depth_logits
_l = torch.where(torch.isnan(_l), torch.full_like(_l, 0), _l)
_l = torch.where(torch.isinf(_l), torch.full_like(_l, 0), _l)

# 2) focal loss + mask + 有效像素归一化 (L181, L189-192)
loss = focal_loss(_l, depth_target)
loss = loss * depth_masks
num_pixels = depth_masks.sum() + 1e-6
loss = loss.sum() / num_pixels
print(loss)          # ≈4.6上下的有限标量(随机logits下每像素≈-log(1/100)≈4.6, focal因子略压低)
assert torch.isfinite(loss)   # 若注释掉步骤1, 这里会因NaN传播而失败
```

**【小结】** 预测侧两道 `torch.where` 冲掉 NaN/Inf 后，focal loss 把 100-bin 分类的难例梯度放大、易例压低；逐像素 loss 乘 lidar 掩码、按有效像素数（+1e-6 防零）归一化成标量。整套流程在 `for grp in self._fv_grps` 里对针孔组、鱼眼组各跑一遍（DepthNet 权重不共享、鱼眼有额外的 logits mask 与 0.5 倍下采样对齐），loss 求和、两组 logits 传给 LSS 投影。

---

## 全章总小结

1. **深度是 100 类分类，不是回归**：DepthNet 输出 `(21,100,88,160)` logits；GT 由 lidar 投影而来，经 `bin_depths`（UD 均匀离散化，`bin_size=(0.12−0)/100=0.0012`≙0.6 m）变成 int64 bin 索引，focal loss 监督——CaDDN/BEVDepth 路线在量产代码中的落地。
2. **一切数值都在"焦距归一化域"**：Depth Loader 已把米制 GT 按各相机焦距（讲者称平方 ⚠）归一化，量程边界同乘 0.002（60 m→0.12），保证 bin 索引不变、多相机共享网络成立；改任何一边必须同步改另一边。
3. **无效值有一条完整的"垃圾桶流水线"**：NaN/负/超 60 m/无投影背景 → dustbin bin 100 → `dustbin_zero` 改写为 0 → `clamp(0,99)` 保险 → LSS 投影阶段丢弃 bin 0；预测侧另有 NaN/Inf 双 `torch.where` 保护。针孔、鱼眼两组独立 DepthNet、共享同一套 loss 代码，loss 相加、logits 双双传给下游。

## 本章存疑清单（⚠）

| # | 位置 | 疑点 | 我的推断 |
|---|------|------|----------|
| 1 | [00:08:04] 句卡 2.2.3 | GT 归一化是"除焦距的**平方**"还是一次方 | 数值旁证（0.002=1/500）更贴近 d/f 一次方（BEVDepth 虚拟深度惯例）；但讲者两次说"平方"。**去 Depth Loader 源码核实** |
| 2 | [00:08:59] 句卡 2.3.3 | 系数 0.002 的出处（配置项名、是否分相机） | 名义/全局系数，写在配置或 Loader；各相机实际系数按焦距缩放 |
| 3 | [00:10:54] 句卡 2.4.2 | "LSS 投影时把置 0 的深度去掉"具体指丢弃 bin 0 的概率分量还是网格错位一格 | 待 LSS 投影章节（后续章）对代码验证 |
| 4 | 帧 00_11_44 L169 句卡 2.4.3 | `sup_dust` 块排在 `dustbin_zero` 之后疑似永久空转 | 代码顺序问题或历史遗留，两开关实际互斥；不影响当前配置的结果 |
| 5 | 帧 00_12_28 L635 句卡 2.5.4 | 注释 `# 14,256,88,160` 与实测 `21×128×88×160` 不符 | 陈旧注释（疑为 2 帧×7 相机旧配置），以 debug 台实测为准 |
| 6 | 帧 00_12_28 L625-626 句卡 2.5.3 | 对 0/1 depth_masks 用 bilinear 插值下采样会产生软值 | 等效边缘降权，多半是将就写法；若要严格 0/1 应用 nearest，可核实是否有意 |
| 7 | 句卡 2.1.3 | 1/8 分辨率深度 GT 在 loader 里的池化方式（min/均值） | BEVDepth 惯例是 patch 内取 min；校正稿头部也标注了此低置信项，需对代码核实 |
| 8 | [00:10:12] 句卡 2.3.7 | 讲者口误"0 到 100 米" | 应为 0~60 米量程 / 100 个 bin，听课勿被带偏 |


---
> [[Ch01_DepthNet网络结构|← Ch1]] · [[00_总览与脉络|📖 总览]] · [[Ch03_Lidar透传与Radar编码图解|Ch3 →]]

