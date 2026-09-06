> [[Ch06_LSS投影输入准备|← Ch6]] · [[00_总览与脉络|📖 总览]] · [[Ch08_多视角与RC与模态融合|Ch8 →]]

# Ch7 LSS 投影本体（00:40:59–00:51:44）

> **你在地图的哪一站**：
> `FPN收尾 → DepthNet → Depth Loss → Lidar Backbone → Radar Backbone → RL融合(UNet) → 【★你在这里：LSS投影(外积+拍平+grid_sample)】 → 多视角融合(针孔+鱼眼) → RC融合 → 模态融合 → MemoryManager → 时序融合 → BEV UNet → CenterPoint头 → Loss → Box解码`
>
> 前面几章里，图像走完 backbone+FPN 拿到了下采样 8 倍的 128 通道特征，DepthNet 对每个像素预测了 100 个深度 bin 的概率分布。**本章是整条图像链路里"从图像视角(PV)跳进 BEV 视角"的那一跳**——DenseBEV 版本的 Lift-Splat：外积升维(lift) → 沿图像高度拍平(splat 的替身) → grid_sample 重采样到自车坐标系。这一跳做完，图像特征和 lidar/radar 特征才第一次站在同一张 448×224(投影时用半分辨率 224×112)的 BEV 网格上,后续的融合、时序、检测头才有共同语言。
>
> **本章涉及的两个文件**（从抽帧面包屑逐字读出）：
> - `e2e/tasks/bev_task/uvp_module/models/fv2bev/bev_projector.py` —— `class BEVProjector(nn.Module)`（外壳：拆帧拆相机、合并成 21、调 single_view_proj、拆回来）；文件后部还有 `class ModuleBevProject(BaseModule)`（更外层的壳，管针孔/鱼眼两轮循环和最终 return）
> - `e2e/tasks/bev_task/uvp_module/models/fv2bev/liftsplat_proj.py` —— `class LiftSplatProjection(nn.Module)`（本体：降通道、外积、拍平、grid_sample）
>
> **本章调试台实测形状**（从 00_44_45 / 00_45_20 / 00_46_45 抽帧的 debug console 逐字抄录，比转写里的口糊数字可靠）：
> ```
> fv_feat.shape            -> torch.Size([3, 7, 128, 88, 160])   # 进 projector 前
> depths_logit.shape       -> torch.Size([3, 7, 100, 88, 160])
> fv_featmap_list_cat.shape-> torch.Size([21, 128, 88, 160])     # 合并 21 之后
> grid_map_list_cat.shape  -> torch.Size([21, 224, 112, 2])
> image_features.shape     -> torch.Size([21, 32, 88, 160])      # 降通道之后
> (外积单份)                -> torch.Size([1, 32, 100, 88, 160])
> (depth unsqueeze后)       -> torch.Size([21, 1, 100, 88, 160])
> bev_feat(hover弹窗)       -> torch.Size([21, 32, 224, 112])，grad_fn=<CudnnGridSamplerBackward>
> ```

---

## Part 1 入口与形状整备：BEVProjector.forward（00:40:59–00:41:32）

**本段在讲什么**：讲者从 solver 单步跳进 `bev_projector.py` 的 `BEVProjector.forward(self, fv_feat, grid_map, depth_probs=None, flip_lr=None, pinhole=True)`。输入是三样东西：FPN 出来的图像特征 `fv_feat`、DepthNet 出来的深度分布 `depth_probs`、离线由标定算好的采样网格 `grid_map`。本段先把这三个张量的形状盘清楚（关键是 batch 维里已经藏了"1 个 batch × 3 帧 = 3"），然后指认代码里两条分支：`remote_proj` 分支这次不走，`optimize_op_num` 分支才是真正执行路径。输出（本段末尾时）还没有任何计算发生——纯粹是"站在门口看清行李"。

---

### 句卡 1-1 ⭐重点句

> **[00:40:59]** 「3×1×7×128×88×160」（承接上一句 [00:40:54]「再变一下就变成了 3×7×1…」）

- 【直译】进投影模块前，图像特征被 reshape 成了一个六维张量：3 × 1 × 7 × 128 × 88 × 160。讲者上一句说到一半改口，其实是在说"先是 (3,7,…) 的五维，再 unsqueeze 一下变成 (3,1,7,…) 的六维"。
- 【形状】逐维拆开读：**3 = batch(1) × 时序帧(3)**——三帧历史帧早在进本模块之前就被折叠进了 batch 维；**1 = 模块内部的 frame_num 占位维**（下一句他自己说"这个 1 可以现在不用管"）；**7 = 七路针孔相机**；**128 = FPN 输出通道**；**88×160 = 下采样 8 倍后的图像特征图高×宽**（原图应约 704×1280，704/8=88，1280/8=160）。抽帧 00_41_28 调试台印证：`fv_feat.shape -> torch.Size([3, 7, 128, 88, 160])` 是 reshape 前的五维形状，加了那个"1"之后才是讲者口中的 (3,1,7,128,88,160)。
- 【代码】等价写法：`fv_feat = fv_feat.reshape(B*T, 7, 128, 88, 160).unsqueeze(1)  # -> (B*T, 1, 7, 128, 88, 160)`。之所以要这个假的 dim1，是因为 `forward` 里第一层循环是 `for i in range(self.frame_num):`（抽帧 00_41_28 第 75 行原文），索引方式是 `fv_feat[:, i, j, ...]`——接口按 (B, frame, view, C, H, W) 设计；本工程把时序帧塞进 batch 维后，就让 frame_num=1 来兼容这个接口。
- 【为什么】把"3 帧"放进 batch 维而不是留在 frame 维，好处是**三帧共享完全相同的投影计算图**——每帧都要独立做一次 LSS（投到各自采集时刻的自车系，时序对齐留给后面 MemoryManager/warp 章节），合并后一次算完，不用写三遍循环，也方便后面进一步和 7 路相机一起合并成 21。
- 【连接】BEVFusion(mmdet3d 版)里对应物是 `LSSTransform.forward` 输入 (B, N_cam, C, fH, fW)——它没有时序帧，所以只有相机数 N 这一个"多视角"维；DenseBEV 等于把 BEVFusion 的 N=6 换成了 T×N=3×7=21。你在 4060 上跑 BEVFusion 时 `img_feats` 的 (1,6,256,32,88) 和这里的 (3,7,128,88,160) 是同一个角色。

### 句卡 1-2

> **[00:41:04]** 「然后预测的 depth 的话会变成 3×7×7×100×88×160」

- 【直译】DepthNet 的输出（每像素 100 个深度 bin 的概率）也做同样的 reshape，和图像特征保持相同的"batch×帧 / frame / view"前缀维。
- 【⚠校正】"3×7×7×100…"是转写口糊：**应为 3×1×7×100×88×160**。证据有二：① 调试台 `depths_logit.shape -> torch.Size([3, 7, 100, 88, 160])`（五维原形，unsqueeze 后即 (3,1,7,100,88,160)）；② depth 必须与 fv_feat 的前三维完全对齐才能在后面用同一套 `[:, i, j, ...]` 索引取出。中间那个"7"应是讲者读 (3,**1**,7,…) 时的口误或转写错误。
- 【形状】(3, 1, 7, **100**, 88, 160)：与图像特征唯一的区别是通道维——图像是 128 个语义通道，depth 是 100 个**深度 bin 的 softmax 概率**（Ch6 讲过：bin0 是"垃圾桶"，预测取 1~99 的 bin；进到这里的 100 维是处理后的概率分布）。
- 【连接】这正是 LSS 论文里的 α（depth distribution）张量；BEVFusion 中对应 `depth = depth_net(x).softmax(dim=1)` 的 (B*N, D, fH, fW)。LSS 原文 D=41（4~45m），这里 D=100，配 0.4m 级 BEV，看得更远更细。

### 句卡 1-3（合并 [00:41:12][00:41:15]，两句为同一意思的车轱辘重复）

> **[00:41:12]** 「这个是 1，可以现在不用管」 **[00:41:15]** 「所以不会那么也是 1」（口糊，意为"depth 那边同样也是 1"）

- 【直译】六维形状里的 dim1 那个"1"是个占位符，图像特征和 depth 两个张量里它都是 1，先忽略。
- 【为什么】这个 1 = `self.frame_num`（模块视角的帧数）。工程演化痕迹：接口留着 frame 维说明这个模块曾经/也可以按"batch 不折帧"的方式用；当前配置下时序帧走 batch 维，于是 frame_num 退化为 1。**读工程代码的重要技能：区分"活着的维度"和"兼容性占位维度"**，后者的典型特征就是讲者这句"可以不用管"。
- 【代码】它的存在只影响一层空转循环：`for i in range(self.frame_num)` 只执行 i=0 一次，`fv_feat[:, 0, j, ...]` 把这个 1 消掉。
- 【连接】你在 BEVFusion 代码里也见过这类占位维：`points` list 外面套的 batch list、`img.unsqueeze(0)`——判断方法一样，顺着索引看它是否恒为 0/1。

### 句卡 1-4（合并 [00:41:24][00:41:27]）

> **[00:41:24]** 「然后这里没有走」 **[00:41:27]** 「然后走这里」

- 【直译】forward 里有两条大分支：上面那条（高亮成蓝色被选中讲解的）这次不执行；执行的是下面 `if self.optimize_op_num:` 这条（IDE 里黄色当前行，抽帧 00_41_28 第 90 行）。
- 【代码】没走的分支原文（抽帧 00_41_28 第 76~88 行逐字）：
  ```python
  if pinhole and self.remote_proj:
      prj_args = [fv_feat[:, i, 1, ...], grid_map[:, -1]]
      if self.prj_with_depth:
          prj_args.insert(1, depth_probs[:, i, 1, ...])
      bev_feat, bev_mask = self.single_view_proj(*prj_args)
      if flip_lr is not None:
          flip_lr = flip_lr.to(bev_feat.device)
          bev_feat = flip_lr_helper(flip_lr, bev_feat)
          bev_mask = flip_lr_helper(flip_lr, bev_mask)
      bev_feat_list.append(bev_feat.unsqueeze(dim=1))
      bev_mask_list.append(bev_mask.unsqueeze(dim=1))
  ```
- 【为什么】⚠ 讲者没解释 `remote_proj` 是什么。从代码硬索引 `fv_feat[:, i, 1, ...]`（固定取 **view=1** 这一路）和 `grid_map[:, -1]`（取**最后一张**额外的 grid）推断：这是给某一路相机（很可能是前向长焦/远距相机）做**额外一次远距投影**的开关——同一路图像特征用另一张覆盖更远范围的 grid 再投一遍。本次配置未开启，听到后面"前向一号相机"可视化时可再对照。存疑，建议向导师确认。
- 【连接】顺带注意 `flip_lr_helper`：训练时如果做了 BEV 左右翻转增广，投影出的 BEV 特征也要同步翻转才能对上 GT——这和 BEVFusion 的 `GlobalRotScaleTrans`/`RandomFlip3D` 需要同步作用到 img2lidar 矩阵是同一个道理，只是这里选择翻转"结果"而不是翻转"矩阵"。

**Part 1 小结**：① 进入投影模块时，batch(1)×3 帧已折叠为 dim0=3，七路相机在 dim2=7，图像特征 128 通道、depth 100 bin，空间都是 88×160。② dim1 的 1 是 frame_num 兼容占位维，索引一次即消。③ 两条分支里 remote_proj（疑似长焦远距二次投影）没走，走的是 optimize_op_num 的合并路径——它是下一 Part 的主角。

### 🔨 动手练习 ch7-1：占位维与"帧折进 batch"

```python
import torch

B, T, V, C, H, W = 1, 3, 7, 128, 88, 160
# DepthNet 阶段: 帧和相机全在 dim0 (21 路)
feat_flat = torch.randn(B * T * V, C, H, W)          # (21,128,88,160)
# 进 projector 前: 拆出相机维,再补 frame_num=1 占位维
fv_feat = feat_flat.reshape(B * T, V, C, H, W)        # (3,7,128,88,160)  <- 调试台看到的
fv_feat = fv_feat.unsqueeze(1)                        # (3,1,7,128,88,160) <- 讲者口中的
print(fv_feat.shape)   # torch.Size([3, 1, 7, 128, 88, 160])

frame_num = fv_feat.shape[1]
for i in range(frame_num):          # 只会跑 i=0
    one = fv_feat[:, i, 2, ...]     # 取第 3 路相机
    print(i, one.shape)             # 0 torch.Size([3, 128, 88, 160])
# 预期输出:
# torch.Size([3, 1, 7, 128, 88, 160])
# 0 torch.Size([3, 128, 88, 160])
```

---

## Part 2 合并成 21：batch×3 帧×7 路进一个维度（00:41:34–00:44:06）

**本段在讲什么**：`optimize_op_num` 分支的全部内容——把 7 路相机各自的 (3,C,H,W) 特征/深度/网格循环取出、`torch.cat` 到 dim0，得到 dim0=21（=1 batch×3 帧×7 路）的三个大张量，再一次性喂给 `self.single_view_proj`（即 LiftSplatProjection）。输入是 (3,1,7,…) 的六维张量，输出是 (21,128,88,160)、(21,100,88,160)、(21,224,112,2) 三个"扁平"张量。本段还交代了 grid 的形状，以及为什么投影只投到 224×112 的半分辨率 BEV。

---

### 句卡 2-1（合并 [00:41:34]~[00:41:43] 五个碎句：「然后在这里会去/会把我们/图像的特征/以及预测的 depth/还有 grid map」）

> **[00:41:34–00:41:43]** 「在这里会把我们图像的特征、以及预测的 depth、还有 grid map（三样一起处理）」

- 【直译】本分支要同时摆弄三个输入：图像特征 fv_feat、深度分布 depth_probs、采样网格 grid_map——后面的合并操作对三者做的是完全同构的事。
- 【代码】对应抽帧 00_42_09 第 91~98 行逐字：
  ```python
  if self.optimize_op_num:
      fv_featmap_list = []
      depth_probs_list = []
      grid_map_list = []
      for j in range(num_views):
          fv_featmap_list.append(fv_feat[:, i, j, ...])
          grid_map_list.append(grid_map[:, j])
          if self.prj_with_depth:
              depth_probs_list.append(depth_probs[:, i, j, ...])
  ```
- 【为什么】grid_map 为什么也要跟着走？因为每一路相机的内外参不同，**每路相机有自己专属的一张采样网格**；相机 j 的特征必须配相机 j 的 grid，所以三个 list 必须以相同的 j 顺序 append，合并后 dim0 上第 k 份特征与第 k 份 grid 才对得上号。
- 【连接】grid_map 是离线用标定（内参+外参+BEV 网格定义）预先算好的，训练时当普通输入喂进来——BEVFusion 的 `lidar2image`/`geom` 也是由标定即时算出，思想一致；预计算换取在线速度，是量产代码和学术代码的典型分野。

### 句卡 2-2 ⭐重点句（合并 [00:41:44]~[00:42:01] 六个碎句）

> **[00:41:44–00:42:01]** 「它的 shape 是 3，是 batch size 乘以 3……我当前设置的 batch size 是 1，所以说这个 D0 维它就是 1×3，是 batch size 乘上 3 帧」

- 【直译】每次从大张量里切出来的单路相机张量，第 0 维是 3——不是"3 帧"这个独立维度，而是 batch_size(1) × 3 帧 = 3。
- 【形状】`fv_feat[:, 0, j, ...]` → (3, 128, 88, 160)；`depth_probs[:, 0, j, ...]` → (3, 100, 88, 160)；`grid_map[:, j]` → (3, 224, 112, 2)（grid 对三帧是广播同一份还是各存一份，画面分辨不清；从 cat 后 21×224×112×2 反推，是每帧各一份、内容相同——同一路相机三帧的标定不变）。
- 【代码】`x = fv_feat[:, i, j, ...]  # (B*T, C, H, W) = (3,128,88,160)`。
- 【为什么】讲者反复强调"batch size 我设的是 1"，是提醒你**调试时看到的 3、21 都是 B=1 的特例**；训练如果 B=4，这里就是 12 和 84。读别人调试录屏时要养成把常数还原成符号的习惯：3=B·T，21=B·T·V。
- 【连接】你的 BEVFusion mini 训练 B=1 时同理：`voxels` 第一维数字直接读没意义，要除回 batch 才知道每样本量级。

### 句卡 2-3 ⭐重点句（合并 [00:42:04]~[00:42:30] 八个碎句）

> **[00:42:04–00:42:30]** 「这一步主要是为了在后面我们做 LSS 投影的时候，能够减小我们的一个显存量。就是如果说 Tensor 太大的话，显存占用会比较多。所以在这里会把每一路相机——每路相机以及它的 batch size 和 3 帧——给它放到一个维度。」

- 【直译】把 B、3 帧、7 路相机统统压进第 0 维（得到 21），是为后面 LSS 的省显存策略做铺垫：第 0 维成为唯一的"份数"维，后面就能沿它切开逐份计算。
- 【代码】合并三件套（抽帧 00_42_09 第 99~103 行逐字）：
  ```python
  fv_featmap_list_cat = torch.cat(fv_featmap_list, 0)   # (21,128,88,160)
  grid_map_list_cat  = torch.cat(grid_map_list, 0)      # (21,224,112,2)
  prj_args = [fv_featmap_list_cat, grid_map_list_cat]
  if self.prj_with_depth:
      prj_args.insert(1, torch.cat(depth_probs_list, 0))  # (21,100,88,160)
  ```
- 【形状】cat 沿 dim0 把 7 份 (3,…) 拼成 (21,…)。**注意拼接顺序是"相机为外、帧为内"**：dim0 = [相机0 的 3 帧, 相机1 的 3 帧, …, 相机6 的 3 帧]。这个顺序到 Part 5 拆回去时会被再次提到（"21 = 7×(B·3)"）。
- 【为什么】这里其实叠了**两个动机**，讲者说的是其一：① `optimize_op_num`（属性名直译"优化算子数量"）——把 7 次 single_view_proj 调用合成 1 次，kernel launch 少、对导出部署图也更干净（else 分支就是老老实实 `for j in range(num_views)` 一路一路调，见抽帧 00_42_09 第 119 行以下）；② 合并后 dim0=21 份彼此独立，为 LiftSplat 内部 `save_memory` 的"拆成 21 份逐份外积"（Part 3）提供了统一的切分轴。⚠ 严格说"合并本身"不省显存（数据总量没变），省显存的是后面按份切开算的策略，合并是它的前置条件——讲者把两步的功劳说在了一步上，理解时要拆开。
- 【连接】"把多路相机折进 batch 维一起过网络"是多视角 BEV 的标准手法：BEVFusion 里 `x = x.view(B*N, C, fH, fW)` 后才过 depth_net/downsample，一模一样。
- 【连接2】和 Transformer 里把 (B, heads, …) 折成 (B·heads, …) 用 bmm 是同一种"维度会计学"：只要各份之间无交互，就可以进 batch 维白嫖并行。

### 句卡 2-4（合并 [00:42:32]~[00:43:02] 六个碎句，其中"permute"的说法需要澄清）

> **[00:42:32–00:43:02]** 「会变得每一路针孔相机，把它对应的 batch size、对应的 3 帧的 feature 和 depth、以及 grid map 会取出来……下面其实就是类似于把我做了一个 permute，就是把我的这个 7 的维度放到最前面去，为了便于后面作为 LSS 投影时候的一个使用。」

- 【直译】循环 j 把每路相机的数据切出来再拼接，等效于把相机维(7)从 dim2 挪到了最前面并与 (B·3) 融合。
- 【代码】"类似于 permute"——严格等价写法其实是一行：
  ```python
  # (3,1,7,C,H,W) --squeeze--> (3,7,C,H,W) --permute--> (7,3,C,H,W) --reshape--> (21,C,H,W)
  out = fv_feat.squeeze(1).permute(1, 0, 2, 3, 4).reshape(-1, C, H, W)
  ```
  循环+cat 与 permute+reshape 结果逐元素相同（练习 ch7-2 会验证）。工程里写成循环，多半是为了导出算子友好（有的推理引擎对高维 permute 支持差）以及和 else 分支代码结构对称。
- 【形状】(3,1,7,128,88,160) → (21,128,88,160)。注意 permute 的语义后果：**新 dim0 的排序是"7 在外、3 在内"**，即索引 k 对应 相机 k//3、帧 k%3。
- 【为什么】"7 放最前面"的实际含义是让"相机"成为切分的最外层——LiftSplat 内 save_memory 沿 dim0 `split(…,1)` 切出的每一份恰好是"某相机某帧"这个最小独立单元。
- 【连接】智谷课程里讲过 permute 只改 stride 不动数据、reshape 要求 contiguous——这里用 cat 直接生产出 contiguous 的结果，也就顺带绕开了 `permute().contiguous()` 的一次显式拷贝。

### 句卡 2-5（合并 [00:43:08]~[00:43:23] 五个碎句）

> **[00:43:08–00:43:23]** 「然后这里循环取出来，然后再沿着 D0 维做一个 cat，就变成了 21。这些都变成了 21 的一个 shape：21×128×88×160 的 feature。」（转写原文"21 x12 x 18 x 18 x 106"为口糊/误听）

- 【⚠校正】"21×12×18×18×106"按调试台照片校正为 **21×128×88×160**（抽帧 00_44_45 底部逐字：`fv_featmap_list_cat.shape -> torch.Size([21, 128, 88, 160])`）。同屏还有 `grid_map_list_cat.shape -> torch.Size([21, 224, 112, 2])`。
- 【直译】合并完成：图像特征 (21,128,88,160)、深度 (21,100,88,160)、网格 (21,224,112,2)，三者 dim0 一一配对。
- 【形状】自查口诀：21 = 1(B)×3(帧)×7(相机)；128 是 FPN 通道；88×160 是 1/8 图像平面；grid 的 224×112 是 BEV 半分辨率网格，最后的 2 是采样坐标 (x,y)。
- 【为什么】做完这一步，"多帧多相机"问题被完全化归为"21 个互不相干的单目 LSS 问题"——这是整章最重要的抽象化简。后面 LiftSplat 的所有代码都只需要按"单相机批量"来写。
- 【连接】LSS 论文的实现同样把 (B,N) 摊平成 B·N 个独立视锥再统一 splat；区别在原版 splat 时要跨相机聚合(同一 BEV 格子接多相机投票)，DenseBEV 把跨相机聚合完全推迟到下一章"多视角融合"，投影阶段保持逐相机独立——解耦得更彻底，也更好部署。

### 句卡 2-6（合并 [00:43:32]~[00:43:45]）

> **[00:43:32–00:43:45]** 「然后这个的话就是生成的 grid，是 21×224×112×2。224 和 112 对应的（是）我们的每个 BEV feature（的）一半，因为我们做 LSS 投影的时候是只把它投到下采样一倍的一个 BEV feature（上）。」（转写原文"21 x 12 x 12 x 12 x 2"校正为 21×224×112×2，调试台为证）

- 【直译】采样网格 grid 的形状是 (21,224,112,2)：给 224×112 的目标 BEV 网格里每个格子存了一个二维采样坐标。全尺寸 BEV 是 448×224，这里只投到它的一半分辨率。
- 【形状】这正是 `F.grid_sample` 要求的 grid 布局 (N, H_out, W_out, 2)：H_out=224（BEV 纵向，前后 179.2m/0.8m），W_out=112（BEV 横向，±44.8m/0.8m），最后一维 2 是归一化到 [-1,1] 的 (x,y)，指向**输入特征图**（Part 4 会看到输入是 (21,32,100,160) 的"深度×图像宽"平面）里的采样位置。
- 【为什么】BEV 几何账：448×224@0.4m ↔ 224×112@0.8m。投影选半分辨率是因为 grid_sample 的计算量正比于输出格子数，448×224 比 224×112 贵 4 倍——投影处在 21 路的公共路径上，是显存和时延大户；先在粗网格上成像，后面 BEV UNet 里再恢复分辨率，代价小收益大。
- 【连接】BEVFusion 同样在 1/8 特征+粗 BEV(如 180×180@0.6m 量级)上做 view transform，再由 BEV encoder 上采样——"投影粗、精修细"是 BEV 流水线的通用性价比策略。

### 句卡 2-7（合并 [00:43:52]~[00:44:06]，含一处未解之词）

> **[00:43:52–00:44:06]** 「如果说是把它投影到 448 和 224 的话，这个计算量以及时延都比较大，所以说会在 224 和 112 的分辨率上做投影。然后会（调用）……的一个模块。」（转写原文"48 和 24""24 和 112"按几何账校正为 448/224 与 224/112；末句原文"然后会是要不新购物的一个模块"⚠无法还原）

- 【直译】重复强调半分辨率投影的动机是算力/时延，然后引出即将进入的下一个模块。
- 【⚠存疑】"新购物的模块"是转写幻听（校正稿文件头也把它列为低置信项）。结合下一句 [00:44:17]"这里就是真正做投影的模块"以及代码第 104 行 `bev_feat, bev_mask = self.single_view_proj(*prj_args)`，最合理的还原是"然后会（调）single_view_proj 的一个模块"——即 `LiftSplatProjection`（抽帧 00_50_14 的 draw.io 框图也写着 `LiftSplatProjection(self.single_view_proj)`，可作旁证）。
- 【为什么】时延敏感是量产车载代码的第一约束——你会看到本章几乎每个设计（半分辨率、合并算子、bmm 选项、自定义 grid_sample 算子）都能用"Orin/Ascend 上跑得动"来解释。
- 【连接】结合你 4060 上 BEVFusion 实测 0.481s/iter 的体感：投影/池化类算子恰是 profile 里的热点，这里的每个开关都是对同类热点的工程回答。

**Part 2 小结**：① 循环+cat 把 (3,1,7,…) 的特征/深度/网格统一压成 dim0=21 的三件套，顺序是相机在外、帧在内，等价于 permute+reshape。② 合并的直接动机是减少 single_view_proj 的调用次数、并为逐份省显存计算提供统一切分轴；这一步本身不减显存。③ grid 为 (21,224,112,2)，投影目标是 448×224 全尺寸 BEV 的一半（224×112@0.8m），纯为算力/时延让步。

### 🔨 动手练习 ch7-2：循环 cat ≡ permute+reshape

```python
import torch

BT, V, C, H, W = 3, 7, 4, 8, 10          # 缩小版: (B*T)=3, 7 路相机
fv = torch.randn(BT, 1, V, C, H, W)

# 写法A: 视频里的循环 + cat
lst = [fv[:, 0, j, ...] for j in range(V)]
cat_a = torch.cat(lst, 0)                 # (21,C,H,W)

# 写法B: 一行 permute + reshape
cat_b = fv.squeeze(1).permute(1, 0, 2, 3, 4).reshape(V * BT, C, H, W)

print(cat_a.shape, torch.allclose(cat_a, cat_b))
# 还原索引: dim0 第 k 份 = 相机 k//BT, 帧 k%BT
k = 10
print(torch.allclose(cat_a[k], fv[10 % BT, 0, 10 // BT]))
# 预期输出:
# torch.Size([21, 4, 8, 10]) True
# True
```

---

## Part 3 LiftSplatProjection 之一：降通道 + 外积升维（00:44:17–00:46:52）

**本段在讲什么**：进入 `liftsplat_proj.py` 的 `LiftSplatProjection.forward(fv_featmap, depth_probs, grid_map)`。输入是刚拼好的 (21,128,88,160)/(21,100,88,160)/(21,224,112,2)。第一步 1×1 卷积把 128 通道降到 32；第二步给两个张量各 unsqueeze 一维，广播相乘做"外积"，生成带深度轴的特征体 (21,32,100,88,160)——这是 LSS 的 **Lift**。因为这个五维体太大，代码在 `save_memory` 开关下把它拆成 21 份逐份相乘。本段输出是 21 份 (1,32,100,88,160) 的特征体列表。

---

### 句卡 3-1

> **[00:44:17]** 「然后对应的这里的话就是真正做投影的一个模块」

- 【直译】前面 BEVProjector 只是"调度壳"，真正的 LSS 数学发生在这个子模块里。
- 【代码】抽帧 00_44_45 面包屑与代码逐字：文件 `fv2bev/liftsplat_proj.py`，第 26 行 `class LiftSplatProjection(nn.Module):`，第 49 行 `def forward(self, fv_featmap, depth_probs, grid_map):`。docstring 原文抄录（信息量很大）：
  ```
  """project fv feature map to bev with vertical conv
  Args:
      fv_featmap (Tensor): (B, C, H, W), front-view feature map
      depth_probs (Tensor): (B, Z + 1, H, W), depth distributions
      grid_map (Tensor):  (B, bev_h, bev_w, 2)
  Returns:
      bev_featmap (Tensor): (B, C', bev_h, bev_w)
      valid_mask (Tensor): (B, bev_h, bev_w)
  """
  ```
- 【⚠注意】docstring 里 depth 写的是 `(B, Z+1, H, W)`——"+1"就是深度垃圾桶 bin；且第 67~68 行有被**注释掉**的 `# remove last depth category (> Max Range)` / `# depth_probs = depth_probs[:, :, :-1]`。结合 Ch6 末尾（[00:40:06]"把为 0 的预测 depth 舍弃掉，只取 1 到 99"）：垃圾桶的裁剪已在进模块前做过/或以置 0 方式处理，这里的注释行是历史方案。当前跑进来的就是 100 个 bin。垃圾桶 bin 的确切编号(0 还是最后一个)在两处代码里说法不一，⚠建议对照 `ModuleBevProjectV2.__init__` 里的 `dustbin_zero / dustbin_additional / dustbin_last` 三个开关核实（抽帧 00_51_06 可见）。
- 【连接】docstring 的"with vertical conv"对应 `weighted_aggr`（沿高度用卷积做加权聚合）选项——本次没启用（见句卡 4-2），说明文档比代码超前/滞后是工程常态。

### 句卡 3-2 ⭐重点句（合并 [00:44:21]~[00:44:31]）

> **[00:44:21–00:44:31]** 「在这里首先会做一个降 channel。然后这里我们 BEV feature 应该是 128（⚠指图像特征通道），然后会把它变成 32。我们图像的 feature 就变成了 21×32×88×160。」

- 【直译】1×1 卷积把通道从 128 压到 32，其余维度不动。
- 【代码】第 63 行逐字：`image_features = self.reduce_channel(fv_featmap)  # (B, C', H, W)`。`reduce_channel` 的实现大概率是 `Conv1x1BnRelu(128, 32)`：证据一，文件末尾测试段注释 `# from .utils import Conv1x1BnRelu`（抽帧 00_48_25 第 141 行）；证据二，抽帧 00_45_20 悬停弹窗显示该张量 `grad_fn=<ReluBackward1>`——链尾是 ReLU。
- 【形状】(21,128,88,160) → **(21,32,88,160)**（调试台逐字：`image_features.shape -> torch.Size([21, 32, 88, 160])`）。
- 【为什么】32 这个数字是被外积逼出来的：下一步特征体大小 ∝ 通道数×深度 bin 数。128 通道的特征体是 21×128×100×88×160 ≈ 37.8G 元素（fp32 约 151GB），必炸；压到 32 后是 9.46 亿元素 ≈ 3.8GB（fp32），配合逐份计算才勉强可行。**通道压缩是外积型 view transform 的标配前置**。
- 【为什么2】语义上也说得通：投影只需要"这个像素是什么"的粗语义，细粒度判别留给 BEV 空间的后续网络；32 维够用。
- 【连接】BEVFusion 的 LSSTransform 同样在 lift 前把 256 压到 80（`self.depthnet` 一并输出 D+C）；LSS 论文 C=64。通道数量级都在几十，原因相同。
- 【⚠口误】讲者说"我们 BEV feature 应该是 128"——此时还没有 BEV feature，指的是**图像** feature 的通道。

### 句卡 3-3 ⭐重点句（合并 [00:44:44]~[00:45:07] 五个碎句）

> **[00:44:44–00:45:07]** 「然后在这里我们会把预测的 depth 和图像的 feature 做一个内积（⚠实为外积），相当于把我们的 feature 沿着预测的 depth 上做了一个扩展，就把它放到了我们所预测的置信度最高的那个 bin 上去。」

- 【直译】把每个像素的 32 维特征向量，按该像素 100 个深度 bin 的概率，按比例"发"到 100 个深度档位上——概率大的 bin 分得多，概率小的分得少。讲者用"放到置信度最高的 bin"是形象化说法，实际是**软分配**（所有 bin 都按权重拿到一份），不是硬 argmax。
- 【⚠术语校正】讲者全程说"内积/内机"，数学上这是**外积**（outer product）：对每个像素位置，32 维特征列向量 × 100 维概率行向量 → 32×100 矩阵。校正稿文件头和总编笔记均按外积理解，代码是广播逐元素乘，语义等价于逐像素外积。
- 【代码】准备动作（第 60~66 行逐字）：
  ```python
  channel_dim = 1
  depth_dim = 2
  image_features = self.reduce_channel(fv_featmap)          # (B, C', H, W)
  if not self.use_bmm:
      image_features = image_features.unsqueeze(depth_dim)   # (B, C', 1, H, W)
      depth_probs = depth_probs.unsqueeze(channel_dim)       # (B, 1, Z, H, W)
  ```
  相乘本体在第 97 行：`frustum_voxel_feats = depth_probs * image_features  # (B, C', Z, H, W)`——PyTorch 广播让 (21,32,**1**,88,160) × (21,**1**,100,88,160) 在互补的 1 维上撑开，得 (21,32,100,88,160)。
- 【形状】这正是转写 [00:45:27]~[00:45:48] 报的两组数：「21×32×1（×）88×160」和「21×1×100×88×160」（调试台照片均可对上）。
- 【为什么】这是 LSS 论文 Lift 步骤的原式：`f(u,v,d) = α_d(u,v) · c(u,v)`——深度概率 α 对上下文向量 c 的外积，把 2D 特征图升维成 3D 视锥（frustum）特征体。变量名 `frustum_voxel_feats` 直接向论文致敬。它的妙处：网络不必硬猜一个深度值（回归会被多峰情况撕裂），而是保留整条分布，让错误深度上的特征以小权重存在，由后续 BEV 网络自行消化。
- 【连接】BEVFusion 的 `depth.unsqueeze(1) * x.unsqueeze(2)` 一字不差是同一行代码；你读过的 LSS 原文公式 (1) 就是它。与之相对的另一派是 BEVFormer 的"反查式"（BEV 查询回图像取特征，无显式深度分布）——DenseBEV 属 LSS 正向派。

### 句卡 3-4（合并 [00:45:50]~[00:46:16] 七个碎句）

> **[00:45:50–00:46:16]** 「然后在这里会把我们的（张量）reshape……会把我们的 21 首先拆分成一个 list，相当于每个 list 里面就是 21 个元素了，（每个元素）就是 1×（1×）32×100×88×160。」

- 【直译】不直接做那个 3.8GB 的大乘法，而是先沿 dim0 把两个输入各切成 21 个薄片。
- 【代码】`save_memory` 分支（抽帧 00_46_45 第 91~95 行逐字）：
  ```python
  else:                                   # not use_bmm
      if self.save_memory:
          depth_probs_split = torch.split(depth_probs, 1)      # 21 × (1,1,100,88,160)
          image_features_split = torch.split(image_features, 1) # 21 × (1,32,1,88,160)
          frustum_voxel_feats_split = [t0 * t1 for t0, t1
                                       in zip(depth_probs_split, image_features_split)]
  ```
  `torch.split(x, 1)` 沿 dim0 切成步长 1 的 21 片，**返回的是视图不拷贝数据**，切分本身零开销。
- 【形状】每片乘积 = (1,32,1,88,160)×(1,1,100,88,160) → **(1,32,100,88,160)**（调试台逐字可见 `torch.Size([1, 32, 100, 88, 160])`；转写 [00:46:11]"1乘1乘以32乘100乘以88乘以160"多念了一个 1，应即此形状）。
- 【为什么】动机（讲者在 [00:42:06] 预告过）：单个大 tensor 的一次性乘法要同时申请约 3.8GB(fp32) 的连续输出；拆成 21 份后单次 kernel 只处理 1/21，还能避开某些推理硬件对单算子尺寸的上限。
- 【⚠我的推断/存疑】就**峰值显存总量**而言，这段代码写法其实省得有限：列表推导会把 21 份乘积**全部保留在 list 里**，直到 Part 4 的 mean 才逐份缩小——峰值仍约等于完整大张量。真正被优化掉的是"单次分配 3.8GB 连续块"的碎片化风险和单算子 workspace。若想真省，应把乘和 mean 融进同一个循环（`[(t0*t1).mean(dim=3) for ...]`）。留意实际主线代码是否如此，此处以录屏所见为准。
- 【连接】和 BEVFusion 里 bev_pool 用 QuickCumsum 自定义算子避免物化全部视锥点是同一类问题的不同解法：LSS 系方法的显存瓶颈永远在 lift 出来的五维体上。

### 句卡 3-5（合并 [00:46:20]~[00:46:52] 九个碎句，讲者此处车轱辘重复较多）

> **[00:46:20–00:46:52]** 「把我们的 feature 和预测的 depth 都拆分开，拆成 21 份。然后每一帧的、以及每一路的相机，它自己做一个外积（原话"内机"）的操作……这里出来的话就变成了也是 21 份，对应的就是（1×）32×100×88×160。」

- 【直译】重申：拆开后每份恰好是"某相机某帧"自己的 lift，互不串扰；输出仍是 21 份，每份 (1,32,100,88,160)。
- 【形状】21 份 × 1×32×100×88×160 ≈ 9.46 亿元素总量——与不拆时完全相同，只是分了 21 次算。
- 【为什么】"每份=某相机某帧"之所以成立，回看 Part 2 的合并顺序：dim0 的第 k 份 = 相机 k//3、帧 k%3。物理上不同相机/不同帧的视锥彼此独立，逐份算不损失任何信息——这就是当初非要把三个维度捏进 dim0 的回报。
- 【连接】调试台此刻刷屏的 `pydevd warning: Computing repr of frustum_voxel_feats_split (list) was slow (took 0.78s)`（抽帧 00_46_45 底部逐字）是个有趣的旁证：仅仅让调试器打印这个 21 份大列表的 repr 都要 0.78 秒——侧面感受这批张量有多大。你远程 debug BEVFusion 时如果 watch 面板卡顿，原因相同：把大张量从 watch 里移除即可。

**Part 3 小结**：① 1×1 Conv(+BN+ReLU) 把 128→32，是外积前的必要节流阀。② Lift = unsqueeze 出互补的 1 维后广播相乘：(21,32,1,88,160)×(21,1,100,88,160)→(21,32,100,88,160)，讲者口中的"内积"实为逐像素外积/软深度分配。③ save_memory 开关把大乘法拆成 21 份逐份做，规避大块显存分配（但列表持有全部结果，峰值节省有限，⚠见句卡 3-4）。

### 🔨 动手练习 ch7-3：外积 lift 与显存账本

```python
import torch

N, C, D, H, W = 21, 32, 100, 88, 160
img = torch.randn(N, C, H, W)
dep = torch.softmax(torch.randn(N, D, H, W), dim=1)   # 每像素100个bin概率和为1

# --- 整体外积 ---
vol = dep.unsqueeze(1) * img.unsqueeze(2)             # (21,32,100,88,160)
print(vol.shape, f"{vol.numel()*4/1024**3:.2f} GB (fp32)")

# --- save_memory: 拆 21 份逐份 ---
vol_split = [d * i for d, i in zip(torch.split(dep.unsqueeze(1), 1),
                                   torch.split(img.unsqueeze(2), 1))]
print(len(vol_split), vol_split[0].shape)
print("等价:", torch.allclose(vol, torch.cat(vol_split, 0)))

# 软分配语义: 沿深度求和应还原原特征 (概率和=1)
print("sum over D 还原:", torch.allclose(vol.sum(dim=2), img, atol=1e-5))
# 预期输出:
# torch.Size([21, 32, 100, 88, 160]) 3.52 GB (fp32)
# 21 torch.Size([1, 32, 100, 88, 160])
# 等价: True
# sum over D 还原: True
```

---

## Part 4 拍平成 pillar：沿图像高度求均值（00:46:55–00:48:19）

**本段在讲什么**：Lift 出来的每份特征体 (1,32,100,88,160) 有五个维度，讲者先把每个维度的物理含义讲透（100=深度=BEV 前后方向，88=图像高度，160=图像宽度≈BEV 横向），然后沿"图像高度 88"这一维求均值，把带高度的特征压成无高度的 pillar 特征 (1,32,100,160)，最后把 21 份 concat 回 (21,32,100,160)。这是 LSS 的 **Splat**——但注意：这里拍平的是**图像高度**，不是原版 LSS 拍平的世界坐标 z 轴。

---

### 句卡 4-1 ⭐重点句（合并 [00:46:55]~[00:47:34] 十个碎句，本章最值得背下来的语义句）

> **[00:46:55–00:47:34]** 「这个是获得了我们在 BEV 上的、其实是还有高度的一个特征。然后对应的 100 其实就是我们的一个深度，其实就对应上的是 BEV feature 上前后的一个范围。然后 88 呢，还是我们图像的一个高度。然后 160 呢，就是我们图像的宽度——其实在这里（就）对应了我们 BEV 的一个宽度。」

- 【直译】五维体 (21,32,100,88,160) 每个维度的身份证：32=语义通道；**100=深度 bin，投到 BEV 上就是"前后/径向距离"方向**；**88=图像纵向（高度），是唯一与 BEV 平面无关、待压掉的维度**；**160=图像横向，透视相机下不同图像列≈不同方位角，投到 BEV 上大致对应"左右/横向"方向**。
- 【为什么】这句是理解后面两步（拍平选谁、grid_sample 怎么采）的钥匙：一个 (depth_bin, image_col) 二维格点，经相机模型即可确定地面上一个 (前后, 左右) 位置——所以 (100,160) 平面**本身就是一张"以相机为原点的极坐标 BEV"**；而 88（图像行）方向对应的是空间中的竖直方向，BEV 表达不需要它，只能把它聚合掉。
- 【⚠精度提醒】"160≈BEV 宽度"是近似说法：图像列对应的是**方位角**而非等距横向坐标，等方位角线在 BEV 上是从相机张开的射线束——这正是 Part 6 可视化里每路相机特征呈扇形/楔形的原因，也是必须再做一次 grid_sample 重采样（极→直角坐标）而不能直接 reshape 的原因。
- 【连接】原版 LSS 的 splat 是把视锥点云按 (x,y) 落格、沿**世界 z 轴** sum-pool 成 pillar；DenseBEV 改为沿**图像行**聚合。二者近似等价的直觉：同一图像列上不同行的像素，在给定深度下大致落在同一 (x,y) 地面格上、只是高度不同。好处是聚合变成规则张量维度上的 mean，一个算子搞定，不需要 scatter/cumsum 这类部署困难的稀疏操作——这是"DenseBEV"之 dense 的一半含义。
- 【连接2】PointPillars 里"pillar"就是把 z 压掉的柱状特征；CenterPoint/BEVFusion 的 BEV 特征同理无高度。所有 BEV 方法殊途同归：高度信息以特征编码方式存活在通道里，而不是显式维度。

### 句卡 4-2（合并 [00:47:35]~[00:47:47]）

> **[00:47:35–00:47:47]** 「然后在这里会沿着第三维给它求一个均值，就把我们的立体的（转写"力气的"）一个 BEV 的特征，给它沿着高度拍平，就变成（21×）32×100×160。」（转写原文"32乘以88乘以100乘以160"中的 88 系口误——88 正是被拍掉的那一维）

- 【直译】对 (B,C,Z,H,W) 的 dim=3（即 H=88，图像高度）取 mean，五维变四维。
- 【代码】三条路径同屏可见（抽帧 00_47_23 第 106~115 行逐字）：
  ```python
  else:
      # Sum pooling over height axis to get pillar features
      if self.is_lite or self.use_bmm:
          frustum_feats = frustum_voxel_feats          # 这两种模式下高度已提前处理
      else:
          if self.save_memory:                          # ← 本次走这里(黄色高亮)
              frustum_feats_split = [t.mean(dim=3) for t in frustum_voxel_feats_split]
              frustum_feats = torch.cat(frustum_feats_split, 0)
          else:
              frustum_feats = frustum_voxel_feats.mean(dim=3)   # (B, C', Z, W)
  ```
- 【形状】每份 (1,32,100,88,160) --mean(dim=3)--> (1,32,100,160)；数据量瞬间缩小 88 倍，显存危机到此解除。
- 【⚠代码考古】注释写"**Sum** pooling"，代码做的是 **mean**——注释陈旧。校正稿文件头曾把"均值还是最大值"列为低置信项，**现以代码照片为准：均值**。mean 与 sum 只差常数因子 1/88（可被后续 BN/卷积吸收），但 mean 数值更稳，不会因 88 项累加把激活推大。
- 【为什么用 mean 不用 max】mean 保留整列的"软证据"且梯度流向所有高度位置，对以概率分布方式摊开的特征更自洽；max 只回传一个位置的梯度，且和"深度概率加权"的软语义相性差。
- 【连接】被 assert 挡掉的第三种方案 `weighted_aggr`（第 98~105 行：`assert not self.weighted_aggr, 'weighted_aggr option is not available'`；把 (B,C',Z,H,W) permute/flatten 成 (B,C'·H,Z,W) 后用 `self.aggr_module` 卷积聚合）就是 docstring 里的"vertical conv"——让网络**学**如何沿高度加权，而不是平均主义。工程上被禁用，估计是收益/时延比不划算。⚠此处为我依代码的推断。

### 句卡 4-3（[00:48:01]~[00:48:19]，合并两句）

> **[00:48:01]** 「然后这里相当于是每一路相机去做这样一个拍平的操作」 **[00:48:06–00:48:19]** 「然后在这里会再把它 concat 起来」

- 【直译】拍平是在 21 份的循环里逐份做的（列表推导），做完 `torch.cat(frustum_feats_split, 0)` 拼回 dim0=21 的整体：(21,32,100,160)。
- 【形状】从这里开始张量重新变小、变整——省显存的"拆"只存在于外积→拍平之间这一小段危险区,一出危险区立刻合并回批量形态，好继续吃 batch 并行。
- 【为什么】cat 放在 mean 之后而不是之前，是整个 save_memory 策略的点睛处：拼接发生在缩小 88 倍之后，拼接产生的新分配只有 21×32×100×160≈1075 万元素（约 41MB fp32），完全无压力。
- 【代码】此刻的 `frustum_feats`：(21, 32, 100, 160)。把它当普通图像看：高=100(深度/前后)、宽=160(图像列/方位)、32 通道——正是下一步 grid_sample 的输入"源图"。
- 【连接】题外的 `use_bmm` 路径（抽帧 00_44_45/00_46_45 第 75~89 行）值得你精读一遍：它把逐列外积+高度均值**融合成一个 bmm**：`image_features.permute(0,3,2,1).reshape(b*w, h, c)` 与 `depth_probs.permute(0,3,1,2).reshape(b*w, z, h)` 后 `frustum_voxel_feats = torch.bmm(depth_probs, image_features) / h  # (B*W, C', Z)`——(z,h)@(h,c) 的矩阵乘在收缩维 h 上求和、再除以 h，恰好=外积后沿高度取 mean，而且**从头到尾不物化五维体**。这是显存最优解，练习 ch7-4 验证等价性。本次调试走的是可读性更好的 save_memory 路径。

**Part 4 小结**：① 五维体的语义：100=深度→BEV 前后，160=图像列→BEV 方位/横向，88=图像行→竖直方向（BEV 不需要）。② Splat 的 DenseBEV 实现 = 沿 dim3(88) 求 mean（注释里的 sum 与备选的 max 均不成立，以代码为准），逐份做完 cat 回 (21,32,100,160)。③ use_bmm 路径用一个 bmm/h 同时完成 lift+splat 且不物化五维体，是部署形态；weighted_aggr(vertical conv) 被 assert 禁用。

### 🔨 动手练习 ch7-4：验证 bmm ≡ 外积+高度均值

```python
import torch

N, C, D, H, W = 2, 8, 10, 6, 12
img = torch.randn(N, C, H, W)
dep = torch.softmax(torch.randn(N, D, H, W), dim=1)

# 路径1: 外积 -> mean(H)   (本次视频走的 save_memory 语义)
vol = dep.unsqueeze(1) * img.unsqueeze(2)     # (N,C,D,H,W)
out1 = vol.mean(dim=3)                        # (N,C,D,W)

# 路径2: use_bmm, 照抄 liftsplat_proj.py 第75~85行
b, c, h, w = img.shape
im2 = img.permute(0, 3, 2, 1).reshape(b * w, h, c)     # (N*W, H, C)
b, z, h, w = dep.shape
dp2 = dep.permute(0, 3, 1, 2).reshape(b * w, z, h)     # (N*W, D, H)
out2 = torch.bmm(dp2, im2) / h                          # (N*W, D, C)
out2 = out2.reshape(b, w, z, c).permute(0, 3, 2, 1)     # -> (N,C,D,W)

print(out1.shape, out2.shape, torch.allclose(out1, out2, atol=1e-5))
# 预期输出: torch.Size([2, 8, 10, 12]) torch.Size([2, 8, 10, 12]) True
```

---

## Part 5 grid_sample：从相机系拍到自车系 BEV（00:48:19–00:49:48）

**本段在讲什么**：拍平后的 (21,32,100,160) 仍是"相机极坐标 BEV"（行=径向深度、列=图像方位角，以各相机为原点）。本段用 `F.grid_sample` 按预生成的 grid (21,224,112,2) 把它重采样到**自车坐标系**的规则 BEV 网格上，得到 (21,32,224,112)，并顺手算一个后续没用到的 valid_mask；最后把 21 拆回 (B·3, 7, …) 的组织方式，与输入对齐。输出交还给外层 BEVProjector。

---

### 句卡 5-1 ⭐重点句（合并 [00:48:19]~[00:48:49] 八个碎句）

> **[00:48:19–00:48:49]** 「然后在这里会做一个 grid_sample。因为当前我们做完（外）积之后得到的这个 feature，其实它还是在那个相机坐标系下。然后（用）我们生成的（转写"Sense的"⚠疑为"预生成的/离线生成的"）那个 grid map，就是由我们相机坐标系下的那个 feature，把它映射到自车坐标系下。然后这是 grid_sample，会得到我们的 BEV 的 feature，然后这个就是在自车坐标系下（的了）。」

- 【直译】frustum_feats 的每一格挂在"相机为原点的 (深度,方位) 网格"上；要变成全车统一、以自车为原点的直角 BEV，就按 grid_map 给的坐标去 frustum_feats 里双线性采样一遍。
- 【代码】抽帧 00_48_25 第 116~121 行逐字：
  ```python
  if not self.convertD:
      if os.getenv('USING_ASCEND_910B') == "1" and os.getenv('USING_CUSTOM_GRID_SAMPLE_2D') == "1":
          bev_featmap = adsop.training.custom_grid_sample_2d(frustum_feats, grid_map, align_corner=True)
      else:
          bev_featmap = F.grid_sample(frustum_feats, grid_map, align_corners=True)   # ← 本次走这行
          valid_mask = (torch.abs(grid_map[:, :, :, 0]) <= 1).float()
  ```
- 【形状】`F.grid_sample(input=(21,32,100,160), grid=(21,224,112,2))` → **(21,32,224,112)**。语义：对目标 BEV 的每个格子 (i,j)，grid_map[n,i,j] 存着"它在相机 n 的 (深度bin, 图像列) 平面上的归一化坐标 (x,y)∈[-1,1]"，grid_sample 去那儿双线性插值取 32 维特征。抽帧 00_49_24 的悬停弹窗直接证实产物：`shape = torch.Size([21, 32, 224, 112])`，且 `grad_fn=<CudnnGridSamplerBackward>`。
- 【为什么】为什么必须重采样而不能 reshape：极坐标网格与直角网格不共形——近处一个图像列覆盖的 BEV 横向很窄、远处很宽。grid_sample 的美妙在于**所有几何(内参、外参、安装位姿、BEV 网格定义)全部离线折叠进 grid_map**，在线只剩一个标准算子，可微、可导出、硬件友好；换标定只换 grid，不改网络。
- 【为什么2】注意坐标方向的取巧：这个 grid 是"**逆向映射**"（目标格子→源坐标），天然没有多个源点写同一目标格的冲突，所以不需要原版 LSS splat 的 scatter_add/cumsum trick——"dense"之名的另一半含义。代价：目标格子拿到的是插值出来的**一个**源位置特征，源平面上多格对一格的信息会被欠采样。
- 【连接】Ascend 分支是给华为昇腾 910B 部署留的自定义算子入口（环境变量 `USING_ASCEND_910B`、`USING_CUSTOM_GRID_SAMPLE_2D`），车 BU 背景的你应当很熟：CANN 生态下原生 grid_sample 长期是性能/精度坑，自定义算子是常规操作。else 分支之外还有 `convertD` 大分支（第 122~135 行，`grid_w_c`、`ads_grid_sampler_nhwc_with_mask_v2` 等），同为部署重写版，训练调试不走。
- 【连接2】MemoryManager/时序融合章的 warp 历史帧，同样用 grid_sample 实现——本章学透这个算子,后面白拿。

### 句卡 5-2（合并 [00:48:52]~[00:49:04]）

> **[00:48:52–00:49:04]** 「然后会在这里会取一个 valid mask——这个是在后续没有用到的。」

- 【直译】顺手算了一张"该 BEV 格子是否真的落在这路相机视野内"的掩码，但当前版本后续没人消费它。
- 【代码】`valid_mask = (torch.abs(grid_map[:, :, :, 0]) <= 1).float()` → (21,224,112)。原理：grid_sample 的归一化坐标里，|x|>1 表示采样点落在源图之外（视野外/深度范围外）；此处只检查了 x 分量。配合 `align_corners=True` 与默认 `padding_mode='zeros'`，视野外的格子在 bev_featmap 里本来就是 0。
- 【为什么它存在又没被用】多相机拼 BEV 时，理想做法是"加权平均 = Σfeat·mask / Σmask"以正确处理重叠区与盲区——mask 是为下一章多视角融合准备的原料。当前版本融合大概率简化成了直接求和/卷积(下一章验证)，于是 mask 被闲置。**但接口仍然把它传出去了**（`return bev_featmap, valid_mask`，第 136 行），这是量产代码"宁可传着不用，不要用时没有"的接口惯性。⚠"未用"以讲者口述为准，建议在下一章代码里确认。
- 【⚠小瑕疵】只查 x 不查 y：y 方向(深度 bin 归一化坐标)越界的格子逃过检查。若深度范围恰好覆盖整个 BEV 前后向，y 几乎不会越界，工程上无伤大雅——但读代码时应看出这层简化。
- 【连接】BEVFusion 的 LSS 里也有类似的 `kept = (x>=0)&(x<W)&...` 有效性过滤,那边是硬过滤参与池化的点,这边是软输出一张 mask,风格差异同样源于 dense/sparse 两种实现路线。

### 句卡 5-3（合并 [00:49:05]~[00:49:48] 九个碎句）

> **[00:49:05–00:49:48]** 「当前我们的 BEV 的 feature 其实应该还是相机的这一维度在前面——就是这里的 21 应该还是 7 乘以 (B·3) 这样一个维度。然后在后续会把我们会重新组织一下它的 shape，会把它变成 (B·3, …)，然后把 7 放到中间，就是为了和我们的输入保持一致。」

- 【直译】此刻 (21,…) 的排序仍是 Part 2 拼接时的"相机在外、帧在内"（21 = 7×3）；返回前要把它拆开重排成"批帧在外(3)、相机在中(7)"，即 (B·3, 7, 32, 224, 112)，与进模块时 fv_feat 的 (3,[1,]7,…) 布局对齐。
- 【代码】外层 BEVProjector 的收尾（抽帧 00_42_09 第 105~117 行逐字）：
  ```python
  bev_feat_split = torch.split(bev_feat, bev_feat.shape[0]//num_views)   # 21//7 → 7份,每份(3,32,224,112)
  bev_mask_split = torch.split(bev_mask, bev_mask.shape[0]//num_views)
  for j in range(num_views):
      bev_feat = bev_feat_split[j]
      bev_mask = bev_mask_split[j]
      if flip_lr is not None:
          ...
      bev_feat_list.append(bev_feat.unsqueeze(dim=1))    # (3,1,32,224,112)
      bev_mask_list.append(bev_mask.unsqueeze(dim=1))
  # 循环外 cat(dim=1) → (3,7,32,224,112)
  ```
  `split(x, 21//7=3)` 沿 dim0 每 3 个切一份——因为拼接时相机在外，**连续的 3 个正是同一相机的(B×3帧)**，所以第 j 份就是相机 j；`unsqueeze(1)` 再沿 dim1 拼，7 就被放到了中间。
- 【形状】(21,32,224,112) → 7×(3,1,32,224,112) → **(3,7,32,224,112)**；mask 同理 → (3,7,224,112)。与抽帧 00_50_14 框图里 outs 的标注 `(bs*3)*7*32*224*112, (bs*3)*7*224*112` 逐字吻合。
- 【为什么】"和输入保持一致"不是洁癖：下一章多视角融合要按 (批帧, 相机, C, H, W) 的约定读这批特征；进出布局对称，模块间的 contract 才简单。**任何"合并进 dim0 干活"的模块，出门前都欠一次逆变换**——这是一对必须配平的括号。
- 【连接】还记得练习 ch7-2 的还原公式吗：k↔(相机 k//3, 帧 k%3)。split+stack 就是这个公式的张量化。写这种代码时最容易犯的错是 `reshape(3,7,...)`（错！那是按"帧在外"解读）而不是 `reshape(7,3,...).transpose(0,1)`——顺序错了张量照样合法，特征却整幅串台，且 loss 还能降,极难排查。此类 bug 在 BEV 工程里俗称"相机窜位"。

**Part 5 小结**：① grid_sample 把相机极坐标平面 (100 深度×160 方位) 的 pillar 特征重采样到自车系 224×112 BEV，全部标定几何离线折叠在 grid_map 里，在线只剩一个可微标准算子。② valid_mask 由 grid x 坐标越界判断而来，接口传出但当前版本无人消费。③ 21 按"每 3 个连续片=同一相机"拆回 7 份，unsqueeze+cat 把相机维放回 dim1，得 (3,7,32,224,112)/(3,7,224,112)，与输入布局对齐。

### 🔨 动手练习 ch7-5：迷你"极坐标→BEV"grid_sample

```python
import torch, torch.nn.functional as F, math

# 源"极坐标"特征: 高=深度D, 宽=方位角A (模拟 (1,C,100,160))
D, A, C = 32, 48, 3
src = torch.zeros(1, C, D, A)
src[:, 0, :, A//2] = 1.0        # 正前方一条径向亮线
src[:, 1, D//2, :] = 1.0        # 等距离一圈亮弧

# 目标 BEV 网格 96x96, 自车在底边中点, 前向 0~20m, 横向±10m, 相机FOV=90°
H, W, max_d, fov = 96, 96, 20.0, math.pi/2
ys = torch.linspace(max_d, 0, H).view(H,1).expand(H,W)      # 前向距离
xs = torch.linspace(-10, 10, W).view(1,W).expand(H,W)       # 横向
r  = (xs**2 + ys**2).sqrt()
az = torch.atan2(xs, ys)
gx = az / (fov/2)               # 方位角 -> 归一化x
gy = r / max_d * 2 - 1          # 距离   -> 归一化y
grid = torch.stack([gx, gy], -1).unsqueeze(0)               # (1,H,W,2)

bev = F.grid_sample(src, grid, align_corners=True)
valid = (grid[..., 0].abs() <= 1).float()
print(bev.shape, valid.shape, f"视野内比例={valid.mean():.2f}")
# 打个字符画: 径向亮线在BEV上应是一条竖线, 等距弧应是一个圆弧, FOV外为0
for row in (bev[0,0] + bev[0,1])[::8]:
    print("".join(".#"[int(v > 0.3)] for v in row[::2]))
# 预期输出: torch.Size([1, 3, 96, 96]) torch.Size([1, 96, 96]) 视野内比例≈0.5
# 字符画中可见楔形(FOV)内的十字/弧线——正是视频可视化里楔形特征的成因
```

---

## Part 6 逐相机可视化与模块输出（00:49:48–00:51:44）

**本段在讲什么**：讲者切到 draw.io 页面（抽帧 00_50_07~00_50_48：浏览器里的 diagrams.net，下方还画着 `ModuleBevProject → for循环遍历针孔/鱼眼 → BEVProjector(prj) → LiftSplatProjection(self.single_view_proj)` 的调用层级框图），把 7 路针孔相机各自投影后的 BEV 特征图逐一贴出来看——每路都是一块 224×112 的黑底图上一束彩色楔形；鱼眼相机的 BEV 特征只有 16×16 的一小块马赛克。随后回到代码，指认整个 ModuleBevProject 的最终返回值：`return [outs, *depth_probs]`。本段输入是 Part 5 的 (3,7,32,224,112)，输出是打包给下一章"多视角融合"的 outs。

---

### 句卡 6-1（合并 [00:49:48]~[00:49:56] 三个碎句）

> **[00:49:48–00:49:56]** 「然后这边我也可视化了一下，就是做完拍平、做完 LSS 投影之后，我们每一路相机它的一个 feature 的一个形式。」

- 【直译】把 (3,7,32,224,112) 里某一帧、逐相机的 32 通道特征渲染成彩色图（通常取若干通道映射 RGB 或 PCA 降维上色），检查投影几何是否正确。
- 【为什么】这是 BEV 开发的黄金调试手段：投影代码的 bug（内外参用错、uv 翻转、归一化坐标算错、相机窜位）在数值上悄无声息,在可视化上一眼即穿——楔形的朝向、张角、近宽远窄的形态都得和该相机的安装位姿严丝合缝。**转岗建议：把"每写完一个投影/warp,先画图再谈 loss"内化成肌肉记忆。**
- 【代码】等价自制版：`plt.imshow(bev_feat[0, j, :3].permute(1,2,0).sigmoid().cpu())`，j 遍历 7 路。
- 【连接】你在 BEVFusion 里可以对 `view_transform` 输出做同款可视化（6 路楔形拼成一朵花）；对比着看能加深"每路相机在 BEV 上各占一个扇区"的空间直觉。

### 句卡 6-2（合并 [00:50:01]~[00:50:20] 七个碎句——逐路点名）

> **[00:50:01–00:50:20]** 「这个是我们前向一号针孔相机做完 LSS 投影之后它的一个 feature；然后这个是二号相机；然后这个是左侧后的；然后这个是左侧前；然后这个是右侧前的；然后右侧后的；然后这个是后向八号相机的。」

- 【直译】7 路针孔相机点名：前向 1 号、前向 2 号、左侧后、左侧前、右侧前、右侧后、后向 8 号——与"7 路针孔"的车辆配置对上（编号 1~8 中缺的编号应是鱼眼或未用位）。抽帧 00_50_14 里恰好是 7 块 224(高)×112(宽) 竖版黑底图+右侧 1 小块鱼眼马赛克。
- 【形状→画面】每块图就是 (32,224,112) 特征的伪彩渲染：纵向是自车前后 179.2m(0.8m/格)、横向是±44.8m。图中彩色楔形从某条边的一点张开——顶点=相机安装位置在 BEV 上的投影，张角=该相机 FOV，方向=安装朝向：前向相机楔形从图下方向上张，侧向的斜着张，后向 8 号从上方向下张（图像里可清楚看到不同块的楔形指向各异）。
- 【为什么楔形外全黑】两重原因叠加：① grid 把视野外格子的采样坐标推到 [-1,1] 之外，grid_sample 补 0；② 那正是 valid_mask=0 的区域。黑区就是"这路相机对这些 BEV 格子无话可说"——多相机融合(下一章)的意义就是让 7 个楔形互补拼出全景。
- 【为什么 1 号楔形窄长】前向 1 号通常是长焦：FOV 窄→楔形张角小,看得远→特征沿径向延伸长。对照第一块图（窄长竖条纹理）与第二块（宽扇形，广角 2 号）完全吻合——**从楔形形状能反读出相机规格**，这是可视化检查的进阶玩法。⚠1号=长焦为推断，与 Part 1 remote_proj 取 view=1 的旁证互洽,建议确认。
- 【连接】nuScenes 6 相机可视化出来是 6 个 60°/120° 扇形；这里多出"左侧前/左侧后"式的 4 侧向布局,是量产车常见的 7V 配置（前2+侧4+后1）。

### 句卡 6-3（合并 [00:50:20]~[00:50:57] 六个碎句）

> **[00:50:20–00:50:57]** 「然后这个是对应我们鱼眼相机做完 LSS 投影的 feature，它的 feature 是 16×16 的。然后我们针孔的相机是那个 448（的一半），是 224×112 的。然后鱼眼相机其实也是一样的（流程），然后这里就不用再讲了。」（转写"48/24x112"按几何与框图校正为 448→224×112）

- 【直译】鱼眼(4 路)走同一套 BEVProjector/LiftSplat 流程,只是它们的目标 BEV 网格小得多——每路 16×16；针孔是全 BEV(448×224)的一半 224×112。
- 【形状】框图 outs 标注逐字（抽帧 00_50_14 蓝框）：`[[(bs*3)*7*32*224*112, (bs*3)*7*224*112], [(bs*3)*4*32*16*16, (bs*3)*4*16*16]]`——外层 list 两个元素=针孔组/鱼眼组,每组 [bev_feat, valid_mask]。鱼眼组 dim0 同为 bs·3,相机数 4,特征 32 通道,网格 16×16。
- 【为什么鱼眼只配 16×16】鱼眼是近距环视(泊车/加塞盲区)传感器,有效测距就几米；16×16 的网格若按 0.8m/格算约覆盖 12.8m×12.8m 的近场——给它 224×112 纯属浪费。⚠具体鱼眼 BEV 网格的物理范围/分辨率视频未讲,16×16 覆盖范围为推断,待后续章节(多视角融合把它贴回大图的方式)反推核实。
- 【为什么"也是一样"】这句话背后是模块化设计的胜利：鱼眼畸变再大,差异全被离线吸进它自己的 grid_map(标定用鱼眼模型),在线代码与针孔共用同一个 LiftSplatProjection——换相机模型不换网络代码。
- 【连接】外层 `ModuleBevProject.forward` 里那个"for 循环遍历针孔/鱼眼"(框图原文)就是分组调用两次 BEVProjector 的壳；这也解释了 Part 1 形参里为什么有 `pinhole=True` 这个开关。

### 句卡 6-4（合并 [00:50:59]~[00:51:24] 七个碎句）

> **[00:50:59–00:51:24]** 「然后这里输出的话就是 out 和 depth。这个（depth）就是我们所预测的一个 depth——就是求完了 softmax 之后、以及把深度为 0 这一维会给它变成 0（之后的）、真正的所预测的 depth。然后在后续这个是没有用到的。」

- 【直译】模块对外返回两样：out（投影结果）和 depth（整理后的深度分布）；后者其实没有下游消费者。
- 【代码】抽帧 00_51_06 第 423 行黄色当前行逐字：`return [outs, *depth_probs]`（`ModuleBevProject.forward` 的收尾；其上第 388~421 行还有 `use_location`/`return_list` 的 parsing_embedding 分支,本次未走）。
- 【为什么 depth 还要传出去】两个工程理由：① Depth Loss 那一章的监督走的是 DepthNet 出口的 logits,这里传的 softmax 版留作调试/可视化/未来模块(如深度引导的融合)接口；② 保持 return 结构稳定,上游 solver 不必因配置增减改解包代码。"传而不用"与 valid_mask 同款——量产代码的接口惯性。
- 【⚠语义提醒】"把深度为 0 这一维变成 0"呼应 Ch6 末尾：bin0 是垃圾桶(无效深度),把它的概率置零,等价于让无效深度的特征在 lift 时权重为 0,不污染 BEV。注意置零后 100 维概率不再归一(和<1),外积后特征幅值略缩,可被 BN 吸收——细节但值得知道。
- 【连接】抽帧 00_51_06 下方还拍到 `class ModuleBevProjectV2(BaseModule)` 与 `dustbin_zero/dustbin_additional/dustbin_last` 三个 cfg 开关——V2 版把"垃圾桶放哪个 bin"做成了配置,佐证团队在深度垃圾桶设计上反复迭代过。

### 句卡 6-5（合并 [00:51:25]~[00:51:44] 收束句,末句为向下一章的过渡）

> **[00:51:25–00:51:44]** 「然后这个 out，out 里面就是我们做完 LSS 投影之后的 BEV 的 feature，以及它对应的一个 mask。做完 LSS 投影之后的话……（[00:51:44] 就后面的话就是涉及到视角以及模态特征的一个融合）」

- 【直译】out = [[针孔 BEV 特征(3,7,32,224,112), 针孔 mask(3,7,224,112)], [鱼眼 BEV 特征(3,4,32,16,16), 鱼眼 mask(3,4,16,16)]]。到此为止图像分支的"视角转换"全部完成；下一章把 7+4 路楔形拼成整张 BEV(视角融合),再和 lidar/radar 的 BEV 会师(模态融合)。
- 【形状】盘点本章总账,一条张量流水线走完：
  ```
  (3,1,7,128,88,160)+(3,1,7,100,88,160)+grid(3,7,224,112,2)
    →cat→ (21,128,88,160)/(21,100,88,160)/(21,224,112,2)
    →reduce→ (21,32,88,160)
    →lift(外积,拆21份)→ 21×(1,32,100,88,160)
    →splat(mean dim3)→cat→ (21,32,100,160)
    →grid_sample→ (21,32,224,112) + mask(21,224,112)
    →split/重排→ (3,7,32,224,112) + (3,7,224,112)  [鱼眼同理→(3,4,32,16,16)]
  ```
- 【为什么这里是全视频的"脊柱关节"】此前一切以像素坐标为家,此后一切以自车坐标为家;检测头、时序 warp、多模态融合能成立,全靠本章把"图像里的哪儿"翻译成了"车周的哪儿"。面试讲 DenseBEV,这条流水线值得一字不差背下来。
- 【连接】对照 BEVFusion 帮你记忆差异点:**同**——外积 lift、深度分布软分配、粗 BEV 投影;**异**——BEVFusion 用视锥点云+bev_pool(scatter 求和,跨相机一步到位),DenseBEV 用高度 mean+grid_sample(逐相机独立,融合后置),后者以少量信息损失换取全 dense 算子、部署无忧。这一对比一句话就能在面试里立住你对两条技术路线的理解。

**Part 6 小结**：① 可视化显示每路针孔在 224×112 BEV 上留下一个与其安装位姿/FOV 严格一致的楔形特征区,楔形外为零;鱼眼只投 16×16 近场小图,流程与针孔完全共用。② 模块最终 `return [outs, *depth_probs]`,outs 按[针孔组,鱼眼组]×[feat,mask]打包,depth 与 mask 均为"传而未用"的接口预留。③ 下一章接力:视角融合(7+4 楔形拼全图)与模态融合。

### 🔨 动手练习 ch7-6：把 21 拆回 (B·3,7) 并画"七楔拼花"

```python
import torch, torch.nn.functional as F, math

BT, V, C, Hb, Wb = 3, 7, 4, 64, 32
# 伪造 grid: 每路相机一个不同朝向的楔形(复用练习ch7-5思路, 此处简化为平移遮罩)
bev21 = torch.zeros(V * BT, C, Hb, Wb)
for cam in range(V):
    r0 = (cam * Hb) // V
    bev21[cam*BT:(cam+1)*BT, :, r0:r0+Hb//V, :] = cam + 1   # 每路占一条横带,值=相机号

# === 视频第105~117行的拆回逻辑 ===
split = torch.split(bev21, bev21.shape[0] // V)      # 7 份, 每份 (3,C,H,W)
out = torch.cat([s.unsqueeze(1) for s in split], 1)  # (3,7,C,H,W)  相机维回到 dim1
print(out.shape)                                     # torch.Size([3, 7, 4, 64, 32])

# 验证没有"相机窜位": 第 j 路的带子值应恒为 j+1
for j in range(V):
    band = out[0, j].amax()
    assert band == j + 1, f"相机{j}窜位!"
print("7 路相机各归各位 ✓")
# 反例: 直接 reshape(BT, V, ...) 会窜位 —— 自己打开注释体会
# wrong = bev21.reshape(BT, V, C, Hb, Wb); print((wrong[0,0]).amax())  # ≠1
# 预期输出:
# torch.Size([3, 7, 4, 64, 32])
# 7 路相机各归各位 ✓
```

---

## 本章存疑清单（全部 ⚠ 汇总）

| # | 位置 | 存疑点 | 我的推断与依据 |
|---|------|--------|----------------|
| 1 | [00:41:04] | 转写"3×7×7×100×88×160" | 应为 (3,1,7,100,88,160)；调试台 depths_logit=[3,7,100,88,160] 加占位维 |
| 2 | [00:41:24] | `remote_proj` 分支用途讲者未讲 | 代码硬取 view=1 与 grid_map[:,-1]，疑为前向长焦的远距二次投影，未启用 |
| 3 | [00:42:06] | "合并到一维是为了省显存" | 合并本身不省；省的是其后 save_memory 逐份计算，合并只是前置条件 |
| 4 | [00:44:06] | "新购物的模块"（转写幻听） | 应为调用 `self.single_view_proj` 即 LiftSplatProjection（draw.io 框图佐证） |
| 5 | [00:44:44] 等多处 | 讲者说"内积/内机" | 实为广播外积（逐像素 32×100 外积）；bmm 路径则是外积+高度均值的融合 |
| 6 | 第 95/112 行 | save_memory 峰值显存是否真降 | 列表推导持有全部 21 份乘积直到 mean，峰值≈不拆；真省需乘+mean 同循环。以录屏代码为据的分析 |
| 7 | 第 107 行注释 | "Sum pooling" vs 代码 mean | 以代码为准：mean(dim=3)。校正稿头部的"均值/最大值待核实"可关闭：是均值 |
| 8 | docstring (B,Z+1,H,W) 与垃圾桶 bin | bin0 还是最后一个 bin 是垃圾桶；被注释的 remove-last 行 | Ch6 口径为 bin0 置零、取 1~99；V2 类有 dustbin_zero/additional/last 三开关，团队迭代过，建议对配置核实 |
| 9 | 第 121 行 | valid_mask 只查 x 不查 y | y(深度向)越界未检；且 mask 后续未用（讲者口述），下一章代码待确认 |
| 10 | [00:50:25] | 鱼眼 16×16 的物理覆盖范围 | 视频未讲；按 0.8m/格推断约 12.8m 近场，待多视角融合章反推 |
| 11 | [00:50:01] | 前向 1 号是否长焦 | 由楔形窄长+remote_proj 取 view=1 推断，未获口述证实 |
| 12 | 转写多处数字 | "21x12x18x18x106"、"48和24"、"32×88×100×160"等 | 均按调试台/框图照片校正：21×128×88×160、448/224、(21,32,100,160) |


---
> [[Ch06_LSS投影输入准备|← Ch6]] · [[00_总览与脉络|📖 总览]] · [[Ch08_多视角与RC与模态融合|Ch8 →]]

