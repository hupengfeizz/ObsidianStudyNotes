> [[Ch11_Loss全解|← Ch11]] · [[00_总览与脉络|📖 总览]]

# Ch12 Box 解码 与 收尾QA（01:53:28–02:06:37）

> 本章是整场 DenseBEV 代码串讲的**最后一章**。前面 11 章把数据从「7 路针孔 + 4 路鱼眼 + lidar + radar」一路搬到了 448×224 的 BEV 特征图，再经 CenterPoint 检测头长出 10 个稠密分支，再算完 heatmap / dense 属性 / 目标级三层 Loss。本章要做的只有一件事：**把这些稠密预测图（dense prediction maps）变回一张「256 行 × 15 列」的目标列表**——也就是自动驾驶下游（tracker、预测、规划）真正能吃的东西。
>
> 讲者本人只用了约 13 分钟走完，但这 13 分钟里塞进了 3 个文件、5 个函数、两重 topk、一次坐标系翻转、一次 log/exp 还原、一次 atan2 还原、一次范围掩码、一次 15 维打包。本章会把每一句都摊开讲。

---

## 0. 本章在全局地图的位置

```
[视频未覆盖：Dataset/DataLoader、图像 backbone]
 → FPN 收尾 → DepthNet → Depth Loss
 → Lidar Backbone(透传) → Radar Backbone(pillar 编码) → RL 融合(UNet)
 → LSS 投影(外积+拍平+grid_sample) → 多视角融合(针孔+鱼眼) → RC 融合 → 模态融合
 → MemoryManager(10Hz 缓存) → 时序融合(warp 历史帧) → BEV UNet backbone
 → CenterPoint 检测头（10 个分支）→ Loss(heatmap + dense 属性 + 目标级回归)
 ★★★ Ch12：Box 解码（双重 topk 256）+ 收尾 QA ★★★  ← 你在这里，终点
```

一句话定位：**前 11 章都在「造图」，本章在「读图」。**

- **输入**：`preds_dict[0]` 这个 dict，里面装着一堆稠密分支的张量，全部是 `[1, C, 448, 224]` 形状（`heatmap` C=5、`reg` C=2、`height` C=1、`dim` C=3、`rot` C=2、`vel` C=2、`movement` C=1、`dir_cls` C=1 …），外加两张给下游用的特征图 `used_bev_feat` / `fusion_feat`。
  ⚠ **注意"分支数"和"解码用到的分支数"不是一回事**：帧 01_55_02 的 console 里还能看到 `close_heatmap [1,5,448,224]` 和 `rot_lidar [1,2,448,224]` 这两个键，**它们从头到尾没被 `decode` 碰过**（详见 0.1 节末尾的说明）。所以本章说的"10 个分支"指的是**参与解码的那 10 个**；`preds_dict` 里实际的键数更多。这在工业代码里是常态——训练时挂的辅助头，推理时并不都消费。
- **输出**：`pred_bboxes[task_id]`，形状 `[1, 256, 15]` 的稠密 tensor（未填的槽位是 `-1`），以及一个 `predictions_dicts` 列表（每个 batch 一个 dict，里面有 `bboxes / labels / scores / movements / directions / indexs / instance_embeddings`）。
- **本质**：一次「稠密 → 稀疏」的转换。**没有 anchor、没有 NMS（center 模式下）、没有 RoIAlign**，全靠 heatmap 峰值 + topk。这正是 CenterPoint / CenterNet 这条技术路线的招牌。

---

## 0.1 本章画面证据清单（帧证）

这一章涉及三个源码文件，讲者在 IDE 里来回跳。为了让你随时知道「他现在在哪个文件的第几行」，先把帧里读到的路径与行号固化下来：

| 文件（面包屑来自帧内 IDE 顶栏） | 类 / 函数 | 行号 | 出现时段 |
|---|---|---|---|
| `pilei/workspace/00_DEBUG_develop/e2e/tasks/bev_task/uvp_module/models/det_head/det_head.py` | `class DetHead(BaseModule)` → `def forward(self, *inputs, **kwargs)` | L16 / L62 | 01:53:28–01:53:57、02:04:45–02:06:37 |
| 同上 | `# decode预测框、获取匹配关系` 分支 | L150–L161 | 01:53:31–01:53:57 |
| `.../models/det_head/centerpoint/centerpoint_head.py` | `class CenterHead(nn.Module)` → `def get_bboxes(self, preds_dicts, gt_indexs=None, record_valid_obj_indexes=None, masks=None, select_bev_feat=False)` | L246 / L1325 | 01:53:58–01:55:16、02:03:26–02:04:45 |
| `.../models/det_head/centerpoint/centerpoint_bbox_coders.py` | `class CenterPointBBoxCoder` | L11 | 01:55:17–02:03:25 |
| 同上 | `def _gather_feat(self, feats, inds, feat_masks=None)` | L44 | （Ch11 用到，本章反复调用） |
| 同上 | `def _topk(self, scores, K=80)` | L65–L103 | 01:55:36–01:59:48 |
| 同上 | `def _transpose_and_gather_feat(self, feat, ind)` | L105 | 01:57:02 起反复出现 |
| 同上 | `def decode(self, used_bev_feat, fusion_feat, heat, rot_sine, rot_cosine, hei, dim, vel, reg=None, task_id=-1, movement=..., direction=..., rot_short_sin=..., rot_short_cos=...)` | L124–L267 | 01:55:17–02:03:25 |
| 内部 wiki 页（浏览器）`...wei.com/domains/153/wiki/4665/WIKI202312012475625` | gt 11 维表 / pred 15 维表 / yaw 与 dir_cls 定义 | — | 02:03:57–02:04:40 |

**调试器实测形状**（来自帧内 Python Console，是本章最硬的证据，全部逐字抄下）：

```text
dir_cls        torch.Size([1, 1, 448, 224])
movement       torch.Size([1, 1, 448, 224])
rot_lidar      torch.Size([1, 2, 448, 224])       # ⚠ 键名，见下方说明
close_heatmap  torch.Size([1, 5, 448, 224])       # ⚠ 第二张 5 通道 heatmap，见下方说明
heatmap        torch.Size([1, 5, 448, 224])

>>> target_box.shape
torch.Size([1, 256, 10])
>>> preds_dict[0]['anno_box'].shape
torch.Size([1, 10, 448, 224])
>>> scores.shape
torch.Size([1, 5, 448, 224])
>>> topk_scores.shape
torch.Size([1, 5, 256])
>>> topk_inds.shape
torch.Size([1, 5, 256])
>>> topk_score.shape
torch.Size([1, 256])
```

**这 6 行 console 输出就是本章的骨架**：`[1,5,448,224]` →（第一重 topk）→ `[1,5,256]` →（第二重 topk）→ `[1,256]`。后面所有解释都围绕这条形状链展开。

**关于上面那两个"多出来的键"（讲者一句都没提，但画面里明明白白，必须补上）：**

- **`close_heatmap [1, 5, 448, 224]`** —— 这是 `preds_dict[0]` 里的**第二张 5 通道分类图**，和主 `heatmap` 同形同类别数。它出现在 console 的键遍历里（帧 01_55_02 底部 Python Console 逐字可读），但**本章的 `get_bboxes` / `decode` 从头到尾一次都没用过它**。
  - 【我的推断（⚠）】名字里的 `close` 最可能是"近距离/近场"：即模型除了主 heatmap，还额外出一张**专门监督近场目标**的分类图。理由有三：(1) 通道数同为 5，说明是同一套类别定义，不是别的任务；(2) 分辨率同为 448×224，是同一张 BEV 图上的第二个分类分支；(3) DenseBEV 的 BEV 前向覆盖 95.4 m，远近目标在 heatmap 上的高斯核尺度差异极大（近处一辆车占几十个格子、远处只占 1~2 个），用一张图同时管远近，近场容易被大高斯核糊掉——单独出一张近场图是很自然的工程解法。
  - 【它为什么不进解码】`decode` 只吃 `batch_heatmap`。所以 `close_heatmap` 要么只在训练时作为**辅助监督**（deep supervision，涨点但不改变推理接口），要么是给别的下游模块（比如近场泊车/AEB 链路）单独用的。**这是"训练图 ≠ 推理用图"的典型例子**，读工业代码时要习惯：`preds_dict` 里的键数 > 解码用到的键数。
  - 【建议核实】`grep -n "close_heatmap" -r models/det_head/` 看它在 loss 里怎么被用。
- **`rot_lidar [1, 2, 448, 224]`** —— 2 通道，形状和 `rot`（sin/cos）完全一致。
  - 【我的推断（⚠）】最可能是**"lidar 坐标系下的朝向"**：即同一个 (sin, cos) 朝向头，但监督目标是 lidar 系而非自车系的 yaw；或者它就是 `rot` 这个键在这份 checkpoint 里的真实名字（`decode` 侧代码读的是 `preds_dict[0]['rot']`，见帧 01_54_24 的 L1366-1367）。
  - 【为什么值得留意】如果确实存在"自车系 yaw"和"lidar 系 yaw"两套，那么 Part 12-8 解出来的 `rot` 到底在哪个坐标系下，就直接决定了下游拿到的朝向对不对。**自车系和 lidar 系之间通常差一个安装外参的 yaw 偏置**，几度到几十度不等。⚠ 这是本章最值得回服务器确认的一条。

**讲者身份**（画面右上角会议软件的"当前说话人"浮层）：
- 01:53:28 – 02:05:22：`pilei p00804189`（本场串讲的主讲人，前 11 章也是他）
- 02:05:28 – 02:06:35：`jiazhiwei j00812467`（另一位同事，做最后的项目层面总结）

这条信息很重要：**本章最后那段「代码是从旧版本迁过来的 / 后面基于 head 改 / Smart 上的优化慢慢回迁 / 现在稳定性差一些」不是主讲人说的，是另一个人（大概率是负责人/组长）在做收尾定调。** 后面 Part 12-11 会专门讲这个区别为什么重要。

---

## 0.2 三分钟速查：本章会出现的所有变量

先把名字过一遍，后面逐句读的时候就不会晕：

| 变量 | 形状 | 含义 | 出生行 |
|---|---|---|---|
| `batch_heatmap` | `[1,5,448,224]` | sigmoid 后的分类置信度图 | `centerpoint_head.py:1350` |
| `batch_reg` | `[1,2,448,224]` | 中心点亚像素偏移（行方向、列方向） | L1352 |
| `batch_hei` | `[1,1,448,224]` | 目标中心 z（离地高度） | L1353 |
| `batch_dim` | `[1,3,448,224]` | **exp 还原后**的 l/w/h | L1356 |
| `batch_rots` / `batch_rotc` | `[1,1,448,224]` ×2 | yaw 的 sin / cos 分量 | L1361–L1367 |
| `batch_vel` | `[1,2,448,224]` | vx / vy | L1372 |
| `batch_movement` | `[1,1,448,224]` | 动静（两阶段时 = prob + class 的加法编码） | L1376–L1385 |
| `batch_direction` | `[1,1,448,224]` | 朝向二分类（sigmoid 后） | L1387–L1389 |
| `used_bev_feat` / `fusion_feat` | `[1,C,448,224]` | 给下游 tracker 抠 instance embedding 用 | L1347–L1348 |
| `scores, inds, clses, ys, xs` | `[1,256]` ×5 | `_topk` 的五元返回 | `bbox_coders.py:147` |
| `temp` | `list[dict]`，len=batch | 每个 batch 一份解析结果 | L1391 / L267 |
| `origin_bbox_preds` | `[1,256,10]` | 未做范围过滤的原始 box（10 列） | L267 |
| `pred_bboxes[task_id]` | `[1,256,15]` | **最终 15 维输出**，初值 `-1` | `centerpoint_head.py:1413` |

---

# Part 12-1｜Loss 收尾，切换到「解析框」分支（01:53:28 – 01:53:57）

### 本段在讲什么

上一章（Ch11）刚把 heatmap loss、dense 属性 loss、目标级 10 维 L1 loss、朝向/速度/动静的分区间加权 loss 全部塞进 `tb_dict`，并强调了 `loss_` 前缀项只是给 TensorBoard 看的、真正反传时会被排除。这一段是**转场**：讲者从 `det_head.py` 的 `forward()` 里 loss 那几行往下滚，滚到第 150 行那句中文注释 `# decode预测框、获取匹配关系`，然后一路点进 `get_bboxes`。

- **输入**：`det_output`（`self.det_head(data_dict)` 的返回），以及训练态下 `get_loss()` 返回的 `gt_inds / record_valid_obj_indexes / masks`。
- **输出**：进入 `CenterHead.get_bboxes`。
- **流水线位置**：Loss 出口 → 解码入口的门。

---

**[01:53:28]** 原话：「然后是这里 Loss 计算完了。」

- 【直译】上一大段（heatmap loss + dense 属性 loss + 目标级 loss + TensorBoard 明细项）到此为止，代码指针要往下走了。
- 【代码】画面上此刻停在 `det_head.py` 第 131–148 行区域：
  ```python
  # 计算loss
  det_loss = torch.tensor(0.0).to(inputs[0])
  tb_dict  = dict()
  if self.training:
      det_loss, tb_dict, gt_inds, record_valid_obj_indexes, masks = self.det_head.get_loss()
  else:
      gt_inds                  = [i[2] for i in data_dict['centerpoint_head_gt']]
      record_valid_obj_indexes = [i[3] for i in data_dict['centerpoint_head_gt']]
      masks                    = [i[4] for i in data_dict['centerpoint_head_gt']]
  ```
- 【为什么】注意这个 `if self.training / else` 的对称写法：**训练时匹配关系是 loss 函数「顺手」算出来的，推理时则直接从 dataset 传进来的 `centerpoint_head_gt` 里取。** 这是很多工业代码的典型做法——推理阶段也要 GT 索引，是为了做「预测框 ↔ GT 框」的配对评测和 DenseBEV 的属性任务采样，而不是为了算 loss。
- 【连接】对照 mmdetection3d 的 `CenterHead`：官方实现里 `loss()` 只返回 `loss_dict`，不吐 `inds/masks`。这里多返回三样，是华为这套 e2e 框架为了「检测 → 跟踪 → 端到端」的贯通做的改造。你在 BEVFusion 里见过的 `get_targets()` 返回 `(heatmaps, anno_boxes, inds, masks)`，就是这几个东西的来源。

---

**[01:53:31]** 原话：「然后可以讲一下这个是怎么通过预测的……」
**[01:53:35]** 原话：「……预测的这些属性去把 BOX 给解析出来。」

- 【直译】接下来的主题：网络吐出来的是一堆「图」（每个像素一个属性值），怎么把它变成一个个真正的 3D 框。
- 【代码】对应 `det_head.py` L150–L161：
  ```python
  # decode预测框、获取匹配关系
  if not self.training or self.export_instance_embeddings:
      for i in range(len(det_output['pred_dict'])):
          for j in range(len(det_output['pred_dict'][i])):
              det_output['pred_dict'][i][j]['bev_mid_feat_list'] = bev_mid_feat_list
              det_output['pred_dict'][i][j]['fusion_feat']       = fusion_feat
              if self.det_head.enable_corner_det:
                  det_output['pred_dict_corner'][i][j]['bev_mid_feat_list'] = bev_mid_feat_list
                  det_output['pred_dict_corner'][i][j]['fusion_feat']       = fusion_feat

      result = self.det_head.head.get_bboxes(
          det_output['pred_dict'], gt_inds, record_valid_obj_indexes, masks,
          select_bev_feat=self.select_bev_feat)
  ```
- 【为什么】这行 `if not self.training or self.export_instance_embeddings:` 很关键：**默认训练时根本不解码**（省算力，训练只需要 loss）；只有推理，或者显式打开 `export_instance_embeddings`（要给下游 tracker 导 instance 特征）时才走解码。这就是为什么讲者能在训练脚本里打断点、却要专门把 `self.training` 关掉才能演示这段。
- 【形状】`det_output['pred_dict']` 是一个 **list of list of dict**：第一层是 task（多任务头，DenseBEV 这里只有 1 个 task），第二层是……实际上代码里 `preds_dicts[0][task_id]` 与 `preds_dict[0]` 混用，说明外层长度为 1。所以 `preds_dict[0]` 就是那唯一的属性字典。
- 【连接】CenterPoint 原版有 `tasks=[dict(num_class=1, class_names=['car']), dict(num_class=2, ...)]` 这种多任务分组（每组一个 heatmap 头）。DenseBEV 把 5 个类别塞进**同一个** task 的 5 通道 heatmap，所以 `task_id` 恒为 0。这也解释了后面为什么第二重 topk 是在 `5×256` 上做——如果分了多 task，就得每个 task 单独 topk 再合并。

---

**[01:53:40]** 原话：「可以说一下 ZIN。」 ⚠
**[01:53:43]** 原话：「这些可以不用看。」

- 【⚠ 存疑】"ZIN" 是 Whisper 的音译残留。结合此刻画面（01:53:44 / 01:53:47 两帧，光标停在 `det_head.py` L136–L148），他正对着这一段：
  ```python
  # Transpose inds
  gt_inds = list(map(list, zip(*gt_inds)))
  gt_inds = [torch.stack(gt_inds_) for gt_inds_ in gt_inds]
  # Transpose record_valid_obj_indexes
  record_valid_obj_indexes = list(map(list, zip(*record_valid_obj_indexes)))
  record_valid_obj_indexes = [torch.stack(r_) for r_ in record_valid_obj_indexes]
  # Transpose inds
  masks = list(map(list, zip(*masks)))
  masks = [torch.stack(masks_) for masks_ in masks]
  ```
  我的推断：他说的大概率是 **`gt_inds`**（「G-T-inds」被听成 ZIN），或者是 **`zip`**（这三段的核心操作就是 `zip(*x)` 转置）。两种读法都指向同一段代码，含义不变：**这里只是把「batch-major 的 list」转成「task-major 的 list」再 stack 成 tensor，是纯数据搬运，跟解码逻辑无关，所以他说"可以不用看"。**
- 【代码】`zip(*x)` 是 Python 的经典转置写法。若 `gt_inds` 原本是 `[[b0t0, b0t1], [b1t0, b1t1]]`（外层 batch、内层 task），`list(map(list, zip(*gt_inds)))` 得到 `[[b0t0, b1t0], [b0t1, b1t1]]`（外层 task、内层 batch），再 `torch.stack` 沿 batch 维堆成 `[B, ...]`。
- 【为什么】DataLoader 的 collate 是按 sample 组织的，而 head 是按 task 组织的，两者维度顺序天然相反，必须转一次。这类"胶水代码"在工业仓库里占比极高，看懂一次就够了。

---

**[01:53:58]** 原话：「在这里就是把我们预测的 heatmap……」
**[01:54:00]** 原话：「……就是分类的 heatmap……」
**[01:54:03]** 原话：「……给它取一个 Sigmoid。」
**[01:54:04]** 原话：「就变成了我们就分类的一个 Score。」

> 这是本章第一句**重点句**，五个角度全给。

- 【直译】网络最后一层卷积吐出来的 heatmap 是「logit」（可正可负、无界），套一层 sigmoid 把它压到 (0,1)，就成了每个 BEV 格子属于每个类别的置信度分数。
- 【代码】`centerpoint_head.py` L1350，画面里这一行被黄色断点条高亮：
  ```python
  batch_heatmap = preds_dict[0]['heatmap'].sigmoid()
  ```
  上面两行是 L1347–L1348：
  ```python
  # 取出使用过的bev_feat, 用于后续取instance_embeddings
  used_bev_feat = preds_dict[0]['used_bev_feat']
  fusion_feat   = preds_dict[0]['fusion_feat']
  ```
- 【形状】`[1, 5, 448, 224] → [1, 5, 448, 224]`（sigmoid 是逐元素的，形状不变）。5 = 类别数（car / truck / bus / VRU / …）；448 = BEV 纵向格数（前 95.4 m + 后 83.8 m = 179.2 m ÷ 0.4 m）；224 = BEV 横向格数（左右各 44.8 m = 89.6 m ÷ 0.4 m）。
- 【为什么】三层原因：
  1. **数值层面**：训练用的是 Gaussian Focal Loss，它要求输入是概率而不是 logit。训练时在 loss 内部做 `clip_sigmoid`（sigmoid 后再 clamp 到 `[1e-4, 1-1e-4]` 防 log(0)）；推理时不需要 clamp，直接 sigmoid。
  2. **语义层面**：topk 要在「可比较的分数」上排序。logit 也能排序（sigmoid 单调），但阈值 `score_threshold`（后面 L225）是按概率设的，所以必须先转概率。
  3. **多标签而非多分类**：注意是 **sigmoid 不是 softmax**。CenterPoint 的 5 个类别通道彼此独立，一个格子可以同时对两个类别都有高响应（然后由 topk 各自选出，产生两个不同 label 的框）。这跟 YOLO 的 objectness×class 或 DETR 的 softmax 分类是不同的哲学。
- 【连接】
  - **对 YOLO 用户**：等价于 YOLOv5/v8 里 `pred[..., 4:5].sigmoid()` 那一步。YOLO 也是 sigmoid 而非 softmax（v3 之后），原因同上——多标签友好。
  - **对 BEVFusion 用户**：`mmdet3d/models/dense_heads/centerpoint_head.py` 的 `get_bboxes` 里也有一模一样的 `batch_heatmap = preds_dict[0]['heatmap'].sigmoid()`。这套 DenseBEV 代码的骨架就是从 mmdet3d 的 CenterHead 长出来的（后面 `_topk` 的注释里那个 `# original:` 标记会直接证明这一点）。
  - **数值直觉（帧 01_54_24 逐字核对过）**：讲者把鼠标悬停在 `batch_heatmap` 上，IDE 弹出的变量 tooltip 显示：
    ```text
    T        = tensor([[[[0.5024],
    data     = tensor([[[[0.5024, 0.5023, 0.5022,  ..., 0.5025, 0.5023, 0.5024],
    device   = device(type='cuda', index=0)
    dtype    = torch.float32
    grad     = None
    grad_fn  = <SigmoidBackward0 object at 0x7fcd222f57c0>
    is_cuda  = True
    is_leaf  = False
    ```
    三条信息量：**(1) 全图都是 0.50 出头**，意味着 logit ≈ 0，这个 checkpoint 基本没训过（或者随机初始化 + 少量 iter）。这是理解后面所有截图数值的前提——**别把 0.51 当成"检测到了车"，它只是"网络还啥都不知道"**。**(2) `grad_fn = SigmoidBackward0` 且 `is_leaf = False`**，证明这一步是在**带梯度的计算图里**跑的（没有 `torch.no_grad()`），也就佐证了 Part 12-1 说的"打开 `export_instance_embeddings` 后训练态也能走解码、梯度能从 instance embedding 回流"。**(3) `dtype = torch.float32`**，不是 fp16——这条在 Part 12-9 讲 `post_center_range` 的 dtype 时会再用到。

---

**[01:54:08]** 原话：「其实它还是 BEV feature 上……」
**[01:54:10]** 原话：「……还是 BEV feature 上的。」
**[01:54:12]** 原话：「1 乘 5 4824。」（即 `[1, 5, 448, 224]`）

- 【直译】强调：sigmoid 之后它**仍然是一张图**，不是目标列表。5 个通道、448×224 个格子，每个格子一个分数。
- 【形状】`[1, 5, 448, 224]`。⚠ 转写把 `448` 和 `224` 连读成了 "4824"，全章反复出现，都要读作 448×224。交叉验证：Ch11 里讲者说过展平后是 `1 × 100352`，而 448×224 = **100352**，完全吻合。
- 【为什么】这一句看似废话，其实是他在建立本章最重要的心智模型：**解码 = 把「图空间」的 100352 个候选，压缩成「列表空间」的 256 个目标。** 后面每一步都在做这件事的某个环节。
- 【连接】在 BEVFusion / CenterPoint 里，这张图的尺寸通常是 `[B, num_class, 180, 180]`（nuScenes，voxel 0.075 m × out_size_factor 8 = 0.6 m，范围 ±54 m）。DenseBEV 用的是 **0.4 m 分辨率、前后不对称的 448×224**，这是量产车「前向看得远、侧向够用就行」的典型取舍。

---

### 🔨 动手练习 ch12-1：造一张 DenseBEV 尺寸的假 heatmap，感受"稠密"的量级

```python
import torch

B, C, H, W = 1, 5, 448, 224          # DenseBEV 的真实尺寸
logits = torch.randn(B, C, H, W) * 0.01   # 模拟"几乎没训练"的网络：logit≈0
heat   = logits.sigmoid()

print("heat.shape        :", tuple(heat.shape))       # (1, 5, 448, 224)
print("每类候选格子数     :", H * W)                    # 100352
print("全部候选(含类别)   :", C * H * W)                # 501760
print("heat 数值范围      :", heat.min().item(), heat.max().item())
print("heat 均值          :", heat.mean().item())      # ≈0.5，和讲者屏幕上的 0.5024 对得上

# 埋两个"真目标"进去看看差别
heat[0, 0, 100, 60] = 0.93     # 第0类，第100行第60列
heat[0, 3, 300, 150] = 0.88    # 第3类
print("top5 全局分数     :", heat.view(-1).topk(5).values)
# 预期输出：tensor([0.9300, 0.8800, 0.5<xx>, 0.5<xx>, 0.5<xx>])
# ——两个人造峰值远高于噪声底噪，这就是 heatmap 检测能靠 topk 直接选目标的原因
```

**你应该看到的**：`100352` 这个数字（= 448×224）和讲者 Ch11 说的 "1 乘以 10 万 0352" 严丝合缝；以及"未训练网络 heatmap 全是 0.5"这一现象。

### 【小结】

1. 解码入口在 `det_head.py:151` 的 `if not self.training or self.export_instance_embeddings:`——**训练默认不解码**，这是省算力的工程设计。
2. `get_bboxes` 的第一件事是 `heatmap.sigmoid()`，把 logit 变成 5 通道、448×224 的独立概率图；**用 sigmoid 不用 softmax**，因为 CenterPoint 的类别通道是多标签独立的。
3. 屏幕上 heatmap 值全在 0.50 附近，说明这是一个几乎没训练的 checkpoint；本章后面所有截图里的数值都要按这个前提去读。

---

# Part 12-2｜把 10 个稠密分支逐个"接出来"（01:54:14 – 01:55:16）

### 本段在讲什么

`get_bboxes` 在真正调 `bbox_coder.decode` 之前，先把 `preds_dict[0]` 里 10 个分支一个一个拆成局部变量，顺手做两件"还原"：**dim 取 exp**、**rot 拆成 sin/cos 两支**。这一段是"备料"，形状全都保持 `[1, C, 448, 224]` 不变，一个目标都还没选出来。

- **输入**：`preds_dict[0]`（dict）
- **输出**：`batch_heatmap / batch_reg / batch_hei / batch_dim / batch_rots / batch_rotc / batch_rots_short / batch_rotc_short / batch_vel / batch_movement / batch_direction` 共 11 个局部变量
- **流水线位置**：解码函数的"参数装配区"

---

**[01:54:14]** 原话：「然后这里对应的 XYZ。」

- 【直译】接下来两行取的是位置相关的分支。
- 【代码】`centerpoint_head.py` L1352–L1353：
  ```python
  batch_reg = preds_dict[0]['reg']        # [1, 2, 448, 224]  中心点亚像素偏移 (dx, dy)
  batch_hei = preds_dict[0]['height']     # [1, 1, 448, 224]  中心点 z
  ```
- 【形状】`reg` 是 2 通道、`height` 是 1 通道。**注意 x、y 不是直接回归的**——x、y 由「格子索引」+「reg 偏移」两部分组成（Part 12-7 详讲），只有 z 是完整回归的。
- 【为什么】BEV 是俯视图，x/y 天然由格子位置提供了强先验（误差 ≤ 0.4 m），网络只需要学 sub-cell 的小残差；而 z 在 BEV 图上完全没有位置先验（BEV 把高度维压掉了），只能硬回归。**这就是为什么 `reg` 是 2 通道而不是 3 通道。**
- 【连接】完全对应 CenterNet 的 `hm + wh + reg` 三件套，只是把 2D 的 `wh` 换成了 3D 的 `dim + height + rot`。也对应 YOLO 的 `bx = σ(tx) + cx`——**格子坐标 + 偏移**是同一个思想，YOLO 用 sigmoid 约束偏移在 [0,1)，CenterPoint 用 L1 loss 软约束。

---

**[01:54:17]** 原话：「然后的话是我们的长宽高。」
**[01:54:21]** 原话：「长宽高我们需要取一个 EXP。」
**[01:54:25]** 原话：「我们其实预测的……」
**[01:54:27]** 原话：「……预测的其实是预测长宽高的一个 Log。」
**[01:54:31]** 原话：「取完 Log 之后的一个结果。」
**[01:54:33]** 原话：「所以说真正……」
**[01:54:34]** 原话：「真正取它的长宽高要算一个 EXP。」

> 讲者在这里连说了 6 句，其实是同一件事的反复强调。**这是重点句，五角度全给。**

- 【直译】网络回归的不是「4.5 米长」，而是「ln(4.5) ≈ 1.504」。所以解码时必须做 `exp` 才能拿回真实米数。
- 【代码】`centerpoint_head.py` L1355–L1358：
  ```python
  if self.norm_bbox:
      batch_dim = torch.exp(preds_dict[0]['dim'])
  else:
      batch_dim = preds_dict[0]['dim']
  ```
  `self.norm_bbox` 是配置开关（mmdet3d 里同名同义），DenseBEV 显然开着。
- 【形状】`[1, 3, 448, 224] → [1, 3, 448, 224]`。3 = (l, w, h) 或 (w, l, h)——按后面 wiki 表的列序是 `3=w, 4=l, 5=h`，所以这里 dim 的 3 个通道顺序是 **(w, l, h)**。
- 【为什么】三个理由，按重要性排序：
  1. **保证正数**。`exp(·) > 0` 恒成立。如果直接回归米数，网络在训练初期完全可能吐出负的长度，box 就崩了（IoU、NMS、可视化全部出错）。
  2. **把"相对误差"变成"绝对误差"**。在 log 空间做 L1 loss，`|log(p) - log(g)| = |log(p/g)|`，惩罚的是**比例误差**。这意味着 2 米长的行人和 12 米长的公交车，被要求达到同样的"百分比精度"，而不是同样的"米数精度"。若在线性空间做 L1，网络会为了压 loss 而把注意力全砸在大车上（大车的绝对误差天然大），小目标尺寸就学不好。
  3. **动态范围压缩**。VRU 的 h≈1.7 m 到卡车的 l≈16 m，跨了近一个数量级；取 log 后落在 [0.5, 2.8] 这个窄区间，对回归头的初始化和学习率友好得多。
- 【连接】
  - **YOLO（用户最熟）**：YOLOv2 之后所有版本都是 `bw = pw · exp(tw)`，`pw` 是 anchor 先验宽。CenterPoint 没有 anchor，等价于 **`pw ≡ 1`**（或者说先验被并进了 bias）。所以 CenterPoint 的 `dim` 头本质是「无 anchor 的 log-scale 回归」，你可以把它理解成 anchor-free 版的 YOLO wh 头。
  - **Faster R-CNN**：`t_w = log(w/w_a)` 同源，这套 log 参数化从 R-CNN 一路传到今天。
  - **BEVFusion / mmdet3d**：`norm_bbox=True` 是 nuScenes 配置的默认值，所以你在 BEVFusion 里跑出来的 `dim` 分支也是 log 空间的。**如果你以后自己写可视化脚本、忘了 exp，会看到一堆"边长 1.5 米的卡车"——这是新人最常踩的坑之一。**
  - **训练侧对应**：Ch11 里目标级 10 维 L1 loss 的 `anno_box` 拼接顺序是 `cat([reg, height, dim, rot, vel], dim=1)`（帧证：`centerpoint_head.py:1021-1025`），其中的 `dim` 就是 GT 取过 log 的值。解码这边 exp、训练那边 log，两边必须严格配对。

---

**[01:54:38]** 原话：「然后这里的话是取出 sinθ 和 cosθ。」

- 【直译】朝向角不是直接回归一个角度值，而是回归它的正弦和余弦两个分量。
- 【代码】`centerpoint_head.py` L1360–L1369（画面 01:54:38 帧，L1360 被高亮）：
  ```python
  if self.enable_corner_det:
      batch_rots       = preds_dict[0]['rot_long'][:, 0].unsqueeze(1)
      batch_rotc       = preds_dict[0]['rot_long'][:, 1].unsqueeze(1)
      batch_rots_short = preds_dict[0]['rot_short'][:, 0].unsqueeze(1)
      batch_rotc_short = preds_dict[0]['rot_short'][:, 1].unsqueeze(1)
  else:
      batch_rots       = preds_dict[0]['rot'][:, 0].unsqueeze(1)
      batch_rotc       = preds_dict[0]['rot'][:, 1].unsqueeze(1)
      batch_rots_short = None
      batch_rotc_short = None
  ```
- 【形状】`preds_dict[0]['rot']` 是 `[1, 2, 448, 224]`；`[:, 0]` 切出来是 `[1, 448, 224]`（少了通道维），`.unsqueeze(1)` 补回来变成 `[1, 1, 448, 224]`。**这个 `.unsqueeze(1)` 不是可有可无的**——后面 `_transpose_and_gather_feat` 要求输入必须是 4 维 `[B, C, H, W]`。
- 【为什么】为什么不直接回归 θ？因为角度有**周期性断点**：`-179.9°` 和 `+179.9°` 在物理上只差 0.2°，但数值上差 359.8°。用 L1/L2 loss 直接回归 θ，网络在这个断点附近会收到巨大的假梯度，永远学不好。改回归 `(sinθ, cosθ)` 后，函数在整个圆周上连续可微，断点消失。
- 【为什么·补充】`enable_corner_det` 分支里的 `rot_long` / `rot_short` 是这套代码的**特色**：对长条形目标（卡车、公交、挂车），车头朝向（long edge）和车身横向（short edge）分别用一组 sin/cos 回归，最后能交叉校验。普通 `center` 模式下 `rot_short` 全 None。
- 【连接】
  - **CenterPoint 原版**：`rot` 头就是 2 通道 `(sin, cos)`，解码用 `torch.atan2(rot_sine, rot_cosine)`——本章 Part 12-8 会看到一模一样的代码。
  - **PointPillars / SECOND**：走的是另一条路——直接回归 θ 但加一个 `dir_cls` 二分类头解决 180° 歧义。DenseBEV **两条路都走了**：既有 sin/cos，又有 `dir_cls`（Part 12-8 详解）。
  - **智谷课程的卷积知识**：这里每个分支都只是 `Conv2d(C_in, C_out, 3, padding=1)` 级别的浅头，真正的"智能"全在共享的 BEV backbone 里。10 个分支 = 10 个并列的小卷积头，参数量占比极低。

---

**[01:54:44]** 原话：「这些全是……然后的话，对，这个是速度。」

- 【代码】`centerpoint_head.py` L1371–L1374：
  ```python
  if 'vel' in preds_dict[0]:
      batch_vel = preds_dict[0]['vel']     # [1, 2, 448, 224]
  else:
      batch_vel = None
  ```
- 【形状】2 通道 = (vx, vy)，单位 m/s，自车坐标系下的**绝对速度**（DenseBEV 有 3 帧时序输入，速度是从时序特征里学出来的）。
- 【为什么】用 `in` 判断而不是 `.get()`，是因为下游要用 `batch_vel is not None` 做分支控制（L1426 决定要不要往 15 维的最后两列填 vx/vy）。这种"配置驱动的可选分支"在多任务头里非常常见。
- 【连接】nuScenes 的官方指标里有 mAVE（平均速度误差），所以 nuScenes 系的检测头基本都带 vel 分支。你在 BEVFusion 的 `code_size=9`（x,y,z,w,l,h,rot_sin,rot_cos,vx,vy 其实是 10，官方 bbox_coder 是 9 或 10 视配置）里见到的就是同一件事。

---

**[01:54:50]** 原话：「然后这些全部都是在 BV feature 上的。」

- 【直译】再次强调：到这一步为止，所有变量都还是 448×224 的**图**，没有任何一个是"目标"。
- 【为什么】讲者反复强调这点，是因为学习这套代码最容易犯的错就是「以为 reg / dim / rot 这些分支只在有目标的地方有值」。**不是的——每一个 BEV 格子都预测了一整套完整的框参数**（448×224 = 100352 套），只不过绝大多数是垃圾，靠 heatmap 分数来挑。这就是"dense prediction"的含义，也是 DenseBEV 名字里 "Dense" 的由来。
- 【连接】训练时对应的是 Ch11 讲的 dense 属性 loss：只在 GT 中心那一个（或高斯核覆盖的几个）像素上加监督，其余像素的属性预测**不受任何约束**（loss mask 掉了）。所以那些格子的 dim/rot 数值是完全没意义的随机值——它们只是从来没人管过而已。

---

**[01:54:57]** 原话：「然后这个是动静。」
**[01:54:58]** 原话：「然后朝向。」
**[01:55:00]** 原话：「所以分别取出来。」

- 【代码】`centerpoint_head.py` L1376–L1389（帧 01:54:49 / 01:55:02 完整可读）：
  ```python
  batch_movement = None
  if 'movement' in preds_dict[0]:
      if self.activate_move_two_stg:
          batch_mov_prob     = F.softmax(preds_dict[0]['mov_two_stage'], dim=1)
          batch_mov_prob_max = torch.max(batch_mov_prob, dim=1, keepdim=True).values
          batch_mov_class    = torch.argmax(batch_mov_prob, dim=1, keepdim=True)
          # tricky here, keep the class and prob at the same time by add them
          batch_movement     = batch_mov_prob_max + batch_mov_class
      else:
          batch_movement = preds_dict[0]['movement'].sigmoid()

  batch_direction = None
  if self.dir_cls_task:
      batch_direction = preds_dict[0]['dir_cls'].sigmoid()
  ```
- 【形状】两者都是 `[1, 1, 448, 224]`（console 实测：`movement torch.Size([1, 1, 448, 224])`、`dir_cls torch.Size([1, 1, 448, 224])`）。
- 【为什么·这是本章最精巧的一个小技巧】看那行英文注释：`# tricky here, keep the class and prob at the same time by add them`。
  - 两阶段动静分类（`activate_move_two_stg`）输出的是 `mov_two_stage` 多通道 logit，softmax 后得到每类概率。
  - 现在问题来了：**下游 15 维输出只给 movement 留了 1 列**，可类别 + 置信度是 2 个数。
  - 解法：`batch_movement = prob_max + class`。因为二分类 softmax 的最大概率必然落在 `[0.5, 1.0]`，所以：
    - class=0（静止）→ 值域 `[0.5, 1.0]`
    - class=1（运动）→ 值域 `[1.5, 2.0]`
  - 两段完全不重叠！下游只要 `cls = int(v)`（或 `v > 1.25`）就能拿回类别，`prob = v - cls` 就能拿回置信度。**用一个 float32 装了两条信息，且完全无损。**
  - 【风险提示】这个技巧的前提是「二分类」且「概率恒 ≥0.5」。如果哪天改成 3 分类，`prob_max` 最小可以到 1/3，`class=1` 时值域变成 `[1.333, 2.0]`、`class=2` 时 `[2.333, 3.0]`，`int()` 仍然可解；但如果 `prob_max` 能低于 0.5 且 class 从 0 开始，`class=0, prob=0.34` 与 `class=0, prob=0.99` 都在 `[0,1)` 内没问题——真正的坑是**如果解码端用 round() 而不是 int()/floor()**，`0.6` 会被 round 成 1，类别就错了。这是典型的"聪明代码"带来的维护风险。
- 【连接】这种"打包编码"在嵌入式/车端很常见（带宽和内存都金贵）。你在 nuScenes 的 attribute 预测里见不到这种做法，因为学术代码不缺内存。这正是**工业代码和学术代码的气质差别**，也是你从"跑通 BEVFusion"到"读懂产线代码"要迈过的坎。

---

**[01:55:01]** 原话：「然后在 CF BoxCoder Decode 里面去解析 Box。」

- 【直译】备料完毕，正式调用 box coder 的 decode 函数。（"CF" 是 "CenterPoint" 的转写残差。）
- 【代码】`centerpoint_head.py` L1391–L1405，帧 01:55:13 完整可读：
  ```python
  temp, origin_bbox_preds = self.bbox_coder.decode(
      used_bev_feat,
      fusion_feat,
      batch_heatmap,
      batch_rots,
      batch_rotc,
      batch_hei,
      batch_dim,
      batch_vel,
      reg=batch_reg,
      task_id=task_id,
      movement=batch_movement,
      direction=batch_direction,
      rot_short_sin=batch_rots_short,
      rot_short_cos=batch_rotc_short)
  ```
- 【形状】14 个入参，前 8 个是位置参数、后 6 个是关键字参数。返回两个东西：`temp`（list，长度 = batch，每项是一个 dict）和 `origin_bbox_preds`（`[1, 256, 10]` 的原始 box 张量，**未经范围过滤**）。
- 【为什么】把 `used_bev_feat` 和 `fusion_feat` 也塞进 decode，是这套代码相对 mmdet3d 的**最大改动**。原版 `decode` 只管几何解码；这里额外在 decode 里"顺手"把每个被选中目标的 BEV 特征向量抠出来（instance embedding），供 tracker / e2e 用。Part 12-6 详讲。
- 【连接】`bbox_coder` 这个抽象来自 mmdetection 的设计哲学：编解码逻辑（GT→回归目标、回归输出→框）单独封装成一个类，head 只管前向和 loss。你在 BEVFusion 的 config 里写的 `bbox_coder=dict(type='CenterPointBBoxCoder', post_center_range=[...], max_num=500, score_threshold=0.1, ...)` 就是在配这个类。

---

**[01:55:17]** 原话：「对，在这里会首先这个 heatmap 了就是我们分类的一个结果，是 1 乘 5 乘以 448 乘 224。」
> ⚠ 转写原文作「1乘5乘以48乘24」——Whisper 把连读的 "四百四十八" / "二百二十四" 丢了音节。本章凡遇 `48乘24`、`4824`、`5乘4乘24` 一律按 **448×224** 还原，依据见 Part 12-1 的 100352 交叉验证。

- 【代码】进入 `centerpoint_bbox_coders.py`，`decode` 的头两行（L145–L147）：
  ```python
  batch, cat, _, _ = heat.size()
  scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)
  ```
- 【形状】`batch=1, cat=5`，H/W 用 `_` 丢弃（decode 里不需要，`_topk` 内部会自己取）。
- 【为什么】`_` 丢弃 H、W 是个小信号：**decode 这一层不关心 BEV 分辨率**，分辨率相关的换算全部延后到 L194–L195 用 `pc_range / voxel_size / out_size_factor` 做（Part 12-8）。这种"关注点分离"让同一个 coder 能服务不同分辨率的配置。
- 【连接】`self.max_num` 就是"最多输出多少个目标"，DenseBEV = 256。mmdet3d 的 nuScenes 配置里通常是 500。这个数是**纯工程取舍**：太小会漏检密集场景，太大会拖慢后处理和 tracker。

---

### 🔨 动手练习 ch12-2：复现"dim 取 log 训练 / 取 exp 解码"的完整闭环

```python
import torch

# ---------- 训练侧：GT 取 log ----------
gt_dim = torch.tensor([[4.6, 1.9, 1.6],     # 轿车 l,w,h
                       [12.0, 2.6, 3.4],    # 公交
                       [0.6, 0.6, 1.7]])    # 行人
gt_log = torch.log(gt_dim)
print("GT 取 log 后 :\n", gt_log)
# 预期：tensor([[1.5261, 0.6419, 0.4700],
#               [2.4849, 0.9555, 1.2238],
#               [-0.5108, -0.5108, 0.5306]])
# 注意动态范围：线性空间 0.6~12（20倍），log 空间 -0.51~2.48（区间宽度仅 3）

# ---------- 网络输出（模拟"每个都差 0.1 的 log 误差"）----------
pred_log = gt_log + 0.1
pred_dim = torch.exp(pred_log)                      # 解码：exp 还原
print("\n解码后的尺寸 :\n", pred_dim)
print("相对误差     :\n", (pred_dim - gt_dim) / gt_dim)
# 预期：三行相对误差全是 0.1052（=e^0.1-1），与目标大小无关
#      —— 这就是"log 空间 L1 = 惩罚相对误差"的直接证据

# ---------- 反例：若在线性空间同样差 0.1 米 ----------
pred_lin = gt_dim + 0.1
print("\n线性空间同样差0.1米时的相对误差 :\n", (pred_lin - gt_dim) / gt_dim)
# 预期：轿车长 0.0217、公交长 0.0083、行人长 0.1667
#      —— 小目标被严重"欺负"，网络会学得偏向大目标
```

**你应该看到的**：log 空间的等误差 ⇒ 等相对精度；线性空间的等误差 ⇒ 小目标吃亏。这就是 `norm_bbox=True` 的全部理由。

### 🔨 动手练习 ch12-3：复现 movement 的"prob + class"打包/解包

```python
import torch, torch.nn.functional as F

B, H, W = 1, 4, 4
mov_two_stage = torch.randn(B, 2, H, W)             # 2 类：0=静止 1=运动

prob      = F.softmax(mov_two_stage, dim=1)
prob_max  = torch.max(prob, dim=1, keepdim=True).values     # [B,1,H,W] ∈ [0.5,1]
cls       = torch.argmax(prob, dim=1, keepdim=True)         # [B,1,H,W] ∈ {0,1}
packed    = prob_max + cls                                   # ★ 打包

print("prob_max 范围 :", prob_max.min().item(), prob_max.max().item())  # 应 ≥ 0.5
print("packed  范围 :", packed.min().item(),   packed.max().item())

# ---------- 解包 ----------
cls_back  = packed.floor().long()      # 用 floor / int，绝不能用 round！
prob_back = packed - cls_back

print("类别还原正确 :", torch.equal(cls_back, cls))                       # True
print("概率还原误差 :", (prob_back - prob_max).abs().max().item())        # ~0 (1e-7)

# ---------- 反例：用 round 会翻车 ----------
bad = packed.round().long()
print("用 round 的错误率 :", (bad != cls).float().mean().item())
# 预期：1.0（即 100% 全错），不是 50%！算一下就知道为什么：
#   class=0 → packed = prob_max ∈ (0.5, 1.0]  → round → 1  ≠ 0  ✗
#   class=1 → packed = 1 + prob ∈ (1.5, 2.0]  → round → 2  ≠ 1  ✗
# 两类各自都被整体平移了 +1 —— 这正是 round 的"四舍五入到最近整数"和
# floor 的"向下取整"在 [n+0.5, n+1) 这个区间上行为相反造成的。
# ★ 换句话说：用 round 解包不是"有时候错"，而是"永远错、且错得很整齐"，
#   在联调时会表现为"动静判断整体反了/整体偏了一类"，很容易被误判成模型没训好。

# ---------- 再看一个边界：3 分类时 floor 还成立吗？----------
p3 = torch.tensor([[0.34], [0.50], [0.99]])      # 3 类时 prob_max 最小可到 1/3
c3 = torch.tensor([[0.],   [1.],   [2.]])
pk3 = p3 + c3
print("\n3分类 packed  :", pk3.flatten().tolist())   # [0.34, 1.50, 2.99]
print("3分类 floor 还原:", pk3.floor().flatten().tolist(), " 期望", c3.flatten().tolist())
# 预期：[0.0, 1.0, 2.0] 全对 —— 因为 prob_max ∈ (0,1]，floor 天然把整数位切出来。
# ★ 唯一真正会崩的情况是 prob_max 恰好 = 1.0：packed = class + 1，floor 会多算一类。
#   二分类 softmax 在 fp32 下 prob_max 达到 1.0 是可能的（另一路 logit 极小时溢出饱和）。
#   所以最鲁棒的解包是 (packed - 1e-6).floor()，或者干脆别用这种打包。
```

**你应该看到的**：`floor` 可以，`round` **100% 必挂**（不是概率性挂）；而且 `floor` 自己也有一个 `prob_max == 1.0` 的边界洞。工业代码里这种"聪明编码"必须配注释——所幸这份代码确实写了 `# tricky here`，但它没写清楚"解包必须用 floor"，这个知识只存在于写代码那个人的脑子里。**这就是"聪明代码"的真实成本。**

### 【小结】

1. 这一段是纯"备料"：把 10 个稠密分支从 dict 拆成局部变量，形状统统保持 `[1, C, 448, 224]`，**一个目标都还没选出来**。
2. 两处"还原"最关键：`dim` 必须 `torch.exp`（训练时 GT 取过 log，为的是正数保证 + 相对误差惩罚 + 动态范围压缩）；`rot` 拆成 sin/cos 两支（为的是消除角度周期性断点）。
3. `movement` 用 `prob_max + class` 把两条信息塞进一个 float——工业代码为省带宽的典型"聪明写法"，解包必须用 `floor` 不能用 `round`。
---

# Part 12-3｜第一重 topk：每类在 448×224 展平上取 256（01:55:26 – 01:56:35）

### 本段在讲什么

这是本章的**技术心脏**。`_topk` 这个 39 行的小函数要完成"从 501760 个 (类别, 格子) 候选里挑出 256 个最强目标，并且知道每个目标的类别、行、列"。它用了一个非常经典但第一次看必然懵的写法：**做两次 `torch.topk`**。这一 Part 讲第一重（类内 topk），下一 Part 讲索引拆解，再下一 Part 讲第二重（跨类 topk）。

- **输入**：`heat`，`[1, 5, 448, 224]`
- **第一重输出**：`topk_scores` `[1, 5, 256]`、`topk_inds` `[1, 5, 256]`
- **流水线位置**：`centerpoint_bbox_coders.py:82`

---

**[01:55:26]** 原话：「然后在这个里面会取 TopK，就是我们当前只会去回归 200……去取 256 个目标。」

- 【直译】讲者先说 200 又改口成 256——**以 256 为准**。这个数就是 `self.max_num`。
- 【代码】`bbox_coders.py:147` → `self._topk(heat, K=self.max_num)`，进入 `_topk`（L65）：
  ```python
  def _topk(self, scores, K=80):
      """Get indexes based on scores.

      Args:
          scores (torch.Tensor): scores with the shape of [B, N, W, H].
          K (int): Number to be kept. Defaults to 80.

      Returns:
          tuple[torch.Tensor]
              torch.Tensor: Selected scores with the shape of [B, K].
              torch.Tensor: Selected indexes with the shape of [B, K].
              torch.Tensor: Selected classes with the shape of [B, K].
              torch.Tensor: Selected y coord with the shape of [B, K].
              torch.Tensor: Selected x coord with the shape of [B, K].
      """
  ```
  注意默认值 `K=80`（CenterNet 的 COCO 默认），但实际调用传的是 `self.max_num=256`。
- 【形状】docstring 写的是 `[B, N, W, H]`，实际是 `[B, C, H, W]`——**这个 docstring 的 W/H 顺序是错的**（从 mmdet3d 抄来时就带着这个笔误）。⚠ 不要被它误导，实测 `scores.shape = [1, 5, 448, 224]`，448 是第 3 维（代码里叫 `height`），224 是第 4 维（代码里叫 `width`）。
- 【为什么 256】这是"召回上限"的工程取舍：
  - **太小**：城区密集车流一帧可能有 60–100 个目标，加上 heatmap 峰值不够尖时同一目标占多个格子，256 已经不算宽裕。
  - **太大**：后续所有 `_transpose_and_gather_feat` 的代价、以及 tracker 的关联代价都是 O(K)，车端算力有限。
  - 对比：mmdet3d nuScenes 默认 `max_num=500`；CenterNet COCO 默认 `K=100`；YOLO 是先按 conf 阈值筛再 NMS，没有硬性 topk 上限。
- 【连接】**为什么 CenterPoint 敢不用 NMS 就 topk？** 因为训练时 heatmap 的 GT 是一个高斯核（中心 1.0，向外按 `exp(-d²/2σ²)` 衰减），配合 Gaussian Focal Loss，网络被强烈鼓励只在中心那一个像素输出高分。理想情况下峰值极尖，topk 自然不会重复选中同一目标。**实际情况没这么理想**——这也是本章最后 jiazhiwei 说"现在稳定性差一些"的技术根源之一。

---

**[01:55:36]** 原话：「然后对当前的这个这里的（scores）就是我们刚刚说的取完 Sigmoid 的一个 heatmap。」
**[01:55:49]** 原话：「当前它的 SIG（shape）是 1 乘 5 乘 448 乘 224。」

- 【代码】`_topk` 的第一行实体代码，L80：
  ```python
  batch, cat, height, width = scores.size()
  ```
- 【形状】`batch=1, cat=5, height=448, width=224`。
- 【帧证】此刻画面下方的 Python Console 里，讲者刚敲过 `scores.shape`，输出 `torch.Size([1, 5, 448, 224])`。**这是第一手证据，不是我推断的。**
- 【为什么】`height=448` / `width=224` 的命名要牢牢记住，因为后面 `topk_inds % width`、`topk_inds / width` 全靠这两个名字，而且**这套代码把 x/y 的命名跟原版对调了**（Part 12-4 详解），是全章最容易搞错的地方。

---

**[01:55:52]** 原话：「就是然后在这里我们首先会取，首先会取一个第一个 TOP 256。」
**[01:56:00]** 原话：「会沿着……会把 448 和 224 给它展平。」
**[01:56:03]** 原话：「就是得到的结果是 1×5×100352。」 ⚠（转写作"15224"）

> **本章第一大重点句，五角度全给。**

- 【直译】把每个类别的那张 448×224 的图拍成一条长度 100352 的向量，然后**在每个类别内部**各挑分数最高的 256 个。
- 【代码】`bbox_coders.py:82`（画面里被黄条高亮）：
  ```python
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
  ```
- 【形状】完整链条：
  ```
  scores                    : [1, 5, 448, 224]
  scores.view(1, 5, -1)     : [1, 5, 100352]     ← 448 × 224 = 100352
  torch.topk(..., K=256)
    → topk_scores           : [1, 5, 256]         （实测 console：torch.Size([1, 5, 256])）
    → topk_inds             : [1, 5, 256]         （实测 console：torch.Size([1, 5, 256])）
  ```
  ⚠ 转写里的 "15224" 是 Whisper 把 "1、5、100352" 听糊了。用 448×224=100352 反推可以确定。
- 【为什么要先做"类内 topk"，而不是直接一次全局 topk？】这是最值得想清楚的一点，也是我要**修正一个流行误解**的地方。
  - 直接 `scores.view(1, -1).topk(256)` 在**数值上**也能给出全局最强 256 个 (类别, 格子) 对，然后 `class = idx // 100352`、`pos = idx % 100352` 也能解出来。
  - **先说结论：两步和一步在结果上严格等价。** 证明很短：全局 top-256 里属于类别 c 的元素最多 256 个，而它们必然是类别 c 内部分数最高的那些，所以一定被第一重的"类内 top-256"完整包含——第一重不会漏掉任何一个最终答案。（练习 ch12-4 会逐值对拍验证。）
  - **也不是为了省算力。** 一步是 `O(C·H·W)` 的一次选择；两步是 `O(C·H·W)` 加上额外的 `O(C·K)`（这里 `C·K = 1280`）再加三次 `gather`。**两步严格更贵**，只是贵得可以忽略。⚠ 网上（和我此前的草稿里）常见的"两步是为了省内存/省算力"的说法**站不住脚**：`torch.topk` 无论在哪种形状上跑，都要遍历同样多的元素，临时 buffer 的量级也由 `K` 和输入元素数决定，分不分两步不改变量级。
  - **那两步的真实理由是什么？三条，按可信度排序：**
    1. **要"类内名次"这个中间量。** `topk_clses = topk_ind // K` 这行之所以成立，完全依赖于"第二重是在 `[batch, cat*K]` 这个 class-major 排布上做的"。如果一步到位，类别得靠 `idx // (H*W)` 解，那是另一套写法——**不是不行，是这份代码没走那条路。**
    2. **历史沿革（最主要）。** 这段是 CenterNet `Objects as Points` 官方实现的 `_topk` 一字未改地传下来的：CenterNet → mmdet3d → 这份代码。CenterNet 当年这么写，后面所有人就都这么写。02:05:41 那句"有些东西也很久都没有变了"说的正是这类代码。
    3. **保留了"每类至少考察 K 个"的语义钩子。** 万一将来要加"每类保底名额"或"类内 NMS"，两步结构留了插入点；一步到位就得推倒重写。
  - **实践建议**：读的时候按"全局取 top256"来理解，写的时候按原样保留。**别为了"优化"把它改成一步——收益是零点几个微秒，风险是整条链路的数值行为变化。**
- 【连接】
  - **CenterNet 原文**（Objects as Points）附录里的 `_topk` 就是这个写法，`mmdet3d` 抄过来，DenseBEV 又抄过来。你在 `mmdet3d/models/task_modules/coders/centerpoint_bbox_coders.py` 里能找到几乎逐字相同的代码。
  - **YOLO 对照**：YOLO 的做法是 `conf > thresh` 筛 + 类内 NMS，没有 topk。两者哲学差异：YOLO 靠 IoU-NMS 去重，CenterPoint 靠"高斯核 + 尖峰"去重。**在 BEV 下 CenterPoint 更优**，因为 BEV 里目标不重叠（俯视图物理上不能穿模），NMS 的价值大打折扣，而 topk 对 GPU 和车端芯片都极友好（固定输出长度，无动态 shape）。**这一点是面试常考。**

---

**[01:56:08]** 原话：「就是每一个类别上会去取 Top 256 个。」
**[01:56:13]** 原话：「然后得到的只是 1×5×256。」

- 【形状】强调输出 `[1, 5, 256]`——**5 个类别各自的 256 个**，总共 1280 个候选。console 实测 `topk_scores.shape → torch.Size([1, 5, 256])`。
- 【为什么】此时还不是最终答案：如果 5 个类别都各留 256 个，总数 1280 > 256。**必须再压一次**，这就是第二重 topk 的必要性。
- 【连接】想象一个极端场景：一帧图里全是车（第 0 类），没有行人。第一重 topk 会给行人类别也强行凑 256 个（分数可能只有 0.501），这些都是垃圾。第二重全局 topk 会把它们全部淘汰，让 256 个名额全给车。**这就是"跨类竞争"的意义**——名额是共享的，不是每类固定配额。

---

**[01:56:19]** 原话：「然后对应的话就是每这 256……」
**[01:56:23]** 原话：「……就是置信度最高的 256 个。」
**[01:56:25]** 原话：「然后对应的在这个展平的这个 heatmap 上它的一个索引。」

- 【直译】`torch.topk` 返回两个东西：分数（values）和**位置**（indices）。位置是在"拍平后的 100352 长向量"里的下标。
- 【代码】`topk_scores, topk_inds = torch.topk(...)`，`topk_inds ∈ [0, 100351]`。
- 【形状】`topk_inds : [1, 5, 256]`，dtype 是 `int64`。
- 【为什么】**索引才是解码的关键，分数只是排序的副产品。** 后面所有属性（reg / dim / rot / vel / …）都要靠这个索引去各自的图上"捞"。可以把 `topk_inds` 理解成 256 张"提货单"。
- 【连接】这跟 nuScenes 数据链里 `get_targets` 生成的 `ind = center_y * feature_map_size_x + center_x` 是**同一套编码**——训练时把 GT 中心编码成扁平索引存进 `ind`，推理时把预测峰值解码成扁平索引。两边必须用同一个 `width` 才对得上，这也是为什么 `_topk` 里用 `height/width` 而不是 `H/W` 硬编码。

---

**[01:56:35]** 原话：「然后这个会除一个 448×224。」 ⚠
**[01:56:39]** 原话：「所以这个区间都是在 448×224 上的。」

> **⚠ 这里讲者口误了，必须纠正。**

- 【原话问题】他说"除"，但代码是**取余**（`%`），不是除法。
- 【代码】`bbox_coders.py:84`：
  ```python
  topk_inds = topk_inds % (height * width)
  ```
- 【形状】`[1, 5, 256] → [1, 5, 256]`，数值不变。
- 【为什么这行其实是个 no-op（空操作）】仔细想：`topk_inds` 是在 `scores.view(batch, cat, -1)` 上做 topk 得到的，**最后一维长度就是 `height*width = 100352`**，所以索引天然满足 `0 ≤ topk_inds < 100352`，取余不改变任何值。
  - 那它为什么存在？**防御性/历史代码。** 在某些 CenterNet 变体里，前一步写的是 `scores.view(batch, -1)`（把 cat 也拍进去），此时索引范围是 `[0, C·H·W)`，必须 `% (H*W)` 才能拿到"格子内位置"。DenseBEV 这份代码用的是 `view(batch, cat, -1)`，所以这行就冗余了，但没人删。
  - **实践意义**：你自己写解码时如果把 `view(batch, cat, -1)` 改成 `view(batch, -1)`，这行就从"冗余"变成"必需"。留着它反而更鲁棒。
- 【连接】讲者说"这个区间都是在 448×224 上的"是对的——他想表达的是"索引的取值范围被限定在单张 BEV 图的格子数之内"，这个理解没问题，只是把 `%` 说成了"除"。**⚠ 请以代码为准。**

---

**[01:56:43 / 01:56:57]** 原话：「然后对应该是在这个……其实它是在 448×224 展平之后的这个位置。」

- 【直译】再次确认：`topk_inds` 是"拍平后的一维坐标"，不是二维行列。
- 【代码】数学关系（记住这条，后面全靠它）：
  ```
  flat_index = row * width + col          # width = 224
  row = flat_index // width               # ∈ [0, 447]
  col = flat_index %  width               # ∈ [0, 223]
  ```
- 【形状】一个 int64 标量 ↔ 一对 `(row, col)`。
- 【为什么用扁平索引而不是直接存 (row, col)？】因为 `torch.gather` 和 `torch.topk` 都只认单维索引。整套 CenterNet 系解码的设计就是"全程扁平索引 + 最后一步才还原行列"，这样所有属性图都能用同一份索引去 gather，代码高度统一。

---

### 🔨 动手练习 ch12-4：亲手验证"两步 topk == 一步全局 topk"

```python
import torch
torch.manual_seed(0)

B, C, H, W, K = 1, 5, 16, 8, 6          # 用小尺寸方便打印；真实是 1,5,448,224,256
scores = torch.rand(B, C, H, W)

# ---------- 方案 A：代码里的两步 topk ----------
topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), K)   # [B,C,K]
print("第一重 topk_scores:", tuple(topk_scores.shape))          # (1, 5, 6)
print("第一重 topk_inds  :", tuple(topk_inds.shape))            # (1, 5, 6)

topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), K)   # [B,K]
cls_A  = (topk_ind / torch.tensor(K, dtype=torch.float)).int()  # 类别 = ind // K
# 用 gather 把类内扁平索引取回来
flat_inds = topk_inds.view(B, -1, 1)                            # [B, C*K, 1]
idx       = topk_ind.unsqueeze(2).expand(B, K, 1)               # [B, K, 1]
pos_A     = flat_inds.gather(1, idx).view(B, K)                 # [B, K]

# ---------- 方案 B：一步全局 topk ----------
gscore, gind = torch.topk(scores.view(B, -1), K)                # 在 C*H*W 上直接选
cls_B = gind // (H * W)
pos_B = gind %  (H * W)

print("\n分数一致 :", torch.allclose(topk_score, gscore))       # True
print("类别一致 :", torch.equal(cls_A.long(), cls_B))           # True
print("位置一致 :", torch.equal(pos_A, pos_B))                  # True
print("\nA: cls", cls_A.tolist(), "pos", pos_A.tolist())
print("B: cls", cls_B.tolist(), "pos", pos_B.tolist())
```

**你应该看到的**：三行 `True`。**两步 topk 与一步全局 topk 完全等价**——所以这段代码的复杂度纯粹是历史包袱 + 中间量复用需求，理解时可以直接按"全局取 top256"来想，写代码时按原样保留即可。

> ⚠ **这个练习有一个前提**：分数不能有并列。`torch.rand` 下并列概率约等于 0，所以三行都是 `True`。但**真实推理时并列是常态**——本视频里这个未训练的 checkpoint，heatmap 全是 0.502x，fp32 下大量格子分数完全相同。此时 `torch.topk` 的 tie-breaking 在 CPU / CUDA / 不同 PyTorch 版本之间**不保证一致**，两步和一步选出来的"并列者"可能不是同一批（但分数集合仍然相同）。
> **这有两个现实后果**：(1) 你在 CPU 上 debug 出来的 256 个框，和 GPU 上跑出来的可能不完全一样，别以为是 bug；(2) 一个训得好的模型 heatmap 峰值分明、并列极少，所以这个问题在生产环境不显眼——**它只在"没训好的模型"上显形**，正好就是本视频演示的这种情况。

### 🔨 动手练习 ch12-5：验证 `% (height*width)` 确实是空操作

```python
import torch
B, C, H, W, K = 1, 5, 448, 224, 256
scores = torch.rand(B, C, H, W)

topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), K)
before = topk_inds.clone()
after  = topk_inds % (H * W)

print("H*W               =", H * W)                       # 100352
print("索引最大值        =", before.max().item())          # < 100352
print("取余前后完全相同  :", torch.equal(before, after))    # True  ← 空操作实锤

# 反例：如果第一步写成 view(B, -1)（把类别也拍进去），取余就是必需的
_, inds2 = torch.topk(scores.view(B, -1), K)
print("\nview(B,-1) 时索引最大值 =", inds2.max().item())    # 可达 501759 (=5*100352-1)
print("此时必须取余才能得到格子位置 :", (inds2 % (H*W)).max().item() < H*W)  # True
```

### 【小结】

1. 第一重 topk 是 `torch.topk(scores.view(batch, cat, -1), K)`：**先把每类的 448×224 拍成 100352 长向量，再在每类内部各取 256 个**，输出 `[1,5,256]` 的分数与索引。
2. `topk_inds % (height*width)` 在当前写法下是**空操作**，是从 CenterNet 抄来的防御性代码；讲者把 `%` 口述成了"除"，⚠ 以代码为准。
3. 两步 topk 与一步全局 topk 结果完全等价；保留两步是历史沿革 + 需要"类内扁平索引"这个中间量给后面的 `_gather_feat` 用。

---

# Part 12-4｜扁平索引 → 行/列，以及那个被对调的 x/y（01:57:04 – 01:57:24）

### 本段在讲什么

拿到扁平索引后要还原成二维格子坐标。这套代码在这里做了一件**与 CenterNet 原版相反**的事，而且把原版代码注释掉了留在旁边——这是理解 DenseBEV 坐标系的钥匙。

- **输入**：`topk_inds` `[1, 5, 256]`（值域 `[0, 100352)`）
- **输出**：`topk_xs` / `topk_ys` `[1, 5, 256]`（float）
- **流水线位置**：`centerpoint_bbox_coders.py:86–93`

---

**[01:57:04]** 原话：「然后这里会除上我的这个……」
**[01:57:06]** 原话：「……heatmap 的一个宽，就是代表它在这个 448×224 上的一个……」
**[01:57:12]** 原话：「……哪一行。」
**[01:57:13]** 原话：「然后 Y 呢是……」
**[01:57:15]** 原话：「……对宽度取一个余。」
**[01:57:18]** 原话：「就是它在……」
**[01:57:21]** 原话：「……呃，哪一列上的。」

> **本章第二大重点句。五角度全给，而且要专门讲清那个"x 是行、y 是列"的反直觉命名。**

- 【直译】索引除以宽度 224 得到"第几行"，索引对 224 取余得到"第几列"。**行 → 叫 x，列 → 叫 y。**
- 【代码】`bbox_coders.py:86–93`，帧 01:56:20 / 01:57:02 可完整看清，**包括被注释掉的两行**：
  ```python
  # topk_ys = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()
  # topk_xs = (topk_inds % width).int().float()

  # original:
  topk_xs = (
      (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()
  )
  topk_ys = (topk_inds % width).int().float()
  ```
- 【形状】`[1, 5, 256] (int64) → [1, 5, 256] (float32)`。中间的 `.int()` 是在做**向零取整的整数除法**，`.float()` 是为了后面能和 `reg` 偏移相加。
- 【为什么 x 是行、y 是列？——这是本章最重要的坐标系知识点】
  - 注意那两行**被注释掉的代码**（L86–L87）：`# topk_ys = inds/width`、`# topk_xs = inds%width`。这个命名与 CenterNet / mmdet3d 原版一致（图像习惯：y 是行、x 是列）。
  - **下面生效的代码（L90–L93）把两者对调了**：`topk_xs` 拿行、`topk_ys` 拿列。
  - 【⚠ 关于 `# original:` 这个标记——我把帧 01_57_02 放大逐行核对后修正了此前的判断】：`# original:`（L89）**紧贴在生效块的正上方**，按注释的常规归属，它标注的是**它下面那一块**，也就是说 —— **生效的 `xs=行 / ys=列` 才是这份仓库的"原始版本"（继承自 SPAS/上一代内部代码），而 L86–L87 那两行被注释掉的、看起来像 mmdet3d 原版的写法，是后来有人想改回图像习惯、试了一下又禁用掉的。**
  - 这个方向很重要：它说明 **x=行 不是一次"迁移时改错又将错就错"的意外，而是这条代码谱系从一开始就采用的自有约定**；后来那次想对齐 mmdet3d 的尝试反而被否决了（大概率是因为改了会打断整条链路的兼容性——正好呼应 02:05:41 那句"有些东西也很久都没有变了"）。
  - 无论 `# original:` 到底指谁，**结论都一样、也都以生效代码为准：`topk_xs` = 行，`topk_ys` = 列。**
  - **为什么要对调？** 因为 DenseBEV 的 BEV 张量布局是：
    ```
    dim=2 (height=448) ←→ 自车坐标系的 x 轴（前后，前 95.4 m + 后 83.8 m = 179.2 m）
    dim=3 (width =224) ←→ 自车坐标系的 y 轴（左右，±44.8 m = 89.6 m）
    ```
    验算：`448 × 0.4 = 179.2` ✓ ；`224 × 0.4 = 89.6` ✓。**完全对上。**
  - 所以"第几行"物理上就是"纵向第几格"= x 方向；"第几列"就是"横向第几格"= y 方向。命名跟着物理走，不跟着图像习惯走。
  - 【⚠ 踩坑警告】如果你直接把 mmdet3d 的 `centerpoint_bbox_coders.py` 拷过来用在这套 BEV 布局上，x/y 会整体转置 90°，可视化会看到所有车"横着开"。这类 bug 在跨仓库迁移时极其常见。
- 【为什么用 `float() / tensor(width, float)` 而不是 `//`？】
  - `topk_inds` 是 int64，`topk_inds // width` 在老版本 PyTorch 里会触发 `__floordiv__` 的 deprecation warning，且在某些导出后端（ONNX/TensorRT）上算子支持不佳。
  - 转成 float 做真除法再 `.int()` 截断，是**为了导出友好**的写法。`torch.tensor(width, dtype=torch.float)` 而不是 python int，同样是为了让 tracing 把它固化成常量张量。
  - **精度隐患**：float32 只能精确表示到 2²⁴ = 16777216 的整数。这里最大索引 100351，远小于阈值，安全。但如果 BEV 分辨率涨到 4096×4096 = 16.7M，就正好卡在边缘了——⚠ 值得记一笔。
- 【连接】
  - **训练侧的镜像操作**：`get_targets` 里生成 `ind` 时用的是 `ind = center_y_int * feature_map_size[0] + center_x_int`（mmdet3d 写法）。这套代码里等价于 `ind = row * width + col`。**推理解码和训练编码必须用同一个 `width`，否则整个坐标全乱。**
  - **智谷课程的卷积基础**：为什么 `H=448` 在 dim=2、`W=224` 在 dim=3？因为 PyTorch 的 `Conv2d` 约定 `[N, C, H, W]`，BEV 特征图是被 2D 卷积（BEV UNet backbone）处理的，必须遵守这个布局。所以"BEV 的 x 轴"被迫放在 H 位。
  - **nuScenes 对照**：nuScenes 的 `point_cloud_range=[-54, -54, -5, 54, 54, 3]` 是正方形对称的，x/y 弄反了也不容易看出来（只是旋转 90°）。DenseBEV 是 448×224 的长方形，**弄反了会直接 shape mismatch 报错**——这反而是好事。

---

### 🔨 动手练习 ch12-6：扁平索引 ↔ 行列 的双向验证（含 x/y 对调）

```python
import torch

H, W = 448, 224                      # DenseBEV 的 BEV 网格
RES  = 0.4                           # 米/格
X_FRONT, X_BACK = 95.4, 83.8         # 前 / 后
Y_HALF = 44.8                        # 左右各

print("纵向覆盖 :", H * RES, "m  (应 =", X_FRONT + X_BACK, ")")   # 179.2 == 179.2 ✓
print("横向覆盖 :", W * RES, "m  (应 =", 2 * Y_HALF, ")")         # 89.6  == 89.6  ✓

# ---- 造几个已知格子 ----
rows = torch.tensor([0, 100, 223, 447])
cols = torch.tensor([0,  60, 112, 223])
flat = rows * W + cols
print("\nflat index :", flat.tolist())     # [0, 22460, 50064, 100351]
#   0*224+0    = 0
# 100*224+60   = 22460
# 223*224+112  = 50064
# 447*224+223  = 100351  ← 正好是最大合法索引 = H*W-1 = 100351

# ---- 按代码的写法还原（注意 xs=行, ys=列）----
topk_xs = (flat.float() / torch.tensor(W, dtype=torch.float)).int().float()   # 行
topk_ys = (flat % W).int().float()                                            # 列
print("还原 xs(行) :", topk_xs.tolist(), " 期望", rows.tolist())
print("还原 ys(列) :", topk_ys.tolist(), " 期望", cols.tolist())
assert torch.equal(topk_xs.long(), rows) and torch.equal(topk_ys.long(), cols)
print("✓ 行列还原正确")

# ---- 反例：用 CenterNet 原版（被注释掉的那两行）会怎样 ----
wrong_ys = (flat.float() / torch.tensor(W, dtype=torch.float)).int().float()
wrong_xs = (flat % W).int().float()
print("\n原版命名下 xs =", wrong_xs.tolist(), "（其实是列！）")
print("原版命名下 ys =", wrong_ys.tolist(), "（其实是行！）")
print("→ 若不改名直接送进后面的 pc_range 换算，x/y 会整体转置 90°")
```

**你应该看到的**：`179.2` 与 `89.6` 两个数字精确对上讲义里的"前95.4/后83.8/左右±44.8、分辨率0.4"，这是判定"H 对应 x、W 对应 y"的铁证。

### 【小结】

1. 扁平索引还原二维用的是最朴素的 `row = idx // W`、`col = idx % W`，但**这套代码把 x 绑到行、y 绑到列**，和 CenterNet/图像习惯相反；`# original:` 标记（帧 01_57_02）贴在生效块正上方，说明"x=行"才是这条代码谱系的自有原版，被注释掉的 L86–L87 反而是后来对齐 mmdet3d 未遂的改动。
2. 原因是 BEV 张量的 `H=448` 对应自车纵向（179.2 m），`W=224` 对应横向（89.6 m）；两个尺寸乘 0.4 m 精确等于 BEV 范围，**这是判定坐标轴归属的硬证据**。
3. 用 `float 除法 + .int()` 而不是 `//`，是为了 ONNX/TensorRT 导出友好；在当前 100352 的量级下 float32 精度完全够用。
---

# Part 12-5｜第二重 topk：5×256 → 256，以及"两个 index"的区别（01:57:24 – 01:59:48）

### 本段在讲什么

第一重 topk 给了每类 256 个候选，一共 1280 个。这一段要在这 1280 个里选出全局最强的 256 个，同时解出**类别**，并把"类内扁平索引 / 行 / 列"这三样东西按新的排名重新排一遍。核心工具是 `_gather_feat`。

- **输入**：`topk_scores` `[1,5,256]`、`topk_inds` `[1,5,256]`、`topk_xs/topk_ys` `[1,5,256]`
- **输出**：`topk_score / topk_inds / topk_clses / topk_ys / topk_xs`，全部 `[1, 256]`
- **流水线位置**：`centerpoint_bbox_coders.py:95–103`

---

**[01:57:24]** 原话：「然后刚刚说的这个 TOP……」
**[01:57:28]** 原话：「TOPK 是过的是 1×5×256。」（转写作「TOPK是过的是15256」，"过"= "结果"）

- 【直译】他在做一个承上启下的复述：刚才第一重 topk 吐出来的东西，形状是 `[1, 5, 256]`。
- 【形状】`topk_scores : [1, 5, 256]`、`topk_inds : [1, 5, 256]`。console 实测两次都是这个值（帧 02_01_36 底部：`topk_scores.shape → torch.Size([1, 5, 256])`、`topk_inds.shape → torch.Size([1, 5, 256])`）。
- 【为什么要重申——这不是废话，是在为下一行做铺垫】接下来那行代码是 `topk_scores.view(batch, -1)`，把 `5` 和 `256` 拍成一维 `1280`。**拍平之前必须先在脑子里锁定这两维的先后顺序**，因为 PyTorch 是 row-major：外层维度变化慢、内层变化快。`[batch, cat, K]` 拍平后的排布必然是 **class-major**：
  ```
  [ 类0的第0..255名 | 类1的第0..255名 | 类2的… | 类3的… | 类4的… ]
     0 ..  255         256 .. 511
  ```
  这个顺序**唯一地决定了**后面那行 `topk_clses = topk_ind // K` 里除的是 `K=256` 而不是 `cat=5`。如果哪天有人把 head 改成 `[batch, K, cat]` 的排布（比如为了别的算子对齐），这行就必须同步改成 `// cat`，否则类别号会全乱且**不报错**。
- 【为什么是 `[1,5,256]` 而不是 `[5,256]`】batch 维哪怕只有 1 也必须留着：`_gather_feat` 内部的 `feats.gather(1, inds)` 是沿 `dim=1` 取的，它假定 `dim=0` 是 batch。挤掉 batch 维会让 gather 沿错误的轴工作——**这是"保留退化维度"在深度学习代码里的典型必要性**，和 numpy 里习惯性 squeeze 的风格正相反。
- 【连接】这一步的心智模型可以借用你熟悉的 YOLO：YOLO 的 `[B, anchors, H, W, 5+C]` 也是先 `view(B, -1, 5+C)` 拍成候选列表再筛。**"把空间维拍平成候选维、再在候选维上做选择"是所有稠密检测器后处理的共同骨架**，只是 YOLO 用阈值+NMS 筛、CenterPoint 用 topk 筛。

---

**[01:57:31]** 原话：「然后在这里会把 5 和 256 给它展成一维。」
**[01:57:38]** 原话：「然后相当于就是我在 5 个类别上去取一个 256。」
**[01:57:42]** 原话：「然后它出来的话就是 256 了。」

- 【直译】把 `[1, 5, 256]` 拍成 `[1, 1280]`，再取 top 256，得到 `[1, 256]`。
- 【代码】`bbox_coders.py:95`：
  ```python
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  ```
- 【形状】
  ```
  topk_scores            : [1, 5, 256]
  .view(1, -1)           : [1, 1280]         ← 1280 = 5 × 256
  torch.topk(..., 256)
    → topk_score         : [1, 256]           （console 实测：torch.Size([1, 256])）
    → topk_ind           : [1, 256]           值域 [0, 1280)
  ```
- 【为什么】这一步实现了**跨类竞争**：256 个名额在 5 个类别之间自由分配。极端情况下 256 个全给车也可以，全给行人也可以。这跟"每类固定分 51 个"完全不同，后者会在稀疏类别上浪费大量名额、在密集类别上漏检。
- 【⚠ 命名地雷】注意这里的变量名：**`topk_score`（单数）vs `topk_scores`（复数）**、**`topk_ind`（单数）vs `topk_inds`（复数）**。差一个 `s`，含义完全不同：
  | 变量 | 形状 | 含义 |
  |---|---|---|
  | `topk_scores` | `[1,5,256]` | 第一重：**类内** top256 的分数 |
  | `topk_score` | `[1,256]` | 第二重：**全局** top256 的分数（最终输出） |
  | `topk_inds` | `[1,5,256]`→`[1,256]` | **格子位置**（在 100352 上的扁平索引） |
  | `topk_ind` | `[1,256]` | **候选名次**（在 1280 上的索引），只是中间量 |
  这是 CenterNet 遗留的糟糕命名，读代码时一定要盯住那个 `s`。讲者 01:58:24–01:58:39 专门花了 15 秒解释这两个 index 的区别，就是因为这里最容易晕。
- 【连接】`torch.topk` 默认 `sorted=True`，所以 `topk_score` 是降序的，`topk_ind` 也按分数降序排列。这个性质后面很有用：**输出的 256 个框天然按置信度从高到低排好序**，下游 tracker 直接按序处理即可，不用再排。

---

**[01:57:45 / 01:57:48]** 原话：「这个……是过（果）。」
**[01:57:48]** 原话：「然后这个呢……」
**[01:57:49]** 原话：「TOPK index 呢……」
**[01:57:50 / 01:57:53 / 01:57:55 / 01:57:56 / 01:58:00]** 原话：「就是呃……它在……这个……5×256 上的一个……就是它的一个索引。」（此处讲者边找词边停顿，五个时间戳同属一句）

- 【直译】`topk_ind` 是在"1280 个候选组成的列表"里的下标，**不是**在 BEV 图上的位置。
- 【代码】`topk_ind ∈ [0, 1280)`，编码规则是 `topk_ind = class_id * 256 + rank_within_class`。
- 【形状】`[1, 256]`，int64。
- 【为什么】这是"索引的索引"（indirect index）。整个 `_topk` 的精妙之处就在这里：**第二重 topk 排的是"候选列表"，所以它的输出索引必须再经过一次 gather 才能变回"BEV 位置"。** 这就是接下来三行 `_gather_feat` 存在的全部意义。
- 【类比】想象一个两级选拔：
  - 第一级：5 个省各自选出 256 名（`topk_inds` 记录"这个人在省内考场的座位号"）。
  - 第二级：从 1280 人里选全国前 256（`topk_ind` 记录"这个人在 1280 人名单里排第几位"）。
  - 要知道全国前 256 名每个人的**考场座位号**，必须用 `topk_ind` 去 1280 人名单里查 `topk_inds`——这就是 `_gather_feat`。

---

**[01:58:06]** 原话：「然后这个所以在除上……」
**[01:58:09]** 原话：「256 就是代表它的分类的一个结果。」
**[01:58:13]** 原话：「就是它是属于……」
**[01:58:14]** 原话：「……哪一个类别。」

> **本章第三大重点句。注意转写里"除上5"与"256"被断在两行，容易误读成"除以 5"——⚠ 实际是除以 K=256。**

- 【直译】把候选名次除以 256（每类的候选数），商就是类别号。
- 【代码】`bbox_coders.py:96`（帧 01:57:31 高亮此行）：
  ```python
  topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
  ```
- 【形状】`[1, 256] → [1, 256]`，int32，值域 `[0, 5)`。
- 【为什么是 `// K` 而不是 `// 5`】关键在 `view` 的展开顺序。`topk_scores` 是 `[batch, cat, K]`，PyTorch 是 row-major（C 序），`view(batch, -1)` 后：
  ```
  index 0    .. 255   → 类别 0 的第 0..255 名
  index 256  .. 511   → 类别 1 的第 0..255 名
  index 512  .. 767   → 类别 2
  index 768  .. 1023  → 类别 3
  index 1024 .. 1279  → 类别 4
  ```
  所以 `class = index // 256 = index // K`。**如果 view 前是 `[batch, K, cat]`，才该除以 `cat=5`。** 讲者口述时把这两个数搅在了一起，⚠ 以代码为准：**除以 K（=256），不是除以类别数 5。**
- 【为什么用 float 除法 + `.int()`】同 Part 12-4：导出友好。`.int()` 是向零截断，对非负数等价于 `floor`，正确。
  - **精度边界**：`topk_ind` 最大 1279，float32 精确表示无虞。但如果 `K` 和 `cat` 都很大（比如 `cat=80, K=1000` → 索引到 80000），float32 依然安全（< 2²⁴）。这套写法的实际风险很低。
- 【连接】
  - **CenterNet 原版**写的是 `topk_clses = (topk_ind / K).int()`，Python 2 时代 `/` 是整除，Python 3 变成真除法后各家都补上了 `.int()`。
  - **YOLO 对照**：YOLO 的类别来自 `argmax(class_logits)`，是**每个候选框独立决定类别**；CenterPoint 的类别来自"它是在哪个类别的 heatmap 上被选中的"，是**位置和类别绑定**的。后者的一个副作用：**同一个 BEV 格子可能同时出现在两个类别的 top256 里**，于是输出两个位置完全相同、label 不同的框。center 模式没有跨类 NMS，这两个框都会被输出。⚠ 这是一个真实存在的行为，下游需要自己处理（或者靠 tracker 的关联逻辑吃掉）。

---

**[01:58:24 / 01:58:27]** 原话：「然后这里的 TOPK index……其实有两个。」
**[01:58:28]** 原话：「第一个 TOPK index 呢……」
**[01:58:30]** 原话：「……其实是代表它在每一个 feature map 上的一个索引。」
**[01:58:33]** 原话：「然后这个 TOPK index 呢……」
**[01:58:35]** 原话：「……是代表它在……」
**[01:58:37]** 原话：「……5 个类别上的一个索引。」

- 【直译】讲者自己也意识到这里绕，专门停下来把两个 index 摊开说。
- 【总结成表】（这张表建议背下来）
  | | `topk_inds`（复数 s） | `topk_ind`（单数） |
  |---|---|---|
  | 来自 | 第一重 `torch.topk` | 第二重 `torch.topk` |
  | 形状 | `[1, 5, 256]` → gather 后 `[1, 256]` | `[1, 256]` |
  | 值域 | `[0, 100352)` | `[0, 1280)` |
  | 语义 | **在 448×224 展平图上的格子位置** | **在 5×256 候选表里的名次** |
  | 用途 | 后面所有 `_transpose_and_gather_feat(attr_map, inds)` 都用它 | 只用来重排 `topk_inds/ys/xs`，之后就丢弃 |
  | 是否返回 | ✅ 作为 `inds` 返回，一路用到最后 | ❌ 函数内部临时量 |
- 【为什么这个区分至关重要】后面 `decode` 里有十几处 `self._transpose_and_gather_feat(xxx, inds)`，**全都用的是 `inds`（格子位置）**。如果误传 `topk_ind`，会去 100352 长的属性向量里取第 0..1279 个元素——不报错，但结果全是垃圾（只取了 BEV 图最前面那一小条）。**这是一个"静默错误"，极难 debug。**

---

**[01:58:39]** 原话：「然后在这里会通过 Gather 的一个方式是去取出来。」
**[01:58:44]** 原话：「然后代表就是取出来之后就能够代表这个 250 个目标……」 ⚠
**[01:58:49]** 原话：「……它具体是在哪一个位置上的。」

> ⚠ **数字勘误**：转写原文这里是「250 个目标」。**正确值是 256**。三重证据：(1) 讲者自己在 01:55:26 就说过"只会去回归 200……去取 256 个目标"；(2) console 实测 `topk_score.shape → torch.Size([1, 256])`；(3) 代码里 `K = self.max_num`，而 `pred_bboxes` 的容器是 `new_ones([batch_size, self.bbox_coder.max_num, self.code_dim])`，`max_num=256`。**讲者在这一小时里已经把 "256" 说成过 "200" 和 "250" 各一次——这是长时间口播的正常磨损，不是代码有两套配置。以 console 为准。**

- 【代码】`bbox_coders.py:97–99`（帧 01:58:27 高亮 L97）：
  ```python
  topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(
      batch, K
  )
  ```
- 【形状】详细拆解：
  ```
  topk_inds                    : [1, 5, 256]
  .view(batch, -1, 1)          : [1, 1280, 1]    ← 把它当成"1280 个长度为 1 的特征向量"
  _gather_feat(·, topk_ind)    : [1, 256, 1]     ← 按 topk_ind 挑出 256 个
  .view(batch, K)              : [1, 256]
  ```
- 【`_gather_feat` 到底做了什么】这个函数在 `bbox_coders.py:44` 定义，本章没完整展示函数体（讲者在 Ch11 01:50:02 用过它）。⚠ 以下是按 CenterNet / mmdet3d 标准实现还原的，函数签名 `def _gather_feat(self, feats, inds, feat_masks=None)` 由帧证确认：
  ```python
  def _gather_feat(self, feats, inds, feat_masks=None):
      dim   = feats.size(2)                                        # 特征维度 C
      inds  = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)   # [B,K] → [B,K,C]
      feats = feats.gather(1, inds)                                # 沿 dim=1 取
      if feat_masks is not None:
          feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
          feats = feats[feat_masks]
          feats = feats.view(-1, dim)
      return feats
  ```
  核心就一句 `feats.gather(1, inds)`：`out[b, k, c] = feats[b, inds[b, k, c], c]`。因为 `inds` 在 C 维上是 expand 出来的（同一份索引复制 C 遍），所以效果是"整行取出"。
- 【为什么不用高级索引 `feats[torch.arange(B)[:,None], inds]`？】
  - `gather` 是单一算子，ONNX/TensorRT 有原生支持（`GatherElements`）；高级索引会被 trace 成一堆 `Range`+`Expand`+`GatherND`，导出后性能差、有时还不支持。
  - **车端部署导向的代码几乎都用 `gather`。** 这是判断一份代码是"论文代码"还是"产线代码"的一个小指标。
- 【连接】这个 `_gather_feat` 在本章后面被 `_transpose_and_gather_feat` 包了一层（`bbox_coders.py:105`）反复调用十几次——它是 CenterNet 系解码的"万能提货器"。

---

**[01:58:54]** 原话：「然后一样的我的这个呃……」
**[01:59:02]** 原话：「这里的话也是通过 Gather 的方式去把……」
**[01:59:06]** 原话：「……对应的……」
**[01:59:10]** 原话：「……对应的哪一行哪一列就是取出来。」

- 【代码】`bbox_coders.py:100–101`（帧 01:58:51 高亮 L100）：
  ```python
  topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
  ```
- 【形状】和 `topk_inds` 一模一样：`[1,5,256] → [1,1280,1] → [1,256,1] → [1,256]`。
- 【为什么要重排 ys/xs】因为 `topk_ys/topk_xs` 是在**第一重排名**下算出来的（顺序是 class-major），而最终输出要用**第二重排名**（全局降序）。三个量必须同步重排，否则第 k 个框的分数、类别、位置会对不上。
- 【冗余提示】其实 `topk_ys/topk_xs` 完全可以在 gather 完 `topk_inds` 之后再从新的 `topk_inds` 现算：
  ```python
  topk_xs = (topk_inds // width).float()
  topk_ys = (topk_inds %  width).float()
  ```
  两行就够，省两次 gather。原代码先算后 gather 是 CenterNet 的历史写法。**面试时如果被问"这段代码能怎么优化"，这是一个标准答案。**

---

**[01:59:15]** 原话：「然后返回值的话，返回值就是……」
**[01:59:24]** 原话：「……置信度最高的 256 个目标。」
**[01:59:26]** 原话：「然后以及他的他在……」
**[01:59:34]** 原话：「……他在展平之后的 448×224 上的一个索引。」
**[01:59:37]** 原话：「然后以及……」
**[01:59:38]** 原话：「……分类的一个结果。」
**[01:59:39]** 原话：「然后以及对应的他属于哪一……」
**[01:59:43]** 原话：「……哪一列哪一行。」

- 【代码】`bbox_coders.py:103`：
  ```python
  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
  ```
  在 `decode` 里被接成：
  ```python
  scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)
  ```
- 【形状】五个都是 `[1, 256]`。dtype：`scores` float32、`inds` int64、`clses` int32、`ys/xs` float32。
- 【为什么 ys/xs 是 float 而不是 int】因为马上要加上 `reg` 亚像素偏移（`xs = xs + reg[...,0:1]`），必须是浮点。
- 【总结这 39 行】`_topk` 的完整流程（**这是本章最该背下来的一段**）：
  ```
  [1,5,448,224]
      │ view(1,5,-1)
  [1,5,100352]
      │ topk(K=256)  ← 第一重：类内竞争
  topk_scores[1,5,256], topk_inds[1,5,256]
      │ % (H*W)                       ← no-op
      │ xs = inds//224 (行), ys = inds%224 (列)
  topk_xs[1,5,256], topk_ys[1,5,256]
      │ topk_scores.view(1,-1) → [1,1280]
      │ topk(K=256)  ← 第二重：跨类竞争
  topk_score[1,256], topk_ind[1,256]
      │ clses = topk_ind // 256
      │ inds/ys/xs 各做一次 _gather_feat(·, topk_ind)
  → scores[1,256], inds[1,256], clses[1,256], ys[1,256], xs[1,256]
  ```

---

### 🔨 动手练习 ch12-7：从零手写 `_topk`，并与 PyTorch 逐值对拍

```python
import torch
torch.manual_seed(42)

def gather_feat(feats, inds):
    """CenterNet 的 _gather_feat（去掉 mask 分支）"""
    dim  = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    return feats.gather(1, inds)

def topk_centerpoint(scores, K):
    """完整复现 centerpoint_bbox_coders.py:65-103"""
    batch, cat, height, width = scores.size()

    # --- 第一重：类内 topk ---
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)                      # no-op
    topk_xs = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()  # 行
    topk_ys = (topk_inds % width).int().float()                                            # 列

    # --- 第二重：跨类 topk ---
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
    topk_inds  = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys    = gather_feat(topk_ys.view(batch, -1, 1),  topk_ind).view(batch, K)
    topk_xs    = gather_feat(topk_xs.view(batch, -1, 1),  topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


# ---------- 用 DenseBEV 真实尺寸跑一遍 ----------
B, C, H, W, K = 1, 5, 448, 224, 256
heat = torch.rand(B, C, H, W) * 0.02 + 0.49          # 模拟"未训练"的 0.5 附近
# 埋 3 个真目标
heat[0, 0, 250, 112] = 0.95     # 类0，正前方 (row=250, col=112)
heat[0, 2,  60,  40] = 0.90     # 类2
heat[0, 0, 251, 112] = 0.88     # 类0，紧挨着上面那个 —— 模拟"峰值不够尖"

s, inds, cls, ys, xs = topk_centerpoint(heat, K)
print("scores.shape :", tuple(s.shape))     # (1, 256)
print("inds.shape   :", tuple(inds.shape))  # (1, 256)
print("\nTop3 分数    :", s[0, :3].tolist())
print("Top3 类别    :", cls[0, :3].tolist())
print("Top3 行(xs)  :", xs[0, :3].tolist())
print("Top3 列(ys)  :", ys[0, :3].tolist())
print("Top3 扁平idx :", inds[0, :3].tolist())

# 校验：扁平索引 == 行*W + 列
assert torch.equal(inds[0, :3], (xs[0, :3] * W + ys[0, :3]).long())
print("\n✓ inds == row*W + col 成立")
# 预期输出：
#   Top3 分数 [0.95, 0.90, 0.88]
#   Top3 类别 [0, 2, 0]
#   Top3 行   [250.0, 60.0, 251.0]
#   Top3 列   [112.0, 40.0, 112.0]
#   —— 注意第 1 名和第 3 名只差一行！这就是"没有 NMS 时同一目标被重复输出"的真实演示
```

**你应该看到的**：
1. 形状链 `[1,5,448,224] → [1,256]` 完整跑通；
2. `inds == row*224 + col` 严格成立；
3. **第 1 名和第 3 名是相邻格子的同类目标**——这就是 center 模式下没有 NMS 的代价，实际部署里靠"高斯核训练让峰值够尖" + 下游 tracker 去重。

### 🔨 动手练习 ch12-8：验证"类别 = topk_ind // K"而不是 "// cat"

```python
import torch
B, C, K = 1, 5, 256
# 构造一个"第 3 类分数全场最高"的极端情况
topk_scores = torch.zeros(B, C, K)
topk_scores[0, 3, :] = 0.9          # 类 3 的 256 个候选全是 0.9
topk_scores[0, 0, :] = 0.8

topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), K)
cls_right = (topk_ind / torch.tensor(K, dtype=torch.float)).int()   # ✅ 除以 K
cls_wrong = (topk_ind / torch.tensor(C, dtype=torch.float)).int()   # ❌ 除以 cat

print("正确类别(除以K=256) 的取值集合 :", set(cls_right[0].tolist()))   # {3}
print("错误类别(除以C=5  ) 的取值集合 :", sorted(set(cls_wrong[0].tolist()))[:5], "...")
# 预期：正确的全是 3（因为 topk_ind ∈ [768,1024)，//256 = 3）
#      错误的会散成 153~204 这种荒谬的"类别号"
```

### 【小结】

1. 第二重 topk 在 `[1,1280]` 上选出全局最强 256 个，实现**跨类名额竞争**；输出天然按分数降序，下游不用再排。
2. **`topk_ind`（名次，值域 1280）和 `topk_inds`（格子位置，值域 100352）是两个东西**，差一个 `s`；后面所有属性 gather 都必须用 `inds`，用错了是静默错误。
3. `topk_clses = topk_ind // K`（除以 **256**，不是除以类别数 5），因为 `view(batch,-1)` 展开顺序是 class-major；讲者口述这里有歧义，⚠ 以代码为准。

---

# Part 12-6｜instance embedding：给 tracker / 端到端预留的"目标特征"（01:59:49 – 02:00:38）

### 本段在讲什么

`decode` 拿到 256 个格子位置之后，第一件事**不是**解框，而是先去 BEV 特征图上把这 256 个位置的特征向量抠出来。这是 DenseBEV 相对 mmdet3d 原版 CenterPoint 的最大改动，也是整个"端到端"（检测 → 跟踪 → 预测）串起来的接口。

- **输入**：`used_bev_feat` / `fusion_feat`（`[1, C, 448, 224]`）、`inds`（`[1, 256]`）
- **输出**：`instance_embeddings` / `fusion_instance_embeddings`（`[1, 256, C]`）
- **流水线位置**：`centerpoint_bbox_coders.py:149–154`

---

**[01:59:49]** 原话：「然后呢……」
**[01:59:50]** 原话：「这里对应的话就是通过 Gather 的一个方式……」
**[01:59:53]** 原话：「……把我的 feature 沿着 h 和 w……」
**[01:59:55]** 原话：「……展平。」
**[01:59:56]** 原话：「然后一些（用）对应的这个 index 去取出来。」

> **重点句，五角度全给。这是理解 `_transpose_and_gather_feat` 的唯一机会。**

- 【直译】把 `[B, C, H, W]` 的特征图先转成 `[B, H*W, C]`（每个格子一个 C 维向量），再用 256 个索引把 256 个向量抠出来。
- 【代码】`bbox_coders.py:149–154`（帧 01:59:28 / 02:00:02 高亮 L150）：
  ```python
  # 获取instance_embeddings
  instance_embeddings = self._transpose_and_gather_feat(used_bev_feat, inds)
  instance_embeddings = instance_embeddings.view(batch, self.max_num, used_bev_feat.shape[1])

  fusion_instance_embeddings = self._transpose_and_gather_feat(fusion_feat, inds)
  fusion_instance_embeddings = fusion_instance_embeddings.view(batch, self.max_num, fusion_feat.shape[1])
  ```
- 【`_transpose_and_gather_feat` 的实现】`bbox_coders.py:105`，docstring 在帧 01:57:02 可读：
  ```python
  def _transpose_and_gather_feat(self, feat, ind):
      """Given feats and indexes, returns the transposed and gathered feats.

      Args:
          feat (torch.Tensor): Features to be transposed and gathered
              with the shape of [B, 2, W, H].
          ind (torch.Tensor): Indexes with the shape of [B, N].

      Returns:
          ...
      """
      feat = feat.permute(0, 2, 3, 1).contiguous()      # [B,C,H,W] → [B,H,W,C]
      feat = feat.view(feat.size(0), -1, feat.size(3))  # → [B, H*W, C]
      feat = self._gather_feat(feat, ind)               # → [B, K, C]
      return feat
  ```
  ⚠ 函数体是按标准实现还原的（画面只完整展示了签名和 docstring）。**旁证**：Ch11 的 loss 部分（帧 01:50:08，`centerpoint_head.py:1045-1047`）把这三行**内联**写了出来，一字不差：
  ```python
  pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
  pred = pred.view(pred.size(0), -1, pred.size(3))
  pred = self._gather_feat(pred, ind)
  ```
  所以还原是可靠的。
- 【形状】以 `used_bev_feat` 为例（设通道数 C）：
  ```
  used_bev_feat            : [1, C, 448, 224]
  .permute(0,2,3,1)        : [1, 448, 224, C]
  .contiguous()            : 同上（但内存重排，让下一步 view 合法）
  .view(1, -1, C)          : [1, 100352, C]
  _gather_feat(·, inds)    : [1, 256, C]
  .view(1, 256, C)         : [1, 256, C]   （已经是这个形状，这行是保险）
  ```
- 【为什么必须 `.contiguous()`】`permute` 只改 stride 不搬数据，得到的张量是 non-contiguous 的；`view` 要求内存连续，不加 `.contiguous()` 会直接抛 `RuntimeError: view size is not compatible with input tensor's size and stride`。**这是 PyTorch 新手最常见的报错之一。**（可以用 `.reshape()` 规避，但 `.contiguous().view()` 的性能行为更可控，且导出更友好。）
- 【为什么这个 permute→view→gather 是"万能提货器"】因为**所有属性图的空间布局完全一致**（都是 448×224，都用同一套扁平索引编码）。所以同一个 `inds` 可以去 heatmap、reg、dim、rot、vel、movement、dir_cls、used_bev_feat、fusion_feat 上取，全部对齐。**这是 CenterNet 系解码代码能写得这么短的根本原因。**
- 【连接】
  - **BEVFusion / mmdet3d**：`_transpose_and_gather_feat` 在 `mmdet3d` 的 `centerpoint_bbox_coders.py` 里同名同实现，你可以直接对照。
  - **训练侧镜像**：Ch11 讲的目标级 10 维 L1 loss 就是用 `_transpose_and_gather_feat(anno_box, ind)` 把 GT 位置上的预测抠出来 `[1, 256, 10]`，再和 `target_box [1, 256, 10]` 算 L1。**训练时索引来自 GT，推理时索引来自 topk——这是同一套机制的两个用法。** console 里那句 `target_box.shape → torch.Size([1, 256, 10])` 就是这个。

---

**[02:00:01]** 原话：「对应的这个呃……」
**[02:00:04]** 原话：「这个是我们后续呃就是是……」
**[02:00:09]** 原话：「……tracker 以及 e2e tracker 这些任务要……」
**[02:00:13]** 原话：「……去做回归时候所用到的一些……」
**[02:00:15]** 原话：「……instance 级别上的 feature。」
**[02:00:18]** 原话：「就对应的置信度最高的那一个（feature）……」
**[02:00:22]** 原话：「……就是把它给抠出来了。」

> **重点句，五角度全给。这是理解整套代码"为什么要这么写"的钥匙。**

- 【直译】这 256×C 的特征向量不是给检测用的，是给**下游跟踪模块**用的。每个检测框都配一个"身份特征"。
- 【代码】两份 embedding 的区别：
  - `instance_embeddings` ← `used_bev_feat`（检测头实际用的那张 BEV 特征）
  - `fusion_instance_embeddings` ← `fusion_feat`（多模态融合后的 BEV 特征，通道更"原始"）
  在 `centerpoint_head.py` 侧，这两个还各自被预分配了容器（L1411–L1412）：
  ```python
  instance_embeddings_list.append(
      preds_dicts[0][task_id]['used_bev_feat'].new_zeros(
          batch_size, self.bbox_coder.max_num, preds_dicts[0][0]['used_bev_feat'].shape[1]))
  fusion_instance_embeddings_list.append(
      preds_dicts[0][task_id]['fusion_feat'].new_zeros(
          batch_size, self.bbox_coder.max_num, preds_dicts[0][0]['fusion_feat'].shape[1]))
  ```
  注意这里用的是 `new_zeros`（而后面 `pred_bboxes` 用的是 `new_ones * -1`）——**embedding 的"空位"填 0，box 的"空位"填 -1**，两种哨兵值，别搞混。
- 【形状】`[1, 256, C]`，C = BEV 特征通道数（配置项，画面没显示具体值；按这类模型的常规是 128 或 256）。
- 【为什么要在 decode 里做，而不是 tracker 自己去抠？】
  1. **索引在这里最全**。出了 `decode` 之后，`inds` 会被范围掩码过滤（L247 `indexs = inds[i, cmask]`），长度变成动态的；在 decode 内部还是固定 256，形状友好。
  2. **省一次 BEV 特征图的搬运**。BEV 特征图是 `[1, C, 448, 224]`，如果传给 tracker 再抠，等于把整张大图跨模块传一遍；在这里抠完只传 `[1, 256, C]`，**数据量小了 100352/256 ≈ 392 倍**。车端内存带宽极其宝贵。
  3. **端到端可微**。如果 tracker 也参与训练（e2e tracker），梯度可以从 tracker 沿着这些 embedding 回流到 BEV backbone。这就是"端到端"的字面含义。
- 【连接】
  - **对标 DETR 系**：`instance_embeddings` 扮演的角色相当于 DETR 的 `object query` 输出、或 MOTR/MUTR3D 的 `track query`。区别是 DETR 的 query 是学出来的、跨层迭代精修的；这里的 embedding 是**从 BEV 图上按位置直接采样的**，更便宜也更"死板"。
  - **对标 ReID**：传统 tracking-by-detection（如 DeepSORT）需要单独跑一个 ReID 网络抽外观特征。这里直接复用检测的 BEV 特征，**零额外算力**。代价是这份特征没有专门为"区分个体"做过监督（除非 e2e 训练时加了对比损失）。
  - **对你转岗的意义**：面试问"你怎么把检测和跟踪串起来"，这段代码就是标准答案的工业版本——**在 decode 里顺手 gather 一份 instance feature 出来**。比背 MOTR 论文实在得多。

---

**[02:00:29]** 原话：「然后这些只是对应……」
**[02:00:32]** 原话：「……分类的一个结果和 score 做一个……」
**[02:00:36]** 原话：「……做一个 shape 的一个变换。」

- 【代码】`bbox_coders.py:156–158`：
  ```python
  # class label
  clses  = clses.view(batch, self.max_num).float()
  scores = scores.view(batch, self.max_num)
  ```
- 【形状】`[1, 256] → [1, 256]`，形状没变，但 `clses` 从 `int32` 转成了 `float32`。
- 【为什么要转 float】因为最终的 15 维输出 tensor 是 float32 的（`heatmap.new_ones(...)` 继承 heatmap 的 dtype），label 要塞进第 7 列，必须是 float。**这也意味着 label 在最终输出里是 `0.0/1.0/2.0/3.0/4.0` 这样的浮点数**，下游取用时要 `int()` 一下。
- 【为什么 `view` 而不是别的】`_topk` 返回的就已经是 `[batch, K]` 了，这两行 `view` 其实是冗余的保险（防止上游改了 `_topk` 的返回形状）。工业代码里这种"防御性 reshape"很常见。

---

### 🔨 动手练习 ch12-9：手写 `_transpose_and_gather_feat` 并验证等价性

```python
import torch
torch.manual_seed(7)

def gather_feat(feats, inds):
    dim  = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    return feats.gather(1, inds)

def transpose_and_gather_feat(feat, ind):
    """centerpoint_bbox_coders.py:105 —— CenterNet 万能提货器"""
    feat = feat.permute(0, 2, 3, 1).contiguous()        # [B,C,H,W] → [B,H,W,C]
    feat = feat.view(feat.size(0), -1, feat.size(3))    # → [B,H*W,C]
    return gather_feat(feat, ind)                        # → [B,K,C]

B, C, H, W, K = 1, 8, 448, 224, 256
bev = torch.randn(B, C, H, W)
inds = torch.randint(0, H * W, (B, K))                   # 假装是 topk 出来的位置

out = transpose_and_gather_feat(bev, inds)
print("输入 :", tuple(bev.shape))    # (1, 8, 448, 224)
print("输出 :", tuple(out.shape))    # (1, 256, 8)

# ---- 用最朴素的循环对拍，确认语义正确 ----
ref = torch.empty(B, K, C)
for b in range(B):
    for k in range(K):
        r, c = inds[b, k].item() // W, inds[b, k].item() % W
        ref[b, k] = bev[b, :, r, c]
print("与朴素循环一致 :", torch.allclose(out, ref))       # True

# ---- 演示不加 .contiguous() 会怎样 ----
try:
    bad = bev.permute(0, 2, 3, 1).view(B, -1, C)
except RuntimeError as e:
    print("\n不加 contiguous 的报错 :", str(e)[:80], "...")
# 预期：RuntimeError: view size is not compatible with input tensor's size and stride...

# ---- 数据量对比：为什么要在 decode 里抠 ----
print("\n整张 BEV 特征元素数 :", bev.numel())              # 802816
print("抠出后元素数        :", out.numel())               # 2048
print("压缩比              :", bev.numel() / out.numel())  # 392.0
```

**你应该看到的**：`(1, 256, 8)`、`True`、以及那个 **392 倍的压缩比**——这就是"在 decode 里顺手抠 instance feature"的工程价值。

### 【小结】

1. `_transpose_and_gather_feat` = `permute(0,2,3,1) → contiguous → view(B,H*W,C) → gather`，是整套解码的"万能提货器"，所有属性图共用同一份 `inds`。
2. `instance_embeddings [1,256,C]` 是给 **tracker / 端到端跟踪** 用的目标级特征，在 decode 里抠比传整张 BEV 图省约 392 倍数据量，且保留端到端梯度通路。
3. 两份 embedding 分别来自 `used_bev_feat`（检测用）和 `fusion_feat`（融合后），空位填 `new_zeros`；而后面的 box 容器空位填 `-1`，两种哨兵值不要混淆。
---

# Part 12-7｜reg 偏移：格子位置 + 亚像素残差（02:00:39 – 02:01:19）

### 本段在讲什么

topk 给的 `xs/ys` 是**整数格子索引**，精度只有 0.4 m。要把定位做到厘米级，必须加上网络回归的亚像素偏移 `reg`。这一段是 CenterPoint 定位精度的关键，也是和 YOLO"格子 + 偏移"思想最直接对应的一段。

- **输入**：`xs/ys` `[1,256]`（整数格子索引的 float 形式）、`reg` `[1,2,448,224]`、`inds` `[1,256]`
- **输出**：`xs/ys` `[1,256,1]`（带亚像素精度的格子坐标）
- **流水线位置**：`centerpoint_bbox_coders.py:160–167`

---

**[02:00:39]** 原话：「然后这里的话会对应的把……」
**[02:00:44]** 原话：「……回归的 x y 上的……」
**[02:00:46]** 原话：「……呃在就是在……」
**[02:00:48]** 原话：「……feature map 上的 x y，然后以及对应的索引，然后把它给取出来。」

- 【代码】`bbox_coders.py:160–167`（帧 02:00:44 高亮 L161，帧 02:01:04 高亮 L163）：
  ```python
  if reg is not None:
      reg = self._transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, self.max_num, 2)
      xs  = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
      ys  = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
  else:
      xs = xs.view(batch, self.max_num, 1) + 0.5
      ys = ys.view(batch, self.max_num, 1) + 0.5
  ```
- 【形状】
  ```
  reg (图)                 : [1, 2, 448, 224]
  _transpose_and_gather_feat: [1, 256, 2]
  .view(1, 256, 2)          : [1, 256, 2]      （保险 reshape）
  xs.view(1, 256, 1)        : [1, 256, 1]
  xs + reg[:, :, 0:1]       : [1, 256, 1]
  ```
  注意 `reg[:, :, 0:1]` 用的是切片而不是索引——`reg[:, :, 0]` 会得到 `[1,256]`（掉一维），`reg[:, :, 0:1]` 保持 `[1,256,1]`。**这是保证广播不出错的常规写法。**
- 【为什么 `reg` 的通道 0 对应 x（行）、通道 1 对应 y（列）？】因为训练侧生成 GT 时用的就是这个顺序。虽然本章没展示 `get_targets`，但从 Ch11 的 `anno_box = torch.cat([reg, height, dim, rot, vel], dim=1)`（共 2+1+3+2+2=10 通道，console 实测 `anno_box.shape → [1, 10, 448, 224]` 完全对上）可知 reg 占前两通道，且必须与解码端同序。
- 【连接】和 Part 12-4 的 x/y 对调一脉相承：既然 `xs` 是行，那 `reg[...,0]` 就必须是行方向的偏移。**训练和推理两端的"行/列 ↔ x/y"约定必须完全一致，否则定位会带一个系统性的 90° 错位。**

---

**[02:00:55]** 原话：「然后这个 x y 呢其实对应的是……」
**[02:00:58]** 原话：「……他在哪一行，然后我们回归的时候其实是回归他相对于这一个位置……」
**[02:01:04]** 原话：「……他的一个偏移量。」
**[02:01:07]** 原话：「所以说这里回归之后就是他在 bev feature 上的对应的……」
**[02:01:12]** 原话：「……x 的一个坐标。」
**[02:01:14]** 原话：「然后这个就是 y 的一个坐标。」

> **重点句，五角度全给。**

- 【直译】最终的格子坐标 = 整数格子索引 + 网络回归的小数偏移。得到的是"带小数的格子坐标"，还不是米。
- 【代码】`xs = xs + reg[:, :, 0:1]`，`ys = ys + reg[:, :, 1:2]`。
- 【形状】`[1,256,1]`，值域约 `[0, 448)` 和 `[0, 224)`（带小数）。
- 【为什么必须有这个偏移？——定量算一笔账】
  - 不加 reg：定位误差 = 半个格子 = 0.2 m（均匀分布下 RMSE ≈ 0.4/√12 ≈ 0.115 m）。
  - 对 nuScenes 的 mAP@0.5m 阈值来说，0.2 m 的系统误差会直接吃掉相当一部分召回。
  - 对量产车的 AEB / ACC 来说，0.2 m 的横向误差在 50 m 外足以让"本车道 vs 邻车道"判错。
  - **加了 reg 之后，定位精度只受网络回归能力限制，理论上可以到几厘米。** 这是 CenterNet 论文里 offset head 的原始动机（补偿下采样量化误差）。
- 【为什么 else 分支加 0.5？】
  - `else` 是"没有 reg 头"的退化配置。此时最佳猜测是**格子中心**，而格子 `k` 的中心在连续坐标下是 `k + 0.5`。
  - **反推一个重要结论**：既然 else 分支加 0.5 才是格心，那么 `if` 分支里 `reg` 的期望值域就应该是 `[0, 1)`（偏移相对于格子左上角），而不是 `[-0.5, 0.5)`。
  - 对应训练侧 `get_targets` 里的写法应该是 `reg_target = center_float - center_int`（`center_int = floor(center_float)`），结果天然落在 `[0,1)`。⚠ 本章没展示 `get_targets`，这是我从 else 分支的 0.5 反推的，**建议你在 4060 服务器上打开这份代码的 `get_targets` 核实一下**——如果那边写的是 `center_float - center_int - 0.5`，那 else 分支的 0.5 就该去掉，两者必有一处不一致。
- 【连接】
  - **YOLO（用户最熟）**：YOLOv3 `bx = σ(tx) + cx`，`cx` 是格子左上角，`σ(tx) ∈ (0,1)` 是偏移 → **和这里的 `xs = topk_xs + reg` 完全同构**。区别：YOLO 用 sigmoid 硬约束在 (0,1)，CenterPoint 用 L1 loss 软约束（所以 reg 理论上可以跑出 [0,1)，属于网络自由度）。YOLOv5 改成 `2σ(tx) - 0.5 ∈ (-0.5, 1.5)` 是为了让格子边界上的目标也能被相邻格子预测——**CenterPoint 不需要这个技巧，因为它的正样本就是峰值格子本身**。
  - **CenterNet 论文**：offset head 的原文动机是"recover the discretization error caused by the output stride"，一模一样。
  - **智谷课程的卷积知识**：为什么会有量化误差？因为 BEV 图是把连续的 3D 空间按 0.4 m 分桶得到的，任何离散化都会引入 ±半桶的误差。`reg` 就是把这个误差学回来。

---

### 🔨 动手练习 ch12-10：量化"有 reg / 无 reg"的定位精度差

```python
import torch
torch.manual_seed(0)

RES = 0.4                          # 米/格
N   = 100000

# 真实中心的连续格子坐标（随机小数）
true_grid = torch.rand(N) * 400

# 方案 A：只用整数格子（无 reg）
pred_A = true_grid.floor()
# 方案 B：整数格子 + 完美 reg
pred_B = true_grid.floor() + (true_grid - true_grid.floor())
# 方案 C：整数格子 + 0.5（else 分支）
pred_C = true_grid.floor() + 0.5
# 方案 D：整数格子 + reg，但 reg 有 0.05 格的回归噪声
pred_D = true_grid + torch.randn(N) * 0.05

for name, p in [("A 无偏移      ", pred_A), ("B 完美reg     ", pred_B),
                ("C 恒加0.5(格心)", pred_C), ("D reg有噪声   ", pred_D)]:
    err_m = (p - true_grid).abs() * RES
    print(f"{name}  平均误差 {err_m.mean():.4f} m   最大误差 {err_m.max():.4f} m")

# 预期输出（约）：
# A 无偏移         平均误差 0.2000 m   最大误差 0.4000 m
# B 完美reg        平均误差 0.0000 m   最大误差 0.0000 m
# C 恒加0.5(格心)  平均误差 0.1000 m   最大误差 0.2000 m
# D reg有噪声      平均误差 0.0160 m   最大误差 0.0900 m
#
# 结论：光是把"取左上角"改成"取格心"就把平均误差砍半；
#      加上一个哪怕不太准的 reg 头（σ=0.05格=2cm），误差再降一个数量级。
```

### 【小结】

1. `xs = topk_xs + reg[...,0:1]`、`ys = topk_ys + reg[...,1:2]`：**整数格子 + 亚像素偏移**，把 0.4 m 的量化误差学回来。
2. `else` 分支加 0.5（格心）反推出 `reg` 的期望值域是 `[0,1)`——⚠ 建议核对 `get_targets` 里 `reg` 的生成方式是否为 `center_float - center_int`。
3. 与 YOLO 的 `bx = σ(tx) + cx` 完全同构；差别只在 CenterPoint 用 L1 软约束而非 sigmoid 硬约束。

---

# Part 12-8｜sin/cos → atan2、高度、尺寸，以及那个"减号"坐标变换（02:01:20 – 02:02:39）

### 本段在讲什么

到这里位置（格子坐标）已经定了，接下来把剩下的框参数一个个 gather 出来并做最后的物理还原：朝向用 `atan2`、尺寸和高度直接 view、**格子坐标乘分辨率 + BEV 区间偏移变成自车坐标系的米**。最后 `torch.cat` 成一张 `[1, 256, 10]` 的表。

- **输入**：`rot_sine/rot_cosine/hei/dim/vel` 的稠密图 + `inds`
- **输出**：`final_box_preds = torch.cat(all_box_attrs, dim=2)`，`[1, 256, 10]`
- **流水线位置**：`centerpoint_bbox_coders.py:169–222`

---

**[02:01:20]** 原话：「然后这里会去对应的去取……」
**[02:01:23]** 原话：「……sinθ 和他的一个 cosθ。」

- 【代码】`bbox_coders.py:169–175`（帧 02:00:44 / 02:01:36 完整可读）：
  ```python
  # rotation value and direction label
  rot_sine   = self._transpose_and_gather_feat(rot_sine, inds)
  rot_sine   = rot_sine.view(batch, self.max_num, 1)

  rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
  rot_cosine = rot_cosine.view(batch, self.max_num, 1)
  rot        = torch.atan2(rot_sine, rot_cosine)
  ```
- 【形状】
  ```
  rot_sine   : [1,1,448,224] → gather → [1,256,1]
  rot_cosine : [1,1,448,224] → gather → [1,256,1]
  rot        : [1,256,1]      弧度，值域 (-π, π]
  ```
- 【为什么用 `atan2(s, c)` 而不是 `atan(s/c)`】
  - `atan(s/c)` 只能返回 `(-π/2, π/2)`，丢掉了象限信息：`(s=1,c=1)` 和 `(s=-1,c=-1)` 的比值都是 1，但角度差 180°。
  - `atan2(s, c)` 同时看 s 和 c 的符号，能恢复完整的 4 个象限，返回 `(-π, π]`。
  - 而且 `atan2` 在 `c=0` 时不会除零（`atan(s/0)` 会 NaN/inf）。
- 【为什么不需要归一化 (s,c) 到单位圆】网络输出的 `(rot_sine, rot_cosine)` 并不保证 `s²+c²=1`。但 `atan2` 只关心**方向**（s:c 的比值 + 符号），模长完全不影响结果。所以不用归一化。
  - 副作用：模长 `√(s²+c²)` 其实携带了"网络对这个朝向有多自信"的信息（训练时 L1 逼近真实 sin/cos，收敛好的地方模长接近 1）。**这是一个免费的置信度信号，但这份代码没用它。** 值得记一笔，是可以做的优化点。
- 【⚠ `atan2` 自己也带一个"分支切割"，这一点常被忽略，而且直接关系到结尾那句"稳定性差"】
  - `atan2` 的值域是 `(-π, π]`，在 **±π 处有一条切割线**。目标朝向恰好接近 180°（车在自车正后方、车头朝着你）时，网络的 `rot_sine` 在 0 附近正负抖动，解出来的 yaw 就在 `+179.9°` 和 `-179.9°` 之间来回跳。
  - **物理上纹丝不动，数值上跳了 360°。** 任何"用相邻帧 yaw 的差值判断朝向是否稳定"的下游逻辑（tracker 的朝向平滑、卡尔曼滤波的观测更新）都会被这一跳骗到，表现为"这辆车突然掉头了"。
  - 我在练习 ch12-11 里用 float32 复现了这个现象：`torch.tensor(math.pi)` 存成 float32 是 3.14159274（略大于 π），`sin` 就变成一个极小的**负数**，`atan2` 落到 `-π` 那一侧。**这不是练习的 bug，是真实系统里每一帧都在发生的事。**
  - 【正确的处理方式】**任何角度差都必须先 wrap 再比较**：`d = atan2(sin(a-b), cos(a-b))`。这行是 3D 感知/跟踪代码里最该背下来的一行工具函数。
  - 【和 sin/cos 回归的关系】注意：**用 (sin, cos) 回归解决的是"训练时的梯度断点"，它并没有解决"解码后表示的分支切割"。** 前者在 loss 里，后者在输出上——两者是两个独立的问题，很多人会混为一谈。这份代码解决了前者，后者留给了下游。
- 【连接】
  - **CenterPoint / mmdet3d**：`rot = torch.atan2(rot_sine, rot_cosine)` 一字不差。
  - **周期性问题的其他解法**：(a) 直接回归 θ + `dir_cls`（SECOND/PointPillars）；(b) 回归 `(sin, cos)`（CenterPoint）；(c) 角度分 bin + bin 内回归（MonoDIS/FCOS3D 的 multi-bin）。**DenseBEV 同时用了 (a) 和 (b)**：sin/cos 给连续朝向，`dir_cls` 给一个额外的正反向二分类（见下）。
  - **Ch11 的伏笔**：讲者在 01:51:40 说"会根据不同的一个朝向角区间去算 sinθ 和 cosθ 的 Loss"——训练时按角度区间分段加权，就是为了让 sin/cos 在各个方向上都学得均匀（否则数据集里"车头朝前"的样本占绝对多数，侧向的学不好）。

---

**[02:01:28]** 原话：「然后回去取他的一个高。」
**[02:01:33]** 原话：「然后以及取他的一个长宽……」
**[02:01:37]** 原话：「……长宽高。」

- 【代码】`bbox_coders.py:177–192`（帧 02:01:36 完整可读）：
  ```python
  rot_short = None
  if rot_short_sin is not None and rot_short_cos is not None:
      rot_short_sin = self._transpose_and_gather_feat(rot_short_sin, inds)
      rot_short_sin = rot_short_sin.view(batch, self.max_num, 1)

      rot_short_cos = self._transpose_and_gather_feat(rot_short_cos, inds)
      rot_short_cos = rot_short_cos.view(batch, self.max_num, 1)
      rot_short     = torch.atan2(rot_short_sin, rot_short_cos)

  # height in the bev
  hei = self._transpose_and_gather_feat(hei, inds)
  hei = hei.view(batch, self.max_num, 1)

  # dim of the box
  dim = self._transpose_and_gather_feat(dim, inds)
  dim = dim.view(batch, self.max_num, 3)
  ```
- 【形状】`hei: [1,256,1]`（米，自车坐标系 z）；`dim: [1,256,3]`（米，已在 L1356 exp 过）。
- 【`rot_short` 是什么】`enable_corner_det` 打开时，模型对长条形目标同时回归"长边朝向"（`rot_long` → `rot`）和"短边朝向"（`rot_short`）。理论上两者应该差 90°，可以互相校验、或对遮挡严重的大车做更稳的朝向估计。center 模式下 `rot_short_sin/cos` 都是 None，这个分支整段跳过。
- 【注释里的 `# height in the bev`】这句英文注释其实**有点误导**：`hei` 不是"BEV 里的高度"（BEV 是俯视图，没有高度），而是**目标中心的 z 坐标**。真正的"高"（box 的 h）在 `dim` 的第 3 通道里。⚠ 读代码时别被注释带偏。

---

**[02:01:40]** 原话：「然后在这里会去计算一下……」
**[02:01:43]** 原话：「……我的会通过……」
**[02:01:46]** 原话：「……就是我其实这里的 xs 其实还是在 bev feature 上的一个位置。」
**[02:01:51]** 原话：「然后在这里会去……」
**[02:01:55]** 原话：「……分辨率，然后……」
**[02:01:58]** 原话：「……以及对应的我的这个 bev 的一个区间。」
**[02:02:01]** 原话：「然后会把它转换到是以自车为坐标系下的一个位置。」
**[02:02:08 / 02:02:11 / 02:02:13]** 原话：「这里的一个……x 和 xs 和 ys……就是在自车坐标系……下的。」

> **本章第四大重点句，也是全章最容易被讲者一带而过、但你必须看清代码的地方。五角度全给。**

- 【直译】把"第几格"换算成"多少米"：乘以分辨率，再加上（这里其实是"用……减去"）BEV 区间的边界值。
- 【代码】`bbox_coders.py:194–195`。**我把这两行做了 5 倍放大逐字确认（帧 02:01:36），请注意是减号不是加号**：
  ```python
  xs = (self.pc_range[0] - xs.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[0])
  ys = (self.pc_range[1] - ys.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[1])
  ```
- 【形状】`[1,256,1] → [1,256,1]`，单位从"格"变成"米"。
- 【为什么是减号？——这是全章最重要的坐标系发现】
  - **mmdet3d / CenterPoint 原版是加号**：
    ```python
    xs = xs.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
    ```
    含义：`pc_range[0]` 是 x 方向的**最小值**（比如 -54），索引从 0 开始向 +x 增长。
  - **DenseBEV 改成了减号**，说明 `pc_range[0]` 存的是 x 方向的**最大值**，索引从 0 开始向 **-x** 增长。也就是：**BEV 张量的第 0 行 = 车头最远处，第 447 行 = 车尾最远处。**
  - **代入本视频给出的配置验证**：`out_size_factor × voxel_size[0] = 0.4 m`（分辨率），`pc_range[0] = 95.4`（前向最远）：
    ```
    row = 0   → x = 95.4 - 0   × 0.4 = +95.4 m   （最前）
    row = 447 → x = 95.4 - 447 × 0.4 = -83.4 m   （最后一格的代表点）
    ```
    ✅ 覆盖范围 `448 × 0.4 = 179.2 m`，正是"前 95.4 + 后 83.8"。
    ```
    col = 0   → y = 44.8 - 0   × 0.4 = +44.8 m   （最左）
    col = 223 → y = 44.8 - 223 × 0.4 = -44.4 m   （最右一格的代表点）
    ```
    ✅ 覆盖范围 `224 × 0.4 = 89.6 m`，正是"左右各 44.8"。
  - **⚠ 别被那 0.4 m 的零头绊住**：`row=447` 算出的是 `-83.4` 而不是边界值 `-83.8`，`col=223` 算出的是 `-44.4` 而不是 `-44.8`。**差的正好是一整格**，原因是 448 个格子之间只有 447 个间隔（"栅栏与栅栏柱"问题），而且 `pc_range[0] - row*res` 给出的是格子的**近端角**而非格心。真正的格心应该是 `pc_range[0] - (row + 0.5)*res`，代入 `row=447` 得 `-83.6`，落在 `[-83.8, -83.4]` 的正中间。
  - **这个 0.5 格的偏置去哪了？** 被 `reg` 吸收了。`reg ∈ [0,1)` 的期望值 0.5 正好补上这半格（Part 12-7），而没有 `reg` 头时代码显式写 `+0.5`。**所以整条链路是自洽的，只是"格角坐标 + 学出来的格内偏移"这种写法把格心约定藏进了网络权重里。**
  - **这意味着 BEV 图画出来是"车头在上、左侧在左"的俯视图**（图像坐标 row 从上往下增大，正好对应 x 从大到小 = 从远到近）。这是很自然的可视化约定，但和 mmdet3d 的"row 从下往上"是**上下翻转**的关系。
- 【⚠ 讲者表述与代码的差异】讲者说"乘分辨率 + 对应 BEV 区间"，听起来像加法；**代码是"区间端点 − 索引×分辨率"**。含义上他没错（都是做仿射变换），但符号方向他没讲。**如果你照着他的口述去写代码，会得到一个前后颠倒的坐标系。以代码为准。**
- 【`out_size_factor` 与 `voxel_size` 的关系】两者相乘才是 BEV 分辨率：
  - 若 `voxel_size = [0.2, 0.2, 8]`、`out_size_factor = 2` → 0.4 m ✓
  - 若 `voxel_size = [0.4, 0.4, 8]`、`out_size_factor = 1` → 0.4 m ✓
  - 两种配法都能得到 0.4，画面上看不到具体值。⚠ 但结合视频前面提到的"下采样一倍到 224×112 上做投影"，更可能是 `voxel_size=0.4 / out_size_factor=1`（BEV backbone 内部再下采样一次，但 head 输出回到 448×224）。**建议在 4060 上打开 config 核实。**
- 【连接】
  - **BEVFusion**：`bbox_coder=dict(type='CenterPointBBoxCoder', pc_range=[-54,-54], out_size_factor=8, voxel_size=[0.075,0.075], ...)`，`8 × 0.075 = 0.6 m`。你可以把 DenseBEV 的这套数字和 BEVFusion 的对照记：**DenseBEV 分辨率更细（0.4 vs 0.6）、范围更长（179 m vs 108 m）、但横向更窄（89.6 m vs 108 m）**——非常典型的"高速+城区量产"取舍。
  - **nuScenes 数据链**：`pc_range` 在 mmdet3d 里是 6 元组 `[x_min,y_min,z_min,x_max,y_max,z_max]`，`bbox_coder` 只用前两个。这里因为符号反了，`pc_range[0]/[1]` 存的应该是 **max** 而非 min ⚠（或者他们单独定义了一个不同语义的 `pc_range`）。这是迁移这套代码时最容易踩的雷。

---

**[02:02:16 / 02:02:20]** 原话：「然后后续这些的话就是……这些属性做一个 concat（转写作「开的」）。」
**[02:02:22]** 原话：「然后取速度。」
**[02:02:25]** 原话：「取速度。」
**[02:02:26]** 原话：「然后也 concat 进去。」

- 【代码】`bbox_coders.py:197–222`（帧 02:02:07 / 02:02:26 完整可读）：
  ```python
  # cat bbox属性
  all_box_attrs = [xs, ys, hei, dim, rot]
  if rot_short is not None:
      all_box_attrs.append(rot_short)

  if vel is not None:
      vel = self._transpose_and_gather_feat(vel, inds)
      vel = vel.view(batch, self.max_num, 2)
      all_box_attrs.append(vel)

  if movement is not None:
      movements = self._transpose_and_gather_feat(movement, inds)
      movements = movements.view(batch, self.max_num, movement.shape[1])

  if direction is not None:
      directions = self._transpose_and_gather_feat(direction, inds)
      directions = directions.view(batch, self.max_num, direction.shape[1])

  final_instance_embeddings        = instance_embeddings
  final_fusion_instance_embeddings = fusion_instance_embeddings
  final_clses    = clses
  final_scores   = scores
  final_box_preds = torch.cat(all_box_attrs, dim=2)
  final_movements  = movements
  final_directions = directions
  ```
- 【形状】center 模式（无 rot_short）下：
  ```
  xs  [1,256,1] + ys [1,256,1] + hei [1,256,1] + dim [1,256,3] + rot [1,256,1] + vel [1,256,2]
  → torch.cat(dim=2) → final_box_preds : [1, 256, 10]
  ```
  **列顺序是 `[x, y, z, w, l, h, yaw, vx, vy]`，一共 9 列**（1+1+1+3+1+2 = 9），**不是 10 列**。

  【这一条我此前标了存疑，现在由帧 02_01_36 直接坐实】画面上 L198 白纸黑字写着 `all_box_attrs = [xs, ys, hei, dim, rot]`（5 项，共 7 列），L199–200 是 `if rot_short is not None: all_box_attrs.append(rot_short)`，L202–205 是 `if vel is not None: ... all_box_attrs.append(vel)`。所以：
  | 模式 | `all_box_attrs` 内容 | `final_box_preds` 列数 |
  |---|---|---|
  | center（本视频演示的） | xs,ys,hei,dim,rot,vel | **9** |
  | corner（`enable_corner_det`） | xs,ys,hei,dim,rot,**rot_short**,vel | **10** |
  | 无 vel 的配置 | xs,ys,hei,dim,rot | 7 |

  ⚠ **不要把它和另外两个"10"搞混**，这是本章最容易串线的地方：
  - `target_box.shape = [1, 256, 10]`（console 实测）是**训练侧**的 GT 目标级张量，10 = `reg2 + hei1 + dim3 + rot2 + vel2`；
  - `preds_dict[0]['anno_box'].shape = [1, 10, 448, 224]`（console 实测）是**训练侧**的稠密预测图，同一套 10 通道；
  - `final_box_preds` 是**推理侧**的，`rot` 已经被 `atan2` 从 2 列压成了 1 列（sin/cos → yaw），所以 10 − 1 = **9**。
  **一个 `atan2` 就是这两个数字差 1 的全部原因。**

  【为什么列数变来变去也不出事】因为下游 `centerpoint_head.py` 的填充只用**两端**的切片：`pred_bboxes[..., :Obj.ry.value+1] = pred['bboxes'][:, :7]` 取前 7 列，`pred_bboxes[..., -2:] = pred['bboxes'][:, -2:]` 取后 2 列。**中间那 0~1 列（rot_short）压根没人读**，所以 9 列 / 10 列都能跑。这是"负索引 + 前缀切片"这种写法的隐性好处——**但也意味着 corner 模式下的 `rot_short` 解出来之后就被默默丢弃了**，它只活在 `final_box_preds` 里，从没进过 15 维输出。⚠ 这是一个值得问作者的点：算了却不用，要么是给别的分支留的口子，要么就是真的漏了。
- 【为什么 movement / direction 不进 `all_box_attrs`？】因为它们要单独放进 `predictions_dict['movements'] / ['directions']`，在 `centerpoint_head.py` 侧再按 `Obj.mov.value / Obj.dir_cls.value` 填到 15 维的**倒数第 3、第 4 列**。box 属性和"语义属性"分开管，是为了让 `final_box_preds` 保持"纯几何"，方便做范围过滤（下一 Part 的 `mask` 只看前 3 列 xyz）。
- 【连接】`torch.cat(dim=2)` 沿最后一维拼——这跟 Ch11 训练侧 `anno_box = torch.cat([reg, height, dim, rot, vel], dim=1)` 沿**通道维**拼是同一件事的两种布局（训练侧是 `[B,C,H,W]` 所以 dim=1，推理侧是 `[B,K,C]` 所以 dim=2）。

---

### 🔨 动手练习 ch12-11：sin/cos → atan2 的完整往返 + 周期性断点演示

```python
import torch, math

# ---------- 1. atan2 能恢复完整 4 象限 ----------
angles = torch.tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi,
                       -3*math.pi/4, -math.pi/2, -math.pi/4])
s, c = torch.sin(angles), torch.cos(angles)
rec_atan2 = torch.atan2(s, c)
rec_atan   = torch.atan(s / c)          # 错误做法
print("原始角度(度) :", (angles * 180/math.pi).round().tolist())
print("atan2 恢复   :", (rec_atan2 * 180/math.pi).round().tolist())
print("atan  恢复   :", (rec_atan  * 180/math.pi).round().tolist())
# 实测输出：
#   atan2 恢复 : [0.0, 45.0, 90.0, 135.0, -180.0, -135.0, -90.0, -45.0]
#   atan  恢复 : [0.0, 45.0, -90.0, -45.0,    0.0,   45.0,  90.0, -45.0]
# → atan 把 135° 变成 -45°、把 180° 变成 0°、把 -135° 变成 45° —— 象限信息全丢，一半的角度是错的。
#
# ★ 但请注意 atan2 那一行的第 5 个值：输入是 +180°，输出是 **-180°**，不是 +180°。
#   物理上两者是同一个方向，数值上却差 360°。原因：float32 存不下 π，
#   torch.tensor(math.pi) 实际是 3.14159274（略大于 π），于是 sin 变成一个极小的负数，
#   atan2 就落到了 -π 那一侧。
#   ★★ 这不是"练习里的小 bug"，而是 Part 12-11 说的"朝向不稳定"的一个真实来源：
#      车头几乎正对 ±180°（也就是车在你正后方、朝你开）时，网络的 sin 输出在 0 附近
#      抖动，解出来的 yaw 就在 +179.99° 和 -179.99° 之间来回跳。数值上跳了 360°，
#      任何"用差值判断是否稳定"的下游逻辑（比如 tracker 的朝向平滑）都会被骗到。
#      正确做法是永远用 wrap 到 (-π, π] 之后的角度差：
#          d = torch.atan2(torch.sin(a - b), torch.cos(a - b))
print("\n180° 附近的分支切割 :", torch.atan2(torch.sin(angles[4:5]), torch.cos(angles[4:5])).item())
diff = torch.atan2(torch.sin(rec_atan2 - angles), torch.cos(rec_atan2 - angles))
print("用 wrap 后的角度差 :", (diff * 180/math.pi).round().tolist(), "← 全 0，说明 atan2 其实没错")

# ---------- 2. 模长不影响结果 ----------
k = torch.tensor([0.1, 1.0, 7.3, 100.0]).view(-1, 1)
ang = torch.tensor([[2.0]])                       # 2 rad
out = torch.atan2(k * torch.sin(ang), k * torch.cos(ang))
print("\n不同模长下的 atan2 :", out.flatten().tolist())   # 全是 2.0 —— 无需归一化

# ---------- 3. 为什么不能直接回归 θ：断点处的假梯度 ----------
gt   = torch.tensor(3.10)                          # 177.6°
pred = torch.tensor(-3.10)                         # -177.6°，物理上只差 4.8°
print("\n直接回归θ 的 L1 loss :", (pred - gt).abs().item())        # 6.20 —— 巨大假梯度
loss_sc = (torch.sin(pred)-torch.sin(gt)).abs() + (torch.cos(pred)-torch.cos(gt)).abs()
print("回归 sin/cos 的 L1   :", loss_sc.item())                    # 0.083 —— 正确反映"很接近"
```

### 🔨 动手练习 ch12-12：BEV 格子 → 自车坐标（含"减号"的验证）

```python
import torch

H, W       = 448, 224
RES        = 0.4                                # = out_size_factor * voxel_size
PC_RANGE   = [95.4, 44.8]                       # ⚠ 这份代码里存的是 MAX 不是 MIN

def grid_to_ego(row, col):
    """复现 centerpoint_bbox_coders.py:194-195"""
    x = PC_RANGE[0] - row * RES
    y = PC_RANGE[1] - col * RES
    return x, y

for r, c, tag in [(0, 0, "左前角"), (0, W-1, "右前角"),
                  (H-1, 0, "左后角"), (H-1, W-1, "右后角"),
                  (H//2, W//2, "中心附近")]:
    x, y = grid_to_ego(r, c)
    print(f"row={r:3d} col={c:3d} ({tag}) → x={x:+7.2f} m  y={y:+7.2f} m")

# ★ 注意：格心到格心的跨度是 (H-1)*RES，不是 H*RES ——差一整格 0.4 m
span_x_center = grid_to_ego(0,0)[0] - grid_to_ego(H-1,0)[0]
span_y_center = grid_to_ego(0,0)[1] - grid_to_ego(0,W-1)[1]
print("\n首末格'代表点'跨度 x :", round(span_x_center,1), "m   = (448-1)*0.4 = 178.8")
print("首末格'代表点'跨度 y :", round(span_y_center,1), "m   = (224-1)*0.4 =  89.2")
print("整块 BEV 覆盖    x :", round(H*RES,1), "m   = 448*0.4 = 179.2  ← 与 前95.4+后83.8 对得上")
print("整块 BEV 覆盖    y :", round(W*RES,1), "m   = 224*0.4 =  89.6  ← 与 左右±44.8 对得上")

# ---- 反例：照 mmdet3d 原版的加号写 ----
def grid_to_ego_mmdet3d(row, col, pc_min=(-83.8, -44.8)):
    return pc_min[0] + row * RES, pc_min[1] + col * RES
print("\n若用原版加号(且 pc_range 存 min)：row=0 →", grid_to_ego_mmdet3d(0,0))
print("→ 第0行变成了车尾最远处，整个 BEV 图前后颠倒")

# 预期输出：
# row=  0 col=  0 (左前角) → x= +95.40 m  y= +44.80 m
# row=  0 col=223 (右前角) → x= +95.40 m  y= -44.40 m
# row=447 col=  0 (左后角) → x= -83.40 m  y= +44.80 m
# row=447 col=223 (右后角) → x= -83.40 m  y= -44.40 m
# 首末格'代表点'跨度 x = 178.8 m ；整块覆盖 x = 179.2 m
# 首末格'代表点'跨度 y =  89.2 m ；整块覆盖 y =  89.6 m
```

**你应该看到的（以及一个必须想清楚的 0.4 m 差值）**：

- 打印出来的"首末格跨度"是 **178.8 / 89.2**，而不是 179.2 / 89.6。**差的正好是一格 0.4 m，这不是 bug，是"格数"和"格间距数"的区别**：448 个格子之间只有 447 个间隔。
- 真正与讲义「前 95.4 / 后 83.8 / 左右 ±44.8」对齐的是 **`H*RES = 179.2`** 和 **`W*RES = 89.6`**，也就是**整块 BEV 的面积覆盖**。这才是"减号 + `pc_range` 存 max"这一解读正确的证明。
- 由此还能反推出一个细节：`row=447` 解出的 `x = -83.4`，而 BEV 后向边界是 `-83.8`。**差的这 0.4 m 说明代码里的 `pc_range[0] - row*res` 给出的是每个格子的"某个角"而不是格心**——如果想要格心，应该是 `pc_range[0] - (row + 0.5)*res`。**这也正好解释了 `decode` 里 `else` 分支为什么要 `+0.5`**（Part 12-7）：有 `reg` 时网络自己学会把这 0.5 补进来，没有 `reg` 时就手动补格心。两处是同一件事的两种写法。

### 【小结】

1. 朝向用 `rot = torch.atan2(rot_sine, rot_cosine)` 还原，能覆盖完整 `(-π, π]` 四象限，且不需要把 (s,c) 归一化到单位圆；模长其实是一个免费但未被利用的置信度信号。⚠ 但 `atan2` 在 ±π 处仍有分支切割，正后方目标的 yaw 会在 ±179.9° 之间跳变——**sin/cos 回归解决的是训练梯度断点，不是输出表示的断点**，后者要靠下游 `atan2(sin(a-b), cos(a-b))` 的 wrap 来吃掉。
2. **`xs = pc_range[0] - xs * out_size_factor * voxel_size[0]` 是减号**，说明这套代码的 `pc_range[0]/[1]` 存的是区间**最大值**，BEV 第 0 行 = 车头最远。代入 0.4 m 分辨率能精确还原「前95.4/后83.8/±44.8」，⚠ 与 mmdet3d 原版的加号写法相反，跨仓库迁移必踩。
3. `final_box_preds = torch.cat([xs, ys, hei, dim, rot, (rot_short), vel], dim=2)`，**center 模式确定是 `[1, 256, 9]`**（帧 02_01_36 的 L198–L205 已坐实），corner 模式多一列 `rot_short` 变 10；它比训练侧的 10 通道 `anno_box` 少 1，原因就是 `atan2` 把 sin/cos 两列压成了 yaw 一列。movement/direction 不进这个 cat，单独走 `predictions_dict`。
---

# Part 12-9｜超界过滤、分数阈值、打包返回（02:02:40 – 02:03:25）

### 本段在讲什么

前面 `xs = pc_range[0] - xs*res` 里的 `xs` 已经带上了 `reg` 偏移，所以有可能被推出 BEV 边界；`hei`（z）更是完全自由回归的，可能给出 100 米高的车。这一段用一个 6 元的 `post_center_range` 做立方体裁剪，再叠一个分数阈值，最后逐 batch 打包成 dict 返回。

- **输入**：`final_box_preds` `[1,256,~10]`、`final_scores` `[1,256]` 等
- **输出**：`predictions_dicts`（list，每 batch 一个 dict）、`final_box_preds`
- **流水线位置**：`centerpoint_bbox_coders.py:224–267`

---

**[02:02:40]** 原话：「然后在这里呢……」
**[02:02:41]** 原话：「然后在这里呢也……在这里就是计算出来的 xy 有可能超过我们预设的一个 bv 的一个范围。」

- 【直译】解出来的米制坐标可能跑到 BEV 覆盖范围之外，这种框是无效的。
- 【为什么会超界？】三个来源：
  1. **reg 偏移把边界格子推出去**：`row=0` 的格子加上一个负的 reg，`x` 就 > 95.4 m。
  2. **未训练/欠训练的网络乱吐**：本视频演示用的 checkpoint heatmap 全是 0.5，被 topk 选中的 256 个格子基本是随机的，属性预测也是随机值，`hei` 完全可能是 ±50 m。
  3. **corner 模式下 `obj_corner_to_center` 的几何反推**：从角点推中心可能推到界外。
- 【连接】mmdet3d 的 `post_center_range` 默认写法是 `[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]`——比 `pc_range` **略大**（留一点余量，避免把贴边的真目标误杀）。DenseBEV 应该也是类似策略。

---

**[02:02:50]** 原话：「说说在这里会通过最大的……最大的 x 最小的 x、最大的 y 最小的 y，然后通过这个范围去做一个 mask，然后把超过 bv 范围上的这些目标给……给过滤掉。」

> **重点句，五角度全给。**

- 【直译】用 6 个数（xyz 的下界 + xyz 的上界）围出一个长方体，框中心落在外面的就丢掉。
- 【代码】`bbox_coders.py:224–231`（帧 02_02_45 高亮 L229，我把 L229 放大 4 倍逐字确认过）：
  ```python
  # use score threshold                                                # L224
  if self.score_threshold is not None:                                 # L225
      thresh_mask = final_scores > self.score_threshold                # L226

  if self.post_center_range is not None:                               # L228
      post_center_range = torch.tensor(self.post_center_range, device=heat.device)   # L229
      mask  = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)             # L230
      mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)             # L231
  ```
  ⚠ 注意 L229 的真实写法是 **`device=heat.device`，并且没有 `dtype=` 参数**（不是常见的 `device=final_box_preds.device, dtype=final_box_preds.dtype`）。这个细节下面单独讲。
- 【形状】
  ```
  final_box_preds[..., :3]                    : [1, 256, 3]     取 x,y,z 三列
  >= post_center_range[:3]                    : [1, 256, 3]  bool（广播）
  .all(2)                                     : [1, 256]     bool，三个维度都满足才 True
  mask &= (... <= post_center_range[3:]).all(2): [1, 256]
  ```
- 【⚠ 讲者说漏了 z】他只说了"最大的 x 最小的 x 最大的 y 最小的 y"，但代码取的是 `[..., :3]`——**x、y、z 三个维度都要过滤**。`post_center_range` 是 6 元组 `[x_min, y_min, z_min, x_max, y_max, z_max]`，`[:3]` 是下界、`[3:]` 是上界。z 的过滤同样重要（能滤掉"飘在空中的车"）。
- 【为什么用 `.all(2)` 而不是逐维判断】`.all(dim=2)` 把 3 个 bool 归约成 1 个：**三个坐标都在范围内才保留**。这是"逻辑与"的向量化写法，比 `m0 & m1 & m2` 更简洁，也更容易适配不同的坐标维数。
- 【为什么要写 `device=heat.device`，又为什么**没写** `dtype=`？——这行值得单独拆开看】
  - **`device=` 是必须的**。`self.post_center_range` 是 config 里的 python list，`torch.tensor(list)` 默认落在 **CPU** 上。而 `final_box_preds` 在 `cuda:0`（帧 01_54_24 的 tooltip 明确显示 `device = device(type='cuda', index=0)`）。CPU 张量和 CUDA 张量直接比较会抛 `RuntimeError: Expected all tensors to be on the same device`。所以这个参数不是"优化"，是**不写就崩**。
  - **为什么用 `heat.device` 而不是 `final_box_preds.device`？** 两者其实同一个设备（`final_box_preds` 就是从 `heat` 的索引结果一路算下来的），所以功能上等价。用 `heat` 是因为它是 `decode` 的**入参**、生命周期最长、最"稳"，属于随手取一个已知在正确设备上的张量当参照——**是习惯写法，不是有意为之**。
  - **没写 `dtype=` 的真实后果**：`torch.tensor([...python float...])` 会用**全局默认 dtype**，即 `torch.float32`（不是 float64——numpy 才默认 float64，PyTorch 默认 float32，这一点很多人记反）。当前 `final_box_preds` 也是 float32（tooltip 已证），所以**比较时不发生任何类型提升，恰好正确**。
  - **⚠ 但这里埋着一个真实的隐患**：如果哪天这套模型开 **AMP / fp16 推理**（车端量化部署几乎一定会开），`final_box_preds` 会变成 `float16`，而 `post_center_range` 仍是 `float32`。此时 PyTorch 的**类型提升规则**会把比较两边都升到 float32——结果仍然正确，只是多一次隐式 cast、多一份临时显存；而在 **TensorRT / ONNX 导出**时，这个"一半 fp16、一半 fp32"的比较节点常常导致算子被迫回落到 fp32 精度、甚至 plugin 不支持。**所以补一个 `dtype=final_box_preds.dtype` 是零成本、纯收益的改进**——这是你读完本章就能提出的第一个具体优化点。
  - 【顺带一个更隐蔽的点】`post_center_range` 每次 `decode` 调用都会**重新构造一次 CPU→GPU 的小张量拷贝**。单次开销极小（6 个 float），但它是一次 **H2D 同步拷贝**，在推理引擎里会打断流水。真正的产线写法应该在 `__init__` 里用 `register_buffer` 建一次。**这也是"很久没变"的历史代码的典型气味。**
- 【连接】
  - **YOLO 对照**：YOLO 里没有这一步，因为图像检测的框天然在图像内。3D 检测需要它，因为回归是无界的。
  - **一个重要的缺失：这里没有 NMS。** center 模式下 `decode` 的全部后处理就是「topk + score_threshold + range mask」。mmdet3d 的 CenterPoint 在 `get_task_detections` 里还会做 `circle_nms` 或 `nms_bev`；**这份代码把它省了**（corner 模式有 `self.corner_nms`，center 模式没有）。⚠ 这可能就是 jiazhiwei 在结尾说"现在稳定性是差一些"的技术原因之一——同一目标的相邻峰值会输出多个框。**建议在服务器上确认 head 的 forward 里是否有 3×3 max-pool NMS（CenterNet 的标准做法：`hmax = F.max_pool2d(heat, 3, 1, 1); heat = heat * (hmax == heat)`）。本章画面里没看到这一步。**

---

**[02:03:15]** 原话：「然后返回保存到这个 dict 里面去。」
**[02:03:16]** 原话：「返回。」

- 【代码】`bbox_coders.py:233–267`（帧 02_02_45 / 02_03_19 完整可读）。**⚠ 缩进很关键，我按帧里的实际层级重排了**——那个 `else` 配的是 **L228 的 `if self.post_center_range is not None:`**，不是 `for` 的 `else`（Python 确实有 `for...else` 语法，这里很容易看岔）：
  ```python
  if self.post_center_range is not None:                    # L228
      post_center_range = torch.tensor(...)                 # L229
      mask  = ...                                           # L230
      mask &= ...                                           # L231

      predictions_dicts = []                                # L233  ← 在 if 里面
      for i in range(batch):                                # L234
          cmask = mask[i, :]                                # L235
          if self.score_threshold:                          # L236
              cmask &= thresh_mask[i]                       # L237

          instance_embeddings        = final_instance_embeddings[i, cmask]        # L239
          fusion_instance_embeddings = final_fusion_instance_embeddings[i, cmask] # L240
          labels     = final_clses[i, cmask]                # L241
          scores     = final_scores[i, cmask]               # L242
          boxes3d    = final_box_preds[i, cmask]            # L243
          movements  = final_movements[i, cmask]  if final_movements  is not None else None   # L244
          directions = final_directions[i, cmask] if final_directions is not None else None   # L245

          indexs = inds[i, cmask]                           # L247

          predictions_dict = {                              # L250
              'instance_embeddings':        instance_embeddings,
              'fusion_instance_embeddings': fusion_instance_embeddings,
              'labels':     labels,
              'scores':     scores,
              'bboxes':     boxes3d,
              'movements':  movements,
              'directions': directions,
              'indexs':     indexs
          }
          predictions_dicts.append(predictions_dict)        # L261
  else:                                                     # L262  ← 配 L228 的 if
      raise NotImplementedError(
          'Need to reorganize output as a batch, only '
          'support post_center_range is not None for now!')  # L265

  return predictions_dicts, final_box_preds                 # L267
  ```
  ⚠ **顺带一个真实的脆弱点**：`predictions_dicts` 是在 `if` 分支里才被创建的。如果 `post_center_range is None`，代码会走到 `else` 抛异常——所以运行时不会出事。但**静态分析器 / IDE 会报 "predictions_dicts might be referenced before assignment"**，因为 L267 的 `return` 在 `if/else` 之外。这类"靠 raise 兜底的伪未初始化"在老代码里很常见，读到时不要以为自己看错了。
- 【形状】**这是全章唯一一处形状变成动态的地方**：
  ```
  final_box_preds[i, cmask] : [N_valid, 9]        N_valid ≤ 256，随输入变化（corner 模式是 10）
  labels                    : [N_valid]
  scores                    : [N_valid]
  indexs                    : [N_valid]
  instance_embeddings       : [N_valid, C]
  ```
  `cmask` 是 bool 索引，PyTorch 会做 boolean mask indexing，输出长度取决于 True 的个数。
- 【为什么最后又要在 `centerpoint_head.py` 里填回固定 256 的 tensor？】正因为这里变成了动态形状！**动态 shape 对 TensorRT / 车端推理引擎是灾难**（要么不支持，要么每次重新编译 kernel）。所以下一 Part 会看到：把这些变长结果再塞回一个固定 `[1, 256, 15]` 的容器，空位填 `-1`。**"内部动态、接口定长"是车端代码的标准套路。**
- 【`raise NotImplementedError` 的信息量】这句话直译是"只支持 post_center_range 不为 None 的情况，否则需要重新组织成 batch 输出"。它暴露了一个设计事实：**`cmask` 逐 batch 长度不同，没法直接 stack 成 batch tensor**，所以必须用 list of dict。如果不做范围过滤（`post_center_range=None`），所有 batch 都是 256 个，本可以直接返回 `[B,256,·]` 的 tensor——但那条路没人实现。
- 【`indexs = inds[i, cmask]` 的用途】这是把"这个框来自哪个 BEV 格子"传出去，`centerpoint_head.py:1439-1441` 会用它和 GT 的 `ind` 做匹配：
  ```python
  pred_index = pred['indexs']
  gt_index   = gt_indexs[task_id][batch]
  pos_inds, pos_assigned_gt_inds = torch.where(pred_index[..., None] == gt_index[masks[task_id][batch]])
  ```
  **用"是否落在同一个 BEV 格子"来判定预测框和 GT 框的配对**——这是 CenterPoint 体系里最省事的匹配方式（不用算 IoU，O(N·M) 的相等比较就行）。这份匹配结果供 DenseBEV 的属性任务和 e2e tracker 的训练使用。

---

### 🔨 动手练习 ch12-13：post_center_range 立方体裁剪 + 分数阈值

```python
import torch

K = 256
torch.manual_seed(1)

# 造 256 个"框"：x,y,z 三列 + 其余 7 列随便
boxes  = torch.zeros(1, K, 10)
boxes[0, :, 0] = torch.empty(K).uniform_(-120, 130)   # x：故意超出 [-83.8, 95.4]
boxes[0, :, 1] = torch.empty(K).uniform_(-60, 60)     # y：故意超出 [-44.8, 44.8]
boxes[0, :, 2] = torch.empty(K).uniform_(-8, 8)       # z
scores = torch.rand(1, K)

post_center_range = [-83.8, -44.8, -5.0, 95.4, 44.8, 3.0]   # [xyz_min, xyz_max]
pcr = torch.tensor(post_center_range, dtype=boxes.dtype)

mask  = (boxes[..., :3] >= pcr[:3]).all(2)
mask &= (boxes[..., :3] <= pcr[3:]).all(2)
print("范围内的框数 :", mask.sum().item(), "/", K)

score_threshold = 0.3
thresh_mask = scores > score_threshold
print("过阈值的框数 :", thresh_mask.sum().item(), "/", K)

cmask = mask[0] & thresh_mask[0]
print("最终保留     :", cmask.sum().item(), "/", K)

kept = boxes[0, cmask]
print("kept.shape   :", tuple(kept.shape))    # (N_valid, 10) —— 动态形状！

# ---- 演示 .all(2) 的作用 ----
one_dim_only = (boxes[..., 0:1] >= pcr[0:1]).all(2)
print("\n只看 x 维保留 :", one_dim_only.sum().item(),
      " vs  xyz 全看保留 :", mask.sum().item())
# 预期：只看 x 会明显多留一些（y 或 z 超界的漏网了）

# ---- 车端痛点演示：动态 shape ----
for seed in range(3):
    torch.manual_seed(seed)
    s = torch.rand(1, K)
    m = (s > 0.3)[0] & mask[0]
    print(f"seed={seed} 输出长度 = {m.sum().item()}")
# 预期：三次长度都不同 —— 这就是为什么最后还要填回固定的 [1,256,15]
```

### 【小结】

1. `post_center_range` 是 6 元组，`[:3]` 是 xyz 下界、`[3:]` 是上界，用 `.all(2)` 归约成逐框 bool；**⚠ 讲者只说了 x/y，实际 z 也在过滤内**。
2. `score_threshold` 与范围 mask 用 `&` 叠加，逐 batch 做 bool 索引，**输出变成动态长度**，所以只能返回 list of dict，代码里还有一句 `raise NotImplementedError` 明说"不支持另一条路"。
3. center 模式下**没有任何 NMS**，只有 topk + 阈值 + 范围裁剪；⚠ 这很可能是结尾"稳定性差一些"的技术根源之一，值得回服务器核实 head 里有没有 max-pool NMS。

---

# Part 12-10｜15 维输出：Obj 枚举、-1 初始化、wiki 里的标签规范（02:03:26 – 02:04:44）

### 本段在讲什么

回到 `centerpoint_head.py`，把 `decode` 返回的变长 `predictions_dicts`，逐条填进一个预分配好的 `[1, 256, 15]` 定长 tensor。这 15 列的语义由一个 `Obj(Enum)` 枚举定义，讲者还特地切到浏览器打开了内部 wiki 页把表格给大家看——**这是本章最有含金量的一分钟**，因为它把整个模型的"对外接口"钉死了。

- **输入**：`temp`（= `predictions_dicts`，list of dict，变长）
- **输出**：`pred_bboxes[task_id]`，`[1, 256, 15]`，未填处为 `-1`
- **流水线位置**：`centerpoint_head.py:1408–1435`

---

**[02:03:26]** 原话：「对，这里的 temp 就是我们最终解析完的一个 box。然后……然后会放到我们自己预设的一个 15 维……这下面这些呢就是把它放到对应的我们预设初始化的一个 15 维的一个 tensor 里面去。」

> **重点句，五角度全给。**

- 【直译】decode 出来的是变长结果，现在把它们抄进一个提前开好的定长 `[1, 256, 15]` 数组里。
- 【代码】`centerpoint_head.py:1408–1414`（帧 02:04:37 是全章最清晰的一帧，逐字可读）：
  ```python
  # concat each batch
  dense_bev_mid_feat_list.append(preds_dicts[0][task_id]['bev_mid_feat_list']
                                 if 'bev_mid_feat_list' in preds_dicts[0][task_id]
                                 else preds_dicts[0][task_id][...])
  dense_bev_feat_list.append(preds_dicts[0][task_id]['used_bev_feat'])
  instance_embeddings_list.append(
      preds_dicts[0][task_id]['used_bev_feat'].new_zeros(
          batch_size, self.bbox_coder.max_num, preds_dicts[0][0]['used_bev_feat'].shape[1]))
  fusion_instance_embeddings_list.append(
      preds_dicts[0][task_id]['fusion_feat'].new_zeros(
          batch_size, self.bbox_coder.max_num, preds_dicts[0][0]['fusion_feat'].shape[1]))
  pred_bboxes.append(
      preds_dicts[0][0]['heatmap'].new_ones(
          [batch_size, self.bbox_coder.max_num, self.code_dim]) * -1)     # ★★★
  detect_sampling_result_list.append([])
  ```
- 【形状】`pred_bboxes[task_id] : [1, 256, 15]`（`batch_size=1`、`max_num=256`、`code_dim=15`）。
- 【★ 为什么是 `new_ones(...) * -1` 而不是 `new_zeros`？】**这是本章最容易被忽略、但工程上最重要的一个细节。**
  - `new_zeros` 会让空位是 0.0。但 0.0 是一个**合法的属性值**：x=0 表示"就在自车正下方"、label=0 表示"车"、conf=0 表示"零置信度"。下游没法区分"这个槽位是空的"和"这里真有个 conf=0 的车"。
  - `-1` 则在所有列上都是**非法值**：label 没有 -1 类、conf 不可能为负、w/l/h 不可能为负。**所以 -1 是一个干净的哨兵值（sentinel）。**
  - 而且它和 wiki 里 `dir_cls` 的定义"**-1 无效方向**"完全一致——整个系统用同一个哨兵约定。
  - 唯一的例外：x/y/z/vx/vy 是可以取到 -1 的。所以下游判"这一行是不是空槽"应该看 **conf 列（第 9 列）是不是 -1**，而不是看 x 列。
  - 【实践建议】你以后自己写这种定长输出接口，一律用 `-1` 或 `NaN` 填空位，别用 0。这是把"未初始化"和"真值 0"区分开的唯一办法。
- 【`new_ones` 的另一层用意】`new_ones` 继承源张量的 **device 和 dtype**（这里从 `heatmap` 继承 → cuda:0 / float32），不需要显式写 `device=` 和 `dtype=`。这是 PyTorch 的 `new_*` 系列 API 的标准用法，比 `torch.ones(..., device=x.device, dtype=x.dtype)` 更简洁、更不容易出错。
- 【连接】对比 `instance_embeddings_list` 用的是 `new_zeros`——**embedding 用 0 填、box 用 -1 填**。为什么不统一？因为 embedding 的每一维本来就没有"非法值"的概念（特征向量取任何实数都合法），0 只是"没有信息"的中性值，正好也是很多下游模块（attention mask）默认的填充值。

---

**[02:03:48]** 原话：「这 15 维呢就是对应了我们的这个……」
**[02:03:53]** 原话：「看一下这个。」

- 【直译】"这 15 维到底是哪 15 个量？我给你们翻文档看看。"——他自己也没背下来，要现查。
- 【画面（帧 02_03_57 / 02_04_25 逐格核对）】讲者切到 Chrome，打开内部 wiki 页 `...wei.com/domains/153/wiki/4665/WIKI202312012475625`。页面内导航面包屑依次是 `ADS-XX知识库 / 个人空间 / 数据 / 模型训练相关 / label标签`，右侧大纲栏只有两节：`1. gt`、`2. pred`。浏览器书签栏能看到 `Object detection t...`、`The latest in Mach...`、`draw.io`、`引擎`、`工作`、`主动安全`、`work项目`、`wiki资料`、`会议记录`、`ADS各业务`——**这排书签本身就是一张该团队日常工作面的切片。**
- 【为什么必须去看 wiki——这一分钟的分量比它看起来重得多】
  1. **15 维是跨部门契约，不是模块内部约定。** 感知输出这 15 列，下游的融合 / 预测 / 规划 / 回灌评测都按这 15 列解析。任何一列的语义漂移都会静默地传播到整车。
  2. **它不在代码里。** 代码里只有 `self.code_dim`（一个数字 15）和 `Obj(Enum)`（一串列号），**没有任何一行注释说明"第 3 列是宽不是长"**。语义只活在 wiki 上。
  3. **所以"读懂这份代码"这件事，靠读代码是读不完的。** 这是学术代码和产线代码最大的体感差别：论文代码里 `bbox` 的定义写在同一个文件的 docstring 里；产线代码里它写在一个你需要内网权限才能打开的页面上。**你转岗后第一周要做的第一件事，就是把这类 wiki 页找齐并存下来。**
- 【为什么是 15 而不是别的数】从 wiki 表反推：几何 7（x,y,z,w,l,h,ry）+ 语义 3（label, tag, conf）+ 预留 1 + 属性 2（dir_cls, mov）+ 运动 2（vx, vy）= **15**。**注意它是"当前需要的 12 个 + 1 个已废弃(tag) + 1 个预留 + 对齐"凑出来的数**，不是什么整齐的设计。工业接口的维数几乎总是这样长出来的。
- 【连接】对照你熟悉的 nuScenes：官方评测的 box 只有 `translation(3) + size(3) + rotation(4, 四元数) + velocity(2) + detection_name + detection_score + attribute_name`。**DenseBEV 用 `ry` 一个标量替代了四元数（BEV 下只有 yaw 有意义）、用 `label` 整数替代了字符串类名、多出了 `dir_cls / mov / tag / conf` 四个量产特有的字段。** 这个差集就是"量产接口 vs 学术接口"的全部距离。

---

**[02:03:56]** 原话：「对应的 15 维就是这些，就是对应的我们的 xyz、长……」
**[02:04:03]** 原话：「w……」
**[02:04:05]** 原话：「……l、h。」
**[02:04:06]** 原话：「然后以及这个是通过 sinθ 和 cosθ 算的一个 yaw。」
**[02:04:13]** 原话：「然后以及 label。然后这个 tags 是一个只是一个 tags，当前是没有用到的。」
**[02:04:19]** 原话：「然后还有的话是我们的一个置信度。」
**[02:04:22]** 原话：「然后一个这些朝向的一个……」
**[02:04:25]** 原话：「……朝向的一个分类，以及动静的一个分类，然后 vx vy，主要是这 15 维。」

> **这是本章的"接口定义"时刻。我把 wiki 页面上的两张表逐格抄了下来。**

**wiki 表 1 —— GT 标签（11 维）**

```
ret_dict['obj_label']         = obj_label:        100*11
ret_dict['obj_state']         = obj_state:        100*1     动静
ret_dict['obj_label_source']  = obj_label_source: 100*1     伪标签、人工标签
ret_dict['obj_dir_cls_label'] = obj_real_yaw:     100*1     原始yaw
```

| 列 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 含义 | x | y | z | w | l | h | ry (yaw) | label | 0 | vx | vy |

**wiki 表 2 —— 预测输出（15 维）** ← 这就是 `pred_bboxes` 的 `code_dim=15`

| 列 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | … | -4 | -3 | -2 | -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 含义 | x | y | z | w | l | h | ry (yaw) | label | tag | conf | (10) | dir_cls | move | vx | vy |

⚠ **名字对不齐的小坑**：wiki 表头把倒数第 3 列写作 **`move`**，但 `centerpoint_head.py` L1430 / L1434 里实际用的是 **`Obj.mov.value`**（帧 02_04_37 逐字可读）。所以枚举成员名是 `mov`，wiki 表头的 `move` 只是人写文档时的顺手拼写。**你以后照着 wiki 敲 `Obj.move` 会直接 `AttributeError`。**

对应的 `Obj(Enum)`（wiki 页里贴了代码截图，帧 02_04_25 可辨认结构；成员名以 `centerpoint_head.py` 的调用处为准）：

```python
class Obj(names.Enum):
    """Column of obj saved in test result "xxx.pkl" """
    x       = 0
    y       = 1
    z       = 2
    w       = 3
    l       = 4
    h       = 5
    ry      = 6
    label   = 7
    tag     = 8
    conf    = 9
    dir_cls = -4
    mov     = -3          # ⚠ 帧证：代码里用的是 Obj.mov.value，不是 Obj.move
    vx      = -2
    vy      = -1
```

- 【关键观察 1：正索引 + 负索引混用】前 10 列用正索引 0..9，后 4 列用负索引 -4..-1。**中间的第 10 列没有名字**（表里是"…"）。
  - 15 列里：`0..9` 用掉 10 列，`-4..-1` = `11..14` 用掉 4 列，**第 10 列是空的/预留的**。
  - 为什么要这么设计？**因为负索引让"尾部字段"与 `code_dim` 解耦**：将来如果中间要插一个新字段（比如加个 `rot_short`），只需要 `code_dim` 从 15 改成 16，`dir_cls/move/vx/vy` 的负索引不用动，下游解析代码也不用改。**这是一个很聪明的向前兼容设计。**
  - ⚠ 第 10 列的实际含义画面上没显示（wiki 表格里就是"…"）。我的推断：**预留位**，或者是 `rot_short`（corner 模式下的短边朝向），或者是某个历史字段。建议在服务器上打印 `pred_bboxes[0, :5, 10]` 看看是不是恒为 -1。
- 【关键观察 2：GT 是 11 维、pred 是 15 维】两者**不能直接相减**。GT 缺 tag/conf/dir_cls/move（GT 的动静在 `obj_state` 单独一列、GT 的朝向在 `obj_dir_cls_label` 单独一列）。评测时要按列名对齐，不能按位置。
- 【关键观察 3：GT 每帧最多 100 个目标（`100*11`），pred 最多 256 个】这解释了 `centerpoint_head.py:1442` 那句注释里的"原始 100 个目标"。**GT 100 / pred 256 的比例（2.56 倍冗余）是为了保证召回。**
- 【关键观察 4：`obj_label_source` 区分"伪标签 / 人工标签"】说明这套数据里混了自动标注（伪标签）和人工标注。训练时大概率会按来源加不同权重。**这是量产数据链的典型做法，学术数据集里见不到。**

**wiki 页上关于 yaw 的四条说明（逐字抄录）：**

```
从标注中读取的yaw（弧度制）：-np.pi~np.pi;  (-180°~180°)
make_real_yaw中会生成real_yaw（弧度制）：-np.pi~np.pi;  (-180°~180°)
process_list中read_heading会读取yaw，把-np.pi~np.pi clip到 -np.pi/2~np.pi/2;（训练监督使用）
朝向使用real_yaw (-np.pi~np.pi)：(np.abs(((real_yaw + np.pi) % (2 * np.pi)) - np.pi) > np.pi / 2).int()，
                                 1代表x正方向，0代表x负方向，-1无效方向;
```

- 【这四条在讲什么——非常重要】
  1. 原始标注 yaw 是完整 360°（`-π ~ π`）。
  2. **但训练回归头只监督"半圈"**：`read_heading` 把 yaw clip 到 `-π/2 ~ π/2`。也就是说 **`rot` 分支（sin/cos）学的是"车身轴线方向"，不区分车头车尾**。
  3. 车头车尾的区分交给 `dir_cls` 这个**二分类**头：判据是 `|wrap(real_yaw)| > π/2`。
  4. 解码时：`最终 yaw = rot（半圈）+ dir_cls ? π : 0`。
- 【为什么要拆成"半圈回归 + 二分类"】
  - 车身轴线（长边方向）在视觉/点云上非常好判断（一个长方形的长边）；但"哪头是车头"需要更精细的线索（车灯、后视镜、行驶方向）。
  - 把简单的部分交给回归、困难的部分交给分类，**分类头即使错了也只导致 180° 翻转，不会污染轴线估计**。如果混在一起回归，网络会在两个模态之间摇摆，输出一个"平均值"（比如 90°），两边都不对。
  - 这是 SECOND / PointPillars 的经典设计（`dir_offset` + `dir_cls`），DenseBEV 沿用了。
- 【⚠ 一处逻辑疑点】wiki 写 `(|wrap(real_yaw)| > π/2).int()`，然后说"**1 代表 x 正方向，0 代表 x 负方向**"。但从几何看：`|yaw| > π/2` ⟺ `cos(yaw) < 0` ⟺ 朝向向量的 x 分量为负 ⟺ **指向 x 负方向**。所以按字面公式，1 应该对应"x 负方向"，与 wiki 的文字说明相反。
  - 三种可能：(a) wiki 文字写反了（最可能）；(b) 他们的 yaw 是从 -x 轴或 y 轴起算的；(c) 由于 Part 12-8 发现 BEV 的行索引方向与自车 +x 相反，这里的"x 正方向"指的是**图像行方向**而非自车 x 轴。
  - **⚠ 强烈建议在服务器上核实**：打印几个已知朝向的 GT，看 `obj_dir_cls_label` 到底是 0 还是 1。这个符号搞反会导致所有车"倒着开"。

---

**[02:04:32]** 原话：「在这里就主要就是把它填充到这 15 维里面去。」

- 【代码】`centerpoint_head.py:1415–1435`（帧 02:04:37 逐字可读，是本章最有价值的一帧）：
  ```python
  for batch, pred in enumerate(temp):
      if IS_CORNER:
          pred = self.corner_nms(pred)
          pred = self.obj_corner_to_center(pred)

      num_pred = pred['bboxes'].shape[0]

      pred_bboxes[task_id][batch, :num_pred, :Obj.ry.value + 1] = pred['bboxes'][:, :Obj.ry.value + 1]
      pred_bboxes[task_id][batch, :num_pred,  Obj.label.value]  = pred['labels']
      pred_bboxes[task_id][batch, :num_pred,  Obj.conf.value]   = pred['scores']

      if batch_vel is not None:
          pred_bboxes[task_id][batch, :num_pred, -2:] = pred['bboxes'][:, -2:]

      if (batch_movement is not None) and (batch_direction is None):
          pred_bboxes[task_id][batch, :num_pred, Obj.mov.value] = pred['movements'][:, 0]
      if (batch_movement is None) and (batch_direction is not None):
          pred_bboxes[task_id][batch, :num_pred, Obj.dir_cls.value + 1] = pred['directions'][:, 0]
      if (batch_movement is not None) and (batch_direction is not None):
          pred_bboxes[task_id][batch, :num_pred, Obj.mov.value]     = pred['movements'][:, 0]
          pred_bboxes[task_id][batch, :num_pred, Obj.dir_cls.value] = pred['directions'][:, 0]
  ```
- 【逐行拆解】
  - `num_pred = pred['bboxes'].shape[0]`：这一帧实际有效的框数（≤ 256，被 Part 12-9 的 mask 过滤过）。
  - `[:num_pred, :Obj.ry.value + 1]` = `[:num_pred, :7]`：**一次性把 x,y,z,w,l,h,ry 这 7 列整体拷贝**。因为 `final_box_preds` 的前 7 列顺序恰好就是 `xs,ys,hei,dim(3),rot` —— **decode 的 cat 顺序是刻意对齐 Obj 枚举设计的**。这不是巧合。
  - `Obj.label.value = 7`、`Obj.conf.value = 9` 分别填 label 和 score。**第 8 列 tag 没人填，永远是 -1**（讲者说"当前是没有用到的"）。
  - `[-2:]` 填 vx/vy——用负索引，与 `final_box_preds[:, -2:]` 对应。**两边都用负索引，所以 center 模式的 9 列和 corner 模式的 10 列都能正确填**（中间那列 `rot_short` 被自动跳过）。这解释了 Part 12-8 里"列数会随配置变"为什么不影响运行——**代价是 corner 模式下 `rot_short` 解出来后就无人读取，等于白算一遍 `atan2`。**
  - `Obj.mov.value = -3`、`Obj.dir_cls.value = -4`。
- 【⚠ 一处可疑写法】第二个分支：
  ```python
  if (batch_movement is None) and (batch_direction is not None):
      pred_bboxes[...][..., Obj.dir_cls.value + 1] = pred['directions'][:, 0]
  ```
  `Obj.dir_cls.value + 1 = -4 + 1 = -3 = Obj.mov.value`。**也就是说：当只有 direction、没有 movement 时，direction 被写进了 mov 那一列。**
  - 可能的解释 (a)：**有意为之的"列压缩"**——只有一个语义属性时，统一放在 -3 列，下游按"有没有 movement 头"去解释这一列。
  - 可能的解释 (b)：**笔误**，本意应该是 `Obj.dir_cls.value`。
  - 从三个分支的对称性看，第一、三分支都用 `Obj.mov.value` 填 movement、第三分支用 `Obj.dir_cls.value` 填 direction，唯独第二分支用了 `+1`——**如果是有意的，应该写注释；没写注释，(b) 笔误的可能性更大**。⚠ 但 DenseBEV 当前配置 movement 和 direction 都开着（走第三分支），所以这个分支根本没被执行到，bug 也就一直没暴露。**这是典型的"死分支里的潜伏 bug"。**
- 【为什么要用 `Obj.xxx.value` 而不是硬编码数字】枚举让"列号"只在一处定义。将来 `code_dim` 从 15 改到 16，只改枚举即可。**这一点做得很好**——比起代码里散落的 `[..., 7]`、`[..., 9]` 强太多。

---

**[02:04:38]** 原话：「这就是我们解析 box 的一个过程。」
**[02:04:41]** 原话：「然后返回。」
**[02:04:45]** 原话：「这应该这些就……」
**[02:04:48]** 原话：「……解析头应该就说完了。」

- 【代码】回到 `det_head.py`（帧 02:04:50），`get_bboxes` 的结果被这样消费：
  ```python
  result = self.det_head.head.get_bboxes(
      det_output['pred_dict'], gt_inds, record_valid_obj_indexes, masks,
      select_bev_feat=self.select_bev_feat)

  if not self.training and self.det_head.enable_corner_det and self.det_head.infer_mode != 'center':
      result_corner = self.det_head.corner_head.get_bboxes(det_output['pred_dict_corner'], IS_CORNER=True)

      bboxes        = result['obj_pred']
      bboxes_corner = result_corner['obj_pred']
      if self.det_head.infer_mode == 'corner':
          bboxes[bboxes[:, Obj.label.value] == 1, Obj.conf.value] = 0.0
          bboxes[bboxes[:, Obj.label.value] == 4, Obj.conf.value] = 0.0
      elif self.det_head.infer_mode == 'corner_priori':
          bboxes[bboxes[:, Obj.label.value] == 1, Obj.conf.value] -= 0.2
          bboxes[bboxes[:, Obj.label.value] == 4, Obj.conf.value] -= 0.2
          bboxes[bboxes[:, Obj.conf.value] < 0.0, Obj.conf.value] = 0.0
      result['obj_pred'] = torch.cat((bboxes, bboxes_corner), dim=1)

  det_output.update(result)
  ```
  这段讲者没讲（属于 corner 模式），但**信息量很大**：
  - `infer_mode` 有三档：`'center'`（只用中心头）、`'corner'`（第 1、4 类完全交给角点头，中心头的置信度直接清 0）、`'corner_priori'`（中心头的第 1、4 类置信度 -0.2，让角点头优先但不完全压死）。
  - **label 1 和 label 4 被特殊对待** —— 结合"5 类 = car/truck/bus/VRU/…"，1 和 4 大概率是**大车（truck/bus）或者形状特殊的类**。对这类目标，角点检测比中心检测更准（大车中心容易被遮挡，角点更可见）。
  - `-0.2` 这个数是**手调的先验偏置**，非常"产线"。
- 【连接】这段代码解释了为什么前面 `enable_corner_det` 分支到处都是——**DenseBEV 实际上跑的是"中心头 + 角点头"双头集成**，用 `infer_mode` 控制融合策略。这在论文里几乎见不到，是纯工程增益。

---

### 🔨 动手练习 ch12-14：完整复现 15 维输出的组装（含 -1 哨兵与负索引）

```python
import torch
from enum import Enum

class Obj(Enum):
    x = 0; y = 1; z = 2; w = 3; l = 4; h = 5; ry = 6
    label = 7; tag = 8; conf = 9
    dir_cls = -4; mov = -3; vx = -2; vy = -1

B, MAX_NUM, CODE_DIM = 1, 256, 15

# ---------- 预分配：new_ones * -1 ----------
heatmap = torch.rand(B, 5, 448, 224)                 # 只是为了借它的 device/dtype
pred_bboxes = heatmap.new_ones([B, MAX_NUM, CODE_DIM]) * -1
print("初始化值(前3行前5列):\n", pred_bboxes[0, :3, :5])   # 全 -1

# ---------- 模拟 decode 出来的 12 个有效框 ----------
# ★ center 模式下 final_box_preds 是 9 列（帧 02_01_36 坐实）：
#   x, y, z, w, l, h, ry, vx, vy
N = 12
pred = {
    'bboxes':     torch.randn(N, 9),
    'labels':     torch.randint(0, 5, (N,)).float(),
    'scores':     torch.rand(N),
    'movements':  torch.rand(N, 1) + torch.randint(0, 2, (N, 1)).float(),  # prob+class
    'directions': torch.rand(N, 1),
}
num_pred = pred['bboxes'].shape[0]

# ---------- 逐段填充（完全照抄 centerpoint_head.py:1422-1435） ----------
pred_bboxes[0, :num_pred, :Obj.ry.value + 1] = pred['bboxes'][:, :Obj.ry.value + 1]
pred_bboxes[0, :num_pred,  Obj.label.value]  = pred['labels']
pred_bboxes[0, :num_pred,  Obj.conf.value]   = pred['scores']
pred_bboxes[0, :num_pred, -2:]               = pred['bboxes'][:, -2:]
pred_bboxes[0, :num_pred,  Obj.mov.value]     = pred['movements'][:, 0]
pred_bboxes[0, :num_pred,  Obj.dir_cls.value] = pred['directions'][:, 0]

print("\n第0个框的15维 :\n", pred_bboxes[0, 0])
print("\n第8列(tag)   :", pred_bboxes[0, :3, 8].tolist(), " ← 永远是 -1，当前未使用")
print("第10列(预留)  :", pred_bboxes[0, :3, 10].tolist(), " ← 也是 -1")
print("\n空槽位(第12行):\n", pred_bboxes[0, 12])          # 全 -1

# ---------- 下游怎么判"这一行是不是空槽" ----------
valid = pred_bboxes[0, :, Obj.conf.value] >= 0
print("\n有效框数 :", valid.sum().item(), "(期望 12)")
# ⚠ 注意不能用 x 列判断：x 完全可以是负数（车在自车后方）
bad_valid = pred_bboxes[0, :, Obj.x.value] >= 0
print("用 x 列判断会得到 :", bad_valid.sum().item(), "← 错的（约 6，因为一半的 x 是负数）")

# ---------- 关键验证：换成 corner 模式的 10 列，同一段填充代码依然正确 ----------
pred_bboxes2 = heatmap.new_ones([B, MAX_NUM, CODE_DIM]) * -1
bboxes10 = torch.randn(N, 10)          # x,y,z,w,l,h,ry, rot_short, vx,vy  ← 中间多一列
bboxes10[:, :7] = pred['bboxes'][:, :7]
bboxes10[:, -2:] = pred['bboxes'][:, -2:]
pred_bboxes2[0, :num_pred, :Obj.ry.value + 1] = bboxes10[:, :Obj.ry.value + 1]
pred_bboxes2[0, :num_pred, -2:]               = bboxes10[:, -2:]

print("\n9列 vs 10列 填出来的前7列一致 :",
      torch.allclose(pred_bboxes[0, :num_pred, :7], pred_bboxes2[0, :num_pred, :7]))
print("9列 vs 10列 填出来的 vx/vy 一致 :",
      torch.allclose(pred_bboxes[0, :num_pred, -2:], pred_bboxes2[0, :num_pred, -2:]))
print("→ 两个 True：'前缀切片 + 负索引'的写法对中间列数完全免疫")
print("→ 但也意味着 corner 模式那一列 rot_short 被静默丢弃了，从没进过 15 维输出")
```

**你应该看到的**：
1. 第 8 列（tag）和第 10 列永远是 `-1`；
2. 判定有效槽位必须看 **conf 列**，用 x 列会漏掉一半（约 6/12）；
3. 最后两行 `True` —— **`[:7]` + `[-2:]` 这种"两端切片"写法对中间列数免疫**，这既是它的优点（配置切换不用改代码），也是它的陷阱（中间列悄悄消失了没人发现）。

### 【小结】

1. `pred_bboxes = heatmap.new_ones([1, 256, 15]) * -1`：**用 -1 而不是 0 做哨兵**，因为 0 是合法属性值；这与 wiki 里"-1 无效方向"的约定统一。下游判有效性要看 conf 列。
2. 15 列的语义由 `Obj(Enum)` 定义，**前 10 列用正索引、后 4 列用负索引**，中间第 10 列预留；负索引让尾部字段与 `code_dim` 解耦，向前兼容。
3. wiki 揭示了朝向的真实设计：**回归头只监督半圈（yaw clip 到 ±π/2），180° 歧义交给 `dir_cls` 二分类**；⚠ wiki 文字"1 代表 x 正方向"与公式 `|yaw|>π/2` 的几何含义相反，建议回服务器核实。
---

# Part 12-11｜收尾 QA：代码迁移、后续规划、稳定性坦白（02:04:51 – 02:06:37）

### 本段在讲什么

主讲人 `pilei` 讲完解析头，问"大家有没有问题"，沉默约 37 秒后，**另一位同事 `jiazhiwei j00812467` 接过话头做项目层面的收尾**。这 70 秒的信息密度极高——它不是技术讲解，而是**对这套代码的"官方定性"**：从哪来、往哪去、现在什么水平。对你（正在转岗、要判断这套代码值不值得深挖）来说，这段可能比前面 12 分钟的技术细节还重要。

**帧证（右上角"当前说话人"浮层）**：
- 02:02:18 / 02:03:02 / 02:05:06 / 02:05:22 → `pilei p00804189`
- 02:05:28 / 02:05:41 / 02:05:46 / 02:06:27 / 02:06:35 → `jiazhiwei j00812467`

**说话人在 02:05:22–02:05:28 之间切换。** 所以从 `[02:05:28]` 起的所有内容都是第二个人说的。这一点转写稿里完全看不出来，只能靠画面确认。

---

**[02:04:51]** 原话（pilei）：「看一看大家有没有什么问题。」

- 【场景还原】此时画面停在 `det_head.py` L161 附近（`result = self.det_head.head.get_bboxes(...)` 被高亮），Python Console 里堆着一路敲出来的 shape。**接下来 37 秒是静默**（转写稿在 02:04:51 到 02:05:28 之间没有任何内容）。
- 【解读】没人提问。对于一场 2 小时的代码串讲，最后无人提问是常态——信息量太大，听众需要时间消化。**这也是你为什么需要这份逐句精讲。**

---

**[02:05:28]** 原话（jiazhiwei）：「到 ⚠Riverbox 的输出，整个一套代码就走下来了。」

- 【⚠ 转写存疑】"Riverbox" 显然是音译残留。结合语境"到 XXX 的输出，整个一套代码就走下来了"，最可能的原词：
  - **"到 raw box 的输出"**（原始框输出）——发音最接近；
  - **"到 pred box 的输出"**；
  - **"到了 box 的输出"**。
  含义无论如何是一致的：**从数据进网络到 box 出来，整条链路今天全部走完了。**
- 【解读】这是一句"结项陈词"。前 11 章 + 本章 = 完整覆盖了 `forward` + `loss` + `decode` 三条路径。**唯一没覆盖的是 Dataset/DataLoader 和图像 backbone**（视频开头就跳过了）。

---

**[02:05:34]** 原话：「大家看还有啥问题？因为这部分代码其实 ⚠JUD 可能比较熟。」
**[02:05:41]** 原话：「跟 JUD 的非常像，然后有些东西也很久都没有变了。」

- 【⚠ 转写存疑】"JUD" 是本章第二个音译谜团。三种可能：
  - (a) **同事姓名/工号缩写**（"跟 JUD 的（代码）非常像" 读起来最顺）；
  - (b) **另一个项目的代号**——注意 02:05:57 他提到 "Smart"，说明团队里确实用英文代号命名项目；那 JUD 很可能是**另一个平行项目**；
  - (c) 某个开源仓库/内部库的名字。
  从"这部分代码其实 JUD 可能比较熟"这句的**主语是人**（"比较熟"的主体只能是人）来判断，**(a) 同事名字的可能性最大**，即"这部分代码 XX 比较熟悉，跟他那边的非常像"。
- 【技术解读】"有些东西也很久都没有变了" —— 这一句是**理解本章所有"怪代码"的钥匙**：
  - `_topk` 里那两行被注释掉的 `# topk_ys / # topk_xs` 和 `# original:` 标记；
  - `topk_inds % (height * width)` 这个空操作；
  - `topk_ys/topk_xs` 先算后 gather 的冗余；
  - 第 8 列 `tag` 从来没人填；
  - `Obj.dir_cls.value + 1` 那个走不到的分支。
  **这些全都是"很久没变"的历史沉积。** 不是设计得不好，是**没人敢动**——动了要重新验证整条链路，成本远高于收益。

---

**[02:05:49]** 原话：「因为我们是从以前版本迁过来的。」

> **这是本段最重要的一句，五角度全给。**

- 【直译】DenseBEV 这套检测头不是从零写的，是从上一代模型的代码库整体搬过来的。
- 【代码证据链】本章里能直接证明"迁移"的痕迹有五处：
  1. `_topk` 的 docstring 写 `[B, N, W, H]`，实际是 `[B, C, H, W]` —— **抄来时没改 docstring**。
  2. `_topk` 的默认 `K=80` —— 这是 **CenterNet 在 COCO 上的默认值**，跟 BEV 检测毫无关系。
  3. `# original:` 注释 + 上面两行被注释掉的 CenterNet 原版 xs/ys —— **改了但保留了原版做对照**。
  4. `topk_inds % (height * width)` 空操作 —— 从别的 view 写法抄来的。
  5. `_transpose_and_gather_feat` 的 docstring 写 `feat ... with the shape of [B, 2, W, H]`，那个 **`2` 是 CenterNet offset head 的通道数**，这里明明是任意 C。
- 【为什么工业界普遍这么干】
  - **风险控制**：一套跑通过、上过车的解码逻辑，其数值行为、边界情况、导出兼容性都是被验证过的。重写一遍等于把所有踩过的坑再踩一遍。
  - **对齐评测**：只有解码逻辑不变，新旧模型的指标才可比。改了解码，指标涨跌就分不清是模型好了还是解码变了。
  - **代价**：代码里堆满历史注释、死分支、名不副实的 docstring。**读代码时必须有"考古"心态**——先判断这行是活的还是化石。
- 【连接】
  - 你在 4060 上跑的 BEVFusion，其 `centerpoint_bbox_coders.py` 和这份代码 80% 重合。**先把 mmdet3d 那份读透，再读这份，你会发现差异点集中在：x/y 对调、pc_range 减号、instance_embeddings、movement/direction 三处。** 这是最高效的学习路径。
  - **对你面试的意义**：如果被问"你读过工业级检测代码吗"，你可以答"读过一份从 CenterNet 迁移过来的量产 BEV 解码头，能指出它哪些是活代码、哪些是历史残留、迁移时改了哪四个地方"。这比背论文有说服力得多。

---

**[02:05:51]** 原话：「后面我们现在也会基于 head 去做一些修改。」

- 【直译】接下来的迭代重点是**检测头**，不是 backbone、不是融合。
- 【为什么头是下一个重点】站在这套代码的现状看，head 上确实还有大量可做的：
  - **没有 NMS**（Part 12-9 指出）→ 重复框问题；
  - **第 8 列 tag 未使用**、第 10 列预留 → 接口有空间加新属性；
  - **corner 头和 center 头的融合还是手调阈值**（`-0.2`）→ 可以学习化；
  - **`rot_short` 只在 corner 模式用** → 可以推广；
  - **instance_embedding 没有专门的监督** → 可以加对比学习让它更适合 ReID。
- 【连接】这条信息对你**转岗后的第一个任务**有直接价值：如果你进的是这个组，大概率就是在 head 上做事。**把本章读透 = 直接具备接手能力。**

---

**[02:05:57]** 原话：「但是之前在 Smart 上的一些优化可以慢慢回迁。」

- 【直译】"Smart" 是另一个（或上一代）项目/平台，那边做过一些优化，可以往 DenseBEV 这边搬。
- 【解读】注意"**回迁**"这个词——说明 Smart 是**从同一个祖先分叉出去**的另一条线，两边各自演进，现在要把 Smart 的改进合并回来。这是典型的"多平台/多车型共用一套感知代码"的组织形态。
- 【上下文佐证】视频前面（Ch1、Ch2）讲者多次提到 "SPAS"（一个更早的 BEV 项目）："以前 SPAS 里面会去需要取图像的一个原始尺寸"、"以前 SPAS BEV 在这里会因为图像的 Token 太多，所以把侧向针孔相机沿着高度做一个拍平"。**所以这条代码谱系大致是：SPAS → (Smart) → DenseBEV**，每一代都继承前一代的代码骨架，同时留下大量前一代的死代码（FPN 里那些"当前没用到"的分支就是明证）。

---

**[02:06:00]** 原话：「有一些物理量的监督和优化还可以再搞一套。」

- 【直译】现在的监督主要是"几何 + 分类"，还可以加入更多**物理量约束**。
- 【技术展开】"物理量的监督"在 3D 检测里通常指：
  - **运动学一致性**：vx/vy 与相邻帧位置差应该自洽（DenseBEV 有 3 帧时序，完全可以加这个约束）；
  - **尺寸先验**：同一个 track 的同一辆车，w/l/h 应该恒定（时序一致性损失）；
  - **地面约束**：z 与局部地面高度的关系（车轮着地）；
  - **朝向-速度一致性**：非横向滑移的车辆，速度方向应该≈yaw 方向。
- 【为什么现在没做】这些约束都需要**跨帧/跨模块**的信息，实现复杂度高、调试成本大。属于"锦上添花"而非"雪中送炭"，所以排在 head 改造之后。
- 【连接】这类"物理先验注入"是当前 BEV 感知论文的热门方向（如各种 motion-aware / kinematics-consistent loss）。**如果你想在转岗后快速做出成绩，这是一个已经被点名、但还没人做的口子。**

---

**[02:06:07]** 原话：「这个可以后续基于这个框架再优化。」
**[02:06:13]** 原话：「现在稳定性是差一些。」

> **这句"稳定性是差一些"是全场唯一一句自我批评，含金量极高。**

- 【直译】当前这套代码/模型的输出还不够稳。
- 【"稳定性"在 BEV 感知语境下通常指什么】
  1. **帧间跳变**：同一个目标的框在连续帧之间位置/尺寸/朝向抖动（时序融合没做好或 heatmap 峰值不稳）。
  2. **朝向翻转**：`dir_cls` 判错导致车头车尾 180° 反转，帧间来回跳——**这是最刺眼的一类不稳定**。
  3. **重复框 / 漏检闪烁**：没有 NMS 导致同一目标输出多个框（Part 12-9 指出），或者置信度在阈值附近抖动导致目标一帧有一帧无。
  4. **训练不收敛/指标波动**。
- 【本章能提供的技术线索】结合前面读到的代码，至少有**五条**可能贡献了"稳定性差"：
  - **(a) center 模式无 NMS**：heatmap 峰值不够尖时，同一目标占据相邻 2–3 个格子，全被 topk 选中 → 重复框。（Part 12-9）
  - **(b) 半圈回归 + dir_cls 的 180° 翻转风险**：`dir_cls` 是一个 sigmoid 二分类，在 0.5 附近抖动就会造成朝向翻转。（Part 12-10）
  - **(c) movement 的 `prob + class` 打包编码**：解包若用错取整方式（练习 ch12-3 演示过 round 会 100% 出错），动静判断会大面积出错。（Part 12-2）
  - **(d) `atan2` 在 ±π 的分支切割**：正后方目标的 yaw 在 `+179.9°` / `-179.9°` 之间跳，物理没动、数值跳 360°，会骗过任何不做 wrap 的下游平滑逻辑。**这一条和 (b) 长得很像但成因完全不同**：(b) 是分类头判错，(d) 是表示本身不连续——**排查时必须分开看，否则会一直在错误的头上调参**。（Part 12-8）
  - **(e) topk 的并列 tie-break 不确定**：分数并列时 CPU/CUDA/不同版本选出的候选不一定相同（练习 ch12-4 的注记）。模型训得越差、heatmap 越平，这个影响越大——**而"稳定性差"本身往往就伴随着"模型没训透"，两者会互相放大**。
  - 【怎么排这个序】(a) 最容易验证也最容易补（加一行 3×3 max-pool，练习 ch12-15 已给出代码）；(d) 是纯下游改造、零风险；(b) 需要看 dir_cls 的训练曲线；(c) 只要 grep 一下解包处就能排除；(e) 会随着模型训好自然消失。**如果我接手这个问题，就按 a → d → c → b → e 的顺序做。**
- 【解读它的分量】一个负责人在总结会上主动说"稳定性差一些"，通常意味着：**这已经是团队内部共识的已知问题，且短期内不会通过重构解决（否则就不会说"可以后续基于这个框架再优化"）。** 换句话说：**这是留给新人的坑，也是留给新人的机会。**

---

**[02:06:20]** 原话（jiazhiwei）：「那今天就这样。」
**[02:06:31]** 原话（jiazhiwei）：「谢谢大家。」

- 【直译】散会。
- 【帧证】02_06_27 / 02_06_33 / 02_06_35 三帧的右上角说话人浮层仍是 `jiazhiwei j00812467`，所以**收尾这两句也是他说的，不是主讲人 pilei**。整场的最后一句话不属于讲代码的人——这个细节本身就说明了这场会的性质：**它是一次"交接/对齐"会，不是一次教学讲座。**
- 【那 11 秒的空档】02:06:20 到 02:06:31 之间空了 11 秒，转写稿里什么都没有。结合前一句"现在稳定性是差一些"和这句"那今天就这样"，这 11 秒大概率是在等有没有人最后提问。**没有人提。**
- 【这场会到此为止，给你留下的是什么】把 Ch1 到 Ch12 连起来看，这 126.6 分钟覆盖了：
  ```
  FPN → DepthNet → Depth Loss → Lidar/Radar Backbone → RL 融合
  → LSS 投影 → 多视角融合 → RC 融合 → 模态融合 → MemoryManager
  → 时序融合 → BEV UNet → CenterPoint 头(10 分支) → 三层 Loss → Box 解码
  ```
  **没覆盖的只有两块**：Dataset/DataLoader（视频开头就跳过了）和图像 backbone。这两块恰恰是最标准化、最容易自学的部分。**换句话说，这场串讲把"只能从人嘴里学到的那部分"讲完了。**
- 【一句实话】这类内部串讲的信息密度高得离谱，但它的默认听众是"已经在这个项目里干了半年的人"。你不是。**所以对你而言，价值不在于听懂了多少，而在于你现在拿到了一张可以逐条去服务器上验证的清单**（附录 C 的 24 条）。听懂是被动的，验证是主动的——**从这一秒开始，主动权回到你手里了。**

---

### 🔨 动手练习 ch12-15：把本章"稳定性差"的三条猜测做成可复现的小实验

```python
import torch, torch.nn.functional as F

# =========== (a) 无 NMS 导致的重复框 ===========
H, W, K = 448, 224, 256
heat = torch.full((1, 1, H, W), 0.05)
# 造一个"胖峰值"：中心 0.9，四邻 0.85（模拟训练不足、峰值不够尖）
heat[0, 0, 200, 100] = 0.90
for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
    heat[0, 0, 200+dr, 100+dc] = 0.85

sc, idx = torch.topk(heat.view(1, 1, -1), 5)
rows, cols = idx[0,0] // W, idx[0,0] % W
print("=== (a) 无 NMS ===")
print("Top5 分数 :", sc[0,0].tolist())
print("Top5 位置 :", list(zip(rows.tolist(), cols.tolist())))
print("→ 同一个目标被输出了 5 次\n")

# 加上 CenterNet 标准的 3x3 max-pool NMS 再看
hmax = F.max_pool2d(heat, 3, stride=1, padding=1)
heat_nms = heat * (hmax == heat).float()
sc2, idx2 = torch.topk(heat_nms.view(1,1,-1), 5)
print("加 maxpool-NMS 后 Top5 分数 :", sc2[0,0].tolist())
print("→ 只剩 1 个 0.9，其余被压成 0.05（底噪）\n")

# =========== (b) dir_cls 在 0.5 附近抖动造成 180° 翻转 ===========
import math
base_yaw = torch.tensor(0.3)                       # 车身轴线（半圈内）
dir_logit_frames = torch.tensor([0.02, -0.01, 0.03, -0.02, 0.01])   # 帧间微小抖动
print("=== (b) 朝向翻转 ===")
for t, lg in enumerate(dir_logit_frames):
    d = (torch.sigmoid(lg) > 0.5).float()
    yaw = base_yaw + d * math.pi
    print(f"  frame{t}: dir_logit={lg:+.3f} → dir={int(d)} → yaw={yaw:.3f} rad ({math.degrees(yaw):+.1f}°)")
print("→ logit 只抖了 ±0.03，yaw 却在 17° 和 197° 之间反复横跳\n")

# =========== (c) movement 打包解码用错取整 ===========
print("=== (c) movement 解包 ===")
prob = torch.tensor([0.52, 0.68, 0.95, 0.51])
cls  = torch.tensor([0.,   0.,   1.,   1.  ])
packed = prob + cls
print("  packed        :", packed.tolist())
print("  floor 解出类别:", packed.floor().tolist(), " ← 正确", cls.tolist())
print("  round 解出类别:", packed.round().tolist(), " ← 错误")
```

**你应该看到的**：
- (a) 加一行 3×3 max-pool 就能把重复框从 5 个压到 1 个——**这是"稳定性差"最容易补的一刀**；
- (b) `dir_cls` 的 logit 抖 ±0.03，朝向就翻 180°——**这类不稳定必须靠时序滤波/tracker 平滑**；
- (c) 取整方式选错，动静判断直接崩。

### 【小结】

1. **02:05:28 起说话人换成了 `jiazhiwei j00812467`**（帧证），这段是项目层面的定性总结，不是主讲人的技术讲解——转写稿完全看不出来。
2. 三条关键定性：**代码从旧版本（SPAS/Smart 谱系）整体迁移而来**（本章能找到 5 处迁移痕迹）、**后续迭代重点在 head**、**Smart 的优化要回迁 + 补物理量监督**。
3. **"现在稳定性是差一些"是全场唯一一句自我批评**；结合代码可定位到**五条**可能原因：center 模式无 NMS、半圈回归+dir_cls 的 180° 翻转、`atan2` 在 ±π 的分支切割、movement 打包编码的解码脆弱性、topk 并列时的 tie-break 不确定。**建议的排查顺序：a → d → c → b → e**。

---

# 全章总结

## 一、一张图记住整条解码链

```
preds_dict[0]  （10 个稠密分支，全部 [1, C, 448, 224]）
   │
   ├─ heatmap [1,5,448,224] ──sigmoid──► batch_heatmap
   ├─ dim     [1,3,448,224] ──exp────► batch_dim         （训练侧取过 log）
   ├─ rot     [1,2,448,224] ──split──► batch_rots/rotc
   ├─ reg/height/vel/movement/dir_cls …
   └─ used_bev_feat / fusion_feat
   │
   ▼  CenterPointBBoxCoder.decode(...)
   │
   ├─ _topk(heat, K=256)
   │    ├ 第一重：view(1,5,-1)=[1,5,100352] → topk → [1,5,256]
   │    ├ inds % (H*W)                                    ← 空操作
   │    ├ xs = inds // 224 (行=自车 x)   ys = inds % 224 (列=自车 y)
   │    ├ 第二重：view(1,-1)=[1,1280] → topk → [1,256]
   │    ├ clses = topk_ind // 256                          ← 除以 K，不是除以 5
   │    └ inds/ys/xs 各做一次 _gather_feat
   │    ⇒ scores[1,256] inds[1,256] clses[1,256] ys[1,256] xs[1,256]
   │
   ├─ instance_embeddings = _transpose_and_gather_feat(used_bev_feat, inds)   → [1,256,C]
   ├─ reg  = _t_a_g_f(reg, inds) → xs += reg[...,0:1] ; ys += reg[...,1:2]    亚像素
   ├─ rot  = atan2(_t_a_g_f(rot_sine,inds), _t_a_g_f(rot_cosine,inds))        → [1,256,1]
   ├─ hei  → [1,256,1] ; dim → [1,256,3]
   ├─ xs = pc_range[0] − xs·out_size_factor·voxel_size[0]   ★减号！→ 米
   ├─ ys = pc_range[1] − ys·out_size_factor·voxel_size[1]
   ├─ final_box_preds = cat([xs,ys,hei,dim,rot,(rot_short),vel], dim=2) → [1,256,~10]
   ├─ mask = (box[...,:3] ≥ pcr[:3]).all(2) & (box[...,:3] ≤ pcr[3:]).all(2)
   ├─ cmask = mask & (scores > score_threshold)
   └─ predictions_dicts[i] = {bboxes, labels, scores, movements, directions, indexs, embeddings}
   │
   ▼  回到 CenterHead.get_bboxes
   │
   ├─ pred_bboxes = heatmap.new_ones([1,256,15]) * −1        ★哨兵 −1
   ├─ [:num_pred, :7]  ← bboxes[:, :7]     (x,y,z,w,l,h,ry)
   ├─ [:num_pred, 7]   ← labels ;  [:num_pred, 9] ← scores
   ├─ [:num_pred, −2:] ← bboxes[:, −2:]    (vx,vy)
   ├─ [:num_pred, −3]  ← movements  ;  [:num_pred, −4] ← directions
   └─ 第 8 列 tag、第 10 列 预留，恒为 −1
   │
   ▼  det_head.forward  →  (corner 融合) → det_output.update(result) → return
```

## 二、五个必须记住的数字

| 数字 | 含义 | 出处 |
|---|---|---|
| **448 × 224 = 100352** | BEV 网格数 = 单类候选数 | console `scores.shape` |
| **256** | `max_num`，两重 topk 的 K，最终目标数 | console `topk_score.shape` |
| **5 × 256 = 1280** | 第二重 topk 的候选池 | `topk_scores.view(batch,-1)` |
| **0.4 m** | `out_size_factor × voxel_size`，BEV 分辨率 | 448×0.4=179.2、224×0.4=89.6 |
| **15 / 11 / 10** | pred 列数 / gt 列数 / anno_box 通道数 | wiki 表 + console `anno_box.shape` |

## 三、本章最容易记错的五处

1. **`topk_clses = topk_ind // K`，除以 256 不是除以 5**（因为 view 展开是 class-major）。
2. **`xs` 是行、`ys` 是列**，与 CenterNet 原版相反（原版那两行被注释保留在旁边）。
3. **`xs = pc_range[0] − xs·res` 是减号**，说明 `pc_range[0]` 存的是**最大值**，BEV 第 0 行是车头最远处。
4. **`topk_inds`（格子位置，值域 100352）≠ `topk_ind`（候选名次，值域 1280）**，差一个 `s`，用错是静默错误。
5. **`pred_bboxes` 用 `new_ones * -1` 初始化**，判有效性要看 conf 列（第 9 列），不能看 x 列。

## 四、和你已有知识的四条桥

| 本章概念 | 你已经会的东西 |
|---|---|
| `dim = exp(pred)` | YOLO 的 `bw = pw·exp(tw)`；Faster R-CNN 的 `t_w = log(w/w_a)` |
| `xs = grid + reg` | YOLO 的 `bx = σ(tx) + cx` |
| 两重 topk + 无 NMS | YOLO 的 conf 阈值 + NMS 的**替代方案**；BEV 下目标不重叠所以 NMS 价值低 |
| `_transpose_and_gather_feat` | BEVFusion / mmdet3d 里同名同实现，可直接对照阅读 |
| `instance_embeddings` | DETR 的 object query 输出 / MOTR 的 track query 的"廉价版" |

---

# 附录 A：本章涉及代码的完整还原

> 说明：以下代码 90% 由帧内画面逐字读出；`_gather_feat` 的函数体和 `_transpose_and_gather_feat` 的函数体（标 ⚠）是按 CenterNet/mmdet3d 标准实现 + Ch11 帧证（`centerpoint_head.py:1045-1047` 的内联版）还原的。行号来自画面。

### A.1 `centerpoint_bbox_coders.py`（核心）

```python
class CenterPointBBoxCoder:                                              # L11

    def _gather_feat(self, feats, inds, feat_masks=None):                # L44  ⚠函数体还原
        dim   = feats.size(2)
        inds  = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):                                       # L65
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].   # ⚠ docstring 有误
            K (int): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()                        # L80

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)   # L82

        topk_inds = topk_inds % (height * width)                         # L84  空操作

        # topk_ys = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()   # L86
        # topk_xs = (topk_inds % width).int().float()                                            # L87

        # original:                                                      # L89
        topk_xs = (                                                      # L90  行 → 自车 x
            (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()
        )
        topk_ys = (topk_inds % width).int().float()                      # L93  列 → 自车 y

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)          # L95
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()         # L96
        topk_inds  = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(  # L97
            batch, K
        )
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)  # L100
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)  # L101

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs       # L103

    def _transpose_and_gather_feat(self, feat, ind):                     # L105
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].                          # ⚠ 那个 2 是 CenterNet 残留
            ind (torch.Tensor): Indexes with the shape of [B, N].
        Returns:
            ...
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()                     # ⚠ 函数体还原
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def decode(self, used_bev_feat, fusion_feat, heat, rot_sine, rot_cosine,      # L124
               hei, dim, vel, reg=None, task_id=-1, movement=None,
               direction=None, rot_short_sin=None, rot_short_cos=None):
        """
        Args:
            ...
            dim (torch.Tensor): Dim of the boxes with the shape of ...            # L135
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].          # L137
            reg (torch.Tensor): Regression value of the boxes in 2D with          # L138
                the shape of [B, 2, W, H]. Default: None.                         # L139
            task_id (int): Index of task. Default: -1.                            # L140
        Returns:
            list[dict]: Decoded boxes.                                            # L143
        """
        batch, cat, _, _ = heat.size()                                            # L145
        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)            # L147

        # 获取instance_embeddings                                                  # L149
        instance_embeddings = self._transpose_and_gather_feat(used_bev_feat, inds)          # L150
        instance_embeddings = instance_embeddings.view(batch, self.max_num, used_bev_feat.shape[1])

        fusion_instance_embeddings = self._transpose_and_gather_feat(fusion_feat, inds)     # L153
        fusion_instance_embeddings = fusion_instance_embeddings.view(batch, self.max_num, fusion_feat.shape[1])

        # class label                                                              # L156
        clses  = clses.view(batch, self.max_num).float()                           # L157
        scores = scores.view(batch, self.max_num)                                  # L158

        if reg is not None:                                                        # L160
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs  = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys  = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:                                                                      # L165
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        # rotation value and direction label                                       # L169
        rot_sine   = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine   = rot_sine.view(batch, self.max_num, 1)
        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)             # L173
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot        = torch.atan2(rot_sine, rot_cosine)                             # L175

        rot_short = None                                                           # L177
        if rot_short_sin is not None and rot_short_cos is not None:                # L178
            rot_short_sin = self._transpose_and_gather_feat(rot_short_sin, inds)
            rot_short_sin = rot_short_sin.view(batch, self.max_num, 1)
            rot_short_cos = self._transpose_and_gather_feat(rot_short_cos, inds)   # L182
            rot_short_cos = rot_short_cos.view(batch, self.max_num, 1)
            rot_short     = torch.atan2(rot_short_sin, rot_short_cos)              # L184

        # height in the bev                                                        # L186
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box                                                           # L190
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)                                     # L192

        xs = (self.pc_range[0] - xs.view(batch, self.max_num, 1)                   # L194  ★减号
              * self.out_size_factor * self.voxel_size[0])
        ys = (self.pc_range[1] - ys.view(batch, self.max_num, 1)                   # L195
              * self.out_size_factor * self.voxel_size[1])

        # cat bbox属性                                                              # L197
        all_box_attrs = [xs, ys, hei, dim, rot]                                    # L198
        if rot_short is not None:
            all_box_attrs.append(rot_short)

        if vel is not None:                                                        # L202
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            all_box_attrs.append(vel)

        if movement is not None:                                                   # L207
            movements = self._transpose_and_gather_feat(movement, inds)
            movements = movements.view(batch, self.max_num, movement.shape[1])

        if direction is not None:                                                  # L211
            directions = self._transpose_and_gather_feat(direction, inds)
            directions = directions.view(batch, self.max_num, direction.shape[1])

        final_instance_embeddings        = instance_embeddings                     # L216
        final_fusion_instance_embeddings = fusion_instance_embeddings
        final_clses      = clses
        final_scores     = scores
        final_box_preds  = torch.cat(all_box_attrs, dim=2)                         # L220
        final_movements  = movements
        final_directions = directions

        # use score threshold                                                      # L224
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold                      # L226

        if self.post_center_range is not None:                                     # L228
            # ★帧证：只有 device=，没有 dtype=；且参照的是 heat 不是 final_box_preds
            post_center_range = torch.tensor(self.post_center_range,
                                             device=heat.device)                   # L229
            mask  = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)     # L230
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)     # L231

            predictions_dicts = []                                                 # L233
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                instance_embeddings        = final_instance_embeddings[i, cmask]   # L239
                fusion_instance_embeddings = final_fusion_instance_embeddings[i, cmask]
                labels     = final_clses[i, cmask]
                scores     = final_scores[i, cmask]
                boxes3d    = final_box_preds[i, cmask]
                movements  = final_movements[i, cmask]  if final_movements  is not None else None
                directions = final_directions[i, cmask] if final_directions is not None else None

                indexs = inds[i, cmask]                                            # L247

                predictions_dict = {                                               # L250
                    'instance_embeddings':        instance_embeddings,
                    'fusion_instance_embeddings': fusion_instance_embeddings,
                    'labels':     labels,
                    'scores':     scores,
                    'bboxes':     boxes3d,
                    'movements':  movements,
                    'directions': directions,
                    'indexs':     indexs
                }
                predictions_dicts.append(predictions_dict)                         # L261
        else:                                                                      # L262
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')                  # L265

        return predictions_dicts, final_box_preds                                  # L267
```

### A.2 `centerpoint_head.py` — `get_bboxes` 关键片段

```python
class CenterHead(nn.Module):                                                       # L246

    def get_bboxes(self, preds_dicts, gt_indexs=None, record_valid_obj_indexes=None,   # L1325
                   masks=None, select_bev_feat=False):
        """
        Args:
            img_metas (list[dict]): Point cloud and image's meta info.             # L1330
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.                 # L1333
        """
        batch_size = preds_dicts[0][0]['heatmap'].shape[0]                         # L1336

        dense_bev_mid_feat_list         = []                                       # L1338
        dense_bev_feat_list             = []
        instance_embeddings_list        = []
        fusion_instance_embeddings_list = []
        pred_bboxes                     = []
        detect_sampling_result_list     = []                                       # L1343

        for task_id, preds_dict in enumerate(preds_dicts):                         # L1345
            # 取出使用过的bev_feat, 用于后续取instance_embeddings                    # L1346
            used_bev_feat = preds_dict[0]['used_bev_feat']
            fusion_feat   = preds_dict[0]['fusion_feat']

            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()                     # L1350

            batch_reg = preds_dict[0]['reg']                                       # L1352
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:                                                     # L1355
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            if self.enable_corner_det:                                             # L1360
                batch_rots       = preds_dict[0]['rot_long'][:, 0].unsqueeze(1)
                batch_rotc       = preds_dict[0]['rot_long'][:, 1].unsqueeze(1)
                batch_rots_short = preds_dict[0]['rot_short'][:, 0].unsqueeze(1)
                batch_rotc_short = preds_dict[0]['rot_short'][:, 1].unsqueeze(1)
            else:                                                                  # L1365
                batch_rots       = preds_dict[0]['rot'][:, 0].unsqueeze(1)
                batch_rotc       = preds_dict[0]['rot'][:, 1].unsqueeze(1)
                batch_rots_short = None
                batch_rotc_short = None                                            # L1369

            if 'vel' in preds_dict[0]:                                             # L1371
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None

            batch_movement = None                                                  # L1376
            if 'movement' in preds_dict[0]:
                if self.activate_move_two_stg:
                    batch_mov_prob     = F.softmax(preds_dict[0]['mov_two_stage'], dim=1)
                    batch_mov_prob_max = torch.max(batch_mov_prob, dim=1, keepdim=True).values
                    batch_mov_class    = torch.argmax(batch_mov_prob, dim=1, keepdim=True)
                    # tricky here, keep the class and prob at the same time by add them   # L1382
                    batch_movement     = batch_mov_prob_max + batch_mov_class
                else:
                    batch_movement = preds_dict[0]['movement'].sigmoid()           # L1385

            batch_direction = None                                                 # L1387
            if self.dir_cls_task:
                batch_direction = preds_dict[0]['dir_cls'].sigmoid()               # L1389

            temp, origin_bbox_preds = self.bbox_coder.decode(                      # L1391
                used_bev_feat, fusion_feat, batch_heatmap,
                batch_rots, batch_rotc, batch_hei, batch_dim, batch_vel,
                reg=batch_reg, task_id=task_id,
                movement=batch_movement, direction=batch_direction,
                rot_short_sin=batch_rots_short, rot_short_cos=batch_rotc_short)    # L1405

            # concat each batch                                                    # L1408
            dense_bev_mid_feat_list.append(...)                                    # L1409
            dense_bev_feat_list.append(preds_dicts[0][task_id]['used_bev_feat'])    # L1410
            instance_embeddings_list.append(
                preds_dicts[0][task_id]['used_bev_feat'].new_zeros(...))           # L1411
            fusion_instance_embeddings_list.append(
                preds_dicts[0][task_id]['fusion_feat'].new_zeros(...))             # L1412
            pred_bboxes.append(preds_dicts[0][0]['heatmap'].new_ones(              # L1413 ★
                [batch_size, self.bbox_coder.max_num, self.code_dim]) * -1)
            detect_sampling_result_list.append([])                                 # L1414

            for batch, pred in enumerate(temp):                                    # L1415
                if IS_CORNER:
                    pred = self.corner_nms(pred)
                    pred = self.obj_corner_to_center(pred)

                num_pred = pred['bboxes'].shape[0]                                 # L1420

                pred_bboxes[task_id][batch, :num_pred, :Obj.ry.value + 1] = \
                    pred['bboxes'][:, :Obj.ry.value + 1]                           # L1422
                pred_bboxes[task_id][batch, :num_pred, Obj.label.value] = pred['labels']   # L1423
                pred_bboxes[task_id][batch, :num_pred, Obj.conf.value]  = pred['scores']   # L1424

                if batch_vel is not None:                                          # L1426
                    pred_bboxes[task_id][batch, :num_pred, -2:] = pred['bboxes'][:, -2:]

                if (batch_movement is not None) and (batch_direction is None):     # L1429
                    pred_bboxes[task_id][batch, :num_pred, Obj.mov.value] = pred['movements'][:, 0]
                if (batch_movement is None) and (batch_direction is not None):     # L1431
                    pred_bboxes[task_id][batch, :num_pred, Obj.dir_cls.value + 1] = \
                        pred['directions'][:, 0]                                   # ⚠ +1 存疑
                if (batch_movement is not None) and (batch_direction is not None): # L1433
                    pred_bboxes[task_id][batch, :num_pred, Obj.mov.value]     = pred['movements'][:, 0]
                    pred_bboxes[task_id][batch, :num_pred, Obj.dir_cls.value] = pred['directions'][:, 0]

                if not IS_CORNER:                                                  # L1437
                    # 获取匹配关系
                    pred_index = pred['indexs']
                    gt_index   = gt_indexs[task_id][batch]
                    pos_inds, pos_assigned_gt_inds = torch.where(
                        pred_index[..., None] == gt_index[masks[task_id][batch]])   # L1441
                    # record_valid_obj_indexes中记录着的100个目标与get_target后的对应关系 …
```

### A.3 `det_head.py` — 前后夹的两段

```python
class DetHead(BaseModule):                                                          # L16
    def forward(self, *inputs, **kwargs):                                           # L62
        if self.use_fusion_instance_embeddings:                                     # L119
            ...
            # 前向网络                                                               # L125
            det_output = self.det_head(data_dict)
            if self.vis_heatmap:
                self.det_head.vis_heatmap(self.vis_base_dir, self.vis_gt_train, ego_loc=self.ego_loc)

            # 计算loss                                                               # L130
            det_loss = torch.tensor(0.0).to(inputs[0])
            tb_dict  = dict()
            if self.training:
                det_loss, tb_dict, gt_inds, record_valid_obj_indexes, masks = self.det_head.get_loss()
            else:                                                                    # L135
                gt_inds                  = [i[2] for i in data_dict['centerpoint_head_gt']]
                record_valid_obj_indexes = [i[3] for i in data_dict['centerpoint_head_gt']]
                masks                    = [i[4] for i in data_dict['centerpoint_head_gt']]

            # Transpose inds                                                         # L140
            gt_inds = list(map(list, zip(*gt_inds)))
            gt_inds = [torch.stack(gt_inds_) for gt_inds_ in gt_inds]
            # Transpose record_valid_obj_indexes                                     # L143
            record_valid_obj_indexes = list(map(list, zip(*record_valid_obj_indexes)))
            record_valid_obj_indexes = [torch.stack(r_) for r_ in record_valid_obj_indexes]
            # Transpose inds                                                         # L146
            masks = list(map(list, zip(*masks)))
            masks = [torch.stack(masks_) for masks_ in masks]

            # decode预测框、获取匹配关系                                                # L150
            if not self.training or self.export_instance_embeddings:                  # L151
                for i in range(len(det_output['pred_dict'])):
                    for j in range(len(det_output['pred_dict'][i])):
                        det_output['pred_dict'][i][j]['bev_mid_feat_list'] = bev_mid_feat_list
                        det_output['pred_dict'][i][j]['fusion_feat']       = fusion_feat
                        if self.det_head.enable_corner_det:                          # L157
                            det_output['pred_dict_corner'][i][j]['bev_mid_feat_list'] = bev_mid_feat_list
                            det_output['pred_dict_corner'][i][j]['fusion_feat']       = fusion_feat

                result = self.det_head.head.get_bboxes(                              # L161
                    det_output['pred_dict'], gt_inds, record_valid_obj_indexes,
                    masks, select_bev_feat=self.select_bev_feat)

                if (not self.training and self.det_head.enable_corner_det            # L163
                        and self.det_head.infer_mode != 'center'):
                    result_corner = self.det_head.corner_head.get_bboxes(
                        det_output['pred_dict_corner'], IS_CORNER=True)              # L164

                    bboxes        = result['obj_pred']                               # L166
                    bboxes_corner = result_corner['obj_pred']
                    if self.det_head.infer_mode == 'corner':                         # L168
                        bboxes[bboxes[:, Obj.label.value] == 1, Obj.conf.value] = 0.0
                        bboxes[bboxes[:, Obj.label.value] == 4, Obj.conf.value] = 0.0
                    elif self.det_head.infer_mode == 'corner_priori':                # L171
                        bboxes[bboxes[:, Obj.label.value] == 1, Obj.conf.value] -= 0.2
                        bboxes[bboxes[:, Obj.label.value] == 4, Obj.conf.value] -= 0.2
                        bboxes[bboxes[:, Obj.conf.value] < 0.0, Obj.conf.value]  = 0.0
                    result['obj_pred'] = torch.cat((bboxes, bboxes_corner), dim=1)   # L175

                det_output.update(result)                                            # L177

            # 筛选出有效的索引，在decode预测框、获取匹配关系时已经把对应关系从get_target时的     # L179
            # 过滤关系的匹配还原到原始100个目标中。这里可以不再使用（暂时保留）
            valid_obj_indexes_mask = [i != -1 for i in record_valid_obj_indexes[0]]  # L180
            valid_obj_indexes = [data[valid_obj_indexes_mask[i]]
                                 for i, data in enumerate(record_valid_obj_indexes[0])]

            # DenseBEV属性任务：训练阶段的匹配结果，推理阶段的匹配结果                    # L183
            if self.training:
                # rl_sample_mask                                                     # L185
                det_output['rl_sample_mask'] = labels['rl_sample_mask']
                # 训练阶段的匹配结果，instance_embeddings与gt框的对应关系                 # L188
                det_output['detect_sampling_result'] = det_output['detect_sampling_result']
            else:                                                                    # L190
                # 推理阶段的匹配结果，instance_embeddings与预测框的对应关系，
                # 其实已经是一一对应的关系了                                            # L191
                det_output['pred_indexes_mask'] = None                               # L192

            return det_output, det_loss, tb_dict                                     # L194
```

---

# 附录 B：🔨 综合练习 ch12-16 —— 端到端复现整条解码链

> 这是本章的压轴练习。40 行内跑完从 `[1,5,448,224]` 到 `[1,256,15]` 的全流程，用真实尺寸和真实参数。跑通它，你就真的掌握了本章。

```python
import torch, math
from enum import Enum

# ============ 配置（对齐 DenseBEV 实测值）============
B, C_CLS, H, W = 1, 5, 448, 224
MAX_NUM, CODE_DIM = 256, 15
RES = 0.4                                     # out_size_factor * voxel_size
PC_RANGE = [95.4, 44.8]                       # ⚠ 存的是 MAX
POST_CENTER_RANGE = [-83.8, -44.8, -5.0, 95.4, 44.8, 3.0]
SCORE_TH = 0.30

class Obj(Enum):
    x=0; y=1; z=2; w=3; l=4; h=5; ry=6; label=7; tag=8; conf=9
    dir_cls=-4; mov=-3; vx=-2; vy=-1

def gather_feat(feats, inds):
    d = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), d)
    return feats.gather(1, inds)

def tgf(feat, ind):                            # _transpose_and_gather_feat
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return gather_feat(feat, ind)

def topk_cp(scores, K):                        # _topk
    b, cat, h, w = scores.size()
    ts, ti = torch.topk(scores.view(b, cat, -1), K)
    ti = ti % (h * w)
    xs = (ti.float() / torch.tensor(w, dtype=torch.float)).int().float()   # 行 → x
    ys = (ti % w).int().float()                                            # 列 → y
    s, ind = torch.topk(ts.view(b, -1), K)
    cls = (ind / torch.tensor(K, dtype=torch.float)).int()
    ti  = gather_feat(ti.view(b, -1, 1), ind).view(b, K)
    ys  = gather_feat(ys.view(b, -1, 1), ind).view(b, K)
    xs  = gather_feat(xs.view(b, -1, 1), ind).view(b, K)
    return s, ti, cls, ys, xs

# ============ 造一批"网络输出" ============
torch.manual_seed(2024)
preds = {
    'heatmap' : torch.full((B, C_CLS, H, W), -3.0),          # logit≈-3 → sigmoid≈0.047
    'reg'     : torch.rand(B, 2, H, W),                       # ∈[0,1)
    'height'  : torch.randn(B, 1, H, W) * 0.3 - 1.0,
    'dim'     : torch.randn(B, 3, H, W) * 0.05 + torch.tensor([0.64, 1.53, 0.47]).view(1,3,1,1),  # log(w,l,h)
    'rot'     : torch.randn(B, 2, H, W) * 0.1,
    'vel'     : torch.randn(B, 2, H, W),
    'movement': torch.randn(B, 1, H, W),
    'dir_cls' : torch.randn(B, 1, H, W),
}
# 埋 3 个真目标（类别, 行, 列）
for cls_id, r, c in [(0, 200, 112), (0, 60, 150), (3, 380, 40)]:
    preds['heatmap'][0, cls_id, r, c] = 3.0                   # sigmoid≈0.953
    preds['rot'][0, 0, r, c] = math.sin(0.3); preds['rot'][0, 1, r, c] = math.cos(0.3)

# ============ Step 1: 分支还原 ============
heat = preds['heatmap'].sigmoid()
dim_m = torch.exp(preds['dim'])                               # ★ exp 还原
rots, rotc = preds['rot'][:, 0].unsqueeze(1), preds['rot'][:, 1].unsqueeze(1)
print("Step1  heat", tuple(heat.shape), " dim(exp后)", tuple(dim_m.shape))

# ============ Step 2: 双重 topk ============
scores, inds, clses, ys, xs = topk_cp(heat, MAX_NUM)
print("Step2  scores", tuple(scores.shape), "inds", tuple(inds.shape),
      " top3分数", [round(v,3) for v in scores[0,:3].tolist()],
      " top3类别", clses[0,:3].tolist())

# ============ Step 3: gather 属性 + reg 偏移 ============
reg = tgf(preds['reg'], inds).view(B, MAX_NUM, 2)
xs  = xs.view(B, MAX_NUM, 1) + reg[:, :, 0:1]
ys  = ys.view(B, MAX_NUM, 1) + reg[:, :, 1:2]
hei = tgf(preds['height'], inds).view(B, MAX_NUM, 1)
dim3= tgf(dim_m, inds).view(B, MAX_NUM, 3)
rs  = tgf(rots, inds).view(B, MAX_NUM, 1)
rc  = tgf(rotc, inds).view(B, MAX_NUM, 1)
rot = torch.atan2(rs, rc)
vel = tgf(preds['vel'], inds).view(B, MAX_NUM, 2)
mov = tgf(preds['movement'].sigmoid(), inds).view(B, MAX_NUM, 1)
dirc= tgf(preds['dir_cls'].sigmoid(),  inds).view(B, MAX_NUM, 1)
print("Step3  top1 yaw =", round(rot[0,0,0].item(), 4), "(埋的是 0.3)")

# ============ Step 4: 格子 → 自车米制（★减号）============
xs = PC_RANGE[0] - xs * RES
ys = PC_RANGE[1] - ys * RES
box = torch.cat([xs, ys, hei, dim3, rot, vel], dim=2)         # [1,256,9]
print("Step4  box", tuple(box.shape), " top1 (x,y,z) =",
      [round(v,2) for v in box[0,0,:3].tolist()])

# ============ Step 5: 范围 + 阈值过滤 ============
pcr = torch.tensor(POST_CENTER_RANGE, dtype=box.dtype)
m  = (box[..., :3] >= pcr[:3]).all(2) & (box[..., :3] <= pcr[3:]).all(2)
m &= (scores > SCORE_TH)
cmask = m[0]
print("Step5  过滤后剩", cmask.sum().item(), "个（阈值", SCORE_TH, "）")

# ============ Step 6: 填进 [1,256,15]，空位 -1 ============
out = heat.new_ones([B, MAX_NUM, CODE_DIM]) * -1
n   = int(cmask.sum())
out[0, :n, :Obj.ry.value+1] = box[0, cmask][:, :Obj.ry.value+1]
out[0, :n,  Obj.label.value] = clses[0, cmask].float()
out[0, :n,  Obj.conf.value]  = scores[0, cmask]
out[0, :n, -2:]              = box[0, cmask][:, -2:]
out[0, :n,  Obj.mov.value]      = mov[0, cmask][:, 0]
out[0, :n,  Obj.dir_cls.value]  = dirc[0, cmask][:, 0]

print("\nStep6  最终输出", tuple(out.shape))
print("第0个目标 15 维 :")
names = ['x','y','z','w','l','h','ry','label','tag','conf','(rsv)','dir_cls','mov','vx','vy']
for i, nm in enumerate(names):
    print(f"   [{i:2d}] {nm:8s} = {out[0,0,i].item():+.4f}")
print("\n空槽位(第 %d 行) conf =" % n, out[0, n, Obj.conf.value].item(), "← -1 哨兵")

# ================= 实测输出（我在 torch 2.6 上真跑过，逐行贴出）=================
# Step1  heat (1, 5, 448, 224)  dim(exp后) (1, 3, 448, 224)
# Step2  scores (1, 256) inds (1, 256)  top3分数 [0.953, 0.953, 0.953]  top3类别 [0, 0, 3]
# Step3  top1 yaw = 0.3     (埋的是 0.3 —— atan2 精确还原到小数点后 4 位)
# Step4  box (1, 256, 9)  top1 (x,y,z) = [15.3, -0.33, -0.73]   ★ 注意是 9 列不是 10 列
# Step5  过滤后剩 3 个（阈值 0.3 ）
# Step6  最终输出 (1, 256, 15)
#    [ 0] x        = +15.3031     ← 95.4 - (200+reg)*0.4，埋的目标在 row=200
#    [ 1] y        =  -0.3303     ← 44.8 - (112+reg)*0.4，col=112 正好是车道中心
#    [ 2] z        =  -0.7324
#    [ 3] w        =  +1.9316     ← exp(0.64)≈1.90，轿车宽度量级 ✓
#    [ 4] l        =  +4.7135     ← exp(1.53)≈4.62，轿车长度量级 ✓
#    [ 5] h        =  +1.6320     ← exp(0.47)≈1.60 ✓
#    [ 6] ry       =  +0.3000     ← 完美还原
#    [ 7] label    =  +0.0000     ← 类0，注意是 float 不是 int
#    [ 8] tag      =  -1.0000     ← ★ 永远没人填
#    [ 9] conf     =  +0.9526     ← sigmoid(3.0)
#    [10] (rsv)    =  -1.0000     ← ★ 预留列，也永远是 -1
#    [11] dir_cls  =  +0.5268
#    [12] mov      =  +0.6350
#    [13] vx       =  +2.2313
#    [14] vy       =  -1.0475
#    空槽位(第 3 行) conf = -1.0
#
# ★★ 三个必须自己确认的点：
#   (1) box 是 9 列 —— 与 Part 12-8 的帧证结论一致；训练侧 anno_box 的 10 通道
#       比它多的那一列，就是被 atan2 吃掉的 sin/cos。
#   (2) 第 8 列和第 10 列恒为 -1 —— 空位哨兵和"没人填的列"长得一模一样，
#       所以下游判空槽必须看 conf 列。
#   (3) 三个目标分数完全并列（都是 sigmoid(3.0)），topk 的 tie-break 决定了
#       谁排第一。你重跑可能拿到 row=60 那个目标——这不是 bug，见 ch12-4 的并列说明。
```

---

# 附录 C：存疑清单（⚠ 汇总，建议在 4060 服务器上逐条核实）

| # | 存疑点 | 依据 / 我的推断 | 建议核实方式 |
|---|---|---|---|
| 1 | 转写"48乘24"/"4824" | 应为 **448×224**；448×224=100352，与 Ch11"1乘10万0352"及 console `scores.shape=[1,5,448,224]` 互证 | 已确认，无需核实 |
| 2 | 转写"15224" | 应为 `[1, 5, 100352]`（view 后） | 已由 448×224 推定 |
| 3 | 讲者说"除一个 448×224" | 代码是 `%`（取余）不是除；且此处是**空操作** | 已由帧证确认代码 |
| 4 | 讲者说"除上5 / 256 代表分类" | 代码是 `topk_ind / K`（**除以 256**），因 view 展开为 class-major | 已由帧证确认 |
| 5 | "可以说一下 **ZIN**"（01:53:40） | 推断为 `gt_inds` 或 `zip`；画面停在 `det_head.py` L141–L148 的 `zip(*x)` 转置段 | 无法进一步确认，语义不影响 |
| 6 | "到 **Riverbox** 的输出"（02:05:28） | 推断为 "raw box / pred box / box"；语义 = "整条链路走完" | 无法确认 |
| 7 | "**JUD** 可能比较熟"（02:05:34） | 推断为同事姓名（"比较熟"的主语是人）；也可能是平行项目代号（对比 "Smart"） | 问部门同事 |
| 8 | console 里的 `rot_lidar torch.Size([1,2,448,224])` | 帧 01_55_02 放大 5 倍后字形确认为 `rot_lidar`；但 `decode` 侧读的键是 `rot`（帧 01_54_24 L1366-1367）。**最可能是"lidar 坐标系下的 yaw"**，与自车系差一个安装外参 yaw 偏置。若两套并存，Part 12-8 解出的 `rot` 究竟在哪个系里直接决定下游朝向对不对 | `print(preds_dict[0].keys())`，并核对 head 里 `rot` / `rot_lidar` 各自的监督目标 |
| 8b | console 里的 `close_heatmap torch.Size([1,5,448,224])` **（新增）** | 与主 `heatmap` 同形同类别数的**第二张分类图**，但 `get_bboxes` / `decode` 全程未使用。推断为**近场（close-range）辅助分类头**——远近目标高斯核尺度差异大，单出一张近场图是常见解法；也可能是纯 deep-supervision 涨点分支 | `grep -rn "close_heatmap" models/det_head/`，看它在 loss 里怎么用、推理时是否真的丢弃 |
| 9 | ~~`final_box_preds` 到底 9 列还是 10 列~~ **已由帧证解决** | 帧 02_01_36 的 L198–L205 白纸黑字：`all_box_attrs = [xs, ys, hei, dim, rot]`（+`rot_short` if corner）（+`vel` if not None）。**center 模式 = 9 列**，corner = 10。console 里的两个 `10`（`target_box`、`anno_box`）是**训练侧**张量，那里 rot 还是 sin/cos 两列，被 `atan2` 压掉一列才成 9 | 已闭环，无需核实 |
| 10 | 15 维里**第 10 列**的含义 | wiki 表里是 "…"，代码里无人填 → 推断为**预留位**（恒 -1） | `print(pred_bboxes[0,:5,10])` |
| 11 | `Obj.dir_cls.value + 1`（L1432） | `-4+1 = -3 = Obj.mov.value`，把 direction 写进了 mov 列；三分支中唯此处 `+1`，**疑似笔误**；当前配置走不到这个分支 | 看 git blame / 问作者 |
| 12 | wiki 的 dir_cls 语义 | 公式 `(|wrap(yaw)| > π/2).int()` 几何上 = 指向 **-x**，但文字写"1 代表 x 正方向"，**相反** | 打印几条已知朝向的 GT 的 `obj_dir_cls_label` |
| 13 | `pc_range[0]/[1]` 存的是 max 还是 min | 代码是 `pc_range[0] - xs*res`（**减号**），代入 95.4/44.8 能精确还原 179.2/89.6 → 存的是 **max** | 打开 config 看 `pc_range` 的值 |
| 14 | `out_size_factor` 与 `voxel_size` 的具体取值 | 两者乘积必须 = 0.4 m；可能是 (1, 0.4) 或 (2, 0.2) | 打开 config |
| 15 | center 模式是否真的**完全没有 NMS** | 本章画面里 `decode` 只有 topk + 阈值 + 范围 mask，未见 3×3 max-pool NMS；corner 模式有 `self.corner_nms` | 在 head 的 `forward` 里搜 `max_pool2d` |
| 16 | `reg` 的 GT 生成方式 | 由 `else` 分支的 `+0.5` 反推，`reg` 应 ∈ `[0,1)`，即 `center_float - center_int` | 看 `get_targets` |
| 17 | `_gather_feat` / `_transpose_and_gather_feat` 的函数体 | 画面只给了签名+docstring；按 CenterNet 标准实现还原，且有 Ch11 帧证（`centerpoint_head.py:1045-1047` 的内联版）佐证 | 直接打开这两个函数看 |
| 18 | `# original:` 注释指的是上面被注释的两行还是下面生效的两行 | 从"被注释掉的是 CenterNet 图像式 ys/xs"判断，**下面生效的是改过的版本**；无论如何以生效代码为准（xs=行, ys=列） | 看 git 历史 |
| 19 | `used_bev_feat` / `fusion_feat` 的通道数 C | 画面未显示具体值；按这类模型常规为 128 或 256 | `print(used_bev_feat.shape)` |
| 20 | `infer_mode == 'corner'` 时被清零的 label 1 和 4 是哪两类 | 结合"5类 car/truck/bus/VRU/…"推断为大车类；角点检测对大车更稳 | 看 config 的 `class_names` |
| 21 | `post_center_range` 没写 `dtype=`（帧 02_02_45 L229） | 当前 `final_box_preds` 是 float32（帧 01_54_24 tooltip 已证），恰好匹配所以不出错；但开 AMP/fp16 或导出 TRT 时会产生隐式提升、算子回落 | 加一句 `dtype=final_box_preds.dtype`，并考虑改成 `register_buffer` 免去每次 H2D 拷贝 |
| 22 | corner 模式下 `rot_short` 算了却没进 15 维 | `all_box_attrs` 里有它，但 `centerpoint_head.py` 的填充只取 `[:7]` 和 `[-2:]`，`rot_short` 所在的中间列被跳过 | 搜 `rot_short` 在整个仓库里还有没有别的消费方；若没有，就是白算 |
| 23 | `det_head.py:175` 的 `torch.cat((bboxes, bboxes_corner), dim=1)` | 同一段里 `bboxes[:, Obj.label.value]` 的写法暗示 `obj_pred` 是 2D `[N,15]`，而 `dim=1` 拼接又更像是在 3D `[B,N,15]` 上沿目标维拼。两者只能对一个 | `print(result['obj_pred'].shape)`（需开 corner 模式） |

---

**Ch12 完 —— 也是整场 DenseBEV 代码串讲的终点。**

从 Ch1 的 FPN 到这里的 `[1, 256, 15]`，你已经完整走过了一条量产 BEV 感知模型的前向 + 损失 + 解码。剩下的事只有一件：**回到 4060 上，把附录 C 的 24 条逐一核实掉（其中 #1~#4、#9 已由帧证闭环，剩下的才是真要动手的）。** 每核实一条，这份代码就从"别人的"变成一点"你的"。


---
> [[Ch11_Loss全解|← Ch11]] · [[00_总览与脉络|📖 总览]]

