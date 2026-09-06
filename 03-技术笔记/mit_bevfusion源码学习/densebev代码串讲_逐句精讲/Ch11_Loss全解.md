> [[Ch10_BEVUNet与CenterPoint检测头|← Ch10]] · [[00_总览与脉络|📖 总览]] · [[Ch12_Box解码与收尾QA|Ch12 →]]

# Ch11 Loss 全解（01:31:16–01:53:28）

> 本章覆盖校正稿第 1323–1707 行（时间戳 `[01:31:16]` 到 `[01:53:28]`），共 **385 行原话**。
> 精读单帧 32 张（`01_31_58 / 01_32_19 / 01_33_02 / 01_33_35 / 01_34_03 / 01_34_38 / 01_35_19 / 01_36_13 / 01_37_03 / 01_37_58 / 01_38_55 / 01_39_02 / 01_39_33 / 01_40_23 / 01_41_29 / 01_42_14 / 01_43_38 / 01_44_47 / 01_45_05 / 01_46_05 / 01_47_30 / 01_48_20 / 01_48_46 / 01_49_46 / 01_50_45 / 01_51_50 / 01_52_09 / 01_52_31 / 01_52_38 / 01_52_45 / 01_52_52 / 01_53_10`）＋总览 sheet_48。
> 本章所有函数名、变量名、行号、张量形状、数值都是从画面里逐字抠出来的，不是猜的；凡是我推断的地方都打了 ⚠ 并写明推断依据。
>
> **【v2 复核记录】** 本章经过一轮逐行复核：(a) 385 行原话按时间戳做了机器比对，无整句遗漏（口语重复与 Whisper 幻听已合并/标注）；(b) 重新放大精读 16 张单帧（`01_31_58 / 01_34_03 / 01_36_13 / 01_41_29 / 01_42_14 / 01_43_38 / 01_44_47 / 01_45_05 / 01_47_30 / 01_48_20 / 01_48_46 / 01_49_46 / 01_51_50 / 01_52_09 / 01_52_45 / 01_52_52 / 01_53_10`），逐字核对函数名/行号/形状；(c) 修正了 6 处与画面或与 PyTorch 实测不符的内容，均在原位标注"**上一版我写错了**"并给出更正依据。主要更正：第 973 行注释是"把**动静**的loss调大3倍"而非"把静的"；`bbox_loss_total += rot_lidar_w_loss * 0.1` 在第 **1130** 行；`permute` 后**不加 `.contiguous()` 也能 view**（实测不报错，见 §11-11）；`attr_heat_masks` 单元素是单通道 `[B,448,224]`（表 B 已改）。

---

## 0. 本章在地图上的位置

```
[已讲完] FPN → DepthNet → Depth Loss → Lidar/Radar Backbone → RL融合(UNet)
       → LSS投影 → 多视角融合 → RC融合 → 模态融合 → MemoryManager
       → 时序融合(warp) → BEV UNet backbone → CenterPoint 检测头(10个分支)
================== 你在这里 ==================
       → 【Ch11】Loss：get_targets 造 GT → heatmap Focal → close_heatmap Focal
                      → dense 属性 L1/BCE → 目标级 gather_feat + L1 → loss_dict 汇总
==============================================
[下一章] Box 解码（sigmoid → topk 256 → exp(dim) → atan2(sin,cos) → NMS）
```

**输入**：`preds_dicts`（上一章检测头吐出来的 11 个分支，全部是 `[B, C, 448, 224]` 的稠密 BEV 图）＋ 一堆从 DataLoader 直接搬进来的 GT 张量（`obj_label / obj_state / obj_dir_cls_label / sample_mask / valid_lidar / lidar_front_mask / is_cls_only / …`）。

**输出**：一个 `loss_dict`（本次调试跑出来正好 **15 项**），以及在外层 `CenterPointHead.get_loss()` 里对它做过滤求和得到的标量 `loss`。

**本章最核心的一句话**：DenseBEV 的 Loss 是**三层监督并存**——
1. **BEV 稠密层（dense）**：在整张 448×224 的图上、在有目标的那些像素上算分类 Focal 和属性 L1/BCE；
2. **近距离稠密层（close）**：把 BEV 图乘一张"近距离区域 mask"，对近处目标再算一遍分类和 yaw；
3. **目标级层（object-level）**：把 10 个回归分支 concat 成 10 通道，`permute→view` 展平成 `[B, 100352, 10]`，用 `inds` 索引 `gather_feat` 抠出 256 个目标，和 `anno_box` 的 `[B, 256, 10]` 算 L1。

这三层同时存在、互相不替代，是这套代码和开源 CenterPoint（只有第 3 层 + heatmap）最大的区别，也是本章最值得你带走的架构认知。

---

## 0.1 先把"事实表"摆出来（后面每一句都会回来查表）

### 表 A：检测头输出 `net_dict` 的 11 个 key（画面 01:31:58 终端逐字抄录）

讲者在终端里跑了一段：

```python
for k in net_dict:
    print(k, net_dict[k].shape)
```

输出（**这是真实打印，不是我编的**）：

| key | shape | 含义 | 后面被谁用 |
|---|---|---|---|
| `reg` | `[1, 2, 448, 224]` | 中心点在格子内的亚像素偏移 dx, dy | 目标级 `anno_box` 前 2 维 |
| `height` | `[1, 1, 448, 224]` | 中心点 z | `anno_box` 第 2 维 |
| `dim` | `[1, 3, 448, 224]` | log(l), log(w), log(h) | `anno_box` 第 3~5 维 |
| `lidar_rot_weight` | `[1, 1, 448, 224]` | "lidar 朝向可信度"权重图 | `rot_lidar_w_loss`（Focal） |
| `rot` | `[1, 2, 448, 224]` | sinθ, cosθ | dense rot Loss + `anno_box` 第 6~7 维 |
| `vel` | `[1, 2, 448, 224]` | vx, vy | dense vel Loss + `anno_box` 第 8~9 维 |
| `dir_cls` | `[1, 1, 448, 224]` | 朝向（前后 180°）二分类 logit | `loss_dir_cls` |
| `movement` | `[1, 1, 448, 224]` | 动/静二分类 logit | dense BCE + 目标级 BCE |
| `rot_lidar` | `[1, 2, 448, 224]` | 由 lidar 前向区域增强的 sinθ, cosθ | `rot_lidar_loss` |
| `close_heatmap` | `[1, 5, 448, 224]` | 近距离分类热图 | `loss_close_heatmap` |
| `heatmap` | `[1, 5, 448, 224]` | 主分类热图（5 类） | `loss_heatmap` |

通道数总和 = 2+1+3+1+2+2+1+1+2+5+5 = **25**。这 25 个通道就是本章要监督的全部东西。

> 【连接】对照 BEVFusion / mmdet3d 的 `CenterHead`：开源版 `common_heads = dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2))` 再加一个 heatmap，一共 5+1 个分支。DenseBEV 在此之上多加了 `dir_cls / movement / rot_lidar / lidar_rot_weight / close_heatmap` 五个分支——**多出来的全是量产车需要而学术数据集不给的东西**（近距离精度、动静状态、朝向 180° 歧义、lidar 可信度）。

### 表 B：`get_targets` 的 9 个返回值（画面 01:33:02，`centerpoint_head.py` 第 519 行）

```python
return heatmaps, anno_boxes, inds, record_valid_obj_indexes, masks, \
       attr_heats, attr_heat_masks, make_concats(movement), make_concats(movement_weight)
```

| 返回值 | 实测 shape | 载体 | 干什么 |
|---|---|---|---|
| `heatmaps` | `[1, 5, 448, 224]` | **BEV 图** | 分类 GT（高斯峰，峰顶=1.0） |
| `anno_boxes` | `[1, 256, 10]` | **目标级** | 10 维回归 GT（xyz+lwh+sincos+vxvy） |
| `inds` | `[1, 256]`（int） | 索引 | 每个目标在展平后 BEV 上的位置（实测首元素 **38416**） |
| `record_valid_obj_indexes` | `[1, 256]` | 索引 | 排序/筛选前后原始 label 的对应关系 |
| `masks` | `[1, 256]`（0/1） | 目标级 | 哪些槽位是真目标（每帧目标数不同，统一补到 256） |
| `attr_heats` | `[1, 9, 448, 224]` | **BEV 图** | dense 属性 GT（9 通道，见表 C） |
| `attr_heat_masks` | `[1, 448, 224]`（**单通道**，见 §11-3 帧证） | **BEV 图** | dense 属性有效性 mask（9 个属性共用同一张） |
| `movement` | `[1, 256, 1]` ⚠ | 目标级 | 动静二分类 GT |
| `movement_weight` | `[1, 256, 1]` ⚠ | 目标级 | 动静 Loss 权重 |

### 表 C：`attr_heats` 9 个通道的确切 layout（由代码三处交叉确认）

这张表是本章我最花力气推出来的东西，三条证据链：

1. 画面 01:42:14 第 872 行：`rot_gt = attr_heats[task_id][:, :2]` → 通道 0,1 = sinθ, cosθ
2. 画面 01:42:14 第 875 行：`v_gt = attr_heats[task_id][:, 2:4]` → 通道 2,3 = vx, vy
3. 画面 01:47:30 第 968 行**代码注释原文**：`# attr_heats[task_id][:, idx] 第4维是二分类动静类别，第5维是二分类loss权重`
4. 画面 01:47:30 第 981 行**代码注释原文**：`# attr_heats[task_id][:, idx] 第6维是四分类动静类别，第7维是四分类loss权重`
5. 画面 01:48:20 第 995 行：`dir_cls_task_idx = (6 if self.enable_corner_det else 4) + (4 if self.activate_move else 0)` → 当前配置 = 4+4 = **8**

| 通道 | 内容 | 取值/约定 |
|---|---|---|
| 0 | rot sinθ | 直接是 sin 值 |
| 1 | rot cosθ | 直接是 cos 值 |
| 2 | vel vx | m/s |
| 3 | vel vy | m/s |
| 4 | 动静**二分类**类别 | 0=无效；1=静；2=动（loss 里做 `-1.0` 变成 0/1） |
| 5 | 动静二分类 **loss 权重** | 由 DataLoader 预先烧进数值；代码注释原文是"**把动静的loss调大3倍**"（帧证 01:47:30 第 973 行，逐字放大核对过），⚠ 注释没说是"静的×3"还是"整个动静任务×3"，见 §11-10 |
| 6 | 动静**四分类**类别 | 有效区间 `[1, 5)` |
| 7 | 动静四分类 **loss 权重** | — |
| 8 | `dir_cls` 朝向分类 | 目标是 `clamp(gt-1, min=0)` |

这也就解释了讲者在 01:35:38 那句"**动静的四个值**"——他说的不是"动静有 4 类"，而是**动静一共占了 4 个通道**（二分类类别+二分类权重+四分类类别+四分类权重）。所以他数出来的 2(朝向)+2(速度)+4(动静)+1(朝向分类) = 9，和 `attr_heats.shape = [1, 9, 448, 224]` 严丝合缝。✅

### 表 D：`anno_box` / `target_box` 的 10 维 layout（画面 01:48:46 第 1021~1025 行）

```python
preds_dict[0]['anno_box'] = torch.cat(
    (preds_dict[0]['reg'], preds_dict[0]['height'],
     preds_dict[0]['dim'], preds_dict[0]['rot'],
     preds_dict[0]['vel']), dim=1)
```

| 维度 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 内容 | dx | dy | z | log l | log w | log h | sinθ | cosθ | vx | vy |
| 来源分支 | `reg` | `reg` | `height` | `dim` | `dim` | `dim` | `rot` | `rot` | `vel` | `vel` |

后面代码里的 `pred[..., -4:-2]` 就是 `[6:8]` = sinθ,cosθ；`pred[..., -2:]` 就是 `[8:10]` = vx,vy；`loss_bbox[..., :6]` 就是 dx,dy,z,l,w,h 这 6 维。**这个 layout 记熟，后面每一次切片都靠它。**

### 表 E：真实跑出来的 `loss_dict`（画面 01:52:52 调试器变量面板，`len() = 15`）

| # | key | 值 | grad_fn | 是否计入总 loss |
|---|---|---|---|---|
| 1 | `task0.loss_movement` | 0.0204 | DivBackward | ✅ |
| 2 | `task0.loss_p_reg_loc` | 1.6824 | SumBackward | ❌（仅日志） |
| 3 | `task0.loss_p_height` | 0.8559 | SumBackward | ❌ |
| 4 | `task0.loss_p_box_size` | 2.6234 | SumBackward | ❌ |
| 5 | `task0.loss_p_rot` | **134.1376** | SumBackward | ❌ |
| 6 | `task0.loss_p_vel` | 3.8619 | SumBackward | ❌ |
| 7 | `task0.loss_p_vel_dense` | 7.9563 | MulBackward | ❌（已并进 loss_bbox） |
| 8 | `task0.loss_p_rot_dense` | 30.5692 | AddBackward | ❌（已并进 loss_bbox） |
| 9 | `task0.loss_p_lidar_rot_w` | 1.6938 | MulBackward | ❌（已 ×0.1 并进 loss_bbox） |
| 10 | `task0.loss_movement_dense` | 6.3973 | DivBackward | ✅ |
| 11 | `task0.loss_dir_cls` | 1.5016 | MulBackward | ✅ |
| 12 | `task0.loss_heatmap` | 5.0235 | MulBackward | ✅ |
| 13 | `task0.loss_bbox` | 43.8560 | AddBackward | ✅ |
| 14 | `task0.loss_close_heatmap` | **13126.7217** | 有 grad_fn（面板右侧被截断，看不到具体算子） | ✅ ⚠ |
| 15 | `task0.loss_p_rl_percentage` | 1.0（python float） | 无梯度 | ❌ |

**我用这张表做了一次算术校验，完美闭合**（这是全章最硬的一条证据）：

```
loss_bbox = loss_bbox[..., :6].sum() + vel_dense_loss + rot_dense_loss + rot_lidar_w_loss * 0.1
          = (1.6824 + 0.8559 + 2.6234) + 7.9563 + 30.5692 + 1.6938*0.1
          = 5.1617 + 7.9563 + 30.5692 + 0.16938
          = 43.8566   ≈  43.8560  ✅（差 0.0006，是面板 4 位小数四舍五入造成的）
```

而这个公式后来在画面 01:52:45 的第 **1127、1130** 行被逐字证实（我把这张帧放大逐行核过行号）：
```python
bbox_loss_total = loss_bbox[..., :6].sum() + vel_dense_loss + rot_dense_loss
...
bbox_loss_total += rot_lidar_w_loss * 0.1
```
**我先用数值反推出公式、再在下一张帧里看到源码——两者一致。** 这说明表 E 的数值可以放心当作理解这套 Loss 的"标尺"。

由此还能读出两个非常重要的事实：
- **`loss_p_rot = 134.14` 这个巨大的数字根本没有进入总 loss**，它只是 TensorBoard 的观测量。真正训练朝向的是 `rot_dense = 30.57`。如果你不知道这一点，看到 TensorBoard 上 rot 一枝独秀会误判"朝向没学好"。
- **`loss_close_heatmap = 13126` 进了总 loss 并且完全主导它**（总 loss ≈ 0.0204+6.3973+1.5016+5.0235+43.856+13126.72 ≈ **13183.5**，close_heatmap 占 99.6%）。这个在 §11-6 会详细分析成因。

---

## Part 11-1　从检测头收尾到 `loss()` 入口（01:31:16 – 01:32:40）

**本段在讲什么**
上一章末尾检测头把 11 个分支塞进了 `ret_dict`，这一段是"交接班"：讲者把 BEV feature 也顺手存进 dict，然后跳进 `CenterHead.loss()`。
输入：`preds_dicts`（11 个 `[1,C,448,224]` 分支）＋ `obj_label / obj_state / obj_dir_cls_label` 三份 GT。
输出：调用 `self.get_targets(...)` 得到 9 个 GT 产物。
最关键的信息量在于讲者点破的一件事：**GT 早在 DataLoader 里就已经被"画"到 BEV 网格上了，网络前向时不再做这件事**——这是一个已落地的性能优化。

---

**[01:31:01] "然后在这里会把我们所用到的一个 BEV 的 feature，也保存到我们的这个 dict 里面去。"**

- 【直译】把送进检测头的那张 BEV 特征图本身也存一份到返回字典里。
- 【代码】
  ```python
  self.ret_dict['bev_feat'] = bev_feat          # [1, C_bev, 448, 224]
  self.ret_dict['pred_dict'] = preds_dicts      # 11 个分支
  ```
- 【为什么】三个用途：(1) 蒸馏/多任务分支（占用、车道线）要复用同一张 BEV；(2) 可视化调试（后面 01:53:10 帧里能看到 `vis_heatmap()` 就是拿 `ret_dict` 里的东西画图）；(3) 时序 MemoryManager 下一帧要拿它当历史帧。
- 【连接】和 BEVFusion 里 `self.pts_bbox_head(x)` 之后就把 `x` 丢掉不同，量产代码几乎总是把中间特征挂在字典上——因为一个 backbone 要喂 N 个下游任务。

**[01:31:08] "这个就是检测的。"**　**[01:31:16] "其实很简单。"**

- 【直译】这两句是讲者收束上一章的语气词，意思是"检测头部分就这些，不难"。
- 此处讲者做的是章节过渡，无新增技术内容。

**[01:31:17] "接下来的话就是去计算 Loss。"**　**[01:31:33] "算 Loss。"**

- 【直译】进入 `loss()` 函数。
- 【代码】画面 01:31:58 显示光标停在 `centerpoint_head.py` 第 802 行：
  ```python
  @force_fp32(apply_to=('preds_dicts'))
  def loss(self, obj_label, obj_state, obj_label_source, preds_dicts,
           sample_mask=None, **kwargs):
      """Loss function for CenterHead."""
  ```
- 【为什么】`@force_fp32` 这个装饰器非常关键：即使外面开了 AMP（fp16），**Loss 计算强制转回 fp32**。原因是 Focal Loss 里有 `log(p)`、`(1-p)^γ`，p 接近 0 时 fp16 会直接下溢成 0 → `log(0) = -inf` → NaN。你在 4060 上跑 BEVFusion 时如果遇到过 loss 变 NaN，十有八九就是这个。
- 【连接】mmdet3d 的 `CenterHead.loss` 上同样挂着 `@force_fp32(apply_to=('preds_dicts'))`，签名是 `loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs)`。DenseBEV 把 `gt_bboxes_3d/gt_labels_3d` 换成了自己的 `obj_label / obj_state / obj_label_source`——因为它的 GT 是**已经栅格化过的张量**，不是 `LiDARInstance3DBoxes` 对象列表。有意思的是 docstring 还留着 mmdet3d 的原文（画面 805–809 行仍写着 `gt_bboxes_3d (list[:obj:'LiDARInstance3DBoxes'])`），**是改代码没改注释的典型痕迹**。

**[01:31:53] "然后这个是我们在这里其实这个 obj_label…"**　**[01:31:57] "obj state。"**　**[01:31:58] "然后以及这个 directionclass label。"**　**[01:32:01] "这些就是我们的一个 GT。"**　**[01:32:04] "我们的一个 GT。"**

- 【直译】`obj_label`、`obj_state`、`obj_dir_cls_label` 这三个入参就是真值。
- 【代码】画面 01:31:58 第 814 行逐字：
  ```python
  obj_label_with_movement = torch.cat(
      (obj_label, obj_state, kwargs['obj_dir_cls_label']), -1)
  ```
- 【形状】`obj_label` 是目标级张量 `[B, 256, K]`（K 含 x,y,z,l,w,h,yaw,vx,vy,cls…），`obj_state` 是 `[B, 256, 1]`（动静状态），`obj_dir_cls_label` 是 `[B, 256, 1]`（朝向分类）。沿最后一维 concat → `[B, 256, K+2]`。
- 【为什么】为什么要 concat 而不是分别传？因为下面的 `get_targets` 内部会对目标做**排序 / 去重 / 高斯半径计算**，顺序会被打乱。如果三份 GT 分开传，就必须三份都跟着重排一次，很容易漏。concat 成一张大表后，"重排一次 = 全部属性跟着走"，这是工程上防错的常用手法。
- 【连接】画面 01:39:33（`load_object.py`）里能看到 GT 的生产端：`obj_state = torch.ones((self.cfg.max_obj_num, 1)).float() * -1`，即所有槽位预填 -1（无效），再逐个目标填真值。`max_obj_num` 就是那个 256。

**[01:32:09] "这里 get target。"**　**[01:32:12] "其实……就是把我们的这个 GT 把它放到我们对应的 BEV 的一个 feature 上。"**　**[01:32:22] "就对应的位置上去。"**

- 【直译】`get_targets` 的职责是把"目标列表"变成"BEV 网格上的监督图"。
- 【代码】画面 01:32:19 第 815 行（这一行长到出屏，我按可读部分＋第 519 行的 return 反推补全）：
  ```python
  heatmaps, anno_boxes, inds, record_valid_obj_indexes, masks, \
  attr_heats, attr_heat_masks, movement, movement_weight = \
      self.get_targets(obj_label_with_movement, **kwargs)
  ```
- 【形状】进：`[B, 256, K+2]` 一张目标属性表。出：见表 B——**5 个是 BEV 图（`[B,·,448,224]`），4 个是目标级（`[B,256,·]`）**。这个"一半图、一半表"的产物结构，正是本章三层监督的物理来源。
- 【为什么】为什么要栅格化？因为卷积头输出的是稠密图，`loss(pred_map, gt_map)` 要求两边同构。CenterPoint 的核心思想就是"把检测变成关键点热图回归"，那 GT 也必须变成热图。
- 【连接】mmdet3d 里这一步叫 `self.get_targets(gt_bboxes_3d, gt_labels_3d)`，内部用 `multi_apply(self.get_targets_single, ...)` 对 batch 里每个样本各算一份再堆起来。画面 01:34:38 第 507–517 行原原本本就是这个 pattern（下一段详解）。

**[01:32:24] "其实我们这个是当现在那个 data 在 DataLoader 的时候就已经给它处理好了的。"**　**[01:32:33] "所以说这就不会在网络计算的时候再去处理 GT。"**

- 【直译】⚠ **这两句是本段最有价值的信息**：真正把目标画成高斯热图的活儿，已经被挪到 DataLoader 的 worker 进程里做掉了；网络前向时 `get_targets` 拿到的基本是搬运和拼接。
- 【代码】对应关系大致是：
  ```python
  # ---- 老写法（mmdet3d 默认）：GPU 主进程里现算 ----
  for k in range(num_objs):
      radius = gaussian_radius((h, w), min_overlap=0.7)
      draw_gaussian(heatmap[cls_id], center_int, radius)   # ← 纯 Python 循环，慢
  # ---- DenseBEV：这段循环在 Dataset.__getitem__ 里，由 num_workers 并行做掉 ----
  ```
- 【为什么】不这么做会怎样？`draw_gaussian` 是**逐目标的 Python for 循环 + numpy 切片**，一帧几十个目标就是几十次 Python 调用。放在训练主循环里，它和 GPU 前向**串行**，GPU 要空等。挪到 DataLoader 后，它和 GPU 前向**流水线并行**（prefetch），等于白送。对 batch=1、iter 时间只有 0.4~0.5s 的场景（正是你在 4060 上跑 BEVFusion 实测的 0.481s/iter 量级），这几十毫秒是实打实的 5%~10%。
- 【为什么·补充】还有一个隐性好处：GT 栅格化搬进 Dataset 后，**数据增强（旋转/翻转/GT 抖动）和栅格化就在同一个地方**，不会出现"增强了框但热图没跟着转"的经典 bug。画面 01:39:33 第 273–277 行正好能看到这类增强：
  ```python
  if (not ctx.is_val_data) and self.cfg.gt_box_with_offset:   # 长短停gt框增加扰动
      heading = self.read_heading(obj)
      if random.random() < 0.2:
          offset_l = obj['size'][0] * random.random() * 0.1 * random.choice([-1, 1])
          offset_w = obj['size'][1] * random.random() * 0.1 * random.choice([-1, 1])
  ```
  即：20% 概率给长/宽各加 ±10% 以内的随机扰动，只在训练集上做。这是针对"长短停"（估计是长时间静止车辆）标注框尺寸不稳的一种正则。
- 【连接】你之前在华为 BEV 组见过的"GT cache / 预生成 target"就是这个思路的极端版本（连高斯图都提前落盘）。这里是折中版：在 worker 里算，不落盘。

**[01:32:41] "然后复得然后复理完之后我们的这个每一种属性它的 GT 的一个含义。"**

- ⚠ Whisper 转写噪声，"复得/复理完"应为"**处理完 / 返回完**"。
- 【直译】`get_targets` 返回之后，我们逐个看这 9 个产物分别是什么意思。
- 这句是过渡句，正文从下一段开始。

---

### 🔨 动手练习 ch11-1：复刻检测头输出 dict 并核对 25 个通道

```python
import torch
# 复刻画面 01:31:58 终端里那 11 行打印
B, H, W, NUM_CLS = 1, 448, 224, 5
net_dict = {
    'reg':              torch.randn(B, 2, H, W),   # dx, dy
    'height':           torch.randn(B, 1, H, W),   # z
    'dim':              torch.randn(B, 3, H, W),   # log l, log w, log h
    'lidar_rot_weight': torch.randn(B, 1, H, W),
    'rot':              torch.randn(B, 2, H, W),   # sin, cos
    'vel':              torch.randn(B, 2, H, W),   # vx, vy
    'dir_cls':          torch.randn(B, 1, H, W),
    'movement':         torch.randn(B, 1, H, W),
    'rot_lidar':        torch.randn(B, 2, H, W),
    'close_heatmap':    torch.randn(B, NUM_CLS, H, W),
    'heatmap':          torch.randn(B, NUM_CLS, H, W),
}
total_c = 0
for k, v in net_dict.items():
    print(f"{k:18s} {tuple(v.shape)}")
    total_c += v.shape[1]
print("total channels =", total_c)          # 期望 25
print("BEV cells H*W  =", H * W)            # 期望 100352
print("一帧监督的标量数 =", total_c * H * W) # 期望 2508800（250 万，注意这才是1帧1个batch）
```
预期输出末三行：
```
total channels = 25
BEV cells H*W  = 100352
一帧监督的标量数 = 2508800
```
> 记住 `448*224 = 100352` 这个数，它会在 §11-11 以 `pred.view(1, -1, 10)` 的形式原样出现。

### 【小结】
1. `loss()` 挂着 `@force_fp32`，Loss 全程 fp32，这是 AMP 下不 NaN 的护身符。
2. 三份 GT（`obj_label / obj_state / obj_dir_cls_label`）先 concat 成一张表再进 `get_targets`，保证后续重排时属性不会走散。
3. **高斯热图的绘制已被前移到 DataLoader**，网络里不再做 GT 处理——这是这套代码相对开源 CenterPoint 的第一个工程优化点。

---

## Part 11-2　`get_targets` 的五大产物：heatmap / anno_box / inds / record / mask（01:32:52 – 01:35:05）

**本段在讲什么**
讲者把 `get_targets` 的返回值一个一个在调试器里点开看形状。
输入：`obj_label_with_movement` `[1, 256, K+2]`。
输出：5 个 BEV 图 + 4 个目标级张量。
本段要建立的核心直觉是：**"BEV 图"和"目标级表"是两套坐标系，`inds` 是它们之间唯一的桥**。你只要牢牢抓住 `inds`，后面 §11-11 的 `gather_feat` 就毫无神秘感。

---

**[01:32:52] "含义可以说一下，这个 heatmap 呢就是它的 shape 是 1 乘 5、448 和 224。"**

- ⚠ 校正稿写作"1乘548和24"，是 Whisper 把 "1×5×448×224" 连读吞了音。画面 01:33:02 的调试器 tooltip 明确显示 `shape = torch.Size([1, 5, 448, 224])`，**以画面为准**。
- 【直译】分类热图，形状 `[batch=1, 类别=5, 前后=448, 左右=224]`。
- 【形状】`448 × 0.4m = 179.2m`（正好 = 前 95.4 + 后 83.8）；`224 × 0.4m = 89.6m`（正好 = 左右各 44.8）。**分辨率 0.4 m/格，网格数 448×224，与全局脉络里的车辆配置完全对上。**
- 【代码】
  ```python
  heatmaps[0].shape   # torch.Size([1, 5, 448, 224])
  ```
- 【为什么】为什么类别放在通道维而不是做一个 5 分类 softmax？因为 CenterPoint 用的是**每类一张独立的二值热图 + Focal Loss**，允许同一个格子上同时出现两类的峰（比如卡车和它拖的挂车中心重合）。softmax 会强制互斥。
- 【连接】和 CenterNet/CenterPoint 论文一致：`Y ∈ [0,1]^{W×H×C}`，`Y_{xyc}=1` 表示 (x,y) 是第 c 类目标的中心。

**[01:32:59] "然后对应的这个 448 和 224 就是我们 BEV 的一个 feature。"**

- 【直译】后两维就是 BEV 特征图的空间尺寸，和检测头输出的空间尺寸一模一样。
- 【为什么】必须一模一样，否则 `loss_cls(pred, gt)` 无法逐像素相减。这也解释了为什么全局脉络里说"下采样一倍在 224×112 上做投影"，但**头和 Loss 都在 448×224 上**——LSS 投影为了省显存在半分辨率上做，BEV backbone（UNet）再上采样回全分辨率，检测和 Loss 在全分辨率上进行。
- 【连接】mmdet3d 里这个叫 `out_size_factor`。画面 01:34:03 第 532 行看到 `self.train_cfg['out_size_factor']`，画面 01:52:52 第 1160 行看到 `bbox_xs * self.train_cfg['out_size_factor'] * self.train_cfg['voxel_size'][0]`。所以配置里 `voxel_size=[0.4, 0.4, ...]`、`out_size_factor=1`（⚠ 推断：因为 448×224×0.4 恰好覆盖全量程，若 out_size_factor=2 则量程会翻倍到 358m，与"前95.4/后83.8"不符）。

**[01:33:03] "5 呢就是……每一种类别。"**（讲者先说"每一种属性"后改口"每一种类别"）

- 【直译】通道 0~4 分别对应 5 个检测类别。
- 【代码】结合上一章 01:30:43–01:30:47 的类别列表和本章 01:43:43、01:43:55 的加权代码，可以把索引钉死：
  ```
  ch0 = car      （代码注释：把car的rotation权重乘3 → _task_heatmap[:, :1]）
  ch3 = VRU      （代码注释：把近处vru的rotation权重乘3 → _task_heatmap[:, 3:4]）
  ```
  ⚠ ch1/ch2/ch4 讲者在上一章念得含糊（"truck / cut / head / sinθ"），大概率是 truck / bus / VRU / cone-tank 一类。代码里还出现 `self.conetank_index` 和 `self.use_conetank_cls`（画面 01:31:58 第 827–830 行），说明有一类是**锥桶/水马**（conetank）。所以我推断 5 类 ≈ `[car, truck, bus, VRU, conetank]`，其中 VRU 在 index 3、car 在 index 0 是**代码确证**的，其余是推断。
- 【连接】nuScenes 是 10 类，这里合并成 5 类是典型的量产取舍：把 bicycle/motorcycle/pedestrian 统一成 VRU（Vulnerable Road User），因为下游规控对它们的处理策略一样。

**[01:33:07] "然后每一种类别，然后它在 448 和 224 上，有对应位置的那个值，它是那个 1。"**（01:33:11–01:33:24 为同一句的断续表述，此处合并）

- 【直译】哪个格子上有目标中心，那个格子的值就是 1；周围按高斯衰减；其余是 0。
- 【代码】
  ```python
  # DataLoader 里已经做完的事，等价于：
  radius = gaussian_radius((box_h_in_cell, box_w_in_cell), min_overlap=0.7)
  draw_umich_gaussian(heatmap[cls_id], (cx_int, cy_int), radius)
  # 峰顶恰好 = 1.0，这一点在 loss 里被用来数目标个数
  ```
- 【为什么】"峰顶恰好等于 1.0" 不是美学，是**功能性契约**：后面 `num_pos = _task_heatmap.eq(1).float().sum()` 就是靠 `== 1.0` 精确相等来数正样本个数的。如果高斯画成 0.999 或者做了归一化，这个统计立刻失效。这是 CenterNet 系列一个隐蔽但重要的约定。
- 【连接】这就是讲者在 01:36:27 要说的"统计有多少个数值为 1 的 pixel，其实对应的就是我有多少个目标"。

**[01:33:27] "anno_box……这个是 anno_box，就是目标的它的一个属性。"**　**[01:33:37] "这个并没有……并没有把它放到 BEV 的那个 feature 上去。"**　**[01:33:47] "这位 tensor 的一个形式还保存的，就是 256 个。"**

- ⚠ "这位 tensor" = "**这（种）tensor**"。
- 【直译】`anno_box` 是**目标级**的属性表，没有被画到 BEV 上，就是一个 256 行的表。
- 【形状】画面 01:33:35 调试器 tooltip 白纸黑字：`shape = torch.Size([1, 256, 10])`，`ndim = 3`。后来在 01:37:58 帧的终端里还能看到讲者自己敲的验证：
  ```
  >>> target_box.shape
  torch.Size([1, 256, 10])
  ```
- 【代码】
  ```python
  anno_boxes[0].shape   # [1, 256, 10] = [B, max_obj, (dx,dy,z,logl,logw,logh,sin,cos,vx,vy)]
  ```
- 【为什么】为什么 xyz/lwh 不像 rot/vel 那样也做成稠密图？两个原因：(1) **中心偏移 dx,dy 只在中心格子上有定义**，邻域格子的"偏移"没有物理意义，画成稠密图等于制造错误监督；(2) 尺寸 l,w,h 在目标内部虽然处处相同，但把它 dense 化会让大目标的权重天然比小目标大 100 倍（面积比），需要额外做面积归一化，得不偿失。而朝向/速度做 dense 是有增益的（见 §11-7），因为它们**在目标内部确实处处相同且不随位置变化**，dense 化等价于免费的数据增广。
- 【连接】这正是 DenseBEV 相对标准 CenterPoint 的第二个特色：**属性分两拨——几何量（xyz/lwh）只在中心点监督，运动/朝向量额外做全图监督**。

**[01:33:53] "然后这个 indices 呢，其实代表的是我的这个每一个目标把它放到 BEV feature 上的一个位置。"**　**[01:34:06] "这个 38416 呢，其实就是把我们的 h 和 w 给它展平之后所对应的一个位置。"**

- 【直译】`inds` 存的是"目标中心格子在展平成一维之后的下标"。
- 【形状/数值】画面 01:34:03 的 tooltip 显示：`tensor([[38416, 0, 0, 0, 0, 0, ...]])`，`len() = 1`（外层 list 只有 1 个 task）。**校正稿里那个存疑的"38416"，画面确认无误。**
- 【代码·把这个数拆开看】展平约定在画面 01:52:52 第 1154–1155 行被写死了：
  ```python
  xs = (ind.float() / width).int().float()     # 行号 = ind // 224
  ys = (ind % width).int().float()             # 列号 = ind %  224
  ```
  所以
  ```python
  ind = 38416
  row = 38416 // 224 = 171      # 171 * 224 = 38304
  col = 38416 %  224 = 112      # 38416 - 38304 = 112
  ```
  **col = 112 恰好是 224 的正中**，也就是横向偏移 0（正前方/正后方那条中轴线）；row = 171。按 0.4m 分辨率与"前 95.4 / 后 83.8"的量程，BEV 图的行索引 0 对应最前方或最后方（取决于 `point_cloud_range` 的符号约定）。⚠ 我不能确定行 0 是最前还是最后（画面第 1160 行写的是 `point_cloud_range[0] - bbox_xs * ...`，是**减号**，暗示行号增加对应 x 减小，即行 0 = 最前方 +95.4m）。若如此，171 行 → x ≈ 95.4 − 171×0.4 = **27.0 m**，即正前方 27 米、横向 0 米处有一个目标。这个数字非常合理（前车）。
- 【为什么】为什么不存 (row, col) 两个数而要展平？因为 `torch.gather` 只能沿**一个**维度取，展平成一维后一次 `gather` 就能把 256 个目标全抠出来，无需两次索引或 `advanced indexing`（后者在 ONNX 导出时经常出问题）。这是为部署友好做的设计。
- 【连接】mmdet3d `centerpoint_head.py` 里一模一样：`ind[k] = y * feature_map_size[0] + x`。你在 BEVFusion 代码里见过的 `_gather_feat(feat, ind)` 就是配套函数。

**[01:34:16] "然后这个呢是记录了我的这个……在给它特别的之后，因为它可能顺序会变，然后这里保存的就是我原始输入的 label 和处理完之后……他们之间的一个对应的一个关系。"**

- ⚠ "给它特别的" 应为 "**给它筛（过滤）掉之后**" 或 "**给它排序之后**"（Whisper 误听）。
- 【直译】`record_valid_obj_indexes` 记录"处理后第 i 个槽位，对应原始标注里的第几个目标"。
- 【形状】`[1, 256]`。
- 【代码】
  ```python
  # 概念伪码
  keep = (in_range) & (num_pts_in_box > 0) & (~ignore)
  kept_objs = objs[keep]                         # 顺序变了、数量变了
  record_valid_obj_indexes = torch.nonzero(keep).squeeze(-1)   # 反查表
  ```
- 【为什么】三个刚需：(1) **可视化/badcase 分析**——训练时发现第 7 个槽位 loss 爆炸，要能回溯到原始 json 里的哪个框；(2) **corner 分支**需要按原始索引去取 `obj_label_corner`（画面 01:52:09 第 1110 行注释掉的那行确实传了 `record_valid_obj_indexes[task_id]`）；(3) 多帧时序里同一个 track 要跨帧对齐。
- 【连接】开源 CenterPoint 没有这个返回值——因为学术评测不需要回溯到原始标注 ID，量产需要（要给标注方提 badcase）。

**[01:34:37] "然后这个 mask 呢是……因为我们在这个路的时候其实有每一帧它的机器数量是不一样的，但我们都把它保存成 100 个目标。"**

- ⚠ 两处 Whisper 噪声：(a)"在这个路的时候"= "**在这个 load（加载）的时候**"；(b)"机器数量"= "**目标数量**"（"机器"是"gt"/"目标"的误听，本章多处出现，后文统一按"目标/GT"理解）。
- ⚠⚠ **"100 个目标"与画面矛盾**：`anno_boxes` 实测 `[1, 256, 10]`，`load_object.py` 里用的是 `self.cfg.max_obj_num`，而讲者自己在 01:51:09 又说"我们当前是预测 256 个目标"。**推断：此处口误，应为 256。** 依据：(1) 同一段视频里两个数字冲突，以有画面证据的 256 为准；(2) 256 = 2^8，是这类代码里 `max_objs` 的常见取值（mmdet3d 默认 500，CenterNet 默认 128）。
- 【直译】每帧真实目标数不同（可能 3 个也可能 80 个），但张量形状必须固定，所以统一开 256 个槽位，`masks` 标记哪些槽位是真的。
- 【形状】`masks[0].shape = [1, 256]`，dtype 是 uint8/float，值 0 或 1。
- 【代码】
  ```python
  masks = torch.zeros(B, 256, dtype=torch.uint8)
  masks[b, :num_real_obj] = 1
  # 用的时候：
  mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()   # [1,256] → [1,256,10]
  ```
  这行 `unsqueeze(2).expand_as(target_box)` 在画面 01:48:46 第 1009 行原样出现。
- 【为什么】不这么做会怎样？变长张量无法 batch，`torch.stack` 直接报错。padding + mask 是所有"目标级"任务（检测、NLP 的序列）的标准解法。
- 【连接】和 DETR 系列的 `num_queries=100/300` 是同一个套路，只不过 DETR 是可学习 query，这里是纯 padding。

**[01:34:57] "所以说对于有些目标是无效的，所以说在这里只是记录了一个哪些目标是有效的一个 Mask。"**

- 【直译】重复上句结论。此处讲者重复上述内容。
- 【补充·代码里还有一层"有效性"】画面 01:48:46 第 1010–1011 行：
  ```python
  isnotnan = (~torch.isnan(target_box)).float()
  mask *= isnotnan
  ```
  即除了"槽位是否被占用"，还要逐**元素**检查 NaN。为什么会有 NaN？因为某些目标可能只标了框没标速度，GT 生产端填了 `float('nan')`。这一步是防止一个坏样本把整批梯度污染成 NaN 的最后一道闸。这是量产代码里非常常见但论文里从不写的细节。

---

### 🔨 动手练习 ch11-2：手搓一个 mini `get_targets`，把 38416 算出来

```python
import torch, math

H, W, NUM_CLS, MAX_OBJ = 448, 224, 5, 256
VOXEL = 0.4
PC_RANGE_X_FRONT = 95.4   # 车前 95.4m 对应 row 0（见正文 ⚠ 推断）

def draw_gaussian(hm, cx, cy, radius):
    """CenterNet 的 draw_umich_gaussian 简化版，峰顶精确 = 1.0"""
    d = 2 * radius + 1
    sigma = d / 6.0
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1),
                          torch.arange(-radius, radius + 1), indexing='ij')
    g = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[radius, radius] = 1.0                       # ← 关键契约：峰顶严格 1.0
    y0, y1 = max(0, cy - radius), min(hm.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(hm.shape[1], cx + radius + 1)
    gy0, gy1 = radius - (cy - y0), radius + (y1 - cy)
    gx0, gx1 = radius - (cx - x0), radius + (x1 - cx)
    hm[y0:y1, x0:x1] = torch.maximum(hm[y0:y1, x0:x1], g[gy0:gy1, gx0:gx1])

# ---- 造一个"正前方 27m、横向 0m 的 car" ----
x_m, y_m, cls_id = 27.0, 0.0, 0          # cls 0 = car
row = int((PC_RANGE_X_FRONT - x_m) / VOXEL)   # 171
col = int(W / 2 + y_m / VOXEL)                # 112
print("row, col =", row, col)

heatmap = torch.zeros(NUM_CLS, H, W)
draw_gaussian(heatmap[cls_id], col, row, radius=4)

inds  = torch.zeros(MAX_OBJ, dtype=torch.long)
masks = torch.zeros(MAX_OBJ, dtype=torch.uint8)
inds[0]  = row * W + col           # ← 展平索引
masks[0] = 1

print("ind      =", inds[0].item())            # 期望 38416
print("反解 row  =", (inds[0] // W).item())     # 期望 171
print("反解 col  =", (inds[0] %  W).item())     # 期望 112
print("num_pos  =", heatmap.eq(1).float().sum().item())   # 期望 1.0
print("heatmap.shape =", tuple(heatmap.unsqueeze(0).shape))  # 期望 (1,5,448,224)
```
预期输出：
```
row, col = 171 112
ind      = 38416
反解 row  = 171
反解 col  = 112
num_pos  = 1.0
heatmap.shape = (1, 5, 448, 224)
```
> 这个 38416 就是调试器里那个数。你亲手算出来一次，`inds` 这个概念以后就再也不会糊了。

### 【小结】
1. `heatmap [1,5,448,224]` 是"图"，`anno_box [1,256,10]` 是"表"，`inds [1,256]` 是连接两者唯一的桥；这三件套构成了 CenterPoint 系检测的全部 GT 骨架。
2. 高斯峰顶必须**精确等于 1.0**，因为后面 `eq(1).sum()` 靠它来数正样本个数并当作 Focal Loss 的 `avg_factor`。
3. 每帧目标数不定统一 padding 到 **256**（讲者口误说成 100），`masks` 标有效槽位，`isnotnan` 再兜一层 NaN 防护。

---

## Part 11-3　dense 属性 GT：`attr_heats` / `attr_heat_masks` / `movement` / `movement_weight`（01:35:05 – 01:36:00）

**本段在讲什么**
讲者继续点开 `get_targets` 的后 4 个返回值。这 4 个是 DenseBEV 特有的（开源 CenterPoint 没有）：**把朝向、速度、动静这些属性也画到 BEV 图上**，于是产生了 `attr_heats [1,9,448,224]` 和它的 mask。
输入：同上。输出：稠密属性 GT + 目标级动静 GT。
本段是理解 §11-7 ~ §11-10 全部 dense loss 的前置知识——**表 C 那 9 个通道的 layout 就是在这里定下来的**。

---

**[01:35:05] "然后这里的属性各 head。"**　**[01:35:09] "其他的一些属性就是对于速度、yaw。"**

- ⚠ "属性各 head" 应为 "**属性的 head**"。
- 【直译】接下来是几个"属性头"对应的 GT：速度、朝向角（yaw）等。
- 【代码】对应的返回值就是 `attr_heats`，被 loss 里这样切片使用（画面 01:42:14 第 871–875 行）：
  ```python
  rot_pred = preds_dict[0]['rot']                 # [1,2,448,224]
  rot_gt   = attr_heats[task_id][:, :2]           # [1,2,448,224]  ← 通道 0,1
  v_pred   = preds_dict[0]['vel']                 # [1,2,448,224]
  v_gt     = attr_heats[task_id][:, 2:4]          # [1,2,448,224]  ← 通道 2,3
  ```

**[01:35:15] "它也是把它的属性也……把它放到 heatmap 上去。"**　**[01:35:21] "就是有值就是有目标位置的那个地方，然后对应的它的一些属性的一个值。"**

- 【直译】和 heatmap 一样按空间铺开：**目标覆盖的那些格子上填该目标的属性值，其余格子留空**。
- 【形状】画面 01:35:19 tooltip 明确：`shape = torch.Size([1, 9, 448, 224])`。
- 【代码·概念伪码】
  ```python
  attr_heats      = torch.zeros(B, 9, 448, 224)
  attr_heat_masks = torch.zeros(B, 9, 448, 224)   # 或 [B,448,224]，见下文 ⚠
  for obj in objs:
      cells = rasterize_box_footprint(obj)        # 该框在 BEV 上覆盖的格子集合
      attr_heats[b, 0, cells] = math.sin(obj.yaw)
      attr_heats[b, 1, cells] = math.cos(obj.yaw)
      attr_heats[b, 2, cells] = obj.vx
      attr_heats[b, 3, cells] = obj.vy
      attr_heats[b, 4, cells] = obj.move_cls_2     # 0=无效,1=静,2=动
      attr_heats[b, 5, cells] = obj.move_w_2
      attr_heats[b, 6, cells] = obj.move_cls_4
      attr_heats[b, 7, cells] = obj.move_w_4
      attr_heats[b, 8, cells] = obj.dir_cls
      attr_heat_masks[b, :, cells] = 1
  ```
  ⚠ 这里"哪些格子被填"我没有直接画面证据（`get_targets_single` 讲者跳过没看）。两种可能：(a) 只填**目标框内部**的格子（真 dense）；(b) 只填**中心点邻域高斯半径内**的格子。我倾向 (a)，依据有二：其一，讲者反复强调"它不仅仅只是对应的哪一个 pixel 上去回归，它其实还会在 BEV 上去回归"（01:41:36），"不仅仅一个 pixel"暗示的是一片区域；其二，如果只是高斯邻域，`attr_heat_masks` 就没必要单独存一份（直接用 `heatmap>0` 即可）。
- 【为什么·这是本章第一个大设计动机】把朝向/速度做成稠密监督有三重收益：
  1. **正样本从 1 个像素变成几十上百个像素**。一辆 4.5m×1.9m 的车在 0.4m 格子上占 `11×5 ≈ 55` 格。原来一个目标只给 1 个像素的 rot 梯度，现在给 55 个，**梯度信号放大 55 倍**，属性收敛显著更快更稳。
  2. **推理时更鲁棒**。真实推理时 heatmap 峰值位置常常偏 1~2 格；如果只在中心点训了属性，偏 1 格取到的属性就是没训过的垃圾值。dense 训练后，中心附近整片区域的属性都是对的，**峰值抖动不再引起属性突变**。这对时序稳定性（不闪、不跳）至关重要，而闪跳正是量产感知最被诟病的问题。
  3. **可以做区域加权**。因为是图，所以能和"近距离 mask"、"lidar 前向 mask"这类空间 mask 直接相乘（§11-6、§11-9 就是这么干的）。目标级的表做不到这一点。
- 【连接】这个思路和 FCOS / CenterNet2 的"center sampling"、以及 BEVDet 的"dense depth supervision"是同源的：**在能 dense 的地方就 dense，正样本越多越好**。

**[01:35:29] "就是有朝向两个值，sinθ、cosθ；然后以及动静速度 VxVy；然后以及动静的四个值；然后还有的话是一个朝向；就是这九种属性。"**

- ⚠ "动静速度 VxVy" 是口误粘连，应为 "**速度 VxVy**"。
- 【直译】9 个通道 = 朝向 2 + 速度 2 + 动静 4 + 朝向分类 1。
- 【为什么这句是本章最有价值的一句】它把 `attr_heats` 的 9 通道 layout 说全了，而**我用三处代码把它逐个钉死**（见表 C）：
  - 通道 0,1 ← `rot_gt = attr_heats[:, :2]`（画面 01:42:14 第 872 行）
  - 通道 2,3 ← `v_gt = attr_heats[:, 2:4]`（画面 01:42:14 第 875 行）
  - 通道 4,5 ← 代码注释原文 "第4维是二分类动静类别，第5维是二分类loss权重"（画面 01:47:30 第 968 行）
  - 通道 6,7 ← 代码注释原文 "第6维是四分类动静类别，第7维是四分类loss权重"（画面 01:47:30 第 981 行）
  - 通道 8 ← `dir_cls_task_idx = (6 if enable_corner_det else 4) + (4 if activate_move else 0) = 4+4 = 8`（画面 01:48:20 第 995 行）
  **讲者说的"动静四个值"= 通道 4,5,6,7，正是"二分类类别+二分类权重+四分类类别+四分类权重"。他说得完全正确，只是没展开。**
- 【代码·把这个 layout 写成常量表，你以后读这段代码就不用数手指】
  ```python
  ATTR = dict(ROT_SIN=0, ROT_COS=1, VX=2, VY=3,
              MOV2_CLS=4, MOV2_W=5, MOV4_CLS=6, MOV4_W=7, DIR_CLS=8)
  ```

**[01:35:46] "然后这个是对应这几个属性的一个 Mask。"**

- 【直译】`attr_heat_masks`，标记 `attr_heats` 里哪些位置是有效监督。
- 【形状】⚠ 画面里讲者没点开它的 shape。但代码用法给出了强约束（画面 01:42:14 第 877–878 行）：
  ```python
  attr_mask = torch.cat([attr_heat_masks[0].unsqueeze(1),
                         attr_heat_masks[0].unsqueeze(1)], dim=1)
  ```
  `unsqueeze(1)` 说明 `attr_heat_masks[0]` 是 **3 维** `[B, 448, 224]`（unsqueeze 后变 `[B,1,448,224]`，cat 两份得 `[B,2,448,224]`，正好匹配 rot/vel 的 2 通道）。**所以 `attr_heat_masks` 是一个 list，元素为 `[B,448,224]` 的单通道 mask，不是 9 通道。** ⚠ 注意这里的索引是硬编码的 `attr_heat_masks[0]` 而不是 `attr_heat_masks[task_id]`——单 task 时没问题，多 task 时是潜在 bug。
- 【为什么】为什么 mask 只需要 1 个通道而属性有 9 个？因为"这个格子上有没有目标"是空间属性，与通道无关。各属性各自的有效性（比如速度无效）另外用 `v_gt < INVALID_VELOCITY` 这种值域判断来处理（见 §11-8）。

**[01:35:50] "然后这个是动静和 wait，这个还是目标级别的，这个是没有把它放到对应的那个 BEV feature 上去。这个是动静的。"**

- ⚠ "wait" = "**weight**"（权重）。
- 【直译】`get_targets` 最后两个返回值 `movement` 和 `movement_weight` 是**目标级**的（`[B,256,·]`），不是 BEV 图。
- 【代码】画面 01:33:02 第 519 行的 return 里它们被 `make_concats(...)` 包了一层：
  ```python
  return heatmaps, anno_boxes, inds, record_valid_obj_indexes, masks, \
         attr_heats, attr_heat_masks, make_concats(movement), make_concats(movement_weight)
  ```
  用法在画面 01:52:31 第 1099、1104 行：
  ```python
  mov_weight = mask[..., -1, None] * movement_weight[task_id] * not_cls_only[..., None]
  loss_movement = loss_movement_fnc(pred_mov, movement[task_id]) / valid_num
  ```
  `mask[..., -1, None]` 是 `[1,256,1]`，所以 `movement[task_id]` 必然也是 `[1,256,1]`（否则广播不上）。✅
- 【为什么】为什么动静**同时**有 dense 版和目标级版？因为它俩解决不同问题：dense 版给足梯度让特征学会"动/静"这个概念；目标级版保证**最终输出的那个像素**上的动静判断准确（推理时只读峰值像素）。这种"dense 训练 + 点位精调"的双保险，在本章会反复出现（rot、vel、movement 全都是两套）。

---

### 🔨 动手练习 ch11-3：验证 `attr_heats` 9 通道切片索引

```python
import torch

B, H, W = 1, 448, 224
ATTR = dict(ROT_SIN=0, ROT_COS=1, VX=2, VY=3,
            MOV2_CLS=4, MOV2_W=5, MOV4_CLS=6, MOV4_W=7, DIR_CLS=8)

# 造一份 attr_heats：每个通道填自己的编号，方便验证切片
attr_heats = torch.zeros(B, 9, H, W)
for name, idx in ATTR.items():
    attr_heats[:, idx] = idx

# --- 复刻源码里的四处切片（行号来自画面） ---
rot_gt = attr_heats[:, :2]                    # 第 872 行
v_gt   = attr_heats[:, 2:4]                   # 第 875 行
print("rot_gt 通道值:", rot_gt[0, :, 0, 0].tolist())   # 期望 [0.0, 1.0]
print("v_gt   通道值:", v_gt[0, :, 0, 0].tolist())     # 期望 [2.0, 3.0]

# 第 969-970 行：动静二分类 mask
mov2_valid = (attr_heats[:, ATTR['MOV2_CLS']] < 3.0) & (attr_heats[:, ATTR['MOV2_CLS']] >= 1.0)
print("mov2 通道号 =", ATTR['MOV2_CLS'], " 权重通道号 =", ATTR['MOV2_W'])

# 第 995 行：dir_cls 索引公式
enable_corner_det, activate_move = False, True
dir_cls_task_idx = (6 if enable_corner_det else 4) + (4 if activate_move else 0)
print("dir_cls_task_idx =", dir_cls_task_idx)          # 期望 8
dir_cls_gt = attr_heats[:, dir_cls_task_idx:dir_cls_task_idx + 1]
print("dir_cls 通道值 =", dir_cls_gt[0, 0, 0, 0].item())  # 期望 8.0

# attr_heat_masks 是 [B,H,W]，靠 unsqueeze(1)+cat 变成 [B,2,H,W]（第 877-878 行）
attr_heat_masks_0 = torch.ones(B, H, W)
attr_mask = torch.cat([attr_heat_masks_0.unsqueeze(1),
                       attr_heat_masks_0.unsqueeze(1)], dim=1)
print("attr_mask.shape =", tuple(attr_mask.shape))     # 期望 (1, 2, 448, 224)
```
预期输出：
```
rot_gt 通道值: [0.0, 1.0]
v_gt   通道值: [2.0, 3.0]
mov2 通道号 = 4  权重通道号 = 5
dir_cls_task_idx = 8
dir_cls 通道值 = 8.0
attr_mask.shape = (1, 2, 448, 224)
```

### 【小结】
1. `attr_heats [1,9,448,224]` = rot(0,1) + vel(2,3) + 动静二分类(4,5) + 动静四分类(6,7) + dir_cls(8)；这张表后面每一个 dense loss 都要查。
2. dense 化属性的三重收益：正样本 ×50 倍、峰值抖动鲁棒、可与空间 mask 相乘。
3. `attr_heat_masks[0]` 是单通道 `[B,448,224]`（由 `unsqueeze(1)` 反推），且代码里索引硬编码为 `[0]` 而非 `[task_id]`，多 task 时需留意。

---

## Part 11-4　`sample_mask` / `valid_lidar` / `num_pos` 与分类 Focal Loss（01:36:01 – 01:37:30）

**本段在讲什么**
终于开始真正算第一个 Loss：分类热图的 Focal Loss。
输入：`preds_dict[0]['heatmap'] [1,5,448,224]`（先过 `clip_sigmoid`）、`heatmaps[task_id] [1,5,448,224]`。
输出：`loss_heatmap`（本次调试实测 5.0235）。
本段三个关键词：**`clip_sigmoid`（数值安全）、`sample_mask`（时序帧有效性）、`num_pos`（归一化因子）**。

---

**[01:36:01] "在这个呢是我们是把我们的预测的分类的一个结果把它做完做一个 Sigmoid。"**　**[01:36:05] "做 Sigmoid。"**

- 【直译】把分类分支的 logit 过 sigmoid 变成 0~1 的概率。
- 【代码】画面 01:36:13 第 819 行逐字：
  ```python
  preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
  ```
- 【为什么用 `clip_sigmoid` 而不是 `torch.sigmoid`】mmdet3d 的实现是：
  ```python
  def clip_sigmoid(x, eps=1e-4):
      return torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
  ```
  把概率钳在 `[1e-4, 1-1e-4]`。目的是让 Focal Loss 里的 `torch.log(p)` 和 `torch.log(1-p)` 永远不会遇到 0。**不 clip 的后果**：训练后期网络很自信，某个背景像素 p→0，`log(1-p)` 没事，但正样本像素若 p→0 则 `log(p) = -inf`，一次 NaN 就毁掉整个 checkpoint。这是 CenterPoint 训练最常见的崩溃点。
- 【形状】不变，`[1,5,448,224]`。
- ⚠ 注意这里是**原地改写** `preds_dict[0]['heatmap']`，之后这个 key 存的就是概率不是 logit 了。Ch12 的 box 解码里讲者说"把预测的 heatmap 取一个 Sigmoid"，那是推理路径（不走 loss），两者互不冲突；但如果有人在 train 时先调 loss 再调 decode，就会**sigmoid 两次**。这是这类原地写法的经典坑。

**[01:36:06] "然后因为我们比如说有多帧，其实有些帧可能目标是无效的，所以说在这里有这样一个 sample mask。"**　**[01:36:18] "比如说帧 1 可能是有效的，帧 2 是无效的，所以说会对应的把这个 GT 也给它做一个 Mask。"**

- ⚠ 校正稿注明"帧1/帧2"原文是 "Bike 1/Bike 2"，是 Whisper 对 "帧" 的误听，已按上下文校正。
- 【直译】时序/多样本 batch 里，有些样本这一帧的 GT 不可用（比如 RL 融合那一路数据缺失），用 `sample_mask` 把它们整体屏蔽。
- 【代码】画面 01:36:13 第 **820–824** 行逐字（放大核对过行号）：
  ```python
  _task_heatmap = heatmaps[task_id]                       # 820
  if sample_mask is not None:
      _task_heatmap = _task_heatmap * sample_mask[..., None, None]
  if kwargs['valid_lidar'] is not None:
      _task_heatmap = _task_heatmap * kwargs['valid_lidar'][..., None, None]
  ```
- 【形状】⚠ **这一条我上一版推错了，现更正，并给出可复现的推导**。我原先写的是"`sample_mask` 是 `[B]`，`[..., None, None]` 补两维变 `[B,1,1]` 再广播"。**这个形状根本乘不起来**：
  ```python
  torch.ones(2,5,448,224) * torch.ones(2)[..., None, None]
  # RuntimeError: The size of tensor a (5) must match the size of tensor b (2)
  ```
  `[B]` 补两维得 `[B,1,1]`，广播时右对齐 ⇒ `B` 会去对 **通道维**（5），而不是 batch 维。
  **正确答案是 `sample_mask` 必须是 `[B, 1]`**，`[..., None, None]` → `[B,1,1,1]` → 与 `[B,5,448,224]` 右对齐广播 ✅。
- 【这个结论是我用"全体用法必须同时成立"反推出来的，四处交叉验证】同一个 `sample_mask` / `not_cls_only` 在本章被用在四种不同秩的张量上，只有 `[B,1]` 能让四处**同时**合法（我逐条在 PyTorch 里跑过）：
  | 用法（行号） | 表达式 | `[B]` | `[B,1]` |
  |---|---|---|---|
  | 822 | `_task_heatmap * sample_mask[..., None, None]`（乘 `[B,5,H,W]`） | ❌ 报错 | ✅ |
  | 1005 | `masks[task_id] * not_cls_only`（乘 `[B,256]`） | ❌ 报错 | ✅ |
  | 1099 | `mask[..., -1, None] * ... * not_cls_only[..., None]`（乘 `[B,256,1]`） | ❌ 报错 | ✅ |
  | 879 | `attr_mask * not_cls_only[..., None, None]`（乘 `[B,2,H,W]`） | ⚠ **只在 B=2 时"成功"，且结果错的** | ✅ |
  最后一行特别值得看：`[B]` 在 B=2 时会**静默地**沿通道维广播成功（因为 rot 恰好是 2 通道），算出来是错的却不报错。**这正是形状 bug 最阴险的形态**，我在写练习 ch11-5 时就实打实踩到了（打印出"样本0 均值 = 0.5"而不是 1.0），才反过来发现自己的形状判断错了。
- 【所以 `[..., None, None]` 到底在补什么】不是"从 `[B]` 补到 4 维"，而是"从 `[B,1]`（batch + 一个占位列）补到 `[B,1,1,1]`（batch + C + H + W）"。GT 生产端把这些帧级标量存成 `[B,1]` 而不是 `[B]`，正是为了让这套 `[..., None]` 写法在任意秩下都对齐 batch 维。**这是一个约定，不是巧合。**
- 【效果】乘完之后，无效样本的整张 GT 热图被清零 ⇒ 它的所有格子都变成"背景"，Focal Loss 里既无正样本也不会被当成漏检罚。
- 【为什么·这个设计比"跳过样本"高明】你可能会想：无效样本干脆从 batch 里剔掉不就行了？不行——batch 里样本数必须固定（DDP 各卡要同步），而且**这个样本的图像分支可能是有效的，只是 lidar 那一路无效**。乘 mask 的做法允许"部分有效"：图像 loss 照算，lidar 相关的 loss 归零。这正是 `valid_lidar` 单独存在的原因。
- 【连接】画面 01:53:10 第 1536–1538 行能看到这些 mask 的来源：
  ```python
  sample_mask     = self.ret_dict.get('rl_sample_mask'),
  lidar_front_mask= self.ret_dict.get('lidar_front_mask'),
  valid_lidar     = self.ret_dict.get('valid_lidar'),
  ```
  `rl_sample_mask` 里的 RL = Radar-Lidar（前面章节的 RL 融合 UNet）。**所以 `sample_mask` 的语义是"这个样本的 RL 分支是否有效"**，而不是泛指的样本有效性。这修正了单看本段容易产生的误解。
- 【连接·数值】表 E 第 15 项 `task0.loss_p_rl_percentage = 1.0`，它的定义在画面 01:52:45 第 1148 行：
  ```python
  loss_dict[f'task{task_id}.loss_p_rl_percentage'] = sample_mask.sum().item() / sample_mask.shape[0]
  ```
  即"这个 batch 里 RL 有效样本的比例"，本次调试是 1.0（全有效）。这是一个纯监控量，用来在 TensorBoard 上看数据质量。

**[01:36:25] – [01:36:55] "然后在这里会在我们的这个 BEV 的这个 feature 上去计算数值为一，就是统计有多少个有数值为一的一个 pixel，其实对应的就是我有多少个目标。"**

（这几句在校正稿里被 Whisper 重复输出了三遍，此处**合并为一句**处理；讲者原意只说了一次。）

- 【直译】数一数 GT 热图上有多少个值恰好等于 1 的像素，这个数就等于目标个数。
- 【代码】画面 01:36:13 第 825 行逐字：
  ```python
  num_pos = _task_heatmap.eq(1).float().sum().item()
  ```
- 【形状】`_task_heatmap [1,5,448,224]` → `eq(1)` 得 bool 同形 → `.sum()` 得标量 → `.item()` 得 python float。
- 【为什么"值为 1 的像素数 = 目标数"】因为高斯是逐目标 `torch.maximum` 叠加的，每个目标只有**峰顶那一个格子**是 1.0，衰减邻域都 < 1。所以 `eq(1)` 精确地数出了目标中心的个数。⚠ **两个例外**：(a) 两个同类目标中心落在同一格 → 只数到 1 个（0.4m 格子下概率极低）；(b) 上面刚乘完 `sample_mask`，被屏蔽的样本贡献 0。后者正是这行代码放在乘 mask **之后**的原因——`num_pos` 必须反映"实际参与训练的目标数"。
- 【为什么要这个数】Focal Loss 的归一化：
  ```python
  loss = focal_sum / max(num_pos, 1)
  ```
  分母用正样本数而非像素总数。若用像素总数（100352×5 = 50 万），一帧只有 10 个目标时正样本 loss 会被稀释 5 万倍，网络直接学成"全预测背景"。用 `num_pos` 归一化，等价于**"平均每个目标承担多少 loss"**，与目标数无关，梯度尺度稳定。
  `max(num_pos, 1)` 的 `1` 是防止空帧（无目标）时除零。

**[01:37:02] "然后这里去算那个……那个分类的 Loss。"**　**[01:37:06] "这个就是我们预测的一个分类结果，就是 1×5×448×224。"**　**[01:37:13] "然后这个是我们的那个 GT。"**

- ⚠ 校正稿写"Bike 3x5x48x24"，是 "1×5×448×224" 的严重误听；画面终端里 `heatmap torch.Size([1, 5, 448, 224])` 是铁证。
- 【直译】把预测热图和 GT 热图送进 Focal Loss。
- 【代码】画面 01:36:13 第 835–841 行**完整逐字**：
  ```python
  loss_heatmap = self.loss_cls(
      preds_dict[0]['heatmap'],          # 预测（已 clip_sigmoid）[1,5,448,224]
      heatmaps[task_id],                 # GT 高斯图        [1,5,448,224]
      nearby_mask=self.nearby_mask,
      weight=self.cls_loss_weight,       # 每类一个权重（5 维向量）
      avg_factor=max(num_pos, 1),
      sample_mask=sample_mask_full_heatmap)
  ```
- 【形状】进两张 `[1,5,448,224]`，出一个标量。实测 `loss_heatmap = 5.0235`。
- 【为什么用 Focal Loss 而不是 BCE】BEV 热图上 `100352×5 = 501760` 个位置，正样本（值=1）只有个位数到几十个，**正负比 1:10000 以上**。BCE 会被海量易分负样本淹没。CenterNet 的 Gaussian Focal Loss 形式：
  ```
  正样本(y=1):   -(1-p)^α · log(p)                 α=2
  负样本(y<1):   -(1-y)^β · p^α · log(1-p)         β=4
  ```
  `(1-y)^β` 这一项是关键：**离目标中心越近，y 越接近 1，`(1-y)^4` 越小，这个"负样本"的惩罚就越轻**。也就是说中心旁边一格预测出高分不会被重罚——这正是我们想要的"软标签"。
- 【连接】这就是 CenterNet 论文里的 `L_k`。你在 BEVFusion 里见到的 `loss_cls=dict(type='GaussianFocalLoss', reduction='mean')` 是同一个东西。DenseBEV 的差别在于**多传了 3 个参数**：`nearby_mask`、`weight`（类别权重）、`sample_mask`。

- 【补充·这三个额外参数分别在改什么，讲者一个都没提，但它们决定了这条 loss 的实际行为】
  | 参数 | 传进来的东西 | 它在 Focal Loss 内部改的是什么 |
  |---|---|---|
  | `weight=self.cls_loss_weight` | 一个长度 5 的**类别权重向量** | 逐通道缩放。5 类的样本量差几十倍（car 满地都是、conetank 几百帧才出一个），靠它把稀有类的梯度顶上去。相当于 mmdet3d 里没有、要自己加的 class-balanced 项。 |
  | `sample_mask=sample_mask_full_heatmap` | `[B,1]` 或 `[B,5,448,224]` | **逐位置屏蔽**。走 else 分支时是 `[B,1]`（整样本开关，见下文形状推导）；开了 conetank 时升成逐类逐像素图（只在近处监督锥桶）。同一个形参吃两种秩，靠内部广播兜住。 |
  | `nearby_mask=self.nearby_mask` | 一张常量图（`self.` 开头 ⇒ 建头时就固定） | ⚠ 画面里只看到传参、没看到定义。从命名+同类物（`close_cls_mask` / `close_rot_heatmap` / `close_conetank_cls_heatmap` 都是 `[448,224]` 的常量图）推断：**它标出"本车周围一圈自遮挡/无观测的区域"**（车身正下方、传感器盲区、拖车挂接区），在这些格子上既不该出正样本也不该罚假正样本，所以要单独屏蔽。 |
  【为什么我敢这么推 `nearby_mask`】三条：(1) 它以 `self.` 存在而不是每帧算，说明是**几何常量**而非数据相关量；(2) 它只传给 `loss_cls`（分类），不传给 `loss_bbox`（回归）——只有"该不该在这里报目标"这个问题需要盲区图，回归不需要；(3) 命名 `nearby` 与 `close` 刻意区分开（`close_*` 一族是"近距离**加练**区"，`nearby` 是"**紧邻本车**"），两者语义不同才会起两个名字。⚠ 仍属推断，列入存疑清单。
- 【⚠ 一处值得注意的不对称】`loss_close_heatmap`（§11-6 第 860–862 行）调用同一个 `self.loss_cls`，但**没有传 `nearby_mask`**。也就是说主 heatmap 会屏蔽本车盲区、close_heatmap 不屏蔽。如果 `nearby_mask` 真是盲区图，那 close 分支就会在盲区里被当成"应该全是背景"来罚——这可能是 §11-6 里 `loss_close_heatmap = 13126` 异常大的**第三个成因**（前两个是 `avg_factor` 退化 + 训练早期瞬态）。这一条是我对照两帧代码时才发现的，讲者没提。

**[01:37:15] – [01:37:29] "真的 GT 的一个……真的读大家看这些就是……得很真的……也要看这些就是……flag 了……那个 Loss。"**

- ⚠ 这几行是 Whisper 在讲者停顿翻页时产生的**幻听垃圾**（校正稿文件头已注明会有这类残留），时间戳还出现了 `[01:37:48]` 排在 `[01:37:29]` 前面的错序。**无有效技术内容，跳过。**
- 【推断依据】(1) 内容语义不成句；(2) 时间戳倒序说明是分段解码拼接错误；(3) 前后两句（01:37:13 讲 GT、01:37:31 讲取属性 mask）语义直接衔接，中间不缺内容。

**[01:37:31] "然后在这里会说，会取出我们目标的其他的属性的一个 GT。"**

- 【直译】分类 loss 算完，接下来处理回归属性的 GT。
- 【代码】画面 01:36:13 第 842 行：
  ```python
  target_box = anno_boxes[task_id]      # [1, 256, 10]
  ```
- 【连接】这一行是从"图"世界切到"表"世界的分界线。前面所有东西都是 `[·,·,448,224]`，从这里开始出现 `[1,256,10]`。

---

**补充：本段没讲但画面里有的 `use_conetank_cls` 分支（画面 01:36:13 第 827–833 行）**

讲者跳过了这段，但它决定了 `sample_mask_full_heatmap` 长什么样，值得补上：
```python
if self.use_conetank_cls and self.close_conetank_cls_heatmap is not None:
    sample_mask_full_heatmap = _task_heatmap.new_ones(_task_heatmap.shape)   # [1,5,448,224] 全 1
    sample_mask_full_heatmap[:, self.conetank_index] *= torch.Tensor(
        self.close_conetank_cls_heatmap[None, ...]).to(_task_heatmap)
    if sample_mask is not None:
        sample_mask_full_heatmap *= sample_mask[..., None, None]
else:
    sample_mask_full_heatmap = sample_mask
```
- 【直译】如果开了"锥桶类"，就造一张**逐类逐像素**的 mask：只有锥桶那一个通道被乘上一张 `close_conetank_cls_heatmap`（一张近距离区域图），其余类别全 1。
- 【为什么】锥桶/水马这类小目标只在近处才检得准（远处 0.4m 格子里根本没几个点），所以**只在近距离区域监督锥桶，远处不算它的 loss**。这是典型的"按类别定义有效区域"的量产做法。
- 【形状】`sample_mask_full_heatmap` 从 `[B,1]`（else 分支）升级成 `[B,5,448,224]`（if 分支）——两种形状都能广播进 `loss_cls`，说明 `loss_cls` 内部对 `sample_mask` 做了兼容处理。⚠ 这是一处可读性较差的写法：同名变量在两个分支里维度差 3 阶。

---

### 🔨 动手练习 ch11-4：Gaussian Focal Loss + num_pos 归一化 + sample_mask

```python
import torch

def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)

def gaussian_focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """CenterNet 版；pred 已是概率，gt 是高斯软标签"""
    pos_inds = gt.eq(1).float()
    neg_inds = 1.0 - pos_inds
    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos_inds
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg_inds
    return pos_loss.sum(), neg_loss.sum()

B, C, H, W = 2, 5, 64, 32                      # 缩小版 BEV，跑得快
gt = torch.zeros(B, C, H, W)
# 样本0：2 个目标；样本1：1 个目标
for (b, c, y, x) in [(0, 0, 20, 16), (0, 3, 40, 8), (1, 0, 10, 10)]:
    gt[b, c, y, x] = 1.0
    gt[b, c, y-1:y+2, x-1:x+2] = torch.maximum(
        gt[b, c, y-1:y+2, x-1:x+2],
        torch.tensor([[0.6, 0.8, 0.6], [0.8, 1.0, 0.8], [0.6, 0.8, 0.6]]))

pred = clip_sigmoid(torch.zeros(B, C, H, W))   # 初始 logit=0 → p=0.5

                                               # ⚠ 形状必须是 [B,1] 不是 [B]，见正文推导
sample_mask = torch.tensor([[1.0], [0.0]])     # [2,1]：样本1 无效（RL 分支缺失）
gt_masked = gt * sample_mask[..., None, None]  # [2,1,1,1] × [2,5,64,32] ✅

# 反面演示：写成 [B] 会怎样
try:
    _ = gt * torch.tensor([1.0, 0.0])[..., None, None]      # [2,1,1] × [2,5,64,32]
except RuntimeError as e:
    print("sample_mask 写成 [B] 的下场:", str(e)[:60], "...")

for name, g in [("不加 mask", gt), ("加 sample_mask", gt_masked)]:
    num_pos = g.eq(1).float().sum().item()
    pos, neg = gaussian_focal_loss(pred, g)
    print(f"{name:14s} num_pos={num_pos:.0f}  "
          f"未归一化={pos+neg:9.1f}  归一化后={(pos+neg)/max(num_pos,1):7.3f}")

# 对比：如果用像素总数归一化会怎样
num_pix = B * C * H * W
pos, neg = gaussian_focal_loss(pred, gt)
print(f"用像素数归一化: {(pos+neg)/num_pix:.6f}   ← 小 4 个数量级，正样本梯度被淹没")
```
预期输出（本机实跑，可复现）：
```
sample_mask 写成 [B] 的下场: The size of tensor a (5) must match the size of tensor b (2) ...
不加 mask       num_pos=3  未归一化=   3544.8  归一化后=1181.604
加 sample_mask  num_pos=2  未归一化=   3546.2  归一化后=1773.090
用像素数归一化: 0.173087   ← 小 4 个数量级，正样本梯度被淹没
```
> 三行都要看：
> 1. **第一行**是 §11-4 正文那个形状推导的直接验证：`sample_mask` 写成 `[B]` 会当场报错，必须是 `[B,1]`。
> 2. **中间两行**：屏蔽掉样本 1 后，分子几乎没变（3544.8 → 3546.2，因为负样本占绝对多数），分母却从 3 掉到 2 ⇒ 归一化后的 loss 反而**变大了 50%**。这说明 `num_pos` 归一化的语义是"**平均每个目标承担多少 loss**"——目标少的帧，每个目标的担子就重。这正是我们要的：loss 尺度与帧内目标数解耦。
> 3. **最后一行**：用像素总数归一化时 loss ≈ 0.1731。⚠ 上一版我在这里写的是 0.693 = ln2 并说"这是 p=0.5 时的 BCE 值"——**算错了**。Gaussian Focal 的负样本项是 `-p^α·(1-y)^β·log(1-p)`，在 p=0.5、α=2、y=0 处等于 `0.25 × 1 × 0.6931 = 0.1733`，那个 `(1-p)^α = 0.25` 的调制因子不能漏。0.1731 ≈ 0.1733（略小是因为极少数正样本格子的贡献被平均掉了）。**结论不变**：0.173 比 1181 小了 4 个数量级，整个 loss 被负样本完全支配——这就是为什么必须用 `num_pos` 而不是像素数。

### 【小结】
1. `clip_sigmoid` 把概率钳到 `[1e-4, 1-1e-4]`，是 Focal Loss 不 NaN 的第一道保险（第二道是 `@force_fp32`）。
2. `sample_mask`（实为 `rl_sample_mask`）和 `valid_lidar` 通过 `[..., None, None]` 广播把无效样本的 GT 整体清零；`num_pos` 在乘 mask **之后**统计，保证归一化因子反映真实参与训练的目标数。
3. `num_pos = heatmap.eq(1).sum()` 这个"数 1 的个数"技巧成立的前提，是 §11-2 里"高斯峰顶严格等于 1.0"的契约。

---

## Part 11-5　`anno_box` 的 10 维含义 与 `is_cls_only` 的身世（01:37:34 – 01:39:50）

**本段在讲什么**
讲者先把 `anno_box` 的 10 维一个一个念出来，然后被 `is_cls_only` 这个字段绊住，现场翻到 `load_object.py` 去查它是从哪来的。
输入：`anno_boxes[task_id] [1,256,10]`、`kwargs['is_cls_only']`。
输出：`target_box`、`not_cls_only`。
本段最大的收获不在 Loss 本身，而在于**看到了一次真实的"代码考古"过程**，并且我在画面里找到了讲者当时没找到的答案（`cls_only` 的写入位置和 git blame）。

---

**[01:37:34] "会取出我们目标的其他的属性的一个 GT，这个是有实为的。"**（01:37:55 重复）

- ⚠ "实为" = "**十维**"（10 维）。这是本段最重要的一处校正，Whisper 把"十维"听成"实为"贯穿整段。
- 【直译】`anno_box` 是 10 维的。
- 【代码】`target_box = anno_boxes[task_id]`，`target_box.shape = [1, 256, 10]`（画面 01:37:58 终端实测）。

**[01:37:56] "它给的 Poke 是对应的……这个对应的十维是中心 XYZ 三维，然后以及长宽高三维，以 sinθ 和 cosθ，然后还有的话是 VXVY，总共这个十维。"**（01:37:58–01:38:12 合并）

- ⚠ "Poke" 疑为 "**box**"；"以 3C 打 Cos2" = "**以及 sinθ cosθ**"。
- 【直译】10 维 = 中心 xyz(3) + 长宽高(3) + sinθ cosθ(2) + vx vy(2)。
- 【形状·但讲者这里有一个小口误】按画面 01:48:46 第 1021–1025 行的 concat 顺序：
  ```python
  torch.cat((reg[2], height[1], dim[3], rot[2], vel[2]), dim=1)   # 2+1+3+2+2 = 10
  ```
  **`reg` 是 2 维（dx, dy）不是 3 维**，z 由单独的 `height` 分支给。所以准确说法是"**中心 xy 的格内偏移(2) + z(1) + 长宽高(3) + sincos(2) + vxvy(2)**"。讲者说"中心 XYZ 三维"在数量上是对的（2+1=3），但把 dx,dy 说成"中心 XY"略含糊——**它们不是绝对坐标，是相对格子中心的亚像素偏移**，绝对坐标要靠 `inds` 反解出格子行列再加上偏移（画面 01:52:52 第 1157–1161 行的 `get_corner` 就是这么还原的）：
  ```python
  bbox_xs = bbox[..., 0:1] + xs[..., None]                    # 格内偏移 + 格子行号
  bbox_xs = (pc_range[0] - bbox_xs * out_size_factor * voxel_size[0])   # 转米制
  ```
- 【为什么用 sinθ/cosθ 而不是直接回归 θ】角度是**周期量**，θ=179° 和 θ=-179° 物理上只差 2°，但 L1 loss 会算成 358° 的巨大误差，梯度方向完全错误。改成 (sin, cos) 后就变成欧氏空间里的两个连续量，L1 表现良好；推理时用 `atan2(sin, cos)` 还原。代价是丢了 180° 的前后区分（sin/cos 能确定 0~360°，但网络容易学成"车头车尾都行"），所以**额外加一个 `dir_cls` 分支专门判前后**——这就是 §11-10 那个 `dir_cls` 的存在理由。
- 【为什么 dim 是 log 空间】画面 01:52:52 第 1164 行 `size2d = bbox[..., 3:5].exp()` 证明网络输出的是 `log(l), log(w), log(h)`。原因：(1) 尺寸恒正，exp 天然保证；(2) 卡车 12m 和行人 0.5m 差 24 倍，log 空间里只差 3.2，**相对误差被均衡**，不会让大目标主导 loss。Ch12 解码时讲者会说"长宽高我们需要取一个 EXP"，就是这里的逆运算。
- 【连接】和 mmdet3d CenterHead 的 `anno_box = torch.cat((center-voxel_center, z, box_dim.log(), rot_sin, rot_cos, vel), dim=?)` 完全一致。DenseBEV 没改这个契约，说明它是从 mmdet3d fork 出来的。

**[01:38:18] "然后对于……这个我看就在代码里面实现，就是从我们的标注结构里面，其实有一个字段，是看是不是主要是分类结果，就是会保存这样一个字段。"**

- 【直译】标注里有一个字段，标记这个目标（或这一帧）是不是"只做分类、不做回归"。
- 【代码】画面 01:36:13 第 844 行：
  ```python
  not_cls_only = 1 - kwargs['is_cls_only']
  ```
  用法遍布全章（画面 01:42:14 第 879 行、01:48:46 第 1005/1007 行、01:50:45 第 1051 行）：
  ```python
  attr_mask = attr_mask * not_cls_only[..., None, None]        # dense 属性
  num  = (masks[task_id] * not_cls_only).float().sum()          # 有效目标计数
  reg_mask = not_cls_only                                       # 目标级回归
  ```
- 【形状】`is_cls_only` 是 **`[B,1]`** 的 0/1（不是 `[B]`——见 §11-4 的四处交叉验证，`[B]` 在三处会直接报错）；`not_cls_only[..., None, None]` → `[B,1,1,1]` 广播到 BEV 图。
- 【为什么这么设计】这是一个**开关**：`is_cls_only=1` 时 `not_cls_only=0`，所有回归 mask 被乘 0 → 该样本只贡献分类 loss（heatmap / close_heatmap 那两项不乘 `not_cls_only`），回归 loss 完全不参与。

**[01:38:37] "它不需要再去回归目标的其他位置这些属性，所以说会有这样一个字段。"**　**[01:38:47] "这个字段是真的什么数据高的呀？"**　**[01:38:54] "这个我昨天看到其实当下还不太清楚。"**

- ⚠ "真的什么数据高的" = "**在哪个数据（集）里给的**"（"高"= "搞/给"的误听）。
- 【直译】"这个目标不需要回归位置/尺寸，所以要有这么个字段标出来"——然后讲者自问"这字段到底是哪批数据在写？"并承认"我昨天看到的时候也没搞清楚"。
- 【这是一处讲者明确表示存疑的地方，且信息量比它看起来大】注意他说的是"**我昨天看到**"——说明他是**为了这次串讲专门提前读了一遍代码**，而不是天天维护这块。这对你判断信息可靠度很有用：**凡是他说"昨天看的"、"应该是"、"感觉"的地方，就是需要你自己去代码里二次确认的地方**；凡是他直接念变量名和数字的地方，可信度就高。本章里属于前者的有：`cls_only` 的数据来源、动静四分类的历史、`rot_lidar` 那句"单帧 RL"、以及过滤前缀 `loss_`（这一条已证实他记错了，见 §11-13）。
- 【为什么"这字段哪来的"是个好问题，不是废话】因为 `is_cls_only=True` 会让**整帧的回归监督全部作废**（§11-5 后面会证明这是帧级 OR 累加）。这意味着：
  - 如果这个字段只在某个早已废弃的实验分支里写过，那它现在恒为 False，这段代码是**死逻辑**，白白增加阅读成本；
  - 如果它在当前主训练集里有一定比例，那它就在**实实在在地削减你的有效训练数据**，而且削减方式很粗暴（整帧废掉）。
  两种情况的处理完全不同。讲者问不出答案是因为要去翻数据；而我在下一段的画面里（01:39:33 的 `load_object.py` + IDE inline blame）恰好找到了它的写入位置和提交来源，**补上了他当场没答出来的那一半**。
- 【给你的行动项】这类"讲者当场没查出来"的字段，是你入职后最容易做出增量贡献的地方：写个小脚本对全量标注 json 统计 `cls_only` 的出现率和分布（按数据批次分组），三十行代码就能给出一个团队里可能没人算过的数。**读代码读不出答案的时候，就去数据里数。**

**[01:38:58] "在这里就 class only，有这样一个字段。"**　**[01:39:11] "相当于它默认就是……默认值。"**

- 【直译】搜到了字段名 `cls_only`，默认 False。
- 【代码】**画面 01:39:33 是本段的最大收获**——讲者切到了 `datasets_v2/load_object.py` 第 272 行，原文逐字：
  ```python
  class LoadObject(LoaderInterface, metaclass=LoaderClass):
      def process_list(self, ctx: DataCtx, obj_list, stop_token_dict, data_version=None):
          ...
          cnt = 0
          is_cls_only = False
          ...
          for i, obj in enumerate(obj_list):
              is_cls_only = is_cls_only or obj.get('cls_only', False)
  ```
  旁边 IDE 的 inline blame 还显示：`w00812388, 20个月前 · 同步到 StreamPETR_develop_POC 分支 @z4.0118@1cfee2651`。
- 【⚠ 我发现的一个重要语义细节，讲者没注意到】`is_cls_only = is_cls_only or obj.get('cls_only', False)` 是**在目标循环里做 OR 累加**。也就是说：
  > **只要这一帧里有任意一个目标带 `cls_only=True`，整帧就变成 cls_only，这一帧所有目标的回归 loss 全部作废。**

  这不是"逐目标屏蔽"，是"**整帧屏蔽**"。粒度非常粗。如果一批数据里只有 5% 的目标是伪标签，但它们均匀散布在各帧，那可能 60% 的帧都被整帧废掉回归监督。⚠ 这是一个值得向导师确认的设计点——是有意为之（宁可少学不可学错）还是历史遗留。
  【推断依据】(1) `is_cls_only` 是循环外初始化、循环内 `or` 累加的**帧级标量**；(2) loss 里 `not_cls_only[..., None, None]` 只补了 2 个维度就去乘 `[B,C,H,W]`，说明它是 **`[B,1]`** 的**样本级**量（§11-4 已用四处用法交叉钉死这个形状），绝不是 `[B,256]` 的目标级量。两条独立证据指向同一结论。

**[01:39:16] "相当于它也会需要去回归它的中心点呀，长宽这些属性。"**　**[01:39:21] "如果说读到了这个字段，那么如果说这个为 True 的话，它就只会去回归分类的 Loss，对其他属性它就不会去回归了。"**

- 【直译】默认（False）时正常回归；读到 `cls_only=True` 时只算分类 loss。
- 【代码·验证】翻回 loss 代码可以确认这个语义严格成立：
  ```python
  loss_heatmap        # 不乘 not_cls_only  → cls_only 时仍然算 ✅
  loss_close_heatmap  # 不乘 not_cls_only  → 仍然算 ✅
  attr_mask *= not_cls_only[..., None, None]   # dense rot/vel → 归零 ✅
  reg_mask  = not_cls_only                     # 目标级 10 维  → 归零 ✅
  mov_weight *= not_cls_only[..., None]        # 目标级动静    → 归零 ✅
  dir_cls_mask = attr_mask_ori * (dir_cls_gt > 0)   # attr_mask_ori 来自已乘过的 attr_mask → 归零 ✅
  ```
  **6 处用法完全一致，讲者的理解正确。**

**[01:39:30] "这个感觉有点奇怪。"**　**[01:39:33] "那是不是以前有一些目标是没有人工标注的，才做了这么一个事情？"**　**[01:39:40] "感觉像是伪标签那种。"**　**[01:39:43] "或者位置不太准确这些。"**　**[01:39:47] "只是我们得看一下哪些数据里有这个东西。"**

- 【直译】讲者猜测 `cls_only` 是给伪标签 / 位置不准的数据用的。
- 【我的判断：讲者的猜测大概率正确，且能补上两条旁证】
  1. **git blame 显示这行来自 "同步到 StreamPETR_develop_POC 分支"**（画面 01:39:33 行尾）。StreamPETR 是一个纯视觉时序 3D 检测方法。POC = Proof of Concept。纯视觉分支做实验时，常见做法就是**用 lidar 模型跑出伪标签来扩充数据**，而伪标签的 3D 框位置/尺寸不可信、类别相对可信——完美对应 "只算分类不算回归"。
  2. 同一个文件里紧挨着的第 273 行是 `gt_box_with_offset`（给框加 ±10% 扰动的增强），第 258–261 行有 `anomaly_tag / anomaly_length / is_child` 等字段。这是一个**明显在处理"标注质量参差"问题**的模块。`cls_only` 是这一族里最激进的一个开关。
  3. 从命名看，`cls_only` 而不是 `pseudo_label`，说明它描述的是"能用什么监督"而非"数据从哪来"，是一个通用开关，可能同时服务于伪标签、远距离粗标注、遮挡严重的框等多种来源。
- 【连接·对你的用处】你在华为做 BEV 数据时如果遇到"部分数据只有 2D 标注 / 只有类别"，这就是标准处理范式：**不要丢掉这批数据，而是给它一个 mask 让它只参与它能参与的那部分 loss**。数据永远比模型稀缺。

---

### 🔨 动手练习 ch11-5：`anno_box` 10 维 concat + `is_cls_only` 开关效果

```python
import torch

B, H, W, MAX_OBJ = 2, 448, 224, 256
preds = {
    'reg':    torch.randn(B, 2, H, W),
    'height': torch.randn(B, 1, H, W),
    'dim':    torch.randn(B, 3, H, W),
    'rot':    torch.randn(B, 2, H, W),
    'vel':    torch.randn(B, 2, H, W),
}
# 画面 01:48:46 第 1021-1025 行
anno_box = torch.cat((preds['reg'], preds['height'], preds['dim'],
                      preds['rot'], preds['vel']), dim=1)
print("anno_box.shape =", tuple(anno_box.shape))      # 期望 (2, 10, 448, 224)

LAYOUT = ['dx','dy','z','log_l','log_w','log_h','sin','cos','vx','vy']
print("切片自检:")
print("  pred[..., -4:-2] →", LAYOUT[-4:-2])          # 期望 ['sin','cos']
print("  pred[..., -2:]   →", LAYOUT[-2:])            # 期望 ['vx','vy']
print("  loss_bbox[:6]    →", LAYOUT[:6])             # 期望 dx..log_h

# ---- is_cls_only 开关 ----
# ⚠ 形状必须是 [B,1]：源码里 not_cls_only 要同时乘 [B,256]、[B,256,1]、[B,C,H,W]
#    三种秩，只有 [B,1] 能让三处同时合法（见 §11-4 的四处交叉验证表）
is_cls_only  = torch.tensor([[0.0], [1.0]])           # [2,1]：样本1 只做分类
not_cls_only = 1 - is_cls_only

masks = torch.zeros(B, MAX_OBJ); masks[0, :7] = 1; masks[1, :5] = 1
num = (masks * not_cls_only).float().sum()            # [2,256] × [2,1] ✅
print("参与回归的目标数 num =", num.item())            # 期望 7.0（样本1 的 5 个被废掉）

attr_mask = torch.ones(B, 2, H, W) * not_cls_only[..., None, None]   # [2,1,1,1] ✅
print("样本0 attr_mask 均值 =", attr_mask[0].mean().item())   # 期望 1.0
print("样本1 attr_mask 均值 =", attr_mask[1].mean().item())   # 期望 0.0

# ---- 反面演示：写成 [B] 会静默算错（B 恰好 == 通道数 2 时不报错！）----
bad = torch.ones(B, 2, H, W) * (1 - torch.tensor([0.0, 1.0]))[..., None, None]
print("[B] 版 样本0 均值 =", bad[0].mean().item(), " 样本1 均值 =", bad[1].mean().item())
print("  ↑ 两个都是 0.5：它沿【通道维】广播了，不是沿 batch 维——不报错但全错")

# ---- 帧级 OR 累加（load_object.py 第 272 行）----
frame_objs = [{'cls_only': False}, {'cls_only': False}, {'cls_only': True}]
flag = False
for o in frame_objs:
    flag = flag or o.get('cls_only', False)
print("这一帧 is_cls_only =", flag, " → 3 个目标全部丢失回归监督")  # 期望 True
```
预期输出：
```
anno_box.shape = (2, 10, 448, 224)
切片自检:
  pred[..., -4:-2] → ['sin', 'cos']
  pred[..., -2:]   → ['vx', 'vy']
  loss_bbox[:6]    → ['dx', 'dy', 'z', 'log_l', 'log_w', 'log_h']
参与回归的目标数 num = 7.0
样本0 attr_mask 均值 = 1.0
样本1 attr_mask 均值 = 0.0
[B] 版 样本0 均值 = 0.5  样本1 均值 = 0.5
  ↑ 两个都是 0.5：它沿【通道维】广播了，不是沿 batch 维——不报错但全错
这一帧 is_cls_only = True  → 3 个目标全部丢失回归监督
```

### 【小结】
1. `anno_box` 10 维 = `reg(dx,dy) + height(z) + dim(log l,w,h) + rot(sin,cos) + vel(vx,vy)`；dx,dy 是格内偏移不是绝对坐标，dim 是 log 空间。
2. `is_cls_only` 是**帧级**开关（`load_object.py` 里用 `or` 对全帧目标累加），一旦为真，该帧所有回归 loss（dense 属性 + 目标级 10 维 + 动静 + dir_cls）全部归零，只剩 heatmap / close_heatmap。
3. git blame 指向 StreamPETR POC 分支，佐证讲者"伪标签/位置不准"的猜测；这是量产数据"分级监督"的标准范式。

---

## Part 11-6　近距离 `close_heatmap`：同一份 GT，第二张预测图（01:39:50 – 01:41:28）

**本段在讲什么**
DenseBEV 给分类任务开了**第二个头** `close_heatmap`，专门监督近距离区域。
输入：`preds_dict[0]['close_heatmap'] [1,5,448,224]`、同一份 `heatmaps[task_id]`、一张预先生成的 `self.close_cls_mask`。
输出：`loss_close_heatmap`（实测 **13126.72**，占总 loss 的 99.6%，本段会分析为什么这么大）。
关键点：**GT 是同一份，mask 不同，预测图不同**——这是"多头共享 GT、按区域分工"的典型做法。

---

**[01:39:50] "然后这个呢是针对于那个……为了就是加增强回归一下近距离的这些目标。"**

- ⚠ "加增强回归" = "**加强 / 增强回归**"。
- 【直译】这一段的目的是加强近距离目标的学习。
- 【为什么近距离要特殊照顾】三个原因：(1) **安全权重不对等**——20m 内的目标漏检直接撞车，80m 外漏检只是舒适性问题；(2) **样本天然不平衡**——BEV 图上 `448×224` 的格子里，近距离（比如 30m 内）只占很小一块面积，远处占大头，均匀 loss 会让网络把容量分配给远处；(3) **近距离精度要求高**——同样 0.4m 的量化误差，在 5m 处是 8% 相对误差，在 80m 处是 0.5%。

**[01:40:00] "然后在这里其实也预测了一个近距离的一个 heatmap。"**

- ⚠ 校正稿写 "headmap"，是 "heatmap" 的误听（文件头已注明该类修复）。
- 【代码】画面 01:41:29 第 846–848 行：
  ```python
  if self.close_cls_mask is not None:
      close_heatmap = clip_sigmoid(preds_dict[0]['close_heatmap'])
      preds_dict[0]['close_heatmap'] = close_heatmap
  ```
- 【形状】`[1, 5, 448, 224]`，和主 heatmap 完全同形（终端打印确认）。

**[01:40:10] "然后因为它提前生成了这样一个 mask，就是对应 448 和 224 这里面近距离区域的一个 mask。"**

- ⚠ 校正稿写"48和224"，缺了个 4。
- 【直译】`self.close_cls_mask` 是一张 `[448, 224]` 的 0/1 图，标出"近距离区域"。
- 【代码】画面 01:41:29 第 849–851 行逐字：
  ```python
  bs = close_heatmap.shape[0]
  _cls_close_mask = torch.Tensor(
      np.tile(self.close_cls_mask[None, None, ...], (bs, 1, 1, 1))).to(_task_heatmap)
  ```
- 【形状】`self.close_cls_mask` 是 numpy `[448, 224]` → `[None, None, ...]` → `[1,1,448,224]` → `np.tile(..., (bs,1,1,1))` → `[bs,1,448,224]` → `.to(_task_heatmap)` 转成同 device/dtype 的 Tensor。
- 【为什么用 `np.tile` 而不是 `torch.Tensor.expand`】`expand` 不复制内存但结果是 view，后面 `*=` 原地乘会报错或产生意外的共享写。`np.tile` 老实复制，代价是每次 forward 都 CPU→GPU 拷一次 `bs×100352` 个 float（约 400KB @ bs=1）。⚠ **这是一处小的性能浪费**：`close_cls_mask` 是常量，完全可以用 `register_buffer` 一次性放到 GPU。在 bs=1、0.5s/iter 的量级下影响可忽略，但在大 batch 下值得优化。
- 【为什么 mask 是"预先生成"而不是按距离现算】因为它可能不是简单的圆/矩形。量产里这张图通常是按**各类别的可检测距离 + 相机 FOV 覆盖 + lidar 有效距离**取交集画出来的不规则区域，画一次存成 npy 最简单。

**[01:40:19] "所以说它能够用相同的一个 GT，然后把它 mask 掉，就只有对应近距离的一个 GT 目标。"**　**[01:40:29] "然后以及预测的近距离的 heatmap，然后去计算这个 Loss。"**

- 【直译】GT 复用主 heatmap 那一份，乘上近距离 mask 得到"只剩近处目标"的 GT，再和 `close_heatmap` 算 loss。
- 【代码】画面 01:41:29 第 852–862 行**完整逐字**：
  ```python
  if sample_mask is not None:
      if self.use_conetank_cls:
          _cls_close_mask = _cls_close_mask * sample_mask_full_heatmap
      else:
          _cls_close_mask *= sample_mask[..., None, None]

  _task_close_heatmap = _task_heatmap * _cls_close_mask          # ← 讲者说的"相乘"
  _close_num_pos = _task_close_heatmap.eq(1).float().sum().item() # ← 近距离目标个数
  loss_close_heatmap = self.loss_cls(
      close_heatmap,                 # 预测：close_heatmap 分支
      heatmaps[task_id],             # GT：注意！传的是【完整】GT，不是 _task_close_heatmap
      weight=self.cls_loss_weight,
      avg_factor=max(_close_num_pos, 1),
      sample_mask=_cls_close_mask)   # 区域限制通过 sample_mask 参数进去
  ```
- 【⚠ 一处讲者口述与代码的细微差异】讲者说"用相同的 GT 把它 mask 掉"，听起来像是把 masked GT 传进 loss。**实际代码传的是完整的 `heatmaps[task_id]`，区域限制是通过 `sample_mask=_cls_close_mask` 这个参数在 loss 内部生效的**。`_task_close_heatmap` 这个变量只用来数 `_close_num_pos`。两种写法在数学上等价（loss 内部会 `loss * sample_mask`），但代码上不是"把 GT mask 掉"。这个区别在你改代码时很重要——如果你以为传的是 masked GT，就会以为近距离外的正样本变成了负样本（会被当成 FP 惩罚），实际不会，它们是被整体屏蔽。

**[01:40:34] "对，这里去取得近距离的一个 mask，然后和我的这个 GT 相乘，其实这里取出来的话就只有近距离的一些 GT 的 heatmap 了。"**

- 【直译】此处讲者重复上述内容（第二遍描述同一个乘法），对应第 858 行 `_task_close_heatmap = _task_heatmap * _cls_close_mask`。
- 【形状·把这次相乘算清楚】`_task_heatmap [1,5,448,224]` × `_cls_close_mask [1,1,448,224]` → 沿通道维广播 → `[1,5,448,224]`。**5 个类别共用同一张空间 mask**，也就是说"近距离区域"的定义与类别无关（唯一的例外是 conetank，它在第 827–833 行有自己的逐类 mask，走 `sample_mask_full_heatmap` 那条路）。
- 【⚠ 但讲者这句话在代码层面并不准确，值得单独钉一次】他说"和 GT 相乘 ⇒ 取出来只有近距离的 GT heatmap 了"，听起来 `_task_close_heatmap` 就是要送进 loss 的 GT。**实际上 `_task_close_heatmap` 这个变量只被用了一次——下一行数 `_close_num_pos`，然后就再也没出现过。** 送进 `self.loss_cls` 的 GT 是**完整的 `heatmaps[task_id]`**（第 861 行）。
- 【这个区别为什么要紧·想清楚它你就真懂 mask 了】假设 30 米外有一辆车，close 区域只到 20 米：
  - 【如果按讲者的字面理解】GT 被 mask 成 0 ⇒ 那个格子从"正样本(y=1)"变成"负样本(y=0)" ⇒ Focal Loss 会**惩罚** close 头在那里输出高分 ⇒ 等于在教网络"远处不许报目标"。
  - 【代码实际做的】GT 保持 y=1，但 `sample_mask=_cls_close_mask` 让那个格子的 loss **整体乘 0** ⇒ 既不奖励也不惩罚，**完全不管**。
  两者对网络的引导方向**完全相反**。前者会让 close 头在远处主动压低分数（可能反过来污染共享的 BEV 特征），后者只是放任。代码选的是后者，这是对的——因为 close 头的输出在远处根本不会被解码器使用，压不压低毫无意义，何必浪费容量。
- 【连接·一条可迁移的经验】以后你在任何 loss 里看到"区域限制"，第一件事就是分清它是 **"把 GT 置零"** 还是 **"把 loss 权重置零"**。前者制造负样本，后者制造"无标注"。这两件事在语义上差一个量级，而在代码上只差一个参数位置。

**[01:40:46] "然后这是算了……我要重点去监督近距离区域的目标的一个个数。"**

- 【直译】统计近距离区域内的目标个数。
- 【代码】第 859 行 `_close_num_pos = _task_close_heatmap.eq(1).float().sum().item()`。
- 【⚠ 这里藏着 `loss_close_heatmap = 13126.72` 这个巨大数值的答案】我们来推一遍：
  - `avg_factor = max(_close_num_pos, 1)`。如果这一帧近距离区域内**一个目标都没有**（比如空旷路段，唯一的目标在 27m 外而 close 区域只到 20m），`_close_num_pos = 0` → `avg_factor = 1`。
  - 此时 Focal Loss 的分子是**整个近距离区域所有负样本的求和**。假设 close 区域约占全图 1/4 ≈ 25000 格 × 5 类 = 125000 个位置，初始 p≈0.5，每个负样本贡献 `-p^2·(1-y)^4·log(1-p) ≈ 0.25 × 1 × 0.693 ≈ 0.173`。
  - 125000 × 0.173 ≈ **21600**，再乘上 `cls_loss_weight`（各类权重，均值可能 0.5~0.7）→ **量级正好落在 13126**。✅
  - 对照主 heatmap：`num_pos ≥ 1`（帧里至少有目标），假设 num_pos ≈ 3~10，分母大了 3~10 倍，同时主 loss 的 `sample_mask` 是标量 1 覆盖全图 50 万个位置……⚠ 这里数量对不太上（全图负样本更多却 loss 更小），说明 `avg_factor` 差异之外还有 `cls_loss_weight` 的差异，或者 `_close_num_pos` 确实是 0 而 `num_pos` 有几十。
- 【我的结论与建议】`loss_close_heatmap` 在这一帧上是"病态值"，成因是 **`avg_factor = max(_close_num_pos, 1)` 在近距离无目标时退化成 1**。它进了总 loss（key 里不含 `.loss_p_`），会让这一帧的梯度被 close 分支完全支配。⚠ **这是一个真实的工程隐患**，稳健写法应该是：无近距离正样本时直接跳过这项，或者用 `max(_close_num_pos, num_pos)`、或者按区域像素数归一化。
  【推断依据】(1) 13126 与其他所有 loss（0.02~44）差 3 个数量级，不可能是正常的相对权重设计；(2) 代码里 `avg_factor=max(_close_num_pos, 1)` 白纸黑字；(3) 主 heatmap 用同一个 `loss_cls` 只有 5.02，唯一的结构差异就是 `avg_factor` 和 `sample_mask`。
  ⚠ 但也存在另一种可能：这是**训练第 0 步**的截图（预测全是 0.5），实际训练几百步后 close 区域的背景概率会被压到 1e-3，负样本 loss 掉到 `1e-6` 量级，13126 会迅速塌缩到个位数。**如果是这样就不算 bug，只是初始瞬态。** 我倾向于两者兼有：初始瞬态被 `avg_factor=1` 放大了。建议你实际训练时把这一项单独画在 TensorBoard 上看它是否收敛。

**[01:40:54] "然后这里也是一个 Focal Loss，去计算近距离的这些回归分类的 Loss。"**

- 【直译】用的还是同一个 `self.loss_cls`（Gaussian Focal Loss）。
- 【为什么复用同一个 loss 函数】保证近处/远处的分类目标函数形式一致，只是样本域不同。如果换成别的 loss，两个头输出的分数就不可比，Ch12 解码时把两张热图融合会出问题。

**[01:41:01] "然后但是近距离的那些回归了两次是什么？"**　**[01:41:07] "它都在同样的一个 feature。"**　**[01:41:09] "对对对，没有，它预测的是不同的一个 feature，是用不同的一个 feature 去预测的。"**　**[01:41:15] "只是它的那个 GT 是相同的。"**

- ⚠ 校正稿全部写作 "factor"，实为 "**feature**"（讲者口音）。这四句是讲者的自问自答，问的是"近距离目标是不是被监督了两次"。
- 【直译】答案：是的，近距离目标同时被 `heatmap` 和 `close_heatmap` 监督了两次；两个分支各有自己的卷积头（不同的 feature/预测值），但 GT 是同一份。
- 【代码·结构图】
  ```
                       ┌── conv → heatmap        [1,5,448,224] ──┐
  BEV feature ─────────┤                                          ├─→ 都拿 heatmaps[task_id] 当 GT
                       └── conv → close_heatmap  [1,5,448,224] ──┘   区别只在 sample_mask
  ```
- 【为什么不直接给主 heatmap 的近距离区域加大权重，非要开第二个头】这是本段最值得琢磨的设计问题。我的分析：
  1. **加权只能改变梯度大小，开新头能改变模型容量分配**。第二个头有自己的卷积参数，可以学一套专门适配近距离尺度/密度的特征变换（近处目标在 BEV 上占的格子多、形状大，和远处的点状目标统计特性完全不同）。
  2. **推理时可以只用其中一个**。部署时如果算力紧张，远距离可以降频跑；或者两张图按距离拼接（近处用 close_heatmap、远处用 heatmap），得到一张分段最优的热图。这在加权方案里做不到。
  3. **调试友好**。两个 loss 分开记录，能直接看出"是近处不行还是远处不行"。
- 【连接】这个思路和 FPN 的"不同尺度用不同层预测"、以及 YOLO 的多尺度检测头是同构的——只不过这里分的不是尺度而是**距离区间**。你正在吃透的 YOLO 里，P3/P4/P5 三个头也是"同一份 GT，按尺寸分配到不同头"。

**[01:41:19] "然后这个 feature 也都是 0.4 米的那个 feature。"**　**[01:41:24] "对，都是 0.4 米的一个分辨率。"**

- 【直译】两个头都在同一张 0.4m 分辨率的 448×224 BEV 上预测，没有做多尺度。
- 【为什么值得说明】因为看到"近距离专用头"，人的第一反应是"是不是用了更高分辨率（比如 0.2m）"。**答案是没有**。近距离头只是区域不同，分辨率相同。这降低了实现复杂度（不用两套 BEV 网格、不用坐标换算），代价是近距离精度上限仍受 0.4m 量化限制（靠 `reg` 分支的亚像素偏移弥补）。
- ⚠ 校正稿这两句里的 "factor" 同样应为 "feature"，且 "0.4米的一个分辨率" 是讲者的确认性重复。

---

### 🔨 动手练习 ch11-6：`close_cls_mask` 与 `avg_factor` 退化复现

```python
import torch, numpy as np

def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)

def gaussian_focal_loss_sum(pred, gt, sample_mask=None, alpha=2.0, beta=4.0):
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    l = (-torch.log(pred) * (1 - pred).pow(alpha) * pos
         - torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg)
    if sample_mask is not None:
        l = l * sample_mask
    return l.sum()

B, C, H, W = 1, 5, 448, 224

# 1) 造一张"近距离区域"mask：只有 row 300~448（车前 20m 内 ⚠ 示意）为 1
close_cls_mask = np.zeros((H, W), dtype=np.float32)
close_cls_mask[300:, :] = 1.0
_cls_close_mask = torch.from_numpy(np.tile(close_cls_mask[None, None, ...], (B, 1, 1, 1)))
print("_cls_close_mask.shape =", tuple(_cls_close_mask.shape))   # 期望 (1,1,448,224)
print("近距离区域占比 = %.1f%%" % (100 * close_cls_mask.mean()))

# 2) GT：造一帧【真实感】的场景——12 个目标全在远处（row < 300），close 区域内一个都没有
#    （上一版我只放 1 个目标，导致 num_pos=1、主 loss 也没被归一化掉，比值反而 <1，
#     那样根本演示不出问题；一帧十几个目标才是实际情况）
gt = torch.zeros(B, C, H, W)
rows = [40, 60, 80, 100, 120, 140, 160, 171, 190, 210, 240, 270]
for i, r in enumerate(rows):
    gt[0, i % 5, r, 60 + i * 12] = 1.0
pred = clip_sigmoid(torch.zeros(B, C, H, W))          # 训练第 0 步，logit=0 → p=0.5

_task_close_heatmap = gt * _cls_close_mask
close_num_pos = _task_close_heatmap.eq(1).float().sum().item()
num_pos       = gt.eq(1).float().sum().item()
print("num_pos =", num_pos, "  close_num_pos =", close_num_pos)   # 期望 12.0 / 0.0

loss_main  = gaussian_focal_loss_sum(pred, gt) / max(num_pos, 1)          # ÷12
loss_close = gaussian_focal_loss_sum(pred, gt, _cls_close_mask) / max(close_num_pos, 1)  # ÷1 !
print("loss_heatmap       = %.1f   ← 分母 = 12 个目标" % loss_main)
print("loss_close_heatmap = %.1f   ← 分母退化成 1" % loss_close)
print("比值 = %.1fx   （close 只覆盖 33%% 的像素，却比主 loss 大 4 倍）" % (loss_close / loss_main))

# 3) 如果训练已收敛（背景概率压到 1e-3），同样的公式会怎样？
pred_conv = torch.full((B, C, H, W), 1e-3)
for i, r in enumerate(rows):
    pred_conv[0, i % 5, r, 60 + i * 12] = 0.9
loss_close2 = gaussian_focal_loss_sum(pred_conv, gt, _cls_close_mask) / max(close_num_pos, 1)
print("收敛后 loss_close_heatmap = %.4f  ← 塌缩，说明 13126 里含大量初始瞬态" % loss_close2)
```
预期输出（我在本机实跑过，数值可复现）：
```
_cls_close_mask.shape = (1, 1, 448, 224)
近距离区域占比 = 33.0%
num_pos = 12.0   close_num_pos = 0.0
loss_heatmap       = 7245.7   ← 分母 = 12 个目标
loss_close_heatmap = 28724.0   ← 分母退化成 1
比值 = 4.0x   （close 只覆盖 33% 的像素，却比主 loss 大 4 倍）
收敛后 loss_close_heatmap = 0.0002  ← 塌缩，说明 13126 里含大量初始瞬态
```
> 这个练习一次演示了两件事：
> 1. **`avg_factor` 退化的放大倍数 ≈ 帧内目标数**。这一帧 12 个目标 → close 被相对放大 4 倍（= 12 × 33%）。真实帧几十个目标时，放大倍数还会更大。这就是 `loss_close_heatmap` 能爬到 13126 而 `loss_heatmap` 只有 5.02 的**结构性原因**。
> 2. 把预测从"随机 0.5"换成"已学会的 1e-3"，close loss 从 2.9 万掉到 0.0002。**所以 13126 里绝大部分是训练早期瞬态**，但 `avg_factor=max(·,1)` 会把这个瞬态放大到危险量级——两个因素叠加，而不是二选一。

### 【小结】
1. `close_heatmap` 是**独立的第二个分类头**（独立卷积参数），和主 heatmap 共用同一份 GT，只靠 `close_cls_mask` 区分作用区域；两个头都在同一张 0.4m / 448×224 的图上，没有多分辨率。
2. 区域限制不是"把 GT 乘 mask 后传进 loss"，而是通过 `sample_mask=_cls_close_mask` 参数传进去；masked GT 只用于统计 `_close_num_pos`。
3. `avg_factor=max(_close_num_pos, 1)` 在近距离无目标时退化为 1，实测让 `loss_close_heatmap` 达到 13126（占总 loss 99.6%）；⚠ 建议实际训练时单独监控该项收敛情况。

---

## Part 11-7　dense 属性总论 + 朝向 Loss（car×3 / VRU×3 / 角度分区间加权）（01:41:28 – 01:44:35）

**本段在讲什么**
进入 `if self.dense_attr:` 大分支——这是 DenseBEV 相对 CenterPoint 最有辨识度的一块。
输入：`preds_dict[0]['rot'] [1,2,448,224]`、`attr_heats[:, :2]`、`attr_heat_masks[0] [1,448,224]`。
输出：`rot_dense_loss`（实测 30.5692，是总 loss 里第二大的有效项）。
本段的技术密度极高，包含 **4 层 mask 相乘 + 2 次类别加权 + 1 次角度分桶加权**，我会把每一层都拆开。

---

**[01:41:28] "然后有一个这样是有一个 dense attribute，就是我们预测的回归的那些属性。"**

- 【直译】进入 dense 属性分支。讲者随口一句"有一个 dense attribute"，但这个 `if` 是**整章的分水岭**——它上面是标准 CenterPoint，它下面（第 864 行到第 1000 行，共 130 多行）全是 DenseBEV 自己加的东西。
- 【代码】画面 01:41:29 第 864 行：`if self.dense_attr:`
- 【这个开关管多大范围·从 01:52:45 的汇总代码反推】第 1123 行还有一个对称的 `if self.dense_attr:` / `else:`，所以这个开关同时控制两件事：
  ```python
  if self.dense_attr:                                   # 864  开始算 dense loss
      ... rot_dense / vel_dense / rot_lidar / lidar_rot_w / movement_dense / dir_cls ...
  ...
  if self.dense_attr:                                   # 1123 汇总时也分叉
      bbox_loss_total = loss_bbox[..., :6].sum() + vel_dense_loss + rot_dense_loss
  else:
      bbox_loss_total = loss_bbox.sum()                 # 1140
  ```
- 【⚠ 关掉 `dense_attr` 会发生什么——这是理解本章架构的关键实验】
  1. `loss_bbox` 从"只取前 6 维"变成"10 维全取" ⇒ **目标级的 rot/vel 突然开始参与训练**。
  2. 而目标级 rot 的加权代码里有 §11-12 那个 `torch.abs(rot_weight)` 的 bug ⇒ **一个错误加权的 loss 直接进了反传**。
  3. 同时 `rot_lidar` / `lidar_rot_weight` / `movement_dense` / `dir_cls` 四条监督全部消失。
  **所以 `dense_attr=False` 不是"少学一点"，而是"换了一套完全不同的训练目标，且踩到一颗雷"。** 这也是我在 §11-12 说那个 bug"现在无害、将来有害"的具体含义。
- 【连接】对照 BEVFusion / mmdet3d：那边根本没有这个开关，`loss_bbox` 永远是 10 维全算。**DenseBEV 的 `dense_attr=True` 路径等于把"朝向和速度"从目标级搬到了稠密级**——同一个信息，换了个监督位置。这是本章最该记住的一次架构取舍。

**[01:41:36] "其实它不仅仅只是对应的哪一个 pixel 上去回归，它其实还会在就是 BEV 的 feature 上去回归它的一些属性。"**

- 【直译】属性不只在目标中心那一个像素上监督，还在整张 BEV 图上监督。
- 【为什么·这一句是本章最重要的设计陈述之一】它明确了 DenseBEV 的双轨监督：
  ```
  目标级（点）：pred → gather_feat(inds) → [1,256,10] ─ L1 ─ anno_box[1,256,10]
  稠密级（面）：pred [1,2,448,224]       ────────── L1 ─ attr_heats[:, :2]
  ```
  同一个 `rot` 分支的输出，**被两个 loss 同时约束**。目标级保证峰值像素准，稠密级保证整片区域准。
- 【连接】用 YOLO 的语言说：这相当于 anchor-free 检测里的 "center sampling" 从 3×3 扩大到整个 box 内部，而且只对部分属性（rot/vel/movement/dir）扩大，对 box 几何量不扩大。

**[01:41:50] "我刚刚说的除了这里的就是这里的 sinθ cosθ，以及 VxVy，还有动静和朝向，它其实还会在 BEV 的那个 feature 上去计算 Loss。"**

- ⚠ 校正稿 "实为的cosθ" = "**sinθ cosθ**"；"factor" = "feature"。
- 【直译】做 dense 监督的属性有四组：朝向 (sin,cos)、速度 (vx,vy)、动静、朝向分类。
- 【代码·对应关系一览】（这四组在后面各占一小节）
  | 属性 | 预测分支 | GT 切片 | loss 类型 | 实测值 |
  |---|---|---|---|---|
  | 朝向 | `rot [1,2,·]` | `attr_heats[:, :2]` | `loss_bbox`(L1) ×2 | `loss_p_rot_dense=30.5692` |
  | 速度 | `vel [1,2,·]` | `attr_heats[:, 2:4]` | `loss_bbox`(L1) ×5 | `loss_p_vel_dense=7.9563` |
  | 动静 | `movement [1,1,·]` | `attr_heats[:, 4]-1` | `BCEWithLogitsLoss` ×10 | `loss_movement_dense=6.3973` |
  | 朝向分类 | `dir_cls [1,1,·]` | `attr_heats[:, 8]` | `loss_dir_cls_func` ×0.5 | `loss_dir_cls=1.5016` |
  | (lidar 增强朝向) | `rot_lidar [1,2,·]` | `attr_heats[:, :2]` | `loss_bbox`(L1) ×2 | 并入 rot_dense |
  | (lidar 朝向权重) | `lidar_rot_weight [1,1,·]` | `close_rot_heatmap` | `loss_cls`(Focal) ×0.1 | `loss_p_lidar_rot_w=1.6938` |

**[01:42:11] "这里是取……这里就是预测的 sinθ 和 cosθ 的一个 GT 和预测值。"**

- 【代码】画面 01:42:14 第 870–872 行逐字：
  ```python
  else:                                       # 非 corner-det 分支（当前配置）
      rot_pred = preds_dict[0]['rot']         # [1,2,448,224]
      rot_gt   = attr_heats[task_id][:, :2]   # [1,2,448,224]
  ```
  对照 if 分支（`enable_corner_det=True` 时，当前未启用）：
  ```python
  rot_pred       = preds_dict[0]['rot_long']
  rot_gt         = attr_heats[task_id][:, :2]
  rot_short_pred = preds_dict[0]['rot_short']
  rot_short_gt   = attr_heats[task_id][:, -3:-1]
  ```
- 【⚠ corner-det 分支的通道账对不上，我记一笔】当前 `attr_heats` 是 9 通道，`[:, -3:-1]` = 通道 6,7，而表 C 里通道 6,7 是"动静四分类类别/权重"。矛盾。**推断：`enable_corner_det=True` 时 `attr_heats` 会多出 2 个通道变成 11，此时 `-3:-1` = 通道 8,9 才是 rot_short**，而 dir_cls 索引公式里的 `(6 if enable_corner_det else 4)` 也印证了 corner 模式下前置通道要多 2 个。由于该分支当前关闭（终端打印里没有 `rot_long/rot_short` 这两个 key），不影响本章理解，但读代码时不要被这行误导。
- 【连接】"long / short" 指的是长边朝向和短边朝向。对方形物体（比如正方形的施工牌），长短边朝向 90° 简并，用两套 sin/cos 分别回归再加一致性约束，是解 90° 歧义的一种做法。

- 【补充·corner 分支的完整三件套，画面 01:44:47 第 913–920 行逐字，讲者完全跳过了，但这段很值得看】
  ```python
  if self.enable_corner_det:
      rot_dense_loss += self.loss_bbox(rot_short_pred, rot_short_gt, attr_mask,        # 914-915
          avg_factor=(attr_mask > 0).eq(1).float().sum().item()) * 2
      # 朝向一致性约束                                                                  # 917
      loss_rot_cross = (rot_pred[:, 0:1, :, :] * rot_short_pred[:, 0:1, :, :]          # 918
                      + rot_pred[:, 1:2, :, :] * rot_short_pred[:, 1:2, :, :]) * ...
      loss_rot_cross = loss_rot_cross.sum() / max(1, (attr_mask > 0).eq(1).float().sum().item())   # 919
      loss_dict[f'task{task_id}.loss_rot_cross'] = torch.abs(loss_rot_cross)           # 920
  ```
  - 【这个式子在算什么】`sin_long·sin_short + cos_long·cos_short` 正是两个单位方向向量的**点积**，也就是 `cos(θ_long − θ_short)`。
  - 【所以约束是什么】外面套了 `torch.abs(...)` 再当 loss 最小化 ⇒ 逼 `|cos(Δθ)| → 0` ⇒ **`Δθ → ±90°`**。也就是强制"长边朝向"和"短边朝向"这两个头**必须相互垂直**。非常干净：不用写 `atan2` 求角、不用处理周期性，一个点积就把正交性约束表达完了，而且梯度处处良好。
  - 【为什么要这么绕】直接回归一个 θ 再加 90° 不就完了？因为方形/近方形物体上长短边**哪个是"长"本身就不确定**，网络会在两种解之间来回跳。开两个头各学各的、再用点积把它们钉成正交，等于给了网络"两个解都可以，但必须成对出现"的自由度——这是处理离散简并的标准手法（和 6D 旋转表示、双分支 keypoint 匹配同源）。
  - 【⚠ 一个容易看漏的点】`loss_rot_cross` 的 key 是 `task{id}.loss_rot_cross`，**不含 `.loss_p_` 前缀** ⇒ 按 §11-13 的过滤规则，它**会直接进入总 loss** 参与反传（不像 `loss_p_rot_dense` 那样只是日志）。当前 `enable_corner_det=False` 所以没跑，但一旦开启 corner 分支，这是一项真实的训练信号。
  - 【顺带确认一件事】第 914 行是 `rot_dense_loss += ...`——**corner 模式下 `rot_short` 的 loss 也是累加进 `rot_dense_loss` 的**，和 §11-9 里 `rot_lidar_loss` 的处理方式一样。所以 `loss_p_rot_dense` 这一个数字最多可能是**三份 loss 之和**（rot + rot_lidar + rot_short），拆不开。这一点在看 TensorBoard 时非常重要。

**[01:42:24] "然后这里是取出了速度。"**

- 【代码】画面 01:42:14 第 874–875 行：
  ```python
  v_pred = preds_dict[0]['vel']              # [1,2,448,224]
  v_gt   = attr_heats[task_id][:, 2:4]       # [1,2,448,224]
  ```

**[01:42:27] "然后这里是对应的属性它的一个 mask。"**

- 【代码】画面 01:42:14 第 877–878 行逐字：
  ```python
  attr_mask = torch.cat([attr_heat_masks[0].unsqueeze(1),
                         attr_heat_masks[0].unsqueeze(1)], dim=1)
  ```
- 【形状】`attr_heat_masks[0]` `[1,448,224]` → `unsqueeze(1)` `[1,1,448,224]` → cat 两份 → **`[1,2,448,224]`**，正好和 rot/vel 的 2 通道对齐。
- 【为什么复制两份而不是 `expand`】`expand` 出来的是 view，后面第 890 行有 `attr_mask = attr_mask * (...)`（非原地，安全）但第 933 行有 `v_attr_mask[vel_mask] *= ...`（**原地索引赋值**），对 expand view 做这个会污染原始数据。用 `cat` 强制复制是稳的。

**[01:42:32] "然后这个就是看是否需要回归它的这个……是否只是……不是只需要去监督分类的。"**

- 【代码】第 879 行（帧证 01:42:14）：
  ```python
  attr_mask = attr_mask * not_cls_only[..., None, None]
  ```
- 【直译】第 1 层 mask：`cls_only` 帧整体归零。讲者这句话结结巴巴改了三次口，但意思很清楚——"这一层是在判断这帧要不要做回归"。
- 【形状】`not_cls_only` 是 **`[B,1]`**（§11-5 第 844 行 `1 - kwargs['is_cls_only']`）→ `[..., None, None]` 补两维得 `[B,1,1,1]` → 与 `attr_mask [B,2,448,224]` 广播相乘。**注意补的是 2 个 None 不是 3 个**——这从形状上反证了 `is_cls_only` 是**样本级**（`[B,1]`）而不是 `[B,256]` 的目标级量，是我在 §11-5 判定它为"帧级开关"的两条独立证据之一。
- 【⚠ 这一行是本章最容易写出静默 bug 的地方】如果 `is_cls_only` 真的是 `[B]`，这一行在 **B=2 时不会报错**（`[2,1,1]` 会沿通道维对上 rot 的 2 通道），但会把"样本 0 的开关"错用成"通道 0 的开关"，算出来全错。同一行代码在 B=1 和 B=4 时又会正常/报错——**batch size 一改行为就变，这类 bug 极难定位**。所以看到 `[..., None, None]` 这种写法，第一件事永远是数清楚被乘对象的秩。
- 【为什么用"乘 0"而不是 `if` 跳过】这是本章反复出现的范式，这里再钉一次：`if is_cls_only: skip` 在 batch 里做不到——同一个 batch 的 8 个样本可能有的要跳有的不跳，而 CUDA kernel 是整批一起跑的。**乘性 mask 是"逐样本 if"在张量世界里的唯一写法。** 代价是那些被乘 0 的样本仍然完整算了一遍前向和 loss（算力白费），换来的是 batch 形状固定、DDP 各卡同步、导出图静态。量产代码几乎永远选后者。
- 【⚠ 这一层的位置很讲究】它乘在 `attr_mask_ori = attr_mask[:, :1]`（第 884 行）**之前**、car×3 / VRU×3（第 889/896 行）**之前**。所以：
  ```
  attr_heat_masks(有目标) → ×not_cls_only(第879行) → ×sample_mask(第882行)
        ├──→ attr_mask_ori（第884行留存，dir_cls 用）      ← 已含前两层，不含权重
        └──→ ×car3 ×近处VRU3 ×角度桶 → rot_attr_mask（rot 用）
  ```
  **两层"有效性"是共享的，三层"重要性"是 rot 独享的。** 这个分叉点设计得很干净：有效性人人都要，重要性因任务而异。
- 【连接】回看 §11-5 我列的那 6 处 `not_cls_only` 用法，这里是第一处。你可以把它当成"数据质量分级"这条主线在 dense 侧的入口——同一批数据，能给什么监督就给什么监督，给不了的乘 0，绝不丢掉整帧。

**[01:42:48] "然后这个 mask 需要相乘。"**　**[01:42:50] "然后呢会和这个 sample mask 也会相乘。"**　**[01:42:54] "sample mask 呢就是看有些目标是否这一帧是否是有效的。"**

- 【代码】第 881–882 行：
  ```python
  if sample_mask is not None:
      attr_mask *= sample_mask[..., None, None]
  ```
- 【直译】第 2 层 mask：RL 分支无效的样本整体归零。
- 【⚠ 讲者措辞小误】他说"有些**目标**是否这一帧是否有效"，但 `sample_mask` 是**样本级**（`[B]`）不是目标级。目标级的有效性是 `masks[task_id]`。这两个概念在本章里容易混，我列一张对照表：

  | mask 名 | 形状 | 语义 | 用在哪 |
  |---|---|---|---|
  | `masks[task_id]` | `[B,256]` | 第 i 个**槽位**是否是真目标 | 目标级回归 |
  | `sample_mask`(=`rl_sample_mask`) | `[B,1]` | 这个**样本**的 RL 分支是否有效 | 全部 loss |
  | `valid_lidar` | `[B,1]` | 这个样本的 **lidar** 是否有效 | heatmap GT |
  | `is_cls_only` | `[B,1]` | 这**帧**是否只做分类 | 全部回归 loss |
  | `attr_heat_masks[0]` | `[B,448,224]` | 这个**格子**上有没有目标 | dense 属性 |
  | `close_cls_mask` | `[448,224]` | 这个格子是否在**近距离区域** | close_heatmap / VRU 加权 |
  | `lidar_front_mask` | `[B,448,224]` ⚠ | 这个格子是否在 **lidar 前向有效区** | rot_lidar |

**[01:42:59] "然后对于速度的 mask 呢，因为我们有些目标可能速度在标注的时候可能给的是一个无效的值。"**　**[01:43:15] "比如说在这里会去判断我的这个 GT 是否小于 50 米每秒。"**

- 【代码】画面 01:42:14 第 884–885 行逐字：
  ```python
  attr_mask_ori = attr_mask[:, :1]                                       # 留一份未加权的
  v_attr_mask   = attr_mask * (v_gt[:, 0] < INVALID_VELOCITY).float().unsqueeze(1)
  ```
- 【形状】`v_gt[:, 0]` `[1,448,224]` → 比较得 bool → `.float().unsqueeze(1)` `[1,1,448,224]` → 与 `attr_mask [1,2,448,224]` 广播 → `v_attr_mask [1,2,448,224]`。
- 【直译】第 3 层 mask（只给速度）：GT 速度 vx 大于等于 `INVALID_VELOCITY` 的位置不参与速度 loss。
- 【关于"50 米每秒"】讲者说阈值是 50 m/s，代码里是常量 `INVALID_VELOCITY`。⚠ 我在画面里没找到它的定义，只能确认名字。50 m/s = 180 km/h，作为"物理上不可能的速度"哨兵值很合理（标注端把未知速度填成 99 或 999）。**讲者说的 50 应该可信**，但注意代码只判了 `v_gt[:, 0]`（**只看 vx**）——如果哨兵值只写进了 vy 而 vx 是 0，这个过滤会失效。⚠ 稳健写法应该是 `(v_gt.abs() < INVALID_VELOCITY).all(dim=1)`。
- 【为什么】不过滤会怎样？一个 999 的假速度进 L1 loss，单点贡献 ~999 的损失和梯度，直接把 `vel` 分支的权重炸飞。这类"哨兵值污染"是量产数据里最常见的训练崩溃原因之一。
- 【连接】目标级那边有对应的一份（画面 01:48:46 第 1027–1028 行）：
  ```python
  valid_velocity = (target_box[..., -2:] < INVALID_VELOCITY).float()
  mask[..., -2:] *= valid_velocity
  ```
  这里就正确地对 vx、vy **两维都判了**。所以 dense 那边只判 `[:, 0]` 更像是笔误。

**[01:43:22] "然后然后嚷嚷……"**

- ⚠ 纯语气词/幻听，无内容。

**[01:43:29] "然后这里的话会把我的那个……对于 GT 里面所引为 0 到 1，其实就是所引为 0 的，其实就是 car，会把 car 的这个权重给它乘上一个 3，会给它加大。"**

- ⚠ "所引为" = "**索引为**"；"卡" = "**car**"。
- 【直译】类别索引 0 是 car，把 car 位置上的朝向 loss 权重乘 3。
- 【代码】画面 01:43:38 第 887–889 行**逐字（含中文注释原文）**：
  ```python
  if self.rot_spec:
      # 把car的rotation权重乘3
      attr_mask = attr_mask * ((_task_heatmap[:, :1] > 0).float() * 2 + 1)
  ```
- 【形状】`_task_heatmap[:, :1]` 是 car 那一个通道 `[1,1,448,224]`；`(>0).float()*2+1` 得到一张**值域 {1, 3}** 的权重图；乘到 `attr_mask [1,2,448,224]` 上广播。
- 【为什么是 `*2+1` 而不是 `*3`】因为要保证**非 car 位置权重是 1 而不是 0**。`(mask)*3` 会把所有非 car 位置清零（连别的类都不学了）。`*2+1` 是"基础 1，car 额外 +2"的标准写法。这是个小但典型的技巧，你写加权代码时会反复用到。
- 【⚠ 注意是 `> 0` 不是 `== 1`】用的是高斯值 `> 0`，也就是**整个高斯衰减邻域**都加权，不只是峰顶。这与 dense 监督的思路一致。
- 【为什么 car 要加权】car 是绝对主类（占目标数 60%+），而且朝向对规控最关键（判断切入/对向）。另一方面，car 的朝向也最容易学——所以这里的 ×3 更可能是为了**压过其他类的数量优势**，保证 car 的朝向精度达到最高标准。
- 【连接】和 §11-8 的速度分区间加权、§11-10 的动静静止×3，构成了这套代码统一的"**乘性权重图**"范式：所有加权都不改 loss 公式，只改那张 mask/weight 图。这种设计非常好维护——所有权重逻辑集中在一处，且天然支持叠乘。

**[01:43:51] "然后对应的话，这个三到四，就是索引为三，其实就是 VRU，会把它的一个权重会给它加大。"**

- ⚠ "塞个例子的" 是 Whisper 幻听，无意义，跳过。
- 【直译】类别索引 3 是 VRU（弱势交通参与者），也加权。
- 【代码】画面 01:43:38 第 890–896 行**逐字（含中文注释原文）**：
  ```python
      # 把近处vru的rotation权重乘3
      if self.close_cls_mask is not None:
          if self.use_conetank_cls:
              vru_pos_mask = (_task_heatmap[:, 3:4] > 0) * _cls_close_mask[:, 0:1]
          else:
              vru_pos_mask = (_task_heatmap[:, 3:4] > 0) * _cls_close_mask
          attr_mask = attr_mask * (vru_pos_mask.float() * 2 + 1)
  ```
- 【⚠ 讲者漏说了一个关键限定词】代码注释写的是"**把近处 vru** 的 rotation 权重乘 3"，`vru_pos_mask` 是 `(VRU 通道 > 0)` **与** `_cls_close_mask`（近距离区域）**相乘**。也就是说：**只有近距离的 VRU 才加权，远处的 VRU 不加权**。讲者只说了"索引为三就是 VRU 会给它加大"，没提近距离限定。**代码是权威。**
- 【为什么只加权近处 VRU】远处的行人/自行车在 0.4m BEV 上只有 1~2 个格子，朝向本身就不可观测（点太少），强行加权只会学到噪声。近处 VRU 的朝向则直接决定"他要不要横穿"，是刹车决策的关键输入。**这是一个非常克制、非常"量产"的设计**：加权只加在信息量确实存在的地方。
- 【形状·叠乘效果】走完这两步，`attr_mask` 的取值可能是：
  ```
  背景空地           : 0    （attr_heat_masks 为 0）
  普通类（truck/bus）: 1
  car                : 3
  近处 VRU           : 3
  car ∩ 近处VRU 重叠 : 9    （极端情况，理论上不该发生）
  ```
  ⚠ 9 这个情况在理论上会出现（同一格子既有 car 高斯尾巴又有近处 VRU 高斯尾巴），代码没有做 clamp。实践中影响不大，但值得知道。

**[01:44:01] "然后这里实现的呢，就是我们按照那个朝向角，就是不同的朝向角的一个区间，会给不同的一个权重值。"**

- 【直译】按 GT 朝向角所在的角度区间，再乘一层权重。
- 【代码】画面 01:43:38 第 898–909 行**完整逐字**：
  ```python
  # 朝向区间不同weight
  rot_sine   = rot_gt[:, 0, :, :]
  rot_cosine = rot_gt[:, 1, :, :]
  rot = torch.atan(rot_sine / (rot_cosine + 1e-10))    # -pi/2~pi/2
  rot = rot * 180 / np.pi
  rot_attr_mask = attr_mask.clone()
  if self.rot_range_weight:
      for rot_i in self.rot_range_weight:
          rot_mask = ((torch.abs(rot) >= self.rot_range_weight[rot_i]['range'][0]) &
                      (torch.abs(rot) <  self.rot_range_weight[rot_i]['range'][1]))
          rot_mask = rot_mask.unsqueeze(1).repeat(1, 2, 1, 1)
          rot_attr_mask[rot_mask] *= self.rot_range_weight[rot_i]['weight']
  ```
- 【形状】`rot_gt[:, 0, :, :]` → `[1,448,224]`；`rot` 同形，单位是**度**，值域 `(-90, 90)`；`rot_mask.unsqueeze(1).repeat(1,2,1,1)` → `[1,2,448,224]` 与 `rot_attr_mask` 同形，然后用**布尔索引原地乘**。
- 【为什么用 `torch.atan` 而不是 `atan2`】`atan(sin/cos)` 的值域是 `(-90°, +90°)`，**主动把 180° 的前后歧义折叠掉**。这是有意的：这里只是想按"车身朝向相对本车的角度"分桶（0° = 与本车同向/对向，90° = 横向），前后不重要。用 `atan2` 得到 `(-180,180]` 反而会让 +170° 和 -170° 落进不同的桶。
  ⚠ 但 `atan(sin/cos)` 在 `cos → 0`（正横向）时会数值爆炸，所以加了 `+1e-10`。当 cos 恰为 0 时 `sin/0 = ±1e10`，`atan(1e10) ≈ ±90°`，结果仍然正确。这个 `1e-10` 是有讲究的（不是随手写的 eps）。
- 【为什么要按角度分桶加权】典型场景：
  - **横向目标（|θ| 接近 90°）**：横穿马路的车/人，最危险，而且 BEV 上横向目标的点云分布最不利于朝向估计（只看得到侧面），**误差最大 → 需要加权**。
  - **同向目标（|θ| 接近 0°）**：前车，数量最多，最好学，可能反而降权避免主导。
  - 具体的 `range/weight` 配置在 yaml 里，画面没显示。⚠ 我只能确定机制，不能确定具体数值。
- 【连接】这和 §11-8 速度按大小分桶、Focal Loss 按难易加权，是同一个哲学：**在 loss 层面把"数据分布不均"和"任务难度不均"显式补偿掉**，而不是指望网络自己学会。

**[01:44:18] "然后回归……在 BEV 上回归的 sinθ 和 cosθ，然后用的 Loss 的话就是 L1Loss，这个是算的朝向角的一个 Loss。"**

- 【代码】画面 01:43:38 第 911–912 行逐字：
  ```python
  rot_dense_loss = self.loss_bbox(rot_pred, rot_gt, rot_attr_mask,
      avg_factor=(rot_attr_mask > 0).eq(1).float().sum().item()) * 2
  ```
- 【形状】进两张 `[1,2,448,224]` + 一张同形权重 → 出标量。实测 `loss_p_rot_dense = 30.5692`。
- 【关于 `self.loss_bbox`】mmdet3d 里通常是 `L1Loss(reduction='none')` 再乘权重求和。讲者说"用的 Loss 就是 L1Loss"，与 mmdet3d 的 `loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25)` 一致。
- 【⚠ `avg_factor` 这里的写法有点怪】`(rot_attr_mask > 0).eq(1).float().sum()` = "权重大于 0 的元素个数"。注意 `.eq(1)` 作用在 bool 张量上，等价于恒等（bool 的 True 就是 1），所以整个表达式 = `(rot_attr_mask > 0).sum()`。**`.eq(1)` 是冗余的**，这个冗余写法在本章出现了至少 5 次（`_close_num_pos`、`v_attr_mask`、`mov_attr_mask`、`rot_attr_mask_2`…），是复制粘贴的痕迹。
- 【⚠ 更重要的一点：分母数的是"元素个数"不是"目标个数"】一个 car 占 55 个格子 × 2 通道 = 110 个元素，权重 3 → `>0` 计数仍是 110（不是 330）。所以 `avg_factor` ≈ 有效元素总数，而分子里 car 的贡献被放大了 3 倍。**净效果是 car 的相对权重确实提升了 3 倍**，符合意图。✅
- 【关于末尾的 `* 2`】整个 dense rot loss 再乘 2，这是任务间的手工权重。结合 §11-8 的 `* 5`（vel）、§11-10 的 `* 10`（movement），可以看出这套代码是**纯手工调 loss_weight**，没有用 uncertainty weighting / GradNorm 那套自动方法。⚠ 不过画面 01:39:02 显示 `solver.py` 里有 `self.weighter.get_weighted_loss(all_task_losses, reweighting=self.check_reweighting(), ...)` 以及 `projected_grad_norms / fused_grad_norm`，说明**更外层还有一套自动的多任务梯度手术（类似 PCGrad / GradNorm）**。所以是"头内手工权重 + 头间自动权重"的两级结构。

---

### 🔨 动手练习 ch11-7：4 层 mask 叠乘 + car×3 + 近处VRU×3 + 角度分桶

```python
import torch, numpy as np

B, H, W, NUM_CLS = 1, 448, 224, 5
CAR, VRU = 0, 3

# ---- 造场景：3 个目标 ----
#  A: car, 正前 27m (row171,col112), 朝向 5°   （同向）
#  B: VRU, 近处 (row380,col100),     朝向 88°  （横穿）
#  C: truck, 远处 (row60,col60),     朝向 45°
heatmap = torch.zeros(B, NUM_CLS, H, W)
attr_heats = torch.zeros(B, 9, H, W)
attr_heat_masks_0 = torch.zeros(B, H, W)

def put(cls_id, r, c, deg, hr=5, wr=2):
    heatmap[0, cls_id, r-hr:r+hr, c-wr:c+wr] = 0.9
    heatmap[0, cls_id, r, c] = 1.0
    th = np.deg2rad(deg)
    attr_heats[0, 0, r-hr:r+hr, c-wr:c+wr] = np.sin(th)
    attr_heats[0, 1, r-hr:r+hr, c-wr:c+wr] = np.cos(th)
    attr_heat_masks_0[0, r-hr:r+hr, c-wr:c+wr] = 1.0

put(CAR, 171, 112, 5);  put(VRU, 380, 100, 88);  put(1, 60, 60, 45)

close_cls_mask = torch.zeros(1, 1, H, W); close_cls_mask[:, :, 300:, :] = 1.0

# ---- 第1层：attr mask + not_cls_only ----
attr_mask = torch.cat([attr_heat_masks_0.unsqueeze(1)] * 2, dim=1)   # [1,2,448,224]
not_cls_only = torch.tensor([1.0])
attr_mask = attr_mask * not_cls_only[..., None, None]
print("L1 唯一权重值:", sorted(set(attr_mask.unique().tolist())))          # [0.0, 1.0]

# ---- 第2层：car ×3 ----
attr_mask = attr_mask * ((heatmap[:, :1] > 0).float() * 2 + 1)
print("L2 唯一权重值:", sorted(set(attr_mask.unique().tolist())))          # [0.0, 1.0, 3.0]

# ---- 第3层：近处 VRU ×3 ----
vru_pos_mask = (heatmap[:, VRU:VRU+1] > 0) * close_cls_mask
attr_mask = attr_mask * (vru_pos_mask.float() * 2 + 1)
print("L3 唯一权重值:", sorted(set(attr_mask.unique().tolist())))          # [0.0, 1.0, 3.0]

# ---- 第4层：角度分桶 ----
rot_gt = attr_heats[:, :2]
rot = torch.atan(rot_gt[:, 0] / (rot_gt[:, 1] + 1e-10)) * 180 / np.pi     # [1,448,224] 度
rot_range_weight = {'near0': dict(range=[0, 30], weight=1.0),
                    'mid':   dict(range=[30, 60], weight=1.5),
                    'lateral': dict(range=[60, 90.1], weight=3.0)}
rot_attr_mask = attr_mask.clone()
for k in rot_range_weight:
    m = (rot.abs() >= rot_range_weight[k]['range'][0]) & (rot.abs() < rot_range_weight[k]['range'][1])
    m = m.unsqueeze(1).repeat(1, 2, 1, 1)
    rot_attr_mask[m] *= rot_range_weight[k]['weight']

print("最终权重直方图:")
for v in sorted(set(rot_attr_mask.unique().tolist())):
    print(f"   w={v:5.1f}  格数={int((rot_attr_mask==v).sum())}")
print("car(5°) 处权重 =", rot_attr_mask[0,0,171,112].item())     # 1*3*1.0 = 3.0
print("VRU(88°,近) 权重 =", rot_attr_mask[0,0,380,100].item())   # 1*3*3.0 = 9.0
print("truck(45°,远) 权重 =", rot_attr_mask[0,0,60,60].item())   # 1*1*1.5 = 1.5

avg_factor = (rot_attr_mask > 0).float().sum().item()
rot_pred = torch.zeros_like(rot_gt)
rot_dense_loss = (torch.abs(rot_pred - rot_gt) * rot_attr_mask).sum() / max(avg_factor, 1) * 2
print("rot_dense_loss =", round(rot_dense_loss.item(), 4))
```
预期输出（关键三行）：
```
car(5°) 处权重 = 3.0
VRU(88°,近) 权重 = 9.0
truck(45°,远) 权重 = 1.5
```
> 亲手跑一遍，你会立刻理解"乘性权重图"这个范式：**每加一条业务规则，就多乘一张图，loss 公式一个字不改。**

### 【小结】
1. dense 属性 loss 的核心是一张逐像素权重图 `rot_attr_mask`，由 4 层相乘叠出来：`有目标 × not_cls_only × sample_mask × car(×3) × 近处VRU(×3) × 角度桶权重`。
2. 代码注释白纸黑字写明 "把 car 的 rotation 权重乘3" / "把**近处** vru 的 rotation 权重乘3"——讲者漏说了 VRU 的"近处"限定，这是真正的设计意图（远处 VRU 朝向不可观测）。
3. `rot = atan(sin/cos)` 主动折叠 180° 歧义、值域 (-90°,90°)、单位换成度再分桶；`+1e-10` 防 cos=0；末尾 `* 2` 是手工任务权重，外层 solver 还有一套自动重加权。

---

## Part 11-8　dense 速度 Loss：模长分桶 + 低速加权 + 欠估惩罚（01:44:35 – 01:45:32）

**本段在讲什么**
和朝向同构的一段，但权重规则更复杂：不只按速度大小分桶，还有一条"预测比 GT 小则额外惩罚"的非对称项。
输入：`v_pred / v_gt [1,2,448,224]`、`v_attr_mask`。
输出：`vel_dense_loss`（实测 7.9563，权重 `* 5`）。
本段最值得学的是那个 **`neg_vel_loss` 非对称惩罚**——它是纯粹从"漏刹车比误刹车危险"这个业务判断推导出来的 loss 设计。

---

**[01:44:36] "然后这里会……这里是去我们对这里的 v GT，其实还是 VXVY。"**

- 【代码】`v_gt = attr_heats[task_id][:, 2:4]`，两个通道分别是 vx、vy（车体系下的地面速度）。
- 【连接】nuScenes 里 velocity 也是 2 维（vx, vy），不含 vz——因为地面车辆的垂直速度可以忽略。

**[01:44:47] "然后这里会算一下它的一个绝对的一个速度，会算一个 Norm。"**

- 【代码】画面 01:45:05 第 923–925 行**逐字（含中文注释原文）**：
  ```python
  # 把低速loss权重乘2 (0.05m/s-1.0m/s)  静止loss权重乘1.5 (< 0.05m/s)
  v_gt_norm   = torch.norm(v_gt,   p=2, dim=1, keepdim=False)
  v_pred_norm = torch.norm(v_pred, p=2, dim=1, keepdim=False)
  ```
- 【形状】`[1,2,448,224]` → `norm(dim=1)` → **`[1,448,224]`**（通道维被 reduce 掉）。这就是为什么后面所有 `v_gt_norm > 0.05` 之类的比较结果都要 `.unsqueeze(1)` 才能乘回 `[1,2,448,224]`。
- 【为什么要算模长】速度权重的分桶依据是"车走得快不快"这个**标量**，不能对 vx、vy 分别判（一个车 vx=0.7, vy=0.7 时模长是 0.99，属于低速，但分量判断会以为两个方向都很慢）。
- 【为什么连 `v_pred_norm` 也算】因为下面的非对称惩罚要比较预测和 GT 的模长大小关系。

**[01:44:55] "然后在这里会去判断我的这个 GT 是……真正的 GT 速度的一个 GT，它是根据不同的一个区间会去给回归……就是算速度的 Loss 的时候也会给它不同的一个权重。"**

- 【代码】画面 01:45:05 第 927–933 行**完整逐字**：
  ```python
  # 速度区间不同weight
  if self.vel_range_weight:
      for vel_i in self.vel_range_weight:
          vel_mask = ((v_gt_norm >= self.vel_range_weight[vel_i]['range'][0]) &
                      (v_gt_norm <  self.vel_range_weight[vel_i]['range'][1]))
          vel_mask = vel_mask.unsqueeze(1).repeat(1, 2, 1, 1)
          v_attr_mask[vel_mask] *= self.vel_range_weight[vel_i]['weight']
  ```
- 【形状】和 rot 那段完全同构：`[1,448,224]` bool → `unsqueeze(1).repeat(1,2,1,1)` → `[1,2,448,224]` → 布尔索引原地乘。
- 【为什么低速要加权（代码注释给了确切数字）】注释原文：
  > `# 把低速loss权重乘2 (0.05m/s-1.0m/s)  静止loss权重乘1.5 (< 0.05m/s)`

  也就是：
  ```
  v < 0.05 m/s        → 权重 1.5    （完全静止的车）
  0.05 ≤ v < 1.0 m/s  → 权重 2.0    （起步/蠕行/行人步速）
  v ≥ 1.0 m/s         → 权重 1.0    （正常行驶）
  ```
  这个权重设计的业务逻辑非常清楚：
  1. **样本分布严重偏斜**——路上大部分时候大部分车在动，静止/低速样本少。加权补偿。
  2. **低速区间的相对误差最致命**——把一辆蠕行的车（0.5 m/s）判成静止（0 m/s），绝对误差只有 0.5，L1 loss 几乎没有惩罚；但下游做"是否可以起步/是否可以变道"的判断时，"动 vs 静"是**离散的、后果完全不同的**两种情况。
  3. **静止误判成移动会引发幽灵刹车**——路边停的车如果被判成 1 m/s 横向移动，规控可能认为它要切入而急刹。
- 【连接】这也解释了为什么还要单独开一个 `movement` 动静二分类分支（§11-10）：**光靠回归速度的连续值不足以把"动/静"这个离散决策做对，必须显式建模成分类任务。**

**[01:45:15] "这里是……给速度的一个权重。"**　**[01:45:23] "然后这里的话就是也是用了 L1Loss 去计算速度的 Loss。"**

- 【代码】画面 01:45:05 第 935–943 行**完整逐字**（这是 `vel_range_weight` 为空时的 else 分支，把上面的注释逻辑硬编码了一遍）：
  ```python
  else:
      if self.neg_vel_loss > 0.0:
          v_attr_mask = v_attr_mask * (
              (v_gt_norm > 0.05) * (v_gt_norm < 1.0) * 1
            + (v_gt_norm < 0.05) * 0.5
            + ((v_gt_norm > 0.2) * (v_pred_norm - v_gt_norm < 0.0))
              * (abs(v_pred_norm - v_gt_norm) * self.neg_vel_loss / (v_gt_norm + 1e-6))
            + 1).float().unsqueeze(1)
      else:
          v_attr_mask = v_attr_mask * (
              (v_gt_norm > 0.05) * (v_gt_norm < 1.0) * 1
            + (v_gt_norm < 0.05) * 0.5
            + 1).float().unsqueeze(1)

  vel_dense_loss = self.loss_bbox(v_pred, v_gt, v_attr_mask,
      avg_factor=(v_attr_mask > 0).eq(1).float().sum().item()) * 5
  ```
- 【拆解 else 分支的加权公式】把 `+1` 提出来看：
  ```
  权重 = 1 + [0.05 < v < 1.0] × 1        → 低速：1+1 = 2      ✅ 对上注释"低速×2"
       + [v < 0.05]        × 0.5         → 静止：1+0.5 = 1.5  ✅ 对上注释"静止×1.5"
       + 非对称惩罚项
  ```
  **注释和代码完全对得上。** 这是本章罕见的"注释可信"案例。
- 【非对称惩罚项 `neg_vel_loss` 详解——本段最精彩的设计】
  ```python
  ((v_gt_norm > 0.2) * (v_pred_norm - v_gt_norm < 0.0))
      * (abs(v_pred_norm - v_gt_norm) * self.neg_vel_loss / (v_gt_norm + 1e-6))
  ```
  逐项读：
  - `v_gt_norm > 0.2`：只对真的在动的目标生效（0.2 m/s 以上）。
  - `v_pred_norm - v_gt_norm < 0.0`：**只在"预测速度比真值小"时生效**（欠估计）。预测偏大不加罚。
  - `abs(Δ) / v_gt_norm`：**相对误差**。把 20 m/s 估成 18（相对 10%）和把 2 m/s 估成 0（相对 100%）区别对待。
  - `× self.neg_vel_loss`：一个可调的系数。
  所以这一项的完整语义是：**"目标在动、你却把它估慢了，按相对欠估比例额外加罚。"**
- 【为什么要这么设计——业务推导】自动驾驶里速度欠估的后果远严重于高估：
  - 把对向来车 15 m/s 估成 8 m/s → 以为有足够时间左转 → **撞车**。
  - 把前车 5 m/s 估成 8 m/s → 以为跟车距离在拉开 → 顶多跟得保守一点。
  L1 loss 本身是对称的（`|pred - gt|`），网络在不确定时会倾向于预测**均值**，而速度分布长尾（大部分低速），均值偏低 → **系统性欠估**。这一项就是把这个系统性偏差用 loss 掰回来。
- 【连接】这是"**非对称/风险敏感损失**"的一个实例。类似的还有 depth 估计里"宁可估近不可估远"（撞障碍物 vs 保守刹车）。你在 DepthNet 那章如果见过类似的不对称项，就是同一个思想。⚠ 值得注意的是这个惩罚项**只加在 dense loss 上，目标级速度 loss 里没有**（§11-12 的目标级 vel 只有区间加权）。
- 【关于末尾的 `* 5`】速度 dense loss 的任务权重是 5（rot 是 2，movement 是 10）。实测 `vel_dense_loss = 7.9563`。

---

### 🔨 动手练习 ch11-8：速度权重公式复现（含非对称欠估惩罚）

```python
import torch

def vel_weight(v_gt_norm, v_pred_norm, neg_vel_loss=0.0):
    """复刻画面 01:45:05 第 936-941 行"""
    w = (v_gt_norm > 0.05) * (v_gt_norm < 1.0) * 1.0 \
      + (v_gt_norm < 0.05) * 0.5
    if neg_vel_loss > 0.0:
        w = w + ((v_gt_norm > 0.2) * (v_pred_norm - v_gt_norm < 0.0)).float() \
              * (torch.abs(v_pred_norm - v_gt_norm) * neg_vel_loss / (v_gt_norm + 1e-6))
    return (w + 1).float()

cases = [
    ("完全静止 v=0.00", 0.00, 0.00),
    ("蠕行     v=0.50", 0.50, 0.50),
    ("正常     v=5.00", 5.00, 5.00),
    ("欠估10%  v=5.00", 5.00, 4.50),
    ("欠估50%  v=5.00", 5.00, 2.50),
    ("高估50%  v=5.00", 5.00, 7.50),
    ("欠估但v小 v=0.10", 0.10, 0.00),
]
print(f"{'场景':<20}{'w(neg=0)':>10}{'w(neg=1)':>10}")
for name, g, p in cases:
    g_t, p_t = torch.tensor(g), torch.tensor(p)
    print(f"{name:<20}{vel_weight(g_t,p_t,0.0).item():>10.3f}{vel_weight(g_t,p_t,1.0).item():>10.3f}")

# ---- 完整 dense vel loss ----
B, H, W = 1, 64, 32
v_gt = torch.zeros(B, 2, H, W); v_gt[0, 0, 20:30, 10:15] = 5.0     # 一辆 5m/s 的车
v_pred = torch.zeros(B, 2, H, W); v_pred[0, 0, 20:30, 10:15] = 2.5 # 严重欠估
attr_mask = torch.zeros(B, 2, H, W); attr_mask[0, :, 20:30, 10:15] = 1.0

INVALID_VELOCITY = 50.0
v_attr_mask = attr_mask * (v_gt[:, 0] < INVALID_VELOCITY).float().unsqueeze(1)
g_n = torch.norm(v_gt, p=2, dim=1); p_n = torch.norm(v_pred, p=2, dim=1)
for neg in (0.0, 1.0):
    m = v_attr_mask * vel_weight(g_n, p_n, neg).unsqueeze(1)
    af = (m > 0).float().sum().item()
    loss = (torch.abs(v_pred - v_gt) * m).sum() / max(af, 1) * 5
    print(f"neg_vel_loss={neg}  vel_dense_loss = {loss.item():.4f}")

# ---- 哨兵值污染演示 ----
v_gt_bad = v_gt.clone(); v_gt_bad[0, 0, 40, 20] = 999.0
attr_mask2 = attr_mask.clone(); attr_mask2[0, :, 40, 20] = 1.0
raw   = (torch.abs(v_pred - v_gt_bad) * attr_mask2).sum().item()
filt_mask = attr_mask2 * (v_gt_bad[:, 0] < INVALID_VELOCITY).float().unsqueeze(1)
filt  = (torch.abs(v_pred - v_gt_bad) * filt_mask).sum().item()
print(f"不过滤哨兵值 sum={raw:.1f}   过滤后 sum={filt:.1f}   ← 差 {raw-filt:.0f}")
```
预期输出（关键行）：
```
场景                  w(neg=0)  w(neg=1)
完全静止 v=0.00           1.500     1.500
蠕行     v=0.50           2.000     2.000
正常     v=5.00           1.000     1.000
欠估10%  v=5.00           1.000     1.100
欠估50%  v=5.00           1.000     1.500
高估50%  v=5.00           1.000     1.000     ← 高估不加罚
欠估但v小 v=0.10          2.000     2.000     ← v_gt<0.2，欠估惩罚不触发
neg_vel_loss=0.0  vel_dense_loss = 6.2500
neg_vel_loss=1.0  vel_dense_loss = 9.3750   ← 欠估 50% 被额外放大了 50%
不过滤哨兵值 sum=1124.0   过滤后 sum=125.0   ← 差 999（那一个 999 的哨兵值被拦掉了）
```

### 【小结】
1. 速度权重 = `1 + [0.05<v<1]×1 + [v<0.05]×0.5 + 欠估惩罚`，代码注释"低速×2 / 静止×1.5"与公式严格对应。
2. `neg_vel_loss` 是一个**只罚欠估、按相对误差缩放**的非对称项，直接来自"漏刹车比误刹车危险"的业务判断；只出现在 dense loss，目标级没有。
3. `INVALID_VELOCITY`（讲者说 50 m/s）过滤哨兵值，dense 侧只判了 `v_gt[:,0]`（vx），目标级侧判了 `[..., -2:]`（vx,vy 都判）——⚠ dense 侧疑似笔误。

---

## Part 11-9　`rot_lidar` 与 `lidar_rot_weight`：用 lidar 前向区增强朝向（01:45:32 – 01:46:30）

**本段在讲什么**
一个很有工程味道的分支：**同一个朝向任务再开一个专用头 `rot_lidar`，只在 lidar 前向有效区域内监督**；再加一个 `lidar_rot_weight` 头去预测"这块区域的 lidar 朝向可不可信"。
输入：`preds_dict[0]['rot_lidar'] [1,2,448,224]`、`kwargs['lidar_front_mask']`、`self.close_rot_heatmap`。
输出：`rot_lidar_loss` 并入 `rot_dense_loss`；`rot_lidar_w_loss`（实测 1.6938，以 ×0.1 并进 `loss_bbox`）。

---

**[01:45:32] "然后这个的话就是优化朝向角的。"**　**[01:45:46] "就是通过单帧的……单帧融合的单帧 R…单帧的 2L 去预测的一个增强 yaw 的一个 sinθ 和 cosθ。"**

- ⚠ **本章最难辨认的一句**。校正稿留了 "Rat / 2L" 这样的碎片。结合画面，正确读法应该是：**"通过单帧的 lidar（前向区域）去预测的一个增强 yaw 的 sinθ 和 cosθ"**。
  【推断依据】画面 01:46:05 第 946–947 行白纸黑字：
  ```python
  rot_lidar = preds_dict[0]['rot_lidar']
  rot_attr_mask_2 = rot_attr_mask * kwargs['lidar_front_mask'].unsqueeze(1)
  ```
  变量名是 `rot_lidar` 和 `lidar_front_mask`，与"单帧 RL/2L"发音接近的是 "lidar"。**"单帧"这个限定我保留**（讲者说了两次），大概率指这一路用的是**当前帧的 lidar**而非时序融合后的特征——因为时序 warp 会引入运动补偿误差，对朝向这种精细量有害。
- 【直译】`rot_lidar` 是一个"lidar 增强版"的朝向预测头。
- 【为什么需要它】朝向估计的信息来源差异极大：
  - **lidar 在近处/前方**能看到车辆的 L 形轮廓（车头+车侧），朝向可以直接从点云拟合出来，精度极高（<2°）。
  - **纯视觉或远处 lidar** 只能看到一团点，朝向靠先验猜（车总是沿车道方向），误差可达 10~30°。
  强行让一个头同时学这两种情况，会被难样本拉低整体精度。**开两个头：一个（`rot`）覆盖全图学"平均水平"，一个（`rot_lidar`）只在 lidar 好使的区域学"高精度版"**，推理时按区域选用或加权融合。
- 【连接】和 §11-6 的 `close_heatmap` 是完全同构的设计模式：**同一份 GT + 不同的空间 mask + 独立的卷积头 = 分区专家**。这套代码里这个模式用了 3 次（close_heatmap、rot_lidar、conetank）。

**[01:45:57] "然后这里去算的话，也是通过 L1Loss 去计算它的 Loss。"**

- 【代码】画面 01:46:05 第 946–950 行**完整逐字**：
  ```python
  if self.rot_spec:
      rot_lidar = preds_dict[0]['rot_lidar']
      rot_attr_mask_2 = rot_attr_mask * kwargs['lidar_front_mask'].unsqueeze(1)
      rot_lidar_loss = self.loss_bbox(rot_lidar, rot_gt, rot_attr_mask_2,
          avg_factor=(rot_attr_mask_2 > 0).eq(1).float().sum().item()) * 2
      rot_dense_loss += rot_lidar_loss
  ```
- 【形状】`kwargs['lidar_front_mask']` 是 `[B,448,224]`（因为 `.unsqueeze(1)` 后才能乘 `[B,2,448,224]`），`rot_attr_mask_2` `[1,2,448,224]`。
- 【关键点 1】`rot_lidar_loss` 复用了 **`rot_gt` 同一份 GT** 和 **`rot_attr_mask` 同一套权重**（car×3、VRU×3、角度桶都继承了），只是额外乘了 `lidar_front_mask`。
- 【关键点 2】`rot_dense_loss += rot_lidar_loss` —— **它被加进了 `rot_dense_loss` 里，不单独记录**。所以表 E 里的 `loss_p_rot_dense = 30.5692` 实际上是 `rot 的 loss + rot_lidar 的 loss` 之和。这一点讲者没说，容易看漏。
- 【关键点 3】`avg_factor` 用的是 `rot_attr_mask_2`（乘过前向 mask 的），所以分子分母都限制在前向区，量级和主 rot loss 可比。✅ 这里 `avg_factor` 处理得比 §11-6 的 close_heatmap 严谨。

**[01:46:05] "然后这个也是近距离的……近距离的……近距离这些目标它的那个 yaw。"**

- 【直译】接下来还有一个和"近距离 yaw"相关的东西。
- 【代码】画面 01:46:05 第 951–965 行**完整逐字**：
  ```python
  if self.close_rot_heatmap is not None:
      rot_lidar_w = preds_dict[0]['lidar_rot_weight']          # [1,1,448,224]
      bs = rot_lidar_w.shape[0]
      rot_lidar_w_gt = torch.Tensor(
          np.tile(self.close_rot_heatmap[None, None, ...], (bs, 1, 1, 1)))
      rot_lidar_w_gt = rot_lidar_w_gt.to(rot_lidar_w.device)
      rot_lidar_w_mask = (rot_lidar_w_gt > 0).float()

      if sample_mask is not None:
          rot_lidar_w_mask *= sample_mask[..., None, None]

      rot_lidar_w_pos_num = rot_lidar_w_mask.eq(1).float().sum().item()
      rot_lidar_w_loss = self.loss_cls(rot_lidar_w, 1 - rot_lidar_w_gt,
          avg_factor=max(rot_lidar_w_pos_num, 1),
          sample_mask=rot_lidar_w_mask)
  ```
- 【⚠ 这一段讲者只说了"近距离的 yaw"就跳过了，但它其实是本章最反直觉的一段代码，我详细拆一下】
  1. `self.close_rot_heatmap` 是一张**常量图** `[448,224]`（和 `close_cls_mask` 同类），描述"哪块区域的 lidar 朝向可信"。
  2. 它被直接当成 **GT**：`rot_lidar_w_gt = tile(close_rot_heatmap)`。
  3. loss 的 target 是 **`1 - rot_lidar_w_gt`**（取补！），用的是 **`self.loss_cls`（Focal Loss）**。
  4. mask 是 `rot_lidar_w_gt > 0`，也就是**只在 close_rot_heatmap 非零的区域算 loss**。
- 【这在干什么？我的解读】`lidar_rot_weight` 这个头预测的是一张"**该像素上应该多信任 `rot_lidar` 的输出**"的权重图。训练时用一张**固定的先验图**去监督它，让网络输出逼近这个先验。至于为什么 target 是 `1 - gt`：
  - 【解读 A（我倾向）】`close_rot_heatmap` 的语义是"**lidar 朝向不可信度**"或者"距离归一化图"，取补后才是"可信度"；网络学出来的 `lidar_rot_weight` 是可信度图，推理时 `rot_final = w * rot_lidar + (1-w) * rot`。
  - 【解读 B】纯粹是符号约定笔误。但考虑到这行代码显然被调试过（loss 值 1.6938 是个正常数），我不倾向笔误说。
  ⚠ **无论哪种解读，我都不能百分之百确定**，因为 `close_rot_heatmap` 的生成代码不在画面里。这是本章我最想问导师的一个点。
- 【为什么用固定先验图当 GT，而不让网络自己学】因为"lidar 在哪好使"是一个**几何事实**（取决于 lidar 安装位置、FOV、地面遮挡），不是数据里能学出来的语义。把它烧成常量图，网络只需学会"在这张图上输出对应的值"——本质上是把一个几何先验**编译进网络权重**，这样部署时不需要额外传一张 mask 进去（对车端算子受限的场景很有用）。这是一个挺聪明的 trick。
- 【连接】表 E 里 `loss_p_lidar_rot_w = 1.6938`，在 §11-13 会以 `bbox_loss_total += rot_lidar_w_loss * 0.1` 的形式贡献 0.169 到总 `loss_bbox`。**权重只有 0.1，说明作者也认为这是个辅助任务。**

---

### 🔨 动手练习 ch11-9：`rot_lidar` 分区专家 + `lidar_rot_weight` 先验蒸馏

```python
import torch, numpy as np
torch.manual_seed(0)          # 固定随机数，保证下面的数值可复现

B, H, W = 1, 448, 224

rot_gt        = torch.zeros(B, 2, H, W); rot_gt[0, 1] = 1.0            # 全部朝向 0°(cos=1)
rot_attr_mask = torch.zeros(B, 2, H, W); rot_attr_mask[:, :, 150:200, 100:130] = 1.0

# lidar 前向有效区：车前 40m 内的一个扇形（这里用矩形近似）
lidar_front_mask = torch.zeros(B, H, W); lidar_front_mask[:, 100:220, 60:170] = 1.0

rot_pred       = torch.randn(B, 2, H, W) * 0.1 + torch.tensor([0., 1.])[None, :, None, None]
rot_lidar_pred = torch.randn(B, 2, H, W) * 0.02 + torch.tensor([0., 1.])[None, :, None, None]

def l1_masked(p, g, m, scale=2.0):
    af = (m > 0).float().sum().item()
    return (torch.abs(p - g) * m).sum() / max(af, 1) * scale

rot_attr_mask_2 = rot_attr_mask * lidar_front_mask.unsqueeze(1)
loss_rot       = l1_masked(rot_pred,       rot_gt, rot_attr_mask)
loss_rot_lidar = l1_masked(rot_lidar_pred, rot_gt, rot_attr_mask_2)
print("有效格子: 全图 %d,  lidar前向区 %d"
      % ((rot_attr_mask > 0).sum(), (rot_attr_mask_2 > 0).sum()))
print("loss_rot        = %.4f" % loss_rot)
print("loss_rot_lidar  = %.4f  ← 专家头误差更小" % loss_rot_lidar)
print("rot_dense_loss  = %.4f  ← 两者相加，不单独记录" % (loss_rot + loss_rot_lidar))

# ---- lidar_rot_weight：用常量先验图当 GT ----
close_rot_heatmap = np.zeros((H, W), dtype=np.float32)
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((yy - 171) * 0.4) ** 2 + ((xx - 112) * 0.4) ** 2)
close_rot_heatmap[d < 40] = np.clip(1 - d[d < 40] / 40, 0, 1)          # 越近越大
rot_lidar_w_gt   = torch.from_numpy(np.tile(close_rot_heatmap[None, None], (B, 1, 1, 1)))
rot_lidar_w_mask = (rot_lidar_w_gt > 0).float()
print("rot_lidar_w_gt.shape =", tuple(rot_lidar_w_gt.shape))
print("监督区域格数 =", int(rot_lidar_w_mask.sum()))
print("target 用的是 1-gt，其值域 = %.5f ~ %.5f"
      % ((1 - rot_lidar_w_gt)[rot_lidar_w_mask > 0].min().item(),
         (1 - rot_lidar_w_gt)[rot_lidar_w_mask > 0].max().item()))
pos_num = rot_lidar_w_mask.eq(1).float().sum().item()
print("rot_lidar_w_pos_num =", pos_num, " → avg_factor =", max(pos_num, 1))
```
预期输出（已 `manual_seed(0)`，可精确复现）：
```
有效格子: 全图 3000,  lidar前向区 3000
loss_rot        = 0.1620
loss_rot_lidar  = 0.0319  ← 专家头误差更小
rot_dense_loss  = 0.1940  ← 两者相加，不单独记录
rot_lidar_w_gt.shape = (1, 1, 448, 224)
监督区域格数 = 31397
target 用的是 1-gt，其值域 = 0.00000 ~ 0.99985
rot_lidar_w_pos_num = 31397.0  → avg_factor = 31397.0
```
> 第 2、3 行是这个练习的核心演示：我故意让 `rot_lidar_pred` 的噪声比 `rot_pred` 小 5 倍（`*0.02` vs `*0.1`），模拟"lidar 前向区里朝向本来就更好估"。结果 `loss_rot_lidar` 只有 `loss_rot` 的 1/5。**这正是开专家头的收益来源**：如果让一个头同时吃这两种难度的样本，它学到的会是两者的折中，两边都不最优。
> 最后一行是这个练习的重点：`rot_lidar_w_mask.eq(1)` 数的是**严格等于 1** 的格子。本例中 mask 由 `(gt > 0).float()` 生成、值域是硬 {0,1}，所以 `pos_num == 监督区域格数 == 31397`，`avg_factor` **没有**退化，这一项是安全的。
> **但这个安全性完全依赖"mask 是硬 0/1"这条隐式契约。** 你可以把 `rot_lidar_w_mask = (rot_lidar_w_gt > 0).float()` 改成 `rot_lidar_w_mask = rot_lidar_w_gt`（直接拿软值当 mask）再跑一次：`eq(1)` 立刻只数出 `d==0` 的那一个中心格（甚至 0 个），`avg_factor` 塌到 1，loss 瞬间放大三万倍。
> **`(x > 0).eq(1)` 这个冗余写法在本章出现了至少 5 次**（`_close_num_pos`、`rot_attr_mask`、`v_attr_mask`、`mov_attr_mask`、`rot_lidar_w_mask`）。它在数学上等价于 `(x > 0).sum()`，但只要有人把 `> 0` 去掉、直接对软权重图 `.eq(1)`，就会静默变成"只数权重恰好等于 1 的格子"——而带了 car×3 / VRU×3 加权的图上，这种格子可能一个都没有。§11-6 的 13126 就是同一类问题的另一个实例。

### 【小结】
1. `rot_lidar` 是"lidar 前向区专家头"：同一份 `rot_gt`、继承 `rot_attr_mask` 的全部权重、额外乘 `lidar_front_mask`，loss `×2` 后**直接 `+=` 进 `rot_dense_loss`**（不单独记录，容易看漏）。
2. `lidar_rot_weight` 头用一张**固定的几何先验图** `close_rot_heatmap` 当 GT（target 取 `1-gt`，⚠ 语义需向导师确认），本质是把几何先验编译进网络权重，部署时不必外传 mask。
3. "同一份 GT + 不同空间 mask + 独立卷积头 = 分区专家"这个模式在本章出现三次（close_heatmap / rot_lidar / conetank），是这套代码的招牌手法。

---

## Part 11-10　动静 Loss（四分类归并成二分类）与朝向分类 `dir_cls`（01:46:24 – 01:48:19）

**本段在讲什么**
两个"分类味"的 dense 任务：动静（moving / static）和朝向 180° 分类。
输入：`preds_dict[0]['movement'] [1,1,448,224]`、`attr_heats[:, 4]`（二分类）、`attr_heats[:, 6]`（四分类）、`attr_heats[:, 8]`（dir_cls）。
输出：`loss_movement_dense = 6.3973`、`loss_dir_cls = 1.5016`。
本段最重要的知识点：**GT 里有 4 类动静标签，但当前只用 `[1, 3)` 区间的两类去训一个二分类头**——这个"标签空间 ≠ 训练空间"的处理，讲者说得对但没说透，我用代码把它钉死。

---

**[01:46:24] "然后这个是动静。"**　**[01:46:30] "然后动静呢，其实我们有标注的，就是可以从有些数据是标注的一个……能够从标注里面得到。"**　**[01:46:40] "然后有些数据是我们通过 car 速度值去给的那个动静的一个区间……动静的一个 GT。"**

- ⚠ "卡" = "**car / 卡（车）**"，这里应为"**通过（目标的）速度值**"。
- 【直译】动静 GT 有两个来源：(a) 人工标注直接给（有些数据集标了"停车/行驶"）；(b) 由标注的速度值按阈值推出来。
- 【为什么两个来源要区分】因为质量不同。人工标注的动静是"语义状态"（一辆等红灯的车，人会标"临时停车"），而速度阈值推出来的是"瞬时运动"（同一辆车 v=0 → 静）。二者在"车辆刚起步/刚刹停"的临界时刻会不一致。这也解释了为什么 GT 有 **4 类**而不是 2 类——大概率是把"确定静止 / 临时停车 / 缓慢移动 / 明确移动"这类中间态也编码进去了。⚠ 4 类的确切语义画面没给，这是推断。
- 【连接】`attr_heats[:, 5]` 和 `[:, 7]` 这两个"loss 权重"通道的存在，正是为了处理来源差异——**人工标注的样本可以给高权重，速度推的给低权重**。这是把"标注置信度"直接编码进 GT 张量的做法，很干净。

**[01:46:51] "然后我们当前是把动静其实分成了几类，但是在我们真正回归的时候只回归了两类。"**　**[01:47:02] "就是在这里，就是动静值小于 3 然后大于 0，其实是……就是有 1 和 2 这两类，其实是把它分为了一类。"**

- ⚠ "把它分为了一类" 应理解为 "**把它当作一个（二分类）任务**"，不是"合并成同一类"。
- 【直译】GT 有多类，但只挑出取值在 `[1, 3)`（即 1 和 2）的那部分，训一个二分类头。
- 【代码】画面 01:47:30 第 967–978 行**完整逐字（含中文注释原文）**：
  ```python
  if self.activate_move:
      # attr_heats[task_id][:, idx] 第4维是二分类动静类别，第5维是二分类loss权重
      mov_attr_mask = ((attr_heat_masks[0]
                        * (attr_heats[task_id][:, 4] < 3.0)
                        * (attr_heats[task_id][:, 4] >= 1.0))
                       * attr_heats[task_id][:, 5]).type(torch.float32).unsqueeze(1)
      if sample_mask is not None:
          mov_attr_mask *= sample_mask[..., None, None]
      # 把动静的loss调大3倍          ← 第 973 行，注释原文（放大帧逐字核对）
      loss_movement_fnc = nn.BCEWithLogitsLoss(weight=mov_attr_mask, reduction='sum')
      valid_num = max(1.0, (mov_attr_mask > 0).eq(1).float().sum().item())
      mov_dense_loss = loss_movement_fnc(
          preds_dict[0]['movement'],
          attr_heats[task_id][:, 4, None] - 1.0) * 10 / valid_num
  ```
- 【逐行拆解——这段值得慢读】
  1. `attr_heats[:, 4]` 是动静二分类**类别值**，取值约定：
     ```
     0     → 无效/未知（不参与训练）
     1     → 静
     2     → 动
     ≥3    → 无效/其它（不参与训练）
     ```
     这就是讲者说的"**小于 3、大于 0**"。
  2. `mov_attr_mask` = `有目标 × (值∈[1,3)) × 该像素的 loss 权重(通道5)`，再 `.unsqueeze(1)` 变 `[1,1,448,224]` 对齐 `movement` 分支。
  3. **注释原文是"把动静的 loss 调大 3 倍"**（第 973 行；我把这一行单独放大到 5 倍逐字核对过，是"动静"两个字，不是"静"一个字。⚠ 我上一版把它读成"把静的 loss 调大3倍"，据此推出"静的填 3、动的填 1"——**那是过度解读，这里更正**）。
     现在能确证的只有两件事：(a) 这个 3 倍**不在这段 loss 代码里**——第 969–977 行从头到尾只有 `* 10 / valid_num`，没有任何 3；(b) 它必然**烧在 `attr_heats[:, 5]` 这个权重通道的数值里**，由 DataLoader 生成 GT 时填进去。
     至于"3 倍加在谁头上"，有两种读法，画面不足以判定：
     - 【读法 A】整个动静任务相对其他 dense 任务整体 ×3（那么通道 5 在有效格子上就是常数 3）；
     - 【读法 B】只有静止目标 ×3、移动目标 ×1（通道 5 取值 {1,3}）。
     我**略倾向 B**，理由是：如果只是整任务放大 3 倍，直接把第 977 行的 `* 10` 改成 `* 30` 就行了，根本不需要专门开一个 `[B,448,224]` 的权重通道来搬运一个常数；开逐像素权重通道，只有在"不同目标权重不同"时才有意义。而"不同目标权重不同"最合理的划分就是动/静。**但这条只是设计合理性推理，不是帧证**，已列入存疑清单。⚠ 无论哪种读法，"**在 loss 代码里找不到那个 3**"这个结论都成立——这是本段最该带走的一点。
  4. `nn.BCEWithLogitsLoss(weight=..., reduction='sum')`：注意用的是 **BCEWith**Logits——所以 `movement` 分支输出的是 **logit，没有先过 sigmoid**（对比 heatmap 是先 `clip_sigmoid` 再进 Focal）。BCEWithLogits 内部用 log-sum-exp 技巧，数值上比"sigmoid + BCE"稳定得多。
  5. **target = `attr_heats[:, 4, None] - 1.0`**：把类别值 1/2 平移成 0/1，正好是 BCE 需要的目标。`[:, 4, None]` 是 `[1,448,224]` → `[1,1,448,224]`。
     ⚠ 注意：无效位置（值 0 或 3）的 target 会变成 -1 或 2，是非法的 BCE 目标，但因为它们的 `weight = 0`，贡献被乘没了。**这是"用 weight 而不是用索引来屏蔽"的典型写法，能保持张量形状规整（对 CUDA/部署友好），代价是可读性差、且依赖 weight 精确为 0。**
  6. `* 10 / valid_num`：`reduction='sum'` 出来的是总和，除以有效元素数得平均，再乘任务权重 10。实测 `loss_movement_dense = 6.3973`。
- 【为什么动静要单独做二分类，而不是从速度阈值算】§11-8 已经埋了伏笔：L1 回归速度不足以保证"动/静"这个**离散决策**准确。分类头直接优化决策边界，配合"静止×3"的权重，能显著降低"路边停车被判成移动"的幽灵刹车率。
- 【连接】这和 YOLO 里 objectness 分支与 box 回归分支并存是同一逻辑：**离散决策用分类头，连续量用回归头，两者互补而非替代。**

**[01:47:19] "这里是取得动静的一个 Mask。"**　**[01:47:26] "然后这里算的话就是我预测的……这就是预测的动静。"**　**[01:47:42] "然后这里是 GT 的一个动静，然后去算 Loss。"**

- 【直译】此处讲者按行复述代码，把第 969–978 行的三个要素点了一遍：mask / pred / gt。内容与上一条卡重合，但正好给了我们一个把三者对齐检查一遍的机会。
- 【形状小结·三者必须严格同形，因为 `BCEWithLogitsLoss(weight=...)` 是逐元素的】
  ```
  pred   : preds_dict[0]['movement']         [1,1,448,224]  (logit，注意没过 sigmoid)
  target : attr_heats[:, 4, None] - 1.0      [1,1,448,224]  (0/1；无效位是 -1 或 2，被 weight 屏蔽)
  weight : mov_attr_mask                     [1,1,448,224]  (0 或 w；w 里含那个"3"，见上条卡)
  ```
  三者形状一致 ⇒ 逐元素相乘求和 ⇒ `reduction='sum'` 出标量 ⇒ `* 10 / valid_num`。
- 【⚠ 这里有一个真实的数值隐患，讲者没提，我推一遍】`BCEWithLogitsLoss` 的公式是
  `l = -[ t·log σ(x) + (1-t)·log(1-σ(x)) ]`。当 `t = -1`（无效位）时，它变成
  `l = -[ -log σ(x) + 2·log(1-σ(x)) ]`——**这是一个完全合法但语义荒谬的值，而且当 `σ(x) → 1` 时 `log(1-σ(x)) → -∞`**。
  也就是说：**无效位置上是可能算出 `inf` 的**。而 `inf × 0 = nan`（IEEE 754），weight 乘 0 **救不回来**！
  那为什么实际没炸？因为 `BCEWithLogitsLoss` 内部对 `log(1-σ(x))` 用了 log-sum-exp 稳定化，把它算成 `-x - log(1+e^{-x})` 这类形式，在 fp32 下 `x` 要大到 ~90 才溢出；而 `movement` 分支的 logit 正常范围是 ±10 以内。**所以这是一个"靠数值范围侥幸不触发"的隐患，不是一个安全设计。**
  ⚠ 稳健写法应该是先 `target = target.clamp(0, 1)` 再进 loss（反正无效位 weight=0，clamp 不影响有效位）。这是一个可以直接向导师提的改进点，成本一行。
- 【为什么我确定无效位的 target 真的是 -1 / 2 而不是别的】因为 target 是 `attr_heats[:, 4, None] - 1.0` 无条件减 1：通道 4 的取值域是 {0(无效), 1(静), 2(动), ≥3(其它无效)}，减 1 后就是 {-1, 0, 1, ≥2}。**代码里没有任何 clamp 或 where**（对比 §11-10 后面 `dir_cls` 那边就老老实实写了 `torch.clamp(dir_cls_gt - 1, min=0)`）。同一个文件里，`dir_cls` 做了防护而 `movement` 没做——这个不一致本身就是"movement 这里是疏忽"的旁证。

**[01:47:47] "然后这个的话是应该是之前有 XX 做的动静四分类。"**　**[01:48:01] "然后这个是当前应该没有用到。"**

- ⚠ 校正稿写"有博做的"，"博"疑为同事姓氏/称呼，无技术含义。
- 【直译】下面这块是四分类动静，当前配置没启用。
- 【代码】画面 01:47:30 第 979–992 行 + 01:48:20 第 988–992 行**完整逐字（含中文注释原文）**：
  ```python
  if self.activate_move_two_stg:
      # 动静四分类，转为onehot形式，loss调大 3 倍
      # attr_heats[task_id][:, idx] 第6维是四分类动静类别，第7维是四分类loss权重
      loss_movement_two_stg_fnc = MultiFocalLoss(4, alpha=[1, 1, 1, 1], gamma=2)
      mov_attr_two_stg_mask = ((attr_heat_masks[0]
                                * (attr_heats[task_id][:, 6] < 5.0)
                                * (attr_heats[task_id][:, 6] >= 1.0))
                               * attr_heats[task_id][:, 7]).type(torch.float32)
      if sample_mask is not None:
          mov_attr_two_stg_mask *= sample_mask[..., None, None]
      mov_two_stg_dense_loss = loss_movement_two_stg_fnc(
          preds_dict[0]['mov_two_stage'], mov_attr_gt, (mov_attr_two_stg_mask > 0))
      valid_num = max(1.0, (mov_attr_two_stg_mask > 0).eq(1).float().sum().item())
      mov_two_stg_dense_loss = torch.sum(
          torch.mul(mov_attr_two_stg_mask, mov_two_stg_dense_loss)) * 0.8 / valid_num
  ```
- 【为什么确定"当前没用到"】终端打印的 11 个 key 里**没有 `mov_two_stage`**（表 A）。检测头压根没建这个分支，所以 `activate_move_two_stg` 必然是 False。✅ 讲者判断正确，且我有独立证据。
- 【结构对照】四分类版和二分类版是完全平行的：
  | | 二分类 | 四分类 |
  |---|---|---|
  | 类别通道 | `attr_heats[:, 4]` | `attr_heats[:, 6]` |
  | 权重通道 | `attr_heats[:, 5]` | `attr_heats[:, 7]` |
  | 有效区间 | `[1, 3)` → 2 类 | `[1, 5)` → 4 类 |
  | 预测分支 | `movement [1,1,·]` | `mov_two_stage [1,4,·]`（未建） |
  | Loss | `BCEWithLogitsLoss` | `MultiFocalLoss(4, alpha=[1,1,1,1], gamma=2)` |
  | 任务权重 | `× 10` | `× 0.8` |
- 【为什么最终选了二分类】我的判断：四分类的中间态（"临时停车" vs "确定静止"）**标注一致性差**，训出来的分类边界不稳定，反而不如二分类可靠；而且下游规控真正需要的就是"这个障碍物会不会动"这一个 bit。**能用 1 bit 解决就不要用 2 bit**——这是量产模型的取舍哲学。⚠ 注意四分类的代码完整保留着（连 GT 通道 6、7 都还在生成），说明随时可以切回来做实验。

**[01:48:03] "然后这个是朝向的一个分类。"**（01:48:08 重复）　**[01:48:11] "这是预测的分类。"**　**[01:48:14] "然后从这里面去取得 GT 的分类结果，去算 Loss。"**

- 【直译】`dir_cls` 分支：朝向的 180° 二分类。
- 【代码】画面 01:48:20 第 994–1000 行**完整逐字**：
  ```python
  if self.dir_cls_task:
      dir_cls_task_idx = (6 if self.enable_corner_det else 4) + (4 if self.activate_move else 0)
      dir_cls_gt   = attr_heats[task_id][:, dir_cls_task_idx:(dir_cls_task_idx + 1)]
      dir_cls_mask = attr_mask_ori * (dir_cls_gt > 0)
      loss_dir_cls = self.loss_dir_cls_func(
          preds_dict[0]['dir_cls'], torch.clamp(dir_cls_gt - 1, min=0), dir_cls_mask)
      loss_dir_cls *= 0.5
  ```
- 【形状】当前配置 `dir_cls_task_idx = 4 + 4 = 8` → `dir_cls_gt = attr_heats[:, 8:9]` `[1,1,448,224]`；`attr_mask_ori = attr_mask[:, :1]` `[1,1,448,224]`（§11-7 第 884 行留的那一份**未加 car/VRU/角度权重**的干净 mask）。
- 【为什么 dir_cls 用 `attr_mask_ori` 而不是 `attr_mask`】因为 car×3、VRU×3、角度桶那些权重是为**朝向角回归**设计的，对"前后判别"这个分类任务没有意义。用干净版是正确的。**这也解释了第 884 行为什么要提前存一份 `attr_mask_ori`**——讲者当时没解释，这里补上。✅
- 【GT 编码】`torch.clamp(dir_cls_gt - 1, min=0)`：GT 取值 0（无效）/1/2，减 1 得 -1/0/1，`clamp(min=0)` 把 -1 拍到 0。配合 `dir_cls_mask = ... * (dir_cls_gt > 0)`，无效位置被 mask 掉，所以 clamp 只是防止非法值进 loss。和 movement 的 `-1.0` 是同一套编码约定（**1-based 类别值，0 表示无效**）。这是本章 GT 编码的统一规范，记住它。
- 【为什么需要 dir_cls】§11-5 说过：`(sin, cos)` 理论上能确定 0~360°，但实践中网络对车头/车尾的区分很弱（BEV 上一辆车的点云前后近似对称，尤其远处）。于是产生"朝向翻转 180°"的典型错误——**一辆前车被判成倒车过来**，规控会急刹。加一个专门的二分类头强制学习前后，是 SECOND / PointPillars 时代就有的经典做法（`dir_offset`、`num_dir_bins=2`）。
- 【关于 `loss_dir_cls *= 0.5`】任务权重 0.5，实测 `loss_dir_cls = 1.5016`。
- 【连接】mmdet3d 的 `SECONDHead` 里有 `loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2)` 和 `get_direction_target()`，思想完全一致。DenseBEV 把它 dense 化了（在整张图上算而不是只在 anchor 上）。

---

### 🔨 动手练习 ch11-10：动静二分类归并 + `dir_cls` 1-based 编码

```python
import torch
import torch.nn as nn

B, H, W = 1, 64, 32
ATTR_MOV2_CLS, ATTR_MOV2_W, ATTR_DIR = 4, 5, 8

attr_heats = torch.zeros(B, 9, H, W)
attr_heat_masks_0 = torch.zeros(B, H, W)

# 目标A: 静止车 (mov2=1, 权重3), 朝向类别1
attr_heats[0, ATTR_MOV2_CLS, 10:20, 5:10] = 1.0
attr_heats[0, ATTR_MOV2_W,   10:20, 5:10] = 3.0     # ← 注释"把动静的loss调大3倍"的那个3，烧在GT通道5里（此处按读法B演示）
attr_heats[0, ATTR_DIR,      10:20, 5:10] = 1.0
# 目标B: 移动车 (mov2=2, 权重1), 朝向类别2
attr_heats[0, ATTR_MOV2_CLS, 30:40, 15:20] = 2.0
attr_heats[0, ATTR_MOV2_W,   30:40, 15:20] = 1.0
attr_heats[0, ATTR_DIR,      30:40, 15:20] = 2.0
# 目标C: 动静未知 (mov2=0) —— 应被完全屏蔽
attr_heats[0, ATTR_MOV2_CLS, 50:55, 25:28] = 0.0
attr_heats[0, ATTR_MOV2_W,   50:55, 25:28] = 1.0
attr_heats[0, ATTR_DIR,      50:55, 25:28] = 0.0
for sl in [(slice(10,20),slice(5,10)), (slice(30,40),slice(15,20)), (slice(50,55),slice(25,28))]:
    attr_heat_masks_0[0][sl] = 1.0

# ---- 动静二分类（第 969-978 行）----
mov_attr_mask = ((attr_heat_masks_0
                  * (attr_heats[:, ATTR_MOV2_CLS] < 3.0)
                  * (attr_heats[:, ATTR_MOV2_CLS] >= 1.0))
                 * attr_heats[:, ATTR_MOV2_W]).type(torch.float32).unsqueeze(1)
print("mov_attr_mask 唯一值:", sorted(set(mov_attr_mask.unique().tolist())))  # [0.0,1.0,3.0]
print("  静止区权重 =", mov_attr_mask[0,0,15,7].item(),   "  (期望 3.0)")
print("  移动区权重 =", mov_attr_mask[0,0,35,17].item(),  "  (期望 1.0)")
print("  未知区权重 =", mov_attr_mask[0,0,52,26].item(),  "  (期望 0.0 → 被屏蔽)")

target = attr_heats[:, ATTR_MOV2_CLS, None] - 1.0
print("target 唯一值:", sorted(set(target.unique().tolist())))   # [-1.0, 0.0, 1.0]
print("  ↑ -1 是非法 BCE 目标，靠 weight=0 屏蔽（本章的统一手法）")

pred_logit = torch.zeros(B, 1, H, W)          # logit 0 → p=0.5
fnc = nn.BCEWithLogitsLoss(weight=mov_attr_mask, reduction='sum')
valid_num = max(1.0, (mov_attr_mask > 0).float().sum().item())
mov_dense_loss = fnc(pred_logit, target) * 10 / valid_num
print("valid_num =", valid_num, " mov_dense_loss = %.4f" % mov_dense_loss.item())

# ---- dir_cls（第 994-1000 行）----
enable_corner_det, activate_move = False, True
idx = (6 if enable_corner_det else 4) + (4 if activate_move else 0)
dir_cls_gt   = attr_heats[:, idx:idx+1]
attr_mask_ori = attr_heat_masks_0.unsqueeze(1)                 # 干净 mask，无 car/VRU 加权
dir_cls_mask = attr_mask_ori * (dir_cls_gt > 0)
dir_target   = torch.clamp(dir_cls_gt - 1, min=0)
print("dir idx =", idx, " dir_target 唯一值:", sorted(set(dir_target.unique().tolist())))
print("dir_cls_mask 有效格数 =", int((dir_cls_mask > 0).sum()),
      " (未知区被排除:", int((attr_heat_masks_0 > 0).sum()) - int((dir_cls_mask > 0).sum()), "格)")
```
预期输出：
```
mov_attr_mask 唯一值: [0.0, 1.0, 3.0]
  静止区权重 = 3.0   (期望 3.0)
  移动区权重 = 1.0   (期望 1.0)
  未知区权重 = 0.0   (期望 0.0 → 被屏蔽)
target 唯一值: [-1.0, 0.0, 1.0]
  ↑ -1 是非法 BCE 目标，靠 weight=0 屏蔽（本章的统一手法）
valid_num = 100.0  mov_dense_loss = 13.8629
dir idx = 8  dir_target 唯一值: [0.0, 1.0]
dir_cls_mask 有效格数 = 100  (未知区被排除: 15 格)
```
> 把 `13.8629` 这个数字拆开验一遍，你就把整条公式吃透了：
> - 静止区 50 格 × 权重 3 + 移动区 50 格 × 权重 1 = **200** 份 `ln2`（logit=0 ⇒ p=0.5 ⇒ BCE = ln2 = 0.6931）；
> - `reduction='sum'` ⇒ 分子 = 200 × 0.6931 = **138.63**；
> - `valid_num` 数的是 `mov_attr_mask > 0` 的**格子数**（100），**不是加权和**（200）——所以分母是 100 不是 200；
> - `138.63 × 10 / 100 = ` **13.86** ✅
> 【⚠ 这里藏着一个设计上的不一致，值得记一笔】分子被权重放大了（静止格贡献 3 份），分母却按未加权的格子数算。所以"静止目标多的帧"算出来的 `mov_dense_loss` 天然偏大——加权的效果不只是"相对重要性"，它同时改变了这一项 loss 的**绝对量级**。想让加权只影响相对重要性而不影响量级，分母应该用 `mov_attr_mask.sum()`（加权和）而不是 `(mov_attr_mask > 0).sum()`（计数）。本章所有用 `(mask>0).sum()` 当 `avg_factor` 的地方都有这个性质，rot 的 car×3 / VRU×3 也一样。
> 视频里实测 `loss_movement_dense = 6.3973`，和这里的 13.86 同一个量级（差一倍是因为真实帧的动静格子配比和权重分布不同）——都对应"logit≈0、p≈0.5"的初始状态，从侧面印证了**画面里那份 loss_dict 就是训练早期的截图**（也就是 §11-6 里 13126 的成因之一）。

### 【小结】
1. 动静 GT 是 **1-based 编码**（0=无效，1=静，2=动，四分类版 1~4），loss 里用 `-1.0` 平移成 BCE 目标，无效位置靠 `weight=0` 屏蔽而不是索引屏蔽。
2. 第 973 行注释原文是"**把动静的 loss 调大 3 倍**"（不是"把静的"），这个 3 **不在 loss 代码里**，而烧在 `attr_heats[:, 5]` 权重通道的数值里——读代码时找不到它是正常的；"3 倍加在静止目标上还是整任务上"画面不足以判定，见存疑清单。
3. `dir_cls` 用的是 `attr_mask_ori`（§11-7 第 884 行提前留的干净 mask），刻意不继承 car×3/VRU×3/角度桶权重，因为那些是为角度回归设计的；`dir_cls` 解的是 sin/cos 无法可靠区分的 180° 前后歧义。

---

## Part 11-11　目标级回归：concat 10 维 → 展平 100352 → `gather_feat` 抠 256 个 → L1（01:48:19 – 01:51:30）

**本段在讲什么**
本章的**技术高潮**。前面所有 loss 都在"图"上算，这一段把 10 个回归分支拼成一张 `[1,10,448,224]` 的图，展平成 `[1, 100352, 10]`，再用 `inds` 一次性 gather 出 256 个目标的预测值，和 `anno_box [1,256,10]` 算 L1。
输入：`preds_dict[0]` 的 reg/height/dim/rot/vel、`inds [1,256]`、`masks [1,256]`、`target_box [1,256,10]`。
输出：`loss_bbox`（10 维逐维的 loss 向量，后面拆开记录）。
**这一段是"图 → 表"转换的唯一入口，也是 CenterPoint 系检测代码里最容易看不懂的一段。我会把每个维度变换都算出具体数字。**

---

**[01:48:19] "然后相对于一些属性，它除了对于这些属性它其实是在 BEV 的一个 feature 上去算的 Loss。"**

- 【直译】承上启下：前面那些（rot / vel / movement / dir_cls）是在 BEV 图上算的，接下来这些不是。
- 【代码位置】画面 01:48:20 第 1002 行有一句代码注释正好给这段话做了标题：
  ```python
  # Regression loss for dimension, offset, height, rotation      # 1002
  ind = inds[task_id]                                            # 1003
  ```
  ⚠ 注意这条注释**已经过时**——它说要回归 dimension / offset / height / **rotation**，但按 §11-13 第 1127 行 `bbox_loss_total = loss_bbox[..., :6].sum() + ...`，**rotation 那 2 维（以及 vel 那 2 维）压根不进总 loss**。注释停留在了 `dense_attr` 还没引入的年代。这是本章第二处"改代码没改注释"的痕迹（第一处是 §11-1 里 docstring 还写着 mmdet3d 的 `LiDARInstance3DBoxes`）。
- 【为什么讲者要在这里专门停下来说一句】因为读者（包括他自己）看到这里最容易犯的错是"**以为属性都做了 dense，几何量也做了 dense**"。他这句话是在划边界：
  ```
  做 dense 的：rot(2) vel(2) movement(1) dir_cls(1) ＋ rot_lidar(2) lidar_rot_weight(1)
  不做 dense的：reg(2) height(1) dim(3)         ← 只在 256 个中心点上监督
  ```
  这个边界不是随意划的，判据是"**这个属性在目标内部是否处处相同**"（下一条卡详述）。
- 【连接·一个记忆锚】把这条边界和 §11-11 结尾的 `loss_bbox[..., :6]` 连起来看，会发现一个漂亮的闭环：**目标级张量的前 6 维（不做 dense 的那些）正好就是进总 loss 的那 6 维，后 4 维（做了 dense 的 rot/vel）正好就是被 dense 版顶替、只留日志的那 4 维。** 10 维 layout 的排列顺序不是随便定的，它让"取前 6 维"这个切片刚好等价于"取所有非 dense 监督量"。

**[01:48:33] "然后对于目标的它的 XY，然后以及长宽高，其实它是……直接把这个 feature 给……把预测的这些属性给它从 BEV 的 feature 上去抠出来，然后再和我的 GT 去算 Loss。"**　**[01:48:54] "它就不是在 BEV feature 上去算的。"**

- 【直译】几何量（中心偏移、z、长宽高）走另一条路：**先从稠密图上"抠"出目标位置的预测值，再和目标级 GT 算 loss**。
- 【为什么"抠"这个动词很准确】英文就是 `gather`。它做的事情就是：`[1, 100352, 10]` 这张大表里，按 `inds` 给的 256 个行号，把对应的 256 行抽出来。
- 【为什么几何量不能 dense】§11-2 已经论证过：`dx, dy` 是"相对本格中心的偏移"，只在中心格有定义；旁边格子的"偏移"物理上应该是不同的值（差 1 格），把中心格的值填过去是**错误监督**。
- 【判据说清楚·一句话区分哪些能 dense 哪些不能】看这个属性**是否随格子位置改变**：
  | 属性 | 在目标覆盖的不同格子上取值 | 能否 dense |
  |---|---|---|
  | `rot` (sin,cos) | 处处相同（整车一个朝向） | ✅ 能 |
  | `vel` (vx,vy) | 处处相同（刚体平移） | ✅ 能 |
  | `movement` / `dir_cls` | 处处相同（整车一个状态） | ✅ 能 |
  | `dim` (l,w,h) | 处处相同，但**大目标格子多 ⇒ 隐式权重按面积翻倍** | ⚠ 理论可、实际不划算 |
  | `height` (z) | 处处相同 | ⚠ 同上 |
  | `reg` (dx,dy) | **每个格子都不同**（偏移是相对本格的） | ❌ 不能 |
  所以严格说，"不能 dense"的只有 `reg` 那 2 维；`height`/`dim` 是**能而不为**——把它们 dense 化需要额外做面积归一化（否则一辆卡车占 200 格、一个行人占 3 格，尺寸 loss 会被卡车彻底主导），收益却不明显，于是作者干脆划到目标级去。**"reg 不能，height/dim 不划算"——这两句话的理由不一样，别混为一谈。**
- 【连接·这条判据在别的任务里也成立】BEVDet 的 dense depth 监督能做，是因为深度在每个像素上有独立真值；而"实例 ID"就不能 dense 回归（它是离散标签，只能做分割）。**判断一个量能不能稠密监督，永远先问"它在每个位置上的真值是不是良定义的"。**

**[01:48:57] "然后主要用的就是这个 Index。"**　**[01:48:59] "Index 其实就是刚刚说的，就是每一个目标它在我的 BEV feature 上的一个索引。"**　**[01:49:06] "这个索引是把 H 和 W 展平之后的一个索引。"**

- 【代码】画面 01:48:20 第 1003 行：`ind = inds[task_id]`，`[1, 256]`，dtype 必须是 **int64**（`torch.gather` 的 index 强制要求 long，传 int32 会直接报 `gather(): Expected dtype int64 for index`）。
- 【回忆·把这个数再拆一遍】§11-2 已经算过 `ind = row * 224 + col`，实测首元素 38416 = 171×224 + 112。**注意乘的是 `width=224` 不是 `height=448`**——行优先（row-major）展平。这个约定必须和 §11-11 里 `permute(0,2,3,1).view(B,-1,C)` 的展平顺序**严格一致**，否则取出来的就是别的格子。两处相隔 500 多行代码，靠的是"大家都用 PyTorch 默认的 C order"这个隐式契约。⚠ 这是移植这段代码时最容易踩的坑：如果 GT 生成端（DataLoader）改用了列优先，loss 不会报错，只会悄悄学不出东西。
- 【自检方法·你以后调这类代码可以直接用】写一行断言就能锁死这个契约：
  ```python
  # 取任意一个有效目标，验证 gather 出来的值 == 直接索引的值
  b, k = 0, 0
  assert torch.allclose(
      pred[b, k],                                             # gather 的结果
      preds_dict[0]['anno_box'][b, :, ind[b,k] // W, ind[b,k] % W])   # 手工索引
  ```
  这一行如果过了，说明 `inds` 的展平约定、`permute` 的维度顺序、`gather` 的 dim 三者全对。**比读十遍代码都可靠。**
- 【为什么讲者反复强调"展平之后的"】因为这是"图坐标"和"表坐标"之间**唯一的换算规则**。前面所有 dense loss 用的是图坐标（`[B,C,H,W]`，靠广播对齐），从这里开始全部改用表坐标（`[B,N,C]`，靠 `inds` 对齐）。两套坐标系混用是这段代码最常见的 bug 来源。
- 【连接】和 mmdet3d `centerpoint_head.py` 的 `get_targets_single` 完全一致：那边写的是 `ind[k] = y * feature_map_size[0] + x`（注意 mmdet3d 的 `feature_map_size` 是 `(W, H)` 顺序，所以 `[0]` 也是宽），配套函数同样叫 `_gather_feat`。你在 BEVFusion 里见过的就是这一套，**DenseBEV 一个字都没改**——这是判断它 fork 自 mmdet3d 的又一条证据。

**[01:49:13] "然后这里去是去计算一下有多少个有效的一个目标。"**

- 【代码】画面 01:48:46 第 1004–1007 行**逐字**：
  ```python
  if sample_mask is None:
      num = (masks[task_id] * not_cls_only).float().sum()
  else:
      num = (masks[task_id] * not_cls_only * sample_mask).float().sum()
  ```
- 【形状】`masks[task_id] [B,256]` × `not_cls_only [B,1]` × `sample_mask [B,1]` → 逐元素广播 → `[B,256]` → 求和得标量。（这一行正是钉死 `[B,1]` 形状的第二处证据：若它们是 `[B]`，`[B,256] * [B]` 在 B≠256 时必然报错。）
- 【为什么三个东西相乘】三层有效性缺一不可：槽位是真目标（`masks`）、这帧要做回归（`not_cls_only`）、这个样本 RL 有效（`sample_mask`）。
- 【它用在哪】后面 `avg_factor=(num + 1e-4)`（第 1056 行）。`+1e-4` 而不是 `max(num,1)` —— ⚠ 这是本章里**另一种**防除零写法。如果 `num=0`，`loss/1e-4` = loss × 10000，比 `max(num,1)` 危险得多。不过 `num=0` 时 `bbox_weights` 全为 0，分子也是 0，`0/1e-4 = 0`，**结果仍然安全**。算是"分子分母同时为零时的巧合安全"。

**[01:49:22] "然后这里是取了一个……去计算目标的一个有效性。"**

- 【代码】画面 01:48:46 第 1009–1011 行**逐字**：
  ```python
  mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
  isnotnan = (~torch.isnan(target_box)).float()
  mask *= isnotnan
  ```
- 【形状·关键一步】
  ```
  masks[task_id]        [1, 256]
    .unsqueeze(2)    →  [1, 256, 1]
    .expand_as(target_box) → [1, 256, 10]     ← 每个目标的 10 维共享同一个有效位
    * isnotnan       →  [1, 256, 10]          ← 再逐元素过滤 NaN
  ```
- 【为什么要展成 10 维而不是保持 `[1,256]`】因为后面还要**逐维**修改它：
  ```python
  mask[..., -2:] *= valid_velocity          # 只把速度那 2 维置零
  bbox_weights = mask * mask.new_tensor(code_weights)   # 每维不同的 code_weight
  ```
  所以 mask 必须是 `[1,256,10]` 这个粒度。**这一步是后面所有"逐维加权"的基础设施。**

**[01:49:31] "然后这里回归的时候会把我预测的 XY，然后 Z 长宽高，sinθ cosθ，然后 VXVY 这些预测值给它 concat 起来。"**

- 【代码】画面 01:48:46 第 1013–1025 行**完整逐字**：
  ```python
  # reconstruct the anno_box from multiple reg heads
  if 'vel' in preds_dict[0]:
      if self.enable_corner_det:
          preds_dict[0]['anno_box'] = torch.cat(
              (preds_dict[0]['reg'], preds_dict[0]['height'],
               preds_dict[0]['dim'], preds_dict[0]['rot_long'],
               preds_dict[0]['rot_short'], preds_dict[0]['vel']), dim=1)
      else:
          preds_dict[0]['anno_box'] = torch.cat(
              (preds_dict[0]['reg'], preds_dict[0]['height'],
               preds_dict[0]['dim'], preds_dict[0]['rot'],
               preds_dict[0]['vel']), dim=1)
      # mask invalid velocity
      valid_velocity = (target_box[..., -2:] < INVALID_VELOCITY).float()
      mask[..., -2:] *= valid_velocity
  else:
      ...
      # drop velocity col
      target_box = target_box[..., :-2]
      mask = mask[..., :-2]
  ```
- 【形状】5 张图沿通道 cat：`2 + 1 + 3 + 2 + 2 = 10` → **`[1, 10, 448, 224]`**。
  画面 01:50:45 终端里讲者自己敲过验证：
  ```
  >>> preds_dict[0]['anno_box'].shape
  torch.Size([1, 10, 448, 224])
  ```
  ✅ 和讲者 01:49:49 说的"应该也是变成了 10 维"完全对上。
- 【注意 else 分支】没有 `vel` 分支时（比如只有 lidar 无时序的配置），target 和 mask 都砍掉最后 2 列变成 8 维。**当前配置有 vel，走的是 if 分支，所以 target_box 保持 10 维**——这和终端打印的 `torch.Size([1, 256, 10])` 一致。✅
- 【速度有效性再确认】`(target_box[..., -2:] < INVALID_VELOCITY)` 这里**对 vx、vy 两维都判了**，比 §11-8 dense 侧只判 `v_gt[:, 0]` 严谨。⚠ 印证了我在 §11-8 提的"dense 侧疑似笔误"。

**[01:49:46] "concat 起来。"**　**[01:49:49] "然后这个应该也是变成了 10 维。"**　**[01:49:55] "这个就是变成了 10 乘以 448 乘 224 的。"**

- ⚠ 校正稿 "10乘以48R式的" = "**10 × 448 × 224**"。另外讲者说的是"变成了 10 维"，但严格讲此刻它还是一张**图**（`[1,10,448,224]`），只是通道数变成 10；真正变成"10 维的表"是再往下两行 `view` + `gather` 之后的事。讲者这里的"维"指的是**通道数**，不是张量的 rank，别被绕进去。
- 【形状】`[1, 10, 448, 224]`。✅ 终端实测确认——画面 01:50:45 左下角终端里讲者自己敲了 `preds_dict[0]['anno_box'].shape`，回显 `torch.Size([1, 10, 448, 224])`（同一张帧上面还能看到 `target_box.shape` → `torch.Size([1, 256, 10])`，以及一次 `NameError: name 'target_box' is not defined` 的失败尝试——说明他是先敲错断点位置、再重敲的，这是**真实调试现场**而非事后整理的截图）。
- 【⚠ 讲者用词"这些预测值给它 concat 起来"要小心理解】被 concat 的是**5 个卷积头的输出张量**，不是 5 个数。也就是说这一步之后，`preds_dict[0]['anno_box']` 是一个**新建的张量**，它和 `reg / height / dim / rot / vel` 这 5 个原张量在计算图上是**父子关系**——梯度会正常沿 `cat` 的反向（即按通道切片）回流到各自的卷积头。这一点很重要：**`anno_box` 上算出来的 loss 会同时训练 5 个头**，而 `rot` / `vel` 这两个头还额外被 dense loss 训练。同一组卷积参数吃两路梯度，这就是 §11-7 说的"双轨监督"在计算图上的真实形态。
- 【为什么要临时拼一张 10 通道图，而不是分别 gather 5 次】(1) **一次 `gather` 比五次快**，而且 `gather` 是访存密集算子，合并后只走一遍 100352 长度的表；(2) 拼成 10 维后才能和 `anno_box [1,256,10]` 这个既有的 GT 契约对齐，`code_weights` 也是长度 10 的向量，全套下游代码都按 10 维写；(3) 部署导出时算子数更少。代价是多分配一块 `[1,10,448,224]` ≈ 4 MB 的显存。
- 【连接】mmdet3d 里这一步一字不差（`preds_dict[0]['anno_box'] = torch.cat((reg, height, dim, rot, vel), dim=1)`），连变量名和注释 `# reconstruct the anno_box from multiple reg heads` 都相同。**"reconstruct"这个词很传神**：检测头为了让每个属性有独立的卷积容量而把它们拆成 5 个分支，算 loss 时又要把它们重新拼回论文里那个统一的 box 向量——拆是为了学，拼是为了算。

**[01:49:59] "然后它后面会去……会这个 gather_feat，就是通过我们取的那个……就是所需要取的索引把它取出来。"**

- 【代码】画面 01:49:46 第 1045–1047 行**逐字**（本章最关键的三行）：
  ```python
  pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
  pred = pred.view(pred.size(0), -1, pred.size(3))
  pred = self._gather_feat(pred, ind)
  ```
- 【形状·把每一步都算出来】
  ```
  起点            [1, 10, 448, 224]            NCHW
  .permute(0,2,3,1)  [1, 448, 224, 10]         NHWC   ← 把通道挪到最后
  .contiguous()      [1, 448, 224, 10]         （物理重排内存；⚠ 本例中 view 不加它也合法，见下）
  .view(1, -1, 10)   [1, 100352, 10]           448×224 = 100352
  _gather_feat(·,ind)[1, 256, 10]              ind [1,256]
  ```
- 【为什么必须先 permute 再 view】这是整段最容易写错的地方。
  - 目标是让"第 i 个格子的 10 个属性"在内存里**连续**，这样 `view` 之后第 i 行就是第 i 个格子。
  - NCHW 布局下内存顺序是 `[c][h][w]`，同一个格子的 10 个通道相隔 `448×224 = 100352` 个元素，**不连续**。直接 `view(1, -1, 10)` 会把"通道 0 的前 10 个格子"当成"格子 0 的 10 个通道"，**完全错乱且不报错**。
  - permute 到 NHWC 后**逻辑顺序**变成 `[h][w][c]`，同一格子的 10 个值在逻辑上紧挨着。`contiguous()` 把这个逻辑顺序落实成物理顺序（permute 本身只改 stride 元数据、不搬数据）。
  - ⚠ **关于 `.contiguous()`，我上一版写错了，这里更正并给出实测**。我原先写的是"漏掉 `.contiguous()` 时 PyTorch 会直接报 `view size is not compatible with input tensor's size and stride`"。**实测（PyTorch 2.6）不会报错，而且结果是对的**：
    ```python
    a = torch.zeros(1, 10, 448, 224)
    p = a.permute(0, 2, 3, 1)                 # [1,448,224,10]
    p.is_contiguous()                          # False
    p.stride()                                 # (1003520, 224, 1, 100352)
    v = p.view(1, -1, 10)                      # ✅ 不报错
    torch.equal(v, p.contiguous().view(1, -1, 10))   # ✅ True
    ```
    **原因**：`view` 只要求"被合并的那几个维度在 stride 上本来就连成一片"。这里被合并的是 H（stride=224）和 W（stride=1）——它们在原始 NCHW 布局里本来就是相邻紧凑的一块，合并成 `H*W`（stride=1）完全合法；通道维（stride=100352）没有被合并，只是挪了位置。所以 `permute(0,2,3,1).view(B,-1,C)` 对 **任何** NCHW 张量都成立，B=1/2/4 我都验过。
    换句话说，**这行 `.contiguous()` 不是正确性必需，而是防御性 + 性能选择**：写上它，后面 `gather` 沿 dim=1 取数时读的是连续内存（对 GPU 访存和 ONNX 导出更友好），也不用担心哪天有人改了 permute 顺序导致 view 真的不合法。
  - 【那真正会静默错乱的写法是什么】是**不 permute 直接 view**：
    ```python
    a.view(1, -1, 10)          # ❌ 把"通道0的前10个格子"当成"格子0的10个通道"
    ```
    这个不会报错、不会警告，取出来全是垃圾值（练习里我做了对照打印）。**这才是这段代码真正的坑**——上一版我把坑指错了地方。
  - `.reshape()` 与 `.view()` 在这里完全等价（`reshape` 在能 view 时就 view，不能时才复制），所以 `permute + reshape` / `permute + contiguous + view` / `permute + view` 三种写法在本例中结果一致。
- 【`_gather_feat` 内部长什么样】mmdet3d 的标准实现：
  ```python
  def _gather_feat(self, feat, ind, mask=None):
      dim = feat.size(2)                                # 10
      ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)   # [1,256] → [1,256,10]
      feat = feat.gather(1, ind)                        # 沿 dim=1 取 → [1,256,10]
      return feat
  ```
  `ind` 要 expand 成和 feat 同 rank，因为 `torch.gather` 要求 index 和 input 维度数一致。
- 【为什么用 gather 而不是 `pred[0, ind[0]]`】三个理由：(1) `gather` 支持 batch 维度并行，advanced indexing 需要额外造 batch index；(2) `gather` 在 ONNX/TensorRT 里有原生算子，advanced indexing 常常导不出去；(3) `gather` 的反向传播是 `scatter_add`，梯度会正确地累加回原位置。

**[01:50:11] "然后在这里也会去通过判断速度的有效性，给速度这一维做一个 Mask。"**

- 【代码】就是上面第 1027–1028 行的 `mask[..., -2:] *= valid_velocity`（帧证 01:48:46）。
- 【形状】`target_box[..., -2:]` `[1,256,2]` → 与 `INVALID_VELOCITY` 比较得 bool → `.float()` `[1,256,2]` → 乘到 `mask[..., -2:]` 的对应 2 列。
- 【为什么是"给速度这一维做 mask"而不是"整个目标屏蔽"】这是本行最值得琢磨的地方：一个目标可能**框标得很准、但速度没标**（例如单帧标注、或静止物本来就没速度字段）。如果因为速度无效就把整个目标丢掉，等于白白扔掉一份可用的 xyz/lwh/yaw 监督。所以这里做的是**逐维屏蔽**——同一个目标，前 8 维照常回归，后 2 维（vx,vy）权重清零。这正是 §11-11 里"mask 必须 `expand` 成 `[1,256,10]` 而不能停在 `[1,256]`"的直接原因：**没有这个粒度，就做不到逐维取舍。**
- 【为什么放在 concat 之后】逻辑上没有先后依赖（它只动 `mask` 和 `target_box`，不动 `pred`），放在 `if 'vel' in preds_dict[0]:` 分支里是为了和 else 分支（无 vel → 直接砍掉最后 2 列，第 1041–1043 行 `# drop velocity col`）形成对称：**有 vel 就"软屏蔽"，没 vel 就"硬砍掉"**。两条路出来后 `target_box` 的宽度不同（10 vs 8），但都和 `pred` 保持同宽。
- 【⚠ 对照】这里判的是 `target_box[..., -2:]`（vx、vy **两维都判**），而 §11-8 的 dense 侧只判了 `v_gt[:, 0]`（只看 vx）。同一个文件、同一个哨兵常量、两种严谨度——这是我判断 dense 侧是笔误的核心依据。

**[01:50:21] "然后这里做一个维度的变换。"**

- 【直译】讲者一句带过，但这一行是整章"图 → 表"的**物理转折点**，值得单独拆。
- 【代码】第 1045 行：`pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()`。
- 【形状】`[1, 10, 448, 224]` → `[1, 448, 224, 10]`。**元素总数不变（1003520 个），变的只是"哪一维排在最外层"。**
- 【为什么这一步必须在 view 之前】用一句话记：**`view` 只会按"从右往左"的顺序把元素铺开，所以你想让谁在一行里，就得先把谁挪到最右边。** 我们想要的一行 = 一个格子的 10 个属性 ⇒ 通道维必须挪到最右。permute 干的就是这件事。
- 【代价】`.contiguous()` 会真实地复制一份 `1×448×224×10 = 100 万` 个 float ≈ **4 MB**（bs=1）。这是这段代码里唯一一次显式的大块内存拷贝，bs=8 时就是 32 MB。⚠ 但它换来的是后面 `gather` 的连续访存，通常是划算的；且如上文所述，本例中它并非正确性必需，真要抠显存是可以去掉的。

**[01:50:26] "然后在这里的话就是我的这个预测的 feature 其实还是……这一步之前其实还是 448×224 的。"**　**[01:50:38] "然后在这里的话就是把 H 和 W 这一维给它展平了。"**

- 【直译】讲者在强调"**这一步之前**"和"**这一步之后**"的分界：之前 BEV 还是一张有空间结构的图（448 行 × 224 列），之后它变成了一张没有空间结构的**长表**。
- 【代码】第 1046 行：`pred = pred.view(pred.size(0), -1, pred.size(3))`。注意这里用的是 `pred.size(3)` 而不是硬编码的 `10`——**因为 corner 模式下通道数是 12**（多了 rot_short 的 2 维，见第 1015–1019 行）。写成 `size(3)` 让这一行对两种配置都成立，是个小而正确的工程习惯。
- 【形状】`[1, 448, 224, 10]` → `[1, 100352, 10]`。H、W 两维被合并成一维。
- 【为什么"丢掉空间结构"反而是对的】这一步之后卷积再也用不了了——但我们也不需要卷积了。**从这里开始，问题从"稠密预测"退化成了"查表"**：GT 已经告诉我们目标在第几行（`inds`），我们只需要把那几行取出来。空间结构的价值已经在 backbone 和检测头里被榨干了。这也是为什么整个 CenterPoint 系的目标级 loss 都长这个样子。

**[01:50:43] "然后它现在的这个维度的话就是 1 乘以 100352。"**　**[01:50:50] "其实这个就是 448 乘以 224。"**

- ⚠ 校正稿 "1乘以10万0352" / "48乘以24"，实为 **100352 = 448 × 224**。**这个数字讲者念对了，是本章的一个锚点。**
- 【形状】`[1, 100352, 10]`。
- 【验算】448 × 224 = 100352 ✅
- 【把这个数记住的理由】它是本章三个"必须能脱口而出"的数字之一：**5（类别）、256（最大目标数）、100352（BEV 格子数）**。看到 100352 就该立刻反应"这是展平后的 BEV"，看到 38416 就该立刻反应"这是 100352 里的一个下标"（38416 < 100352 ✅，这也是一个随手可做的自检）。
- 【顺带算一笔账】这张表有 `100352 × 10 = 100 万`个数，而我们只会用到其中 `256 × 10 = 2560` 个——**99.75% 的计算量在目标级 loss 里被浪费了**。为什么不心疼？因为这 100 万个数在 dense loss（§11-7~§11-10）里是被用上的：`rot`、`vel`、`movement`、`dir_cls` 全在整张图上算 loss。**目标级 loss 是"顺便"从已有的稠密预测里取几行，不是为它单独算的。** 想通这一点，你就理解了 DenseBEV 为什么敢同时上三层监督——增量成本其实很低。

**[01:50:52] "然后在这里通过索引去把对应的那个 pixel，就是有目标的那个 pixel，给它的这些 feature 预测值给它取出来。"**

- 【代码】第 1047 行：`pred = self._gather_feat(pred, ind)`。
- 【形状】`[1, 100352, 10]` + `ind [1, 256]` → `[1, 256, 10]`。
- 【直译·用一句大白话】"这张 10 万行的表里，我只要 `inds` 点名的那 256 行。"
- 【⚠ 一个容易忽略的细节：padding 槽位取到的是什么】`inds` 里只有前 N 个是真目标，其余槽位是 **0**（见 01:34:03 的 tooltip：`[[38416, 0, 0, 0, ...]]`）。所以 `gather` 会把**BEV 图左上角第 0 个格子**的预测值重复取 256−N 遍填进去。这些值是垃圾，但完全无害——因为紧接着 `bbox_weights` 里对应的行全是 0（`masks` 为 0），乘完就没了。**"取垃圾但乘 0"** 是这类定长张量代码的标准做法，比"先筛后取"更 GPU 友好、也更好导出。
- 【连接】这正好解释了 §11-2 里 `masks` 存在的必要性：`inds` 负责"取哪儿"，`masks` 负责"哪些取回来的算数"。两者必须配套使用，单看一个都会出错。

**[01:51:06] "然后这个就是变成了……我们当前是预测 256 个目标，变成 256 乘以 10。"**　**[01:51:14] "然后我们的 GT 也是 256。"**

- 【形状】`pred [1, 256, 10]`，`target_box [1, 256, 10]`。**两边同构，可以直接逐元素做 L1。** 到这一步为止，整章那条"图 → 表"的转换链条终于闭合了：
  ```
  BEV feature ─(检测头,10个卷积分支)→ 10 张图 ─(cat)→ [1,10,448,224]
              ─(permute)→ [1,448,224,10] ─(view)→ [1,100352,10]
              ─(gather by inds)→ [1,256,10]  ⟷  anno_box [1,256,10]  ← GT
  ```
  **注意 GT 那一侧从头到尾没动过**——它在 DataLoader 里就是 `[1,256,10]`，是预测值一路变形去凑它。这是刻意的：GT 的产生代价高（Python 循环），预测值的变形代价低（改 stride），所以让便宜的那一方去适配贵的那一方。
- 【⚠ 讲者口误订正：01:34:52 说的"100 个目标"】他在 §11-2 说"我们都把它保存成 100 个目标"，在这里说"我们当前是预测 256 个目标"。**以 256 为准**，三条铁证：(1) 画面 01:33:35 调试器 tooltip `torch.Size([1, 256, 10])`；(2) 画面 01:37:58 / 01:50:45 终端里 `target_box.shape` 两次回显 `[1, 256, 10]`；(3) 他自己在这里改口说了 256。
  【为什么会口误成 100】我的猜测：`max_obj_num` 这个配置项在早期版本很可能真的是 100，或者他把它和 DETR 系的 `num_queries=100` 记混了。⚠ 这类"讲者念错常数"的地方，本章还有一处（§11-13 的过滤前缀），**处理原则统一：终端/调试器画面 > 源码画面 > 口述**。
- 【256 这个数怎么定的·顺带给你一个实用判据】`max_obj_num` 的选取只有一个硬约束：**必须 ≥ 数据集里单帧最大目标数**，否则 GT 生成时会静默截断（多出来的目标直接丢掉，没有任何报错）。选大了只是浪费一点显存（`256×10` 个 float 才 10 KB，可以忽略），选小了则是**永久性的标注浪费**。所以工程上一律往大了选。mmdet3d nuScenes 默认 500、CenterNet 默认 128，DenseBEV 取 256——处在中间，说明他们统计过自己数据的单帧目标数分布。你接手这类代码时可以顺手验一下：`max([len(frame['objects']) for frame in dataset])`，如果这个数逼近 256，就该报警了。

**[01:51:24] "然后这里去通过 L1Loss 去算这 10 维的一个……10 维属性的一个 Loss。"**

- 【代码】画面 01:50:45 第 1049–1057 行**完整逐字**：
  ```python
  code_weights = self.train_cfg.get('code_weights', None)
  bbox_weights = mask * mask.new_tensor(code_weights)
  reg_mask = not_cls_only
  if sample_mask is not None:
      reg_mask *= sample_mask

  loss_bbox = self.loss_bbox(
      pred, target_box, bbox_weights, avg_factor=(num + 1e-4),
      sample_mask=reg_mask, keep_last_dim=True)
  ```
- 【`code_weights` 是什么】一个长度 10 的列表，给 10 个维度各自一个权重。mmdet3d 里 nuScenes 的典型配置是
  ```python
  code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
  #               dx   dy   z    l    w    h   sin  cos   vx   vy
  ```
  速度只给 0.2 是因为速度 GT 噪声大（nuScenes 的速度是相邻帧差分出来的）。⚠ DenseBEV 的具体数值画面没给，但机制相同。
- 【形状】`bbox_weights = mask * code_weights` → `[1,256,10] * [10]` 广播 → `[1,256,10]`。
- 【`keep_last_dim=True` 是本段的点睛之笔】普通 L1Loss 会 reduce 成标量，这里保留最后一维，**返回一个长度 10 的向量**：
  ```
  loss_bbox = [L_dx, L_dy, L_z, L_l, L_w, L_h, L_sin, L_cos, L_vx, L_vy]
  ```
  为什么要保留？两个用途，后面都会用到：
  1. **分维度记录到 TensorBoard**（§11-13 的 `name_and_dim` 循环）；
  2. **分维度替换**（§11-12 的 `loss_bbox[-4:-2] = loss_rot`）。
  这个设计非常巧妙——用一个向量同时承载"可拆解的日志"和"可局部替换的中间结果"。
- 【验证】表 E 里 `loss_p_reg_loc = 1.6824`（= L_dx + L_dy）、`loss_p_height = 0.8559`（= L_z）、`loss_p_box_size = 2.6234`（= L_l+L_w+L_h），正是这个向量按 `name_and_dim` 切段求和的结果。✅

---

### 🔨 动手练习 ch11-11：完整复现 permute → view → gather_feat（含"忘记 permute"的错误演示）

```python
import torch

B, H, W, MAX_OBJ = 1, 448, 224, 256
# 直接造 concat 后的 anno_box，在 (row=171, col=112) 写入可识别的标记值 100..109
anno_box = torch.zeros(B, 10, H, W)
anno_box[0, :, 171, 112] = torch.arange(10, dtype=torch.float32) + 100   # 100..109
anno_box[0, :, 0, 0]     = torch.arange(10, dtype=torch.float32) + 200   # 干扰项

ind = torch.zeros(B, MAX_OBJ, dtype=torch.long)
ind[0, 0] = 171 * W + 112       # 38416
print("ind[0,0] =", ind[0, 0].item())

def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)

# ---------- 正确路径 ----------
pred = anno_box.permute(0, 2, 3, 1).contiguous()
print("permute 后:", tuple(pred.shape))               # (1, 448, 224, 10)
pred = pred.view(pred.size(0), -1, pred.size(3))
print("view 后   :", tuple(pred.shape))               # (1, 100352, 10)
pred = _gather_feat(pred, ind)
print("gather 后 :", tuple(pred.shape))               # (1, 256, 10)
print("目标0 取到 :", pred[0, 0].tolist())            # 期望 [100..109]

# ---------- 真正的错误路径：忘记 permute，直接 view ----------
wrong = _gather_feat(anno_box.reshape(B, -1, 10), ind)
print("忘记permute取到:", wrong[0, 0].tolist())      # 期望全 0 —— 静默错乱，不报错！
print("→ 与正确路径一致:", torch.allclose(pred[0, 0], wrong[0, 0]))   # 期望 False

# ---------- 忘记 contiguous 会怎样？实测：什么都不会发生 ----------
p_nc = anno_box.permute(0, 2, 3, 1)
print("permute 后 is_contiguous =", p_nc.is_contiguous())   # False
print("stride =", p_nc.stride())                            # (1003520, 224, 1, 100352)
v_nc = p_nc.view(B, -1, 10)                                 # ✅ 不报错
print("不加 contiguous 也能 view，结果一致:",
      torch.equal(v_nc, p_nc.contiguous().view(B, -1, 10)))  # 期望 True
print("  ↑ 因为被合并的 H,W 在原 NCHW 里本就 stride 连续，view 合法；"
      "contiguous 在这里是防御/性能，不是正确性必需")

# ---------- 等价写法对照 ----------
r1 = _gather_feat(anno_box.permute(0,2,3,1).contiguous().view(B,-1,10), ind)
r2 = _gather_feat(anno_box.permute(0,2,3,1).reshape(B,-1,10), ind)
r3 = _gather_feat(anno_box.permute(0,2,3,1).view(B,-1,10), ind)
print("三种写法一致:", torch.equal(r1, r2) and torch.equal(r2, r3))   # 期望 True

# ---------- code_weights + keep_last_dim ----------
target_box = torch.zeros(B, MAX_OBJ, 10); target_box[0, 0] = torch.arange(10.) + 100
masks = torch.zeros(B, MAX_OBJ); masks[0, 0] = 1
# ⚠ 注意 .clone()：expand_as 出来的是 view，多个位置共享同一块内存，
#    直接对它做 *= 会报 "more than one element ... refers to a single memory location"。
#    源码第 1009-1011 行之所以能原地乘，是因为 masks 的 dtype 不是 float32，
#    那里的 .float() 触发了一次真实的类型转换（=复制），顺带把 view 变成了独立张量。
#    → 这是一条隐式依赖：哪天有人把 masks 存成 float32，第 1011 行就会当场报错。
mask = masks.unsqueeze(2).expand_as(target_box).float().clone()
mask *= (~torch.isnan(target_box)).float()
code_weights = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,0.2,0.2])
bbox_weights = mask * code_weights
num = masks.sum()
loss_vec = (torch.abs(pred - target_box) * bbox_weights).sum(dim=(0,1)) / (num + 1e-4)
print("loss_bbox 向量 (keep_last_dim=True) 长度 =", loss_vec.numel())   # 10
name_and_dim = [('reg_loc',2), ('height',1), ('box_size',3), ('rot',2), ('vel',2)]
cur = 0
for name, d in name_and_dim:
    print(f"  loss_p_{name:9s} = {loss_vec[cur:cur+d].sum().item():.4f}")
    cur += d
```
预期输出（本机 PyTorch 2.6 实跑，可复现）：
```
ind[0,0] = 38416
permute 后: (1, 448, 224, 10)
view 后   : (1, 100352, 10)
gather 后 : (1, 256, 10)
目标0 取到 : [100.0, 101.0, ..., 109.0]
忘记permute取到: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
→ 与正确路径一致: False
permute 后 is_contiguous = False
stride = (1003520, 224, 1, 100352)
不加 contiguous 也能 view，结果一致: True
三种写法一致: True
loss_bbox 向量 (keep_last_dim=True) 长度 = 10
  loss_p_reg_loc   = 0.0000
  loss_p_height    = 0.0000
  loss_p_box_size  = 0.0000
  loss_p_rot       = 0.0000
  loss_p_vel       = 0.0000
```
> 最后 5 行**全是 0 恰恰是对的**：我把 `target_box[0,0]` 也设成了 100..109，和 gather 出来的预测值完全一致 ⇒ L1 = 0。**这等于把"gather 取对了行"这件事又验证了一遍**——如果 `inds` 或 permute 有任何一点错，这里立刻会出现非零值。你可以把 `ind[0,0]` 改成 38417 再跑，5 行马上全变成非零。
>
> 这段练习的价值全在那两行对照上：
> - **`anno_box.reshape(B,-1,10)`（不 permute）取出来全是 0，而且不报错、不告警。** 它把"通道 0 的第 0~9 个格子"当成了"格子 0 的 10 个属性"。这才是 CenterPoint 系代码真正的隐形坑。
> - **少写 `.contiguous()` 什么都不会发生。** 上一版我说它会抛 `view size is not compatible...`，是错的——被合并的 H、W 在原 NCHW 布局里本来就 stride 连续，view 完全合法。记住这条，你以后 review 这类代码时才不会把注意力放错地方。

### 【小结】
1. `permute(0,2,3,1).contiguous().view(B,-1,C)` 是 NCHW→"格子表"的标准三连；**先 permute 是为了让同一格子的 10 个通道在逻辑顺序上相邻**，漏了 permute 直接 view 会**静默错乱、不报错**。⚠ 而中间那个 `.contiguous()` 实测**不是正确性必需**（H、W 本就 stride 连续，view 合法），它是防御性写法 + 让 gather 读连续内存。
2. `448×224 = 100352` → `_gather_feat(pred, ind)` → `[1,256,10]`，和 `anno_box [1,256,10]` 同构，一次 L1 搞定 10 维。
3. `keep_last_dim=True` 让 `loss_bbox` 保持成**长度 10 的向量**，既方便分维度记 TensorBoard，又为 §11-12 的"局部替换"留了口子——这是本章最优雅的一处设计。

---

## Part 11-12　目标级 rot / vel 区间加权 + 目标级动静（01:51:31 – 01:52:30）

**本段在讲什么**
把 §11-7 / §11-8 的"区间加权"思路原样搬到目标级，重新算 rot 和 vel 这两段 loss，然后**用 `loss_bbox[-4:-2] = loss_rot` 这种切片赋值把结果替换回去**。最后再做一次目标级的动静 BCE。
输入：`pred [1,256,10]`、`target_box [1,256,10]`、`mask [1,256,10]`。
输出：修正后的 `loss_bbox` 向量、`loss_movement`（实测 0.0204）。
**本段我发现了一个真实的代码 bug**，详见 `[01:51:38]` 那一条。

---

**[01:51:31] "然后在这里呢，是在这里，我会其实会重新再去计算一下，根据不同的一个朝向角区间去算一下它的那个回归 sinθ 和 cosθ 它的 Loss。"**

- 【直译】按朝向角区间加权，重算 sin/cos 这 2 维的 loss。
- 【代码】画面 01:51:50 第 1059–1077 行**完整逐字**：
  ```python
  # 朝向区间不同weight
  if self.rot_range_weight:
      pred_rot   = pred[..., -4:-2]              # [1,256,2]  = sin,cos
      target_rot = target_box[..., -4:-2]         # [1,256,2]

      rot_sine   = target_rot[..., 0]             # [1,256]
      rot_cosine = target_rot[..., 1]             # [1,256]
      rot = torch.atan(rot_sine / (rot_cosine + 1e-10))   # -pi/2~pi/2
      rot = rot * 180 / np.pi

      rot_weight = mask[..., -4:-2].clone()       # [1,256,2]

      for rot_i in self.rot_range_weight:
          rot_mask = ((torch.abs(rot_weight) >= self.rot_range_weight[rot_i]['range'][0]) &
                      (torch.abs(rot_weight) <  self.rot_range_weight[rot_i]['range'][1]))
          rot_weight[rot_mask] *= self.rot_range_weight[rot_i]['weight']

      loss_rot = self.loss_bbox(pred_rot, target_rot, rot_weight,
          avg_factor=(num + 1e-4), sample_mask=reg_mask, keep_last_dim=True)
      loss_bbox[-4:-2] = loss_rot
  ```

**[01:51:38] ⚠⚠ 【本章最重要的发现：这段代码有 bug】**

对比 §11-7 的 dense 版（画面 01:43:38 第 905–909 行）和这里的目标级版（画面 01:51:50 第 1071–1074 行）：

```python
# ---- dense 版（正确）----
rot = torch.atan(rot_sine / (rot_cosine + 1e-10)) * 180 / np.pi
rot_attr_mask = attr_mask.clone()
for rot_i in self.rot_range_weight:
    rot_mask = ((torch.abs(rot) >= ...['range'][0]) &          # ← 用 rot（角度）
                (torch.abs(rot) <  ...['range'][1]))
    rot_mask = rot_mask.unsqueeze(1).repeat(1,2,1,1)
    rot_attr_mask[rot_mask] *= ...['weight']

# ---- 目标级版（BUG）----
rot = torch.atan(rot_sine / (rot_cosine + 1e-10)) * 180 / np.pi     # ← 算出来了
rot_weight = mask[..., -4:-2].clone()
for rot_i in self.rot_range_weight:
    rot_mask = ((torch.abs(rot_weight) >= ...['range'][0]) &   # ← 用的是 rot_weight！
                (torch.abs(rot_weight) <  ...['range'][1]))
    rot_weight[rot_mask] *= ...['weight']
```

我把这张帧放大过（`/tmp/rotw.png`），**逐字确认是 `torch.abs(rot_weight)` 而不是 `torch.abs(rot)`**。

- 【这个 bug 的后果】`rot_weight` 是 `mask[..., -4:-2].clone()`，值域是 `{0, 1}`（乘过 code_weight 前的 mask）。而 `rot_range_weight[·]['range']` 是**角度区间（单位：度，比如 [0,30]、[30,60]、[60,90]）**。用 0/1 去和角度区间比：
  - 落在 `[0, 30)` 这个桶里的是**所有 mask=0 和 mask=1 的元素**（0 和 1 都 < 30）→ 全部乘上第一个桶的 weight；
  - 其余桶一个元素都匹配不上。
  净效果：**朝向的分区间加权完全失效，退化成"全体乘以第一个桶的权重"**。而且 `rot` 这个变量算出来之后**从未被使用**（dead variable）——这是编译器/linter 一眼能看出来的信号。
- 【为什么我确信这是 bug 而不是有意为之】三条独立证据：
  1. **dense 版是 `torch.abs(rot)`**，同一文件同一逻辑，两处只差一个变量名，典型的复制粘贴漏改。
  2. **紧挨着的速度版是对的**（画面 01:52:31 第 1086–1089 行）：
     ```python
     vel_mask = ((torch.norm(target_vel, p=2, dim=-1) >= ...['range'][0]) &
                 (torch.norm(target_vel, p=2, dim=-1) <  ...['range'][1]))
     ```
     速度用的是 `torch.norm(target_vel)`（真实物理量），没有用 `vel_weight`。**同一段代码里 rot 用 weight、vel 用物理量，只能是笔误。**
  3. **`rot` 变量声明后未使用**。如果是有意设计，不会留一个 4 行才算出来的死变量。
- 【它有多严重】中等偏低。因为：(a) 目标级 rot loss 在最终 `bbox_loss_total = loss_bbox[..., :6].sum() + ...` 里**根本没被用上**（只取前 6 维！见 §11-13）——所以这个 bug 的实际影响是"一个本来就不参与训练的量算错了"，仅仅让 TensorBoard 上的 `loss_p_rot` 数值失真；(b) 真正训练朝向的是 dense 版，那边是对的。
  ⚠ 但如果哪天有人关掉 `dense_attr`，`bbox_loss_total = loss_bbox.sum()` 就会把这个错误加权的 rot loss 送进训练。**这是一颗埋着的雷。**
- 【给你的行动建议】这是一个可以直接向导师提的高质量问题："`centerpoint_head.py` 第 1072 行是不是应该是 `torch.abs(rot)`？dense 版第 906 行用的是 `rot`，而且这里 `rot` 算出来没被用。" 这种问题能立刻显示你真的逐行读了代码。

**[01:51:51] "然后这里的速度也是一样的，去计算了一下速度的一个……通过给不同的一个权重去算 Loss。"**

- 【代码】画面 01:52:31 第 1080–1092 行**完整逐字**：
  ```python
  # 速度区间不同weight
  if self.vel_range_weight:
      pred_vel   = pred[..., -2:]            # [1,256,2]
      target_vel = target_box[..., -2:]      # [1,256,2]
      vel_weight = mask[..., -2:].clone()    # [1,256,2]

      for vel_i in self.vel_range_weight:
          vel_mask = ((torch.norm(target_vel, p=2, dim=-1) >= self.vel_range_weight[vel_i]['range'][0]) &
                      (torch.norm(target_vel, p=2, dim=-1) <  self.vel_range_weight[vel_i]['range'][1]))
          vel_weight[vel_mask] *= self.vel_range_weight[vel_i]['weight']

      loss_vel = self.loss_bbox(pred_vel, target_vel, vel_weight,
          avg_factor=(num + 1e-4), sample_mask=reg_mask, keep_last_dim=True)
      loss_bbox[-2:] = loss_vel
  ```
- 【形状】`torch.norm(target_vel, p=2, dim=-1)` → `[1,256]`（对最后一维 reduce）；`vel_weight` 是 `[1,256,2]`。**注意这里 `vel_mask [1,256]` 直接去索引 `vel_weight [1,256,2]`**——PyTorch 的布尔索引会把 `[1,256]` 的 True 位置映射成 `vel_weight` 的前两维，选出 `[N, 2]` 的子张量，`*=` 正好把 vx、vy 两维一起加权。✅ 写法比 dense 版（要 `unsqueeze(1).repeat(1,2,1,1)`）更简洁，因为目标级张量把通道放在了最后一维。
- 【`loss_bbox[-2:] = loss_vel` 这个"局部替换"】这就是 §11-11 里 `keep_last_dim=True` 埋的伏笔。整个流程是：
  ```
  第一次算：loss_bbox = [L_dx,L_dy,L_z,L_l,L_w,L_h,  L_sin,L_cos,  L_vx,L_vy]   (无区间加权)
  替换 rot：loss_bbox[-4:-2] = loss_rot                 ← 换成带角度权重的版本
  替换 vel：loss_bbox[-2:]   = loss_vel                 ← 换成带速度权重的版本
  ```
  ⚠ **代价是 sin/cos 和 vx/vy 这 4 维被算了两遍**（第一次白算）。轻微浪费，但换来了代码的可读性。

**[01:52:01] "然后 Active Movement。"**　**[01:52:05] "然后在这里的话，也是目标级别上的动静的一个 Loss。"**　**[01:52:13] "就也不是在那个 featureMap 上去的。"**

- 【直译】进入 `if self.activate_move:` 的目标级分支。
- 【代码】画面 01:52:31 第 1094–1105 行**完整逐字**：
  ```python
  if self.activate_move:
      # TODO: T3P, compute loss of movement
      pred_mov = preds_dict[0]['movement'].permute(0, 2, 3, 1).contiguous()
      pred_mov = pred_mov.view(pred_mov.size(0), -1, pred_mov.size(3))
      pred_mov = self._gather_feat(pred_mov, ind)
      mov_weight = mask[..., -1, None] * movement_weight[task_id] * not_cls_only[..., None]
      if sample_mask is not None:
          mov_weight = mov_weight * sample_mask[..., None]
      loss_movement_fnc = nn.BCEWithLogitsLoss(weight=mov_weight, reduction='sum')
      valid_num = max(1.0, (mov_weight > 0).float().sum().item())
      loss_movement = loss_movement_fnc(pred_mov, movement[task_id]) / valid_num
      loss_dict[f'task{task_id}.loss_movement'] = loss_movement
  ```

**[01:52:18] "也在这里也是通过 gather_feat 以及这个 Index，把我的 BEV feature 上的那个动静预测值通过 SEN 给它取出来，然后去算 Loss。"**

- ⚠ "通过 SEN" 疑为 "**通过（这个）索引**" 或 "通过 `_gather_feat`" 的口误/幻听。
- 【形状·完整链路】
  ```
  preds_dict[0]['movement']         [1, 1, 448, 224]
    .permute(0,2,3,1).contiguous()  [1, 448, 224, 1]
    .view(1, -1, 1)                 [1, 100352, 1]
    _gather_feat(·, ind)            [1, 256, 1]
  movement[task_id]                 [1, 256, 1]     ← GT
  mov_weight                        [1, 256, 1]
  ```
  **和 §11-11 的 10 维版本是完全一样的三连**，只是通道数从 10 变成 1。你只要把 §11-11 那段吃透，这里一眼就懂。
- 【`mask[..., -1, None]` 这个写法】`mask` 是 `[1,256,10]`，`[..., -1]` 取最后一维（vy 那一维）得 `[1,256]`，`None` 补回来得 `[1,256,1]`。⚠ 为什么用最后一维（vy）而不是第 0 维？因为 `mask[..., -2:]` 已经被 `valid_velocity` 修改过——**用 `-1` 意味着"只有速度有效的目标才算动静 loss"**。这可能是有意的（速度无效 → 动静 GT 也不可信），也可能是随手写的。⚠ 存疑，倾向于有意。
- 【为什么 dense 和目标级两套动静 loss 都要】表 E 里 `loss_movement = 0.0204`（目标级）而 `loss_movement_dense = 6.3973`（dense）。差 300 倍，因为：dense 版是 `× 10` 且覆盖几千个像素；目标级版只有几个目标、无额外任务权重。目标级这一项**权重极小，更像是一个"点位校准"**，主要学习靠 dense 那一路。
- 【`# TODO: T3P, compute loss of movement`】代码里留了一个 TODO，说明这块还在演进中。⚠ T3P 可能是某个内部项目代号。

---

### 🔨 动手练习 ch11-12：复现 rot 加权 bug + `loss_bbox` 局部替换

```python
import torch, numpy as np

B, MAX_OBJ = 1, 256
# 4 个目标，朝向分别 5° / 45° / 85° / 5°
angles = [5., 45., 85., 5.]
target_box = torch.zeros(B, MAX_OBJ, 10)
mask_raw   = torch.zeros(B, MAX_OBJ)
for i, a in enumerate(angles):
    th = np.deg2rad(a)
    target_box[0, i, 6] = np.sin(th)     # sin
    target_box[0, i, 7] = np.cos(th)     # cos
    target_box[0, i, 8] = 3.0            # vx
    mask_raw[0, i] = 1
mask = mask_raw.unsqueeze(2).expand_as(target_box).clone().float()

rot_range_weight = {'a': dict(range=[0, 30],   weight=1.0),
                    'b': dict(range=[30, 60],  weight=2.0),
                    'c': dict(range=[60, 90.1],weight=5.0)}

target_rot = target_box[..., -4:-2]
rot = torch.atan(target_rot[..., 0] / (target_rot[..., 1] + 1e-10)) * 180 / np.pi
print("解算出的角度:", [round(v,1) for v in rot[0,:4].tolist()])   # [5.0, 45.0, 85.0, 5.0]

def apply_weight(use_bug: bool):
    w = mask[..., -4:-2].clone()
    for k in rot_range_weight:
        src = w if use_bug else rot.unsqueeze(-1).expand_as(w)   # ← bug 用 w，正确用 rot
        m = (src.abs() >= rot_range_weight[k]['range'][0]) & \
            (src.abs() <  rot_range_weight[k]['range'][1])
        w[m] *= rot_range_weight[k]['weight']
    return w

w_bug = apply_weight(True)
w_ok  = apply_weight(False)
print("BUG 版 前4个目标的 rot 权重:", w_bug[0,:4,0].tolist())
print("正确版 前4个目标的 rot 权重:", w_ok[0,:4,0].tolist())
print("→ BUG 版把所有目标都塞进了 [0,30) 这个桶（因为 mask 值 0/1 都 <30）")

# ---- loss_bbox 局部替换 ----
pred = torch.zeros(B, MAX_OBJ, 10)
code_weights = torch.tensor([1.]*8 + [0.2, 0.2])
bbox_weights = mask * code_weights
num = mask_raw.sum()

def l1_keep_last(p, g, w):
    return (torch.abs(p - g) * w).sum(dim=(0, 1)) / (num + 1e-4)

loss_bbox = l1_keep_last(pred, target_box, bbox_weights)        # [10]
print("替换前 loss_bbox =", [round(v,3) for v in loss_bbox.tolist()])
loss_rot = l1_keep_last(pred[..., -4:-2], target_rot, w_ok)     # [2]
loss_bbox[-4:-2] = loss_rot
print("替换后 loss_bbox =", [round(v,3) for v in loss_bbox.tolist()])
print("注意: bbox_loss_total 只取 loss_bbox[:6]，所以 rot/vel 这 4 维只进 TensorBoard")
print("  loss_bbox[:6].sum() =", round(loss_bbox[:6].sum().item(), 4))
```
预期输出（关键行）：
```
解算出的角度: [5.0, 45.0, 85.0, 5.0]
BUG 版 前4个目标的 rot 权重: [1.0, 1.0, 1.0, 1.0]
正确版 前4个目标的 rot 权重: [1.0, 2.0, 5.0, 1.0]
→ BUG 版把所有目标都塞进了 [0,30) 这个桶（因为 mask 值 0/1 都 <30）
```

### 【小结】
1. ⚠ **画面 01:51:50 第 1072 行 `torch.abs(rot_weight)` 应为 `torch.abs(rot)`**——dense 版对、目标级速度版对、只有目标级朝向版错，且 `rot` 变量算完未用，三条证据指向复制粘贴笔误。影响有限（该维不进总 loss），但关掉 `dense_attr` 后会变成真雷。
2. `loss_bbox[-4:-2] = loss_rot` / `loss_bbox[-2:] = loss_vel` 是"先粗算再局部替换"的写法，代价是这 4 维被算两遍，收益是代码可读。
3. 目标级动静走的是和 §11-11 一模一样的 `permute→view→gather_feat` 三连，只是通道数 1；权重取 `mask[..., -1]`（即受 `valid_velocity` 影响的那一维），实测 loss 只有 0.0204，主要学习靠 dense 那一路。

---

## Part 11-13　`loss_dict` 汇总与 `.loss_p_` 前缀的真相（01:52:31 – 01:53:28）

**本段在讲什么**
把前面算出来的十几个 loss 装进 `loss_dict`，其中一部分只用于 TensorBoard 展示、不进反向传播。
输入：前面所有 loss 变量。
输出：`loss_dict`（15 项）和外层 `get_loss()` 求和出的标量 `loss`。
本段我做了两件事：**逐字读出了那个过滤字符串（不是讲者说的 `loss_`，而是 `.loss_p_`）**，以及**用表 E 的数值把汇总公式反推验证了一遍**。

---

**[01:52:31] "然后后续这里的话，其实就是把前面算的 Loss 给它放到 dict 里面去。"**

- 【代码】画面 01:52:09 第 1112–1121 行 + 01:52:45 第 1118–1149 行，我把两帧拼起来还原完整段落：
  ```python
  # for tensorboard log
  name_and_dim = [('reg_loc', 2), ('height', 1), ('box_size', 3), ('rot', 2)]
  if 'vel' in preds_dict[0]:
      name_and_dim.append(('vel', 2))
  cur_dim = 0
  for name, dim in name_and_dim:
      end_dim = cur_dim + dim
      loss_dict[f'task{task_id}.loss_p_{name}'] = loss_bbox[cur_dim:end_dim].sum()
      cur_dim = end_dim

  if self.dense_attr:
      loss_dict[f'task{task_id}.loss_p_vel_dense'] = vel_dense_loss
      loss_dict[f'task{task_id}.loss_p_rot_dense'] = rot_dense_loss

      bbox_loss_total = loss_bbox[..., :6].sum() + vel_dense_loss + rot_dense_loss
      if self.close_rot_heatmap is not None:
          loss_dict[f'task{task_id}.loss_p_lidar_rot_w'] = rot_lidar_w_loss
          bbox_loss_total += rot_lidar_w_loss * 0.1

      if self.activate_move:
          loss_dict[f'task{task_id}.loss_movement_dense'] = mov_dense_loss
          if self.activate_move_two_stg:
              loss_dict[f'task{task_id}.loss_movement_two_stg_dense'] = mov_two_stg_dense_loss

      if self.dir_cls_task:
          loss_dict[f'task{task_id}.loss_dir_cls'] = loss_dir_cls
  else:
      bbox_loss_total = loss_bbox.sum()

  loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
  loss_dict[f'task{task_id}.loss_bbox']    = bbox_loss_total
  if self.close_cls_mask is not None:
      loss_dict[f'task{task_id}.loss_close_heatmap'] = loss_close_heatmap

  if sample_mask is not None:
      loss_dict[f'task{task_id}.loss_p_rl_percentage'] = \
          sample_mask.sum().item() / sample_mask.shape[0]

  return loss_dict, inds, record_valid_obj_indexes, masks
  ```
- 【`name_and_dim` 这个循环在干嘛】把长度 10 的 `loss_bbox` 向量按段求和，记成 5 个可读的名字：
  ```
  loss_bbox[0:2].sum() → loss_p_reg_loc    = 1.6824
  loss_bbox[2:3].sum() → loss_p_height     = 0.8559
  loss_bbox[3:6].sum() → loss_p_box_size   = 2.6234
  loss_bbox[6:8].sum() → loss_p_rot        = 134.1376
  loss_bbox[8:10].sum()→ loss_p_vel        = 3.8619
  ```
  和表 E 的前 6 项完全对上。✅
- 【⚠ `loss_p_rot = 134.14` 为什么这么大】它是目标级 rot loss，且经过了 §11-12 那个有 bug 的 `rot_weight` 加权。134 这个量级说明 `rot_range_weight` 的第一个桶权重不小（bug 让所有元素都吃了第一个桶的权重）。**幸好它不进总 loss。**

**[01:52:38] "然后现在 Loss……Loss 里面就是动静的 Loss、XY 的 Loss、高的 Loss、然后一些分类的 Loss、Heading，然后这里有 Loss。"**

- 【直译】讲者在念调试器里 `loss_dict` 的条目。对照表 E：
  - "动静的 Loss" → `loss_movement` / `loss_movement_dense`
  - "XY 的 Loss" → `loss_p_reg_loc`
  - "高的 Loss" → `loss_p_height`
  - "一些分类的 Loss" → `loss_heatmap` / `loss_close_heatmap` / `loss_dir_cls`
  - "Heading" → `loss_p_rot` / `loss_p_rot_dense`
- 【画面证据】01:52:52 的调试器 tooltip 把 15 项连值带梯度函数全列出来了（见表 E），这是本章信息密度最高的一张截图。

**[01:53:05] "带有 loss_ 前缀项的，其实它就是一些……其实主要是为了在 TensorBoard 里面显示，就是一些详细的一个 Loss。"**　**[01:53:14] "所以说真正算 Loss 的时候，是把这些给排除掉的。"**

- 【直译】带某个前缀的项只用于日志，不参与反向传播。
- 【⚠ 讲者说的"loss_ 前缀"不准确，真实字符串是 `.loss_p_`】画面 01:53:10 第 1548 行，我把它放大过（`/tmp/losssum.png`）逐字确认：
  ```python
  loss = sum([v for k, v in loss_dict.items() if '.loss_p_' not in k])
  return loss, loss_dict, inds, record_valid_obj_indexes, masks
  ```
  - 如果真按 `loss_` 过滤，那所有 key 都含 `loss_`（`task0.loss_heatmap` 也含），**总 loss 会是 0**。所以必然是更长的模式。
  - `.loss_p_` 里的 `p` 我推断是 **"print" / "plot"**（只打印用）。⚠ 无直接证据，但从功能看合理。
  - 前面还有个 `.`，是为了避免误伤形如 `xxxloss_p_yyy` 的 key（防御性写法）。
- 【验证这个过滤规则】拿表 E 逐项过一遍：

  | key | 含 `.loss_p_`? | 计入总 loss |
  |---|---|---|
  | `task0.loss_movement` | ✗ | ✅ 0.0204 |
  | `task0.loss_p_reg_loc` | ✓ | ❌ |
  | `task0.loss_p_height` | ✓ | ❌ |
  | `task0.loss_p_box_size` | ✓ | ❌ |
  | `task0.loss_p_rot` | ✓ | ❌ |
  | `task0.loss_p_vel` | ✓ | ❌ |
  | `task0.loss_p_vel_dense` | ✓ | ❌ |
  | `task0.loss_p_rot_dense` | ✓ | ❌ |
  | `task0.loss_p_lidar_rot_w` | ✓ | ❌ |
  | `task0.loss_movement_dense` | ✗ | ✅ 6.3973 |
  | `task0.loss_dir_cls` | ✗ | ✅ 1.5016 |
  | `task0.loss_heatmap` | ✗ | ✅ 5.0235 |
  | `task0.loss_bbox` | ✗ | ✅ 43.8560 |
  | `task0.loss_close_heatmap` | ✗ | ✅ 13126.7217 |
  | `task0.loss_p_rl_percentage` | ✓ | ❌（还好，它是无梯度的 python float，进了会报错） |

  **总 loss = 0.0204 + 6.3973 + 1.5016 + 5.0235 + 43.8560 + 13126.7217 = 13183.52**
- 【这个设计的精妙之处】`loss_p_reg_loc / height / box_size` 这三项**加起来正好等于** `loss_bbox[..., :6].sum()`，也就是 `loss_bbox` 的一部分。如果不过滤，**同一份 loss 会被算两次**，梯度直接翻倍。`.loss_p_` 前缀就是用命名约定来防止这个错误——比手工维护一个"哪些要加"的列表可靠得多。
- 【⚠ 但也埋了一个坑】任何人新加一个 loss 项时，如果忘了加 `.loss_p_` 前缀，它会**静默地进入总 loss**；反之如果误加了前缀，它会**静默地不参与训练**。两种错误都不会报错。这种"靠字符串约定的开关"是很脆弱的设计。更稳的做法是 `loss_dict` 和 `log_dict` 分开两个字典。
- 【连接】mmdet 的标准做法是 `parse_losses()` 里 "所有 key 含 'loss' 的都求和"，DenseBEV 反过来用"含特定标记的都排除"。两种约定都常见，关键是团队内统一。

**[01:53:28] "然后是这里 Loss 计算完了。"**

- 【直译】本章结束。
- 【外层还有什么】画面 01:53:10 显示这个 `loss()` 之上还有 `CenterPointHead.get_loss()`（第 1515 行），里面除了求和还做了一件事——**corner head 的 loss 以 0.5 权重合并进来**：
  ```python
  if self.enable_corner_det and 'obj_label_corner' in self.ret_dict:
      loss_dict_corner, _, _, _ = self.corner_head.loss(
          self.ret_dict['obj_dense_label_corner'], ...,
          sample_mask=self.ret_dict.get('rl_sample_mask'),
          lidar_front_mask=self.ret_dict.get('lidar_front_mask'),
          valid_lidar=self.ret_dict.get('valid_lidar'),
          is_cls_only=self.ret_dict.get('is_cls_only'),
          obj_dir_cls_label=self.ret_dict.get('obj_dense_dir_cls_label_corner'),
          is_drop_lidar=self.ret_dict.get('is_drop_lidar', None),
          rl_crop_index=self.ret_dict.get('rl_crop_index', None),
          centerpoint_head_gt=self.ret_dict.get('centerpoint_head_gt_corner', None),
          obj_label_corner=self.ret_dict.get('obj_label_corner', None),)
      for k, v in loss_dict_corner.items():
          loss_dict[k + '_corner'] = v * 0.5
  ```
  这段把本章讲的整个 `loss()` **又跑了一遍**（用 corner 版的 GT 和 pred），结果乘 0.5 加进来。当前配置 `enable_corner_det=False` 所以没跑，但这解释了为什么 `loss()` 里到处都是 `if self.enable_corner_det` 分支。
  ⚠ 这里还能看到几个本章没出现的 kwargs：`is_drop_lidar`（lidar dropout 数据增强的标记）、`rl_crop_index`（RL 特征裁剪索引）、`centerpoint_head_gt`。这些是 Ch10/Ch12 或视频未覆盖部分的内容。
- 【再往上】画面 01:39:02 显示的 `solver.py` 第 190–218 行：
  ```python
  def loss_backward(self, loss, task_id, task_pg, lock=None):
      loss, extra_back = self.weighter.get_weighted_loss(all_task_losses,
          reweighting=self.check_reweighting(), shared_parameters=..., ...)
      if self.amp_type == 1:
          self.grad_scaler.scale(loss).backward()
      elif self.amp_type == 2:
          with apex_amp.scale_loss(loss, self.optimizer) as scaled_loss:
              scaled_loss.backward()
      else:
          loss.backward()
      ...
      shared_grads = self.inspector.all_gather_grads(self.task_ranks, self.weighter.grad_reduction, task_pg)
      cur_grad_norm, grad_norms, projected_grad_norms, fused_grad_norm, extra_post = \
          self.weighter.post_backward(all_task_losses, reweighting=..., ...)
  ```
  **`projected_grad_norms` 这个名字基本坐实了外层用的是梯度投影类的多任务方法（PCGrad / GradVac / CAGrad 一族）**：当两个任务的梯度夹角 > 90°（互相冲突）时，把一个投影到另一个的法平面上再相加。这对 DenseBEV 这种"一个 backbone 喂检测/占用/车道线"的多任务模型很关键。
  ⚠ 这段不在本章范围内，我只标出它的存在，让你知道**本章算出来的 loss 并不是直接 `.backward()` 的，中间还隔着一层任务加权和梯度手术**。

---

### 🔨 动手练习 ch11-13：复现 `loss_dict` 汇总 + `.loss_p_` 过滤 + 数值核对

```python
import torch

# 完全照抄画面 01:52:52 调试器里的 15 项
loss_dict = {
    'task0.loss_movement':          torch.tensor(0.0204),
    'task0.loss_p_reg_loc':         torch.tensor(1.6824),
    'task0.loss_p_height':          torch.tensor(0.8559),
    'task0.loss_p_box_size':        torch.tensor(2.6234),
    'task0.loss_p_rot':             torch.tensor(134.1376),
    'task0.loss_p_vel':             torch.tensor(3.8619),
    'task0.loss_p_vel_dense':       torch.tensor(7.9563),
    'task0.loss_p_rot_dense':       torch.tensor(30.5692),
    'task0.loss_p_lidar_rot_w':     torch.tensor(1.6938),
    'task0.loss_movement_dense':    torch.tensor(6.3973),
    'task0.loss_dir_cls':           torch.tensor(1.5016),
    'task0.loss_heatmap':           torch.tensor(5.0235),
    'task0.loss_bbox':              torch.tensor(43.8560),
    'task0.loss_close_heatmap':     torch.tensor(13126.7217),
    'task0.loss_p_rl_percentage':   1.0,
}
print("len(loss_dict) =", len(loss_dict))                      # 期望 15

# ---- 1) 反推 bbox_loss_total（画面 01:52:45 第 1127/1131 行）----
d = {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
recon = (d['task0.loss_p_reg_loc'] + d['task0.loss_p_height'] + d['task0.loss_p_box_size']
         + d['task0.loss_p_vel_dense'] + d['task0.loss_p_rot_dense']
         + d['task0.loss_p_lidar_rot_w'] * 0.1)
print(f"重建的 bbox_loss_total = {recon:.4f}   实际 = {d['task0.loss_bbox']:.4f}   "
      f"误差 = {abs(recon - d['task0.loss_bbox']):.4f}")     # 误差应 < 0.001

# ---- 2) 复刻 get_loss 的过滤规则（画面 01:53:10 第 1548 行）----
total = sum([v for k, v in loss_dict.items() if '.loss_p_' not in k])
print(f"总 loss = {float(total):.4f}")
print("参与反传的项:")
for k, v in loss_dict.items():
    if '.loss_p_' not in k:
        val = v.item() if torch.is_tensor(v) else v
        print(f"   {k:32s} {val:12.4f}   占比 {100*val/float(total):5.2f}%")

# ---- 3) 如果写成讲者说的 'loss_' 会怎样 ----
wrong = [k for k in loss_dict if 'loss_' not in k]
print(f"\n若按 'loss_' 过滤，剩下 {len(wrong)} 项 → 总 loss = 0，训练直接死掉")

# ---- 4) 如果忘了过滤（重复计入）----
naive = sum([v for k, v in loss_dict.items() if torch.is_tensor(v)])
print(f"不过滤的话 = {float(naive):.4f}，其中 reg_loc/height/box_size 被重复计了一遍")
```
预期输出：
```
len(loss_dict) = 15
重建的 bbox_loss_total = 43.8566   实际 = 43.8560   误差 = 0.0006
总 loss = 13183.5205
参与反传的项:
   task0.loss_movement                    0.0204   占比  0.00%
   task0.loss_movement_dense              6.3973   占比  0.05%
   task0.loss_dir_cls                     1.5016   占比  0.01%
   task0.loss_heatmap                     5.0235   占比  0.04%
   task0.loss_bbox                       43.8560   占比  0.33%
   task0.loss_close_heatmap           13126.7217   占比 99.57%

若按 'loss_' 过滤，剩下 0 项 → 总 loss = 0，训练直接死掉
不过滤的话 = 13366.9014，其中 reg_loc/height/box_size 被重复计了一遍
```
> 这个练习一跑，你就同时验证了：(a) `bbox_loss_total` 的重建公式；(b) 过滤字符串必须是 `.loss_p_` 而不是 `loss_`；(c) close_heatmap 在这一帧占了 99.57%。三条都是硬结论。

### 【小结】
1. 过滤字符串是 **`.loss_p_`** 不是讲者说的 `loss_`（按 `loss_` 过滤总 loss 会是 0）；`p` 我推断是 print/plot。
2. 用表 E 的 15 个真实数值反推出 `bbox_loss_total = loss_bbox[:6].sum() + vel_dense + rot_dense + lidar_rot_w*0.1`，误差 0.0006，随后在下一张帧的源码里得到逐字证实。
3. 这个 `loss()` 上面还有两层：`CenterPointHead.get_loss()`（corner head 结果 ×0.5 合并）和 `solver.loss_backward()`（`weighter.get_weighted_loss` + `projected_grad_norms` 的多任务梯度手术 + AMP scaler）。本章算的 loss 不是直接 backward 的。

---

## 11-14　全章总账：DenseBEV 的 Loss 到底由哪些部分构成

把本章 385 句话压缩成一张图：

```
                       ┌──────────────────────────────────────────────┐
  obj_label            │            CenterHead.loss()                 │
  obj_state    ─cat─►  │  ① get_targets → 9 个 GT 产物                 │
  obj_dir_cls_label    │     heatmaps[1,5,448,224]  anno_boxes[1,256,10]│
                       │     inds[1,256]  masks[1,256]  record[1,256]   │
                       │     attr_heats[1,9,448,224]  attr_heat_masks   │
                       │     movement[1,256,1]  movement_weight         │
                       └──────────────────────────────────────────────┘
                                          │
        ┌─────────────────┬───────────────┼───────────────┬────────────────┐
        ▼                 ▼               ▼               ▼                ▼
  【A 分类·全图】    【B 分类·近距】  【C dense属性】  【D 目标级几何】  【E 目标级动静】
  clip_sigmoid       clip_sigmoid     rot / vel /      permute→view      permute→view
  Focal(num_pos)     Focal(close_pos) movement /       →gather_feat      →gather_feat
                     ×close_cls_mask  dir_cls          [1,256,10]        [1,256,1]
                                      + rot_lidar      L1 + code_w       BCEWithLogits
                                      + lidar_rot_w    keep_last_dim
        │                 │               │               │                │
   loss_heatmap    loss_close_heatmap  各 dense loss   loss_bbox[10维]  loss_movement
     =5.0235         =13126.7217           ↓               ↓              =0.0204
                                    ┌──────┴───────────────┴───────┐
                                    │ bbox_loss_total =            │
                                    │   loss_bbox[:6].sum()        │  =43.8560
                                    │ + vel_dense(7.9563)          │
                                    │ + rot_dense(30.5692)         │
                                    │ + lidar_rot_w(1.6938)*0.1    │
                                    └──────────────────────────────┘
                                                   │
                        loss = sum(v for k,v in loss_dict if '.loss_p_' not in k)
                             = 13183.52
                                                   │
                        CenterPointHead.get_loss()  ← corner head loss ×0.5
                                                   │
                        solver.loss_backward()      ← weighter + 梯度投影 + AMP
```

### 权重清单（本章出现的所有手工系数，一处不落）

| 位置 | 系数 | 代码位置（画面/行号） |
|---|---|---|
| dense rot loss | `× 2` | 01:43:38 / 912 |
| rot_lidar loss | `× 2`（累加进 rot_dense） | 01:46:05 / 949 |
| dense vel loss | `× 5` | 01:45:05 / 943 |
| dense movement（二分类） | `× 10` | 01:47:30 / 977 |
| dense movement（四分类，未启用） | `× 0.8` | 01:48:20 / 992 |
| dir_cls | `× 0.5` | 01:48:20 / 1000 |
| lidar_rot_weight | `× 0.1` | 01:52:45 / **1130** |
| corner head 整体 | `× 0.5` | 01:53:10 / 1546 |
| car 的 rot 权重 | `×3`（`*2+1`） | 01:43:38 / 889 |
| 近处 VRU 的 rot 权重 | `×3`（`*2+1`） | 01:43:38 / 896 |
| 低速 vel 权重 (0.05~1.0 m/s) | `×2`（`+1`） | 01:45:05 / 936 |
| 静止 vel 权重 (<0.05 m/s) | `×1.5`（`+0.5`） | 01:45:05 / 936 |
| 动静 movement 权重 | `×3`（烧在 `attr_heats[:,5]` 里，loss 代码里看不到） | 01:47:30 / **973** 注释"把动静的loss调大3倍" |
| 朝向角分区间 | `rot_range_weight`（配置） | 01:43:38 / 908；01:51:50 / 1074 |
| 速度分区间 | `vel_range_weight`（配置） | 01:45:05 / 933；01:52:31 / 1089 |
| 10 维回归 | `code_weights`（配置） | 01:50:45 / 1050 |
| 类别 | `cls_loss_weight`（配置，car/VRU 加权） | 01:36:13 / 839 |

### 归一化因子（avg_factor）清单——这是最容易出问题的地方

| loss | avg_factor | 风险 |
|---|---|---|
| `loss_heatmap` | `max(num_pos, 1)` | 空帧时退化为 1 |
| `loss_close_heatmap` | `max(_close_num_pos, 1)` | ⚠ **近距离无目标时退化为 1 → 实测 13126** |
| `rot_dense_loss` | `(rot_attr_mask>0).sum()` | 无目标时为 0 → 除零 ⚠ |
| `vel_dense_loss` | `(v_attr_mask>0).sum()` | 同上 ⚠ |
| `mov_dense_loss` | `max(1.0, (mov_attr_mask>0).sum())` | ✅ 安全 |
| `rot_lidar_w_loss` | `max(rot_lidar_w_pos_num, 1)` | ✅ 安全 |
| `loss_bbox` / `loss_rot` / `loss_vel` | `num + 1e-4` | 分子同时为 0，巧合安全 |
| `loss_movement`（目标级） | `max(1.0, (mov_weight>0).sum())` | ✅ 安全 |

⚠ `rot_dense_loss` 和 `vel_dense_loss` 的 `avg_factor` 没有做 `max(·, 1)` 保护。当一帧完全没有目标（或全被 `is_cls_only` / `sample_mask` 屏蔽）时会 `x / 0 = nan`。**这是一个潜在的 NaN 源，值得实测确认 `self.loss_bbox` 内部是否有兜底。**

### 三个层次的监督，各自负责什么（本章最该带走的架构图）

| 层次 | 覆盖范围 | 监督的量 | 主要作用 |
|---|---|---|---|
| **稠密全图** | 448×224 全部有目标的格子 | heatmap(5)、rot、vel、movement、dir_cls | **提供绝大部分梯度**（正样本 ×50）、让特征学会"这块区域是什么、怎么动" |
| **稠密近距** | close_cls_mask 区域 | close_heatmap、rot_lidar、lidar_rot_weight | **在信息量最足、安全最关键的区域加练** |
| **目标级** | 256 个中心点 | dx,dy,z,l,w,h（+rot,vel 但不进总 loss）、movement | **保证推理时读取的那一个像素上，几何量精确** |

一句话记忆：**dense 管"学得会"，目标级管"读得准"。**

---

## 11-15　把本章接到你已有的知识上

### 和 BEVFusion / mmdet3d `CenterHead` 的差异对照

| 项目 | mmdet3d CenterHead | DenseBEV CenterHead |
|---|---|---|
| GT 栅格化位置 | 网络前向时（`get_targets` 内 Python 循环） | **DataLoader worker 里**（本章 01:32:24） |
| 分支数 | 5 reg + 1 heatmap | **10 reg + 1 heatmap**（多 dir_cls/movement/rot_lidar/lidar_rot_weight/close_heatmap） |
| 属性监督 | 只在中心点 | **中心点 + 全图 dense 双轨** |
| 类别权重 | 无 | car ×3、近处 VRU ×3 |
| 角度/速度加权 | 无 | 分区间加权 + 非对称欠估惩罚 |
| 近距离特殊处理 | 无 | close_heatmap / close_rot_heatmap / conetank 区域 |
| 数据质量分级 | 无 | `is_cls_only`、`sample_mask`、`valid_lidar`、`movement_weight` |
| loss 汇总 | `parse_losses` 按 'loss' 求和 | `.loss_p_` 排除法 + 外层梯度投影 |

**结论**：骨架完全是 mmdet3d 的（`get_targets` / `_gather_feat` / `anno_box` / `code_weights` 一字未改），**所有增量都在"处理真实数据的脏"和"照顾业务关键区域"这两件事上**。这就是"论文代码"和"量产代码"的差别，也是你从 BEVFusion 转到工作代码时最需要补的那一块。

### 和 YOLO 的类比（你正在吃透的那条线）

| CenterPoint / DenseBEV | YOLO |
|---|---|
| heatmap（每类一张、高斯软标签） | objectness + class map（网格分类） |
| `inds` 展平索引 + `gather_feat` | `build_targets` 里的 `gi, gj` 索引 + `pi[b, a, gj, gi]` |
| `reg`（格内亚像素偏移 dx,dy） | `tx, ty`（sigmoid 后的格内偏移） |
| `dim` 预测 log 空间 | `tw, th` 预测 `log(w/anchor_w)` |
| Gaussian Focal Loss | BCE + Focal（YOLOv3/v5 可选） |
| `code_weights` 逐维权重 | `box/obj/cls` 三项 gain |
| dense 属性监督 | v8 的 DFL / center-sampling 扩大正样本 |

**最大的共同点**：都是"把检测变成网格上的稠密预测 + 用索引把网格和目标列表对应起来"。你把 §11-11 的 `permute→view→gather_feat` 和 YOLO 的 `build_targets` 对着看一遍，两边都会通透很多。

### 和 nuScenes / BEVFusion 训练实践的挂钩

- 你在 4060 上跑 BEVFusion mini 时如果遇到 loss 突然 NaN，本章给了三个排查方向：(1) 是否关掉了 `@force_fp32` 或 clip_sigmoid；(2) GT 里是否有 NaN / 哨兵值没过滤（本章的 `isnotnan` 和 `INVALID_VELOCITY`）；(3) 某个 `avg_factor` 是否退化成 0。
- 本章的"无 AMP"提示也对上了：`solver.py` 里 `amp_type == 1` 用 `grad_scaler`、`== 2` 用 apex、`else` 直接 backward。你的训练计划里"无 AMP"就是走 else 分支。

---

## 11-16　⚠ 存疑清单（按重要性排序，建议向导师逐条确认）

| # | 存疑点 | 我的推断 | 证据强度 |
|---|---|---|---|
| 1 | **第 1072 行 `torch.abs(rot_weight)` 疑为 `torch.abs(rot)`** | 复制粘贴 bug；dense 版第 906 行用的是 `rot`，且 `rot` 在目标级版算完未使用；相邻的 vel 版用的是 `torch.norm(target_vel)` | **强**（三条独立证据） |
| 2 | `loss_close_heatmap = 13126.72` 占总 loss 99.6% | **三因素叠加**：(a) `avg_factor = max(_close_num_pos, 1)` 在近距离无目标时退化为 1（放大倍数 ≈ 帧内目标数，练习 ch11-6 实测 12 个目标 → 4 倍）；(b) 这是训练早期截图（logit≈0，收敛后同一公式掉到 1e-4 量级）；(c) **`loss_close_heatmap` 调用 `loss_cls` 时没有传 `nearby_mask`**（主 heatmap 传了，第 838 行 vs 第 860–862 行），若 `nearby_mask` 是本车盲区图，close 分支就会在盲区里被当纯背景罚 | 中强（(a)(b) 练习可复现；(c) 两帧代码对照确认，但依赖 `nearby_mask` 语义推断） |
| 3 | `rot_lidar_w_loss` 的 target 是 `1 - rot_lidar_w_gt` | `close_rot_heatmap` 语义可能是"不可信度"，取补得可信度 | 弱（无生成代码，两种解读都说得通） |
| 4 | dense 速度过滤只判 `v_gt[:, 0]`（vx），目标级判 `[..., -2:]`（vx,vy） | dense 侧笔误 | 中（同文件两处不一致） |
| 5 | `rot_dense_loss` / `vel_dense_loss` 的 `avg_factor` 无 `max(·,1)` 保护 | 空帧时可能除零产 NaN，除非 `self.loss_bbox` 内部兜底 | 中（代码逐字确认，兜底与否未知） |
| 6 | `attr_heat_masks[0]` 硬编码索引 `[0]` 而非 `[task_id]` | 单 task 无影响，多 task 时是 bug | 中（画面第 877–878 行逐字） |
| 7 | `is_cls_only` 是**帧级** OR 累加（任一目标 cls_only → 全帧作废回归） | `load_object.py` 第 272 行 `is_cls_only = is_cls_only or obj.get('cls_only', False)`，循环外初始化 | **强**（源码逐字） |
| 8 | 讲者说"保存成 **100** 个目标"（01:34:52） | 口误，实为 **256**（终端 `[1,256,10]`；他自己 01:51:09 也说 256） | **强** |
| 9 | 讲者说过滤前缀是 "**loss_**"（01:53:05） | 实为 `.loss_p_`（按 `loss_` 过滤总 loss 会是 0） | **强**（放大帧逐字） |
| 10 | 讲者说 "单帧的 RL / 2L 去预测增强 yaw"（01:45:46） | 应为"单帧的 **lidar**"，对应 `rot_lidar` + `lidar_front_mask` | 中强（变量名对应 + 发音相近） |
| 11 | 讲者说 VRU 加权（01:43:51）未提"近处" | 代码注释明写"把**近处** vru 的 rotation 权重乘3"，`vru_pos_mask` 与 `_cls_close_mask` 相乘 | **强**（注释+代码） |
| 12 | 讲者说"中心 XYZ 三维"（01:38:01） | `reg` 只有 2 维（dx,dy 格内偏移），z 来自 `height`；数量对但语义需分清 | **强** |
| 13 | 5 个类别的确切名称与索引 | 已确证 **ch0 = car**、**ch3 = VRU**；另有 `conetank_index` 说明有锥桶类。推断 ≈ `[car, truck, bus, VRU, conetank]` | ch0/ch3 强，其余弱 |
| 14 | `attr_heats` corner-det 分支的 `[:, -3:-1]` 与 9 通道 layout 冲突 | corner 模式下 `attr_heats` 应为 11 通道（`dir_cls_task_idx` 公式里的 `6` 佐证） | 中（推理自洽但无直接画面） |
| 15 | `attr_heats` 各属性填的是"框内全部格子"还是"高斯邻域" | 倾向"框内全部格子" | 弱（`get_targets_single` 未展示） |
| 16 | `INVALID_VELOCITY` 的具体数值 | 讲者说 50 m/s | 中（只有口述，无画面） |
| 17 | `out_size_factor` 的值 | 推断为 1（448×224×0.4m 恰好覆盖前95.4/后83.8、左右±44.8） | 中强（几何自洽） |
| 18 | `.loss_p_` 里 `p` 的含义 | print / plot | 弱（功能推断） |
| 19 | BEV 行索引 0 对应车前还是车后 | 推断行 0 = 最前方（第 1160 行 `pc_range[0] - bbox_xs*...` 是减号）；据此 `ind=38416` → 正前方 27m、横向 0m | 中 |
| 20 | 讲者说"动静的四个值"（01:35:38） | 指**4 个通道**（二分类类别+权重、四分类类别+权重），不是"4 个类别"；讲者表述正确只是没展开 | **强**（代码注释确证） |
| 21 | `self.nearby_mask` 是什么（第 838 行只传参、无定义） | 推断为一张 `[448,224]` 的**本车周围盲区/自遮挡区常量图**，在这些格子上不算分类 loss。三条依据：以 `self.` 存在 ⇒ 几何常量；只传给 `loss_cls` 不传给 `loss_bbox` ⇒ 只关"该不该报目标"；与 `close_*` 一族刻意起了不同的名字 ⇒ 语义不同 | 中（命名+用法推理，无定义画面） |
| 22 | 注释"把动静的loss调大3倍"里的 3 倍加在谁头上 | 通道 5 逐像素权重存在本身说明"不同目标权重不同"，否则直接改 `* 10` 即可 ⇒ 略倾向"**静止目标×3、移动×1**"；但也可能是整任务×3 | 弱（设计合理性推理；注释原文只说"动静"不说"静"，帧证已放大核对） |
| 23 | corner 分支的 `loss_rot_cross`（第 918–920 行）会进总 loss | key 是 `task{id}.loss_rot_cross`，**不含 `.loss_p_`** ⇒ 按第 1548 行规则会被求和进 `loss`。当前 `enable_corner_det=False` 未触发 | **强**（两帧代码逐字 + 过滤规则确证） |
| 24-a | 第 977 行 `movement` 的 BCE target 未 clamp（无效位是 -1 / ≥2 的非法 BCE 目标） | 靠 `weight=0` 屏蔽，但 `inf × 0 = nan`，理论上不安全；实际靠 `BCEWithLogitsLoss` 的 log-sum-exp 稳定化和 logit 的正常取值范围侥幸不触发。同文件的 `dir_cls`（第 999 行）就写了 `torch.clamp(gt-1, min=0)`，两处不一致 ⇒ 疏忽。建议加一行 clamp | 中强（代码逐字 + BCE 公式推导 + 同文件不一致） |
| 24 | `loss_p_rot_dense` 这一个数字里到底含几份 loss | 至少含 `rot`(×2) + `rot_lidar`(×2)（第 950 行 `+=`）；开 corner 时还含 `rot_short`(×2)（第 914 行 `+=`）。**三份混在一个标量里，TensorBoard 上拆不开** | **强**（第 914、950 行均为 `+=`，逐字确认） |

---

## 11-17　下一章预告

Ch12 从 `[01:53:31]` 开始讲 **Box 解码**：
```python
heatmap → sigmoid → score            # 本章 clip_sigmoid 的推理版
dim     → exp()                      # 本章说的 log 空间的逆运算
rot     → atan2(sin, cos)            # 本章 atan(sin/cos) 折叠歧义的完整版
reg     → 格内偏移 + inds 反解行列 → 米制  # 本章 get_corner 第 1154-1161 行已剧透
                                     # → topk 256 → NMS
```
你会发现 Ch12 的每一步都是本章某个 loss 的镜像。**把本章的表 A / 表 D 记牢，Ch12 会非常轻松。**


---
> [[Ch10_BEVUNet与CenterPoint检测头|← Ch10]] · [[00_总览与脉络|📖 总览]] · [[Ch12_Box解码与收尾QA|Ch12 →]]

