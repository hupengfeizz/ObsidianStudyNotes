---
tags: [BEV, BEVFusion, create_data, nuScenes, 逐句精讲, 辅导课]
日期: 2026-08-30（辅导发生于 8-29 深夜～8-30 凌晨）
时长: 1:39:54
形式: 一对一远程辅导（说话人1=朋友/导师，@iris=你；他远程控制你的工作站 VS Code 讲解）
材料: 语音转写 2248 行 + 抽帧 207 张（--drift 8 --min-gap 20）+ 工作站真实代码逐行核对
帧位置: /Users/apple/Documents/36/bev_study/bev视频/36视频抽帧/bev_create_data/
---

# 《bev_create_data》辅导课逐句精讲

> **本篇按"真·逐句"标准生产**：对话的每个有内容的意群都有原话+解释，所有代码论断
> 已对照工作站 `~/project/bevfusion-main` 逐行核实（行号为你机器实况），
> 朋友的 6 处口误/含糊已在纠错表更正——其中 2 处是实质性错误，面试照搬会翻车。

## ⭐ 课程定位与总体评价

| 维度 | 评分 | 说明 |
|---|---|---|
| 教学质量 | ★★★★☆ 4/5 | 实战派讲法：不念概念，直接开 json 文件和源码对着讲；"去中心化"画图讲解是亮点。扣分：现场找文件/找不到 main 耗了约 8 分钟，两处实质性口误（见纠错表 B） |
| 干货密度 | ★★★★☆ 4/5 | 100 分钟基本无废话，问答节奏快；你的提问质量高（bin 序号、样本级 token、时序、可见性字段——有两问问到了他答错/答不上的点） |
| 对你的价值 | ★★★★★ 5/5 | **这节课把你已有的四条知识线全部接通了**（详见"与你的知识体系接通"）：max_sweeps↔叠帧消融、他车补偿↔专题结论、−yaw−π/2↔你的yaw修正坑、token链↔厨房类比的索引卡。且讲的正是你亲手跑过 40 分钟的那条命令背后的代码 |

**一句话主旨**：`create_data.py` 把 nuScenes 的十几个 json（互相用 token 链接的"小数据库"）
翻译成两类产物——①`infos_train/val.pkl`：每帧的路径+标定链+时序链+GT 框七要素，训练读数据全靠它；
②`gt_database/ + dbinfos_train.pkl`：把每个目标的点抠出来去中心化存成 bin，供 GT-sampling 数据增强"贴目标"用。

## 目录

| Part | 内容 | 时间 |
|---|---|---|
| 0 | 开场（共享调试，折叠） | 00:00–00:56 |
| 1 | 产物总览：三个 pkl + gt_database + GT-sampling 概念 | 00:56–07:43 |
| 2 | 找 main 函数与"代码被改过"插曲 | 07:43–11:17 |
| 3 | create_nuscenes_infos 入参与数据集划分 | 11:17–13:21 |
| 4 | scene 是什么：场景≠样本 | 13:21–18:42 |
| 5 | token 体系：场景级/帧级/时序链 | 18:42–29:24 |
| 6 | 标定链：lidar2ego 与 ego2global | 29:24–36:57 |
| 7 | info 的其他字段 + 四元数转矩阵 | 36:57–43:23 |
| 8 | 六相机循环：obtain_sensor2top 链式变换 | 43:23–54:26 |
| 9 | 关键帧 vs sweep + 两级运动补偿 | 54:26–58:57 |
| 10 | sample_annotation：3D 框七要素 | 58:57–01:04:40 |
| 11 | 速度坐标转换 + NameMapping + SECOND 角度 | 01:04:40–01:09:24 |
| 12 | info 汇总落盘：两个 infos pkl 诞生 | 01:09:24–01:13:48 |
| 13 | create_data vs create_gt_database | 01:13:48–01:17:37 |
| 14 | create_gt_database 逐行：从读帧到抠点 | 01:17:39–01:34:23 |
| 15 | 去中心化：白板推演 | 01:34:23–01:37:28 |
| 16 | 收尾方法论：怎么学这种没教程的东西 | 01:37:28–01:39:54 |

---

## Part 0 · 开场（00:00–00:56）

> 🗀 折叠：飞书共享黑屏调试、账号问题，无技术内容。00:56 进入正题。

---

## Part 1 · 产物总览：三个 pkl + gt_database（00:56–07:43）

### 1.1 create_data 最终生成什么（00:56–01:14）

> **朋友**：它最终生成的是几个文件？三个 pkl，还有一个 nuscenes_gt_database。

四个产物（你 data/nuscenes/ 下都有，当时生成花了约 40 分钟）：

| 产物 | 你机器上的实物 | 用途 |
|---|---|---|
| nuscenes_infos_**train**.pkl（转写作"券点pkl"） | 473MB / 28130 帧 | 训练集每帧的全部索引信息 |
| nuscenes_infos_**val**.pkl（转写作"view点/喂点"） | 98MB / 6019 帧 | 验证集同上 |
| nuscenes_**dbinfos**_train.pkl（转写作"DB infos 圈点PQ"） | 200MB | GT-sampling 增强的目标清单 |
| nuscenes_**gt_database**/ | 7.9GB / 823476 个 bin | 每个目标抠出来的点云库 |

### 1.2 infos pkl 装什么（01:14–02:31）

> **朋友**：把你的所有数据——每一帧点云对应的图像是哪一个、点的数量、旋转矩阵——这一帧对应的都给它写好、一一匹配好。

**infos pkl = 每帧的"户口本"**：不存数据本身，存"这帧的点云文件在哪、六张图在哪、
标定矩阵是什么、前后帧是谁、GT 框有哪些"。（和你解剖台账里"pkl=菜单/索引卡"的厨房类比完全一致，这节课等于把索引卡逐字段翻开了。）

### 1.3 dbinfos = GT-sampling 数据增强的清单（02:31–04:39）「帧 00_04_32」

> **朋友**：3D 检测有一个最重要的数据增强，叫 GT sampling……把你这一帧里面的目标的点单独抠出来，存到 nuscenes_gt_database 里去。

帧 00_04_32 拍到了它在你配置里的位置——`configs/nuscenes/default.yaml` 的
`ObjectPaste`（GT-sampling 在 mmdet3d 里的名字）：

```yaml
type: ObjectPaste
stop_epoch: ${gt_paste_stop_epoch}   # 训练后期会关掉这个增强
db_sampler:
  info_path: ${dataset_root + "nuscenes_dbinfos_train.pkl"}
  prepare:
    filter_by_min_points:   # 入库门槛：框内至少 5 个点才配进库
      car: 5
      truck: 5
      ...（全类=5）
  sample_groups:            # 每帧要贴到多少个（每类不同！）
    car: 2
    truck: 3
    construction_vehicle: 7
    bus: 4
    trailer: 6
    barrier: 2
    motorcycle: 6
    bicycle: 6
    pedestrian: 2
    traffic_cone: 2
```

> ⚠️ **精确化**：朋友口头说"必须保证每类要 5 个"，把两个参数说混了——
> **5 是入库的点数门槛**（filter_by_min_points，少于 5 点的目标不进库），
> **每帧贴多少由 sample_groups 决定且每类不同**（car 贴 2 个、construction_vehicle 贴 7 个——越稀有的类贴越多）。
> 另注意 `stop_epoch`：贴目标增强在训练末期会停用（让模型最后见真实分布）。

### 1.4 bin 文件名的三段结构（04:39–05:46）

> **朋友**：前面从 0A 开始到 004 是这一帧的名字；杠 bus 说明这是单独的一个 bus 目标；这个 7 代表它在整个数据集里排行第七。

文件名 `{帧token}_{类别}_{i}.bin`。**注意：最后一段的语义他答错了**——
代码（create_gt_database.py:320-321）是 `for i in range(num_obj): filename = f"{image_idx}_{names[i]}_{i}.bin"`，
**i 是"这一帧里第 i 个框"的帧内序号，不是全数据集序号**。你在 07:26 追问"是当前帧的第七个还是整个视频的第七个"，
他答"整个视频"——你的问题问得准，他的答案错了。详见纠错表 B-1。

### 1.5 贴目标怎么贴：库→帧（05:46–07:26）

> **你**：他是从当前帧做一个复制，还是可以把目标贴到其他帧上去？
> **朋友**：先把所有目标都放到 gt_database 这个库，相当于生成了一个库。训练第 0 帧时从库里选目标贴进去，训练第 1 帧再从库里选、再贴。

**两阶段**：生产期（现在讲的）建库；训练期每帧从**全数据集的库**里抽目标贴入。
目的：有些帧目标太少，贴完之后"点的数量变大、目标数量变多"。

> 💡 **与 CBGS 辨析**（你已实战过 CBGS，别混淆）：两者都治类别不平衡，但机制不同——
> **CBGS 在"发号"环节重复抽整帧**（稀有类的帧多训几遍），**GT-sampling 在"帧内"贴目标**（把库里的稀有目标粘进当前帧）。
> BEVFusion 默认两个同时开；你的消融实验里 CBGS 被你关了（--use_cbgs False），ObjectPaste 一直开着。

---

## Part 2 · 找 main 函数与"代码被改过"插曲（07:43–11:17）

### 2.1 现场混乱实录（07:43–11:17）

现场找不到 main：朋友以为 nuscenes_converter.py 底部该有 main，你的没有；他断言"你这代码百分之百改过了"（你提到"cloud/Claude 帮我跑过实验"）。当时没对齐就跳过了。

### 2.2 【真相核实】你的代码没有他说的那个问题 ✅

我逐一核对了你工作站的实际代码：

- **入口 main 一直都在 `tools/create_data.py`**（第 99–122 行：`if args.dataset == "nuscenes" ...` 分发 trainval/mini），它 import `nuscenes_converter` 再调 `create_nuscenes_infos`（create_data.py:31）。
- **`nuscenes_converter.py` 本来就没有 main**——它是被调用的模块，MIT 原版就这样。朋友记忆里的"converter 底部有 main"可能是 mmdetection3d 另一版本的布局。
- 你的仓库确实被改过，但改的是**实验开关**（default.yaml 的 sweeps_num 变量化、train.py 的 CBGS 开关、pointpillars.yaml 的 grid_size 修复——全部有台账记录），**数据生产链一行没动**。你当时生成的 850 场景/28130 帧全量 pkl 也验证过是好的。

结论：**不用重新下源码**，这个插曲翻篇。

---

## Part 3 · create_nuscenes_infos 入参与数据集划分（11:17–13:21）

### 3.1 version 与 max_sweeps（11:17–11:53）「代码：nuscenes_converter.py:42-43」

> **朋友**：version 是 v1.0-trainval 就是完整数据集，mini 的话就 v1.0-mini。max_sweeps 什么意思？就是我一直跟你说的——
> **你**：叠帧的是吗？
> **朋友**：对，叠帧的参数，默认就是 10。

`def create_nuscenes_infos(root_path, info_prefix, version="v1.0-trainval", max_sweeps=10)`。
**这个 10 你太熟了**：pkl 里每个关键帧最多记 10 个历史 sweep（converter.py:234 `while len(sweeps) < max_sweeps`），
训练时 LoadPointsFromMultiSweeps 的 sweeps_num 再决定实际叠几帧——你的整个 T1 消融（单帧 vs 十帧）动的就是下游那个开关，上游配额在这定。

### 3.2 splits：训练/验证场景是官方定死的（11:53–13:21）

> **朋友**：nuScenes 一共划分很多个 scene……90 个训练 10 个验证这种，都是 nuScenes 里自带的。get_available_scenes 把名字拿出来，哪个是 train 哪个是 val 它给你划分好。

官方 splits 固定（全量 850 场景 = train 700 + val 150，你生成时日志里见过这两个数）。
「帧 00_14_26」拍到了 `get_available_scenes`（converter.py:122-160）的正文：遍历 nusc.scene、
取每场景第一帧的 lidar_path、`mmcv.is_filepath` 检查文件真的存在，不存在的场景剔除——
所以它打印 `exist scene num: 850` 时，等于顺带做了一次数据完整性体检（你当时验收清单里的那行日志就是这么来的）。

> 💡 你的 q25 子集（175 场景）就是在这套"按场景划分"体系上二次切分的——按场景切防泄漏的道理，跟官方 train/val 按场景分是同一个。

---

## Part 4 · scene 是什么：场景≠样本（13:21–18:42）

### 4.1 scene.json：官方定死的场景清单（14:24–15:23）

> **朋友**：scene-0061、scene-0103……nuScenes 自动默认划分好的，就读 scene.json，这里面所有 json 全是自带的。

### 4.2 你的追问：scene 是"白天雨天"那种场景吗？name 是一个样本吗？（14:19 / 16:36）

> **朋友**：不是按天气，就是一段采集。一个 name 不是一个样本，是**一个场景**——街道、乡下、高速这种。场景下有很多样本。

层级敲定：**scene（≈20 秒的一段采集，约 40 帧）→ sample（帧，=一个训练样本）**。
「帧 00_17_17」可见 scene.json 的 description 字段：如 "parking lot, parked bicycles, bus, many pedestrians…"——每个场景带一句英文描述（晚上/大街道/停车区等）。

### 4.3 mini vs 全量（18:19–18:42）

现场开的是 mini（10 场景），朋友强调全量"很大"。你机器上两套都有：mini 用来快速做实验/复现，全量 850 场景训练用。

---

## Part 5 · token 体系：场景级/帧级/时序链（18:42–29:24）

### 5.1 进入 _fill_trainval_infos：从 LIDAR_TOP 开始（18:42–21:57）

> **朋友**：调入 nuScenes 的 sample，把 data 下面的 LIDAR_TOP（转写作"雷达top"）的 token 取出来。

核心函数 `_fill_trainval_infos`（converter.py:163，「帧 00_14_26」可见签名 max_sweeps=10）。
每个 sample 的 `data` 字典里按传感器名（LIDAR_TOP、CAM_FRONT…）存着对应 sample_data 的 token。

### 5.2 token 是什么：加密版的索引（21:57–23:03）

> **朋友**：正常命名可能就 0001、0002 这种简单 ID 映射。有些公司为了加密，把它转成哈希那种 DZ8B3966……你理解为就是索引就行。
> **你**：我在公司里见过。

**token = 主键**。nuScenes 全库用 32 位十六进制哈希做主键，各 json 之间靠 token 互相引用——
这就是你在 01:37:57 感慨的"很像个小的数据库"：**它就是一组以 token 为外键的关系表**。
「帧 00_33_53」右半的知乎关系图值得存档：scene / sample / sample_data / ego_pose / calibrated_sensor / sample_annotation / instance / category 的连接关系一图全览。

### 5.3 两级 token：场景级 + 帧级（23:03–25:30）

> **朋友**：我这个场景在 100 个场景里是哪一个位置，我这一帧在场景里对应哪个 token——两个目标。
> **你**：那就是除了样本级的还有帧级别的 token……中间没有样本级的吗？
> **朋友**：样本就是帧。点云是按帧来的，单位不可能是视频。

你带着公司里"一个视频=一个样本"的习惯来问，这里被纠正：**nuScenes 的一个样本 = 一帧**（一个 sample 打包了这一时刻的 1 份点云 + 6 张图 + 标注）。

### 5.4 时序怎么串：你问"没有时序信息了吗"（25:30–27:43）

> **你**：一个样本是一帧，那时序信息是不是就没了？
> **朋友**：有啊……我把当前帧的前面一帧和后面一帧都记录到当前帧里去，拼接的时候按索引一帧一帧找，就可以按序拼接，叠帧不就可以叠了吗。

**问得非常好**——时序不靠"视频"保存，靠**链表**：每帧记着前一帧和后一帧的 token，
叠帧时顺着链一路往前找。（他起初把字段说成 first/last_sample_token，27:55 自己纠正——见下一卡。）

### 5.5 字段名精确化：scene 的 first/last vs sample 的 prev/next（27:43–29:24）「帧 00_28_35」

> **朋友**（自己找到了）：sample 级的也有啊——prev 什么意思？前面一帧；next 是不是下面一帧？
> **你**：我们这个 next 是空的，那说明就截止了。
> **朋友**：最后一帧了……肯定有一帧 prev 是空的，说明它是第一帧。

标准答案（帧 00_28_35 的 sample.json 实拍可见）：

| 字段 | 挂在哪 | 含义 |
|---|---|---|
| first_sample_token / last_sample_token | **scene** 上 | 这个场景的第一帧/最后一帧 |
| **prev / next** | **sample** 上 | 时序链表的前驱/后继；prev="" 是场景首帧，next="" 是末帧 |
| scene_token | sample 上 | 我属于哪个场景（同场景的帧此字段相同） |

> 💡 **接你的实测**：`prev=""` 的场景首帧就是你体素化实验踩过的退化样本
> （idx0 无历史可叠、pad_empty_sweeps 自我复制凑数）——链表的"头结点"在数据侧长这样。

---

## Part 6 · 标定链：lidar2ego 与 ego2global（29:24–36:57）

### 6.1 calibrated_sensor.json：外参 + （相机才有的）内参（29:24–31:16）

> **朋友**：从这一帧找到 calibrated_sensor，标定参数。因为每一帧必须保留内外参。
> **你**：为什么（内参）是个空的？

**你当时的困惑值得讲透**：calibrated_sensor.json 每条记录有三个字段——
`translation`、`rotation`（该传感器到自车的**外参**）+ `camera_intrinsic`（**内参，只有相机有**）。
你们当时看的是 **LIDAR_TOP 的记录，雷达没有内参，所以 camera_intrinsic 是空列表 []**——不是数据缺失。
（朋友 30:27 一度也说绕了"这是内参"，见纠错表 B-4。）

### 6.2 lidar2ego：ego 就是 IMU 的位置（31:16–34:19）「帧 00_33_53」

> **朋友**：自车指的就是你 IMU 的位置……雷达 to ego 其实就是雷达 to IMU。

他专门让你打开知乎那张传感器布局图（帧 00_33_53）：车顶中央 LIDAR_TOP，车身中心 IMU（=ego 原点），
6 个 CAM、5 个 RADAR 环布。**info 里的 `lidar2ego_translation/rotation` 就从 calibrated_sensor 的 CS record 里抄**。

### 6.3 ego2global：车在地球上的位置（35:38–36:57）

> **朋友**：相当于 GPS、北斗一样——车在北京在上海跑，我得知道它在 GNSS/GPS 坐标系下在哪个位置。

从 **ego_pose.json**（pose record）拿 translation/rotation → info 的 `ego2global_*`。
**这两级标定链是你早就用过的**：viz_moc.py 里 `pts @ R.T + t` 做自车运动补偿，用的正是这里存进 pkl 的 ego_pose——今天算是看到了它的"出生地"。

---

## Part 7 · info 的其他字段 + 四元数转矩阵（36:57–43:23）

### 7.1 时间戳与点云路径（36:57–37:58）

每帧 info 记 `timestamp`（微秒级——点云第 5 维那个"时间戳偏移"就是拿相邻帧的这个字段相减）和 `lidar_path`（按 token 直接映射到 bin 文件路径，"后面读点的时候拿这个路径直接读"）。

### 7.2 log.json：这帧是哪辆车、何时、在哪采的（37:58–38:32）

> **朋友**：这个车叫 0N015，2018 年采集的，位置在 Singapore……这个在波士顿。

log = 采集日志（车辆号/日期/地点）。nuScenes 数据一半新加坡（右舵！）一半波士顿。

### 7.3 简名复制：l2e_r 那批变量（39:10–39:42）

代码把 lidar2ego、ego2global 的旋转平移复制成短名（l2e_r / l2e_t / e2g_r / e2g_t），纯粹为了后面公式写起来短，无新信息。

### 7.4 Quaternion：四元数 → 旋转矩阵（39:42–43:23）

> **朋友**：json 里存的是四元数，我要转成旋转矩阵，因为我们计算都是 numpy 矩阵计算……就跟把 list 转 numpy array 一样，是个格式转换。

`pyquaternion.Quaternion(...).rotation_matrix`——课上现场用豆包查了这个库（还有一句名场面："你先不要用百度，你这个百度太拉了"）。
**够用的理解**：四元数=旋转的 4 数紧凑存法（无万向节死锁），要做矩阵乘就转成 3×3 旋转矩阵。深究数学此刻不必要。

---

## Part 8 · 六相机循环：obtain_sensor2top 链式变换（43:23–54:26）

### 8.1 六相机逐个登记（43:23–44:30）

定义 6 个 camera 名，for 循环逐个：取该相机的 sample_data → calibrated_sensor 拿内参（相机有！）→ 调 `obtain_sensor2top`（converter.py:295）。

### 8.2 obtain_sensor2top 在算什么（44:30–48:26）

> **朋友**：获取你当前这个传感器到顶部雷达的 RT matrix，旋转和平移矩阵。
> **你**：这传感器指哪一个？IMU 吗？相机？
> **朋友**：看 for 循环 for 的是 camera，那肯定是相机啊。

**输出 = camera→LIDAR_TOP 的外参**（sensor2lidar_rotation / translation），顺带把该相机的
data_path、内参、时间戳都打包进 info 的 cams 字典。

### 8.3 链式变换的路径（48:26–51:46）

> **朋友**：sweep to ego，ego to global，然后 global 再回到雷达……一步步乘。
> **你**：这是一整个的坐标变换，从（相机）到自车的一个坐标变换。

数学上就是「帧 00_33_53」知乎页顶部那条公式：
**T_cam→lidar = (T_ego→lidar 链) ∘ (T_global→ego') ∘ (T_ego→global) ∘ (T_cam→ego)** ——
相机先升到自车、再升到世界（大地坐标系/GNSS，49:50 他考了你这个词），再从世界降回雷达那一时刻的自车、降到雷达。
**为什么绕道世界系**：相机和雷达的标定都只有"各自到自车"，而两者时间戳不同、自车已移动——世界系是唯一的公共桥。

### 8.4 这些矩阵将来干什么用（51:46–53:25）

> **朋友**：做多模态的时候，比如点云投影到图像上，就得用它的旋转和平移。
> **你**：这个操作是不是每一帧都得做？……不是对某个车某个人做的吧？
> **朋友**：对，是的（整帧统一变换）。

**BEVFusion 的 camera 支路**（图像特征投到 BEV）用的就是这批矩阵——你解剖地图 v2 里"视锥→投到 BEV"那一站的原料，在数据生产期就备好了。

### 8.5 两层 for 的辨析（53:38–54:26）

> **你**：for sample 这个循环是对六个相机是吧？……上面还有个 for sample 呢？
> **朋友**：外层是每一帧（你不是只有一帧啊），内层才是六相机。

外层遍历 28130 个 sample，内层遍历 6 相机——嵌套结构你自己读出来了。

---

## Part 9 · 关键帧 vs sweep + 两级运动补偿（54:26–58:57）

### 9.1 keyframe 概念（54:26–55:33）

> **朋友**：sample 是采样的关键帧，sweep 是非关键帧。比如 100 帧每隔几帧抽一标注。

**关键帧=有人工标注的帧（2Hz）**；雷达实际 20Hz 扫描，中间未标注的转圈叫 sweep。
（他举例"每隔 5 抽 1"是示意；nuScenes 实际约每 10 个 sweep 出 1 个关键帧。）

### 9.2 你的好问题：不连续标注没影响吗？（55:33–56:14）

> **你**：它不连续进行标注，没有影响吗？
> **朋友**：有什么影响呢？训练的时候叠帧，把前四帧点云拼到当前帧上来，**框还是当前关键帧的框**。

**要点**：叠帧只借历史帧的**点**（加密点云），**监督信号（框）永远是当前关键帧的**——历史 sweep 本来就没有标注。

### 9.3 两级补偿：你主动输出，朋友背书（56:14–57:09）

> **你**：运动补偿有两级，一个对自车、一个对他车。他车好像需要标注，MIT 这块没做。
> **朋友**：他车你做不了。你只有自己车上的传感器，只能做自车运动补偿……
> **你**：相当于（他车）不一样长，那些动态目标会有一个残影是吧？
> **朋友**：会有残影的。

**这一分钟是你整场最高光的时刻**——你把专题实验的核心结论（两级补偿、第②级推理时不可用、动目标残影）主动讲了出来，被实战工程师当场确认。你的专题文档结论获得了外部背书（含你实测的残影率 49.3% vs 14.3%，比"会有残影"更进一步——你有数字）。

### 9.4 sweeps 信息也进 info（57:13–58:57）

> **朋友**：关键帧 005 下面包含 004、003、002、001 的信息……非关键帧的信息都写到关键帧的 info 里。

converter.py:234 `while len(sweeps) < max_sweeps:` 沿 prev 链收集至多 10 个 sweep（每个也过一遍 obtain_sensor2top 存标定），挂到该关键帧 info 的 `sweeps` 字段——**这就是你查过的 `infos[idx]['sweeps']`**（idx0 那个空列表=场景首帧无历史，一切都对上了）。

---

## Part 10 · sample_annotation：3D 框七要素（58:57–01:04:40）

### 10.1 标注从哪来 + instance token（58:57–01:00:23）

> **朋友**：3D 框从 sample_annotation 里拿……每个目标有实例级 token，instance 就是每个目标。

sample_annotation.json 每条 = 一个目标在一帧里的标注；`instance_token` 标识"同一个物体"跨帧的身份（跟踪就靠它）。

### 10.2 3D 框的七要素 + 附加字段（01:00:23–01:02:22）

> **朋友**：translation 对应中心点，size 长宽高，rotation 旋转角度，还有速度，还有框里有多少个雷达点——你看这个框才 5 个点，这个才 2 个点。

| 字段 | 含义 |
|---|---|
| translation | 框中心点（global 坐标系） |
| size | 宽长高（nuScenes 序：w, l, h） |
| rotation | 朝向（四元数） |
| velocity | 速度（由前后帧标注差分而来）——**mAVE 的真值源头，也是第②级补偿只能用于真值生产的原因** |
| num_lidar_pts | 框内雷达点数（你看到 5 个点、2 个点——远处目标就这么稀疏，呼应你 mAP 塌方在小目标的结论） |

### 10.3 你的追问：visibility 字段（01:02:22–01:04:03）

> **你**：好像还有个可见性的字段？
> **朋友**（现场查）：目标在相机图像里没被遮挡的比例。1=0~40% 严重遮挡，2=40~60%，3=60~80% 轻度，4=80~100% 基本可见。**BEVFusion 里不会用到这个。**

好问题+诚实的现场查证。（评测时 nuScenes 官方会用 visibility 过滤，训练确实不用。）

---

## Part 11 · 速度转换 + NameMapping + SECOND 角度（01:04:40–01:09:24）

### 11.1 速度从 global 转到雷达系（01:04:03–01:04:40）

> **朋友**：速度是在 global 下的，给它转到雷达坐标系下——做一个坐标系的转换。

真值速度必须和点云同坐标系，模型预测的 vx,vy 才有意义（你 T1 里 mAVE 的比较基准）。

### 11.2 NameMapping：细类→粗类（01:04:40–01:08）

> **朋友**：nuScenes 命名非常完整，比如 vehicle.bus.bendy，训练时映射成粗类（bus）。
> **你**：实际用的是右边（映射后）这个吗？
> **朋友**：肯定用映射完之后的。左边只是标注时更详细。

converter.py:272-273：`names[i] = NuScenesDataset.NameMapping[names[i]]`。
细粒度 23 类 → 训练 10 类（你评测表里的 car/truck/bus/trailer/CV/pedestrian/motorcycle/bicycle/barrier/cone）。

### 11.3 ★ SECOND 格式与 −yaw−π/2：你的 yaw 坑的官方出处（01:07:56–01:09:24）

> **朋友**：second 也是一篇 3D 检测论文，它是 KITTI 数据集的角度格式，X 朝前 Y 朝左……rot 减去二分之派，做了个角度转换。
> **你**：就是相当于绕角会变一下。

代码原文（converter.py:276）：

```python
gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
```

> 💡 **闭环时刻**：你在 Part1 可视化时踩过"框画歪"的坑，逆向猜出"pkl 存的 yaw = −几何朝向−90°"（修复后框内点 441→558）。
> **这行代码就是那个约定的出生地**——nuScenes 的 yaw 转成 SECOND/KITTI 习惯（x 前 y 左）时取负再减 π/2。
> 中心点和长宽高不受影响，只有角度定义变（他的原话"长和中心坐标不变，角度定义有点差"）。

---

## Part 12 · info 汇总落盘：两个 infos pkl 诞生（01:09:24–01:13:48）

### 12.1 单帧 info 集齐（01:09:24–01:10:33）

> **朋友**：名字有了、速度有了、多少个雷达点有了、遮挡信息有了——当前这一帧的详细信息都放到 info 里面。
> **你**：那就相当于拿到了一个样本的所有信息了。

### 12.2 train/val 分开、写盘（01:10:33–01:13:01）

> **朋友**：train 和 val 肯定分开放……创建 {前缀}_infos_train.pkl，前缀叫 nuscenes——生成 nuscenes_infos_train.pkl，保存的就是所有训练帧的 info；验证集同理 val。

打印 train/val 样本数（你全量时是 28130/6019）→ 两个 pkl 落盘。**到此四个产物完成两个。**

### 12.3 阶段小结（01:13:01–01:13:48）

> **朋友**：坐标转换关系、雷达和图像的（标定）关系、每帧的 box 信息——全都写到这里面来了。还剩两个没生成：dbinfos 和 gt_database。
> **你**：数据增强的东西。

---

## Part 13 · create_data vs create_gt_database（01:13:48–01:17:37）

### 13.1 职责划分（01:16:04–01:17:06）

> **你**：create_data 和 create_gt_database 有啥区别？
> **朋友**：create_data 生成数据的 pkl 信息；create_gt 单独生成每一个 3D 样本的独立信息（dbinfos + gt_database）。
> **你**：那训练之前这两个都得做吗？……哦我理解，create_gt_database 就是为了做数据增强用的。
> **朋友**：对，专门为数据增强做的。不做 GT-sampling 就可以不生成。

（工程事实补充：你机器上这两步是 `create_data.py` 一条命令里连着跑的——create_nuscenes_infos 完了接着调 create_groundtruth_database，当时 778 秒 + 26 分钟的两段就是它俩。）

### 13.2 为什么一定要做：提 5 个点（01:17:06–01:17:37）

> **朋友**：我之前做实验发现，加了 GT sampling，mAP 至少能提 5 个点以上，提点非常明显。

一手实战数据。（结合 1.3 的 stop_epoch：提点主要来自训练早中期的富集监督。）

---

## Part 14 · create_gt_database 逐行（01:17:39–01:34:23）

### 14.1 入口与多数据集兼容（01:17:39–01:19:11）

> **朋友**：mmdetection3d 是框架，既能做 KITTI 也能做 nuScenes，只关心 nuScenes，KITTI 跳过。

`create_groundtruth_database`（create_gt_database.py:111）先建 dataset_cfg 字典，按 dataset_class_name 分支。

### 14.2 pipeline 定义：这里只是"声明"不是"执行"（01:19:11–01:20:43）

> **朋友**：LoadPointsFromFile 单帧导入……LoadPointsFromMultiSweeps 多帧、拼 10 帧……LoadAnnotations3D 导入 3D 框。只是定义 pipeline，没有做任何函数操作。

NuScenesDataset 分支实际参数（create_gt_database.py:180-199，你机器核实）：

```python
LoadPointsFromFile(load_dim=5, use_dim=5)                 # 单帧 5 维
LoadPointsFromMultiSweeps(sweeps_num=10, use_dim=[0..4],
                          pad_empty_sweeps=True, remove_close=True)
LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True)
```

> ⚠️ **"16 维"更正**：课上说"导入的维度是 16 维"——16 是**另一分支**（`load_augmented`，
> 虚拟点/painted points 用，create_gt_database.py:208/215）的参数；**默认走的分支是 load_dim=5**。见纠错表 B-2。
>
> 💡 **两个彩蛋**：①抠目标用的点云是**叠了 10 帧的稠密版**（sweeps_num=10）——库里存的目标比单帧看到的肥；
> ②`pad_empty_sweeps=True` 又见面了——场景首帧在这儿同样自我复制凑数（你的体素化实验结论横跨到了数据生产链）。

### 14.3 空壳先建好（01:20:14 前后）「帧 01_20_14」

> **朋友**：生成 nuscenes_gt_database 文件夹、nuscenes_dbinfos_train.pkl——注意还没导入内容，只是创建了一个空文件夹和空文件。

（db_info_save_path 逻辑在 create_gt_database.py:256-257。）

### 14.4 with_mask：2D 框兼容，跳过（01:20:43–01:21:07）

> **朋友**：做图像时有 2D 框（比如 YOLO），这里只为代码兼容。BEVFusion 没有 2D 框信息，只有 3D 框，关掉不管。

### 14.5 主循环：整帧读入（01:21:07–01:22:33）

> **朋友**：100 帧数据从 0 遍历……第 0 帧对应的点云、第 0 帧对应的框都读进来了。
> **你**：它是某一个点吗？还是这个目标的所有点？
> **朋友**：一帧点，所有的一帧点全部读进来，**它还没开始提取呢**。

先整帧进内存（点+框），抠取在后面。

### 14.6 ★ 抠点的那一行：points_in_rbbox（01:22:33–01:25:28）

> **朋友**：gt_box 是一个 7（XYZ 中心点 + 长宽高 + 一个角度）……288 行这一步才开始抠：根据 3D 框，把这一帧里属于这个框的 3D 点抠出来——point_indices 就是属于框内的点的索引。
> **你**：他这里抠出来，比如一个目标有 6 个点，就只是 6 个点的（索引）？
> **朋友**：对，把这 6 个点的索引拿出来就行了。

`point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)`（create_gt_database.py:288）——
一个 (点数 × 框数) 的布尔矩阵，第 i 列为 True 的行就是落在第 i 个框内的点。
（你 Part1 算残影率时手写过"点是否在框内"的判断——官方版就是这个函数。）

### 14.7 保存：文件名拼装与写盘（01:29:00–01:30:40）

> **朋友**：337~339 行，名字 = 帧名 + 类别名 + 序号，保存到 nuscenes_gt_database 路径下，一一对应好，直接 tofile 保存下来。
> **你**：他就获得了每个目标的一个点云。
> **朋友**：对，写了一大堆，最终目的就是获取每个目标的点云信息存成 bin。

实际行号（你机器）：320-321 拼 filename，326 取点，327 减中心（下一 Part），338-339 `gt_points.tofile(f)`。

### 14.8 3D 框存哪了：你的收尾追问（01:30:40–01:32:36）

> **你**：它的 3D 框是不是也有？……复制粘贴只需要复制点云，不需要 3D 框？
> **朋友**：db_info 里名字、路径、gt 索引、**box3d 框的信息、点的信息全部保存到一起**，写到 all_db_infos，最后存成 nuscenes_dbinfos_train.pkl。训练读的时候直接读。

**bin 里只有点；框七要素在 dbinfos pkl 里**——贴目标时把点按框位姿变换过去、框也一起带过去（贴入帧的 GT 里会新增这个框）。

---

## Part 15 · 去中心化：白板推演（01:26:21–01:37:28）「帧 01_36_44」

### 15.1 那一行减法（01:26:21–01:28:01）

> **朋友**：提出来之后做了一步操作——减去框的中心点。所有目标都以坐标原点为单位……我给你画个图。

`gt_points[:, :3] -= gt_boxes_3d[i, :3]`（create_gt_database.py:327）。

### 15.2 白板例子（01:35:32–01:37:17，帧 01_36_44 实拍红笔）

> **朋友**：框从 (1,1) 到 (2,2)，中心 (1.5,1.5)。框里每个点减 (1.5,1.5)——(2,2) 变 (0.5,0.5)，全都移到以 (0,0) 为中心。每个目标减去自己的中心，都做了**去中心化**。
> **你**：好像确实……只要减中心点就可以做到。
> **朋友**：目的就是去完中心化之后，接下来**贴的时候方便**。

**为什么方便**：库里的目标统一躺在原点，贴到任何一帧时只需"旋转到目标朝向 + 平移到落点"一步到位；
若带着原始坐标存，每次贴都得先减原位姿再加新位姿，还容易错。**存归一化的、用时再变换**——工程通用套路。

---

## Part 16 · 收尾方法论（01:37:28–01:39:54）

### 16.1 你的两个真实感受（01:37:41 / 01:37:57）

> **你**：这块确实比 2D 复杂好多，数据挺杂的……一个 token 一个 token 的，很像个小的数据库，挺抽象的。
> **朋友**：2D 就一个平面，没有角度。token 它把索引搞复杂了点，**你自己做的时候不用这么复杂，只要知道一一映射的关系是什么就可以**。

### 16.2 怎么学：朋友的三步法（01:38:22–01:39:16）

> **你**：达到你这种程度的理解，我需要怎么去学习？
> **朋友**：首先把代码跑通；跑通之后不需要全懂，起码把具体流程搞清楚，自己理一下、写清楚；**知道每个函数做什么、输入是什么、输出是什么就可以了，后面再慢慢抠细节。前面抠细了没有意义。**

> 💡 与你已定的"从零到一六步法"完全互证（跑通→流程→输入输出→细节），而且**第一步你早就完成了**——create_data 你 8-22 就跑通过，今天是在补"流程搞清楚"这一步，本精讲就是他说的"自己理一下、写清楚"。

### 16.3 没有教程（01:39:16–01:39:37）

> **你**：这方面有什么教程吗？都是你们全自学的吗？
> **朋友**：没有教程，这玩意肯定没教程……工作之后自己学的。有这种教程你早就搜到了。

---

## ✎ 转录纠错表

### A. 语音转写错误（txt 的锅，读转写时自动替换）

| 转写写成 | 实际是 |
|---|---|
| 券点 pkl / 圈点 / 圈 | **train**（nuscenes_infos_train.pkl） |
| view点 / 喂 / will | **val** |
| inforce / enforce | **infos / dbinfos** |
| 雷达 top | **LIDAR_TOP** |
| new sense / nuance / newson / newscase / user的 | **nuScenes** |
| PTR / PQL / PQ / pico / people | **pkl** |
| 面函数 / 面点 py | **main 函数 / main** |
| GDbox / GGbox / JDbox / 基地 box | **gt_box（GT 框）** |
| Ability | **velocity（速度）** |
| 死格式 / second format | **SECOND 格式**（论文名，这个转写倒是对的） |
| 基因SS | **GNSS** |
| U2R / Citation / Py Citation | **Quaternion / pyquaternion** |
| sample 写作 simple / symbol | **sample** |
| K 里面 / KD | **KITTI** |
| receive 成 | **reshape** |
| GT sampling 写作 GD sampling / gdtest / gdc base | **GT sampling / gt_database** |

### B. 讲者口误 / 含糊（实质内容，面试别照搬）

| # | 课上说法 | 应为 | 依据 |
|---|---|---|---|
| B-1 | bin 文件名的"7"是**整个数据集**排行第七的 bus | **帧内第 i 个框**（0 起）的序号 | create_gt_database.py:320 `for i in range(num_obj)`，i 在每帧重置 |
| B-2 | "导入的维度是 16 维" | 默认分支 **load_dim=5**；16 维是 load_augmented（虚拟点）分支 | create_gt_database.py:186 vs :208 |
| B-3 | 用 first_sample_token/last_sample_token 讲帧间前后索引 | 帧间链表是 sample 的 **prev/next**；first/last 挂在 **scene** 上指场景首末帧（他 27:55 自己纠正了一半） | sample.json 实拍（帧 00_28_35） |
| B-4 | 指着 rotation/translation 说"这是内参" | 那是**外参**（sensor→ego）；内参是 camera_intrinsic 且只有相机有（雷达为空列表——你当时的疑问） | calibrated_sensor.json 结构 |
| B-5 | "100 帧每隔 5 抽一标 20 帧" | 示意性说法；nuScenes 实际雷达 20Hz、关键帧 2Hz，**约每 10 帧标 1** | nuScenes 官方规格 |
| B-6 | "必须保证每类 5 个" | **5 = 入库点数门槛**（filter_by_min_points）；每帧贴多少由 **sample_groups** 定且每类不同（car 2 / CV 7 …） | default.yaml（帧 00_04_32 实拍） |

---

## ◈ 与你的知识体系接通（六处闭环）

1. **max_sweeps=10 ↔ 你的 T1 消融**：pkl 生产期每帧最多记 10 个 sweep（上游配额），训练期 sweeps_num 决定实际叠几帧（你动的下游开关）。上下游今天接通。
2. **−rots−π/2（converter.py:276）↔ 你的 yaw 修正坑**：viz_moc 框画歪的根因，官方出处找到了。
3. **他车补偿做不了 ↔ 专题文档结论**：实战工程师背书，而且你手里有他没有的数字（残影率 49.3%/14.3%、mAVE ×2.4）。
4. **prev="" 的场景首帧 ↔ 体素化退化样本**：pad_empty_sweeps 在数据生产链（gt_database 的 pipeline）里同样存在。
5. **token 关系表 ↔ 厨房类比**："pkl=索引卡"如今展开成完整 schema（帧 00_33_53 的关系图值得打印贴墙）。
6. **GT-sampling ≠ CBGS**：一个帧内贴目标、一个重复抽帧；你的消融关的是 CBGS，ObjectPaste 一直开着（stop_epoch 前）。

## ✓ 行动清单

1. **朋友布置的**：把流程"自己理一下、写清楚"——本精讲已代完成初稿，**建议你用自己的话把"四个产物各是什么"对着目录讲一遍**（对镜/对 chat 均可）。
2. **顺手可做的实物验证**（工作站，一条命令看 dbinfos 长什么样）：
   ```bash
   python -c "import pickle; d=pickle.load(open('data/nuscenes/nuscenes_dbinfos_train.pkl','rb')); k=list(d.keys()); print('类别:',k); e=d['car'][0]; print('car样例:',{x:e[x] for x in ['name','path','num_points_in_gt']})"
   ```
3. **面试储备**：把 B-1（bin 序号）、B-6（两个 5 的区别）当"我读过源码"的证据讲——纠正过权威的人才算真读过。

## ☆ 学习建议

- **不用整段重看视频**：本精讲已按逐句标准覆盖；要回味的只有两段——33:24 传感器布局图讲解、01:35:32 白板去中心化（各 2 分钟）。
- **消化顺序**：先把"目录表+一句话主旨"讲给自己听 → 再过六处闭环 → 纠错表 B 过一遍防面试翻车。
- **与解剖体系的关系**：本课覆盖了解剖地图"共同起点：Dataset/DataLoader"再往上游的一层（原始 json→pkl）。至此你的数据链完整了：**json →(create_data)→ pkl →(Dataset/pipeline)→ 张量 →(体素化)→ 柱子**。
- 下一个自然衔接：训练期 GT-sampling 怎么"贴"（ObjectPaste/db_sampler 的运行时代码）——朋友只讲了生产期，运行期在 mmdet3d/datasets/pipelines 里，需要时说一声我备路标。
