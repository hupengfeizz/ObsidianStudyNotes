---
tags: [BEV, BEVFusion, 调试方法论, 配置系统, spconv]
创建: 2026-08-23
---

# BEVFusion 实验环境搭建实录与方法论

> 这份文档记录 2026-08-22~23 做的每一处改动：**为什么改、怎么改、你自己该怎么做**。
> 关联：[[BEVFusion训练计划_4060工作站]]、[[BEVFusion阶段0基准测量结果]]

---

## 0. 改动清单（每一条都可撤销）

所有改动都在 git 仓库里，`git diff` 随时可查，且都留了 `.bak` 备份。

| # | 文件 | 改了什么 | 撤销方法 |
|---|---|---|---|
| 1 | `configs/nuscenes/default.yaml` | 数据路径 mini → 全量 | `git checkout configs/` |
| 2 | `configs/.../lidar/pointpillars.yaml` | 补 `test_cfg.grid_size` | 同上 |
| 3 | `tools/train.py` | 加 CBGS 开关 | `cp tools/train.py.bak tools/train.py` |
| 4 | `mmdet3d/ops/spconv/conv.py` | 空张量保护（**未生效**） | `cp .../conv.py.bak .../conv.py` |
| 5 | `mmdet3d/ops/spconv/ops.py` | 诊断打印（**临时，用完要删**） | `cp .../ops.py.bak .../ops.py` |
| 6 | conda 环境 | protobuf 走纯 Python 实现 | 删 `envs/bev/etc/conda/activate.d/protobuf_fix.sh` |

查看全部改动：

```bash
cd ~/project/bevfusion-main && git status --short && git diff --stat
```

---

## 1. 数据子集：为什么按 scene 切

### 问题

全量 28130 帧，voxelnet 一个 epoch 要 1.9 小时。8 组消融实验跑不完。
要缩小数据规模，但**不能随便缩**。

### 为什么不能按帧随机切

nuScenes 一个 scene 是**连续 20 秒行车片段**，约 40 个关键帧，采样频率 2 Hz。
相邻两帧之间车只开了 2~3 米，画面几乎一模一样。

按帧随机切的后果：同一场景的第 10 帧进训练集、第 11 帧进验证集
→ 模型验证时看到的是训练时几乎见过的画面
→ **验证分数虚高，而且高得没道理**，做出来的消融结论全是假的。

按 scene 切：整个场景要么全进训练、要么全进验证，训练和验证走**完全不同的路段**。

> 这是数据划分的通用原则：**按"独立单元"切，不按样本切**。
> 医疗影像按病人切（不按切片），语音按说话人切，推荐系统按用户切。
> 判断标准：切完之后，两边的样本之间还有没有信息泄漏。

### 怎么做

`scene_token` 不在 infos pkl 里（info 字典只有 `token` = 样本 token），
所以要从 `v1.0-trainval/sample.json` 建立 `样本token → 场景token` 的映射。

脚本：`tools/make_subset.py`（我写的，你可以读一遍，逻辑很简单）

```bash
# 先看分布，不写文件
python tools/make_subset.py --infos data/nuscenes/nuscenes_infos_train.pkl \
    --meta data/nuscenes/v1.0-trainval --stats-only
```

```bash
# 切 25%，训练集和验证集必须用同一个 seed
python tools/make_subset.py --infos data/nuscenes/nuscenes_infos_train.pkl \
    --meta data/nuscenes/v1.0-trainval --frac 0.25 --seed 0 \
    --out data/nuscenes/nuscenes_infos_train_q25.pkl
```

**关键点：只生成新的 pkl，一个图片/点云文件都不复制。**
pkl 本质是个 list of dict，切子集就是从列表里挑一部分存成新文件。

### 实测结果

| | 全量 | 25% 子集 |
|---|---|---|
| 训练 | 700 场景 / 28130 帧 | 175 场景 / **7026 帧** |
| 验证 | 150 场景 / 6019 帧 | 38 场景 / **1529 帧** |

场景比例 25.0%，帧数比例 25.0%——**成比例说明切法没有偏差**。

### 公司里怎么做

同样的原则，但规模不同：

- 元数据不会是 pkl，而是**数据库表**（Hive/Iceberg/ClickHouse），
  切子集是一条 SQL：`WHERE scene_id IN (SELECT ... TABLESAMPLE ...)`
- 划分结果要**版本化**并落库，不能是某个人机器上的一个 pkl
  （否则三个月后没人能复现你的实验）
- 通常还会做**分层抽样**：保证子集里的天气、城市、时段、类别分布和全量一致，
  而不是纯随机。我这个脚本只做了随机抽场景 + 打印城市分布供人工核对

---

## 2. CBGS：它是什么，为什么要关，怎么关

### 它是什么

**Class-Balanced Grouping and Sampling**，出自论文 *Class-balanced Grouping and
Sampling for Point Cloud 3D Object Detection*。

nuScenes 类别极度不平衡：car 33.9 万个，而 bicycle 只有 8185 个，差 40 倍。
直接训练，模型会疯狂优化 car，忽略长尾类别。

CBGS 的做法：**对含稀有类别的帧重复采样**，让每个 epoch 里各类别出现频率接近。
代价是 epoch 变长——实测在 q25 子集上 **7026 → 31150，放大 4.43 倍**。

### 为什么要关

不是因为它不好，是因为**时间预算**：

| | 一个 epoch |
|---|---|
| 开 CBGS | 31150 iter × 0.24s = **2.08 小时** |
| 关 CBGS | 7026 iter × 0.24s = **28 分钟** |

8 组实验 × 4 epoch，开着 CBGS 要 66 小时，关掉只要 15 小时。

> 而且从实验设计角度：CBGS 本身就该是**一个独立的消融变量**，
> 而不是所有实验都默认开着的背景噪声。

### 怎么关（这里有个坑）

配置里 CBGS 不是一个布尔开关，而是**包了一层数据集**：

```yaml
data:
  train:
    type: CBGSDataset          # 外层包装
    dataset:                   # 内层才是真正的 NuScenesDataset
      type: NuScenesDataset
      ann_file: ...
```

想关掉，就要把 `data.train` 整个换成内层那个 dict。

**坑：torchpack 的 YAML 合并是"深合并"，不是"替换"。**

我实测验证过：如果在子配置里写

```yaml
data:
  train:
    type: NuScenesDataset
    ann_file: ...
```

得到的结果是：

```
data.train 的键: ['type', 'dataset', 'ann_file', ...]
                          ^^^^^^^^ 父配置的 dataset 键还在！
```

而 `NuScenesDataset.__init__` 没有 `**kwargs`，多出来的 `dataset` 参数会直接 `TypeError`。

**所以纯靠 YAML 覆盖是绕不过去的。**

### 解法：给 train.py 加开关

在 `tools/train.py` 里 `build_dataset` 之前插入：

```python
if cfg.data.train.get("type") == "CBGSDataset" and not cfg.get("use_cbgs", True):
    logger.info("use_cbgs=False -> 剥离 CBGSDataset 包装")
    cfg.data.train = cfg.data.train.dataset
```

**设计考量**：
- 默认 `use_cbgs=True`，不传这个参数时**行为和原版一模一样**，不影响别人
- 通过配置键控制，所以能用 torchpack 的 `--use_cbgs False` 从命令行传
- 只有 3 行，且加在 tools/ 而不是 mmdet3d/ 库代码里，影响面最小

用法：

```bash
torchpack dist-run -np 1 python tools/train.py <config> --use_cbgs False ...
```

### 公司里怎么做

- **不要直接改上游代码**。正确做法是把这类开关做成**配置项**，
  在你们自己的配置层（而不是第三方库）里控制
- 如果非改不可，改动要：① 有默认值保持原行为 ② 有注释说明原因
  ③ 记录在 CHANGELOG 或 wiki 里，让接手的人知道这不是上游原版
- 更规范的做法是维护一个 **patch 文件**或 fork 分支，升级上游时能重新应用

---

## 3. torchpack 配置系统：三层机制

MIT BEVFusion 用的不是 mmdet3d 标准的 Python 配置，而是 torchpack 的 YAML 系统。
搞懂这三层，配置就不会再懵。

### 第一层：递归继承

```python
configs.load(args.config, recursive=True)
```

`recursive=True` 会**从根目录一路向下**，把每一级目录里的 `default.yaml` 依次加载，
最后加载你指定的那个文件。

比如加载 `configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml`，
实际加载顺序是：

```
configs/default.yaml
configs/nuscenes/default.yaml
configs/nuscenes/det/default.yaml
configs/nuscenes/det/transfusion/default.yaml
configs/nuscenes/det/transfusion/secfpn/default.yaml
configs/nuscenes/det/transfusion/secfpn/lidar/default.yaml
configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

**后加载的覆盖先加载的，且是深合并。**

> 这解释了一个现象：`gt_paste_stop_epoch` 在 `configs/nuscenes/default.yaml` 里是 `-1`，
> 但在 `.../lidar/default.yaml` 里被覆盖成 `15`。
> 所以纯激光配置有 GT-Paste，融合配置没有——因为融合配置的路径里没有 `lidar/` 这一级。

### 第二层：`${}` 变量插值

```yaml
dataset_root: data/nuscenes/
ann_file: ${dataset_root + "nuscenes_infos_train.pkl"}
```

`${}` 里可以写 Python 表达式，在 `recursive_eval(configs)` 时求值。
所以改一个 `dataset_root`，所有引用它的地方都跟着变。

### 第三层：命令行覆盖

```python
args, opts = parser.parse_known_args()   # opts 是没被识别的参数
configs.update(opts)                      # 交给 torchpack 处理
```

用法是 `--嵌套.键.路径 值`：

```bash
--data.train.dataset.ann_file data/nuscenes/nuscenes_infos_train_q25.pkl
--max_epochs 4
--optimizer.lr 0.0002
```

值会先尝试 `literal_eval`（所以 `4` 变 int，`False` 变 bool），失败则当字符串。

**优先级：命令行 > 指定的 yaml > 各级 default.yaml**

### 怎么自己验证配置最终长什么样

这是最有用的一招——**不要靠猜，直接打印出来**：

```bash
python -c "
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval
configs.load('configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml', recursive=True)
configs.update(['--use_cbgs','False','--max_epochs','4'])
cfg = Config(recursive_eval(configs))
print(cfg.data.train.type)
print(cfg.max_epochs)
"
```

我今天定位 PointPillars 的 bug、验证 CBGS 开关、确认数据路径，全靠这一招。

---

### 补充：一个参数值究竟存在于哪里（2026-08-23）

以学习率为例。`--optimizer.lr 3e-5` 跑完之后，这个值在**三个物理位置**：

| 位置 | 完整路径 | 值 | 性质 |
|---|---|---|---|
| **源文件** | `/home/hpf/project/bevfusion-main/configs/nuscenes/det/transfusion/secfpn/default.yaml` 第 34-36 行 | `lr: 1.0e-4` | 手写，git 跟踪，**从未被改** |
| **命令行** | 只存在于你敲的那条命令里 | `3e-5` | 临时，只对这一次运行有效 |
| **运行存档** | `/home/hpf/project/bevfusion-main/runs/<run-dir>/configs.yaml` | `lr: 3.0e-05` | 程序自动 dump，**这次实验实际用的值** |

源文件长这样（`configs/nuscenes/det/transfusion/secfpn/default.yaml`）：

```yaml
optimizer:
  type: AdamW
  lr: 1.0e-4          # PointPillars 和 voxelnet 都继承这一行
  weight_decay: 0.01
```

**在源码里搜 `3e-5` 是搜不到的**——命令行覆盖只改运行时内存，不写磁盘。

数据流：

```
配置文件 1.0e-4  ──┐
                   ├──→ 内存中的 cfg ──→ 训练用它 ──→ dump 到 runs/xxx/configs.yaml
命令行   3e-5    ──┘         ↑
                        3e-5 覆盖 1.0e-4
```

验证源文件确实没动：

```bash
cd /home/hpf/project/bevfusion-main && git diff configs/nuscenes/det/transfusion/secfpn/default.yaml
```

输出为空 = 没被修改过。

### 补充：`--optimizer.lr` 不是"定义好的参数"

`tools/train.py` **全文只定义了 2 个参数**：

```bash
grep -n 'add_argument' /home/hpf/project/bevfusion-main/tools/train.py
```

```
24: parser.add_argument("config", metavar="FILE")     # 配置文件（位置参数）
25: parser.add_argument("--run-dir", metavar="DIR")   # 输出目录
```

`--optimizer.lr`、`--max_epochs`、`--use_cbgs`、`--data.train.dataset.ann_file`
**全都没有定义**。它们能生效，是因为 torchpack 把点号当成**配置树的路径**：

```
--optimizer.lr 3e-5
      ↓ 按 . 拆开
configs['optimizer']['lr'] = 3e-5
```

**所以配置树里任何一条路径都能从命令行改**，不需要谁"支持"它：

```bash
--model.heads.object.num_proposals 300
--model.encoders.lidar.voxelize.max_voxels "[60000, 80000]"
--lr_config.target_ratio "[3, 0.0001]"
```

torchpack 的实现在：

```bash
python -c "import torchpack.utils.config as c; print(c.__file__)"
```

里面 `update()` 有两个 `@multimethod` 重载——传字典走配置文件合并那条，传列表走命令行那条。

### ⚠️ 坑：拼错的参数名会静默失效

torchpack **不校验路径是否存在**。少写一个点：

```bash
--optimizer_lr 3e-5      # 少了点号
```

会被当成新建一个顶层键 `optimizer_lr`，**不报任何错，但对训练毫无影响**，
而 `optimizer.lr` 还是默认的 `1.0e-4`。

自己复现这个坑：

```bash
cd /home/hpf/project/bevfusion-main && python -c "
from torchpack.utils.config import configs
configs.load('configs/nuscenes/det/transfusion/secfpn/lidar/pointpillars.yaml', recursive=True)
configs.update(['--optimizer_lr', '3e-5'])
print('optimizer.lr 实际值:', configs.optimizer.lr)
print('多出来的野键     :', configs.get('optimizer_lr'))
"
```

**所以每起一个训练，都该查一眼 run 目录里的 `configs.yaml` 确认参数真的生效了：**

```bash
grep -A3 '^optimizer' /home/hpf/project/bevfusion-main/runs/<run-dir>/configs.yaml
```

### 判断标准：什么该改文件，什么该走命令行

| | 改源文件 | 命令行覆盖 |
|---|---|---|
| 持久性 | 永久 | 只对这条命令有效 |
| 出现在 `git diff` | ✅ | ❌ |
| 适用 | **长期不变的环境配置** | **实验变量** |
| 本项目实例 | `dataset_root`（数据路径） | `--optimizer.lr`、`--max_epochs`、`--use_cbgs` |

**判断依据只有一条：这个值会不会在不同实验之间变化。**

会变的走命令行——好处是**看命令就知道这次实验用了什么**，不用去翻文件当时是什么状态；
也不会因为忘记改回来而污染下一组实验。


---

## 4. spconv 崩溃排查实录（方法论重点）

这一段最值得看，因为**结论不重要，过程才重要**。
我在这个问题上走了三次弯路，每次都是被数据打脸后修正的。

### 现象

```
RuntimeError: mmdet3d/ops/spconv/include/tensorview/helper_launch.h 17
N > 0 assert faild. CUDA kernel launch blocks must be positive, but got N= 0
```

间歇性：同配置同数据，有时跑 500 步没事，有时 50 步内就崩。

### 弯路一：怀疑数据

**假设**：某个点云文件是空的或损坏的，导致体素化产出 0 个体素。

**怎么验证**：

```bash
# 找零字节文件
find samples/LIDAR_TOP sweeps/LIDAR_TOP -name '*.bin' -size -1c | wc -l
# 找最小的文件（每个点 20 字节）
find samples/LIDAR_TOP -name '*.bin' -printf '%s %p\n' | sort -n | head -5
```

**结果**：0 个空文件，最小的也有 685 KB（34272 个点）。**假设被否定。**

### 弯路二：怀疑 GPU 架构

**假设**：4060 Ti 是 sm_89，而扩展只编译到 sm_86，kernel 静默失败返回 0。

**怎么验证**：

```bash
# 看编译产物里有哪些架构
/usr/local/cuda-11.8/bin/cuobjdump --list-elf mmdet3d/ops/spconv/sparse_conv_ext...so | grep -oE 'sm_[0-9]+' | sort -u
# 看有没有 PTX（可 JIT 到新架构）
/usr/local/cuda-11.8/bin/cuobjdump --list-ptx ...so | grep -oE 'compute_[0-9]+' | sort -u
```

**结果**：只有 sm_70/75/80/86，没有 PTX。看起来假设成立。

**但被两条证据否定**：
1. **CUDA 二进制兼容规则**：同一大版本内（8.x），低小版本的 cubin 可以在高小版本上跑。
   sm_86 的代码在 sm_89 上是合法的。
2. **GitHub issue #297 的报告者用的是 A100（sm_80）**，同样的错误。

> 教训：**看到"版本不匹配"就下结论是最常见的误判**。
> 要么找到确凿的机制证据，要么找到反例。我这次是靠反例（A100 也崩）才醒过来的。

### 弯路三：拦错了地方

**假设**：N=0 就是输入的活跃体素数为 0，在卷积入口拦一下就行。

**做法**：在 `conv.py` 的 `forward` 里加：

```python
if indices.shape[0] == 0:
    return 空的 SparseConvTensor
```

**结果**：崩溃照旧，而且**保护的计数器是 0——从来没触发过**。

说明 `N` 不是 `indices.shape[0]`，是 `get_indice_pairs` **内部**的另一个量。

> 教训：**加了保护要验证它真的触发过**。
> 我特意在保护里加了计数和 warning，才发现它形同虚设。
> 如果只看"没崩就是修好了"，会得出完全错误的结论（那次恰好是运气好）。

### 正确的下一步：让程序自己告诉你

不再猜，直接在失败点把所有输入打出来：

```python
try:
    return get_indice_pairs_func(indices, batch_size, out_shape, spatial_shape, ...)
except RuntimeError:
    print(f"indices.shape={tuple(indices.shape)} batch_size={batch_size} "
          f"spatial_shape={spatial_shape} out_shape={out_shape} "
          f"ksize={ksize} stride={stride} padding={padding}", file=sys.stderr)
    raise
```

候选嫌疑：
- `batch_size` 为 0？
- `out_shape` 某个维度被反复降采样到 0？（sparse_shape 的 z 只有 41，
  经过几次 stride-2 后可能变得很小）
- `indices` 里的坐标越界被丢弃？

**这一步还没跑，是留给你的作业。**

### 通用的调试方法论

我今天用到的，按优先级排序：

1. **读完整堆栈，从最深一帧往上看**。最深的那帧才是真正出错的地方，
   上面都是调用链。今天两次都是这样定位到 `get_indice_pairs`。
2. **做能区分假设的实验**，而不是能确认假设的实验。
   比如"是数据问题还是环境问题"→ 用已知好的 mini 数据跑同一个配置，一次就分清了。
3. **搜上游 issue**。这个 bug 在 GitHub 上有两个 issue（#82、#297），
   而且 #297 直接否定了我的架构假设。**搜索比自己猜快十倍。**
4. **加保护要能观测**。计数器 + warning，否则你不知道它有没有生效。
5. **打印比猜快**。到第三次弯路才想起来直接 print，早该这么做。

### 公司里怎么做

- **必须先搜内部 issue / wiki / 群聊记录**。你遇到的问题八成有人遇到过
- 定位到是上游 bug 后，**先看上游有没有修复**（新版本、PR、fork）
- 自己打的补丁要：① 有注释写明来源（issue 链接）② 有开关或降级路径
  ③ 有可观测性（日志、指标），不能是"悄悄吞掉错误"
- **不要在生产训练里静默跳过异常样本**。要记录下来，事后分析。
  否则模型在某类数据上一直学不到东西，你还不知道

---

## 5. 当前状态与你的下一步

### 已就绪

- 全量数据 + 25% 子集（train 7026 帧 / val 1529 帧）
- 基准数据：pointpillars 0.218s、voxelnet 0.241s、camera 0.289s、fusion 0.477s
- CBGS 开关可用
- PointPillars 配置已修复，**且它不用 spconv，不受这个 bug 影响**

### 待办

1. **跑诊断，抓住 spconv 崩溃时的参数**（作业，见 §4）
2. 决定路线：
   - **路线 A**：先在 PointPillars 上做完整套消融（不受 bug 影响，且对应你的简历项目）
   - **路线 B**：先修 spconv，再做需要 voxelnet/fusion 的模态消融
3. 清理临时的诊断补丁（`ops.py`），别忘了

### 需要你自己敲的命令

清理诊断补丁（用完之后）：

```bash
cd ~/project/bevfusion-main && cp mmdet3d/ops/spconv/ops.py.bak mmdet3d/ops/spconv/ops.py
```

跑 PointPillars 的第一组实验（路线 A）：

```bash
cd ~/project/bevfusion-main && torchpack dist-run -np 1 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/pointpillars.yaml --use_cbgs False --data.train.dataset.ann_file data/nuscenes/nuscenes_infos_train_q25.pkl --data.val.ann_file data/nuscenes/nuscenes_infos_val_q25.pkl --data.test.ann_file data/nuscenes/nuscenes_infos_val_q25.pkl --max_epochs 4 --run-dir runs/E1-pointpillars-q25
```

预计 7026 iter × 0.218s × 4 epoch ≈ **1.7 小时**（外加每个 epoch 结束的评测）。
