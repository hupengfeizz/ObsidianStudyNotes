---
title: BEVDepth与BEVFusion区别 复习笔记
type: 专题
tags: [专题, BEVDepth, BEVFusion]
source: 简历/BEVDepth与BEVFusion区别_复习笔记.md
updated: 2026-08-21
---
# BEVDepth 与 BEVFusion 区别复习笔记

## 1. 最核心的一句话

不要把 BEVDepth 和 BEVFusion 放在同一条线上硬比较，它们关注的维度不同：

```text
BEVDepth：主要关注 Camera 分支的深度估计怎么做得更准
BEVFusion：主要关注 Camera / LiDAR 等多模态 BEV 特征怎么融合
```

更适合面试表达的一句话是：

> 判断 Camera 分支是不是 BEVDepth-style，主要看训练时有没有用 LiDAR 投影深度作为显式 depth loss；判断整体是不是 BEVFusion-style，主要看推理时有没有 LiDAR / Camera 两路 BEV 特征融合。

## 2. 两条轴要分开看

这块最容易混淆的原因是：BEVDepth 和 BEVFusion 不是互斥概念。

应该拆成两条轴：

```text
轴 1：Camera 分支是否使用 BEVDepth-style 显式深度监督
轴 2：整体模型是否使用 BEVFusion-style 多模态融合
```

也就是说，一个模型可以同时是：

```text
BEVFusion 主框架 + BEVDepth-style Camera 分支
```

这在工程里很常见。

## 3. BEVDepth 解决什么问题

BEVDepth 主要解决的是纯视觉 BEV 检测里深度估计不准的问题。

典型流程：

```text
image
→ image backbone / FPN
→ DepthNet 预测 depth distribution
→ lift 到 frustum
→ splat / BEVPool 到 BEV
→ detection head
```

BEVDepth 的关键是：训练时用 LiDAR 投影到图像平面，生成像素级深度真值，然后监督 DepthNet。

训练期：

```text
LiDAR 点云
→ 投影到 image plane
→ 生成 depth GT
→ 监督 DepthNet
```

loss 通常可以理解成：

```text
total_loss = detection_loss + λ * depth_loss
```

推理期：

```text
通常只用 camera
不需要 LiDAR 作为输入
```

所以 BEVDepth 里的 LiDAR 更像是：

```text
训练期老师 / depth label source
```

## 4. BEVFusion 解决什么问题

BEVFusion 主要解决的是多模态融合问题。

典型流程：

```text
Camera branch:
image → camera BEV feature

LiDAR branch:
point cloud → lidar BEV feature

Fusion:
camera BEV feature + lidar BEV feature
→ fused BEV feature
→ detection / segmentation head
```

BEVFusion 的重点不是“是否显式监督 DepthNet”，而是：

```text
推理时 LiDAR 是否作为真实输入分支参与融合
```

训练期：

```text
image → camera branch → camera BEV
point cloud → lidar branch → lidar BEV
camera BEV + lidar BEV → fused BEV
fused BEV → detection_loss
```

推理期：

```text
image + point cloud
→ camera BEV + lidar BEV
→ fusion
→ final prediction
```

所以 BEVFusion 里的 LiDAR 更像是：

```text
训练期和推理期都存在的真实输入模态
```

## 5. 显式 depth loss 是什么

显式 depth loss 的意思是：DepthNet 不只是通过最终检测任务间接学习，而是有一个专门的深度监督信号。

DepthNet 输出：

```text
pred_depth[h, w, d]
```

表示像素 `(h, w)` 在不同深度 bin `d` 上的概率分布。

LiDAR 投影生成：

```text
gt_depth[h, w]
```

然后把真实深度离散到对应 depth bin，单独计算深度损失：

```text
depth_loss = CE(pred_depth, gt_depth_bin)
```

或者也可能是：

```text
depth_loss = BCE / focal loss / KL loss
```

最终 loss：

```text
total_loss = detection_loss + λ * depth_loss
```

这种情况下，DepthNet 会收到两类训练信号：

```text
1. depth_loss 的直接监督
2. detection_loss 反传回来的间接监督
```

这就是 BEVDepth-style 的核心。

## 6. 非显式 depth loss 是什么

非显式 depth loss 不是说 DepthNet 不训练，而是说没有单独的深度标签监督 DepthNet。

如果模型仍然是 LSS-style view transform，它可能仍然有：

```text
DepthNet → depth distribution → lift → BEV → detection head
```

但训练时只有最终检测损失：

```text
total_loss = detection_loss
```

DepthNet 的梯度来自最终检测任务：

```text
detection_loss
→ detection head
→ BEV feature
→ lift / splat
→ DepthNet
```

所以它是间接学深度。

一句话：

```text
显式 depth loss：depth 有自己的 loss
非显式 depth loss：depth 没有自己的 loss，只靠 detection loss 间接带着学
```

## 7. DepthNet 在两种模型里都有吗

不一定，要看具体 view transform 方法。

可以分三类：

```text
BEVDepth:
一定有 DepthNet
DepthNet 是核心模块
并且训练时有显式 depth supervision
```

```text
基于 LSS 的 BEVFusion:
通常也有 DepthNet / depth head
因为 camera branch 需要预测 depth distribution 来 lift image feature
但它不一定有显式 depth supervision
```

```text
非 LSS 式 BEVFusion:
不一定有 DepthNet
例如一些 transformer cross-attention / query-based view transform 方法
可能不显式预测像素级 depth distribution
```

所以核心不是“有没有 DepthNet”，而是：

```text
DepthNet 有没有被 LiDAR 投影出来的 depth GT 直接监督
```

## 8. 四种组合

把“两条轴”组合起来，可以得到四种情况。

### 1. Camera-only + 无显式 depth loss

```text
代表：LSS / BEVDet 风格
```

特点：

```text
只有 camera 输入
可能有 DepthNet
没有 LiDAR 投影深度监督
DepthNet 通过 detection_loss 间接学习
```

### 2. Camera-only + 有显式 depth loss

```text
代表：BEVDepth 风格
```

特点：

```text
训练期用 LiDAR 投影生成 depth GT
DepthNet 有 depth_loss 直接监督
推理期通常不需要 LiDAR
```

### 3. Camera-LiDAR fusion + 无显式 depth loss

```text
代表：普通 BEVFusion 风格
```

特点：

```text
Camera branch 输出 camera BEV
LiDAR branch 输出 lidar BEV
两者融合后做检测
LiDAR 是训练和推理时的真实输入
但不一定额外监督 Camera DepthNet
```

### 4. Camera-LiDAR fusion + 有显式 depth loss

```text
代表：BEVFusion 主框架 + BEVDepth-style depth supervision
```

特点：

```text
训练期：
LiDAR 一方面作为 LiDAR branch 输入
另一方面投影到图像上生成 depth GT，监督 DepthNet

推理期：
LiDAR 仍然作为 LiDAR branch 输入
Camera BEV 和 LiDAR BEV 融合
```

这种模型里的点云有双角色：

```text
1. 训练期老师：生成深度真值监督 DepthNet
2. 真实输入模态：进入 LiDAR encoder，生成 LiDAR BEV 特征参与融合
```

## 9. 训练期和推理期的对比

### 纯 BEVDepth

训练期：

```text
image → camera branch → BEV → detection
LiDAR → 投影到 image plane → depth GT → depth_loss
```

loss：

```text
total_loss = detection_loss + λ * depth_loss
```

推理期：

```text
image → camera branch → BEV → detection
```

LiDAR 通常不参与推理。

### 普通 BEVFusion

训练期：

```text
image → camera branch → camera BEV
point cloud → lidar branch → lidar BEV
camera BEV + lidar BEV → fused BEV → detection
```

loss：

```text
total_loss = detection_loss
```

注意：这里不排除某些实现额外加 depth_loss，但普通 BEVFusion 的定义不依赖它。

推理期：

```text
image + point cloud
→ camera BEV + lidar BEV
→ fusion
→ detection
```

LiDAR 是真实输入。

### BEVFusion + BEVDepth-style

训练期：

```text
image → camera branch → camera BEV
point cloud → lidar branch → lidar BEV
point cloud → 投影到 image plane → depth GT → supervise DepthNet
camera BEV + lidar BEV → fused BEV → detection
```

loss：

```text
total_loss = detection_loss + λ * depth_loss
```

推理期：

```text
image + point cloud
→ camera BEV + lidar BEV
→ fusion
→ detection
```

这种就是项目里最容易被问到的“双角色”。

## 10. 为什么不能简单说“纯 BEVFusion 式”

如果项目里只是：

```text
LiDAR branch 输出 BEV 特征
Camera branch 输出 BEV 特征
然后两者融合
```

那可以说是 BEVFusion-style。

但如果项目里还做了：

```text
LiDAR 投影到图像
生成深度真值
显式监督 DepthNet
```

那就不只是“纯 BEVFusion 式”了，因为 Camera 分支采用了 BEVDepth-style 的训练机制。

更严谨的表达是：

```text
整体是 BEVFusion 多模态融合框架，
Camera 分支采用 LSS / BEVDepth 式 view transformation，
并引入 LiDAR 投影深度监督训练 DepthNet。
```

## 11. 面试口语版回答

如果面试官问：

> 你们这个到底是 BEVDepth 还是 BEVFusion？

可以回答：

> 我们整体更接近 BEVFusion，因为推理时 LiDAR 和 Camera 都是输入，分别生成 LiDAR BEV 和 Camera BEV，然后在 BEV 空间融合。但 Camera 分支的 view transformation 借鉴了 BEVDepth：训练时把 LiDAR 点投影到图像上生成像素级深度真值，用显式 depth loss 监督 DepthNet，让图像分支的深度估计更准。所以点云在我们模型里有两个角色：训练时作为深度监督来源，推理时作为 LiDAR 分支的真实输入。

如果面试官继续追问：

> 那显式 depth loss 和没有显式 depth loss 的区别是什么？

可以回答：

> 显式 depth loss 是 DepthNet 除了接受最终 detection loss 的反传之外，还会额外用 LiDAR 投影得到的 depth GT 单独计算一个 depth loss，例如 CE 或 BCE。没有显式 depth loss 时，DepthNet 仍然可能存在，但它没有单独的深度标签，只能通过最终 detection loss 间接学习深度分布。

如果面试官问：

> 拔掉 LiDAR 后模型还能不能推理？

可以回答：

> 如果是纯 BEVDepth / camera-only 模型，训练完之后推理可以不需要 LiDAR；但如果整体是 BEVFusion 框架，LiDAR branch 是推理路径上的真实输入，拔掉 LiDAR 这条分支会断，除非额外做缺失模态训练或设计 camera-only fallback。

## 12. 最终记忆版

```text
BEVDepth 看 Camera 分支：
有没有用 LiDAR 投影深度 GT 显式监督 DepthNet。

BEVFusion 看整体框架：
推理时有没有 LiDAR / Camera 多模态 BEV 融合。

显式 depth loss：
total_loss = detection_loss + λ * depth_loss

非显式 depth loss：
total_loss = detection_loss

项目最稳表述：
BEVFusion 主框架 + BEVDepth-style Camera 深度监督。
```
