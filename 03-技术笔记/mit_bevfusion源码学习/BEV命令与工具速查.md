---
tags: [BEV, 工具速查, shell]
创建: 2026-08-28
用途: 实验中用过的命令与 shell 工具知识的家。对话里讲过的工具知识一律沉淀到这里。
---

# BEV 命令与工具速查

## tail —— 取输入的最后 N 行（2026-08-28 问答沉淀）

- 管道是流水线：`日志文件 → grep(筛选) → tail(截尾) → 屏幕`。
  **grep 的输出 = tail 的输入**，两种说法指同一堆东西；单独用时输入就是文件本身
- **N 的选法 = 你要的东西占几行**：
  - `tail -1` 日志最后 1 行 → "训练此刻进展到哪"
  - `tail -2` 最后 2 条评测巨型行 → 最后两次评测对比
  - `tail -3` 评测器每次打印 mAP/mAVE/NDS 三行清爽汇总 → 最后一次评测一整套
- 坑：`grep '^mAP:'` 行首锚定在混有进度条的日志里会漏行（见踩坑 26）

## grep 常用姿势

- `grep -c 'pattern'` 只数行数（查 nan 专用：`grep -c 'grad_norm: nan'`）
- `grep -oP 'lr: \K[0-9.e-]+'` 只抠出数值本身（⚠️ `[0-9.]+` 匹配不了 nan/inf，
  先专项查再抠数，踩坑 25）
- `grep -E "A|B"` 多模式任一匹配

## 进程管理

- `pgrep -cf '[t]rain.py'` 数训练进程（方括号防止 pgrep 匹配到它自己）
- `pgrep ... && tail ...` 的坑：进程没了 pgrep 返回失败，`&&` 后面**不执行**；
  收工检查用 `pgrep ... || echo 已结束`
- `nohup 命令 > 日志 2>&1 &`：后台+免疫断线和 VS Code 幽灵 ^C；`2>&1` 把报错也进日志

## 睡前检查（防 GPU 空转，两次前科后的土办法）

```bash
pgrep -cf '[t]rain.py' && tail -1 ~/project/bevfusion-main/runs/<当前训练>.log
```

## ffmpeg（2026-08-30 装于 Mac，brew install ffmpeg）

- **容器 vs 编码**：MOV/MP4 只是"盒子"（封装格式），画质由里面的编码+码率决定。
  同内容换盒子画质零差别；"转完变糊"是重编码压了码率，不是格式的锅
- 无损换封装（只换盒子不动内容，秒级）：
  `ffmpeg -i in.mov -c copy out.mp4`（-c copy = 流原样复制，不重编码）
- 查视频参数/时长：`ffprobe -v quiet -print_format json -show_format -show_streams 文件`
  （Mac 土办法备用：`mdls -name kMDItemDurationSeconds 文件`）
- 拼接：编码参数一致 → concat demuxer + -c copy 无损直拼；不一致 → 需重编码
- 拼接实战（2026-08-30 bev代码串讲两段 MOV→mp4）：
  `printf "file '路径1'\nfile '路径2'\n" > list.txt`
  `ffmpeg -f concat -safe 0 -i list.txt -map 0:v:0 -map 0:a:0 -c copy -tag:v hvc1 -strict unofficial 出.mp4`
  三个坑：①Apple 空间音频轨(apac) mp4 容器不支持→只映射主音轨；②杜比视界元数据
  默认被丢→加 -strict unofficial 保住；③QuickTime 元数据流(data)不进 mp4，无内容损失。
  验收三件套：ffprobe 时长=两段之和、拼缝两侧各抽一帧、耳听衔接。
  识别错拷贝：两个文件字节数完全相同=同一内容（ls -la 比大小）
- 两段有重叠的无损去重拼接（2026-08-30 实战）：①音频互相关定位——两段各抽 3 分钟
  8kHz 单声道 wav，短时能量包络滑动点积，峰值即对齐点（精度±0.02s，z>5 才可信）；
  ②剪第一段的尾巴而非第二段的头（尾裁逐包精确，不受关键帧限制，可保持 -c copy）；
  ③concat 列表里给第一段加 `outpoint <秒>` 即可。两次架机的画面角度跳变无法消除属正常

## 本地语音转文字（Whisper，2026-08-30 建立）

- 环境：`~/.venv_whisper`（独立 venv，装 mlx-whisper，M2 GPU 加速）
- 流程：ffmpeg 抽 16kHz 单声道 wav → mlx_whisper.transcribe(large-v3-turbo,
  language=zh, **initial_prompt=领域词表**) → 实测 126 分钟音频 9.8 分钟转完
- 质量：结构远好于会议自动字幕，但领域词仍错（针孔→"真空"、Depth→"Deps"），
  **必须跟一遍术语校正**（有序替换表 ~150 组 + 幻听清除）
- Whisper 三类垃圾要清：静默段空行、重复幻听循环（"就是啊"×N）、
  广告词幻听（"请不吝点赞订阅…"——静默处的经典幻觉）
- 校正低置信处必须在文件头标 ⚠ 声明，不能装作全对

