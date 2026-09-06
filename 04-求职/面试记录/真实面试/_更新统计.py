#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面试统计更新脚本 v2（可持续更新机制的核心工具）

用法：
    cd 到 真实面试/ 文件夹，运行：  python3 _更新统计.py
    输出两部分（markdown 表格，可直接粘贴进《面试问题统计_减分点总库》）：
      ① 减分点编码排行榜 —— 来自各场 frontmatter 的 `减分点编码: [..]`
      ② 问题级主题排行榜 —— 来自各场文末「## 全问题索引」表的主题标签列

新增一场面试的登记流程（4 步，详见总库 §五）：
    1. 按模板写 面试实录_YYYY-MM-DD_HHMM_xxx.md：
       - frontmatter 填 `减分点编码: [B03, B03, K05, ...]`（出现几次写几次）和 `评分: NN`
       - 文末带「## 全问题索引」表：| # | 时刻 | 主题标签 | 问题 | 我的回答概要 | 评 | 减分点 |
         主题标签用《问题主题词表》里的标签（可 1-2 个，逗号分隔）；评 用 ✅/⚠️/❌
    2. 在 面试分析进度总表.md 登记该场
    3. 运行本脚本，用输出更新总库的两张排行榜
    4. 新问题类型：编码按 B16…/K23… 顺延、标签在词表新增一行——绝不改写旧含义
"""
import os, re, glob, sys
from collections import defaultdict

# ============ 减分点编码表（与总库 §一 严格同步） ============
NAMES = {
    "B01": "用分工回答能力（“有专人负责”）",
    "B02": "签收降格总结",
    "B03": "“不清楚/忘了/记不清”裸放弃",
    "B04": "数字问题（零/错/漂移/矛盾/无法自辩）",
    "B05": "弱动机表述",
    "B06": "答非所问·答what不答why·被纠偏多轮",
    "B07": "归属弱化（协助/我们/帮忙）",
    "B08": "关键设计题答太短/太浅",
    "B09": "超长独白无控制点",
    "B10": "拿到信息不转化",
    "B11": "先讲组织再讲我",
    "B12": "简历与实际不符",
    "B13": "不会的题停在“没有”",
    "B14": "自我缩小（不带切割）",
    "B15": "谈判与临场职业素养",
    "B16": "公司功课缺失（对公司/岗位零调研）",
    "K01": "自车vs目标运动补偿/残影",
    "K02": "BEVPoolv2机制",
    "K03": "DepthNet深度概率分布（vs argmax）",
    "K04": "SmoothL1/focal/CIoU等损失函数",
    "K05": "自己系统的参数口径",
    "K06": "ODD/失效感知/“全场景”",
    "K07": "BEV高度归约/投影几何/空间相关性",
    "K08": "YOLO系（标签分配/loss/FPN）",
    "K09": "标定/外参精度",
    "K10": "C++（内存泄漏/weak_ptr/RAII）",
    "K11": "ROS/中间件",
    "K12": "工程基础（Docker/Linux/Git）",
    "K13": "端到端/BEVFormer/前沿架构",
    "K15": "毫米波/弱模态/失效融合鲁棒性",
    "K17": "量化排查/校准原理/两端对齐",
    "K19": "跟踪/ID/MOTR检测跟踪一体",
    "K20": "深度学习/算法基础",
    "K22": "数据增强与采样策略",
    "K23": "Python 语言机制（GIL/内存管理/方法类型）",
}

def read_head(path, n=8192):
    with open(path, encoding="utf-8") as f:
        return f.read(n)

def read_all(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def parse_frontmatter(text):
    codes, score = [], None
    m = re.search(r"减分点编码:\s*\[([^\]]*)\]", text)
    if m:
        codes = [c.strip() for c in m.group(1).split(",") if c.strip()]
    m = re.search(r"评分:\s*(\d+)", text)
    if m:
        score = int(m.group(1))
    return codes, score

def parse_qindex(text):
    """解析「## 全问题索引」表 → [(tags,[...]), verdict, question)]"""
    out = []
    m = re.search(r"##\s*全问题索引(.*?)(?:\n## |\Z)", text, re.S)
    if not m:
        return out
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("#", "---") or set(cells[0]) <= {"-", " "}:
            continue
        tags = [t.strip() for t in cells[2].replace("，", ",").split(",") if t.strip() and "." in t]
        verdict = "✅" if "✅" in cells[5] else ("❌" if "❌" in cells[5] else ("⚠️" if "⚠" in cells[5] else "?"))
        if tags:
            out.append((tags, verdict, cells[3]))
    return out

def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(folder, "面试实录_2*.md")))
    if not files:
        print("未找到 面试实录_*.md，请在 真实面试/ 文件夹内运行。")
        return

    # ---------- ① 减分点编码 ----------
    sess_count = defaultdict(set); inst_count = defaultdict(int)
    scores, no_codes = [], []
    # ---------- ② 问题级主题 ----------
    t_sess = defaultdict(set); t_cnt = defaultdict(int)
    t_bad = defaultdict(int); t_warn = defaultdict(int); t_good = defaultdict(int)
    no_index = []
    total_q = 0

    for path in files:
        name = os.path.basename(path)
        text = read_all(path)
        codes, score = parse_frontmatter(text[:8192])
        if codes:
            if score is not None:
                scores.append(score)
            for c in codes:
                sess_count[c].add(name); inst_count[c] += 1
        else:
            no_codes.append(name)
        qs = parse_qindex(text)
        if not qs:
            no_index.append(name)
        for tags, verdict, _ in qs:
            total_q += 1
            for t in tags:
                t_sess[t].add(name); t_cnt[t] += 1
                if verdict == "❌": t_bad[t] += 1
                elif verdict == "⚠️": t_warn[t] += 1
                elif verdict == "✅": t_good[t] += 1

    n = len(files) - len(no_codes)
    print(f"扫描 {len(files)} 个场次文档：编码有效 {n} 场（实例 {sum(inst_count.values())} 次）；"
          f"问题索引有效 {len(files)-len(no_index)} 场（问题 {total_q} 条）。")
    if scores:
        print(f"平均分 {sum(scores)/len(scores):.1f}（{min(scores)}~{max(scores)}）")
    for lbl, lst in (("缺 减分点编码", no_codes), ("缺 全问题索引", no_index)):
        if lst:
            print(f"⚠ {lbl}：{'、'.join(x.replace('面试实录_','').replace('.md','') for x in lst)}")

    print("\n## ① 减分点编码排行榜\n")
    print("| 排名 | 编码 | 问题 | 出现场次 | 场次占比 | 总次数 |")
    print("|---|---|---|---|---|---|")
    ranked = sorted(sess_count, key=lambda c: (-len(sess_count[c]), -inst_count[c], c))
    for i, c in enumerate(ranked, 1):
        s = len(sess_count[c])
        print(f"| {i} | {c} | {NAMES.get(c,'⚠未登记，请在NAMES和总库§一补充')} | {s} | {s*100//max(n,1)}% | {inst_count[c]} |")
    unknown = [c for c in ranked if c not in NAMES]
    if unknown:
        print(f"\n⚠ 未登记编码 {unknown}")

    print("\n## ② 问题级主题排行榜（按被问场次排序）\n")
    print("| 排名 | 主题标签 | 被问场次 | 被问次数 | ❌ | ⚠️ | ✅ | 答好率 |")
    print("|---|---|---|---|---|---|---|---|")
    tr = sorted(t_sess, key=lambda t: (-len(t_sess[t]), -t_cnt[t], t))
    for i, t in enumerate(tr, 1):
        g, w, b = t_good[t], t_warn[t], t_bad[t]
        tot = g + w + b
        rate = f"{g*100//tot}%" if tot else "-"
        print(f"| {i} | {t} | {len(t_sess[t])} | {t_cnt[t]} | {b} | {w} | {g} | {rate} |")

if __name__ == "__main__":
    main()
