"""Summarize the closed-loop repair experiment from its recorded artifacts."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import sleep
from zoneinfo import ZoneInfo


def json_file(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def json_lines(path):
    if not path.is_file():
        return []
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(("\n", "\r"))]


def csv_file(path):
    if not path.is_file():
        return []
    with path.open(newline="") as source:
        return [row for row in csv.DictReader(source) if None not in row.values()]


def verdict(rows):
    if not rows:
        return "待输出"
    n = len(rows)
    success = sum(r["success"] == "True" for r in rows)
    p, z = success/n, 1.95996398454
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    width = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return f"{success}/{n}（{p:.1%}，Wilson 95% {max(0,center-width):.1%}–{min(1,center+width):.1%}）"


def evaluate_rows(run, folder, cohort=None):
    data = csv_file(run/folder/"episodes.csv")
    return [row for row in data if cohort is None or row.get("cohort") == cohort]


def navigation_row(run, name, aligned, offset):
    values = [verdict(evaluate_rows(run, aligned))]
    values.extend(verdict(evaluate_rows(run, offset, c)) for c in ("in_view", "out_of_view"))
    return f"| {name} | " + " | ".join(values) + " |"


def report(run, original):
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
    previous = f"../{original.name}"
    repaired = run/"repaired"
    audit = json_file(run/"action-audit.json")
    gates = json_file(repaired/"gate-decision.json")
    decision = json_file(repaired/"early-decision.json")
    data = json_file(run/"mix/summary.json")
    final = bool(gates and (not gates["formal_permitted"] or decision))
    lines = [
        "# 阶段 3 闭环控制修复与 30 格复核",
        "",
        f"北京时间 {now}。报告状态为{'已完成门槛或正式判定' if final else '运行中'}。",
        "",
        "## 五项诊断",
        "",
    ]
    if audit:
        counts = audit["train_roundtrip"]
        copy = audit["validation_previous_action"]
        lines.extend([
            f"- **相机映射**。{counts['actions']:,} 条冻结训练回程动作，"
            f"token 不一致 {counts['token_mismatches']}，环境量化动作不一致 {counts['environment_mismatches']}。"
            f"yaw 第 {audit['center_yaw_bin']} 档为 0°。原始连续角度的舍入归入量化误差。 [动作核验](action-audit.json)",
            f"- **复制上一步**。{copy['copy_exact']:,}/{copy['actions']:,}"
            f"（{copy['copy_exact']/copy['actions']:.2%}）整条完全匹配。"
            f"动作变化子集有 {copy['changed']} 条，返回首步复制为"
            f" {copy['initial_copy_exact']}/{copy['initial_return_actions']}。",
        ])
    aligned_cpu = json_lines(run/"cpu-aligned-probe/start-probes.jsonl")
    if aligned_cpu:
        sample = aligned_cpu[0]
        lines.extend([
            "",
            f"**首步输入逐 token 比对**。失败种子 `{sample['seed']}` 的"
            f"输入差异字段为 `{', '.join(sample['differing_input_fields'])}`。"
            "固定同一画面，只改历史动作可检验首步生成是否受历史影响。 "
            f"[完整输入](cpu-aligned-probe/seed{sample['seed']}-input-diff.json)",
            "",
            "| 首步条件 | 前进 | yaw | 教师目标匹配 |",
            "| --- | ---: | ---: | --- |",
        ])
        for key, value in sample["controls"].items():
            lines.append(f"| `{key}` | {value['forward']} | {value['yaw']}° | {'是' if value['exact'] else '否'} |")
        lines.extend([
            "",
            "同一首帧的干预支持动作历史依赖。JPEG 对照保留了原先的错误动作。"
            " [动作 token](cpu-aligned-probe/start-probes.jsonl)",
        ])
    starts = json_lines(run/"input-and-history-probes/start-probes.jsonl")
    if starts:
        lines.extend(["", f"**偏角首步干预**已记录 {len(starts)}/20 个场景。",
                      " [逐场首步干预](input-and-history-probes/start-probes.jsonl)", "",
                      "| 条件 | 合计精确匹配 | 视野内 | 视野外 |",
                      "| --- | ---: | ---: | ---: |"])
        groups = [starts, *[[r for r in starts if r["cohort"] == c] for c in ("in_view", "out_of_view")]]
        for name in starts[0]["controls"]:
            values = [f"{sum(r['controls'][name]['exact'] for r in group)}/{len(group)}" for group in groups]
            lines.append(f"| `{name}` | " + " | ".join(values) + " |")
    traces = sorted((run/"no-memory-step-traces").glob("*-control.jsonl"))
    lines.extend(["", f"**故障逐步记录**已产出 {len(traces)}/6 个回合。"])
    if traces:
        lines.extend(["", "| 回合 | 控制步 | 首步前进 | 首步 yaw | 相邻生成 token 重复 |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for path in traces:
            trace = json_lines(path)
            if trace:
                first = trace[0]
                repeats = sum(left["raw_token_ids"] == right["raw_token_ids"]
                              for left, right in zip(trace, trace[1:]))
                lines.append(f"| [{path.stem}](no-memory-step-traces/{path.name}) | {len(trace)} | "
                             f"{first['buttons']['forward']} | {first['yaw']}° | {repeats}/{len(trace)-1} |")
    lines.extend([
        "", "## 原权重闭环结果", "",
        "| 模型 | 5 格正前方 | 偏角视野内 | 偏角视野外 |",
        "| --- | --- | --- | --- |",
        navigation_row(run, "无记忆", f"{previous}/eval-no_memory-nav", f"{previous}/eval-no_memory-nav_offset"),
        navigation_row(run, "空间记忆", "current-spatial-nav", "current-spatial-nav_offset"),
        "", "**30 格诊断，原权重导航门槛未通过**。每组计划 30 个冻结场景，GPU 7 每批推进两个实例。",
        "", "| 模型 | 诊断成功 | 死亡 | 位姿跳变 |",
        "| --- | --- | ---: | ---: |",
    ])
    for name in ("no_memory", "spatial"):
        folder = f"diagnostic-d30-{name}"
        rows = evaluate_rows(run, folder)
        assert all(r["mode"] == "diagnostic_d30" for r in rows)
        label = f"[`{name}`]({folder}/episodes.csv)" if rows else f"`{name}`"
        lines.append(f"| {label} | {verdict(rows)} | {sum(int(r['deaths']) for r in rows)} | "
                     f"{sum(int(r['pose_jumps']) for r in rows)} |")
    lines.extend([
        "", "## DART 与同设置继续训练", "",
        "两组从原末步适配器初始化，空间记忆组同时加载空间模块。"
        "训练时以 0.5 概率使用统一历史占位，评测固定使用占位。"
        "扰动动作与教师纠正动作分别记录，混合索引使用两来源各 6,000 条回程监督。",
    ])
    if data:
        lines.extend([
            "",
            f"实际采集 **{data['dart_raw_frames']:,} 帧**，冻结 **{sum(data['dart_selected_frames'].values()):,} 帧**。"
            f"训练混合索引的场景、世界及冻结评测交集均为 0。"
            f"索引 SHA-256 为 `{data['mixed_index_sha256']}`。 [数据与种子核验](mix/summary.json)",
        ])
    lines.extend(["", "| 模型 | 优化步 | 最近 100 步平均训练损失 |",
                  "| --- | ---: | ---: |"])
    for model in ("no_memory", "spatial"):
        train = csv_file(repaired/f"train-{model}/loss.csv")
        update = train[-1]["update"] if train else "待启动"
        loss = f"{mean(float(r['loss']) for r in train[-100:]):.5f}" if train else "待输出"
        lines.append(f"| `{model}` | {update} | {loss} |")
    lines.extend(["", "| 模型 | 验证步 | 损失 | token 准确率 | 完整动作 | 动作变化匹配 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for model in ("no_memory", "spatial"):
        for row in csv_file(repaired/f"train-{model}/val_loss.csv"):
            change = row["changed_action_exact_accuracy"]
            suffix = f"{float(change):.2%}（n={row['changed_actions']}）" if change else "无变化样本"
            lines.append(f"| `{model}` | {row['update']} | {float(row['loss']):.5f} | "
                         f"{float(row['token_accuracy']):.2%} | {float(row['exact_action_accuracy']):.2%} | {suffix} |")
    lines.extend([
        "", "## 修复后门槛与正式结果", "",
        "| 模型 | 5 格正前方 | 偏角视野内 | 偏角视野外 |",
        "| --- | --- | --- | --- |",
    ])
    for model in ("no_memory", "spatial"):
        lines.append(navigation_row(repaired, model, f"eval-{model}-nav", f"eval-{model}-nav_offset"))
    lines.append("")
    if gates:
        for model, state in gates["variants"].items():
            condition = "通过" if state["passed"] else "，".join(state["blockers"])
            lines.append(f"- `{model}` 门槛结果为 {condition}。")
    else:
        lines.append("等待两组各完成 ≥8/10 原版及 ≥12/15 偏角视野内的门槛核验。偏角还需比恒定前进多成功至少 3 回合。")
    if decision:
        lines.extend(["", f"正式 30 格继续判断为 **{'继续' if decision['proceed'] else '止损'}**。"])
        for model in ("no_memory", "spatial"):
            folder = f"eval-{model}-d30"
            rows = evaluate_rows(repaired, folder)
            lines.append(f"- `{model}` 正式 30 格为 {verdict(rows)}。 [逐回合](repaired/{folder}/episodes.csv)")
    elif gates and not gates["formal_permitted"]:
        lines.extend(["", "两组的门槛未同时达到，正式 30 格未启动。"])
    else:
        lines.extend(["", "正式 30 格继续或止损条件仍待测量。"])
    return "\n".join(lines) + "\n", final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    destination = args.run/"control-repair-report.md"
    while True:
        document, completed = report(args.run, args.original)
        staged = destination.with_suffix(".md.tmp")
        staged.write_text(document)
        staged.replace(destination)
        if completed or not args.watch:
            return
        sleep(60)


if __name__ == "__main__":
    main()
