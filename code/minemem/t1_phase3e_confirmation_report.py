"""Render the revised Phase 3e confirmation from archived machine-readable results."""

import argparse
import json
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def counts(score):
    return f"{score['successes']}/{score['n']}"


def interval(bounds):
    return " 至 ".join(f"{value:.1%}" for value in bounds)


def arm(score):
    return (f"{counts(score)} [{interval(score['wilson95'])}]，选对 {score['room_correct']}，"
            f"干扰柱 {score['distractor_first']}，超时 {score['budget']}")


def pct(numerator, denominator):
    return f"{numerator}/{denominator} ({numerator/denominator:.1%})" if denominator else "0/0（无适用步）"


def report(main, generalization):
    conclusion = read(main / "confirmation-decision.json", {})
    decision = read(main / "main-decision.json")
    plan = read(main / "seed-plan.json")
    cpu = read(main / "frozen/metrics.json")["distances"]
    lines = [
        "# 方案 P 原 v3 主确认与布景泛化",
        "",
        "主确认沿用 `c09b0c31` 预注册的原 v3 场景，四档各 30 个种子。60 格的唯一缺口使用原种子补跑，"
        "地面准备局部清除残留砾石。CPU 基线作为实测对照列入同表。采用这一规则是因为原 10% 门槛来自单目标布景，"
        "而主比较在同一规划器和控制器下交换视觉记忆输入。放宽前的原始验收记录另行归档。 "
        "[原确认报告](REPORT_phase3e_confirmation.md) · [布景与砾石核查](REPORT_phase3e_scene_repair.md)",
        "",
        "## A. 主确认",
        "",
        "方括号内为 Wilson 95% 区间。每组同时报告选对围挡、先到干扰柱与超时的绝对场数。",
        "",
        "| 格数 | 原视觉 | 打乱视觉 | 教师 | 转 180° | RGB 扫视 | RGB 找结构 | 真值位姿折返 | 回到起点 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in (10, 30, 60, 100):
        result = read(main / f"A-{d}.json", {"groups": {}})
        groups = result["groups"]
        learning = [arm(groups[name]) if name in groups else "待完成" for name in ("visual", "shuffle_visual")]
        scripts = [counts(cpu[str(d)][name]) for name in
                   ("teacher", "turn_180", "scan", "find_structure", "reverse_route", "start_pose")]
        lines.append("| " + " | ".join([str(d), *learning, *scripts]) + " |")
    lines.extend(("", f"[脚本逐场结果]({main.name}/frozen/acceptance.csv) · "
                      f"[CPU 指标区间]({main.name}/frozen/metrics.json)", ""))
    if decision:
        gap = decision["visual_minus_best_script"]
        lo, hi = [100 * value for value in gap["bootstrap95"]]
        verdict = "通过" if decision["passed"] else "未通过"
        lines.extend((f"30 格主判定**{verdict}**。原视觉 {decision['visual_successes']}/30，"
                      f"比打乱视觉多 {decision['visual_minus_shuffled']} 场，"
                      f"比最好的无记忆脚本 RGB 找结构（{decision['best_script_successes']}/30）多 {gap['gap']} 场。"
                      f"在同一种子上配对的成功率差为 {100*gap['gap']/gap['n']:.1f} 个百分点，"
                      f"95% 区间为 {lo:.1f} 至 {hi:.1f} 个百分点。"
                      "区间通过枚举配对百分位 bootstrap 分布得到。"
                      f"仅原视觉成功 {gap['visual_only']} 场，仅找结构成功 {gap['baseline_only']} 场。 "
                      f"[机器判定]({main.name}/main-decision.json)", ""))
    else:
        lines.extend(("30 格主判定等待原视觉与打乱视觉两组完成。", ""))
    longest = read(main / "A-100.json")
    if longest:
        footprints = [row for group in longest["groups"].values() for row in group["footprint"]]
        oversized = sum(not row["fits_within_128"] for row in footprints)
        lost = sum(row["discarded_cells"] for row in footprints)
        lines.extend((f"100 格两组共 {len(footprints)} 场，{oversized} 场的轨迹跨度超过 128 格，"
                      f"记录的丢弃格数为 {lost}。逐场 x、z 跨度见指标。 [100 格结果]({main.name}/A-100.json)", ""))
    lines.extend(("## B. 30 格消融与动作", "",
                  "| 消融 | 成功数及 Wilson 95% 区间 | 强制转向步 | 稳定器步 | 末 200 步航点跳变总数 |",
                  "| --- | --- | ---: | ---: | ---: |"))
    for key, label in (("b1", "B1 关闭强制转向"), ("b2", "B2 脚本控制器"), ("b3", "B3 真值航点")):
        score = read(main / f"B-{key}.json")
        row = (f"{counts(score)} [{interval(score['wilson95'])}] | {score['heading_guard_steps']} | "
               f"{score['stabilizer_steps']} | {score['waypoint_jumps_tail200']}") if score else "待完成 | — | — | —"
        lines.append(f"| {label} | {row} |")
    lines.extend(("", "| 主确认组 | 7B 提出前进 | 仅相机转动 | 其他 | 强制转向占比 | 覆盖时动作一致 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"))
    for d in (10, 30, 60, 100):
        result = read(main / f"A-{d}.json", {"groups": {}})
        for name, label in (("visual", "原视觉"), ("shuffle_visual", "打乱视觉")):
            if name not in result["groups"]:
                continue
            actions = result["groups"][name]["actions"]
            values = [pct(actions[k], actions["steps"]) for k in ("forward", "turn", "other", "heading_guard_steps")]
            values.append(pct(actions["guard_agrees"], actions["heading_guard_steps"]))
            lines.append("| " + " | ".join([f"{d} 格 {label}", *values]) + " |")
    if not any((main / f"A-{d}.json").is_file() for d in (10, 30, 60, 100)):
        lines.append("| 待完成 | — | — | — | — | — |")

    lines.extend(("", "## C. 30 格材料交换", ""))
    pairs = conclusion.get("C")
    if pairs:
        lines.extend(("单元格依次列出金柱在 0、1 侧时的预测围挡与成功判定。"
                      "两臂的采样位姿逐帧相同，原始记录含渲染散列。", "",
                      "| 种子 | 原视觉 | 打乱视觉 |", "| --- | --- | --- |"))
        for pair in pairs["pairs"]:
            cells = []
            for name in ("visual", "shuffle_visual"):
                values = pair["arms"][name]
                cells.append(" / ".join(f"{slot}（{'成功' if ok else reason}）" for slot, ok, reason in
                                         zip(values["predicted_slots"], values["success"], values["reasons"], strict=True)))
            lines.append(f"| {pair['seed']} | {cells[0]} | {cells[1]} |")
        lines.append("")
        for name, label in (("visual", "原视觉"), ("shuffle_visual", "打乱视觉")):
            row = pairs["summary"][name]
            lines.append(f"{label}双臂都成功 {row['both_success']}/{row['pairs']}，"
                         f"随材料交换翻转 {row['flips']}/{row['pairs']}。")
    else:
        lines.append("待 10 对双臂闭环结果汇总。")

    lines.extend(("", "## 第二版布景泛化", "",
                  "第二版在 30 格使用原有的第二版确认种子，原视觉与打乱视觉各 30 场。该结果单独解释布景变化下的表现。"))
    extra = read(generalization / "A-30.json")
    if extra:
        lines.extend(("", f"原视觉 {arm(extra['visual'])}。打乱视觉 {arm(extra['shuffle_visual'])}。 "
                          f"[逐组指标]({generalization.name}/A-30.json)"))
    else:
        lines.extend(("", "泛化闭环待完成。"))
    lines.extend(("", "## 冻结源码与记录", "",
                  f"规划器与控制链使用 `{plan['frozen_commit']}`，由 SHA-256 清单逐项验证。"
                  "7B、房间读出、稳定器和控制条件遵循冻结评测。布景泛化使用已归档的第二版场景。 "
                  f"[冻结源码校验]({main.name}/frozen-code.sha256) · [实验工具校验]({main.name}/harness.sha256) · "
                  f"[种子校验]({main.name}/seed-plan.sha256) · [GPU 前置记录]({main.name}/gpu-eligibility.json)", ""))
    if conclusion:
        lines.append(f"[完整机器汇总]({main.name}/confirmation-decision.json)")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("main", "generalization", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.write_text(report(args.main, args.generalization))
