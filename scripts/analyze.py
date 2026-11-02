"""分析评测结果 JSON，输出消融表格。

用法：
    .venv/Scripts/python.exe scripts/analyze.py results/ablation_v3.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ORDER = [
    "A_baseline",
    "A_native",
    "B_context",
    "B_handwritten",
    "B_native",
    "C_reflection",
    "D_guarded",
    "E_guarded_reflect",
]


def pass_at_k(outcomes: List[bool], k: int) -> float:
    """pass^k = C(success, k) / C(trials, k)。"""
    n = len(outcomes)
    if n < k:
        return float("nan")
    s = sum(1 for x in outcomes if x)
    return math.comb(s, k) / math.comb(n, k)


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ablation_v3.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]

    variants = [v for v in ORDER if any(r["variant"] == v for r in results)]
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)

    tasks: List[str] = []
    for r in results:
        if r["task_id"] not in tasks:
            tasks.append(r["task_id"])

    print("=" * 110)
    print("变体总览   模型=%s   运行数=%d   任务数=%d" % (data.get("model"), len(results), len(tasks)))
    print("=" * 110)
    header = "%-20s %-8s %-9s %-8s %-8s %-10s %-9s %s" % (
        "变体", "运行数", "pass@1", "pass^3", "DB", "NL", "工具调用", "判定异常")
    print(header)
    print("-" * 110)
    for v in variants:
        items = by_variant[v]
        trials = len({r["trial"] for r in items})
        per_task = defaultdict(list)
        for r in items:
            per_task[r["task_id"]].append(r["success"])
        p3 = sum(pass_at_k(o, 3) for o in per_task.values() if len(o) >= 3)
        p3 = p3 / max(1, sum(1 for o in per_task.values() if len(o) >= 3))
        judge_err = sum(1 for r in items for d in r.get("nl_details", []) if d.get("passed") is None)
        print(
            "%-20s %-8d %-9.3f %-8.3f %-8.3f %-10.3f %-9.2f %d"
            % (
                v,
                len(items),
                sum(1 for r in items if r["success"]) / len(items),
                p3,
                sum(r["db_reward"] for r in items) / len(items),
                sum(r["nl_reward"] for r in items) / len(items),
                sum(r["n_tool_calls"] for r in items) / len(items),
                judge_err,
            )
        )

    print()
    print("逐任务 pass 率（每次试验的成功次数/总次数）")
    print("-" * 110)
    print("%-8s %s" % ("任务", " ".join("%-14s" % v for v in variants)))
    for t in tasks:
        cells = []
        for v in variants:
            items = [r for r in by_variant[v] if r["task_id"] == t]
            if not items:
                cells.append("%-14s" % "-")
                continue
            ok = sum(1 for r in items if r["success"])
            mark = "" if ok == len(items) else "  <-"
            cells.append("%-14s" % ("%d/%d%s" % (ok, len(items), mark)))
        print("%-8s %s" % (t, " ".join(cells)))

    print()
    print("失败归因分布（动作级）")
    print("-" * 110)
    for v in variants:
        items = by_variant[v]
        fails = Counter(r["failure_mode"] for r in items if not r["success"])
        print("%-20s %s" % (v, dict(fails) if fails else "无失败"))

    print()
    print("工具调用与报错（**中位数为主**：均值会被失控运行拉偏）")
    print("-" * 110)
    print(
        "%-20s %-12s %-12s %-12s %-14s %-16s %s"
        % ("变体", "中位工具调用", "平均工具调用", "P90", "失控运行数", "格式错误运行数", "有工具报错运行数")
    )
    for v in variants:
        items = by_variant[v]
        tc = sorted(r["n_tool_calls"] for r in items)
        med = tc[len(tc) // 2]
        p90 = tc[int(len(tc) * 0.9)]
        runaway = sum(1 for r in items if r.get("runaway"))
        fmt = sum(1 for r in items if r.get("format_error"))
        print(
            "%-20s %-12.0f %-12.2f %-12.0f %-14s %-16s %d/%d"
            % (
                v,
                med,
                sum(tc) / len(tc),
                p90,
                "%d/%d" % (runaway, len(items)),
                "%d/%d" % (fmt, len(items)),
                sum(1 for r in items if r.get("tool_errors")),
                len(items),
            )
        )

    # 任务区分度：**不同变体在该任务上结果不同**的任务占比
    discriminating = []
    for t in tasks:
        rates = []
        for v in variants:
            items = [r for r in by_variant[v] if r["task_id"] == t]
            if items:
                rates.append(round(sum(1 for r in items if r["success"]) / len(items), 3))
        if len(set(rates)) > 1:
            discriminating.append(t)
    print()
    print("任务区分度：%d/%d 个任务能区分变体（各变体在该任务上的通过率不全相同）" % (len(discriminating), len(tasks)))
    print("  -> %s" % (", ".join(discriminating) or "无"))

    # 配对差异：同一任务上逐对比较，比总体 pass^3 灵敏得多
    # （总体粒度受限于任务数 1/17 ≈ 0.059，配对粒度是 1 个任务）
    print()
    print("配对差异（同一任务上谁赢谁输；> 表示左边更好）")
    print("-" * 110)
    pairs = [
        ("A_baseline", "B_crop"),
        ("A_baseline", "B_points"),
        ("A_baseline", "B_context"),
        ("B_crop", "B_context"),
        ("B_points", "B_context"),
        ("B_context", "B_handwritten"),
    ]
    print("%-34s %-10s %-10s %s" % ("对比", "左赢", "右赢", "净"))
    for left, right in pairs:
        if left not in by_variant or right not in by_variant:
            continue
        left_only = right_only = 0
        for t in tasks:
            li = [r for r in by_variant[left] if r["task_id"] == t]
            ri = [r for r in by_variant[right] if r["task_id"] == t]
            if not li or not ri:
                continue
            l_rate = sum(1 for r in li if r["success"]) / len(li)
            r_rate = sum(1 for r in ri if r["success"]) / len(ri)
            if l_rate > r_rate:
                left_only += 1
            elif r_rate > l_rate:
                right_only += 1
        print(
            "%-34s %-10d %-10d %+d"
            % ("%s vs %s" % (left, right), left_only, right_only, left_only - right_only)
        )


if __name__ == "__main__":
    main()
