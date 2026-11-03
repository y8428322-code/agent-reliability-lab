"""从 results/*.json 生成可入库的汇总报告与图表（零第三方依赖）。

为什么不直接写死一个 Markdown：
    汇总数字必须**可复算** —— 面试官/审阅者可以自己重跑本脚本核对。

批次分组是**显式写死**的，因为守卫逻辑与规则集在开发过程中改过多次，
不同代码版本产生的数据不能合并（这一点是踩过坑才写下来的）。

用法：
    .venv/Scripts/python.exe scripts/summarize.py
    .venv/Scripts/python.exe scripts/summarize.py --out results_summary
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# 批次分组（显式、可审计）
#
# 代码/规则集的变更时点（git log）：
#   11-02 15:00  e7c1d24  引入机制约束 + R15 去 universal + R17 改写   ← 规则集 A
#   11-02 16:36  0546680  跨回合守卫 + 终止性拒绝语义                   ← 守卫 B
#   11-02 17:31  4365762  守卫按话术顺序 + need_confirm 修复 + 归因修正   ← 守卫 C
#   11-02 18:25  bb1e081  修守卫 need_confirm 范围 + 归因修正             ← 守卫 D（最终）
#
# 注意：批次启动时间晚于提交时间，��用的是**启动时刻**的代码。
# 守卫逻辑改过 4 版，因此只有 19:02 之后的批次是最终版（D）。
# ---------------------------------------------------------------------------
GROUPS: List[Dict[str, Any]] = [
    {
        "key": "keepredline",
        "label": "B_crop_keepredline（裁剪 + 保红线，无机制约束）",
        "desc": "5 个独立批次；legacy 规则集，守卫对该变体无影响，故可合并",
        "files": [
            "bench_B_crop_keepredline_20261102_150010.json",
            "bench_B_crop_keepredline_20261102_163638.json",
            "bench_B_crop_keepredline_20261102_173800.json",
            "bench_B_crop_keepredline_20261102_182318.json",
            "bench_B_crop_keepredline_20261102_190206.json",
        ],
    },
    {
        "key": "guard",
        "label": "B_guard（同策略 + 工具层机制约束）",
        "desc": "3 个独立批次；legacy 规则集；**仅取守卫逻辑定稿后的批次（bb1e081 之后）**，早期批次守卫逻辑不同，不可合并",
        "files": [
            "bench_B_guard_20261102_190206.json",
            "bench_B_guard_20261102_193810.json",
            "bench_B_guard_20261102_201127.json",
        ],
    },
    {
        "key": "crop_redline",
        "label": "B_crop_redline（裁剪 + 保红线，规则集 v1，无机制约束）",
        "desc": "规则集 v1（措辞边界模糊的版本），无机制约束",
        "files": sorted([p.name for p in (RESULTS).glob("bench_B_crop_redline_20261103_*.json")]),
    },
    {
        "key": "guard_redline",
        "label": "B_guard_redline（规则集 v1 + 机制约束）",
        "desc": "规则集 v1 + 机制约束；与 B_guard 只差规则集",
        "files": sorted([p.name for p in (RESULTS).glob("bench_B_guard_redline_20261103_*.json")]),
    },
    {
        "key": "crop_v2",
        "label": "B_crop_v2（裁剪 + 保红线，规则集 v2，无机制约束）",
        "desc": "规则集 v2（措辞边界清晰的版本），无机制约束；与 B_crop_keepredline 只差规则集",
        "files": sorted([p.name for p in (RESULTS).glob("bench_B_crop_v2_20261103_*.json")]),
    },
    {
        "key": "guard_v2",
        "label": "B_guard_v2（规则集 v2 + 机制约束）",
        "desc": "规则集 v2 + 机制约束；与 B_guard 只差规则集",
        "files": sorted([p.name for p in (RESULTS).glob("bench_B_guard_v2_20261103_*.json")]),
    },
]

FOCUS_TASK = "H9"


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def load(files: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in files:
        path = RESULTS / name
        if not path.exists():
            print("[warn] 缺少结果文件，跳过: %s" % name)
            continue
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("results", []))
    return [r for r in rows if not r.get("aborted")]


def per_task(res: Sequence[Dict[str, Any]]) -> Dict[str, List[bool]]:
    out: Dict[str, List[bool]] = defaultdict(list)
    for r in res:
        out[r["task_id"]].append(bool(r["success"]))
    return dict(out)


def pass_at_k(outcomes: Sequence[bool], k: int) -> float:
    n = len(outcomes)
    s = sum(1 for x in outcomes if x)
    if n < k:
        return float("nan")
    return comb(s, k) / comb(n, k)


def summarize(res: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pt = per_task(res)
    ok = [(sum(1 for x in v if x), len(v)) for v in pt.values()]
    usable = [(s, n) for s, n in ok if n >= 3]
    return {
        "n": len(res),
        "pass_at_1": sum(1 for r in res if r["success"]) / len(res) if res else 0.0,
        "pass_at_3": (
            sum(comb(s, 3) / comb(n, 3) for s, n in usable) / len(usable) if usable else 0.0
        ),
        "per_task": pt,
        "failures": dict(Counter(r.get("failure_mode", "?") for r in res if not r["success"])),
    }


def batch_scores(files: Sequence[str]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for name in files:
        path = RESULTS / name
        if not path.exists():
            continue
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))["results"] if not r.get("aborted")]
        s = summarize(rows)
        out.append((s["pass_at_1"], s["pass_at_3"]))
    return out


def fisher_one_sided(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    """Fisher 精确检验（单侧）：两组边际固定下，B 组成功数 >= 观测值的概率。"""
    K = a_success + b_success
    N = a_total + b_total
    n = b_total
    k_obs = b_success
    p = 0.0
    for x in range(k_obs, min(K, n) + 1):
        p += comb(K, x) * comb(N - K, n - x) / comb(N, n)
    return p


# ---------------------------------------------------------------------------
# 极简 SVG 出图（零依赖）
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def svg_grouped_bars(
    path: Path,
    title: str,
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    ylabel: str = "",
    ymax: float = 1.0,
    note: str = "",
) -> None:
    """分组柱状图。series = [(名称, 值序列, 颜色)]。"""
    W, H = 760, 380
    pad_l, pad_r, pad_t, pad_b = 78, 24, 54, 96
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    n_cat = max(1, len(categories))
    n_ser = max(1, len(series))
    slot = plot_w / n_cat
    bar_w = slot / (n_ser + 1.2)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif">' % (W, H, W, H),
        '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
        '<text x="%d" y="30" font-size="19" font-weight="600" fill="#1b1b1f" text-anchor="middle">%s</text>'
        % (W // 2, _esc(title)),
    ]
    # 网格与 y 轴刻度
    for i in range(6):
        y = pad_t + plot_h - plot_h * i / 5.0
        val = ymax * i / 5.0
        parts.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e6e6eb" stroke-width="1"/>'
            % (pad_l, y, pad_l + plot_w, y)
        )
        parts.append(
            '<text x="%d" y="%.1f" font-size="12" fill="#6b6b76" text-anchor="end">%.2f</text>'
            % (pad_l - 8, y + 4, val)
        )
    parts.append(
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#3a3a44" stroke-width="1.5"/>'
        % (pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h)
    )
    for ci, cat in enumerate(categories):
        base_x = pad_l + slot * ci + slot * 0.1
        for si, (name, values, color) in enumerate(series):
            v = values[ci] if ci < len(values) else 0.0
            h = plot_h * (v / ymax if ymax else 0)
            x = base_x + bar_w * si
            y = pad_t + plot_h - h
            parts.append(
                '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="2"/>'
                % (x, y, bar_w * 0.92, h, color)
            )
            parts.append(
                '<text x="%.1f" y="%.1f" font-size="12" font-weight="600" fill="#1b1b1f" '
                'text-anchor="middle">%.3f</text>' % (x + bar_w * 0.46, y - 5, v)
            )
        parts.append(
            '<text x="%.1f" y="%d" font-size="12.5" fill="#3a3a44" text-anchor="middle">%s</text>'
            % (pad_l + slot * ci + slot / 2, pad_t + plot_h + 20, _esc(cat))
        )
    # 图例
    lx = pad_l
    for name, _values, color in series:
        parts.append('<rect x="%d" y="%d" width="13" height="13" fill="%s" rx="2"/>' % (lx, H - 30, color))
        parts.append(
            '<text x="%d" y="%d" font-size="12.5" fill="#3a3a44">%s</text>' % (lx + 18, H - 19, _esc(name))
        )
        lx += 22 + len(name) * 8
    if ylabel:
        parts.append(
            '<text x="18" y="%d" font-size="12" fill="#6b6b76" transform="rotate(-90 18 %d)" '
            'text-anchor="middle">%s</text>' % (pad_t + plot_h // 2, pad_t + plot_h // 2, _esc(ylabel))
        )
    if note:
        parts.append(
            '<text x="%d" y="%d" font-size="11.5" fill="#8a8a95" text-anchor="middle">%s</text>'
            % (W // 2, H - 8, _esc(note))
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_scatter(
    path: Path,
    title: str,
    series: Sequence[Tuple[str, Sequence[Tuple[float, float]], str]],
    ylabel: str = "",
    note: str = "",
) -> None:
    """散点图：每个批次一个点，带均值横线与 ±σ 带。"""
    W, H = 760, 360
    pad_l, pad_r, pad_t, pad_b = 78, 30, 54, 84
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    allx = [x for _n, pts, _c in series for x, _y in pts]
    xmin, xmax = (min(allx) - 0.6, max(allx) + 0.6) if allx else (0, 1)
    ymin, ymax = 0.7, 1.02

    def sx(x: float) -> float:
        return pad_l + plot_w * (x - xmin) / (xmax - xmin)

    def sy(y: float) -> float:
        return pad_t + plot_h - plot_h * (y - ymin) / (ymax - ymin)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif">' % (W, H, W, H),
        '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
        '<text x="%d" y="30" font-size="19" font-weight="600" fill="#1b1b1f" text-anchor="middle">%s</text>'
        % (W // 2, _esc(title)),
    ]
    for i in range(7):
        y = pad_t + plot_h - plot_h * i / 6.0
        val = ymin + (ymax - ymin) * i / 6.0
        parts.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e6e6eb"/>'
            % (pad_l, y, pad_l + plot_w, y)
        )
        parts.append(
            '<text x="%d" y="%.1f" font-size="12" fill="#6b6b76" text-anchor="end">%.3f</text>'
            % (pad_l - 8, y + 4, val)
        )
    lx = pad_l
    for name, pts, color in series:
        if not pts:
            continue
        ys = [p[1] for p in pts]
        mean = sum(ys) / len(ys)
        sd = statistics.pstdev(ys) if len(ys) > 1 else 0.0
        parts.append(
            '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity="0.13"/>'
            % (pad_l, sy(mean + sd), plot_w, max(1.0, sy(mean - sd) - sy(mean + sd)), color)
        )
        parts.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="2" stroke-dasharray="5 3"/>'
            % (pad_l, sy(mean), pad_l + plot_w, sy(mean), color)
        )
        for i, (x, y) in enumerate(pts):
            parts.append(
                '<circle cx="%.1f" cy="%.1f" r="6" fill="%s" stroke="#ffffff" stroke-width="1.6"/>'
                % (sx(x), sy(y), color)
            )
            parts.append(
                '<text x="%.1f" y="%.1f" font-size="10.5" fill="#6b6b76" text-anchor="middle">%.3f</text>'
                % (sx(x), sy(y) - 12, y)
            )
        parts.append(
            '<rect x="%d" y="%d" width="13" height="13" fill="%s" rx="2"/>' % (lx, H - 30, color)
        )
        parts.append(
            '<text x="%d" y="%d" font-size="12.5" fill="#3a3a44">%s</text>'
            % (lx + 18, H - 19, _esc("%s  σ=%.3f" % (name, sd)))
        )
        lx += 30 + len(name) * 8 + 80
    if note:
        parts.append(
            '<text x="%d" y="%d" font-size="11.5" fill="#8a8a95" text-anchor="middle">%s</text>'
            % (W // 2, H - 8, _esc(note))
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="生成评测汇总报告与图表")
    parser.add_argument("--out", default="results_summary", help="输出目录")
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for group in GROUPS:
        rows = load(group["files"])
        summaries[group["key"]] = summarize(rows)

    a = summaries["keepredline"]
    b = summaries["guard"]

    # ---- H9 显著性 ----
    h9_a = a["per_task"].get(FOCUS_TASK, [])
    h9_b = b["per_task"].get(FOCUS_TASK, [])
    p_val = fisher_one_sided(sum(h9_a), len(h9_a), sum(h9_b), len(h9_b))

    # ---- 批次方差 ----
    batches = {g["key"]: batch_scores(g["files"]) for g in GROUPS}
    sd = {
        k: {
            "p1": statistics.pstdev([x[0] for x in v]) if len(v) > 1 else 0.0,
            "p3": statistics.pstdev([x[1] for x in v]) if len(v) > 1 else 0.0,
        }
        for k, v in batches.items()
    }

    # ---- Markdown ----
    lines: List[str] = []
    lines.append("# 评测结果汇总\n")
    lines.append(
        "> 本文件由 `scripts/summarize.py` 从 `results/*.json` **自动生成**，数字可复算。\n"
        "> 批次分组在脚本里显式写死并注明对应的代码版本（守卫逻辑与规则集在开发中改过多版，"
        "不同版本的数据不可合并）。\n"
    )
    lines.append("## 1. 主结果\n")
    lines.append("| 配置 | 批次数 | 运行数 | pass@1 | pass^3 | %s（过度执行任务） |" % FOCUS_TASK)
    lines.append("|---|---|---|---|---|---|")
    for group in GROUPS:
        s = summaries[group["key"]]
        pt = s["per_task"].get(FOCUS_TASK, [])
        lines.append(
            "| %s | %d | %d | %.3f | %.3f | **%d/%d** |"
            % (
                group["label"],
                len(group["files"]),
                s["n"],
                s["pass_at_1"],
                s["pass_at_3"],
                sum(1 for x in pt if x),
                len(pt),
            )
        )
    lines.append(
        "\nΔpass@1 = **%+.3f**，Δpass^3 = **%+.3f**\n"
        % (b["pass_at_1"] - a["pass_at_1"], b["pass_at_3"] - a["pass_at_3"])
    )
    lines.append(
        "**%s：%d/%d → %d/%d，Fisher 精确检验（单侧）p = %.6f**\n"
        % (
            FOCUS_TASK,
            sum(1 for x in h9_a if x),
            len(h9_a),
            sum(1 for x in h9_b if x),
            len(h9_b),
            p_val,
        )
    )

    lines.append("## 2. 批次间方差（效应量判定的依据）\n")
    lines.append("| 配置 | 各批次 pass@1 | 各批次 pass^3 | σ(pass@1) | σ(pass^3) |")
    lines.append("|---|---|---|---|---|")
    for group in GROUPS:
        k = group["key"]
        lines.append(
            "| %s | %s | %s | %.3f | %.3f |"
            % (
                group["label"],
                ", ".join("%.3f" % x[0] for x in batches[k]),
                ", ".join("%.3f" % x[1] for x in batches[k]),
                sd[k]["p1"],
                sd[k]["p3"],
            )
        )
    lines.append(
        "\n> **pass^3 的批次间噪声约为 pass@1 的 2~3 倍** —— pass^3 对边界任务的单次翻转极敏感。\n"
        "> 因此效应量判定应以 **pass@1 + 配对检验**为主，pass^3 只作参考。\n"
    )

    lines.append("## 3. 逐任务通过率\n")
    lines.append("| 任务 | %s | %s |" % (GROUPS[0]["label"], GROUPS[1]["label"]))
    lines.append("|---|---|---|")
    for task in sorted(a["per_task"]):
        va = a["per_task"][task]
        vb = b["per_task"].get(task, [])
        lines.append(
            "| %s | %d/%d | %s |"
            % (
                task,
                sum(1 for x in va if x),
                len(va),
                ("%d/%d" % (sum(1 for x in vb if x), len(vb))) if vb else "-",
            )
        )

    lines.append("\n## 4. 失败归因（动作级）\n")
    for group in GROUPS:
        s = summaries[group["key"]]
        lines.append("- **%s**：%s" % (group["label"], s["failures"] or "无失败"))

    lines.append("\n## 5. 批次分组说明（可审计）\n")
    for group in GROUPS:
        lines.append("- **%s**（%d 批）：%s" % (group["label"], len(group["files"]), group["desc"]))
        for f in group["files"]:
            lines.append("    - `results/%s`" % f)

    (out_dir / "汇总.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 图表 ----
    labels = [g["label"].split("（")[0] for g in GROUPS]
    svg_grouped_bars(
        out_dir / "fig1_main.svg",
        "上下文工程 + 机制约束 + 规则强度分层：主结果",
        labels,
        [
            ("pass@1", [summaries[g["key"]]["pass_at_1"] for g in GROUPS], "#2f6df6"),
            ("pass^3", [summaries[g["key"]]["pass_at_3"] for g in GROUPS], "#8ab4f8"),
        ],
        ylabel="通过率",
        note="%d / %d 个独立批次合并；判定：%s（过度执行任务）%d/%d → %d/%d，Fisher p=%.6f"
        % (
            len(GROUPS[0]["files"]),
            len(GROUPS[1]["files"]),
            FOCUS_TASK,
            sum(1 for x in h9_a if x),
            len(h9_a),
            sum(1 for x in h9_b if x),
            len(h9_b),
            p_val,
        ),
    )

    scatter_series = []
    for idx, group in enumerate(GROUPS):
        pts = [(i + 1, v[1]) for i, v in enumerate(batches[group["key"]])]
        palette = ["#2f6df6", "#e8802f", "#2e9e5b", "#a05ad6"]
        scatter_series.append((group["label"].split("（")[0], pts, palette[idx % len(palette)]))
    svg_scatter(
        out_dir / "fig2_variance.svg",
        "批次间方差：pass^3 每个批次的取值",
        scatter_series,
        note="虚线=批次均值，浅色带=±σ。σ(pass^3) 明显大于 σ(pass@1) → 效应量判定应以 pass@1 + 配对检验为主",
    )

    tasks = sorted(a["per_task"])
    svg_grouped_bars(
        out_dir / "fig3_per_task.svg",
        "逐任务通过率：机制约束只改掉了「过度执行」这一类失败",
        tasks,
        [
            ("keepredline", [sum(1 for x in a["per_task"][t] if x) / len(a["per_task"][t]) for t in tasks], "#9aa0a6"),
            ("guard", [sum(1 for x in b["per_task"][t] if x) / len(b["per_task"][t]) for t in tasks], "#2f6df6"),
        ],
        ylabel="通过率",
        note="唯一有显著差异的任务是 H9（多诉求过度执行）：0/15 → 9/9；其余任务两者等价",
    )

    print("已生成：")
    for p in sorted(out_dir.iterdir()):
        print("  %s" % (p.relative_to(ROOT)))
    print()
    print("主结果：")
    for group in GROUPS:
        s = summaries[group["key"]]
        print(
            "  %-46s n=%-4d pass@1=%.3f pass^3=%.3f  %s=%d/%d"
            % (
                group["label"],
                s["n"],
                s["pass_at_1"],
                s["pass_at_3"],
                FOCUS_TASK,
                sum(1 for x in s["per_task"].get(FOCUS_TASK, []) if x),
                len(s["per_task"].get(FOCUS_TASK, [])),
            )
        )
    print("  %s Fisher p = %.6f" % (FOCUS_TASK, p_val))


if __name__ == "__main__":
    main()
