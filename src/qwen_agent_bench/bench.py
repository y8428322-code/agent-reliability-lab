"""评测床命令行入口。

用法（在项目根目录执行）：
    export PYTHONPATH=src
    .venv/Scripts/python.exe -m qwen_agent_bench.bench --variant A_baseline --trials 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 必须在 import qwen_agent 之前设置：上游 `qwen_agent/settings.py` 在**导入时**读取该变量，
# 用它限制一次 agent 运行内的 LLM 调用次数（默认 20）。
# 实测：模型在原生 function calling 下遇到工具报错会「扫射式」重试
# （单轮 87 次工具调用），不设上限会烧掉大量 token 且不产生正确结果。
os.environ.setdefault("QWEN_AGENT_MAX_LLM_CALL_PER_RUN", "12")

from qwen_agent_demo.config import load_app_config  # noqa: E402
from qwen_agent_demo.model_factory import MissingApiKeyError, build_llm  # noqa: E402

from .domain.loader import load_tasks  # noqa: E402
from .evaluator import TaskResult  # noqa: E402
from .runner import run_task  # noqa: E402
from .variants import VARIANTS, variant_uses_native  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "results"


def pass_at_k(results: List[TaskResult], k: int) -> float:
    """pass^k = C(success_count, k) / C(num_trials, k)。

    同一任务多次试验，只有 k 次全成功才计入分子。
    """
    by_task: Dict[str, List[bool]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r.success)
    if not by_task:
        return 0.0
    total = 0.0
    for outcomes in by_task.values():
        n = len(outcomes)
        s = sum(1 for x in outcomes if x)
        if n < k:
            continue
        total += math.comb(s, k) / math.comb(n, k)
    return total / len(by_task)


def summarize(results: List[TaskResult]) -> Dict[str, Any]:
    """按变体汇总。"""
    out: Dict[str, Any] = {}
    by_variant: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    for variant, items in by_variant.items():
        n = len(items)
        out[variant] = {
            "runs": n,
            "pass@1": sum(1 for r in items if r.success) / n if n else 0.0,
            "db_reward": sum(r.db_reward for r in items) / n if n else 0.0,
            "nl_reward": sum(r.nl_reward for r in items) / n if n else 0.0,
            "avg_tool_calls": sum(r.n_tool_calls for r in items) / n if n else 0.0,
            "avg_tool_errors": sum(len(r.tool_errors) for r in items) / n if n else 0.0,
            "failures": {},
        }
        for r in items:
            if not r.success:
                out[variant]["failures"][r.failure_mode] = out[variant]["failures"].get(r.failure_mode, 0) + 1
    return out


def print_report(results: List[TaskResult]) -> None:
    print("")
    print("=" * 96)
    print("%-8s %-14s %-8s %-8s %-8s %-8s %s" % ("任务", "变体", "DB", "NL", "总分", "轮数", "失败归因"))
    print("-" * 96)
    for r in results:
        print(
            "%-8s %-14s %-8.2f %-8.2f %-8.2f %-8d %s"
            % (
                r.task_id,
                r.variant,
                r.db_reward,
                r.nl_reward,
                r.reward,
                len(r.turns),
                r.failure_mode or "-",
            )
        )
    print("-" * 96)
    for variant in sorted({r.variant for r in results}):
        items = [r for r in results if r.variant == variant]
        trials = len({r.trial for r in items})
        judge_err = sum(
            1 for r in items for d in r.nl_details if d.get("passed") is None
        )
        print(
            "变体 %-14s pass@1=%.3f  pass^%d=%.3f  db=%.3f  nl=%.3f  平均工具调用=%.1f  平均工具报错=%.2f  判定异常=%d"
            % (
                variant,
                sum(1 for r in items if r.success) / len(items),
                trials,
                pass_at_k(items, trials),
                sum(r.db_reward for r in items) / len(items),
                sum(r.nl_reward for r in items) / len(items),
                sum(r.n_tool_calls for r in items) / len(items),
                sum(len(r.tool_errors) for r in items) / len(items),
                judge_err,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="自建 agent 评测床")
    parser.add_argument("--variant", action="append", choices=VARIANTS, help="要跑的变体，可重复；默认全部")
    parser.add_argument("--task", action="append", help="只跑指定任务 id，可重复")
    parser.add_argument("--trials", type=int, default=1, help="每个任务的重复试验次数")
    parser.add_argument("--no-nl", action="store_true", help="跳过 NL 断言判定（只判 DB，速度快）")
    parser.add_argument("--out", default="", help="结果 JSON 输出路径")
    args = parser.parse_args()

    variants = args.variant or VARIANTS
    tasks = load_tasks()
    if args.task:
        wanted = set(args.task)
        tasks = [t for t in tasks if t["id"] in wanted]

    app_cfg = load_app_config()
    try:
        llm_textual = build_llm(app_cfg.model, use_raw_api=False)
        llm_native = build_llm(app_cfg.model, use_raw_api=True)
    except MissingApiKeyError as exc:
        raise SystemExit(str(exc))

    llm_by_mode = {"native": llm_native, "textual": llm_textual}

    print("模型: %s   端点: %s" % (app_cfg.model.model, app_cfg.model.base_url))
    print("变体: %s   任务数: %d   重复: %d   NL 判定: %s" % (variants, len(tasks), args.trials, not args.no_nl))

    results: List[TaskResult] = []
    for variant in variants:
        llm = llm_by_mode["native" if variant_uses_native(variant) else "textual"]
        for task in tasks:
            for trial in range(1, args.trials + 1):
                res = run_task(
                    task,
                    variant,
                    llm,
                    app_cfg.model,
                    trial=trial,
                    use_nl_judge=not args.no_nl,
                )
                results.append(res)
                if res.aborted:
                    # 中断 = 基础设施故障，不是模型答错；单独计数，不进正确率
                    print(
                        "[ABRT] %s %s trial=%d  <%s>"
                        % (variant, task["id"], trial, res.error)
                    )
                    continue
                flag = "OK " if res.success else "FAIL"
                print(
                    "[%s] %s %s trial=%d db=%.0f nl=%.2f tool=%d err=%d%s"
                    % (
                        flag,
                        variant,
                        task["id"],
                        trial,
                        res.db_reward,
                        res.nl_reward,
                        res.n_tool_calls,
                        len(res.tool_errors),
                        ("  <%s>" % res.error) if res.error else "",
                    )
                )

    print_report(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT_DIR / "last_run.json"
    payload = {
        "model": app_cfg.model.model,
        "base_url": app_cfg.model.base_url,
        "summary": summarize(results),
        "results": [
            {
                "task_id": r.task_id,
                "variant": r.variant,
                "trial": r.trial,
                "success": r.success,
                "db_reward": r.db_reward,
                "nl_reward": r.nl_reward,
                "reward": r.reward,
                "gold_hash": r.gold_hash,
                "pred_hash": r.pred_hash,
                "failure_mode": r.failure_mode,
                "n_tool_calls": r.n_tool_calls,
                "tool_errors": r.tool_errors,
                "audit": r.audit,
                "aborted": r.aborted,
                "runaway": r.runaway,
                "runaway_turns": r.runaway_turns,
                "nl_details": r.nl_details,
                "transcript": r.transcript,
                "turns": [
                    {
                        "user": t.user,
                        "assistant_text": t.assistant_text,
                        "tool_calls": t.tool_calls,
                        "tool_results": t.tool_results,
                    }
                    for t in r.turns
                ],
                "error": r.error,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n结果已写入: %s" % out_path)


if __name__ == "__main__":
    main()
