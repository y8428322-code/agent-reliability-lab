"""判分器：DB 终态比对（客观）+ 自然语言断言（LLM 判定）。

借鉴 τ²-bench 的 EnvironmentEvaluator：
  参考轨迹回放到干净 DB 得到目标终态；agent 的写操作发生在同一份 DB 上得到预测终态；
  比对状态哈希。等价终态即通过，不要求调用序列一致。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qwen_agent_demo.model_factory import chat

from .domain.db import MockDB
from .domain.tools import build_tools


@dataclass
class TurnRecord:
    """一轮对话记录。"""

    user: str = ""
    assistant_text: str = ""
    tool_calls: List[Tuple[str, str]] = field(default_factory=list)
    tool_results: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class TaskResult:
    """一个任务的一次试验结果。"""

    task_id: str = ""
    variant: str = ""
    trial: int = 1
    success: bool = False
    db_reward: float = 0.0
    nl_reward: float = 0.0
    reward: float = 0.0
    gold_hash: str = ""
    pred_hash: str = ""
    turns: List[TurnRecord] = field(default_factory=list)
    nl_details: List[Dict[str, Any]] = field(default_factory=list)
    tool_errors: List[str] = field(default_factory=list)
    failure_mode: str = ""
    reflection: str = ""
    n_llm_calls: int = 0
    n_tool_calls: int = 0
    error: str = ""

    @property
    def assistant_text(self) -> str:
        return "\n".join(t.assistant_text for t in self.turns if t.assistant_text)

    @property
    def transcript(self) -> str:
        """完整对话记录（用于 LLM 判定）。"""
        lines: List[str] = []
        for turn in self.turns:
            lines.append("用户: %s" % turn.user)
            for name, args in turn.tool_calls:
                lines.append("客服调用工具 %s(%s)" % (name, args))
            for name, content in turn.tool_results:
                lines.append("工具 %s 返回: %s" % (name, content[:200]))
            if turn.assistant_text:
                lines.append("客服: %s" % turn.assistant_text)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# DB 判分
# --------------------------------------------------------------------------
# 会改变 DB 状态的工具（写操作）。gold 回放时只有这些失败才说明任务定义有 bug；
# 读操作失败（例如用户先报错了手机号）是任务刻意设计的一部分，不算错误。
WRITE_TOOLS = ("create_repair_ticket", "replace_device", "cancel_ticket", "offer_refund")


def replay_actions(db: MockDB, actions: List[Dict[str, Any]]) -> List[str]:
    """在给定 DB 上回放一组工具调用，返回逐条对应的错误列表（无错误的位置为空串）。"""
    tools = {t.name: t for t in build_tools(db)}
    errors: List[str] = []
    for action in actions:
        name = action.get("name")
        args = action.get("arguments") or {}
        tool = tools.get(name)
        if tool is None:
            errors.append("未知工具: %s" % name)
            continue
        out = tool.call(json.dumps(args, ensure_ascii=False))
        try:
            parsed = json.loads(out)
        except Exception:  # noqa: BLE001
            errors.append("%s 返回非 JSON: %s" % (name, out[:120]))
            continue
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            errors.append("%s 失败: %s" % (name, parsed.get("error")))
        else:
            errors.append("")
    return errors


def gold_hash(task: Dict[str, Any]) -> Tuple[str, List[str]]:
    """回放参考轨迹，得到目标 DB 哈希。

    只有**写操作**失败才说明任务定义有 bug；读操作失败是任务刻意设计（例如
    用户第一次把手机号报错），不报错。
    """
    db = MockDB()
    actions = task.get("reference_actions") or []
    errors = replay_actions(db, actions)
    write_errors = [
        err
        for action, err in zip(actions, errors)
        if err and action.get("name") in WRITE_TOOLS
    ]
    return db.hash(), write_errors


def db_reward(task: Dict[str, Any], pred_db: MockDB) -> Tuple[float, str, str]:
    """DB 维度奖励。返回 (reward, gold_hash, pred_hash)。"""
    g_hash, gold_errors = gold_hash(task)
    if gold_errors:
        # 参考轨迹本身有问题 —— 评测集有 bug，必须暴露而不是静默判失败
        raise RuntimeError("任务 %s 的参考轨迹执行失败: %s" % (task["id"], "; ".join(gold_errors)))
    p_hash = pred_db.hash()
    return (1.0 if g_hash == p_hash else 0.0), g_hash, p_hash


# --------------------------------------------------------------------------
# 自然语言断言判定
# --------------------------------------------------------------------------
_TRUE_RE = re.compile(r"^\s*(true|yes|是|满足|通过|符合)\b", re.IGNORECASE)
_FALSE_RE = re.compile(r"^\s*(false|no|否|不满足|不通过|不符合)\b", re.IGNORECASE)


def judge_assertion(cfg, assertion: str, transcript: str) -> Tuple[Optional[bool], str]:
    """用模型判定一条自然语言断言是否被满足。

    返回 (True/False/None, 错误信息)。None = 判定本身失败，**不等于断言不成立**。
    区分二者很重要：否则端点抖动会被误算成模型能力问题。

    注意 max_tokens 不能太小：MiniMax 会先输出 reasoning_content，
    预算太小会被思考过程吃光导致正文为空（实测踩过）。
    """
    prompt = (
        "你是一个严格的评测员。下面给出一段客服与用户的完整对话记录，以及一条断言。\n"
        "请判断对话记录**是否满足**这条断言。只输出 true 或 false，不要输出任何其他内容。\n\n"
        "【对话记录】\n%s\n\n"
        "【断言】\n%s\n" % (transcript, assertion)
    )
    try:
        answer = chat(cfg, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
    except Exception as exc:  # noqa: BLE001
        return None, "调用失败: %s" % exc
    answer = answer.strip()
    if _TRUE_RE.match(answer):
        return True, ""
    if _FALSE_RE.match(answer):
        return False, ""
    return None, "无法识别的判定输出: %r" % answer[:120]


def judge_all(cfg, task: Dict[str, Any], transcript: str) -> Tuple[float, List[Dict[str, Any]]]:
    """判定任务的所有 NL 断言。返回 (通过比例, 明细)。"""
    assertions = task.get("nl_assertions") or []
    if not assertions:
        return 1.0, []
    details: List[Dict[str, Any]] = []
    passed = 0
    for assertion in assertions:
        ok, err = judge_assertion(cfg, assertion, transcript)
        if ok is None:
            ok, err = judge_assertion(cfg, assertion, transcript)  # 重试一次
        details.append({"assertion": assertion, "passed": ok, "error": err})
        if ok:
            passed += 1
    return passed / float(len(assertions)), details


# --------------------------------------------------------------------------
# 失败模式归因
# --------------------------------------------------------------------------
def attribute_failure(task: Dict[str, Any], result: TaskResult) -> str:
    """规则化失败归因（先用确定性规则，规则无法判定时返回 unknown）。"""
    called = [name for turn in result.turns for name, _ in turn.tool_calls]
    gold_calls = [a["name"] for a in (task.get("reference_actions") or [])]

    if result.tool_errors:
        return "参数错"
    if result.db_reward == 1.0 and result.nl_reward < 1.0:
        return "未闭环"
    if result.db_reward == 0.0:
        write_calls = [c for c in called if c in ("create_repair_ticket", "replace_device", "cancel_ticket", "offer_refund")]
        gold_writes = [c for c in gold_calls if c in ("create_repair_ticket", "replace_device", "cancel_ticket", "offer_refund")]
        if gold_writes and not write_calls:
            return "该办未办"
        if write_calls and not gold_writes:
            return "不该办却办了"
        return "状态不符"
    return "unknown"
