"""agent 策略变体（消融实验的自变量）。

- A_baseline ：朴素基线 —— 原始 policy 全文 + 默认工具描述
- B_context  ：上下文工程 —— 结构化/裁剪后的 policy + 改写过的工具描述
- C_reflection：B + Reflexion 式自省反思（反思结论注入下一轮）
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qwen_agent.agents import Assistant
from qwen_agent.llm.base import BaseChatModel

from .domain.db import MockDB
from .domain.loader import build_digest_context, load_policy_raw
from .domain.tools import apply_enhanced_descriptions, baseline_descriptions, build_tools

VARIANTS = ["A_baseline", "B_context", "C_reflection"]

BASE_PERSONA = (
    "你是某电信运营商的在线客服坐席，正在通过在线对话为用户办理业务。\n"
    "你只能通过提供的工具查询和办理业务，不要凭空编造系统数据。\n"
    "所有面向用户的回复请使用中文。"
)

_REFLECTION_PROMPT = (
    "你是这次客服对话的质量检查员。下面给出系统业务规范和到目前为止的对话记录。\n"
    "请检查：业务规范中的哪些要求在当前对话里**还没有被满足**（例如该问的信息没问、"
    "该办的业务没办、该拒绝的请求没拒绝、该告知的关键信息没告知）。\n"
    "如果都已满足，只输出「无」。\n"
    "如果还有未满足项，用不超过 5 行中文列出待办事项，每行一条，不要复述对话。\n\n"
    "【业务规范】\n%s\n\n"
    "【对话记录】\n%s\n"
)


def variant_uses_digest(variant: str) -> bool:
    """是否使用结构化策略摘要。"""
    return variant in ("B_context", "C_reflection")


def build_context(variant: str, task: Dict[str, Any], reflection: str = "") -> str:
    """构造 system message（上下文）。"""
    if variant_uses_digest(variant):
        policy = build_digest_context(task.get("policy_sections") or [])
    else:
        policy = load_policy_raw()

    parts = [BASE_PERSONA, "", "【业务规范】", policy]
    if reflection:
        parts.extend(["", "【上一轮的自查结论（请务必逐条落实）】", reflection])
    return "\n".join(parts)


def build_agent(
    variant: str,
    task: Dict[str, Any],
    db: MockDB,
    llm: BaseChatModel,
    reflection: str = "",
) -> Tuple[Assistant, List[Any]]:
    """按变体构造 agent 与工具实例。"""
    tools = build_tools(db)
    if variant_uses_digest(variant):
        apply_enhanced_descriptions(tools)
    else:
        baseline_descriptions(tools)

    system = build_context(variant, task, reflection)
    agent = Assistant(
        llm=llm,
        name="telecom_cs",
        description="电信客服坐席",
        system_message=system,
        function_list=tools,
    )
    return agent, tools


def reflection_prompt(policy_text: str, transcript: str) -> str:
    """构造自省反思的 prompt。"""
    return _REFLECTION_PROMPT % (policy_text, transcript)
