"""agent 策略变体（消融实验的自变量）。

变体阶梯（每一步只加一个东西，便于归因）：

- ``A_baseline``    原始 policy 全文 + 默认工具描述
- ``A_native``      同 A，但走上游的**原生** function calling（隔离「调用协议」这一个变量）
- ``B_context``     结构化/裁剪后的 policy + 改写过的工具描述（上下文工程）
- ``B_native``      同 B，但走原生 function calling
- ``C_reflection``  同 B + Reflexion 式自省反思（**单向**：只问「还有什么没做」）
- ``D_guarded``     同 B + **减法机制**（工具级调用前置条件 + 系统级动作边界）
- ``E_guarded_reflect``  同 D + **双向反思**（同时问「该做的没做」和「不该做的做了」）

设计依据见 ``docs/实验报告-v2.md``：当前主失败模式是「做多了」，
因此 D/E 引入的是**减法**机制，而不是更多的补做清单。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qwen_agent.agents import Assistant
from qwen_agent.llm.base import BaseChatModel

from .compat import CompatAssistant
from .domain.db import MockDB
from .domain.loader import load_policy_digest
from .domain.render import load_policy_rules, render
from .domain.tools import apply_enhanced_descriptions, baseline_descriptions, build_tools

VARIANTS = [
    "A_baseline",         # 长叙述版 policy（全量、无要点化、无红线前置）
    "A_native",           # 同上，但走原生 function calling
    "B_crop",             # **只裁剪**（按场景），仍是长段落
    "B_points",           # **只要点化**（全量规则要点化），不裁剪、不红线前置
    "B_crop_points",      # 裁剪 + 要点化，**不**红线前置（用于单独隔离「裁剪」的效应）
    "B_points_redline",   # 要点化 + 红线前置，**不**裁剪（用于单独隔离「红线前置」的效应）
    "B_points_trunc",     # **等长对照**：全量要点化但截断到与裁剪版等长
    "B_context",          # 裁剪 + 要点化 + 红线前置（= 完整上下文工程）
    "B_handwritten",      # 同 B，但 policy 用**人工手写**的摘要 —— 与自动版对照
    "B_native",           # 同 B，走原生 function calling
    "C_reflection",       # 同 B + 单向自省反思
    "D_guarded",          # 同 B + 减法机制
    "E_guarded_reflect",  # 同 D + 双向自省反思
]

# 会写库的写操作工具（减法机制只对它们加约束）
WRITE_TOOL_NAMES = (
    "create_repair_ticket",
    "replace_device",
    "cancel_ticket",
    "offer_refund",
)

BASE_PERSONA = (
    "你是某电信运营商的在线客服坐席，正在通过在线对话为用户办理业务。\n"
    "你只能通过提供的工具查询和办理业务，不要凭空编造系统数据。\n"
    "所有面向用户的回复请使用中文。"
)

# ---------------------------------------------------------------- 减法机制
GUARD_SYSTEM = (
    "【动作边界（最高优先级，优先于下面所有业务规范）】\n"
    "1. 写操作工具（建单 / 换机 / 撤单 / 退款）**只能在用户明确要求该业务时**调用。"
    "用户没有明确要求的业务，一律不要代办。\n"
    "2. 一次只办理一个诉求。用户同时提出多个诉求（例如既说宽带断了、又说设备要换）时，"
    "只办理**第一个**，其余的先向用户说明并**等用户确认后**再办。\n"
    "3. 未经用户确认，不要执行任何写操作。\n"
    "4. 如果你认为某个诉求不该办理，说明原因即可，**不要先做了再解释**。"
)

# ---------------------------------------------------------------- 反思
_REFLECTION_ONESHOT = (
    "你是这次客服对话的质量检查员。下面给出系统业务规范和到目前为止的对话记录。\n"
    "请检查：业务规范中的哪些要求在当前对话里**还没有被满足**（例如该问的信息没问、"
    "该办的业务没办、该拒绝的请求没拒绝、该告知的关键信息没告知）。\n"
    "如果都已满足，只输出「无」。\n"
    "如果还有未满足项，用不超过 5 行中文列出待办事项，每行一条，不要复述对话。\n\n"
    "【业务规范】\n%s\n\n"
    "【对话记录】\n%s\n"
)

_REFLECTION_BIDIRECTIONAL = (
    "你是这次客服对话的质量检查员。下面给出系统业务规范和到目前为止的对话记录。\n"
    "请分别检查两个方向的问题：\n"
    "（一）**该做而没做**：业务规范里哪些要求还没被满足（例如该问的信息没问、"
    "该办的业务没办、该拒绝的请求没拒绝、该告知的关键信息没告知）。\n"
    "（二）**不该做而做了**：agent 是否执行了用户**没有明确要求**的写操作、"
    "或违反了「一次只办一件事」「不得越权办理」这类限制。"
    "特别注意：工具调用里出现的每一项写操作（建单、换机、撤单、退款），"
    "都要对照用户原话判断用户是否真的要求过。\n"
    "如果两个方向都没有问题，只输出「无」。\n"
    "有问题时用不超过 6 行中文列出，每行一条，以「漏：」或「多：」开头，不要复述对话。\n\n"
    "【业务规范】\n%s\n\n"
    "【对话记录】\n%s\n"
)


def variant_uses_digest(variant: str) -> bool:
    """是否使用结构化策略摘要（A_* 系列统一用原始策略，便于隔离变量）。"""
    return not variant.startswith("A_")


def variant_uses_native(variant: str) -> bool:
    """是否走上游的原生 function calling。"""
    return variant.endswith("_native")


def variant_uses_guard(variant: str) -> bool:
    """是否启用减法机制（D / E）。"""
    return variant.startswith("D_") or variant.startswith("E_")


def variant_uses_bidirectional_reflection(variant: str) -> bool:
    """是否使用双向反思（只有 E）。"""
    return variant.startswith("E_")


def variant_uses_reflection(variant: str) -> bool:
    """是否启用自省反思（C 单向 / E 双向）。"""
    return variant.startswith("C_") or variant.startswith("E_")


def apply_guard_descriptions(tools: List[Any]) -> None:
    """给写操作工具的描述追加**按工具定制**的「调用前置条件」（减法机制的工具级部分）。"""
    from .domain.tools import GUARD_LINES

    for tool in tools:
        line = GUARD_LINES.get(tool.name)
        if line and line not in tool.description:
            tool.description = tool.description + "\n" + line


def policy_for_variant(variant: str, task: Dict[str, Any]) -> str:
    """该变体使用的策略文本。

    **全部来自同一个事实源**（`domain/policy_rules.yaml`），只是三个正交开关的组合不同：

    | 变体 | 裁剪 | 要点化 | 红线前置 |
    |---|---|---|---|
    | `A_baseline` | ✗ | ✗ | ✗ |
    | `B_crop` | ✓ | ✗ | ✗ |
    | `B_points` | ✗ | ✓ | ✗ |
    | `B_context`（及 C/D/E） | ✓ | ✓ | ✓ |
    | `B_handwritten` | 例外：人工手写摘要 | — | — |

    反思提示必须与 agent 上下文里的策略**同源**——否则反思依据的规则
    和 agent 实际看到的规则不一致，反思结论可能指向 agent 并不在意的条款。
    """
    if variant == "B_handwritten":
        return load_policy_digest()

    spec = load_policy_rules()
    scenes = [s for s in (task.get("policy_sections") or []) if s != "redline"]
    if variant == "B_crop":
        return render(spec, scenes=scenes, points=False, redline_first=False)
    if variant == "B_points":
        return render(spec, scenes=None, points=True, redline_first=False)
    if variant == "B_crop_points":
        return render(spec, scenes=scenes, points=True, redline_first=False)
    if variant == "B_points_redline":
        return render(spec, scenes=None, points=True, redline_first=True)
    if variant == "B_points_trunc":
        # **等长对照**：全量要点化版本，但按字符数截断到与裁剪版等长。
        # 用来区分「裁剪掉的内容缺失」与「上下文变短本身」两种解释。
        full = render(spec, scenes=None, points=True, redline_first=False)
        cropped = render(spec, scenes=scenes, points=True, redline_first=False)
        return full[: len(cropped)]
    if variant_uses_digest(variant):
        return render(spec, scenes=scenes, points=True, redline_first=True)
    return render(spec, scenes=None, points=False, redline_first=False)


def build_context(variant: str, task: Dict[str, Any], reflection: str = "") -> str:
    """构造 system message（上下文）。

    减法约束放在**最前面**（业务规范之前）—— 实测放在末尾时模型基本会忽略。
    """
    policy = policy_for_variant(variant, task)

    parts = [BASE_PERSONA]
    if variant_uses_guard(variant):
        parts.extend(["", GUARD_SYSTEM])
    parts.extend(["", "【业务规范】", policy])
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
    if variant_uses_guard(variant):
        apply_guard_descriptions(tools)

    system = build_context(variant, task, reflection)
    agent = CompatAssistant(
        llm=llm,
        name="telecom_cs",
        description="电信客服坐席",
        system_message=system,
        function_list=tools,
    )
    return agent, tools


def reflection_prompt(variant: str, policy_text: str, transcript: str) -> str:
    """构造自省反思的 prompt（C 单向 / E 双向）。"""
    template = (
        _REFLECTION_BIDIRECTIONAL
        if variant_uses_bidirectional_reflection(variant)
        else _REFLECTION_ONESHOT
    )
    return template % (policy_text, transcript)
