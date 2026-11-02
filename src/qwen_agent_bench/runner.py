"""任务运行器：脚本化用户 + Qwen-Agent 会话 + 判分。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from qwen_agent_demo.model_factory import chat, normalize_args as _normalize_args

from .domain.db import MockDB
from .domain.tools import TurnGuard
from .domain.loader import load_policy_raw
from .evaluator import (
    TaskResult,
    TurnRecord,
    attribute_failure,
    db_reward,
    judge_all,
    unexecuted_broken_calls_for_turn,
)
from .variants import (
    build_agent,
    build_context,
    policy_for_variant,
    reflection_prompt,
    variant_uses_mechanism_guard,
    variant_uses_reflection,
)

# 单轮工具调用数超过该阈值即视为「失控」（模型陷入重复生成）
RUNAWAY_TOOL_CALL_THRESHOLD = 15


def _to_dict(msg: Any) -> Dict[str, Any]:
    return msg if isinstance(msg, dict) else msg.model_dump()


def collect_turn(batches: List[List[Any]]) -> Tuple[List[Dict[str, str]], List[Tuple[str, str]], str]:
    """把 agent.run() 的流式产出整理成一条完整回合。

    返回 (工具调用列表, 工具返回列表, 最终文本)。

    四个坑（实测踩过）：
      1. 上游是 `yield response + output`，**累积消息会被反复吐出**（同一个 function_call
         消息在每个流式 chunk 里重复出现），必须按唯一键去重；
      2. 上游的 `extra.function_id` **在多轮之间会重号**（每轮从 1 重新计数），
         不能用它做跨轮次的唯一键；
      3. 上游默认走 Nous **文本式** function calling（一次调用一条完整消息）；
         原生流式模式下同一次调用的 arguments 是**分片增量**，
         因此用「同名且参数为严格前缀延长」判定为同一次调用的分片，否则算新调用；
      4. 模型偶尔吐出**多余右括号**的 JSON 参数，去重键必须先规范化。

    去重规则：按唯一键保留**最后一次出现**（内容类消息保留最后一次累积值）。
    """
    calls: List[Dict[str, str]] = []
    call_index: Dict[Tuple[str, str], int] = {}
    results: List[Tuple[str, str]] = []
    result_index: Dict[Tuple[str, str], int] = {}
    final_text = ""

    for batch in batches:
        for msg in batch:
            data = _to_dict(msg)

            fc = data.get("function_call")
            if fc:
                cd = fc if isinstance(fc, dict) else fc.model_dump()
                name = cd.get("name")
                args = cd.get("arguments") or ""
                key = (name, _normalize_args(args))
                # 同名 + 参数为严格前缀延长 => 同一次调用在流式下的分片增量
                merged = False
                for back in range(len(calls) - 1, -1, -1):
                    prev = calls[back]
                    if prev["name"] != name:
                        continue
                    if len(args) > len(prev["raw"]) and args.startswith(prev["raw"]):
                        # 参数被补全 -> 去重索引的键也要同步，否则后续同一条消息
                        # 会因为「旧键已失效」被当成新调用重复计入
                        call_index.pop((prev["name"], prev["key"]), None)
                        prev["raw"] = args
                        prev["key"] = _normalize_args(args)
                        prev["arguments"] = prev["key"] or args
                        call_index[(name, prev["key"])] = back
                        merged = True
                    break
                if merged:
                    continue
                # 完全相同的 (name, 规范化参数) 是上游重复吐出的同一条消息，去重
                if key in call_index:
                    continue
                call_index[key] = len(calls)
                calls.append(
                    {
                        "name": name,
                        "arguments": _normalize_args(args) or args,
                        "raw": args,
                        "key": _normalize_args(args),
                    }
                )
                continue

            role = data.get("role")
            if role == "function":
                key = (data.get("name"), data.get("content") or "")
                idx = result_index.get(key)
                if idx is None:
                    result_index[key] = len(results)
                    results.append(key)
                continue

            if role == "assistant":
                content = data.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        str(c.get("text", "")) for c in content if isinstance(c, dict)
                    )
                if content.strip():
                    final_text = content

    return calls, results, final_text


def _history_entries(
    calls: List[Dict[str, str]],
    results: List[Tuple[str, str]],
    final_text: str,
) -> List[Dict[str, Any]]:
    """把一回合还原成可继续对话的消息序列（格式对齐上游内部约定）。

    function_id 由我们自己按顺序分配，保证 tool_call 与 tool 消息的 id 一一对应
    （上游的多轮重号会导致 OpenAI 侧 tool/tool_call id 不匹配）。
    """
    entries: List[Dict[str, Any]] = []
    for idx, call in enumerate(calls):
        entries.append(
            {
                "role": "assistant",
                "content": "",
                "function_call": {"name": call["name"], "arguments": call["arguments"]},
                "extra": {"function_id": str(idx + 1)},
            }
        )
    for idx, (name, content) in enumerate(results):
        entries.append(
            {
                "role": "function",
                "name": name,
                "content": content,
                "extra": {"function_id": str(idx + 1)},
            }
        )
    if final_text:
        entries.append({"role": "assistant", "content": final_text})
    return entries


def run_task(
    task: Dict[str, Any],
    variant: str,
    llm,
    cfg,
    trial: int = 1,
    use_nl_judge: bool = True,
) -> TaskResult:
    """跑一个任务的一次试验。"""
    result = TaskResult(task_id=task["id"], variant=variant, trial=trial)

    db = MockDB()
    # 机制约束（工具层）：同一回合只准一次写操作。守卫在整轮任务内共享，每回合重置。
    guard = TurnGuard() if variant_uses_mechanism_guard(variant) else None
    try:
        agent, _tools = build_agent(variant, task, db, llm, guard=guard)
    except Exception as exc:  # noqa: BLE001
        result.error = "构造 agent 失败: %s" % exc
        return result

    user_turns: List[str] = task.get("user_turns") or ["你好"]
    max_turns = min(int(task.get("max_turns") or len(user_turns)), len(user_turns))

    history: List[Dict[str, Any]] = []
    reflection = ""

    for turn_index in range(max_turns):
        user_msg = user_turns[turn_index]
        history.append({"role": "user", "content": user_msg})
        if guard is not None:
            guard.begin_turn(user_msg)

        if variant_uses_reflection(variant) and reflection:
            # 反思结论注入下一轮：重建 agent（system_message 变化），历史通过 messages 传入
            agent, _tools = build_agent(
                variant, task, db, llm, reflection=reflection, guard=guard
            )
            reflection = ""

        batches: List[List[Any]] = []
        try:
            for batch in agent.run(messages=list(history)):
                batches.append(batch)
        except Exception as exc:  # noqa: BLE001
            result.error = "第 %d 轮执行异常: %s" % (turn_index + 1, exc)
            # **中断的运行不是"任务失败"** —— 那是基础设施故障（限流/超时），
            # 计入正确率会系统性压低分数。必须单独统计并排除。
            result.aborted = True
            break

        result.n_llm_calls += 1
        calls, results, final_text = collect_turn(batches)

        # 失控检测：模型陷入重复生成时，单轮工具调用会异常膨胀。
        # 标记出来而不是让它污染均值（v3 曾出现一次 137 次调用的离群运行）。
        if len(calls) > RUNAWAY_TOOL_CALL_THRESHOLD:
            result.runaway = True
            result.runaway_turns.append(turn_index + 1)

        record = TurnRecord(user=user_msg, assistant_text=final_text)
        for call in calls:
            record.tool_calls.append((call["name"], call["arguments"]))
            result.n_tool_calls += 1

        # 工具调用格式错误：模型写了 <tool_call> 但格式坏了（未闭合/含零宽字符），
        # 上游解析失败 -> 工具没执行，残缺文本被原样吐给用户。
        # 这是**独立于成败的质量指标**（上游缺陷），失败时它就是失败原因。
        broken = unexecuted_broken_calls_for_turn(record)
        if broken:
            result.format_error = True
            for name in broken:
                if name not in result.broken_calls:
                    result.broken_calls.append(name)
        for name, content in results:
            record.tool_results.append((name, content))
            try:
                parsed = json.loads(content)
            except Exception:  # noqa: BLE001
                parsed = None
            if isinstance(parsed, dict) and parsed.get("ok") is False:
                message = "%s: %s" % (name, parsed.get("error"))
                if "终止性拒绝" in str(parsed.get("error", "")):
                    # 机制拒绝**不是**参数错误，单独统计，避免误归因
                    result.guard_rejections.append(message)
                else:
                    result.tool_errors.append(message)
        result.turns.append(record)

        history.extend(_history_entries(calls, results, final_text))

        # C / E 变体：回合结束后做一次自省，为下一轮提供待办（E 为双向）
        if variant_uses_reflection(variant) and turn_index < max_turns - 1:
            transcript = "\n".join(
                "用户: %s\n客服: %s" % (t.user, t.assistant_text) for t in result.turns
            )
            try:
                reflection = chat(
                    cfg,
                    [
                        {
                            "role": "user",
                            "content": reflection_prompt(
                                variant, policy_for_variant(variant, task), transcript
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
                result.n_llm_calls += 1
            except Exception as exc:  # noqa: BLE001
                result.tool_errors.append("反思失败: %s" % exc)
                reflection = ""

    # ---- 判分 ----
    try:
        d_reward, g_hash, p_hash = db_reward(task, db)
    except Exception as exc:  # noqa: BLE001
        result.error = "DB 判分失败: %s" % exc
        result.aborted = True
        return result
    result.db_reward = d_reward
    result.gold_hash = g_hash
    result.pred_hash = p_hash

    if use_nl_judge:
        try:
            n_reward, details = judge_all(cfg, task, result.transcript)
        except Exception as exc:  # noqa: BLE001
            n_reward, details = 0.0, [{"error": str(exc)}]
        result.nl_reward = n_reward
        result.nl_details = details

    result.audit = list(db.audit_log)
    result.reward = result.db_reward * result.nl_reward
    result.success = result.reward >= 1.0
    result.failure_mode = attribute_failure(task, result)
    return result
