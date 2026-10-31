"""端到端演示：Function Calling / ReAct / 多 Agent 编排。

用法（项目根目录执行）：
    set PYTHONPATH=src && .venv\\Scripts\\python.exe -m qwen_agent_demo.demo
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from qwen_agent.agents import Assistant, GroupChat, ReActChat

from .config import AppConfig, load_app_config
from .model_factory import MissingApiKeyError, build_llm, check_endpoint
from .tools import default_tools


def _to_dict(msg: Any) -> Dict[str, Any]:
    """上游 Message 是 pydantic 模型（BaseModelCompatibleDict），统一转 dict 再读。"""
    return msg if isinstance(msg, dict) else msg.model_dump()


def _text_of(content: Any) -> str:
    """content 可能是 str 或多模态 ContentItem 列表。"""
    if isinstance(content, list):
        return " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return str(content or "")


def run_and_report(bot, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """跑一轮对话并输出：工具调用轨迹 + 工具返回 + 最终回答。

    上游是流式返回：
      - assistant 内容是**累积文本**，每轮 yield 都是当前全文；
      - function_call 的 arguments 也是**分片增量**（"{" -> '{"numbers"' -> 完整 JSON）。
    因此这里按工具名合并分片（保留最长的参数），最后统一打印。
    """
    call_order: List[str] = []
    call_args: Dict[str, str] = {}
    tool_results: List[Any] = []
    final_text = ""
    seen_results = set()
    react_actions: List[str] = []

    for response in bot.run(messages=messages):
        for msg in response:
            data = _to_dict(msg)
            role = data.get("role", "?")

            call = data.get("function_call")
            if call:
                call_data = call if isinstance(call, dict) else call.model_dump()
                name = call_data.get("name")
                args = call_data.get("arguments") or ""
                if name not in call_args:
                    call_order.append(name)
                    call_args[name] = args
                elif len(args) > len(call_args[name]):
                    call_args[name] = args

            content = _text_of(data.get("content"))
            if role == "function" and data.get("name"):
                sig = (data.get("name"), content)
                if sig not in seen_results:
                    seen_results.add(sig)
                    tool_results.append(sig)
            elif role == "assistant" and content.strip():
                final_text = content
                # ReAct 风格（非原生 function calling）的工具调用写在正文里
                for line in content.splitlines():
                    if line.strip().startswith("Action:"):
                        act = line.strip()[len("Action:"):].strip()
                        if act and act not in react_actions:
                            react_actions.append(act)

    for name in call_order:
        print("  -> 调用工具 %s(%s)" % (name, call_args.get(name, "")))
    for act in react_actions:
        print("  -> (ReAct 文本动作) %s" % act.replace("\n", " ")[:120])
    for name, content in tool_results:
        print("  <- %s 返回: %s" % (name, content[:200]))

    # ReAct 场景的最终答案藏在 "Final Answer:" 之后
    if "Final Answer:" in final_text:
        final_text = final_text.split("Final Answer:")[-1].strip()

    print("  最终回答: %s" % (final_text.strip()[:400] or "(空)"))
    return {
        "tool_calls": [(n, call_args.get(n, "")) for n in call_order],
        "tool_results": tool_results,
        "final_text": final_text.strip(),
    }


def run_function_calling(llm) -> None:
    """场景一：Assistant + 本地工具，验证 function calling 链路。"""
    print("\n===== 场景一：Function Calling =====")
    bot = Assistant(
        llm=llm,
        name="计算助手",
        description="能调用本地工具完成数值与文本统计的助手",
        system_message=(
            "你是一个计算助手。需要求和时调用 sum_numbers，"
            "需要统计文本时调用 text_stats，然后用一句话给出结论。"
        ),
        function_list=default_tools(),
    )
    messages = [{"role": "user", "content": "请计算 12、30、8 这三个数的和，并用一句话告诉我结果。"}]
    run_and_report(bot, messages)


def run_react_chat(llm) -> None:
    """场景二：ReActChat，验证多轮推理 + 工具调用。"""
    print("\n===== 场景二：ReAct Chat =====")
    bot = ReActChat(
        llm=llm,
        name="文本分析助手",
        description="通过工具分步骤完成文本分析",
        system_message="你是一个文本分析助手，先规划再调用工具，最后总结。",
        function_list=default_tools(),
    )
    messages = [
        {"role": "user", "content": "统计下面这段文本的字符数、词数和行数：\n海上升明月\n天涯共此时"},
    ]
    run_and_report(bot, messages)


def run_group_chat(llm) -> None:
    """场景三：GroupChat，验证多 Agent 编排。"""
    print("\n===== 场景三：多 Agent 编排 =====")
    cfgs = {
        "background": "一个数据分析小组的群聊",
        "agents": [
            {
                "name": "计算员",
                "description": "负责数值计算，会使用 sum_numbers 工具",
                "system_message": "你是计算员，负责算数，需要时调用 sum_numbers 工具。",
                "selected_tools": ["sum_numbers"],
            },
            {
                "name": "文本员",
                "description": "负责文本统计，会使用 text_stats 工具",
                "system_message": "你是文本员，负责统计文本，需要时调用 text_stats 工具。",
                "selected_tools": ["text_stats"],
            },
        ],
    }
    bot = GroupChat(agents=cfgs, llm=llm, function_list=default_tools())
    messages = [{"role": "user", "content": "计算 5、6、7 的和。"}]
    run_and_report(bot, messages)


def main() -> None:
    cfg: AppConfig = load_app_config()
    print("模型: %s" % cfg.model.model)
    print("端点: %s" % cfg.model.base_url)

    try:
        tool_name = check_endpoint(cfg.model)
    except Exception as exc:  # noqa: BLE001 - 顶层演示，失败即退出
        raise SystemExit("协议级冒烟失败: %s" % exc)
    print("协议冒烟: 端点工具调用正常 (tool=%s)" % tool_name)

    try:
        llm = build_llm(cfg.model)
    except MissingApiKeyError as exc:
        raise SystemExit(str(exc))

    scenarios = {
        "function_calling": run_function_calling,
        "react_chat": run_react_chat,
        "group_chat": run_group_chat,
    }
    for name in cfg.demo.enabled_scenarios:
        scenarios[name](llm)

    print("\n全部场景执行完毕。")


if __name__ == "__main__":
    main()
