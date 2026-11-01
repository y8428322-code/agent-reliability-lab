"""与上游 Qwen-Agent 的兼容层修复。

只改**我们自己的封装层**，不修改上游源码（`upstream/` 保持 pristine）。
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple

from qwen_agent.agents import Assistant


def repair_args(raw: Any) -> Any:
    """修复被上游拼接坏的工具参数。

    上游 `qwen_agent/llm/oai.py::_chat_stream` 假定模型服务把工具参数**分片流式**返回，
    于是用 `+=` 累加：

        full_tool_calls[-1]['function_call']['arguments'] += tc.function.arguments

    若某个服务在每个 chunk 里都返回**完整**的参数，累计结果就会变成
    `{"a":1}{"a":1}{"a":1}...` 这种非法 JSON，工具侧会报
    `Parameters must be formatted as a valid JSON!`。

    这里从可能被重复拼接的字符串里，取出**最后一个**可解析的完整 JSON 对象。
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw
    text = raw.strip()
    try:
        json.loads(text)
        return text
    except Exception:  # noqa: BLE001
        pass
    # 依次尝试每个 '{' 开头的后缀，取第一个能解析的（即最后一个完整对象）
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        candidate = text[idx:]
        try:
            json.loads(candidate)
            return candidate
        except Exception:  # noqa: BLE001
            continue
    return text


class CompatAssistant(Assistant):
    """在调用工具前修复上游拼接坏的参数。

    只影响"参数被拼接成非法 JSON"这一条路径；正常路径完全等价。
    """

    def _call_tool(self, tool_name: str, tool_args: Any = "{}", **kwargs) -> str:
        try:
            repaired = repair_args(tool_args)
        except Exception:  # noqa: BLE001
            repaired = tool_args
        return super()._call_tool(tool_name, repaired, **kwargs)
