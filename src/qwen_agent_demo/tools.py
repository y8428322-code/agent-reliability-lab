"""演示用的本地确定性工具（无外部依赖，可单元测试）。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("sum_numbers", allow_overwrite=True)
class SumTool(BaseTool):
    """对一组数字求和。"""

    name = "sum_numbers"
    description = "对给定的一组数字求和并返回结果。numbers 是数字列表，例如 [1, 2, 3]。"
    parameters = [
        {
            "name": "numbers",
            "type": "array",
            "description": "需要求和的数字列表。",
            "required": True,
        },
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        numbers = args.get("numbers") or []
        if not isinstance(numbers, list):
            return json.dumps({"error": "numbers 必须是列表"}, ensure_ascii=False)
        try:
            total = sum(float(x) for x in numbers)
        except (TypeError, ValueError):
            return json.dumps({"error": "numbers 含有非数值元素"}, ensure_ascii=False)
        return json.dumps({"count": len(numbers), "sum": total}, ensure_ascii=False)


@register_tool("text_stats", allow_overwrite=True)
class TextStatsTool(BaseTool):
    """统计一段文本的字符数、词数与行数。"""

    name = "text_stats"
    description = "统计文本的字符数、空白分隔词数与行数。text 是待统计的文本。"
    parameters = [
        {
            "name": "text",
            "type": "string",
            "description": "待统计的文本内容。",
            "required": True,
        },
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        text = args.get("text")
        if not isinstance(text, str):
            return json.dumps({"error": "text 必须是字符串"}, ensure_ascii=False)
        return json.dumps(
            {
                "characters": len(text),
                "words": len(text.split()),
                "lines": len(text.splitlines()) if text else 0,
            },
            ensure_ascii=False,
        )


def default_tools() -> List[BaseTool]:
    """返回演示用的工具实例列表。"""
    return [SumTool(), TextStatsTool()]


def tool_specs() -> List[Dict[str, Any]]:
    """返回工具的 OpenAI 兼容 schema，用于协议级冒烟测试。"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p["name"]: {"type": p["type"], "description": p["description"]}
                        for p in tool.parameters
                    },
                    "required": [p["name"] for p in tool.parameters if p.get("required")],
                },
            },
        }
        for tool in default_tools()
    ]
