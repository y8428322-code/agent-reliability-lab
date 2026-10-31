"""本地工具的单元测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_agent_demo.tools import (  # noqa: E402
    SumTool,
    TextStatsTool,
    default_tools,
    tool_specs,
)


class SumToolTest(unittest.TestCase):
    def test_sum(self):
        result = json.loads(SumTool().call(json.dumps({"numbers": [1, 2, 3]})))
        self.assertEqual(result["sum"], 6)
        self.assertEqual(result["count"], 3)

    def test_sum_with_float(self):
        result = json.loads(SumTool().call(json.dumps({"numbers": [1.5, 2.5]})))
        self.assertAlmostEqual(result["sum"], 4.0)

    def test_invalid_input(self):
        result = json.loads(SumTool().call(json.dumps({"numbers": "abc"})))
        self.assertIn("error", result)

    def test_missing_required_param(self):
        with self.assertRaises(ValueError):
            SumTool().call(json.dumps({}))


class TextStatsToolTest(unittest.TestCase):
    def test_stats(self):
        text = "海上升明月\n天涯共此时"
        result = json.loads(TextStatsTool().call(json.dumps({"text": text})))
        self.assertEqual(result["characters"], len(text))
        self.assertEqual(result["lines"], 2)
        self.assertEqual(result["words"], 2)

    def test_empty_text(self):
        result = json.loads(TextStatsTool().call(json.dumps({"text": ""})))
        self.assertEqual(result["characters"], 0)
        self.assertEqual(result["lines"], 0)

    def test_wrong_type(self):
        result = json.loads(TextStatsTool().call(json.dumps({"text": 123})))
        self.assertIn("error", result)


class ToolSpecTest(unittest.TestCase):
    def test_default_tools(self):
        names = [t.name for t in default_tools()]
        self.assertEqual(names, ["sum_numbers", "text_stats"])

    def test_tool_specs_are_openai_compatible(self):
        for spec in tool_specs():
            self.assertEqual(spec["type"], "function")
            func = spec["function"]
            self.assertIn("name", func)
            self.assertIn("description", func)
            self.assertEqual(func["parameters"]["type"], "object")
            self.assertTrue(func["parameters"]["required"])

    def test_tools_are_registered_in_global_registry(self):
        """GroupChat 的成员工具只能按名字从全局 TOOL_REGISTRY 查找，必须已注册。"""
        from qwen_agent.tools.base import TOOL_REGISTRY

        self.assertIn("sum_numbers", TOOL_REGISTRY)
        self.assertIn("text_stats", TOOL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
