"""评测床的单元测试（不调用模型）。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_agent_bench.bench import pass_at_k  # noqa: E402
from qwen_agent_bench.domain.db import MockDB  # noqa: E402
from qwen_agent_bench.domain.loader import (  # noqa: E402
    build_digest_context,
    load_tasks,
    split_policy_sections,
)
from qwen_agent_bench.domain.tools import build_tools  # noqa: E402
from qwen_agent_bench.evaluator import (  # noqa: E402
    TaskResult,
    db_reward,
    gold_hash,
    replay_actions,
)
from qwen_agent_bench.runner import _history_entries, collect_turn  # noqa: E402


def _msg(role, content="", function_call=None, extra=None):
    data = {"role": role, "content": content}
    if function_call is not None:
        data["function_call"] = function_call
    if extra is not None:
        data["extra"] = extra
    return data


class DbTest(unittest.TestCase):
    def test_hash_is_deterministic(self):
        a, b = MockDB(), MockDB()
        self.assertEqual(a.hash(), b.hash())

    def test_audit_log_not_hashed(self):
        db = MockDB()
        db.log("noop", "C1001", {})
        self.assertEqual(db.hash(), MockDB().hash())

    def test_ticket_id_is_deterministic(self):
        db = MockDB()
        first = db.create_ticket("C1001", "断网")
        second = db.create_ticket("C1002", "慢速")
        self.assertEqual(first, "TK0002")
        self.assertEqual(second, "TK0003")

    def test_refund_amount_normalized_to_float(self):
        db = MockDB()
        db.create_refund("C1001", 30, "重复扣费")
        self.assertEqual(db.refunds["RF0001"]["amount"], 30.0)


class ToolTest(unittest.TestCase):
    def setUp(self):
        self.db = MockDB()
        self.tools = {t.name: t for t in build_tools(self.db)}

    def test_find_customer(self):
        out = json.loads(self.tools["find_customer_by_phone"].call(json.dumps({"phone": "13800001111"})))
        self.assertTrue(out["ok"])
        self.assertEqual(out["customer_id"], "C1001")

    def test_find_customer_missing_returns_error(self):
        out = json.loads(self.tools["find_customer_by_phone"].call(json.dumps({"phone": "13000000000"})))
        self.assertFalse(out["ok"])
        self.assertIn("未找到", out["error"])

    def test_warranty_out_of_date(self):
        out = json.loads(self.tools["check_device_warranty"].call(json.dumps({"device_serial": "D002"})))
        self.assertFalse(out["in_warranty"])

    def test_warranty_in_date(self):
        out = json.loads(self.tools["check_device_warranty"].call(json.dumps({"device_serial": "D001"})))
        self.assertTrue(out["in_warranty"])

    def test_replace_device_rejected_out_of_warranty(self):
        out = json.loads(
            self.tools["replace_device"].call(
                json.dumps({"customer_id": "C1002", "device_serial": "D002", "new_device_serial": "N001"})
            )
        )
        self.assertFalse(out["ok"])
        self.assertEqual(self.db.customers["C1002"]["device_serial"], "D002")

    def test_refund_above_tool_hard_cap_rejected(self):
        out = json.loads(
            self.tools["offer_refund"].call(
                json.dumps({"customer_id": "C1001", "amount": 800, "reason": "设备故障"})
            )
        )
        self.assertFalse(out["ok"])
        self.assertEqual(self.db.refunds, {})

    def test_refund_below_tool_cap_allowed(self):
        """工具层只管硬上限，授权与否由 policy 约束 agent —— 所以 300 元能调通。"""
        out = json.loads(
            self.tools["offer_refund"].call(
                json.dumps({"customer_id": "C1001", "amount": 300, "reason": "设备故障"})
            )
        )
        self.assertTrue(out["ok"])
        self.assertEqual(len(self.db.refunds), 1)

    def test_cancel_ticket_only_when_not_dispatched(self):
        out = json.loads(self.tools["cancel_ticket"].call(json.dumps({"ticket_id": "TK0001", "reason": "用户取消"})))
        self.assertTrue(out["ok"])
        self.assertEqual(self.db.tickets["TK0001"]["status"], "已撤销")

    def test_cancel_dispatched_ticket_rejected(self):
        db = MockDB()
        db.create_ticket("C1001", "断网")  # 新建的是「已受理」，不是「未派工」
        tools = {t.name: t for t in build_tools(db)}
        out = json.loads(tools["cancel_ticket"].call(json.dumps({"ticket_id": "TK0002", "reason": "用户取消"})))
        self.assertFalse(out["ok"])

    def test_missing_required_param_raises(self):
        """字典型 parameters 走 jsonschema 校验，抛 ValidationError（不是 ValueError）。"""
        import jsonschema

        with self.assertRaises(jsonschema.ValidationError):
            self.tools["find_customer_by_phone"].call(json.dumps({}))


class TaskSetTest(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks()

    def test_tasks_have_required_fields(self):
        for task in self.tasks:
            self.assertIn("id", task)
            self.assertTrue(task["user_turns"])
            self.assertIn("reward_basis", task)
            self.assertTrue(task["reference_actions"])

    def test_reference_actions_all_succeed(self):
        for task in self.tasks:
            h, errors = gold_hash(task)
            self.assertEqual(errors, [], "任务 %s 参考轨迹有错: %s" % (task["id"], errors))
            self.assertTrue(h)

    def test_gold_hash_differs_between_write_and_no_write_tasks(self):
        hashes = {t["id"]: gold_hash(t)[0] for t in self.tasks}
        self.assertNotEqual(hashes["T01"], hashes["T03"])  # T01 建单, T03 不改状态
        self.assertNotEqual(hashes["T06"], hashes["T04"])  # T06 退款, T04 不改状态

    def test_db_reward_detects_wrong_action(self):
        task = [t for t in self.tasks if t["id"] == "T01"][0]
        good = MockDB()
        replay_actions(good, task["reference_actions"])
        reward, g, p = db_reward(task, good)
        self.assertEqual(reward, 1.0)
        self.assertEqual(g, p)

        bad = MockDB()
        replay_actions(bad, [{"name": "create_repair_ticket", "arguments": {"customer_id": "C1002", "fault_type": "断网"}}])
        reward, g, p = db_reward(task, bad)
        self.assertEqual(reward, 0.0)
        self.assertNotEqual(g, p)

    def test_db_reward_detects_extra_write(self):
        """越权多写一步 -> 状态多出记录 -> 判失败。"""
        task = [t for t in self.tasks if t["id"] == "T04"][0]
        bad = MockDB()
        replay_actions(bad, [{"name": "offer_refund", "arguments": {"customer_id": "C1001", "amount": 300, "reason": "设备故障"}}])
        reward, _, _ = db_reward(task, bad)
        self.assertEqual(reward, 0.0)


class CollectTurnTest(unittest.TestCase):
    def test_dedupes_repeated_function_call_messages(self):
        """上游会把累积消息反复吐出，同一条 function_call 只能计一次。"""
        call = _msg("assistant", "", {"name": "find_customer_by_phone", "arguments": '{"phone":"1"}'})
        batches = [[call], [call], [call]]
        calls, results, text = collect_turn(batches)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 0)
        self.assertEqual(text, "")

    def test_merges_streaming_argument_fragments(self):
        batches = [
            [_msg("assistant", "", {"name": "create_repair_ticket", "arguments": '{"customer_id"'})],
            [_msg("assistant", "", {"name": "create_repair_ticket", "arguments": '{"customer_id":"C1001"'})],
            [_msg("assistant", "", {"name": "create_repair_ticket", "arguments": '{"customer_id":"C1001","fault_type":"断网"}'})],
        ]
        calls, _, _ = collect_turn(batches)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], '{"customer_id":"C1001","fault_type":"断网"}')

    def test_distinct_calls_are_kept(self):
        batches = [
            [
                _msg("assistant", "", {"name": "create_repair_ticket", "arguments": '{"customer_id":"C1001","fault_type":"断网"}'}),
                _msg("assistant", "", {"name": "create_repair_ticket", "arguments": '{"customer_id":"C1002","fault_type":"慢速"}'}),
            ]
        ]
        calls, _, _ = collect_turn(batches)
        self.assertEqual(len(calls), 2)

    def test_identical_repeated_calls_collapse(self):
        """完全相同的调用会被上游重复吐出，去重后计一次（已知局限，DB 判分才是权威）。"""
        m = _msg("assistant", "", {"name": "check_service_status", "arguments": '{"customer_id":"C1001"}'})
        calls, _, _ = collect_turn([[m], [m]])
        self.assertEqual(len(calls), 1)

    def test_final_text_taken_from_last_content(self):
        batches = [
            [_msg("assistant", "第一段")],
            [_msg("assistant", "第一段第二段")],
        ]
        _, _, text = collect_turn(batches)
        self.assertEqual(text, "第一段第二段")

    def test_history_ids_pair_calls_with_results(self):
        calls = [{"name": "a", "arguments": "{}"}, {"name": "b", "arguments": "{}"}]
        results = [("a", "ra"), ("b", "rb")]
        entries = _history_entries(calls, results, "done")
        call_ids = [e["extra"]["function_id"] for e in entries if e["role"] == "assistant" and e.get("function_call")]
        fn_ids = [e["extra"]["function_id"] for e in entries if e["role"] == "function"]
        self.assertEqual(call_ids, ["1", "2"])
        self.assertEqual(fn_ids, ["1", "2"])


class PassAtKTest(unittest.TestCase):
    def _results(self, task_id, outcomes):
        return [TaskResult(task_id=task_id, variant="v", success=o) for o in outcomes]

    def test_pass_1_is_success_rate(self):
        self.assertAlmostEqual(pass_at_k(self._results("t", [True, True, False]), 1), 2 / 3)

    def test_pass_3_requires_all_three(self):
        self.assertAlmostEqual(pass_at_k(self._results("t", [True, True, True]), 3), 1.0)
        self.assertAlmostEqual(pass_at_k(self._results("t", [True, True, False]), 3), 0.0)

    def test_comb_formula(self):
        # 4 次里成功 2 次 -> C(2,2)/C(4,2) = 1/6
        self.assertAlmostEqual(pass_at_k(self._results("t", [True, True, False, False]), 2), 1 / 6)


class PolicyTest(unittest.TestCase):
    def test_digest_sections_split(self):
        from qwen_agent_bench.domain.loader import load_policy_digest

        sections = split_policy_sections(load_policy_digest())
        self.assertIn("redline", sections)
        self.assertIn("refund", sections)
        self.assertIn("R5", sections["redline"])

    def test_digest_context_always_keeps_redline(self):
        ctx = build_digest_context(["identity"])
        self.assertIn("红线", ctx)
        self.assertIn("身份核验", ctx)
        self.assertNotIn("退款细则", ctx)

    def test_digest_redline_mentions_refund_authority(self):
        ctx = build_digest_context([])
        self.assertIn("50", ctx)


if __name__ == "__main__":
    unittest.main()
