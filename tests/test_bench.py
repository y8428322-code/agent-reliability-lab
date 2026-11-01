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
        base = len(db.tickets)
        first = db.create_ticket("C1001", "断网")
        second = db.create_ticket("C1002", "慢速")
        self.assertEqual(first, "TK%04d" % (base + 1))
        self.assertEqual(second, "TK%04d" % (base + 2))

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
        new_id = db.create_ticket("C1001", "断网")  # 新建的是「已受理」，不是「未派工」
        tools = {t.name: t for t in build_tools(db)}
        out = json.loads(tools["cancel_ticket"].call(json.dumps({"ticket_id": new_id, "reason": "用户取消"})))
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
            self.assertEqual(errors, [], "任务 %s 参考轨迹的写操作有错: %s" % (task["id"], errors))
            self.assertTrue(h)

    def test_read_failure_in_reference_is_tolerated(self):
        """H10 刻意让第一次手机号查询失败（用户报错一位），不应判为任务定义有 bug。"""
        task = [t for t in self.tasks if t["id"] == "H10"][0]
        h, write_errors = gold_hash(task)
        self.assertEqual(write_errors, [])
        self.assertNotEqual(h, MockDB().hash())  # 最终确实建了工单

    def test_task_set_has_both_positive_and_negative_gold(self):
        """一半以上任务必须是「正向写」，否则「什么都不做」也能满分。"""
        base = MockDB().hash()
        positive = [t for t in self.tasks if gold_hash(t)[0] != base]
        negative = [t for t in self.tasks if gold_hash(t)[0] == base]
        self.assertGreaterEqual(len(positive), 5)
        self.assertGreaterEqual(len(negative), 5)

    def test_same_name_customers_exist(self):
        """H6 依赖「同名不同人」的干扰项。"""
        names = [c["name"] for c in MockDB().customers.values()]
        self.assertGreater(len(names), len(set(names)))

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

    def test_ticket_creation_order_does_not_affect_hash(self):
        """工单 id 按创建顺序分配，顺序不同不应被误判为状态不一致。"""
        a = MockDB()
        a.create_ticket("C1010", "慢速")
        a.create_ticket("C1010", "掉线")
        b = MockDB()
        b.create_ticket("C1010", "掉线")
        b.create_ticket("C1010", "慢速")
        # id 与故障类型的对应关系确实不同
        map_a = {tid: t["fault_type"] for tid, t in a.tickets.items()}
        map_b = {tid: t["fault_type"] for tid, t in b.tickets.items()}
        self.assertNotEqual(map_a, map_b)
        self.assertEqual(a.hash(), b.hash())  # 但业务状态相同

    def test_duplicate_ticket_is_still_detected(self):
        """顺序无关不能掩盖「重复建单」——重复项要保留在比较里。"""
        a = MockDB()
        a.create_ticket("C1010", "慢速")
        a.create_ticket("C1010", "掉线")
        b = MockDB()
        b.create_ticket("C1010", "慢速")
        b.create_ticket("C1010", "慢速")
        self.assertNotEqual(a.hash(), b.hash())

    def test_task_set_has_positive_and_negative_gold(self):
        base = MockDB().hash()
        positive = [t for t in self.tasks if gold_hash(t)[0] != base]
        negative = [t for t in self.tasks if gold_hash(t)[0] == base]
        self.assertGreaterEqual(len(positive), 6)
        self.assertGreaterEqual(len(negative), 6)

    def test_multi_fault_task_expects_two_tickets(self):
        """T11：两个独立故障现象 -> 按故障类型分别建单。"""
        task = [t for t in self.tasks if t["id"] == "T11"][0]
        db = MockDB()
        replay_actions(db, task["reference_actions"])
        own = [t for t in db.tickets.values() if t["customer_id"] == "C1010"]
        self.assertEqual(len(own), 2)
        self.assertEqual({t["fault_type"] for t in own}, {"掉线", "慢速"})


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
        self.assertEqual(json.loads(calls[0]["arguments"]), {"customer_id": "C1001", "fault_type": "断网"})

    def test_malformed_extra_brace_args_are_normalized(self):
        """模型偶尔吐多余右括号，同一次调用不能因为字符串不同被算成两次。"""
        a = _msg("assistant", "", {"name": "replace_device", "arguments": '{"customer_id":"C1","new_device_serial":"N001"}'})
        b = _msg("assistant", "", {"name": "replace_device", "arguments": '{"customer_id":"C1","new_device_serial":"N001"}}'})
        calls, _, _ = collect_turn([[a], [b]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["arguments"]), {"customer_id": "C1", "new_device_serial": "N001"})

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

    def test_cumulative_tool_call_lists_are_deduped(self):
        """原生模式：上游每个 chunk 都会重吐**累积的** tool_calls 列表。

        回归用例：参数先是分片、后被补全，若去重索引的键没同步，
        同一条调用会被计两次（实测导致 10 个 tool_call 对 5 个 tool 返回 -> 端点 500）。
        """
        partial = _msg("assistant", "", {"name": "find_customer_by_phone", "arguments": '{"phone"'})
        full = _msg("assistant", "", {"name": "find_customer_by_phone", "arguments": '{"phone":"13800001111"}'})
        other = _msg("assistant", "", {"name": "get_account_details", "arguments": '{"customer_id":"C1001"}'})
        batches = [
            [partial],                     # chunk1: 只有第一条，参数是分片
            [full, other],                 # chunk2: 累积列表，参数补全 + 第二条
            [full, other],                 # chunk3: 又重吐一遍
        ]
        calls, _, _ = collect_turn(batches)
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(calls[0]["arguments"]), {"phone": "13800001111"})
        self.assertEqual(calls[1]["name"], "get_account_details")

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


class VariantTest(unittest.TestCase):
    def test_variant_ladder_flags(self):
        from qwen_agent_bench.variants import (
            variant_uses_bidirectional_reflection,
            variant_uses_digest,
            variant_uses_guard,
            variant_uses_native,
        )

        self.assertFalse(variant_uses_digest("A_baseline"))
        self.assertFalse(variant_uses_digest("A_native"))
        self.assertTrue(variant_uses_digest("B_context"))
        self.assertTrue(variant_uses_digest("D_guarded"))
        self.assertTrue(variant_uses_digest("E_guarded_reflect"))

        self.assertTrue(variant_uses_native("A_native"))
        self.assertTrue(variant_uses_native("B_native"))
        self.assertFalse(variant_uses_native("B_context"))

        self.assertTrue(variant_uses_guard("D_guarded"))
        self.assertTrue(variant_uses_guard("E_guarded_reflect"))
        self.assertFalse(variant_uses_guard("C_reflection"))

        self.assertTrue(variant_uses_bidirectional_reflection("E_guarded_reflect"))
        self.assertFalse(variant_uses_bidirectional_reflection("C_reflection"))

    def test_guard_system_only_in_guarded_variants(self):
        from qwen_agent_bench.variants import build_context

        task = {"policy_sections": ["redline"]}
        self.assertNotIn("动作边界", build_context("B_context", task))
        self.assertIn("动作边界", build_context("D_guarded", task))
        self.assertIn("动作边界", build_context("E_guarded_reflect", task))

    def test_guard_descriptions_only_touch_write_tools(self):
        from qwen_agent_bench.domain.tools import build_tools
        from qwen_agent_bench.variants import apply_guard_descriptions

        tools = build_tools(MockDB())
        before = {t.name: t.description for t in tools}
        apply_guard_descriptions(tools)
        for tool in tools:
            has_guard = "调用前置条件" in tool.description
            if tool.name in ("create_repair_ticket", "replace_device", "cancel_ticket", "offer_refund"):
                self.assertTrue(has_guard, tool.name)
            else:
                self.assertFalse(has_guard, tool.name)
                self.assertEqual(tool.description, before[tool.name])

    def test_reflection_prompt_is_bidirectional_only_for_e(self):
        from qwen_agent_bench.variants import reflection_prompt

        oneshot = reflection_prompt("C_reflection", "policy", "transcript")
        both = reflection_prompt("E_guarded_reflect", "policy", "transcript")
        self.assertIn("还没有被满足", oneshot)
        self.assertNotIn("不该做而做了", oneshot)
        self.assertIn("不该做而做了", both)
        self.assertIn("该做而没做", both)


class AttributionTest(unittest.TestCase):
    """动作级失败归因：把「状态不符」变成可定位的原因。"""

    def _result(self, calls, db_reward=0.0, nl_reward=1.0, tool_errors=None):
        from qwen_agent_bench.evaluator import TaskResult, TurnRecord

        res = TaskResult(task_id="X", variant="v", db_reward=db_reward, nl_reward=nl_reward)
        res.tool_errors = list(tool_errors or [])
        turn = TurnRecord(user="u", assistant_text="a")
        turn.tool_calls = list(calls)
        res.turns.append(turn)
        return res

    def test_extra_write_is_named(self):
        from qwen_agent_bench.evaluator import attribute_failure

        task = {"reference_actions": [{"name": "create_repair_ticket", "arguments": {}}]}
        res = self._result(
            [
                ("create_repair_ticket", '{"customer_id":"C1","fault_type":"断网"}'),
                ("replace_device", '{"customer_id":"C1"}'),
            ]
        )
        self.assertEqual(attribute_failure(task, res), "状态不符(多换机 x1)")

    def test_missing_write_is_named(self):
        from qwen_agent_bench.evaluator import attribute_failure

        task = {"reference_actions": [{"name": "create_repair_ticket", "arguments": {}}]}
        res = self._result([])
        self.assertEqual(attribute_failure(task, res), "状态不符(少建单 x1)")

    def test_wrong_parameter_is_named(self):
        from qwen_agent_bench.evaluator import attribute_failure

        task = {
            "reference_actions": [
                {"name": "replace_device", "arguments": {"new_device_serial": "N001"}}
            ]
        }
        res = self._result([("replace_device", '{"new_device_serial":"N004"}')])
        self.assertEqual(attribute_failure(task, res), "状态不符(换机参数不符)")

    def test_db_ok_but_nl_bad_is_unclosed(self):
        from qwen_agent_bench.evaluator import attribute_failure

        task = {"reference_actions": []}
        res = self._result([], db_reward=1.0, nl_reward=0.5)
        self.assertEqual(attribute_failure(task, res), "未闭环")

    def test_tool_error_takes_precedence(self):
        from qwen_agent_bench.evaluator import attribute_failure

        task = {"reference_actions": []}
        res = self._result([], tool_errors=["x: y"])
        self.assertEqual(attribute_failure(task, res), "参数错")


if __name__ == "__main__":
    unittest.main()
