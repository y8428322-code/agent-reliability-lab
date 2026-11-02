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
            variant_uses_mechanism_guard,
            variant_uses_native,
            variant_uses_prompt_guard,
        )

        self.assertFalse(variant_uses_digest("A_baseline"))
        self.assertFalse(variant_uses_digest("A_native"))
        self.assertTrue(variant_uses_digest("B_context"))
        self.assertTrue(variant_uses_digest("D_guarded"))
        self.assertTrue(variant_uses_digest("E_guarded_reflect"))

        self.assertTrue(variant_uses_native("A_native"))
        self.assertTrue(variant_uses_native("B_native"))
        self.assertFalse(variant_uses_native("B_context"))

        # prompt 级减法（D/E）与工具级机制约束（B_guard）是两回事，不能混用同一个函数名
        self.assertTrue(variant_uses_prompt_guard("D_guarded"))
        self.assertTrue(variant_uses_prompt_guard("E_guarded_reflect"))
        self.assertFalse(variant_uses_prompt_guard("B_guard"))
        self.assertTrue(variant_uses_mechanism_guard("B_guard"))
        self.assertFalse(variant_uses_mechanism_guard("D_guarded"))

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


class PolicyRenderTest(unittest.TestCase):
    """策略单一事实源与两种渲染版式。"""

    def setUp(self):
        from qwen_agent_bench.domain.render import load_policy_rules

        self.spec = load_policy_rules()

    def test_single_source_of_truth(self):
        """所有变体的策略都必须能追溯到同一份规则定义。"""
        rules = self.spec["rules"]
        self.assertGreaterEqual(len(rules), 10)
        ids = [r["id"] for r in rules]
        self.assertEqual(len(ids), len(set(ids)))
        for r in rules:
            self.assertTrue(r.get("body", "").strip(), "规则 %s 缺 body" % r["id"])
            self.assertTrue(r.get("criteria"), "规则 %s 缺 criteria" % r["id"])

    def test_narrative_contains_all_rules(self):
        """长叙述版按分节拼正文，**规则标题不出现**（规则埋在段落里是这一版的设计）。"""
        from qwen_agent_bench.domain.render import render_narrative

        text = render_narrative(self.spec)
        for r in self.spec["rules"]:
            first_line = r["body"].strip().splitlines()[0]
            self.assertIn(first_line[:24], text, "长叙述版缺少规则 %s 的正文" % r["id"])

    def test_structured_exposes_rule_titles(self):
        """结构化版相反：规则标题与判据都显式列出。"""
        from qwen_agent_bench.domain.render import render_structured

        text = render_structured(self.spec, scenes=None)
        for r in self.spec["rules"]:
            self.assertIn(r["title"], text, "结构化版缺少规则标题 %s" % r["title"])
            self.assertIn(r["criteria"][0][:16], text, "结构化版缺少判据 %s" % r["id"])

    def test_structured_puts_redlines_first(self):
        from qwen_agent_bench.domain.render import render_structured

        text = render_structured(self.spec, scenes=["identity"])
        redline_idx = text.find("红线")
        self.assertTrue(redline_idx >= 0)
        # 红线段应在其它分节之前
        for r in self.spec["rules"]:
            if not r.get("redline"):
                continue
            self.assertIn(r["title"], text)
        # 非红线规则在 scenes 裁剪外时不应出现
        refund_rules = [r for r in self.spec["rules"] if r["category"] == "refund" and not r.get("redline")]
        for r in refund_rules:
            self.assertNotIn(r["title"], text)

    def test_structured_scene_filter(self):
        from qwen_agent_bench.domain.render import render_structured

        text = render_structured(self.spec, scenes=["identity", "repair", "speech"])
        self.assertIn("身份核验", text)
        self.assertIn("故障受理", text)
        self.assertNotIn("工单撤销", text)

    def test_variants_use_same_source_different_rendering(self):
        """A 与 B 必须同源：规则集合一致，只是版式不同。"""
        from qwen_agent_bench.variants import policy_for_variant

        task = {"policy_sections": ["redline", "identity", "repair", "speech"]}
        a = policy_for_variant("A_baseline", task)
        b = policy_for_variant("B_context", task)
        self.assertNotEqual(a, b)
        # 同一份规则：两版都应提到身份核验的关键判据
        for text in (a, b):
            self.assertIn("手机号", text)

    def test_handwritten_variant_differs_from_auto(self):
        from qwen_agent_bench.variants import policy_for_variant

        task = {"policy_sections": ["redline", "identity", "repair", "speech"]}
        auto = policy_for_variant("B_context", task)
        hand = policy_for_variant("B_handwritten", task)
        self.assertNotEqual(auto, hand)

    def test_three_context_mechanisms_are_orthogonal(self):
        """上下文工程的三个子机制必须能独立开关，否则无法归因。

        注意：断言的是**呈现形式**而不是长度——长度会随规则集与措辞变化，
        形式（要点 vs 段落、红线块位置、包含哪些分节）才是稳定的不变量。
        """
        from qwen_agent_bench.variants import policy_for_variant

        task = {"policy_sections": ["redline", "identity", "repair", "speech"]}
        a = policy_for_variant("A_baseline", task)
        crop = policy_for_variant("B_crop", task)
        points = policy_for_variant("B_points", task)
        full = policy_for_variant("B_context", task)

        # 裁剪：不含无关分节
        self.assertNotIn("工单撤销", crop)
        self.assertIn("故障受理", crop)
        # 要点化：呈现为要点列表而非长段落
        self.assertIn("- ", points)
        self.assertNotIn("- ", a)
        # 红线前置：红线块在最前
        self.assertTrue(full.startswith("## 0. 红线"))
        self.assertFalse(a.startswith("## 0. 红线"))
        # 四者互不相同
        self.assertEqual(len({a, crop, points, full}), 4)

    def test_keep_and_position_are_decoupled(self):
        """「是否保留」与「是否置顶」必须是两件事。

        v6 之前把两者混在一起，导致「按场景裁剪」的伤害被高估、机制被误判。
        """
        from qwen_agent_bench.domain.render import load_policy_rules, render

        spec = load_policy_rules()
        scenes = ["identity", "device", "speech"]
        top = render(spec, scenes=scenes, points=True, redline_first=True)
        bottom = render(spec, scenes=scenes, points=True, redline_first=False)
        self.assertTrue(top.startswith("## 0."))
        self.assertFalse(bottom.startswith("## 0."))
        # 两种位置下，红线内容都在
        self.assertIn("R2", top)
        self.assertIn("R2", bottom)

    def test_cropping_strategies_differ_as_designed(self):
        """三档裁剪策略必须真的不同：旧裁剪连红线一起裁，中/新裁剪都保住红线。"""
        from qwen_agent_bench.variants import policy_for_variant

        task = {"policy_sections": ["redline", "identity", "device", "speech"]}
        old = policy_for_variant("B_crop_points", task)
        mid = policy_for_variant("B_crop_keepredline", task)
        safe = policy_for_variant("B_crop_safe", task)

        r2 = "先查账户/宽带状态"          # R2 红线（category=repair，本场景不含）
        r15 = "不代替用户做未经确认的决定"  # R15（category=multi，本场景不含）

        # 旧裁剪：红线被一起裁掉
        self.assertNotIn(r2, old)
        self.assertNotIn(r15, old)
        # 中/新裁剪：红线保住
        self.assertIn(r2, mid)
        self.assertIn(r2, safe)
        self.assertLess(len(old), len(mid))

    def test_guard_variant_shares_policy_with_keepredline(self):
        """B_guard 与 B_crop_keepredline 的**策略必须完全相同**，
        这样「机制约束」才是唯一变量。"""
        from qwen_agent_bench.variants import policy_for_variant

        task = {"policy_sections": ["redline", "identity", "device", "speech"]}
        self.assertEqual(
            policy_for_variant("B_guard", task),
            policy_for_variant("B_crop_keepredline", task),
        )

    def _guard_tools(self):
        import json as _json

        from qwen_agent_bench.domain.tools import TurnGuard, build_tools

        db = MockDB()
        guard = TurnGuard()
        tools = {t.name: t for t in build_tools(db, guard=guard)}
        return db, guard, tools, _json

    def test_turn_guard_blocks_second_business_in_a_multi_business_turn(self):
        """用户同一轮提了多个业务 -> 本轮只准办第一个。"""
        db, guard, tools, js = self._guard_tools()
        guard.begin_turn("我家宽带断了，而且光猫也坏了，你直接给我换一台新的，顺便报修")

        first = js.loads(
            tools["create_repair_ticket"].call(
                js.dumps({"customer_id": "C1001", "fault_type": "断网"})
            )
        )
        self.assertTrue(first["ok"])

        second = js.loads(
            tools["replace_device"].call(
                js.dumps(
                    {"customer_id": "C1001", "device_serial": "D001", "new_device_serial": "N001"}
                )
            )
        )
        self.assertFalse(second["ok"])
        self.assertIn("一次只办理一项业务", second["error"])
        # 终止性语义：明确不是参数错误
        self.assertIn("终止性", second["error"])
        self.assertEqual(db.customers["C1001"]["device_serial"], "D001")

    def test_turn_guard_blocks_second_business_in_later_turn(self):
        """跨回合：需用户单独确认的业务，在用户没单独提到时仍被挡（v8 的缺口）。"""
        db, guard, tools, js = self._guard_tools()
        guard.begin_turn("我家宽带断了，而且光猫也坏了，你直接给我换一台新的，顺便报修")
        js.loads(
            tools["create_repair_ticket"].call(
                js.dumps({"customer_id": "C1001", "fault_type": "断网"})
            )
        )
        # 下一轮用户没说换机
        guard.begin_turn("好的麻烦快些")
        blocked = js.loads(
            tools["replace_device"].call(
                js.dumps(
                    {"customer_id": "C1001", "device_serial": "D001", "new_device_serial": "N001"}
                )
            )
        )
        self.assertFalse(blocked["ok"])
        self.assertIn("需要用户单独确认", blocked["error"])

        # 用户单独确认换机后 -> 放行
        guard.begin_turn("对，那就换机吧")
        allowed = js.loads(
            tools["replace_device"].call(
                js.dumps(
                    {"customer_id": "C1001", "device_serial": "D001", "new_device_serial": "N001"}
                )
            )
        )
        self.assertTrue(allowed["ok"])

    def test_turn_guard_allows_single_business_across_turns(self):
        """H2 场景：用户只提一个业务，下一轮补信息后才办 -> 必须放行。"""
        db, guard, tools, js = self._guard_tools()
        guard.begin_turn("我这个光猫坏了，给我换一台新的吧")
        guard.begin_turn("张伟，手机号 13800001111")
        out = js.loads(
            tools["replace_device"].call(
                js.dumps(
                    {"customer_id": "C1001", "device_serial": "D001", "new_device_serial": "N001"}
                )
            )
        )
        self.assertTrue(out["ok"])

    def test_turn_guard_allows_same_business_twice(self):
        """T11 场景：同一业务（建单）可以多次，重复次数由策略里的具体规则管。"""
        db, guard, tools, js = self._guard_tools()
        guard.begin_turn("我网络特别慢，而且每隔几分钟就掉线，这两个毛病你都得给我报修")
        a = js.loads(
            tools["create_repair_ticket"].call(
                js.dumps({"customer_id": "C1001", "fault_type": "慢速"})
            )
        )
        b = js.loads(
            tools["create_repair_ticket"].call(
                js.dumps({"customer_id": "C1001", "fault_type": "掉线"})
            )
        )
        self.assertTrue(a["ok"])
        self.assertTrue(b["ok"])

    def test_read_tools_never_blocked(self):
        db, guard, tools, js = self._guard_tools()
        guard.begin_turn("我要退款")
        read = js.loads(tools["check_service_status"].call(js.dumps({"customer_id": "C1001"})))
        self.assertTrue(read["ok"])


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

    def test_broken_tool_call_is_its_own_failure_mode(self):
        """模型写了 tool_call 但格式坏了（未闭合/含零宽字符）→ 工具没执行。

        这与「压根没尝试」是两回事，改进手段完全不同（协议/解析 vs 决策），
        必须能单独归因。
        """
        from qwen_agent_bench.evaluator import TurnRecord, attribute_failure

        task = {"reference_actions": [{"name": "create_repair_ticket", "arguments": {}}]}
        res = TaskResult(task_id="X", variant="v", db_reward=0.0, nl_reward=1.0)
        turn = TurnRecord(
            user="u",
            assistant_text=(
                'tool_call>\n{"name": "create_repair_ticket", "arguments": {}}\n'
                "</tool_call>"  # 闭合标签内含零宽字符
            ),
        )
        res.turns.append(turn)
        self.assertEqual(attribute_failure(task, res), "工具调用格式错误")

    def test_successful_run_is_not_labelled_as_failure(self):
        """成功运行即使残留了坏片段，也不该带失败标签（那是独立质量指标）。"""
        from qwen_agent_bench.evaluator import TurnRecord, attribute_failure

        task = {"reference_actions": []}
        res = TaskResult(task_id="X", variant="v", db_reward=1.0, nl_reward=1.0)
        res.turns.append(
            TurnRecord(user="u", assistant_text='tool_call>{"name":"x"}</tool_call>')
        )
        self.assertEqual(attribute_failure(task, res), "unknown")

    def test_broken_fragment_for_already_called_tool_is_harmless(self):
        """片段提到的工具若已执行，只是无害冗余，不算格式错误。"""
        from qwen_agent_bench.evaluator import TurnRecord, attribute_failure

        task = {"reference_actions": []}
        res = TaskResult(task_id="X", variant="v", db_reward=1.0, nl_reward=1.0)
        turn = TurnRecord(user="u", assistant_text='tool_call>{"name":"find_customer_by_phone"}</tool_call>')
        turn.tool_calls.append(("find_customer_by_phone", "{}"))
        res.turns.append(turn)
        self.assertEqual(attribute_failure(task, res), "unknown")


if __name__ == "__main__":
    unittest.main()
