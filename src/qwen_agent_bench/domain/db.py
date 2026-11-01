"""评测床的模拟数据库与确定性哈希。

设计要点（借鉴 τ²-bench 的 EnvironmentEvaluator）：
  - 参考轨迹回放到干净 DB 得到「目标终态」，agent 的写操作回放到副本得到「预测终态」
  - 两者按**哈希**比对，任何产生等价终态的路径都算通过
  - 只有**写操作**造成的状态参与比对；`audit_log` 与自由文本字段不参与，
    避免 agent 措辞不同导致假失败
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# 域内固定的"当前时间"，保证 gold 与 predicted 的时间戳一致（时间戳参与哈希）
DOMAIN_NOW = "2026-10-31T09:00:00"
DOMAIN_DATE = "2026-10-31"

# 参与哈希的状态字段（audit_log 不参与）
HASHED_KEYS = ("customers", "devices", "inventory", "tickets", "refunds")


def domain_dir() -> Path:
    """域数据目录。"""
    return Path(__file__).resolve().parent


def load_seed() -> Dict[str, Any]:
    """读取初始 DB 数据。"""
    with (domain_dir() / "db.json").open("r", encoding="utf-8") as f:
        return json.load(f)


class MockDB:
    """模拟数据库。"""

    def __init__(self, seed: Optional[Dict[str, Any]] = None) -> None:
        self.data: Dict[str, Any] = copy.deepcopy(seed) if seed is not None else load_seed()
        self.audit_log: List[Dict[str, Any]] = []

    # ---------- 只读访问 ----------
    @property
    def customers(self) -> Dict[str, Any]:
        return self.data["customers"]

    @property
    def devices(self) -> Dict[str, Any]:
        return self.data["devices"]

    @property
    def inventory(self) -> Dict[str, Any]:
        return self.data["inventory"]

    @property
    def tickets(self) -> Dict[str, Any]:
        return self.data["tickets"]

    @property
    def refunds(self) -> Dict[str, Any]:
        return self.data["refunds"]

    # ---------- 写操作 ----------
    def next_ticket_id(self) -> str:
        """确定性工单号：TK + 四位序号。按创建顺序递增。"""
        return "TK%04d" % (len(self.tickets) + 1)

    def next_refund_id(self) -> str:
        return "RF%04d" % (len(self.refunds) + 1)

    def create_ticket(self, customer_id: str, fault_type: str, note: str = "") -> str:
        """创建故障工单。note 是自由文本，只进 audit_log，不参与哈希。"""
        ticket_id = self.next_ticket_id()
        self.tickets[ticket_id] = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "fault_type": fault_type,
            "status": "已受理",
            "created_at": DOMAIN_NOW,
            "appointment": "待预约",
        }
        self.log("create_repair_ticket", customer_id, {"ticket_id": ticket_id, "fault_type": fault_type}, note)
        return ticket_id

    def cancel_ticket(self, ticket_id: str, reason: str = "") -> bool:
        """撤销工单。仅当状态为「未派工」时允许（业务约束由工具层校验）。"""
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            return False
        if ticket["status"] != "未派工":
            return False
        ticket["status"] = "已撤销"
        self.log("cancel_ticket", ticket.get("customer_id", ""), {"ticket_id": ticket_id}, reason)
        return True

    def replace_device(self, customer_id: str, device_serial: str, new_device_serial: str) -> bool:
        """更换设备。保修期等业务约束由工具层校验。"""
        customer = self.customers.get(customer_id)
        if customer is None:
            return False
        customer["device_serial"] = new_device_serial
        self.log(
            "replace_device",
            customer_id,
            {"old": device_serial, "new": new_device_serial},
            "",
        )
        return True

    def create_refund(self, customer_id: str, amount: float, reason: str = "") -> str:
        """创建退款记录。amount 参与哈希；reason 是自由文本，只进 audit_log。"""
        refund_id = self.next_refund_id()
        self.refunds[refund_id] = {
            "refund_id": refund_id,
            "customer_id": customer_id,
            "amount": amount,
            "status": "已发起",
        }
        self.log("offer_refund", customer_id, {"refund_id": refund_id, "amount": amount}, reason)
        return refund_id

    # ---------- 审计 ----------
    def log(self, action: str, customer_id: str, args: Dict[str, Any], note: str = "") -> None:
        self.audit_log.append(
            {
                "action": action,
                "customer_id": customer_id,
                "args": args,
                "note": note,
                "at": DOMAIN_NOW,
            }
        )

    # ---------- 哈希 ----------
    def state_for_hash(self) -> Dict[str, Any]:
        """参与哈希的状态切片（不含 audit_log）。

        **工单与退款用「排序后的列表」而不是字典**：
        它们的 id 是按创建顺序自动分配的（TK0003、TK0004……），
        若 agent 与参考轨迹以**不同顺序**创建同样的工单，字典比较会因 id 不同而误判。
        因此这里剔除自动 id，只按 (customer_id, fault_type, status, created_at) 等**业务字段**
        排序比较——顺序无关，但重复仍会被发现（列表保留重复项）。
        """
        state: Dict[str, Any] = {}
        for key in HASHED_KEYS:
            value = self.data.get(key)
            if key == "tickets":
                state[key] = sorted(
                    [
                        [
                            t.get("customer_id", ""),
                            t.get("fault_type", ""),
                            t.get("status", ""),
                            t.get("created_at", ""),
                        ]
                        for t in (value or {}).values()
                    ]
                )
            elif key == "refunds":
                state[key] = sorted(
                    [
                        [
                            r.get("customer_id", ""),
                            r.get("amount", 0),
                            r.get("status", ""),
                        ]
                        for r in (value or {}).values()
                    ]
                )
            else:
                state[key] = value
        return state

    def hash(self) -> str:
        """状态哈希。同一状态必然同一哈希。"""
        blob = json.dumps(self.state_for_hash(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
