"""telecom_cn 域的工具集。

分层原则（与 τ²-bench 一致）：
  - 工具层只校验**数据/业务前置条件**（保修期、状态机、金额硬上限），越权与否由 policy 约束 agent
  - 所有失败都返回**结构化 JSON** 而不是抛异常，让 agent 有机会纠正
  - 自由文本只进 audit_log，**不参与状态哈希**（避免措辞差异造成假失败）
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from qwen_agent.tools.base import BaseTool

from .db import DOMAIN_DATE, MockDB

FAULT_TYPES = ["断网", "慢速", "掉线", "电视信号"]
CANCEL_REASONS = ["用户取消", "重复报修", "客户改约"]
REFUND_REASONS = ["设备故障", "网络故障", "重复扣费"]
MAX_REFUND_AMOUNT = 500.0

# 会写库的写操作工具
WRITE_TOOL_NAMES = ("create_repair_ticket", "replace_device", "cancel_ticket", "offer_refund")

# 写操作 -> 业务名（业务名用于判断"是不是同一件事"）
WRITE_TOOL_BUSINESS = {
    "create_repair_ticket": "报修工单",
    "replace_device": "设备更换",
    "cancel_ticket": "工单撤销",
    "offer_refund": "退款",
}

# 业务名 -> 用户可能用的说法（用于判断用户本轮是否"单独提到"了这个业务）
BUSINESS_KEYWORDS = {
    "报修工单": ["报修", "报个修", "维修", "派人上门", "上门修", "派工", "修一下", "派人来修", "上门维修"],
    "设备更换": ["换机", "换一台", "换新的", "更换设备", "换设备", "换个新的", "换一台新的"],
    "退款": ["退款", "退钱", "退我", "退给我", "退下"],
    "工单撤销": ["撤单", "取消工单", "把工单取消", "撤销工单"],
}

# 终止性拒绝的固定措辞：必须明确表达"这不是参数错误、不要重试"
_TERMINAL = "这是终止性拒绝，本回合不要再重试该业务。"


def businesses_in_text(text: str) -> List[str]:
    """从用户消息里识别提到的业务名，**按它们在文本中出现的先后排序**。

    顺序很重要：守卫要挡的是「用户先提的那个业务之外的其它业务」，
    所以必须知道用户**先**提了哪个。
    """
    if not text:
        return []
    hits: List[Tuple[int, str]] = []
    for business, keywords in BUSINESS_KEYWORDS.items():
        positions = [text.find(kw) for kw in keywords if kw in text]
        if positions:
            hits.append((min(positions), business))
    hits.sort()
    return [business for _, business in hits]


class TurnGuard:
    """**机制约束**：一次只办理一项业务。

    背景（`docs/实验报告-v7.md` / `v8.md`）：
    - 「一次只办一件事」写成 prompt 里的**文字禁令**会被 agent 读成"别多做事"，
      连必要的查询都不做了 → 从「过度执行」翻转成「该做不做」。
    - 而实测 H9 的失败是**跨回合**的过度执行（第 1 回合办第一个诉求，
      第 2 回合办第二个），单纯的"每回合一次写操作"压不住。

    所以约束落在**工具层**，且区分两种情形：

    1. **用户在同一轮里提出了多个业务** → 本轮最多办一个，其余的记为
       「需用户单独确认」，直到用户**在某一轮里单独提到它**为止。
    2. **业务已在「需单独确认」状态** → 用户本轮没单独提到 → 拒绝。

    这样能同时满足：
    - H2（用户只提一个业务，下一轮才补信息）→ 放行 ✓
    - H9（用户同轮提两个业务，第二轮只说"好的麻烦快些"）→ 拒绝 ✓

    拒绝消息必须是**终止性**语义，与「工具参数错误」区分开 ——
    否则 agent 会把它当成可纠正的参数错误反复重试（v8 实测因此出现 3 次「参数错」）。
    """

    def __init__(self) -> None:
        self.done_businesses: List[str] = []       # 本会话已办理过的业务
        self.need_confirm: List[str] = []          # 需用户单独确认的业务
        self.turn_businesses: List[str] = []       # 用户本轮提到的业务
        self.turn_handled: List[str] = []          # 本轮已办理的业务

    # ---------- 每回合更新 ----------
    def begin_turn(self, user_text: str = "") -> None:
        """新回合开始：解析用户本轮提到的业务，更新「需单独确认」集合。"""
        mentioned = businesses_in_text(user_text)
        self.turn_businesses = mentioned
        self.turn_handled = []

        if len(mentioned) >= 2:
            # 本轮用户提了多个业务 -> **只有用户后提的那些**需要单独确认；
            # 用户先提的那个（第一个）本轮就应该办理。
            # （v9 实测 bug：把全部都标成需确认，结果连本该办的第一个业务也被挡。）
            for business in mentioned[1:]:
                if business not in self.need_confirm:
                    self.need_confirm.append(business)
        elif len(mentioned) == 1:
            # 用户单独提到了它 -> 视为确认
            business = mentioned[0]
            if business in self.need_confirm:
                self.need_confirm.remove(business)

    # ---------- 写操作检查 ----------
    def check(self, tool_name: str) -> Optional[str]:
        """若该写操作不被允许，返回拒绝原因（终止性语义）；否则返回 None。"""
        if tool_name not in WRITE_TOOL_NAMES:
            return None
        business = WRITE_TOOL_BUSINESS[tool_name]

        # 1) 本轮用户提了多个业务时，本轮**只允许办理用户先提出的那个**
        if len(self.turn_businesses) >= 2:
            first = self.turn_businesses[0]
            if business != first:
                return (
                    "本轮用户同时提出了多个业务（%s），一次只办理一项业务："
                    "本轮只能办理用户先提出的「%s」。请先办理「%s」，"
                    "其余业务需用户单独确认后再办。%s"
                    % ("、".join(self.turn_businesses), first, first, _TERMINAL)
                )

        # 2) 该业务需用户单独确认，而用户本轮没有单独提到它
        if business in self.need_confirm and business not in self.turn_businesses:
            return (
                "「%s」需要用户单独确认后才能办理（用户当轮同时提出了多个业务）。"
                "请先向用户说明并获得其单独确认，在用户确认后的下一轮再办理。%s"
                % (business, _TERMINAL)
            )

        # 3) 同一业务已办理过 -> 允许（重复次数由策略里的具体规则约束，如「不重复建单」）
        return None

    def record(self, tool_name: str) -> None:
        if tool_name not in WRITE_TOOL_NAMES:
            return
        business = WRITE_TOOL_BUSINESS[tool_name]
        if business not in self.done_businesses:
            self.done_businesses.append(business)
        if business not in self.turn_handled:
            self.turn_handled.append(business)


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _err(message: str, **payload: Any) -> str:
    return json.dumps({"ok": False, "error": message, **payload}, ensure_ascii=False)


class _DbTool(BaseTool):
    """带数据库引用的工具基类。"""

    def __init__(self, db: MockDB, guard: Optional["TurnGuard"] = None) -> None:
        super().__init__(None)
        self.db = db
        self.guard = guard

    def _guard_block(self, tool_name: str) -> Optional[str]:
        """机制约束检查：返回拒绝原因（None 表示放行）。"""
        if self.guard is None:
            return None
        return self.guard.check(tool_name)

    def _guard_record(self, tool_name: str) -> None:
        if self.guard is not None:
            self.guard.record(tool_name)


class FindCustomerByPhone(_DbTool):
    name = "find_customer_by_phone"
    description = "根据手机号查询客户，返回客户编号与姓名。"
    enhanced_description = (
        "【何时使用】用户来电后，你需要先确认这是哪位客户。"
        "凡是要办理任何业务（查宽带、报修、换机、退款）之前，必须先调用本工具完成身份定位与核验。\n"
        "【何时不用】如果用户还没给出手机号，不要臆造号码，应该先向用户询问。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "phone": {"type": "string", "description": "客户手机号，11位数字"},
        },
        "required": ["phone"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        phone = str(args["phone"]).strip()
        for cid, cust in self.db.customers.items():
            if cust["phone"] == phone:
                return _ok(customer_id=cid, name=cust["name"], phone=phone)
        return _err("未找到该手机号对应的客户，请让用户核对手机号", phone=phone)


class GetAccountDetails(_DbTool):
    name = "get_account_details"
    description = "根据客户编号查询账户详情，包括账户状态、宽带状态、设备与保修期。"
    enhanced_description = (
        "【何时使用】已通过 find_customer_by_phone 定位到 customer_id 之后，"
        "需要查看账户/宽带/设备状态时调用。\n"
        "【注意】返回的字段是后续判断能否办理业务（如换机是否在保修期）的唯一依据，"
        "不要凭用户口述下结论。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "客户编号，形如 C1001"},
        },
        "required": ["customer_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        cid = str(args["customer_id"]).strip()
        cust = self.db.customers.get(cid)
        if cust is None:
            return _err("客户编号不存在", customer_id=cid)
        return _ok(**cust)


class CheckServiceStatus(_DbTool):
    name = "check_service_status"
    description = "查询客户的宽带服务状态与历史故障工单。"
    enhanced_description = (
        "【何时使用】用户报修宽带故障前调用，用来确认是账号问题（欠费/停机）还是线路故障，"
        "避免给欠费用户开故障单。\n"
        "【何时不用】已确认是线路故障且要直接建单时不必重复调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "客户编号，形如 C1001"},
        },
        "required": ["customer_id"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        cid = str(args["customer_id"]).strip()
        cust = self.db.customers.get(cid)
        if cust is None:
            return _err("客户编号不存在", customer_id=cid)
        tickets = [t for t in self.db.tickets.values() if t["customer_id"] == cid]
        return _ok(
            account_status=cust["account_status"],
            broadband_status=cust["broadband_status"],
            tickets=tickets,
        )


class CheckDeviceWarranty(_DbTool):
    name = "check_device_warranty"
    description = "查询设备是否在保修期内，以及保修截止日期。"
    enhanced_description = (
        "【何时使用】用户要求换机时**必须**先调用本工具判断是否在保，"
        "不要根据用户自称的购买时间判断。\n"
        "【当前日期】%s，超过 warranty_until 即为过保。" % DOMAIN_DATE
    )
    parameters = {
        "type": "object",
        "properties": {
            "device_serial": {"type": "string", "description": "设备序列号，形如 D001"},
        },
        "required": ["device_serial"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        serial = str(args["device_serial"]).strip()
        device = self.db.devices.get(serial)
        customer = None
        for cust in self.db.customers.values():
            if cust["device_serial"] == serial:
                customer = cust
                break
        if device is None:
            return _err("设备序列号不存在", device_serial=serial)
        warranty_until = customer["warranty_until"] if customer else ""
        return _ok(
            device_serial=serial,
            model=device["model"],
            purchase_date=device["purchase_date"],
            warranty_until=warranty_until,
            in_warranty=bool(warranty_until) and warranty_until >= DOMAIN_DATE,
            current_date=DOMAIN_DATE,
        )


class CreateRepairTicket(_DbTool):
    name = "create_repair_ticket"
    description = "为客户创建宽带故障工单，返回工单号。"
    enhanced_description = (
        "【何时使用】确认是线路故障（而非欠费/停机）且用户要求上门维修后调用。\n"
        "【前置条件】必须已定位 customer_id；fault_type 必须取自允许值 %s。\n"
        "【何时不用】宽带状态为「欠费」时不要建单，应先告知欠费。" % FAULT_TYPES
    )
    parameters = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "客户编号，形如 C1001"},
            "fault_type": {
                "type": "string",
                "description": "故障类型",
                "enum": FAULT_TYPES,
            },
        },
        "required": ["customer_id", "fault_type"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        cid = str(args["customer_id"]).strip()
        fault_type = str(args["fault_type"]).strip()
        if cid not in self.db.customers:
            return _err("客户编号不存在，不能建单", customer_id=cid)
        blocked = self._guard_block("create_repair_ticket")
        if blocked:
            return _err(blocked)
        ticket_id = self.db.create_ticket(cid, fault_type)
        self._guard_record("create_repair_ticket")
        return _ok(ticket_id=ticket_id, customer_id=cid, fault_type=fault_type, sla="24小时内上门")


class ReplaceDevice(_DbTool):
    name = "replace_device"
    description = "为客户更换光猫设备。"
    enhanced_description = (
        "【何时使用】用户设备故障且**已确认在保修期内**时调用。\n"
        "【前置条件】device_serial 必须是客户当前在用设备；new_device_serial 必须取自库存清单。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "客户编号，形如 C1001"},
            "device_serial": {"type": "string", "description": "客户当前在用设备序列号"},
            "new_device_serial": {"type": "string", "description": "新设备序列号，取自库存清单"},
        },
        "required": ["customer_id", "device_serial", "new_device_serial"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        cid = str(args["customer_id"]).strip()
        old = str(args["device_serial"]).strip()
        new = str(args["new_device_serial"]).strip()
        cust = self.db.customers.get(cid)
        if cust is None:
            return _err("客户编号不存在", customer_id=cid)
        if cust["device_serial"] != old:
            return _err("设备序列号与客户当前在用设备不一致", customer_id=cid, device_serial=old)
        if cust["warranty_until"] < DOMAIN_DATE:
            return _err(
                "设备已过保修期（保修截止 %s，当前 %s），不能走保修换机" % (cust["warranty_until"], DOMAIN_DATE),
                warranty_until=cust["warranty_until"],
                current_date=DOMAIN_DATE,
            )
        if new not in self.db.inventory:
            return _err("新设备序列号不在库存清单中", new_device_serial=new)
        blocked = self._guard_block("replace_device")
        if blocked:
            return _err(blocked)
        self.db.replace_device(cid, old, new)
        self._guard_record("replace_device")
        return _ok(customer_id=cid, old_device_serial=old, new_device_serial=new, warranty_until=cust["warranty_until"])


class CancelTicket(_DbTool):
    name = "cancel_ticket"
    description = "撤销客户的故障工单。"
    enhanced_description = (
        "【何时使用】用户要求撤销已提交但尚未派工的工单时调用。\n"
        "【前置条件】工单状态必须是「未派工」；已派工的工单无法撤销。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "description": "工单号，形如 TK0001"},
            "reason": {
                "type": "string",
                "description": "撤单原因",
                "enum": CANCEL_REASONS,
            },
        },
        "required": ["ticket_id", "reason"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        ticket_id = str(args["ticket_id"]).strip()
        reason = str(args["reason"]).strip()
        ticket = self.db.tickets.get(ticket_id)
        if ticket is None:
            return _err("工单号不存在", ticket_id=ticket_id)
        if ticket["status"] != "未派工":
            return _err(
                "工单当前状态为「%s」，只有「未派工」的工单才能撤销" % ticket["status"],
                ticket_id=ticket_id,
                status=ticket["status"],
            )
        blocked = self._guard_block("cancel_ticket")
        if blocked:
            return _err(blocked)
        self.db.cancel_ticket(ticket_id, reason)
        self._guard_record("cancel_ticket")
        return _ok(ticket_id=ticket_id, status="已撤销", reason=reason)


class OfferRefund(_DbTool):
    name = "offer_refund"
    description = "为客户发起退款。"
    enhanced_description = (
        "【何时使用】仅在你已确认自己有权办理该笔退款时才调用。\n"
        "【注意】单笔金额不得超过 %s 元；超出部分本工具会直接报错。" % MAX_REFUND_AMOUNT
    )
    parameters = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "客户编号，形如 C1001"},
            "amount": {"type": "number", "description": "退款金额，单位元"},
            "reason": {
                "type": "string",
                "description": "退款原因",
                "enum": REFUND_REASONS,
            },
        },
        "required": ["customer_id", "amount", "reason"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        args = self._verify_json_format_args(params)
        cid = str(args["customer_id"]).strip()
        amount = float(args["amount"])
        reason = str(args["reason"]).strip()
        if cid not in self.db.customers:
            return _err("客户编号不存在", customer_id=cid)
        if amount <= 0:
            return _err("退款金额必须大于 0", amount=amount)
        if amount > MAX_REFUND_AMOUNT:
            return _err(
                "单笔退款不得超过 %s 元，该笔需升级人工处理" % MAX_REFUND_AMOUNT,
                amount=amount,
                limit=MAX_REFUND_AMOUNT,
            )
        blocked = self._guard_block("offer_refund")
        if blocked:
            return _err(blocked)
        refund_id = self.db.create_refund(cid, amount, reason)
        self._guard_record("offer_refund")
        return _ok(refund_id=refund_id, customer_id=cid, amount=amount)


class ListInventory(_DbTool):
    name = "list_inventory"
    description = "查询当前可用的备件库存清单，返回设备序列号与型号。"
    enhanced_description = (
        "【何时使用】需要为用户选择替换设备时**必须先调用本工具**获取合法的设备序列号，"
        "不要凭记忆或猜测填写 new_device_serial。\n"
        "【怎么选】优先选与用户原设备**同型号**的条目；同型号缺货时再向用户确认是否接受其他型号。"
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        self._verify_json_format_args(params)
        items = [
            {"device_serial": serial, "model": item["model"], "type": item["type"]}
            for serial, item in sorted(self.db.inventory.items())
        ]
        return _ok(inventory=items)


# 写操作工具的「调用前置条件」（减法机制的工具级部分）。
# 按工具写成**具体动作**，避免笼统的"请谨慎操作"被模型忽略。
GUARD_LINES = {
    "create_repair_ticket": (
        "【调用前置条件】仅当用户**明确要求派人上门维修（报修）**时才调用本工具。"
        "用户只是描述故障现象、或只问「能不能修」而没有要求办理时，不要调用；"
        "同一客户同一故障类型不要重复建单。"
    ),
    "replace_device": (
        "【调用前置条件】仅当用户**明确要求更换设备（换机）**时才调用本工具。"
        "用户只是说设备坏了、或在询问能否换机而没有明确要求换机时，不要直接调用。"
    ),
    "cancel_ticket": (
        "【调用前置条件】仅当用户**明确要求撤销工单**时才调用本工具。"
    ),
    "offer_refund": (
        "【调用前置条件】仅当用户**明确要求退款**，且符合普通坐席自助退款条件"
        "（重复扣费且单笔不超过 50 元）时才调用本工具。"
        "其他任何情形都不要调用本工具——即使用户坚持，也只能说明并升级人工。"
    ),
}


TOOL_CLASSES = [
    FindCustomerByPhone,
    GetAccountDetails,
    CheckServiceStatus,
    CheckDeviceWarranty,
    ListInventory,
    CreateRepairTicket,
    ReplaceDevice,
    CancelTicket,
    OfferRefund,
]


def build_tools(db: MockDB, guard: Optional["TurnGuard"] = None) -> List[_DbTool]:
    """构造绑定到指定数据库的工具实例。

    Args:
        guard: 机制约束守卫；``None`` 表示不启用（即 v6/v7 的行为）。
    """
    return [cls(db, guard) for cls in TOOL_CLASSES]


def tool_names() -> List[str]:
    return [cls.name for cls in TOOL_CLASSES]


def apply_enhanced_descriptions(tools: List[_DbTool]) -> None:
    """把工具描述替换为「上下文工程」版本（变体 B/C 使用）。"""
    for tool in tools:
        enhanced = getattr(tool, "enhanced_description", None)
        if enhanced:
            tool.description = enhanced


def baseline_descriptions(tools: List[_DbTool]) -> None:
    """恢复为朴素描述（变体 A 使用）。"""
    for tool in tools:
        cls = type(tool)
        tool.description = cls.__dict__.get("description", "")
