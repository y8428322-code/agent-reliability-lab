# Qwen-Agent 复现演示（接入 MiniMax）

基于 GitHub 高星开源框架 **Qwen-Agent**（17,236★ / Apache-2.0 / 阿里）的本地复现工程：
拉取上游源码 → 建立环境 → 用 MiniMax 的 **OpenAI 兼容端点**跑通 **Function Calling / ReAct Chat / 多 Agent 编排** 三个场景。

> 上游源码保持 pristine（未做任何魔改）。魔改方案待本工程跑通后再定。
> 评测 / 调优 / 安全管控部分**只出方法论文档**，不实现代码，见 `docs/`。

---

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12（`py -3.12`）。**本机 `python` / `pip` 是应用商店占位程序，不可用** |
| 虚拟环境 | `./.venv`（由 `scripts/setup_env.sh` 创建） |
| 上游源码 | `upstream/Qwen-Agent`，commit `31a4d36`（由 `scripts/clone_upstream.sh` 拉取，已 gitignore） |
| 模型服务 | MiniMax OpenAI 兼容端点 `https://api.minimaxi.com/v1`，模型 `MiniMax-M3.1-Flash` |
| 密钥 | 环境变量 `MINIMAX_API_KEY`；未设置时回退 `ANTHROPIC_AUTH_TOKEN`（见 `.env.example`） |
| PyPI 源 | 默认走阿里云镜像（本机到 pypi.org 直连很慢，索引页 30s+） |

**网络前提**（本机实测）：GitHub 的 git 协议被重置，因此拉源码走 `gitclone.com` 镜像 / `gh-proxy.com` 转发 / `codeload` tarball 三级兜底，脚本已内置。

---

## 目录

```
agent/
├── README.md                     # 本文件
├── requirements.txt              # 本项目额外依赖（PyYAML / requests / numpy 等）
├── .env.example                  # 密钥变量示例
├── .gitignore
├── configs/
│   └── model.yaml                # 模型 / 端点 / 场景开关配置
├── src/qwen_agent_demo/          # ① 演示层：证明底座可信
│   ├── config.py                 # YAML 配置 -> dataclass，密钥取值链
│   ├── model_factory.py          # 装配 LLM + 协议级冒烟 + 带重试的文本对话
│   ├── tools.py                  # 两个本地确定性工具（sum_numbers / text_stats）
│   └── demo.py                   # 三个演示场景入口
├── src/qwen_agent_bench/         # ② 评测床：自建，零新增依赖
│   ├── domain/
│   │   ├── db.py                 # 模拟数据库 + 确定性状态哈希
│   │   ├── db.json               # 初始数据（5 客户 / 5 设备 / 库存 / 预置工单）
│   │   ├── tools.py              # 8 个域工具（4 读 4 写）
│   │   ├── policy.md             # 原始业务策略（基线变体使用）
│   │   ├── policy_digest.md      # 结构化策略摘要（按 section 可裁剪）
│   │   ├── tasks.json            # 6 条中文任务（含越权 / 诱导 / 信息保护）
│   │   └── loader.py             # 任务与策略加载
│   ├── evaluator.py              # DB 终态判分 + NL 断言判定 + 失败归因
│   ├── variants.py               # A 基线 / B 上下文工程 / C +反思 三个变体
│   ├── runner.py                 # 脚本化用户 + 会话循环 + 判分
│   └── bench.py                  # 命令行入口（批量跑 + pass^k + 报告）
├── tests/                        # stdlib unittest（47 项，全部不调模型）
│   ├── test_config.py
│   ├── test_tools.py
│   └── test_bench.py
├── scripts/
│   ├── clone_upstream.sh         # 拉取上游源码（三级镜像兜底）
│   ├── setup_env.sh              # 建 venv + 装依赖
│   ├── verify.sh                 # 一键复现验证
│   ├── run_demo.sh               # 跑演示
│   └── run_bench.sh              # 跑评测床
├── docs/
│   ├── 调研报告-选型.md          # 框架选型 + 本机网络/环境实测
│   ├── 评测床设计.md             # 自建评测床的规格与实施记录
│   ├── 评测与调优方法论.md        # Reflexion / OpenJudge / Claw-Eval 方法与步骤
│   └── 安全管控方法论.md          # 腾讯朱雀 A.I.G + Agent 三道关
└── upstream/Qwen-Agent/          # 上游源码（gitignore）
```

---

## 命令

### 一键复现

```bash
# 1) 拉取上游源码
bash scripts/clone_upstream.sh

# 2) 建环境装依赖
bash scripts/setup_env.sh

# 3) 全量验证（源码检查 -> 单元测试 -> 端到端 demo）
bash scripts/verify.sh
```

### 分步执行

```bash
# 单元测试
.venv/Scripts/python.exe -m unittest discover -s tests -v

# 端到端 demo（需要 PYTHONPATH 指向 src）
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe -m qwen_agent_demo.demo
```

Windows PowerShell 下：

```powershell
$env:PYTHONPATH="src"
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -m qwen_agent_demo.demo
```

### 切换模型

改 `configs/model.yaml` 的 `model` / `base_url` 即可，已验证可用的模型名见 `https://api.minimaxi.com/v1/models`。
任何 OpenAI 兼容服务都能接（上游 `TextChatAtOAI` 原生读 `cfg['base_url']`）。

---

## ② 自建评测床（`src/qwen_agent_bench`）

一个**中文电信客服**域的 agent 评测床，**零新增依赖**（只用已装的 Qwen-Agent + 标准库）。
规格与实施记录见 `docs/评测床设计.md`。

```bash
# 全量：3 个变体 × 6 个任务 × 1 次
bash scripts/run_bench.sh

# 只跑基线 / 跑 3 次取 Pass^3
bash scripts/run_bench.sh --variant A_baseline
bash scripts/run_bench.sh --trials 3
```

**判分方式**（借鉴 τ²-bench）：参考轨迹回放到干净 DB 得到**目标终态**，agent 的写操作发生在同一份 DB 上得到**预测终态**，比对状态哈希——等价终态即通过，不要求调用序列一致。奖励 = DB 维度 × NL 断言维度。

**五个消融变体**（阶梯式，每级只加一个东西）：

| 变体 | 上下文 | 工具描述 | 额外机制 | function calling |
|---|---|---|---|---|
| `A_baseline` | 原始 policy 全文 | 默认 | — | 上游默认（Nous 文本式） |
| `A_native` | 原始 policy 全文 | 默认 | — | **原生**（`use_raw_api`） |
| `B_context` | 结构化摘要（红线前置 + 按任务裁剪） | 改写（何时该用/何时不该用/必填参数） | — | 文本式 |
| `B_native` | 同 B | 同 B | — | 原生 |
| `C_reflection` | 同 B | 同 B | 单向反思（只问"还没做什么"） | 文本式 |
| `D_guarded` | 同 B + **动作边界**（置于 system 最前） | 同 B + **按工具定制的调用前置条件** | 减法机制 | 文本式 |
| `E_guarded_reflect` | 同 D | 同 D | 减法机制 + **双向反思**（同时问"该做而没做"和"不该做而做了"） | 文本式 |

**任务集 17 条**（原 6 条回归基线 + 11 条有区分度的）：标准受理 / 欠费 / 过保换机 / 越权退款 / 信息保护 / 合规退款 / 重复报修不重复建单 / **换机需选对同型号库存** / **退款边界 50 元可办·51 元不可办** / 用户谎称在保 / 同名客户需问手机号 / 欠费账户要求换机 / 声称授权仍不得代查 / 多诉求不得越权 / 手机号报错后自恢复 / **多现象分别建单**。
正向写 gold 8 条 / 不写库 9 条 —— 避免"什么都不做也能满分"。

**核心指标**：pass@1 / pass^k（`C(success,k)/C(trials,k)`）/ DB 维度分 / NL 维度分 / 工具调用数（**看中位数**）/ 动作级失败归因。

**最终结果**（`docs/实验报告-v4.md`，255 次运行）：

| 变体 | pass@1 | **pass^3** | DB | 中位工具调用 |
|---|---|---|---|---|
| A_baseline | 0.843 | 0.765 | 0.863 | 4 |
| **B_context** | **0.941** | **0.941** | 0.941 | **3** |
| C_reflection | 0.941 | 0.941 | 0.941 | 3 |
| D_guarded | 0.941 | 0.882 | 0.941 | 3 |
| E_guarded_reflect | 0.941 | 0.882 | **0.980** | 3 |

> 已知局限（写在这里是为了不被误读）：只有一个域、一个模型，**用户是脚本化的**（非 LLM 模拟），无外部分数对照——因此**只做组内消融对比，不宣称绝对水平**。

---

## demo 覆盖的场景

| 场景 | 上游 API | 验证目标 |
|---|---|---|
| Function Calling | `Assistant(function_list=[...])` | 模型能正确发起 `tool_calls`，本地工具能正确执行并回填 |
| ReAct Chat | `ReActChat(function_list=[...])` | 多轮推理 + 工具调用链路 |
| 多 Agent 编排 | `GroupChat(agents={...})` | 多角色 agent 自动路由 |
| 协议冒烟 | 直接打 `/v1/chat/completions` 带 `tools` | 底层协议与工具调用的回归基线（不依赖框架） |

实际运行输出：

```
模型: MiniMax-M3.1-Flash
端点: https://api.minimaxi.com/v1
协议冒烟: 端点工具调用正常 (tool=add)

===== 场景一：Function Calling =====
  -> 调用工具 sum_numbers({"numbers": [12, 30, 8]})
  <- sum_numbers 返回: {"count": 3, "sum": 50.0}
  最终回答: 12、30、8 这三个数的和是 50。

===== 场景二：ReAct Chat =====
  -> (ReAct 文本动作) text_stats
  最终回答: 字符数：11，词数：2，行数：2。

===== 场景三：多 Agent 编排 =====
  -> 调用工具 sum_numbers({"numbers": [5, 6, 7]})
  <- sum_numbers 返回: {"count": 3, "sum": 18.0}
  最终回答: 18

全部场景执行完毕。
```

---

## 上游踩坑（已绕过，详见 `docs/调研报告-选型.md` 第 8 节）

1. **上游 `setup.py` 漏声明 `numpy` / `soundfile` / `ipykernel` / `regex` / `tqdm`**，但这些是模块级 import，不装则 `import qwen_agent` 直接失败 —— 已补进 `requirements.txt`。
2. **`reasoning_content` 会吞掉 `tool_calls`**：上游 `_chat_no_stream` 在响应含 `reasoning_content` 时只返回 content，丢掉工具调用；MiniMax 恒返回该字段。**保持流式**（上游默认即是）可规避。
3. **流式返回是累积式**：assistant 内容每轮是全文，`function_call.arguments` 是分片增量 —— `demo.py` 已做合并去重。
4. **`GroupChat` 成员工具必须用 `@register_tool` 注册**，否则 `Tool xxx is not registered`。

---

## 已知限制

1. **未做 RAG 场景**：Qwen-Agent 的 RAG 默认依赖 dashscope embedding（需另购 key）或 HuggingFace 本地 embedding（`huggingface.co` 在本机未验证可达）。后续接入方式见 `docs/调研报告-选型.md`。
2. **未做评测/调优/安全管控实现**：按约定只出方法与原理，见 `docs/评测与调优方法论.md`、`docs/安全管控方法论.md`。
3. **上游源码不入库**：`upstream/` 已 gitignore，需要时执行 `bash scripts/clone_upstream.sh` 重新拉取。
4. **密钥不落盘**：真实 key 只从环境变量读取，仓库内只有 `.env.example`。
5. **上游 `code_interpreter` 工具未挂载**：该工具可执行任意代码，生产化前需沙箱化（见 `docs/安全管控方法论.md` 第 5 节风险清单）。
6. **模型输出非确定性**：同一问题多次运行结果可能不同，做回归基线时建议按 Claw-Eval 的 Pass^3 口径连跑 3 次。
