# 向上游 Qwen-Agent 提交修复

> 背景：本项目在把 Qwen-Agent 接到 MiniMax（OpenAI 兼容端点）时，定位到 **5 处上游缺陷**。
> 这一步的目标是：能自己修的**提 PR**，需要讨论的**开 issue**，已有的**不重复提交**。

---

## 1. 调研结论：哪些能提、哪些已经有人提了

| # | 缺陷 | 上游状态 | 本项目的动作 |
|---|---|---|---|
| 1 | 原生 function calling 的工具结果用 `id` 而非 OpenAI 规范的 **`tool_call_id`** | **已有 open PR #917**（nankingjing，2026-07-10） | 不重复提；本项目在封装层绕过 |
| 2 | `setup.py` 漏声明 `numpy`/`soundfile`/`ipykernel`/`regex`/`tqdm` | **已有 open PR #968**（Linxiushen，2026-09-21） | 不重复提；本项目补进 `requirements.txt` |
| 3 | **非流式**回复同时含 `reasoning_content` 与 `tool_calls` 时，**tool_calls 被丢弃** | **上游未覆盖** | ✅ **提 PR** |
| 4 | `yield response + output` 反复吐出累积消息 | 上游未覆盖，但属内部设计 | 本项目在收集器侧去重 |
| 5 | 并行 tool_call 缺 `extra.function_id` 时 **id 重复** → 端点 `400 duplicate tool_call id` | **上游未覆盖** | ✅ **开 issue 讨论**（见下） |

> #1 和 #2 虽然已有 PR，但**至今未合并**（分别是 4 个月前、2 个月前提交的）。
> 重复提交同类 PR 会被标记 duplicate，价值不大；但"独立发现 + 可复现 + 带回归测试"这件事本身仍有价值，
> 所以本项目保留了对这两个缺陷的**独立复现**与规避方案。

---

## 2. 材料清单

| 文件 | 用途 |
|---|---|
| `0001-fix-keep-tool-calls-with-reasoning-content.patch` | PR 的补丁（167 行，含代码改动 + 3 个回归测试） |
| `PR-修复non-streaming丢tool_calls.md` | PR 描述（英文，含真实端点复现证据） |
| `ISSUE-并行tool_call_id重复.md` | issue 正文（英文，含三种候选修法与配对约束分析） |
| `../../scripts/submit_upstream_pr.py` | 带 `GH_TOKEN` 一键完成：fork → push → 开 PR → 开 issue |

---

## 3. 缺陷 #3 的复现证据（真实端点）

`MiniMax-M3.1-Flash`，非流式，带工具：

```
Q: 请用工具计算 (137*89 - 42) / 7 的结果
   reasoning_content=有(297字) | tool_calls=['calc'] | finish=tool_calls
```

上游 `_chat_no_stream` 命中 `hasattr(message, 'reasoning_content')` 分支 →
只返回 `content` + `reasoning_content`，**tool_calls 全丢** → agent 永远不会调工具。

流式路径（`_chat_stream`）正常，所以这是个**流式/非流式的不对称**。
影响面覆盖所有返回 `reasoning_content` 的模型（MiniMax / DeepSeek-R1 / Qwen3-thinking…）。

**端点实测的第二个案例**（一次返回 4 个 tool_calls）同时触发了缺陷 #5。

---

## 4. 缺陷 #5 为什么只开 issue、不直接提 PR

修 id 唯一性有个**配对约束**：tool_call 的 id 必须与对应 tool 结果消息的 id 一致。
只改一侧会引入**静默错配**（tool 结果挂到错误的 tool call 上），比 400 更危险。

issue 里给了三种候选：

- **A** 按序号对称生成（`call_1`/`call_2`）—— 依赖"tool 消息紧跟其 assistant 消息"这一假设
- **B** 缺 `extra.function_id` 时直接报错 —— 最小最安全，但属破坏性变更
- **C** 只改 assistant 侧 —— **不推荐**（会静默错配）

让 maintainer 选，比我们自己拍板更专业。

---

## 5. 怎么提交

### 方式 A：脚本（推荐）

```bash
# 1) 建 token：GitHub → Settings → Developer settings → Personal access tokens
#    勾选 public_repo
export GH_TOKEN=ghp_xxxxxxxx

# 2) 先看流程不提交
.venv/Scripts/python.exe scripts/submit_upstream_pr.py --dry-run

# 3) 真的提交
.venv/Scripts/python.exe scripts/submit_upstream_pr.py
```

脚本会：fork → 把分支 push 到你的 fork → 开 PR → 开 issue。
token 通过 `GIT_ASKPASS` 传给 git，**不出现在命令行参数里**。

### 方式 B：手动

```bash
# 1) 网页 fork: https://github.com/QwenLM/Qwen-Agent/fork
# 2) 本地（已在临时 clone 里准备好分支 fix/keep-tool-calls-when-reasoning-content）
git clone https://github.com/<你的用户名>/Qwen-Agent.git
cd Qwen-Agent
git checkout -b fix/keep-tool-calls-when-reasoning-content
git apply ../docs/上游贡献/0001-fix-keep-tool-calls-with-reasoning-content.patch
git commit -m "fix(llm): keep tool_calls when a non-streaming reply also carries reasoning_content"
git push origin fix/keep-tool-calls-when-reasoning-content

# 3) 开 PR：https://github.com/QwenLM/Qwen-Agent/compare/main...<你的用户名>:fix/keep-tool-calls-when-reasoning-content
#    正文粘贴 docs/上游贡献/PR-修复non-streaming丢tool_calls.md
# 4) 开 issue：https://github.com/QwenLM/Qwen-Agent/issues/new
#    正文粘贴 docs/上游贡献/ISSUE-并行tool_call_id重复.md
```

---

## 6. 提交前自检（已做）

- [x] 补丁可干净应用到**干净上游**（`git apply --check` 通过）
- [x] 补丁前：`test_no_stream_keeps_tool_calls_when_reasoning_content_present` 失败，
      报错 `tool_calls were dropped by the reasoning_content branch`
- [x] 补丁后：#3 的两个测试通过；#5 的测试**仍失败**（故意留作 issue 的回归护栏）
- [x] 上游测试套件基线 50 failed（全部因缺 `DASHSCOPE_API_KEY` 的环境性失败）→ 补丁后**无新增失败**
- [x] 端到端验证：补丁后 `_chat_no_stream` 同时返回 `function_call` + `reasoning_content` + `extra.function_id`
