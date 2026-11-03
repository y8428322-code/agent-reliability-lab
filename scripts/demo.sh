#!/usr/bin/env bash
# 3 分钟看到全貌：环境自检 + 一次真实对话 + 打印已归档的消融结果。
#
#   bash scripts/demo.sh          # 默认（约 1~3 分钟，取决于模型响应速度）
#   bash scripts/demo.sh --full   # 额外跑一轮真实评测（约 30 分钟）
#
# 说明：完整版消融（多批次 × 多变体）要 2.5 小时以上，见 scripts/run_bench.sh。
# 本脚本**不重跑**全量实验，而是打印已归档的结果 —— 那些结果可用
# `scripts/summarize.py` 从原始 JSON 重新算出并核对。
set -u

# Windows 控制台默认 GBK，Python 输出 ✓/✗ 等字符会抛 UnicodeEncodeError
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/Scripts/python.exe"
FAIL=0

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- 0. 环境自检
say "[0/4] 环境自检"

if [ ! -x "$VPY" ]; then
    echo "  ✗ 未找到虚拟环境 .venv"
    echo "    请先执行: bash scripts/setup_env.sh"
    exit 1
fi
echo "  ✓ 虚拟环境就绪"

if [ ! -d "$ROOT/upstream/Qwen-Agent" ]; then
    echo "  ! 未找到上游源码，正在拉取（首次约 1~2 分钟）..."
    bash "$ROOT/scripts/clone_upstream.sh" || { echo "  ✗ 上游拉取失败"; exit 1; }
fi
echo "  ✓ 上游源码就绪: upstream/Qwen-Agent ($(git -C "$ROOT/upstream/Qwen-Agent" rev-parse --short HEAD 2>/dev/null || echo '?') )"

"$VPY" - <<'PY' || FAIL=1
import importlib, sys
for mod in ("qwen_agent", "yaml"):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        print("  ✗ 依赖缺失: %s (%s)" % (mod, exc))
        sys.exit(1)
print("  ✓ 依赖就绪 (qwen_agent, yaml)")
PY
[ "$FAIL" -eq 0 ] || exit 1

# ---------------------------------------------------------------- 1. 模型连通
say "[1/4] 模型服务连通性"

if [ -z "${MINIMAX_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    MINIMAX_API_KEY="$(py -3 -c "
import json, os
try:
    env = json.load(open(os.path.expanduser('~/.claude/settings.json'), encoding='utf-8')).get('env', {})
    print(env.get('ANTHROPIC_AUTH_TOKEN', ''))
except Exception:
    print('')
")"
    export MINIMAX_API_KEY
fi

if [ -z "${MINIMAX_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    echo "  ✗ 未找到 API 密钥。请设置 MINIMAX_API_KEY（见 .env.example）"
    exit 1
fi

PYTHONPATH="$ROOT/src" "$VPY" - <<'PY' || FAIL=1
from qwen_agent_demo.config import load_app_config
from qwen_agent_demo.model_factory import check_endpoint

cfg = load_app_config().model
try:
    tool = check_endpoint(cfg)
except Exception as exc:
    print("  ✗ 端点不可用: %s" % exc)
    raise SystemExit(1)
print("  ✓ 端点可用: %s | 模型 %s | 工具调用正常 (%s)" % (cfg.base_url, cfg.model, tool))
PY
[ "$FAIL" -eq 0 ] || exit 1

# ---------------------------------------------------------------- 2. 跑一次真实对话
say "[2/4] 真实对话（Function Calling 链路）"

PYTHONPATH="$ROOT/src" PYTHONIOENCODING=utf-8 "$VPY" - <<'PY' || FAIL=1
from qwen_agent_demo.config import load_app_config
from qwen_agent_demo.model_factory import build_llm
from qwen_agent_demo.tools import default_tools
from qwen_agent_demo.demo import run_and_report
from qwen_agent.agents import Assistant

cfg = load_app_config()
llm = build_llm(cfg.model)
bot = Assistant(
    llm=llm,
    name="计算助手",
    description="能调用本地工具完成数值与文本统计的助手",
    system_message=(
        "你是一个计算助手。需要求和时调用 sum_numbers，"
        "需要统计文本时调用 text_stats，然后用一句话给出结论。"
    ),
    function_list=default_tools(),
)
run_and_report(bot, [{"role": "user", "content": "请计算 12、30、8 这三个数的和，并用一句话告诉我结果。"}])
PY
[ "$FAIL" -eq 0 ] || exit 1

# ---------------------------------------------------------------- 3. 打印已归档结果
say "[3/4] 已归档的消融结果（可用 scripts/summarize.py 复算）"

if [ -f "$ROOT/results_summary/汇总.md" ]; then
    sed -n '/## 1. 主结果/,/^## 2/p' "$ROOT/results_summary/汇总.md" | head -20
    echo "  完整汇总: results_summary/汇总.md"
    echo "  图表:    results_summary/fig1_主结果.svg, fig2_批次间方差.svg, fig3_逐任务通过率.svg"
else
    echo "  ! 未找到 results_summary/汇总.md"
    echo "    可运行: .venv/Scripts/python.exe scripts/summarize.py   （需要 results/*.json）"
fi

# ---------------------------------------------------------------- 4. 可选：真实评测
say "[4/4] 真实评测"
if [ "${1:-}" = "--full" ]; then
    echo "  跑一轮评测（约 30 分钟）..."
    bash "$ROOT/scripts/run_bench.sh" --variant B_crop_keepredline --variant B_guard --trials 3 --jobs 2 || FAIL=1
else
    echo "  已跳过。执行以下命令可跑完整消融（约 2.5 小时，多批次）："
    echo "    bash scripts/run_bench.sh --variant B_crop_keepredline --variant B_guard --trials 3 --jobs 2"
    echo "  或加 --full 跑一轮（约 30 分钟）"
fi

printf '\n'
if [ "$FAIL" -eq 0 ]; then
    printf '\033[1;32m✔ 演示完成\033[0m\n'
else
    printf '\033[1;31m✘ 有步骤失败\033[0m\n'
fi
exit "$FAIL"
