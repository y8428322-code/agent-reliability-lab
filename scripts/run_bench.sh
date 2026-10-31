#!/usr/bin/env bash
# 跑自建评测床。
# 用法：
#   bash scripts/run_bench.sh                          # 三个变体 × 6 个任务 × 1 次
#   bash scripts/run_bench.sh --variant A_baseline      # 只跑基线
#   bash scripts/run_bench.sh --trials 3                # 每个任务重复 3 次（Pass^3 口径）
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$VPY" ]; then
    echo "[bench] 未找到虚拟环境，请先执行 bash scripts/setup_env.sh" >&2
    exit 1
fi

# 密钥兜底：环境变量 -> 本机 Claude Code 配置
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

if [ -z "${MINIMAX_API_KEY:-}" ]; then
    echo "[bench] 未找到 API 密钥，请设置 MINIMAX_API_KEY（见 .env.example）" >&2
    exit 1
fi

export PYTHONPATH="$ROOT/src"
export PYTHONIOENCODING=utf-8

OUT="$ROOT/results/ablation_$(date +%Y%m%d_%H%M%S).json"
echo "[bench] 结果将写入: $OUT"
exec "$VPY" -m qwen_agent_bench.bench --out "$OUT" "$@"
