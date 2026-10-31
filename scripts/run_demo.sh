#!/usr/bin/env bash
# 运行端到端 demo（自动准备环境变量）。
#
# 密钥取值：环境变量 MINIMAX_API_KEY > ANTHROPIC_AUTH_TOKEN > 本机 ~/.claude/settings.json 里的 env.ANTHROPIC_AUTH_TOKEN
# 用法：bash scripts/run_demo.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/Scripts/python.exe"

if [ ! -x "$VPY" ]; then
    echo "[demo] 未找到虚拟环境，请先执行 bash scripts/setup_env.sh" >&2
    exit 1
fi

# 最后兜底：从本机 Claude Code 配置里读取同一把 key（不打印内容）
if [ -z "${MINIMAX_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    SETTINGS="$HOME/.claude/settings.json"
    if [ -f "$SETTINGS" ]; then
        MINIMAX_API_KEY="$(py -3 -c "
import json, os, sys
try:
    env = json.load(open(os.path.expanduser('~/.claude/settings.json'), encoding='utf-8')).get('env', {})
    print(env.get('ANTHROPIC_AUTH_TOKEN', ''))
except Exception:
    print('')
")"
        export MINIMAX_API_KEY
    fi
fi

if [ -z "${MINIMAX_API_KEY:-}" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
    echo "[demo] 未找到 API 密钥，请设置 MINIMAX_API_KEY（或 ANTHROPIC_AUTH_TOKEN），见 .env.example" >&2
    exit 1
fi

export PYTHONPATH="$ROOT/src"
export PYTHONIOENCODING=utf-8

exec "$VPY" -m qwen_agent_demo.demo "$@"
