#!/usr/bin/env bash
# 跑自建评测床。**多变体并行**，每个变体一个进程、各写一个 JSON，分析脚本可合并读取。
#
# 用法：
#   bash scripts/run_bench.sh                              # 全部变体，并行跑
#   bash scripts/run_bench.sh --variant A_baseline --variant B_context
#   bash scripts/run_bench.sh --trials 3
#   bash scripts/run_bench.sh --jobs 2                     # 限制并发数
#
# 并行的理由：单变体一轮 17 任务 × 3 次约 25 分钟，串行 5 个变体要 2 小时以上；
# 注意：并发太高会触发端点限流（实测 4 并发时 23/204 次运行被限流中断），
# 而中断若被当成"任务失败"会系统性压低分数。默认 2 并发。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$VPY" ]; then
    echo "[bench] 未找到虚拟环境，请先执行 bash scripts/setup_env.sh" >&2
    exit 1
fi

# 解析参数
VARIANTS=()
TRIALS=3
TASKS=()
JOBS=2
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANTS+=("$2"); shift 2;;
        --task)    TASKS+=("$2"); shift 2;;
        --trials)  TRIALS="$2"; shift 2;;
        --jobs)    JOBS="$2"; shift 2;;
        --out)     OUT="$2"; shift 2;;
        *) echo "未知参数: $1" >&2; exit 1;;
    esac
done

if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(A_baseline B_crop B_points B_context B_handwritten)
fi

# 密钥兜底：环境变量优先，其次从本机 Claude Code 配置读取
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
    echo "[bench] 未找到 API 密钥，请设置 MINIMAX_API_KEY（或 ANTHROPIC_AUTH_TOKEN），见 .env.example" >&2
    exit 1
fi

export PYTHONPATH="$ROOT/src"
export PYTHONIOENCODING=utf-8

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/results"
FILES=()

echo "[bench] 变体: ${VARIANTS[*]} | 重复: $TRIALS | 并发: $JOBS"

PIDS=()
for v in "${VARIANTS[@]}"; do
    f="$ROOT/results/bench_${v}_${STAMP}.json"
    FILES+=("$f")
    args=(--variant "$v" --trials "$TRIALS" --out "$f")
    for t in "${TASKS[@]}"; do args+=(--task "$t"); done
    "$VPY" -m qwen_agent_bench.bench "${args[@]}" > "$ROOT/results/bench_${v}_${STAMP}.log" 2>&1 &
    PIDS+=($!)
    # 控制并发
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "[bench] 各变体结果："
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then echo "   $f"; else echo "   [缺失] $f"; FAIL=1; fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "[bench] 汇总分析："
    "$VPY" "$ROOT/scripts/analyze.py" "${FILES[@]}"
fi
exit "$FAIL"
