#!/usr/bin/env bash
# 一键复现验证：环境 -> 单元测试 -> 端到端 demo。
# 用法：bash scripts/verify.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VPY="$ROOT/.venv/Scripts/python.exe"
FAIL=0

step() {
    echo ""
    echo "============================================================"
    echo ">>> $1"
    echo "============================================================"
}

step "1/4 检查上游源码"
if [ -f "$ROOT/upstream/Qwen-Agent/qwen_agent/llm/oai.py" ]; then
    echo "OK  上游源码存在: upstream/Qwen-Agent"
    echo "    commit: $(git -C "$ROOT/upstream/Qwen-Agent" rev-parse HEAD 2>/dev/null || echo '无 git 历史')"
else
    echo "FAIL 上游源码缺失，请先执行 bash scripts/clone_upstream.sh"
    FAIL=1
fi

step "2/4 检查虚拟环境"
if [ ! -x "$VPY" ]; then
    echo "FAIL 虚拟环境不存在，请先执行 bash scripts/setup_env.sh"
    exit 1
fi
"$VPY" -c "import qwen_agent; print('OK  qwen_agent 可导入')" || FAIL=1

step "3/4 单元测试"
"$VPY" -m unittest discover -s "$ROOT/tests" -v || FAIL=1

step "4/4 端到端 demo（真实调用模型服务）"
export PYTHONPATH="$ROOT/src"
export PYTHONIOENCODING=utf-8
"$VPY" -m qwen_agent_demo.demo || FAIL=1

echo ""
echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
    echo ">>> 全部通过"
else
    echo ">>> 存在失败项，见上方输出"
fi
echo "============================================================"
exit "$FAIL"
