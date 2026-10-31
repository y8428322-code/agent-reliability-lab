#!/usr/bin/env bash
# 建立项目虚拟环境并安装依赖（幂等）。
# 注意：本机 `python` / `pip` 是应用商店占位程序，必须用 `py -3.12` 或 venv 内的解释器。
# 网络：本机到 pypi.org 直连很慢（索引页 30s+），默认走阿里云镜像，可用 PIP_INDEX_URL 覆盖。
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYVER="${PYVER:-3.12}"
INDEX="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
FALLBACK_INDEX="https://pypi.org/simple"

if [ ! -d "$ROOT/.venv" ]; then
    echo "[env] 创建虚拟环境 (Python $PYVER)"
    py "-$PYVER" -m venv "$ROOT/.venv"
else
    echo "[env] 虚拟环境已存在，跳过创建"
fi

VPY="$ROOT/.venv/Scripts/python.exe"

pip_install() {
    # 优先走镜像，失败再退回官方源
    "$VPY" -m pip install -i "$INDEX" --timeout 60 "$@" \
        || "$VPY" -m pip install -i "$FALLBACK_INDEX" --timeout 60 "$@"
}

echo "[env] 补齐构建工具 (index=$INDEX)"
pip_install --upgrade pip setuptools wheel

echo "[env] 安装上游 Qwen-Agent（源码 editable 安装，保证与 upstream 源码一致）"
pip_install -e "$ROOT/upstream/Qwen-Agent"

echo "[env] 安装本项目额外依赖"
pip_install -r "$ROOT/requirements.txt"

echo "[env] 自检"
"$VPY" -c "import qwen_agent, yaml; print('qwen_agent', getattr(qwen_agent, '__version__', 'ok')); print('yaml ok')"
echo "[env] 完成。"
