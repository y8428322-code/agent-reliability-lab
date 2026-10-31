#!/usr/bin/env bash
# 拉取上游 Qwen-Agent 源码。
# 本机到 github.com 的 git 协议被重置（Recv failure: Connection was reset），
# 因此按镜像优先级依次尝试，全部失败再退回 codeload tarball。
# 注意：脚本不执行递归删除，失败时保留临时目录供人工排查。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$ROOT/upstream"
DEST="$UP/Qwen-Agent"
REPO="QwenLM/Qwen-agent"

echo "[clone] 目标目录: $DEST"

if [ -f "$DEST/qwen_agent/llm/oai.py" ]; then
    echo "[clone] 上游源码已存在，跳过。需重装请手动删除: $DEST"
    exit 0
fi

mkdir -p "$UP"

try_git_clone() {
    local url="$1" label="$2" tmp="$UP/.clone-try-$3"
    echo "[clone] 尝试: $label"
    if git clone --depth 1 "$url" "$tmp" 2>&1 | tail -3; then
        if [ -f "$tmp/qwen_agent/llm/oai.py" ]; then
            mv "$tmp" "$DEST"
            echo "[clone] 成功（$label）"
            git -C "$DEST" rev-parse HEAD
            return 0
        fi
    fi
    echo "[clone] $label 未取到完整源码，临时目录保留在: $tmp"
    return 1
}

try_git_clone "https://gitclone.com/github.com/${REPO}.git" "gitclone.com 镜像" 1 \
    || try_git_clone "https://gh-proxy.com/https://github.com/${REPO}.git" "gh-proxy.com 转发" 2

if [ -f "$DEST/qwen_agent/llm/oai.py" ]; then
    exit 0
fi

echo "[clone] 尝试: codeload tarball"
TGZ="$UP/qwen-agent-main.tar.gz"
if curl -sL --max-time 300 -o "$TGZ" "https://codeload.github.com/${REPO}/tar.gz/refs/heads/main" \
   && tar -xzf "$TGZ" -C "$UP"; then
    if [ -d "$UP/Qwen-Agent-main/qwen_agent" ]; then
        mv "$UP/Qwen-Agent-main" "$DEST"
        echo "[clone] 成功（codeload tarball，无 git 历史）"
        exit 0
    fi
fi

echo "[clone] 全部尝试失败，请检查网络/镜像可用性。" >&2
exit 1
