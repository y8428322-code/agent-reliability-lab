#!/usr/bin/env bash
# 发布本项目到 GitHub，并推上游修复分支。
#
# 前置（一次性）：
#   1) 把 ~/.ssh/id_rsa.pub 贴到 GitHub → Settings → SSH and GPG keys
#   2) 在 https://github.com/new 建一个**空**仓库（不要勾 README/.gitignore/license）
#      建议名：agent-reliability-lab
#   3) fork 上游：https://github.com/QwenLM/Qwen-Agent/fork
#
# 用法：
#   bash scripts/publish.sh                       # 推本项目 + 推上游分支
#   bash scripts/publish.sh --project-only         # 只推本项目
#   bash scripts/publish.sh --upstream-only       # 只推上游分支
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GH_USER="${GH_USER:-y8428322-code}"
PROJECT_REPO="${PROJECT_REPO:-agent-reliability-lab}"
UPSTREAM_REPO="QwenLM/Qwen-Agent"
BRANCH="fix/keep-tool-calls-when-reasoning-content"
LOCAL_CLONE="${LOCAL_CLONE:-/tmp/_upstream_test}"

MODE="all"
case "${1:-}" in
    --project-only) MODE="project";;
    --upstream-only) MODE="upstream";;
esac

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# ------------------------------------------------------------ 上传本项目
if [ "$MODE" = "all" ] || [ "$MODE" = "project" ]; then
    say "[1/3] 推本项目到 github.com/$GH_USER/$PROJECT_REPO"
    if ! ssh -o BatchMode=yes -T git@github.com >/dev/null 2>&1; then
        echo "  ✗ SSH 未就绪。请先把 ~/.ssh/id_rsa.pub 加到 GitHub 的 SSH keys。"
        exit 1
    fi
    if git remote get-url origin >/dev/null 2>&1; then
        git remote set-url origin "git@github.com:$GH_USER/$PROJECT_REPO.git"
    else
        git remote add origin "git@github.com:$GH_USER/$PROJECT_REPO.git"
    fi
    git push -u origin HEAD 2>&1 | tail -3
    echo "  → https://github.com/$GH_USER/$PROJECT_REPO"
fi

# ------------------------------------------------------------ 推上游分支
if [ "$MODE" = "all" ] || [ "$MODE" = "upstream" ]; then
    say "[2/3] 推上游修复分支到你的 fork"
    if [ ! -d "$LOCAL_CLONE/.git" ]; then
        echo "  ! 未找到本地上游 clone（$LOCAL_CLONE），先克隆："
        echo "      git clone https://github.com/$GH_USER/Qwen-Agent.git $LOCAL_CLONE"
        exit 1
    fi
    cd "$LOCAL_CLONE"
    git remote set-url fork "git@github.com:$GH_USER/Qwen-Agent.git" 2>/dev/null \
        || git remote add fork "git@github.com:$GH_USER/Qwen-Agent.git"
    git push fork "$BRANCH:refs/heads/$BRANCH" 2>&1 | tail -3
    cd "$ROOT"
fi

# ------------------------------------------------------------ 给出提 PR 的入口
say "[3/3] 开 PR / issue 的入口"
cat <<EOF

  PR  （正文在 docs/上游贡献/PR-修复non-streaming丢tool_calls.md）
    https://github.com/$UPSTREAM_REPO/compare/main...$GH_USER:$BRANCH?expand=1
    打开后把 PR 正文文件内容整段粘进 Description 即可。

  Issue（正文在 docs/上游贡献/ISSUE-并行tool_call_id重复.md）
    https://github.com/$UPSTREAM_REPO/issues/new
    标题用：Parallel tool calls without extra.function_id get duplicate tool_call ids (400 duplicate tool_call id)

EOF
