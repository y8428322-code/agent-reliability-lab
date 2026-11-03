"""把本地分支推到 GitHub fork 并向上游开 PR / issue。

需要环境变量：
    GH_TOKEN      GitHub personal access token（需 repo 权限）
    可选 GH_USERNAME  你的 GitHub 用户名（不填则自动查）

用法：
    # 先看会做什么（不提交）
    .venv/Scripts/python.exe scripts/submit_upstream_pr.py --dry-run

    # 真的提交
    .venv/Scripts/python.exe scripts/submit_upstream_pr.py

流程：
    1. 在你的账号下 fork QwenLM/Qwen-Agent（没有则建）
    2. 把本地分支 push 到 fork
    3. 用 API 开 PR（或 issue）

注意：token 通过 GIT_ASKPASS 传给 git，不出现在命令行参数里。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "QwenLM/Qwen-Agent"
API = "https://api.github.com"
DOCS = ROOT / "docs" / "上游贡献"

DEFAULT_PR_BODY = DOCS / "PR-修复non-streaming丢tool_calls.md"
DEFAULT_ISSUE_BODY = DOCS / "ISSUE-并行tool_call_id重复.md"


def api(method: str, path: str, token: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = path if path.startswith("http") else API + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "qwen-agent-fix-submitter",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit("GitHub API %s %s 失败 (%s):\n%s" % (method, path, exc.code, detail[:600]))


def current_branch() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT), capture_output=True, text=True
    )
    return out.stdout.strip()


def ensure_fork(token: str, username: str) -> str:
    """确保 fork 存在，返回 fork 的 full_name。"""
    try:
        api("GET", "/repos/%s/forks" % UPSTREAM, token)
        print("[fork] 已存在 fork")
    except SystemExit:
        pass
    try:
        api("GET", "/repos/%s/forks" % UPSTREAM, token)
    except SystemExit:
        print("[fork] 创建 fork ...")
        api("POST", "/repos/%s/forks" % UPSTREAM, token, {"organization": None})
    # 轮询等待 fork 就绪
    for _ in range(20):
        try:
            info = api("GET", "/repos/%s" % ("%s/%s" % (username, UPSTREAM.split("/")[1])), token)
            if info.get("full_name"):
                print("[fork] 就绪: %s" % info["full_name"])
                return info["full_name"]
        except SystemExit:
            pass
        import time

        time.sleep(3)
    raise SystemExit("fork 创建超时，请到网页确认: https://github.com/%s/%s" % (username, UPSTREAM.split("/")[1]))


def push_branch(token: str, fork_full_name: str, branch: str, dry_run: bool) -> None:
    url = "https://github.com/%s.git" % fork_full_name
    print("[push] %s -> %s (%s)" % (branch, fork_full_name, "dry-run" if dry_run else "实际推送"))
    if dry_run:
        return
    # 用 GIT_ASKPASS 传 token，避免出现在命令行/进程列表里
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nprintf '%%s\\n' \"$GH_TOKEN\"\n")
        askpass = f.name
    os.chmod(askpass, 0o700)
    env = dict(os.environ)
    env["GIT_ASKPASS"] = askpass
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GH_TOKEN"] = token
    try:
        subprocess.run(
            ["git", "push", url, "%s:refs/heads/%s" % (branch, branch)],
            cwd=str(ROOT),
            env=env,
            check=True,
        )
    finally:
        os.unlink(askpass)


def open_pr(token: str, username: str, branch: str, title: str, body: str, dry_run: bool) -> None:
    payload = {
        "title": title,
        "head": "%s:%s" % (username, branch),
        "base": "main",
        "body": body,
    }
    print("[pr] 标题: %s" % title)
    if dry_run:
        print("[pr] dry-run，未创建。payload:\n%s" % json.dumps(payload, ensure_ascii=False, indent=2)[:1200])
        return
    res = api("POST", "/repos/%s/pulls" % UPSTREAM, token, payload)
    print("[pr] 已创建: %s" % res.get("html_url"))


def open_issue(token: str, title: str, body: str, dry_run: bool) -> None:
    payload = {"title": title, "body": body}
    print("[issue] 标题: %s" % title)
    if dry_run:
        print("[issue] dry-run，未创建。")
        return
    res = api("POST", "/repos/%s/issues" % UPSTREAM, token, payload)
    print("[issue] 已创建: %s" % res.get("html_url"))


def main() -> None:
    parser = argparse.ArgumentParser(description="向上游 Qwen-Agent 提交 PR / issue")
    parser.add_argument("--dry-run", action="store_true", help="只演示流程，不真正提交")
    parser.add_argument("--branch", default="", help="要推送的本地分支（默认当前分支）")
    parser.add_argument("--pr-title", default="fix(llm): keep tool_calls when a non-streaming reply also carries reasoning_content")
    parser.add_argument("--pr-body", default=str(DEFAULT_PR_BODY))
    parser.add_argument("--issue-title", default="Parallel tool calls without extra.function_id get duplicate tool_call ids (400 duplicate tool_call id)")
    parser.add_argument("--issue-body", default=str(DEFAULT_ISSUE_BODY))
    parser.add_argument("--skip-issue", action="store_true", help="只提 PR，不开 issue")
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "未找到 GH_TOKEN。\n"
            "  GitHub → Settings → Developer settings → Personal access tokens\n"
            "  勾选 public_repo（fork + PR 需要），然后：\n"
            "      export GH_TOKEN=ghp_xxx"
        )

    username = os.environ.get("GH_USERNAME", "").strip()
    if not username:
        me = api("GET", "/user", token)
        username = me["login"]
    print("[whoami] %s" % username)

    branch = args.branch or current_branch()
    print("[branch] %s" % branch)

    fork_full_name = ensure_fork(token, username)
    push_branch(token, fork_full_name, branch, args.dry_run)
    open_pr(
        token,
        username,
        branch,
        args.pr_title,
        Path(args.pr_body).read_text(encoding="utf-8"),
        args.dry_run,
    )
    if not args.skip_issue:
        open_issue(
            token,
            args.issue_title,
            Path(args.issue_body).read_text(encoding="utf-8"),
            args.dry_run,
        )
    print("\n完成。")


if __name__ == "__main__":
    main()
