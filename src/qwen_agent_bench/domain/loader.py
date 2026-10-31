"""域数据加载：任务集、策略文本、策略分节。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import domain_dir


def load_tasks(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取任务集。"""
    task_path = Path(path) if path else domain_dir() / "tasks.json"
    with task_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_task_by_id(task_id: str, path: Optional[Path] = None) -> Dict[str, Any]:
    for task in load_tasks(path):
        if task["id"] == task_id:
            return task
    raise KeyError("任务不存在: %s" % task_id)


def load_policy_raw(path: Optional[Path] = None) -> str:
    """读取原始策略全文（基线变体使用）。"""
    p = Path(path) if path else domain_dir() / "policy.md"
    return p.read_text(encoding="utf-8")


def load_policy_digest(path: Optional[Path] = None) -> str:
    """读取结构化策略摘要全文。"""
    p = Path(path) if path else domain_dir() / "policy_digest.md"
    return p.read_text(encoding="utf-8")


_SECTION_RE = re.compile(r"<!--\s*section:([a-z_]+)\s*-->")


def split_policy_sections(text: str) -> Dict[str, str]:
    """把带 <!-- section:xxx --> 标记的摘要拆成 {section_name: 正文}。"""
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # 去掉该节的一级标题行（保留内容）
        lines = [ln for ln in body.splitlines() if not ln.startswith("# ")]
        sections[match.group(1)] = "\n".join(lines).strip()
    return sections


def build_digest_context(section_names: List[str], digest_path: Optional[Path] = None) -> str:
    """按任务类型裁剪策略摘要：红线永远保留，其余按需注入。"""
    sections = split_policy_sections(load_policy_digest(digest_path))
    ordered = ["redline"] + [s for s in section_names if s != "redline"]
    parts: List[str] = []
    for name in ordered:
        body = sections.get(name)
        if not body:
            continue
        parts.append(body)
    return "\n\n".join(parts)
