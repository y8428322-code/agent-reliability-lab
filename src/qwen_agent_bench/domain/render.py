"""把业务规范（单一事实源）渲染成不同版式。

这是 P4-2 的核心：**同一份规则，渲染成两种版式**，差异 100% 来自呈现方式，
从而把「上下文工程」的收益干净地归因给呈现方式，而不是人工撰写内容的差异。

- :func:`render_narrative`   长叙述版 —— 规则散落在段落里，判据不突出（变体 A）
- :func:`render_structured` 结构化版 —— 红线前置 + 分节 + 关键判据突出 + 按场景裁剪（变体 B）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .db import domain_dir


def load_policy_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取规则单一事实源。"""
    p = Path(path) if path else domain_dir() / "policy_rules.yaml"
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rules_by_category(spec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rule in spec.get("rules", []):
        out.setdefault(rule["category"], []).append(rule)
    return out


def render_narrative(spec: Dict[str, Any]) -> str:
    """长叙述版：按分节把规则正文拼成长段落。

    刻意**不做**任何结构化优化：红线不置顶、判据不单列、不按场景裁剪、
    规则混在段落里 —— 这正是「原始 policy」的样子。
    """
    titles = {s["id"]: s["title"] for s in spec.get("sections", [])}
    lines: List[str] = []
    for category, rules in _rules_by_category(spec).items():
        lines.append("## %s" % titles.get(category, category))
        lines.append("")
        for rule in rules:
            lines.append(rule["body"].strip())
            lines.append("")
    return "\n".join(lines).strip()


def render_structured(spec: Dict[str, Any], scenes: Optional[Sequence[str]] = None) -> str:
    """结构化版：红线前置 + 按场景分节 + 关键判据要点化。

    Args:
        scenes: 任务关心的分节 id（如 ``["identity", "repair", "speech"]``）。
            ``None`` 表示全部分节都要。红线**始终**包含。
    """
    wanted = set(scenes or [])
    titles = {s["id"]: s["title"] for s in spec.get("sections", [])}
    rules = spec.get("rules", [])

    parts: List[str] = []

    redlines = [r for r in rules if r.get("redline")]
    if redlines:
        parts.append("## 0. 红线（任何场景都适用，违反即判失败）")
        for rule in redlines:
            parts.append("")
            parts.append("**%s %s**" % (rule["id"], rule["title"]))
            for item in rule.get("criteria", []):
                parts.append("- %s" % item)
        parts.append("")

    by_cat = _rules_by_category(spec)
    for category, cat_rules in by_cat.items():
        if wanted and category not in wanted:
            continue
        shown = [r for r in cat_rules if not r.get("redline")]
        if not shown:
            continue
        parts.append("## %s" % titles.get(category, category))
        for rule in shown:
            parts.append("")
            parts.append("**%s %s**" % (rule["id"], rule["title"]))
            for item in rule.get("criteria", []):
                parts.append("- %s" % item)
        parts.append("")

    return "\n".join(parts).strip()


def render_all(spec: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """两种版式的渲染结果（便于导出成文件供人工阅读）。"""
    spec = spec or load_policy_rules()
    return {
        "narrative": render_narrative(spec),
        "structured": render_structured(spec, scenes=None),
    }


def export_to_files(out_dir: Optional[Path] = None) -> List[Path]:
    """把两种渲染结果导出为 markdown 文件（生成产物，非手工维护）。"""
    spec = load_policy_rules()
    target = Path(out_dir) if out_dir else domain_dir()
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in render_all(spec).items():
        path = target / ("policy_%s.md" % name)
        path.write_text(text + "\n", encoding="utf-8")
        written.append(path)
    return written
