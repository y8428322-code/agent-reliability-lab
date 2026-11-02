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


def render_narrative(spec: Dict[str, Any], scenes: Optional[Sequence[str]] = None) -> str:
    """长叙述版：按分节把规则正文拼成长段落。

    刻意**不做**任何结构化优化：判据不单列、红线不置顶、规则混在段落里。

    Args:
        scenes: 若给定，则**只渲染这些分节**（裁剪），否则全量。
    """
    wanted = set(scenes or [])
    titles = {s["id"]: s["title"] for s in spec.get("sections", [])}
    lines: List[str] = []
    for category, rules in _rules_by_category(spec).items():
        if wanted and category not in wanted:
            continue
        lines.append("## %s" % titles.get(category, category))
        lines.append("")
        for rule in rules:
            lines.append(rule["body"].strip())
            lines.append("")
    return "\n".join(lines).strip()


def render_structured(
    spec: Dict[str, Any],
    scenes: Optional[Sequence[str]] = None,
    redline_first: bool = True,
    keep_redline: bool = True,
    keep_universal: bool = True,
) -> str:
    """结构化版：判据要点化 + 可选的场景裁剪 + 可选的红线/通用约束保留。

    这里刻意把两件事**解耦**：

    - **是否保留**（``keep_redline`` / ``keep_universal``）——决定这条规则会不会被场景裁掉
    - **是否置顶**（``redline_first``）——只决定它在上下文里的位置

    v6 之前把两者混在一起（``redline_first`` 一关，红线也跟着被裁），
    导致"按场景裁剪"的伤害被严重高估、机制被误判。

    Args:
        scenes: 任务关心的分节 id。``None`` 表示全量。
        redline_first: 保留块放在最前（True）还是最后（False）。
        keep_redline: 红线规则是否**永不被裁掉**。
        keep_universal: 跨场景通用约束（``universal: true``）是否**永不被裁掉**。
    """
    wanted = set(scenes or [])
    titles = {s["id"]: s["title"] for s in spec.get("sections", [])}
    rules = spec.get("rules", [])

    always: List[Dict[str, Any]] = []
    if keep_redline:
        always.extend(r for r in rules if r.get("redline"))
    if keep_universal:
        always.extend(
            r for r in rules if r.get("universal") and not r.get("redline")
        )
    always_ids = {id(r) for r in always}

    def _block(items: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for rule in items:
            out.append("")
            out.append("**%s %s**" % (rule["id"], rule["title"]))
            for item in rule.get("criteria", []):
                out.append("- %s" % item)
        return out

    head: List[str] = []
    tail: List[str] = []
    if always:
        title = "## 0. 红线与通用约束（任何场景都适用）"
        if redline_first:
            head.append(title)
            head.extend(_block(always))
            head.append("")
        else:
            tail.append(title)
            tail.extend(_block(always))
            tail.append("")

    body: List[str] = []
    by_cat = _rules_by_category(spec)
    for category, cat_rules in by_cat.items():
        if wanted and category not in wanted:
            continue
        shown = [r for r in cat_rules if id(r) not in always_ids]
        if not shown:
            continue
        body.append("## %s" % titles.get(category, category))
        body.extend(_block(shown))
        body.append("")

    if always and not redline_first:
        pass

    return "\n".join(head + body + tail).strip()


def render(
    spec: Dict[str, Any],
    scenes: Optional[Sequence[str]] = None,
    points: bool = False,
    redline_first: bool = True,
    keep_redline: bool = True,
    keep_universal: bool = True,
) -> str:
    """统一入口：把「上下文工程」的子机制做成**正交开关**。

    | 子机制 | 开关 | 含义 |
    |---|---|---|
    | 裁剪 | ``scenes`` 非空 | 只注入与本任务相关的规则 |
    | 要点化 | ``points=True`` | 规则呈现为要点列表，而不是长段落 |
    | 置顶 | ``redline_first=True`` | 红线/通用约束块放在上下文最前 |
    | **保红线** | ``keep_redline=True`` | 红线永不被场景裁掉 |
    | **保通用约束** | ``keep_universal=True`` | 跨场景通用约束永不被裁掉 |

    「保留」与「置顶」是**两件事**：v6 之前把两者混在一起，
    导致"按场景裁剪"的伤害被高估、机制被误判。
    """
    if points:
        return render_structured(
            spec,
            scenes=scenes,
            redline_first=redline_first,
            keep_redline=keep_redline,
            keep_universal=keep_universal,
        )
    return render_narrative(spec, scenes=scenes)


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
