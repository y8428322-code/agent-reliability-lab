"""自建 agent 评测床（零新增依赖）。

设计规格见 docs/评测床设计.md。借鉴 τ²-bench 的任务 schema 与判分思想，
不引入其代码与依赖，只用已装的 Qwen-Agent + 标准库。
"""

__version__ = "0.1.0"
