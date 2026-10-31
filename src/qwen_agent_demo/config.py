"""模型接入配置加载（YAML + 环境变量）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def repo_root() -> Path:
    """项目根目录。"""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """默认配置文件路径。"""
    return repo_root() / "configs" / "model.yaml"


@dataclass
class ModelConfig:
    """模型接入配置。"""

    model: str
    base_url: str
    api_key_env: str = "MINIMAX_API_KEY"
    fallback_env: Optional[str] = None
    model_type: str = "oai"
    generate_cfg: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.generate_cfg is None:
            self.generate_cfg = {}

    def resolve_api_key(self) -> Optional[str]:
        """按 api_key_env -> fallback_env 的顺序取密钥，取不到返回 None。"""
        for name in (self.api_key_env, self.fallback_env):
            if not name:
                continue
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return None

    def to_chat_cfg(self) -> Dict[str, Any]:
        """转成 qwen_agent.llm.get_chat_model 接受的配置字典。

        TextChatAtOAI 原生读取 cfg['base_url'] 与 cfg['api_key']，
        因此任意 OpenAI 兼容服务都可以直接接入。

        注意：采样参数必须放在 cfg['generate_cfg'] 里（上游 BaseChatModel 只从
        generate_cfg 取 API 级参数）；顶层放 'stream' 会被静默忽略，且若塞进
        generate_cfg 会与上游显式传入的 stream=True 冲突。
        """
        cfg: Dict[str, Any] = {
            "model": self.model,
            "model_type": self.model_type,
        }
        if self.base_url:
            cfg["base_url"] = self.base_url
        api_key = self.resolve_api_key()
        if api_key:
            cfg["api_key"] = api_key
        if self.generate_cfg:
            cfg["generate_cfg"] = dict(self.generate_cfg)
        return cfg


@dataclass
class DemoConfig:
    """演示场景开关。"""

    function_calling: bool = True
    react_chat: bool = True
    group_chat: bool = True

    @property
    def enabled_scenarios(self) -> List[str]:
        names = []
        if self.function_calling:
            names.append("function_calling")
        if self.react_chat:
            names.append("react_chat")
        if self.group_chat:
            names.append("group_chat")
        return names


@dataclass
class AppConfig:
    """完整应用配置。"""

    model: ModelConfig
    demo: DemoConfig


def load_app_config(path: Optional[Path] = None) -> AppConfig:
    """读取 YAML 配置。"""
    cfg_path = Path(path) if path else default_config_path()
    if not cfg_path.exists():
        raise SystemExit("配置文件不存在: %s" % cfg_path)

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model_raw = raw.get("model") or {}
    if not model_raw.get("model"):
        raise SystemExit("configs/model.yaml 缺少 model.model 字段")

    model_cfg = ModelConfig(
        model=model_raw["model"],
        base_url=model_raw.get("base_url", ""),
        api_key_env=model_raw.get("api_key_env", "MINIMAX_API_KEY"),
        fallback_env=model_raw.get("fallback_env"),
        model_type=model_raw.get("model_type", "oai"),
        generate_cfg=model_raw.get("generate_cfg") or {},
    )

    demo_raw = raw.get("demo") or {}
    demo_cfg = DemoConfig(
        function_calling=bool(demo_raw.get("function_calling", True)),
        react_chat=bool(demo_raw.get("react_chat", True)),
        group_chat=bool(demo_raw.get("group_chat", True)),
    )

    return AppConfig(model=model_cfg, demo=demo_cfg)
