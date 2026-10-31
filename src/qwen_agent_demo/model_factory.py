"""模型实例工厂：把配置装配成 qwen_agent 的 LLM 对象。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.base import BaseChatModel

from .config import ModelConfig


class MissingApiKeyError(RuntimeError):
    """未找到可用的 API 密钥。"""


def build_llm(cfg: ModelConfig) -> BaseChatModel:
    """按配置构造 LLM 实例。

    Raises:
        MissingApiKeyError: 环境变量中找不到任何可用密钥。
    """
    if not cfg.resolve_api_key():
        raise MissingApiKeyError(
            "未找到 API 密钥。请设置环境变量 %s（参见 .env.example）；"
            "或设置回退变量 %s。" % (cfg.api_key_env, cfg.fallback_env)
        )
    return get_chat_model(cfg.to_chat_cfg())


def check_endpoint(cfg: ModelConfig, timeout: Optional[int] = None) -> str:
    """直接对 OpenAI 兼容端点发一次带 tools 的请求，验证协议与工具调用。

    这是脱离框架的协议级回归基线：即使框架升级，底层协议变了这里也会先炸。
    """
    import json
    import urllib.request

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "What is 1+1? Use the tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % cfg.resolve_api_key(),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout or cfg.generate_cfg.get("timeout", 120)) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    choice = body["choices"][0]
    tool_calls = choice["message"].get("tool_calls") or []
    if not tool_calls:
        raise RuntimeError(
            "端点未返回 tool_calls，工具调用不可用。响应: %s" % json.dumps(choice, ensure_ascii=False)[:300]
        )
    return tool_calls[0]["function"]["name"]


def chat(
    cfg: ModelConfig,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: Optional[int] = None,
    retries: int = 3,
) -> str:
    """纯文本对话（不传 tools），用于评判器 / 反思等旁路调用。

    带重试：批量跑评测时连续请求容易触发端点限流/超时，
    判定失败必须与「断言不成立」区分开，否则会把基础设施抖动误判成模型能力问题。
    """
    import json
    import time
    import urllib.error
    import urllib.request

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % cfg.resolve_api_key(),
        },
        method="POST",
    )
    limit = timeout or cfg.generate_cfg.get("timeout", 120)
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=limit) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (body["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("模型调用失败(重试 %d 次): %s" % (retries, last_error))
