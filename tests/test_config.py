"""配置加载的单元测试。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_agent_demo.config import (  # noqa: E402
    DemoConfig,
    ModelConfig,
    default_config_path,
    load_app_config,
    repo_root,
)


class ConfigTest(unittest.TestCase):
    def test_repo_root_contains_configs(self):
        root = repo_root()
        self.assertTrue((root / "configs").is_dir())
        self.assertTrue(default_config_path().exists())

    def test_load_app_config_defaults(self):
        cfg = load_app_config()
        self.assertIsInstance(cfg.model, ModelConfig)
        self.assertIsInstance(cfg.demo, DemoConfig)
        self.assertEqual(cfg.model.model, "MiniMax-M3.1-Flash")
        self.assertEqual(cfg.model.model_type, "oai")

    def test_demo_scenarios_from_yaml(self):
        cfg = load_app_config()
        self.assertEqual(
            cfg.demo.enabled_scenarios, ["function_calling", "react_chat", "group_chat"]
        )

    def test_api_key_resolution_order(self):
        cfg = ModelConfig(
            model="m",
            base_url="http://x/v1",
            api_key_env="DEMO_PRIMARY_KEY",
            fallback_env="DEMO_FALLBACK_KEY",
        )
        os.environ.pop("DEMO_PRIMARY_KEY", None)
        os.environ.pop("DEMO_FALLBACK_KEY", None)
        self.assertIsNone(cfg.resolve_api_key())

        os.environ["DEMO_FALLBACK_KEY"] = "fallback-value"
        self.assertEqual(cfg.resolve_api_key(), "fallback-value")

        os.environ["DEMO_PRIMARY_KEY"] = "primary-value"
        self.assertEqual(cfg.resolve_api_key(), "primary-value")

        del os.environ["DEMO_PRIMARY_KEY"]
        del os.environ["DEMO_FALLBACK_KEY"]

    def test_to_chat_cfg_contains_base_url(self):
        cfg = ModelConfig(
            model="MiniMax-M3.1-Flash",
            base_url="https://api.minimaxi.com/v1",
            api_key_env="DEMO_KEY_X",
        )
        os.environ["DEMO_KEY_X"] = "secret"
        chat_cfg = cfg.to_chat_cfg()
        self.assertEqual(chat_cfg["model"], "MiniMax-M3.1-Flash")
        self.assertEqual(chat_cfg["model_type"], "oai")
        self.assertEqual(chat_cfg["base_url"], "https://api.minimaxi.com/v1")
        self.assertEqual(chat_cfg["api_key"], "secret")
        del os.environ["DEMO_KEY_X"]

    def test_missing_config_file(self):
        with self.assertRaises(SystemExit):
            load_app_config(Path("__not_exist__.yaml"))


if __name__ == "__main__":
    unittest.main()
