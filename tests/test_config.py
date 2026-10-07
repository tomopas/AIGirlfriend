from __future__ import annotations

import os
from pathlib import Path

import pytest

from agf.config import Config

YAML = """\
ollama_url: http://example:11434
llm_model: my-model
nsfw: true
proactive_min_hours: 1
proactive_max_hours: 2
proactive_image_prob: 0.5
"""


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text(YAML)
    (tmp_path / ".env").write_text(
        "TELEGRAM_BOT_TOKEN=test-token\nALLOWED_CHAT_ID=123456\n"
    )
    for key in _ENV_CANDIDATES:
        os.environ.pop(key, None)
    return tmp_path


_ENV_CANDIDATES = [
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_CHAT_ID",
    "OLLAMA_URL",
    "COMFYUI_URL",
    "LLM_MODEL",
    "EMBED_MODEL",
    "IMAGE_WORKFLOW",
    "VIDEO_WORKFLOW",
    "NSFW",
    "PROACTIVE_MIN_HOURS",
    "PROACTIVE_MAX_HOURS",
    "PROACTIVE_IMAGE_PROB",
    "PROACTIVE_MIN_IDLE_MINUTES",
    "PROACTIVE_QUIET_START",
    "PROACTIVE_QUIET_END",
    "DATA_DIR",
    "PERSONA_PATH",
    "TEMPERATURE",
]


def test_defaults() -> None:
    cfg = Config()
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.llm_model == "dolphin-llama3"
    assert cfg.nsfw is True
    assert cfg.db_path == Path("data") / "memory.db"


def test_load_yaml_and_env(project_dir: Path) -> None:
    cfg = Config.load(config_path=str(project_dir / "config.yaml"), env_path=str(project_dir / ".env"))
    assert cfg.ollama_url == "http://example:11434"
    assert cfg.llm_model == "my-model"
    assert cfg.telegram_token == "test-token"
    assert cfg.allowed_chat_ids == [123456]
    assert cfg.proactive_min_hours == 1.0
    assert cfg.proactive_image_prob == 0.5


def test_env_overrides_yaml(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OLLAMA_URL=http://env:11434\n")
    os.environ.pop("OLLAMA_URL", None)
    cfg = Config.load(env_path=str(tmp_path / ".env"))
    assert cfg.ollama_url == "http://env:11434"


def test_allows_everyone(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ALLOWED_CHAT_ID=*\n")
    os.environ.pop("ALLOWED_CHAT_ID", None)
    cfg = Config.load(env_path=str(tmp_path / ".env"))
    assert cfg.allows_everyone is True
    assert cfg.allowed_chat_ids == []


def test_negative_group_ids(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ALLOWED_CHAT_ID=-123456, 789\n")
    for key in _ENV_CANDIDATES:
        if key != "ALLOWED_CHAT_ID":
            os.environ.pop(key, None)
    os.environ.pop("ALLOWED_CHAT_ID", None)
    cfg = Config.load(config_path=str(tmp_path / "missing.yaml"), env_path=str(tmp_path / ".env"))
    assert cfg.allowed_chat_ids == [-123456, 789]


def test_quiet_hours_defaults() -> None:
    cfg = Config()
    assert cfg.proactive_min_idle_minutes == 30.0
    assert cfg.proactive_quiet_start is None
    assert cfg.proactive_quiet_end is None