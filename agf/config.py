from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)


def load_dotenv(path: str | os.PathLike) -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_ENV_KEYS = {
    "TELEGRAM_BOT_TOKEN": "telegram_token",
    "ALLOWED_CHAT_ID": "allowed_chat_id",
    "OLLAMA_URL": "ollama_url",
    "COMFYUI_URL": "comfyui_url",
    "LLM_MODEL": "llm_model",
    "EMBED_MODEL": "embed_model",
    "IMAGE_WORKFLOW": "image_workflow",
    "VIDEO_WORKFLOW": "video_workflow",
    "NSFW": "nsfw",
    "PROACTIVE_MIN_HOURS": "proactive_min_hours",
    "PROACTIVE_MAX_HOURS": "proactive_max_hours",
    "PROACTIVE_IMAGE_PROB": "proactive_image_prob",
    "PROACTIVE_MIN_IDLE_MINUTES": "proactive_min_idle_minutes",
    "PROACTIVE_QUIET_START": "proactive_quiet_start",
    "PROACTIVE_QUIET_END": "proactive_quiet_end",
    "DATA_DIR": "data_dir",
    "PERSONA_PATH": "persona_path",
    "TEMPERATURE": "temperature",
}


@dataclass
class Config:
    telegram_token: Optional[str] = None
    allowed_chat_id: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    comfyui_url: str = "http://localhost:8188"
    llm_model: str = "dolphin-llama3"
    embed_model: str = "nomic-embed-text"
    image_workflow: Optional[str] = None
    video_workflow: Optional[str] = None
    nsfw: bool = True
    proactive_min_hours: float = 2.0
    proactive_max_hours: float = 6.0
    proactive_image_prob: float = 0.3
    proactive_min_idle_minutes: float = 30.0
    proactive_quiet_start: Optional[int] = None
    proactive_quiet_end: Optional[int] = None
    data_dir: str = "data"
    persona_path: str = "persona.yaml"
    temperature: float = 0.85

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "memory.db"

    @property
    def allowed_chat_ids(self) -> list[int]:
        raw = (self.allowed_chat_id or "").strip()
        if not raw:
            return []
        ids: list[int] = []
        for part in raw.replace(";", ",").replace(" ", ",").split(","):
            part = part.strip()
            if not part or part == "*":
                continue
            sign = 1
            digits = part
            if digits.startswith(("+", "-")):
                if digits[0] == "-":
                    sign = -1
                digits = digits[1:]
            if digits.isdigit():
                try:
                    ids.append(sign * int(digits))
                except ValueError:
                    continue
        # de-dup preserving order
        seen: set[int] = set()
        out: list[int] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    @property
    def allows_everyone(self) -> bool:
        raw = (self.allowed_chat_id or "").strip().lower()
        return raw.startswith("*")

    @classmethod
    def load(cls, config_path: str = "config.yaml", env_path: str = ".env") -> "Config":
        load_dotenv(env_path)
        raw: dict = {}
        config_file = Path(config_path)
        if config_file.exists():
            loaded = yaml.safe_load(config_file.read_text())
            if isinstance(loaded, dict):
                raw = loaded
        cfg = cls()
        for key, value in raw.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        for env, attr in _ENV_KEYS.items():
            value = os.environ.get(env)
            if value is None or (isinstance(value, str) and value.strip() == "" and attr not in ("allowed_chat_id",)):
                continue
            current = getattr(cfg, attr)
            try:
                if isinstance(current, bool):
                    setattr(cfg, attr, value.strip().lower() in ("1", "true", "yes", "on"))
                elif isinstance(current, float):
                    setattr(cfg, attr, float(value))
                elif current is None and attr in ("proactive_quiet_start", "proactive_quiet_end"):
                    v = value.strip()
                    setattr(cfg, attr, int(v) if v not in ("", "none", "null") else None)
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(cfg, attr, int(float(value)))
                else:
                    if attr in ("image_workflow", "video_workflow") and isinstance(value, str) and value.strip() == "":
                        setattr(cfg, attr, None)
                    else:
                        setattr(cfg, attr, value)
            except (ValueError, AttributeError):
                continue
        # normalize empty workflow strings to None
        if isinstance(cfg.image_workflow, str) and not cfg.image_workflow.strip():
            cfg.image_workflow = None
        if isinstance(cfg.video_workflow, str) and not cfg.video_workflow.strip():
            cfg.video_workflow = None
        # sanity: swap if min > max
        if cfg.proactive_min_hours > cfg.proactive_max_hours:
            cfg.proactive_min_hours, cfg.proactive_max_hours = cfg.proactive_max_hours, cfg.proactive_min_hours
        # warn on unknown yaml keys (typos are otherwise silent)
        known = {f.name for f in fields(cls)}
        for key in raw:
            if key not in known:
                log.warning("unknown config key %r in %s — ignored", key, config_path)
        cfg.normalize()
        return cfg

    def normalize(self) -> list[str]:
        """Clamp/validate ranges in place. Returns list of warnings."""
        warnings: list[str] = []
        if not 0.0 <= self.temperature <= 2.0:
            warnings.append(f"temperature {self.temperature} out of range [0, 2]; clamped")
            self.temperature = min(max(self.temperature, 0.0), 2.0)
        if not 0.0 <= self.proactive_image_prob <= 1.0:
            warnings.append(
                f"proactive_image_prob {self.proactive_image_prob} out of range [0, 1]; clamped"
            )
            self.proactive_image_prob = min(max(self.proactive_image_prob, 0.0), 1.0)
        if self.proactive_min_hours < 0 or self.proactive_max_hours < 0:
            warnings.append("proactive hours must be >= 0; clamped")
            self.proactive_min_hours = max(0.0, self.proactive_min_hours)
            self.proactive_max_hours = max(0.0, self.proactive_max_hours)
        if self.proactive_min_hours > self.proactive_max_hours:
            self.proactive_min_hours, self.proactive_max_hours = (
                self.proactive_max_hours,
                self.proactive_min_hours,
            )
        for attr in ("proactive_quiet_start", "proactive_quiet_end"):
            value = getattr(self, attr)
            if value is not None and not 0 <= value <= 23:
                warnings.append(f"{attr} {value} out of range [0, 23]; disabled")
                setattr(self, attr, None)
        if not self.data_dir:
            warnings.append("data_dir empty; using 'data'")
            self.data_dir = "data"
        if not self.persona_path:
            warnings.append("persona_path empty; using 'persona.yaml'")
            self.persona_path = "persona.yaml"
        for warning in warnings:
            log.warning("%s", warning)
        return warnings

    def validate(self) -> list[str]:
        """Non-mutating checks for startup. Returns list of error strings."""
        errors: list[str] = []
        if not self.telegram_token:
            errors.append("TELEGRAM_BOT_TOKEN not set")
        if not self.allowed_chat_ids and not self.allows_everyone:
            errors.append("ALLOWED_CHAT_ID not set — bot will refuse everyone")
        for label, path in (("image_workflow", self.image_workflow), ("video_workflow", self.video_workflow)):
            if path and not Path(path).exists():
                errors.append(f"{label} {path!r} not found")
        if not Path(self.persona_path).exists():
            errors.append(f"persona {self.persona_path!r} not found (fallback persona will be used)")
        return errors