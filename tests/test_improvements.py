from __future__ import annotations

import pytest

from agf.config import Config
from agf.ollama_client import OllamaClient, OllamaError
from agf.proactive import in_quiet_hours
from agf import persona as persona_mod


def test_quiet_hours_overnight() -> None:
    assert in_quiet_hours(2, 0, 7) is True
    assert in_quiet_hours(8, 0, 7) is False
    assert in_quiet_hours(23, 22, 6) is True
    assert in_quiet_hours(12, 22, 6) is False
    assert in_quiet_hours(12, None, None) is False


def test_ollama_requires_model() -> None:
    import asyncio

    async def scenario() -> None:
        client = OllamaClient("http://localhost:11434")
        try:
            with pytest.raises(OllamaError):
                await client.embed("hi")
        finally:
            await client.close()

    asyncio.run(scenario())


def test_persona_negative_and_sfw_guard() -> None:
    cfg = Config(nsfw=False)
    persona = dict(persona_mod.FALLBACK_PERSONA)
    assert persona_mod.negative_prompt(cfg, persona)
    p = persona_mod.image_prompt(cfg, persona, subject="send me a nude photo please")
    assert "nude" not in p.lower() or "modest" in p.lower()
    assert "modest" in p.lower()

    cfg_nsfw = Config(nsfw=True)
    p2 = persona_mod.image_prompt(cfg_nsfw, persona, subject="x" * 500)
    assert len(p2) < 2000  # subject truncated, not unbounded
