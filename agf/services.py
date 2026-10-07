from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

from . import persona as persona_mod
from .comfy_client import ComfyClient
from .config import Config
from .memory import MemoryStore
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)


@dataclass
class Reply:
    text: str
    media: Optional[bytes] = None
    media_type: str = "photo"
    caption: Optional[str] = None


_PHOTO_WORDS_RE = re.compile(
    r"\b(?:photo|pic|pics|picture|selfie|snap|image|img)\b", re.I
)
_VIDEO_WORDS_RE = re.compile(r"\b(?:video|clip|vid|vids|reel)\b", re.I)
# Explicit request verbs that strongly signal media intent (covers "send me X", "I want a clip", "pic please")
_MEDIA_REQUEST_RE = re.compile(
    r"\b(?:send|give|gimme|show|share|post|drop|make|generate|create|want|need|wish|please)\b",
    re.I,
)


def detect_intent(text: str) -> str:
    lowered = text.lower().strip()
    # explicit slash commands always win
    if lowered.startswith("/photo"):
        return "photo"
    if lowered.startswith("/video"):
        return "video"
    wants_media = bool(_MEDIA_REQUEST_RE.search(text))
    # \"pic please\", \"selfie?\" style short asks count even without a verb
    short_ask = len(text.split()) <= 6 and "?" in text
    if _VIDEO_WORDS_RE.search(text) and (wants_media or short_ask):
        return "video"
    if _PHOTO_WORDS_RE.search(text) and (wants_media or short_ask):
        return "photo"
    return "chat"


def _profile_key(fact: str) -> str:
    head, _, _ = fact.partition(":")
    head = head.strip()
    if head and len(head) <= 60:
        return head
    return fact.strip()[:60] or "detail"


def parse_facts(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    facts = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact", "")).strip()
        if not fact:
            continue
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        kind = item.get("kind", "fact")
        if kind not in ("fact", "pref", "private"):
            kind = "fact"
        facts.append({"fact": fact, "importance": min(max(importance, 0.0), 1.0), "kind": kind})
    return facts


class GirlfriendService:
    def __init__(
        self,
        cfg: Config,
        ollama: OllamaClient,
        comfy: ComfyClient,
        memory: MemoryStore,
    ):
        self.cfg = cfg
        self.ollama = ollama
        # ensure the client falls back to the configured model when callers omit it
        if not getattr(ollama, "default_model", None):
            ollama.default_model = cfg.llm_model
        self.comfy = comfy
        self.memory = memory
        self.persona = persona_mod.load_persona(cfg.persona_path)
        self._bg_tasks: set[asyncio.Task] = set()

    def _track(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._bg_tasks.discard(t)
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                return
            if exc:
                log.warning("background task failed: %s", exc)

        task.add_done_callback(_done)
        return task

    async def shutdown(self) -> None:
        if not self._bg_tasks:
            return
        await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

    def _options(self) -> dict:
        return {"temperature": self.cfg.temperature, "num_ctx": 8192}

    async def _embed(self, text: str):
        return await self.ollama.embed(text, model=self.cfg.embed_model)

    async def build_messages(self, user_text: str) -> list[dict]:
        profile = await self.memory.recall_profile()
        system = persona_mod.build_system_prompt(self.cfg, self.persona, profile)
        memories = await self.memory.recall(user_text, k=5)
        if memories:
            block = "\n\nRelevant things you remember:\n" + "\n".join(
                f"- {m['text']}" for m in memories
            )
            system = system + block
        messages: list[dict] = [{"role": "system", "content": system}]
        for message in await self.memory.recent_chat(24):
            messages.append({"role": message["role"], "content": message["text"]})
        messages.append({"role": "user", "content": user_text})
        return messages

    async def reply(self, user_text: str) -> Reply:
        messages = await self.build_messages(user_text)
        text = await self.ollama.complete(
            messages, model=self.cfg.llm_model, options=self._options()
        )
        await self.memory.add_chat("user", user_text)
        await self.memory.add_chat("assistant", text)
        self._track(self._extract_memories(f"{user_text}\n\n{text}"))
        action = detect_intent(user_text)
        if action == "photo" and self.cfg.image_workflow and os.path.exists(self.cfg.image_workflow):
            try:
                image = await self.generate_image(subject=user_text)
                return Reply(text=text, media=image, media_type="photo", caption=self.photo_caption())
            except Exception as exc:
                log.warning("photo generation failed: %s", exc)
                return Reply(text=f"{text}\n\n(she tried to send you a pic but the image server "
                                  "didn't cooperate \uD83D\uDE22)")
        if action == "video" and self.cfg.video_workflow and os.path.exists(self.cfg.video_workflow):
            try:
                video = await self.generate_video(subject=user_text)
                return Reply(text=text, media=video, media_type="video", caption=self.photo_caption())
            except Exception as exc:
                log.warning("video generation failed: %s", exc)
        return Reply(text=text)

    async def _extract_memories(self, conversation: str) -> None:
        try:
            raw = await self.ollama.complete(
                [
                    {"role": "system", "content": persona_mod.extraction_prompt()},
                    {"role": "user", "content": conversation},
                ],
                model=self.cfg.llm_model,
                options={"temperature": 0.2},
            )
        except Exception as exc:
            log.warning("memory extraction failed: %s", exc)
            return
        for fact in parse_facts(raw):
            if fact["importance"] < 0.45:
                continue
            if fact["kind"] in ("pref", "private"):
                await self.memory.set_profile(_profile_key(fact["fact"]), fact["fact"])
            try:
                await self.memory.add_memory(fact["fact"], kind=fact["kind"], importance=fact["importance"])
            except ValueError:
                continue
        try:
            await self.memory.prune(max_memories=1000)
        except Exception as exc:
            log.debug("memory prune failed: %s", exc)

    async def generate_image(self, subject: Optional[str] = None) -> bytes:
        if not self.cfg.image_workflow:
            raise RuntimeError("no image workflow configured")
        graph = ComfyClient.load_graph(self.cfg.image_workflow)
        prompt = persona_mod.image_prompt(self.cfg, self.persona, subject)
        negative = persona_mod.negative_prompt(self.cfg, self.persona)
        seed = random.randint(0, 2**31 - 1)
        return await self.comfy.generate(graph, positive=prompt, negative=negative, seed=seed)

    async def generate_video(self, subject: Optional[str] = None) -> bytes:
        if not self.cfg.video_workflow:
            raise RuntimeError("no video workflow configured")
        graph = ComfyClient.load_graph(self.cfg.video_workflow)
        prompt = persona_mod.image_prompt(self.cfg, self.persona, subject)
        negative = persona_mod.negative_prompt(self.cfg, self.persona)
        seed = random.randint(0, 2**31 - 1)
        return await self.comfy.generate(graph, positive=prompt, negative=negative, seed=seed)

    def photo_caption(self) -> str:
        captions = self.persona.get("photo_captions") or ["hehe, something for you \u2661"]
        return random.choice(captions)

    async def proactive_message(self) -> str:
        profile = await self.memory.recall_profile()
        system = persona_mod.build_system_prompt(self.cfg, self.persona, profile)
        prompt = (
            "It's been a while since you last texted the user. Reach out right now with a short, "
            "natural, affectionate in-character message. Vary it between sweet, teasing and "
            "flirty. One or two sentences max. Something only a real girlfriend would send."
        )
        try:
            text = await self.ollama.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                model=self.cfg.llm_model,
                options=self._options(),
            )
            if text and text.strip():
                await self.memory.add_chat("assistant", text.strip())
            return text
        except Exception as exc:
            log.warning("proactive text failed: %s", exc)
            fallback = random.choice(
                [
                    "hey babe, was just thinking about you ♡",
                    "you busy? I'm bored and it's your fault for being so cute.",
                    "mm, I miss you. come say hi?",
                ]
            )
            try:
                await self.memory.add_chat("assistant", fallback)
            except Exception:
                pass
            return fallback

    async def status(self) -> dict:
        models = []
        try:
            models = await self.ollama.tags()
        except Exception as exc:
            models = [f"error: {exc}"]
        comfy_ok = await self.comfy.health()
        return {
            "ollama_models": models,
            "comfyui_reachable": comfy_ok,
            "image_workflow": self.cfg.image_workflow,
            "video_workflow": self.cfg.video_workflow or None,
            "nsfw": self.cfg.nsfw,
        }