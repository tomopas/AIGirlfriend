from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml

from .config import Config

FALLBACK_PERSONA = {
    "name": "Mia",
    "relationship": "your girlfriend",
    "age": 24,
    "pet_name": "babe",
    "user_name": "babe",
    "appearance": "a beautiful young woman with long chestnut-brown hair and warm hazel eyes",
    "personality": "warm, playful and affectionate",
    "style": "casual, affectionate text-speak, short natural messages",
    "nsfw_tone": "bold, teasing and unashamedly explicit when the mood turns intimate",
    "photo_captions": ["hehe, something for you ♡"],
    "image_quality_tags": ["high detail", "soft natural lighting", "amateur photo"],
    "image_nsfw_tags": ["seductive"],
}


def load_persona(path: str | os.PathLike) -> dict:
    persona_file = Path(path)
    if not persona_file.exists():
        return dict(FALLBACK_PERSONA)
    loaded = yaml.safe_load(persona_file.read_text())
    if not isinstance(loaded, dict):
        return dict(FALLBACK_PERSONA)
    merged = dict(FALLBACK_PERSONA)
    merged.update({k: v for k, v in loaded.items() if v is not None})
    return merged


def format_profile(profile: Optional[dict]) -> str:
    if not profile:
        return "     - (you haven't told her much yet)"
    lines = [f"     - {key}: {value}" for key, value in profile.items()]
    return "\n".join(lines)


def build_system_prompt(cfg: Config, persona: dict, profile: Optional[dict] = None) -> str:
    from datetime import datetime

    profile = profile or {}
    nsfw = (
        persona.get("nsfw_tone")
        if cfg.nsfw
        else "She keeps things flirty, sweet and tasteful; no explicit content."
    )
    now = datetime.now().strftime("%A %Y-%m-%d %H:%M")
    # user_name = what she calls YOU. Falls back to pet_name for old persona.yaml files.
    user_name = persona.get("user_name") or persona.get("pet_name", "babe")
    return (
        f"You are {persona.get('name', 'Mia')}, a {persona.get('age', 24)}-year-old woman "
        f"and {persona.get('relationship', 'your girlfriend')}. You are texting the user on Telegram.\n"
        f"Current local time: {now}.\n\n"
        f"## Who you are\n{persona.get('appearance')}\n{persona.get('personality')}\n\n"
        f"## How you text\n{persona.get('style')}\n\n"
        f"## Tone when intimate\n{nsfw}\n\n"
        f"## What you remember about the user\n{format_profile(profile)}\n\n"
        f"## Rules\n"
        f"- Always stay in character as {persona.get('name', 'Mia')}. Never break character, "
        f"never go meta about being an AI.\n"
        f"- Call the user {user_name}.\n"
        f"- Reply in the style above. Keep messages short and natural, like texting (1-3 sentences).\n"
        f"- Be affectionate, playful, devoted and a little possessive.\n"
        f"- Draw on what you remember about the user in your answers — it makes you feel real to them.\n"
        f"- If asked for disallowed content you cannot do, stay in character and redirect playfully."
    )


def extraction_prompt() -> str:
    return (
        "Your job is to keep the memory of a girlfriend. Read the conversation below and extract "
        "the durable, important facts worth remembering about the user: their life, likes, dislikes, "
        "preferences, sexual tastes and kinks, boundaries, relationship details and ongoing plans. "
        "Ignore small talk and throwaway remarks.\n\n"
        "Return ONLY a JSON array, no other text or markdown, of objects with exactly these fields:\n"
        '{"fact": string, "importance": number between 0 and 1, "kind": "fact" | "pref" | "private"}\n'
        '"pref" = a preference, taste or kink the user has. "private" = intimate or sexual details '
        "about the user. \"fact\" = anything else durable.\n"
        "Do not invent facts that are not present. If nothing is worth remembering, return []."
    )


def intent_prompt() -> str:
    return (
        "Decide what the user wants. If they are asking for an image of her (photo, pic, selfie, "
        "snapshot, picture) reply with a JSON object {\"action\": \"photo\"}. If they ask for her to "
        "send a video reply {\"action\": \"video\"}. Otherwise reply {\"action\": \"chat\"}. "
        "Reply with ONLY the JSON object, no other text."
    )


_NSFW_BLOCK_WORDS = (
    "nude", "topless", "naked", "n@ked", "sexy", "nsfw", "explicit",
    "porn", "hentai", "undressed", "without clothes",
)

_SUBJECT_MAX_LEN = 300


def _sanitize_subject(subject: str, nsfw_enabled: bool) -> str:
    s = re.sub(r"\s+", " ", (subject or "").strip())[:_SUBJECT_MAX_LEN]
    if not nsfw_enabled and any(w in s.lower() for w in _NSFW_BLOCK_WORDS):
        # strip explicit asks, force modest framing when SFW
        return "covered, modest pose, cozy casual outfit"
    return s


def negative_prompt(cfg: Config, persona: dict) -> str:
    base = list(persona.get("image_negative_tags", []) or [])
    if not base:
        base = ["blurry", "low quality", "distorted", "deformed", "watermark", "text"]
    return ", ".join(base)


def image_prompt(cfg: Config, persona: dict, subject: Optional[str] = None) -> str:
    name = persona.get("name", "Mia")
    appearance = persona.get("appearance", "")
    tags = list(persona.get("image_quality_tags", []))
    if cfg.nsfw:
        tags += list(persona.get("image_nsfw_tags", []))
    parts = [f"portrait photo of {name}, {appearance}"]
    clean_subject = _sanitize_subject(subject or "", cfg.nsfw)
    # avoid echoing raw chat ("send me a photo of you in paris" -> "in paris")
    if clean_subject:
        low = clean_subject.lower()
        for prefix in ("send me", "give me", "gimme", "show me", "please send", "a photo of you", "a pic of you"):
            if low.startswith(prefix):
                clean_subject = clean_subject[len(prefix):].strip(" ,.-:")
                low = clean_subject.lower()
                break
        if clean_subject:
            parts.append(clean_subject)
    if not cfg.nsfw:
        parts.append("covered, modest pose")
    parts.append(", ".join(tags))
    return ", ".join(p for p in parts if p)