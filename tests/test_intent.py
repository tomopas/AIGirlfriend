from __future__ import annotations

from agf.services import detect_intent, parse_facts


def test_chat_intent() -> None:
    assert detect_intent("hi, how are you today") == "chat"
    assert detect_intent("miss you") == "chat"


def test_photo_intent() -> None:
    assert detect_intent("send me a photo of you") == "photo"
    assert detect_intent("can you send a selfie?") == "photo"
    assert detect_intent("pic please") == "photo"


def test_video_intent() -> None:
    assert detect_intent("send me a video") == "video"
    assert detect_intent("I want a clip of you") == "video"


def test_no_false_positives() -> None:
    # mentioning media without asking must stay chat
    assert detect_intent("I watched a video today, it was funny") == "chat"
    assert detect_intent("my camera roll has a nice picture from paris") == "chat"
    assert detect_intent("the video game was fun") == "chat"


def test_parse_facts_json() -> None:
    raw = '[{"fact": "loves dark chocolate", "importance": 0.8, "kind": "pref"}, ' \
          '{"fact": "birthday in june", "importance": 0.6, "kind": "fact"}]'
    facts = parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["kind"] == "pref"
    assert facts[0]["importance"] == 0.8


def test_parse_facts_with_preamble() -> None:
    raw = 'here is what i found:\n```json\n[{"fact": "hates mornings", "importance": 0.5, "kind": "fact"}]\n```'
    facts = parse_facts(raw)
    assert facts and facts[0]["fact"] == "hates mornings"


def test_parse_facts_empty() -> None:
    assert parse_facts("nothing worth remembering") == []
    assert parse_facts("") == []