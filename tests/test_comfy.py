from __future__ import annotations

import json
from pathlib import Path

from agf.comfy_client import ComfyClient

GRAPH = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative words", "clip": ["4", 1]}},
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
}


def test_load_graph(tmp_path: Path) -> None:
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(GRAPH))
    loaded = ComfyClient.load_graph(str(path))
    assert loaded["3"]["class_type"] == "KSampler"


def test_prepare_replaces_positive_only() -> None:
    prepared = ComfyClient.prepare(GRAPH, positive="a brand new prompt", seed=424242)
    assert prepared["6"]["inputs"]["text"] == "a brand new prompt"
    assert prepared["7"]["inputs"]["text"] == "negative words"
    assert prepared["3"]["inputs"]["seed"] == 424242
    assert prepared["4"]["inputs"]["ckpt_name"] == "model.safetensors"
    assert GRAPH["6"]["inputs"]["text"] == "positive"


def test_prepare_checkpoint_override() -> None:
    prepared = ComfyClient.prepare(
        GRAPH, positive="x", seed=1, checkpoint="better-checkpoint.safetensors"
    )
    assert prepared["4"]["inputs"]["ckpt_name"] == "better-checkpoint.safetensors"


def test_prepare_original_graph_untouched() -> None:
    before = json.dumps(GRAPH)
    ComfyClient.prepare(GRAPH, positive="changed", seed=999)
    assert json.dumps(GRAPH) == before


def test_prepare_negative_prompt() -> None:
    prepared = ComfyClient.prepare(GRAPH, positive="pos", negative="neg words")
    assert prepared["6"]["inputs"]["text"] == "pos"
    assert prepared["7"]["inputs"]["text"] == "neg words"