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


def test_prepare_overrides_linked_prompt_text() -> None:
    """A prompt node fed by an enhancement chain gets the literal prompt."""
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ["2", 0], "clip": ["0", 1]}},
        "2": {"class_type": "StringConcatenate", "inputs": {"string_a": ["3", 0]}},
        "3": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "stale default"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
        "5": {
            "class_type": "KSampler",
            "inputs": {"seed": 1, "positive": ["1", 0], "negative": ["1", 0]},
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }
    prepared = ComfyClient.prepare(graph, positive="fresh persona prompt", seed=9)
    assert prepared["1"]["inputs"]["text"] == "fresh persona prompt"
    # Enhancement chain + preview pruned; output chain kept.
    assert set(prepared) == {"1", "5", "8", "9"}
    assert prepared["5"]["inputs"]["seed"] == 9


def test_prepare_without_save_node_keeps_everything() -> None:
    prepared = ComfyClient.prepare(GRAPH, positive="x", seed=1)
    assert set(prepared) == set(GRAPH)