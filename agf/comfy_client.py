from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx


class ComfyError(RuntimeError):
    pass


@dataclass
class MediaRef:
    filename: str
    subfolder: str = ""
    type: str = "output"


def outputs_empty(entry: dict) -> bool:
    outputs = entry.get("outputs") or {}
    if not outputs:
        return True
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for key in ("images", "gifs", "videos"):
            if output.get(key):
                return False
    return True


class ComfyClient:
    _OUTPUT_KEYS = ("images", "gifs", "videos")

    def __init__(self, base_url: str, timeout: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def health(self) -> bool:
        try:
            response = await self._http.get("/system_stats", timeout=5.0)
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    @staticmethod
    def load_graph(path: str | os.PathLike) -> dict:
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ComfyError(f"cannot read workflow {path}: {exc}") from exc
        # Detect UI-format exports (they have "nodes"/"links", not API-format ids).
        if isinstance(raw, dict) and ("nodes" in raw or "links" in raw):
            raise ComfyError(
                f"workflow {path} looks like ComfyUI UI-format (has 'nodes'/'links'). "
                "In ComfyUI click 'Save (API Format)' and save that JSON instead."
            )
        if not isinstance(raw, dict) or not raw:
            raise ComfyError(f"workflow {path} is empty or invalid (expected API-format dict)")
        if not any(isinstance(n, dict) and "class_type" in n for n in raw.values()):
            raise ComfyError(
                f"workflow {path} has no 'class_type' nodes — "
                "it is not an API-format export. Use 'Save (API Format)'."
            )
        return raw

    @staticmethod
    def _positive_node_ids(graph: dict) -> list[str]:
        encoders = [
            node_id
            for node_id, node in graph.items()
            if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode"
        ]
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or "positive" not in inputs:
                continue
            ref = inputs["positive"]
            if isinstance(ref, list) and len(ref) == 2 and ref[0] in encoders:
                return [ref[0]]
        negative_refs = {
            ref[0]
            for node_id, node in graph.items()
            if isinstance(node, dict)
            and isinstance(node.get("inputs"), dict)
            and isinstance(node["inputs"].get("negative"), list)
            and isinstance(node["inputs"]["negative"][0], str)
        }
        fallback = [node_id for node_id in encoders if node_id not in negative_refs]
        return fallback or encoders

    @staticmethod
    def _negative_node_ids(graph: dict) -> list[str]:
        encoders = [
            node_id
            for node_id, node in graph.items()
            if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode"
        ]
        positive_ids = set(ComfyClient._positive_node_ids(graph))
        negatives: list[str] = []
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict) or "negative" not in inputs:
                continue
            ref = inputs["negative"]
            if isinstance(ref, list) and len(ref) == 2 and ref[0] in encoders:
                negatives.append(ref[0])
        if negatives:
            return negatives
        # fall back: any encoder that is not the positive one
        return [n for n in encoders if n not in positive_ids]

    @staticmethod
    def prepare(
        graph: dict,
        positive: str | None = None,
        negative: str | None = None,
        seed: int | None = None,
        checkpoint: str | None = None,
    ) -> dict:
        prepared = copy.deepcopy(graph)
        target_ids = set(ComfyClient._positive_node_ids(prepared)) if positive is not None else set()
        neg_ids = set(ComfyClient._negative_node_ids(prepared)) if negative is not None else set()
        seed_counter = 0
        for node_id, node in prepared.items():
            if not isinstance(node, dict) or "inputs" not in node:
                continue
            inputs = node["inputs"]
            class_type = node.get("class_type", "")
            if positive is not None and node_id in target_ids:
                inputs["text"] = positive
            if negative is not None and node_id in neg_ids:
                inputs["text"] = negative
            if seed is not None and "seed" in inputs:
                # Distinct seed per sampler node: same run, uncorrelated noise.
                inputs["seed"] = int(seed) + seed_counter
                seed_counter += 1
            if checkpoint is not None and class_type == "CheckpointLoaderSimple":
                inputs["ckpt_name"] = checkpoint
        return prepared

    @staticmethod
    def validate_workflow(path: str | os.PathLike) -> list[str]:
        """Return a list of problems (empty = OK). Never raises."""
        try:
            graph = ComfyClient.load_graph(path)
        except Exception as exc:
            return [str(exc)]
        problems: list[str] = []
        if not ComfyClient._positive_node_ids(graph):
            problems.append("no CLIPTextEncode positive node found")
        if not ComfyClient._negative_node_ids(graph):
            problems.append("no negative prompt node found (continuing with positive only)")
        has_sampler = any(
            isinstance(n, dict) and "seed" in (n.get("inputs") or {})
            for n in graph.values()
        )
        if not has_sampler:
            problems.append("no sampler/seed node found — seeds will not be randomized")
        return problems

    async def queue(self, graph: dict, client_id: str | None = None) -> str:
        payload = {"prompt": graph, "client_id": client_id or str(uuid.uuid4())}
        response = await self._http.post("/prompt", json=payload)
        if response.status_code != 200:
            raise ComfyError(f"comfy prompt failed ({response.status_code}): {response.text[:400]}")
        return response.json()["prompt_id"]

    async def wait(self, prompt_id: str, timeout: float = 600.0) -> list[MediaRef]:
        deadline = time.monotonic() + timeout
        errors = 0
        polls = 0
        while True:
            try:
                response = await self._http.get(f"/history/{prompt_id}")
            except httpx.HTTPError as exc:
                errors += 1
                if errors > 5:
                    raise ComfyError(f"comfy history polling failed: {exc}") from exc
                await asyncio.sleep(2.0)
                continue
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                entry = data.get(prompt_id)
                if entry is not None:
                    status = entry.get("status") or {}
                    status_str = str(status.get("status_str", ""))
                    messages = status.get("messages") or []
                    if status_str == "error" or any(
                        "error" in str(m).lower() for m in messages
                    ):
                        raise ComfyError(f"comfy workflow failed: {messages or status}")
                    if status.get("completed") is False and outputs_empty(entry):
                        # explicit failure without outputs
                        raise ComfyError(f"comfy workflow failed: {status}")
                    outputs = entry.get("outputs") or {}
                    if status.get("completed") or outputs:
                        refs = []
                        for output in outputs.values():
                            for key in self._OUTPUT_KEYS:
                                for item in output.get(key, []):
                                    if not isinstance(item, dict) or "filename" not in item:
                                        continue
                                    refs.append(
                                        MediaRef(
                                            filename=item["filename"],
                                            subfolder=item.get("subfolder", ""),
                                            type=item.get("type", "output"),
                                        )
                                    )
                        if refs:
                            return refs
            if time.monotonic() > deadline:
                raise TimeoutError(f"comfyui did not finish prompt {prompt_id} in {timeout}s")
            # Back off while queued: 1s -> 5s max so we don't hammer ComfyUI.
            polls += 1
            await asyncio.sleep(min(1.0 + polls * 0.5, 5.0))

    async def fetch(self, ref: MediaRef) -> bytes:
        params = {"filename": ref.filename, "type": ref.type}
        if ref.subfolder:
            params["subfolder"] = ref.subfolder
        try:
            response = await self._http.get("/view", params=params)
        except httpx.HTTPError as exc:
            raise ComfyError(f"comfy fetch failed for {ref.filename}: {exc}") from exc
        if response.status_code != 200:
            raise ComfyError(f"comfy fetch failed ({response.status_code}): {response.text[:300]}")
        if not response.content:
            raise ComfyError(f"comfy returned empty media for {ref.filename}")
        return response.content

    async def generate(
        self,
        graph: dict,
        positive: str,
        negative: str | None = None,
        seed: int | None = None,
        checkpoint: str | None = None,
        timeout: float = 600.0,
    ) -> bytes:
        medias = await self.generate_all(
            graph, positive=positive, negative=negative, seed=seed,
            checkpoint=checkpoint, timeout=timeout, limit=4,
        )
        # Workflows often emit a preview + final image. Prefer the largest
        # payload (usually the final), falling back to the last output.
        best = max(range(len(medias)), key=lambda i: (len(medias[i]), i))
        return medias[best]

    async def generate_all(
        self,
        graph: dict,
        positive: str,
        negative: str | None = None,
        seed: int | None = None,
        checkpoint: str | None = None,
        timeout: float = 600.0,
        limit: int = 4,
    ) -> list[bytes]:
        prepared = self.prepare(graph, positive=positive, negative=negative, seed=seed, checkpoint=checkpoint)
        try:
            prompt_id = await self.queue(prepared)
        except httpx.HTTPError as exc:
            raise ComfyError(f"comfy queue failed: {exc}") from exc
        refs = await self.wait(prompt_id, timeout=timeout)
        out: list[bytes] = []
        for ref in refs[: max(1, limit)]:
            out.append(await self.fetch(ref))
        if not out:
            raise ComfyError("comfy returned no media")
        return out