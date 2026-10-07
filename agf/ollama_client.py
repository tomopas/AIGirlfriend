from __future__ import annotations

import json
from typing import AsyncIterator, List, Optional

import httpx
import numpy as np


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 120.0, default_model: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    def _resolve_model(self, model: Optional[str]) -> str:
        resolved = model or self.default_model
        if not resolved:
            raise OllamaError("no Ollama model specified — pass model= or set default_model")
        return resolved

    async def close(self) -> None:
        await self._http.aclose()

    async def tags(self) -> List[str]:
        try:
            response = await self._http.get("/api/tags")
        except httpx.HTTPError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    async def stream_chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> AsyncIterator[str]:
        resolved = self._resolve_model(model)
        payload: dict = {
            "model": resolved,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise OllamaError(f"ollama chat failed ({response.status_code}): {body[:300]}")
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("done"):
                        break
                    chunk = (obj.get("message") or {}).get("content", "")
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            raise OllamaError(f"ollama chat request failed: {exc}") from exc

    async def complete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> str:
        chunks = []
        async for chunk in self.stream_chat(messages, model=model, options=options):
            chunks.append(chunk)
        return "".join(chunks)

    async def embed(self, text: str, model: Optional[str] = None) -> np.ndarray:
        resolved = self._resolve_model(model)
        try:
            response = await self._http.post("/api/embed", json={"model": resolved, "input": text})
        except httpx.HTTPError as exc:
            raise OllamaError(f"ollama embed request failed: {exc}") from exc
        if response.status_code == 404:
            legacy = await self._http.post(
                "/api/embeddings", json={"model": resolved, "prompt": text}
            )
            legacy.raise_for_status()
            return np.asarray(legacy.json()["embedding"], dtype="float32")
        response.raise_for_status()
        data = response.json().get("embeddings")
        if not data:
            raise OllamaError("ollama returned no embeddings")
        vector = data[0] if isinstance(data[0], list) else data
        return np.asarray(vector, dtype="float32").reshape(-1)