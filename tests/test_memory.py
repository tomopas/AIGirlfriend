from __future__ import annotations

import numpy as np
import pytest

from agf.memory import MemoryStore


def fake_embed(text: str) -> np.ndarray:
    tokens = text.lower().split()
    if any("python" in token for token in tokens):
        return np.array([1.0, 0.0], dtype="float32")
    if any("pizza" in token for token in tokens):
        return np.array([0.0, 1.0], dtype="float32")
    return np.array([0.5, 0.5], dtype="float32")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db", embed_fn=fake_embed)


def test_add_and_recall(store: MemoryStore) -> None:
    await_store = store
    import asyncio

    async def scenario() -> None:
        await await_store.add_memory("she loves coding in python")
        await await_store.add_memory("best pizza place in town")
        hits = await await_store.recall("python side projects", k=3)
        assert hits
        assert hits[0]["text"] == "she loves coding in python"
        assert hits[0]["score"] > 0.0

    asyncio.run(scenario())


def test_recall_without_embeddings(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    import asyncio

    async def scenario() -> None:
        await store.add_memory("user works at a bakery")
        await store.add_memory("user hates cold coffee")
        hits = await store.recall("bakery work", k=3)
        assert any("bakery" in h["text"] for h in hits)

    asyncio.run(scenario())


def test_profile_roundtrip(store: MemoryStore) -> None:
    import asyncio

    async def scenario() -> None:
        await store.set_profile("kink", "enjoys roleplay")
        profile = await store.recall_profile()
        assert profile["kink"] == "enjoys roleplay"

    asyncio.run(scenario())


def test_chat_history(store: MemoryStore) -> None:
    import asyncio

    async def scenario() -> None:
        await store.add_chat("user", "hi babe")
        await store.add_chat("assistant", "hey you!")
        history = await store.recent_chat(5)
        assert [h["role"] for h in history] == ["user", "assistant"]
        assert history[0]["text"] == "hi babe"

    asyncio.run(scenario())


def test_forget_all(store: MemoryStore) -> None:
    import asyncio

    async def scenario() -> None:
        await store.add_memory("something to forget")
        deleted = await store.forget_all()
        assert deleted >= 1
        assert await store.list_memories() == []

    asyncio.run(scenario())


def test_sync_embed_actually_used(tmp_path) -> None:
    import asyncio
    import sqlite3

    store = MemoryStore(tmp_path / "memory.db", embed_fn=fake_embed)
    try:
        async def scenario() -> None:
            await store.add_memory("she loves coding in python")
            rows = await store._submit(store._all_memories)
            assert rows and rows[0]["vector"] is not None

        asyncio.run(scenario())
    finally:
        store.close()


def test_dedup_returns_same_id(tmp_path) -> None:
    import asyncio

    store = MemoryStore(tmp_path / "memory.db")
    try:
        async def scenario() -> None:
            first = await store.add_memory("User Loves Pizza")
            second = await store.add_memory("user loves pizza")
            assert first == second
            assert len(await store.list_memories()) == 1

        asyncio.run(scenario())
    finally:
        store.close()


def test_delete_cleans_embeddings(tmp_path) -> None:
    import asyncio
    import sqlite3

    store = MemoryStore(tmp_path / "memory.db", embed_fn=fake_embed)
    try:
        async def scenario() -> None:
            mid = await store.add_memory("she loves coding in python")
            await store.delete_memory(mid)
            assert await store.list_memories() == []
            con = sqlite3.connect(str(store.path))
            try:
                n = con.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
            finally:
                con.close()
            assert n == 0

        asyncio.run(scenario())
    finally:
        store.close()


def test_dim_mismatch_falls_back_to_keywords(tmp_path) -> None:
    import asyncio

    def embed_v2(text: str):
        return np.array([1.0, 0.0, 0.0], dtype="float32")

    store = MemoryStore(tmp_path / "memory.db", embed_fn=fake_embed)
    try:
        async def scenario() -> None:
            await store.add_memory("bakery fresh bread")
            # swap to incompatible dim
            store._embed = embed_v2  # type: ignore
            hits = await store.recall("bakery fresh", k=3)
            assert any("bakery" in h["text"] for h in hits)

        asyncio.run(scenario())
    finally:
        store.close()