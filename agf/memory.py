from __future__ import annotations

import asyncio
import inspect
import os
import queue
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

import numpy as np

EmbedFn = Callable[[str], Union[Awaitable[np.ndarray], np.ndarray]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    ts INTEGER NOT NULL
);
"""


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MemoryStore:
    """SQLite-backed memory with optional embedding-based recall.

    All database work runs on a single dedicated worker thread so the sqlite
    connection (which is strictly thread-bound) stays consistent even when the
    store is used from asyncio code.
    """

    def __init__(self, db_path: str | os.PathLike, embed_fn: Optional[EmbedFn] = None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._embed = embed_fn
        self._closing = threading.Event()
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="memory-db", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        while not self._closing.is_set() or not self._queue.empty():
            try:
                func, future, loop = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = func(conn)
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, result)
        conn.close()

    def _submit(self, func: Callable) -> asyncio.Future:
        if not self._thread.is_alive():
            raise RuntimeError("memory store is closed")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put((func, future, loop))
        return future

    def close(self) -> None:
        self._closing.set()
        self._thread.join(timeout=5.0)

    async def _embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._embed is None:
            return None
        try:
            result = self._embed(text)
            if inspect.isawaitable(result):
                result = await result
            arr = np.asarray(result, dtype="float32").reshape(-1)
            if arr.size == 0:
                return None
            return arr
        except Exception:
            return None

    async def add_memory(self, text: str, kind: str = "fact", importance: float = 0.5) -> int:
        clean = (text or "").strip()
        if not clean:
            raise ValueError("memory text must not be empty")
        # dedup: skip near-exact duplicates (case-insensitive)
        existing = await self._submit(lambda conn: self._find_duplicate(conn, clean))
        if existing is not None:
            return int(existing)
        vector: Optional[bytes] = None
        vec = await self._embed_text(clean)
        if vec is not None:
            vector = vec.tobytes()
        return await self._submit(lambda conn: self._add(conn, clean, kind, float(importance), vector))

    @staticmethod
    def _find_duplicate(conn: sqlite3.Connection, text: str) -> Optional[int]:
        cursor = conn.execute(
            "SELECT id FROM memories WHERE lower(text) = lower(?) LIMIT 1",
            (text,),
        )
        row = cursor.fetchone()
        return int(row["id"]) if row else None

    @staticmethod
    def _add(conn: sqlite3.Connection, text: str, kind: str, importance: float, vector: Optional[bytes]) -> int:
        cursor = conn.execute(
            "INSERT INTO memories (text, kind, importance, created_at) VALUES (?,?,?,?)",
            (text, kind, importance, _now()),
        )
        memory_id = cursor.lastrowid
        if vector is not None:
            conn.execute(
                "INSERT INTO memory_embeddings (memory_id, vector) VALUES (?,?)",
                (memory_id, vector),
            )
        conn.commit()
        return int(memory_id)

    @staticmethod
    def _all_memories(conn: sqlite3.Connection, limit: int = 500) -> list[dict]:
        cursor = conn.execute(
            "SELECT m.id, m.text, m.kind, m.importance, me.vector "
            "FROM memories m LEFT JOIN memory_embeddings me ON me.memory_id = m.id "
            "ORDER BY m.id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    async def recall(self, query: str, k: int = 5, min_score: float = 0.15) -> list[dict]:
        rows = await self._submit(lambda conn: self._all_memories(conn))
        if not rows:
            return []
        query_vector = await self._embed_text(query)
        scored: list[tuple[float, dict]] = []
        if query_vector is not None:
            qdim = query_vector.shape[0]
            for row in rows:
                if row.get("vector") is None:
                    continue
                try:
                    vector = np.frombuffer(row["vector"], dtype="float32")
                except (ValueError, TypeError):
                    continue
                if vector.shape[0] != qdim:
                    continue  # embedding model changed; skip stale vectors
                sim = _cosine(query_vector, vector)
                if sim < min_score:
                    continue
                scored.append((sim * (0.5 + row["importance"]), row))
            if not scored:
                # fall through to keyword search if all vectors stale/missing
                query_vector = None
        if query_vector is None and not scored:
            query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
            for row in rows:
                words = set(re.findall(r"[a-z0-9]+", row["text"].lower()))
                overlap = len(query_words & words)
                if overlap:
                    scored.append((float(overlap) * (0.5 + row["importance"]), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:k]
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "kind": row["kind"],
                "importance": row["importance"],
                "score": round(score, 3),
            }
            for score, row in top
        ]

    async def list_memories(self, limit: int = 100) -> list[dict]:
        def _run(conn: sqlite3.Connection) -> list[dict]:
            cursor = conn.execute(
                "SELECT id, text, kind, importance, created_at FROM memories ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

        return await self._submit(_run)

    async def delete_memory(self, memory_id: int) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()

        await self._submit(_run)

    async def prune(self, max_memories: int = 1000) -> int:
        """Keep only the newest/most important memories. Returns rows deleted."""

        def _run(conn: sqlite3.Connection) -> int:
            cursor = conn.execute("SELECT COUNT(*) AS n FROM memories")
            count = int(cursor.fetchone()["n"])
            if count <= max_memories:
                return 0
            to_delete = count - max_memories
            cursor = conn.execute(
                "SELECT id FROM memories ORDER BY importance ASC, id ASC LIMIT ?",
                (to_delete,),
            )
            ids = [r["id"] for r in cursor.fetchall()]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})", ids)
            cur2 = conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            conn.commit()
            return int(cur2.rowcount)

        return await self._submit(_run)

    async def forget_all(self, kind: str | None = None) -> int:
        def _run(conn: sqlite3.Connection) -> int:
            if kind:
                cursor = conn.execute("SELECT id FROM memories WHERE kind = ?", (kind,))
                ids = [r["id"] for r in cursor.fetchall()]
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    conn.execute(
                        f"DELETE FROM memory_embeddings WHERE memory_id IN ({placeholders})", ids
                    )
                cursor = conn.execute("DELETE FROM memories WHERE kind = ?", (kind,))
            else:
                conn.execute("DELETE FROM memory_embeddings")
                cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return int(cursor.rowcount)

        return await self._submit(_run)

    async def set_profile(self, key: str, value: str) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO profile (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, _now()),
            )
            conn.commit()

        await self._submit(_run)

    async def recall_profile(self) -> dict[str, str]:
        def _run(conn: sqlite3.Connection) -> dict[str, str]:
            cursor = conn.execute("SELECT key, value FROM profile ORDER BY updated_at")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

        return await self._submit(_run)

    async def add_chat(self, role: str, text: str, max_history: int = 500) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO chat_history (role, text, ts) VALUES (?,?,?)",
                (role, text, _now()),
            )
            # bound table size: drop oldest beyond cap
            conn.execute(
                "DELETE FROM chat_history WHERE id NOT IN "
                "(SELECT id FROM chat_history ORDER BY id DESC LIMIT ?)",
                (max_history,),
            )
            conn.commit()

        await self._submit(_run)

    async def last_activity_ts(self) -> Optional[int]:
        def _run(conn: sqlite3.Connection) -> Optional[int]:
            cursor = conn.execute("SELECT MAX(ts) AS ts FROM chat_history")
            row = cursor.fetchone()
            return int(row["ts"]) if row and row["ts"] is not None else None

        return await self._submit(_run)

    async def recent_chat(self, n: int = 40) -> list[dict]:
        def _run(conn: sqlite3.Connection) -> list[dict]:
            cursor = conn.execute(
                "SELECT role, text FROM ("
                "SELECT id, role, text FROM chat_history ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (n,),
            )
            return [dict(row) for row in cursor.fetchall()]

        return await self._submit(_run)

    async def clear_chat(self) -> None:
        def _run(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM chat_history")
            conn.commit()

        await self._submit(_run)