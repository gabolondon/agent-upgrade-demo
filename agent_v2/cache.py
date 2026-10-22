"""sqlite report cache with per-key single-flight."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path


class ReportCache:
    def __init__(self, path: str | Path = ".cache/reports.sqlite"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS reports (key TEXT PRIMARY KEY, value TEXT)")
        self.locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM reports WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO reports VALUES (?, ?)", (key, value))
        self.db.commit()

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[str]]
    ) -> tuple[str, bool]:
        """Return (value, cached). Concurrent callers for one key compute only once."""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            hit = self.get(key)
            if hit is not None:
                return hit, True
            value = await compute()  # an exception here propagates: nothing is stored
            if value and value.strip():  # never store an empty result
                self.put(key, value)
            return value, False
