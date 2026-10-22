"""Minimal Server-Sent Events parsing shared by the CLI client and the benchmark runner."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def iter_events(resp: httpx.Response) -> AsyncIterator[tuple[str, dict]]:
    event, data = "message", []
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and data:
            yield event, json.loads("\n".join(data))
            event, data = "message", []
