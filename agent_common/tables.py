"""Helpers for the two payload shapes (verbose list-of-dicts and compact header/table)."""

from __future__ import annotations

import json
from typing import Any


def rows_of(payload: Any) -> list[dict]:
    """Normalize any supported payload into a list of row dicts."""
    if isinstance(payload, dict) and "worked" in payload:
        payload = payload.get("data", [])
    if isinstance(payload, dict) and "header" in payload and "table" in payload:
        cols = payload["header"]
        return [dict(zip(cols, r, strict=False)) for r in payload["table"]]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def typed_result(payload: Any) -> dict:
    """Wrap tool data as {"worked": true, "data": ...}; empty data is still a call that worked."""
    return {"worked": True, "data": payload}


def failed_result(reason: str) -> dict:
    return {"worked": False, "reason": reason}


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))
