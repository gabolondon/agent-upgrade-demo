"""Content fingerprint for the opening-report cache.

The raw API payload is not a stable key: v1 shuffles rows per request, floats can differ in
their last digits, and every response carries volatile fields (a trace id, a retrieval date).
stable_form() normalises all of that (sorted keys, sorted rows, rounded floats, volatile keys
dropped) so the same data always hashes the same.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

FLOAT_DIGITS = 6
VOLATILE_KEYS = {"retrieved_on", "trace_id"}


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=32).hexdigest()


def _normalise(obj: Any) -> Any:
    if isinstance(obj, bool) or obj is None or isinstance(obj, str | int):
        return obj
    if isinstance(obj, float):
        r = round(obj, FLOAT_DIGITS)
        return 0.0 if r == 0 else (int(r) if r == int(r) else r)
    if isinstance(obj, dict):
        return {
            str(k): _normalise(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            if k not in VOLATILE_KEYS
        }
    if isinstance(obj, list | tuple):
        items = [_normalise(v) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")))
    raise TypeError(f"cannot normalise {type(obj).__name__}")


def stable_form(obj: Any) -> str:
    """Order-, float- and noise-insensitive JSON text for `obj`."""
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"))


def raw_digest(text: str) -> str:
    """Hash of the payload exactly as received (the unstable baseline)."""
    return _digest(text)


def fingerprint_key(data: Any, generator: str, scope: dict[str, Any]) -> str:
    """A fingerprint of the stable form of the data, stamped with an opaque stamp of what produced the report.

    `generator` is an opaque stamp of what produced the report; `scope` is a small dict of
    report settings.
    """
    return _digest(stable_form({"data": data, "generator": generator, "scope": scope}))
