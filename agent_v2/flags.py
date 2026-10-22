"""Agent v2 feature flags. Every improvement can be switched off to ablate it."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

from common.config import env_bool

FLAG_NAMES = ["CACHE", "DIRECT_TOOLS", "COMPACT", "MODEL_PER_ROLE", "STREAMING"]


@dataclass(frozen=True)
class Flags:
    cache: bool = True
    direct_tools: bool = True
    compact: bool = True
    model_per_role: bool = True
    streaming: bool = True

    @classmethod
    def from_env(cls) -> Flags:
        return cls(**{f.name: env_bool(f.name.upper(), f.default) for f in fields(cls)})

    @classmethod
    def from_cli(cls, spec: str) -> Flags:
        """'CACHE=0,COMPACT=0' style overrides on top of the environment."""
        base = asdict(cls.from_env())
        for part in filter(None, (p.strip() for p in spec.split(","))):
            k, _, v = part.partition("=")
            base[k.strip().lower()] = v.strip().lower() in ("1", "true", "yes", "on", "")
        return cls(**base)

    def label(self) -> str:
        off = [n for n in FLAG_NAMES if not getattr(self, n.lower())]
        return "all-on" if not off else "off:" + ",".join(off)


def apply_to_env(flags: Flags) -> None:
    for n in FLAG_NAMES:
        os.environ[n] = "1" if getattr(flags, n.lower()) else "0"
