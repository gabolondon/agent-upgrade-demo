"""Environment-driven settings shared by every component."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TRUE = {"1", "true", "yes", "on"}


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader: KEY=VALUE lines, never overrides the real environment."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE


def demo_today() -> date:
    """The 'today' of the example data (fixed so the data and benchmark are reproducible)."""
    return date.fromisoformat(os.environ.get("REFERENCE_DATE", "2026-09-30"))


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    triage_model: str
    answer_model: str
    judge_model: str
    fake: bool
    api_url: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            llm_api_key=os.environ.get("LLM_API_KEY", "none"),
            triage_model=os.environ.get("TRIAGE_MODEL", "small-triage"),
            answer_model=os.environ.get("ANSWER_MODEL", "big-answer"),
            judge_model=os.environ.get("JUDGE_MODEL", "big-judge"),
            fake=env_bool("LLM_FAKE", False),
            api_url=os.environ.get("PORTFOLIO_API_URL", "http://localhost:8210"),
        )
