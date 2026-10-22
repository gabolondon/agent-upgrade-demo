"""Agent v2, "the upgraded one": every improvement sits behind a flag (see flags.py).

A triage step on the light TRIAGE_MODEL decides portfolio question vs. small talk, then ONE answer
agent on ANSWER_MODEL calls the data tools directly.

Events are plain dicts {"event": working|token|cited|done|failed, "data": {...}} so the server can
turn them into SSE or collapse them into one JSON response. A turn is complete only with `done`.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.prebuilt import create_react_agent

from agent_common import prompts
from agent_common.llm import make_chat_model
from agent_common.report import gather_report_data
from agent_common.tables import dumps, rows_of
from agent_common.tools import ApiClient, make_tools
from agent_v1.graph import AgentV1
from agent_v2.cache import ReportCache
from agent_v2.fingerprint import fingerprint_key
from agent_v2.flags import Flags
from common.config import Settings, demo_today
from portfolio_api.data import NICKNAMES

Event = dict

# Bump when the report output should be regenerated.
REPORT_GENERATION = "gen-3"

# Report settings that take part in the cache key.
REPORT_SCOPE = {"currency": "USD", "locale": "en"}

STATUS_LABELS = {
    "get_holdings": "Looking up your holdings",
    "get_transactions": "Reading your transactions",
    "get_prices": "Fetching prices",
    "search_notes": "Searching the research notes",
}


def ev(event: str, **data) -> Event:
    return {"event": event, "data": data}


def finish(answer: str, items: list[dict], **extra) -> list[Event]:
    """The closing pair: `cited` (sent once) then `done`."""
    return [ev("cited", items=items), ev("done", answer=answer, **extra)]


def failed(exc: Exception | str) -> Event:
    message = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else exc
    return ev("failed", message=message)


def to_messages(history: list[dict] | None) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for h in history or []:
        out.append(HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]))
    return out


def note_sources(tool_messages: list[ToolMessage], api_url: str) -> list[dict]:
    seen, out = set(), []
    for m in tool_messages:
        if m.name != "search_notes":
            continue
        try:
            rows = rows_of(json.loads(m.content))
        except json.JSONDecodeError:
            continue
        for r in rows:
            if r["slug"] not in seen:
                seen.add(r["slug"])
                out.append({"title": r["title"], "href": f"{api_url}/notes/{r['slug']}"})
    return out


async def collapse(events: AsyncIterator[Event]) -> dict:
    """Success -> {"answer", "cited", ...}; failure -> {"message"}. Callers tell them apart by key."""
    cited: list[dict] = []
    async for e in events:
        if e["event"] == "cited":
            cited = e["data"]["items"]
        elif e["event"] == "done":
            return {**e["data"], "cited": cited}
        elif e["event"] == "failed":
            return e["data"]
    return {"message": "stream ended without a done event"}


class AgentV2:
    def __init__(
        self,
        flags: Flags | None = None,
        settings: Settings | None = None,
        api: ApiClient | None = None,
        cache: ReportCache | None = None,
    ):
        self.flags = flags or Flags.from_env()
        self.settings = settings or Settings.from_env()
        self.api = api or ApiClient(self.settings.api_url)
        self.cache = cache or (
            ReportCache(os.environ.get("REPORT_CACHE_PATH", ".cache/reports.sqlite"))
            if self.flags.cache
            else None
        )
        self.answer_model = make_chat_model(self.settings.answer_model, self.settings)
        self.triage_model = (
            make_chat_model(self.settings.triage_model, self.settings)
            if self.flags.model_per_role
            else self.answer_model
        )
        self.tools = make_tools(self.api, compact=self.flags.compact, typed=True)
        self.legacy = None
        if not self.flags.direct_tools:  # ablation: fall back to the v1 planner/specialists/writer
            self.legacy = AgentV1(
                self.settings, self.api, compact=self.flags.compact, answer_model=self.answer_model
            )

    # ------------------------------------------------------------------ chat
    async def chat_events(
        self, account_id: str, message: str, history: list[dict] | None = None
    ) -> AsyncIterator[Event]:
        try:
            async for e in self._chat(account_id, message, history):
                yield e
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed event
            yield failed(exc)

    async def _chat(self, account_id, message, history) -> AsyncIterator[Event]:
        today = demo_today().isoformat()
        yield ev("working", label="Understanding your question")
        verdict = await self.triage_model.ainvoke(
            [SystemMessage(prompts.TRIAGE_V2), HumanMessage(message)]
        )
        needs_data = str(verdict.content).strip().lower() == "portfolio"

        if self.legacy is not None:
            yield ev("working", label="Asking the specialists")
            st = await self.legacy.run(account_id, message, history)
            sources = [
                {"title": n["title"], "href": f"{self.settings.api_url}/notes/{n['slug']}"}
                for n in st.get("notes", [])
            ]
            for e in finish(st["answer"], sources):
                yield e
            return

        tools = list(self.tools.values()) if needs_data else []
        agent = create_react_agent(
            self.answer_model,
            tools,
            prompt=prompts.ANSWER_V2.format(
                account_id=account_id, nickname=NICKNAMES.get(account_id, "friend"), today=today
            ),
        )
        msgs = [*to_messages(history), HumanMessage(message)]
        if not self.flags.streaming:
            out = await agent.ainvoke({"messages": msgs})
            tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
            for e in finish(
                str(out["messages"][-1].content), note_sources(tool_msgs, self.settings.api_url)
            ):
                yield e
            return

        answer, tool_msgs = [], []
        async for _mode, payload in agent.astream({"messages": msgs}, stream_mode=["messages"]):
            chunk, _meta = payload
            if isinstance(chunk, ToolMessage):
                tool_msgs.append(chunk)
            elif isinstance(chunk, AIMessageChunk):
                if chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        yield ev(
                            "working", label=STATUS_LABELS.get(tc.get("name") or "", "Working")
                        )
                elif chunk.content:
                    answer.append(str(chunk.content))
                    yield ev("token", text=str(chunk.content))
        for e in finish("".join(answer).strip(), note_sources(tool_msgs, self.settings.api_url)):
            yield e

    async def chat(self, account_id: str, message: str, history: list[dict] | None = None) -> dict:
        """Collapse the event stream into one payload (used when STREAMING is off)."""
        return await collapse(self.chat_events(account_id, message, history))

    # ------------------------------------------------------------------ opening report
    async def open_events(self, account_id: str) -> AsyncIterator[Event]:
        """Opening report. Only a report that reaches `done` is ever cached."""
        try:
            yield ev("working", label="Checking your data")
            data = await gather_report_data(self.api, account_id, self.flags.compact)

            async def generate() -> str:
                msg = await self.answer_model.ainvoke(
                    [SystemMessage(prompts.REPORT_V2), HumanMessage("DATA:\n" + dumps(data))]
                )
                return str(msg.content)

            if self.flags.cache and self.cache is not None:
                key = fingerprint_key(data, REPORT_GENERATION, REPORT_SCOPE)
                hit = self.cache.get(key) is not None
                yield ev(
                    "working", label="Loading your saved report" if hit else "Writing your report"
                )
                text, cached = await self.cache.get_or_compute(key, generate)
            else:
                yield ev("working", label="Writing your report")
                text, cached = await generate(), False
            for e in finish(text, [], cached=cached):
                yield e
        except Exception as exc:  # noqa: BLE001
            yield failed(exc)

    async def open_report(self, account_id: str) -> dict:
        return await collapse(self.open_events(account_id))
