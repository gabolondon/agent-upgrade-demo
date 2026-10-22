"""FastAPI server: POST /ask and POST /briefing, serving agent v1 or v2 by config.

v1 answers with one JSON body. v2 streams Server-Sent Events (working, token, cited, done | failed)
when STREAMING is on, and falls back to one JSON body when it is off.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent_v1.graph import AgentV1
from agent_v2.agent import AgentV2
from agent_v2.flags import Flags
from common.config import Settings, env_bool, load_dotenv
from server.dev_console import debug_router


class ChatIn(BaseModel):
    account_id: str
    message: str
    history: list[dict] = Field(default_factory=list)


class OpenIn(BaseModel):
    account_id: str


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def sse_stream(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for e in events:
        yield sse(e["event"], e["data"])


def create_app(
    version: str | None = None,
    flags: Flags | None = None,
    settings: Settings | None = None,
    agent=None,
    debug: bool | None = None,
) -> FastAPI:
    version = version or os.environ.get("AGENT_VERSION", "v2")
    settings = settings or Settings.from_env()
    flags = flags or Flags.from_env()
    holder: dict = {"agent": agent}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if holder["agent"] is None:
            holder["agent"] = AgentV1(settings) if version == "v1" else AgentV2(flags, settings)
        yield

    app = FastAPI(title=f"agent-{version}", lifespan=lifespan)

    if env_bool("ADVISOR_DEBUG", False) if debug is None else debug:
        app.include_router(debug_router(settings))

    @app.get("/health")
    async def health():
        return {"version": version, "flags": flags.label() if version == "v2" else None}

    @app.post("/ask")
    async def ask(body: ChatIn):
        a = holder["agent"]
        if version == "v2" and flags.streaming:
            return StreamingResponse(
                sse_stream(a.chat_events(body.account_id, body.message, body.history)),
                media_type="text/event-stream",
            )
        if version == "v2":
            res = await a.chat(body.account_id, body.message, body.history)
            return JSONResponse(res, status_code=502 if "message" in res else 200)
        return await a.chat(body.account_id, body.message, body.history)

    @app.post("/briefing")
    async def briefing(body: OpenIn):
        a = holder["agent"]
        if version == "v2" and flags.streaming:
            return StreamingResponse(
                sse_stream(a.open_events(body.account_id)), media_type="text/event-stream"
            )
        if version == "v2":
            res = await a.open_report(body.account_id)
            return JSONResponse(res, status_code=502 if "message" in res else 200)
        res = await a.open_report(body.account_id)
        return {"answer": res["report"], "cited": [], "cached": False}

    return app


def app_from_env() -> FastAPI:
    load_dotenv()
    return create_app()
