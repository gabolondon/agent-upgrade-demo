"""web-gateway: what the CLI/frontend talks to. Requires X-Demo-Token, forwards to the advisor."""

from __future__ import annotations

import hmac
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from common.config import load_dotenv


def create_app(
    token: str | None = None,
    advisor_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    token = token if token is not None else os.environ.get("DEMO_TOKEN", "")
    advisor_url = advisor_url or os.environ.get("ADVISOR_URL", "http://advisor:8102")
    http = client or httpx.AsyncClient(base_url=advisor_url, timeout=120)
    app = FastAPI(title="web-gateway")

    def authorized(request: Request) -> bool:
        got = request.headers.get("X-Demo-Token", "")
        return bool(token) and hmac.compare_digest(got, token)

    @app.get("/health")
    async def health():
        return {"ok": True}

    async def forward(request: Request, path: str):
        if not authorized(request):
            return JSONResponse({"detail": "missing or wrong X-Demo-Token"}, status_code=401)
        req = http.build_request(
            "POST", path, content=await request.body(), headers={"content-type": "application/json"}
        )
        resp = await http.send(req, stream=True)
        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("text/event-stream"):
            return StreamingResponse(
                resp.aiter_raw(),
                status_code=resp.status_code,
                media_type=ctype,
                background=BackgroundTask(resp.aclose),
            )
        body = await resp.aread()
        await resp.aclose()
        return Response(body, status_code=resp.status_code, media_type=ctype or None)

    @app.post("/ask")
    async def ask(request: Request):
        return await forward(request, "/ask")

    @app.post("/briefing")
    async def briefing(request: Request):
        return await forward(request, "/briefing")

    return app


def app_from_env() -> FastAPI:
    load_dotenv()
    return create_app()
