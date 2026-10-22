"""Invented personal-finance data API with a verbose /v1 and a compact /v2."""

from __future__ import annotations

import asyncio
import os
import random
import re
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from portfolio_api import data as D

DEFAULT_WINDOW_DAYS = 90
MAX_WINDOW_DAYS = 366


def create_app(
    *,
    overhead_ms: float | None = None,
    ms_per_1k_rows: float | None = None,
    fail_rate: float | None = None,
    world: D.World | None = None,
) -> FastAPI:
    overhead = float(os.environ.get("API_OVERHEAD_MS", 80)) if overhead_ms is None else overhead_ms
    per_1k = (
        float(os.environ.get("API_LATENCY_MS_PER_1K_ROWS", 15))
        if ms_per_1k_rows is None
        else ms_per_1k_rows
    )
    rate = float(os.environ.get("FAIL_RATE", 0.3)) if fail_rate is None else fail_rate
    w = world or D.build_world()
    fail_rng = random.Random(7)
    notes = {slug: (title, body) for slug, title, body in D.NOTES}
    app = FastAPI(title="Invented portfolio API")

    async def delay(rows: int) -> None:
        await asyncio.sleep((overhead + per_1k * rows / 1000) / 1000)

    def volatile(rows: list[dict]) -> list[dict]:
        """v1 responses carry per-request noise: a trace id and a retrieval date."""
        trace = f"{random.getrandbits(48):012x}"
        return [dict(r, retrieved_on=w.ref.isoformat(), trace_id=trace) for r in rows]

    def account_or_404(aid: str) -> None:
        if aid not in D.ACCOUNTS:
            raise HTTPException(404, "unknown account")

    def window(frm: date | None, to: date | None) -> tuple[date, date]:
        to = to or w.ref
        frm = frm or (to - timedelta(days=DEFAULT_WINDOW_DAYS))
        if frm > to:
            raise HTTPException(422, "from must be <= to")
        if (to - frm).days > MAX_WINDOW_DAYS:
            raise HTTPException(422, f"window exceeds {MAX_WINDOW_DAYS} days")
        return frm, to

    def holdings_rows(aid: str) -> list[list]:
        return [[t, D.INSTRUMENTS[t][0], q] for t, q in sorted(D.HOLDINGS[aid].items())]

    def maybe_fail(aid: str) -> None:
        if aid == D.FLAKY_ACCOUNT and fail_rng.random() < rate:
            raise HTTPException(503, "upstream ledger unavailable")

    def search(q: str) -> list[list]:
        terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2]
        out = []
        for slug, (title, body) in notes.items():
            hay = f"{title} {body}".lower()
            score = sum(t in hay for t in terms)
            if score:
                out.append((score, slug, title, body[:120]))
        out.sort(key=lambda r: (-r[0], r[1]))
        return [[s, t, sn] for _, s, t, sn in out[:3]]

    # ----- meta and notes -------------------------------------------------
    @app.get("/health")
    async def health():
        return {"ok": True, "today": w.ref.isoformat()}

    @app.get("/notes/{slug}", response_class=PlainTextResponse)
    async def note(slug: str):
        if slug not in notes:
            raise HTTPException(404)
        title, body = notes[slug]
        return f"# {title}\n\n{body}\n"

    @app.get("/v1/accounts/{aid}")
    async def account_info(aid: str):
        account_or_404(aid)
        await delay(1)
        return {"id": aid, "name": D.ACCOUNTS[aid]}

    # ----- v1: verbose, whole history, unstable order ---------------------
    @app.get("/v1/accounts/{aid}/holdings")
    async def v1_holdings(aid: str):
        account_or_404(aid)
        rows = [dict(ticker=t, name=n, quantity=q) for t, n, q in holdings_rows(aid)]
        random.shuffle(rows)
        rows = volatile(rows)
        await delay(len(rows))
        return rows

    @app.get("/v1/accounts/{aid}/transactions")
    async def v1_tx(aid: str):
        account_or_404(aid)
        maybe_fail(aid)
        rows = [
            dict(date=a, ticker=b, side=c, quantity=d, price=e)
            for a, b, c, d, e in w.transactions[aid]
        ]
        random.shuffle(rows)
        rows = volatile(rows)
        await delay(len(rows))
        return rows

    @app.get("/v1/prices/{ticker}")
    async def v1_prices(ticker: str):
        if ticker not in D.INSTRUMENTS:
            raise HTTPException(404, "unknown instrument")
        rows = [dict(date=a, ticker=ticker, close=b, volume=c) for a, b, c in w.prices[ticker]]
        random.shuffle(rows)
        rows = volatile(rows)
        await delay(len(rows))
        return rows

    @app.get("/v1/notes")
    async def v1_notes(q: str = ""):
        res = search(q)
        await delay(len(res))
        return [dict(slug=s, title=t, snippet=sn) for s, t, sn in res]

    # ----- v2: compact, stable, bounded -----------------------------------
    @app.get("/v2/accounts/{aid}/holdings")
    async def v2_holdings(aid: str):
        account_or_404(aid)
        rows = holdings_rows(aid)
        await delay(len(rows))
        return {"header": ["ticker", "name", "quantity"], "table": rows}

    @app.get("/v2/accounts/{aid}/transactions")
    async def v2_tx(
        aid: str,
        window_start: date | None = None,
        window_end: date | None = None,
    ):
        account_or_404(aid)
        maybe_fail(aid)
        frm, to_ = window(window_start, window_end)
        rows = [
            list(r)
            for r in sorted(w.transactions[aid])
            if frm.isoformat() <= r[0] <= to_.isoformat()
        ]
        await delay(len(rows))
        return {"header": ["date", "ticker", "side", "quantity", "price"], "table": rows}

    @app.get("/v2/prices/{ticker}")
    async def v2_prices(
        ticker: str,
        window_start: date | None = None,
        window_end: date | None = None,
    ):
        if ticker not in D.INSTRUMENTS:
            raise HTTPException(404, "unknown instrument")
        frm, to_ = window(window_start, window_end)
        rows = [list(r) for r in w.prices[ticker] if frm.isoformat() <= r[0] <= to_.isoformat()]
        await delay(len(rows))
        return {"header": ["date", "close", "volume"], "table": rows}

    @app.get("/v2/notes")
    async def v2_notes(q: str = ""):
        res = search(q)
        await delay(len(res))
        return {"header": ["slug", "title", "snippet"], "table": res}

    return app


def app_from_env() -> FastAPI:
    return create_app()
