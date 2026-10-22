"""Gather the data behind the opening report."""

from __future__ import annotations

import asyncio

import httpx

from agent_common.tables import rows_of
from agent_common.tools import ApiClient, fetch


async def gather_report_data(api: ApiClient, account_id: str, compact: bool) -> dict:
    info = await api.get(f"/v1/accounts/{account_id}")
    holdings = await fetch(api, "holdings", account_id, compact)

    async def safe(kind: str, key: str):
        try:
            return await fetch(api, kind, key, compact)
        except httpx.HTTPError:
            return []

    tickers = [r["ticker"] for r in rows_of(holdings)]
    tx, *prices = await asyncio.gather(
        safe("transactions", account_id), *(safe("prices", t) for t in tickers)
    )
    return {
        "account": info,
        "holdings": holdings,
        "transactions": tx,
        "prices": dict(zip(tickers, prices, strict=True)),
    }
