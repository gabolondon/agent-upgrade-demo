"""Data tools over the portfolio API.

Two flavours, matching the two agents:
- untyped (v1): returns the raw payload as text; any failure or empty result becomes the same
  vague "No data available." string.
- typed (v2): returns {"worked": true, "data": ...} or {"worked": false, "reason": "..."}.
Either flavour can read /v1 (verbose) or /v2 (compact, bounded) via `compact`.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
from langchain_core.tools import BaseTool, tool

from agent_common.tables import dumps, failed_result, rows_of, typed_result
from common.config import demo_today

VAGUE_EMPTY = "No data available."


class ApiClient:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None, timeout: float = 30):
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def get(self, path: str, params: dict | None = None):
        r = await self.client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        await self.client.aclose()


def _window(start: str, end: str, days: int = 90) -> dict:
    end_d = end or demo_today().isoformat()
    start_d = start or (demo_today() - timedelta(days=days)).isoformat()
    return {"window_start": start_d, "window_end": end_d}


async def fetch(api: ApiClient, kind: str, key: str, compact: bool, start="", end=""):
    """Fetch one dataset. kind: holdings | transactions | prices | notes."""
    v = "v2" if compact else "v1"
    if kind == "holdings":
        return await api.get(f"/{v}/accounts/{key}/holdings")
    if kind == "transactions":
        return await api.get(
            f"/{v}/accounts/{key}/transactions", _window(start, end) if compact else None
        )
    if kind == "prices":
        return await api.get(f"/{v}/prices/{key}", _window(start, end) if compact else None)
    if kind == "notes":
        return await api.get(f"/{v}/notes", {"q": key})
    raise ValueError(kind)


def make_tools(api: ApiClient, *, compact: bool, typed: bool) -> dict[str, BaseTool]:
    async def run(kind: str, key: str, start: str = "", end: str = "") -> str:
        try:
            payload = await fetch(api, kind, key, compact, start, end)
        except (httpx.HTTPError, ValueError) as exc:
            return dumps(failed_result(f"{type(exc).__name__}: {exc}")) if typed else VAGUE_EMPTY
        if typed:
            return dumps(typed_result(payload))
        return dumps(payload) if rows_of(payload) else VAGUE_EMPTY

    @tool
    async def get_holdings(account_id: str) -> str:
        """Current holdings (ticker, name, quantity) for a client id like 'c1'."""
        return await run("holdings", account_id)

    @tool
    async def get_transactions(account_id: str, start: str = "", end: str = "") -> str:
        """Transactions for a client. Optional ISO dates start/end (YYYY-MM-DD)."""
        return await run("transactions", account_id, start, end)

    @tool
    async def get_prices(ticker: str, start: str = "", end: str = "") -> str:
        """Daily closing prices for a ticker such as 'KELP'. Optional ISO dates start/end."""
        return await run("prices", ticker, start, end)

    @tool
    async def search_notes(query: str) -> str:
        """Search the research notes. Returns slug, title and snippet of the best matches."""
        return await run("notes", query)

    return {t.name: t for t in (get_holdings, get_transactions, get_prices, search_notes)}
