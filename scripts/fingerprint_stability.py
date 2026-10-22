"""Fetch the same /v1 payloads 25 times and count distinct hashes: raw versus stable_form.

    uv run python scripts/fingerprint_stability.py [--url http://localhost:8210] [--n 25]

With no --url it serves the API in-process, so it needs nothing running.
"""

from __future__ import annotations

import argparse
import json

import httpx

from agent_v2.fingerprint import raw_digest, stable_form
from portfolio_api.app import create_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--account", default="c1")
    args = ap.parse_args()
    client = httpx.Client(base_url=args.url, timeout=60) if args.url else None
    raw, stable = set(), set()
    if client is None:
        import asyncio

        async def fetch_all() -> list[str]:
            app = create_app(overhead_ms=0, ms_per_1k_rows=0)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://x"
            ) as c:
                return [(await c.get("/v1/prices/KELP")).text for _ in range(args.n)]

        texts = asyncio.run(fetch_all())
    else:
        texts = [client.get("/v1/prices/KELP").text for _ in range(args.n)]
    for t in texts:
        raw.add(raw_digest(t))
        stable.add(raw_digest(stable_form(json.loads(t))))
    print(f"fetched /v1/prices/KELP {args.n} times")
    print(f"raw payload hashes:    {len(raw)} distinct")
    print(f"stable_form hashes:    {len(stable)} distinct")


if __name__ == "__main__":
    main()
