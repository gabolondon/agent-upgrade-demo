"""Terminal chat client (rich). Works against v1 (one JSON body) and v2 (SSE)."""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx
from rich.console import Console

from client.sse import iter_events

console = Console()


async def reveal(text: str, delay: float = 0.03) -> None:
    for word in text.split(" "):
        console.print(word, end=" ")
        await asyncio.sleep(delay)
    console.print()


def print_sources(sources: list) -> None:
    for s in sources:
        if isinstance(s, dict):
            console.print(f"  [link={s['href']}]{s['title']}[/link]", style="cyan")
        else:
            console.print(f"  {s}", style="dim")


async def ask(http: httpx.AsyncClient, path: str, body: dict, pace: float) -> dict | None:
    steps: list[str] = []
    streamed = False
    async with http.stream("POST", path, json=body) as resp:
        resp.raise_for_status()
        if "text/event-stream" not in resp.headers.get("content-type", ""):
            await resp.aread()
            payload = resp.json()
            await reveal(payload["answer"], 0)
            print_sources(payload.get("cited", []))
            return payload
        cited: list = []
        async for event, data in iter_events(resp):
            if event == "working":
                steps = (steps + [data["label"]])[-3:]
                console.print("  " + " > ".join(steps), style="dim")
            elif event == "token":
                streamed = True
                console.print(data["text"], end="")
            elif event == "cited":
                cited = data["items"]
            elif event == "failed":
                console.print(f"[red]Could not finish: {data['message']}[/red]")
                return None
            elif event == "done":
                if streamed:
                    console.print()
                else:
                    if data.get("cached") and pace:
                        time.sleep(pace)  # a cached report should not land jarringly fast
                    await reveal(data["answer"])
                print_sources(cited)
                return data
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8102")
    ap.add_argument("--account", default="c1")
    ap.add_argument(
        "--pace", type=float, default=0.0, help="seconds to pause before a cached report"
    )
    args = ap.parse_args()
    history: list[dict] = []
    async with httpx.AsyncClient(base_url=args.url, timeout=120) as http:
        console.print(f"[bold]Opening report for {args.account}[/bold]")
        await ask(http, "/briefing", {"account_id": args.account}, args.pace)
        while True:
            try:
                q = console.input("[bold green]you> [/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q in {"exit", "quit"}:
                break
            res = await ask(
                http, "/ask", {"account_id": args.account, "message": q, "history": history}, 0
            )
            if res:
                history += [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": res["answer"]},
                ]


if __name__ == "__main__":
    asyncio.run(main())
