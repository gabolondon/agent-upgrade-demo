"""Plays the scenario conversations against a target URL and judges every answer."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import yaml

from bench.judge import judge_answer
from client.sse import iter_events

SCENARIOS = Path(__file__).parent / "scenarios.yaml"


def read_scenario_file(path: str | Path = SCENARIOS) -> dict:
    return yaml.safe_load(Path(path).read_text())


async def call(http: httpx.AsyncClient, path: str, body: dict, timeout: float) -> dict:
    """One request. Returns timing plus the final answer (v1 JSON or v2 SSE)."""
    rec = {
        "first_event_s": 0.0,
        "total_s": 0.0,
        "timeout": False,
        "error": False,
        "answer": "",
        "cited": [],
        "cached": False,
    }
    t0 = time.perf_counter()
    try:
        async with http.stream("POST", path, json=body, timeout=timeout) as resp:
            if resp.status_code >= 400 and "text/event-stream" not in resp.headers.get(
                "content-type", ""
            ):
                await resp.aread()
                rec["error"] = True
            elif "text/event-stream" in resp.headers.get("content-type", ""):
                first = True
                cited: list = []
                async for event, data in iter_events(resp):
                    if first:
                        rec["first_event_s"], first = time.perf_counter() - t0, False
                    if event == "cited":
                        cited = data["items"]
                    elif event == "done":
                        rec.update(
                            answer=data["answer"], cited=cited, cached=data.get("cached", False)
                        )
                    elif event == "failed":
                        rec["error"] = True
            else:
                payload = (await resp.aread()) and resp.json()
                rec["first_event_s"] = time.perf_counter() - t0
                rec.update(
                    answer=payload.get("answer", ""),
                    cited=payload.get("cited", []),
                    cached=payload.get("cached", False),
                )
    except httpx.TimeoutException:
        rec["timeout"] = True
    except httpx.HTTPError:
        rec["error"] = True
    rec["total_s"] = time.perf_counter() - t0
    return rec


async def run_suite(
    url: str,
    scenarios: dict,
    judge_model,
    *,
    timeout: float = 60.0,
    judge_samples: int = 3,
    run_index: int = 0,
) -> dict:
    rules = scenarios["rules"]
    out = {"questions": [], "opens": []}
    async with httpx.AsyncClient(base_url=url) as http:
        for conv in scenarios["conversations"]:
            cid, history = conv["account_id"], []
            for kind in ("open", "reopen"):  # a user closes the chat and comes back
                r = await call(http, "/briefing", {"account_id": cid}, timeout)
                out["opens"].append(
                    {
                        "conversation": conv["id"],
                        "kind": kind,
                        "run": run_index,
                        "total_s": r["total_s"],
                        "cached": r["cached"],
                        "error": r["error"],
                        "timeout": r["timeout"],
                    }
                )
            for item in conv["questions"]:
                r = await call(
                    http,
                    "/ask",
                    {"account_id": cid, "message": item["q"], "history": history},
                    timeout,
                )
                rule_set = {rid: rules[rid] for rid in item["rules"]}
                if r["timeout"] or r["error"]:
                    verdict = {
                        "rules": dict.fromkeys(rule_set, False),
                        "grade": 0,
                        "reason": "no answer (timeout or error)",
                        "raw": "",
                    }
                else:
                    verdict = await judge_answer(
                        judge_model, item["q"], r["answer"], r["cited"], rule_set, judge_samples
                    )
                    history += [
                        {"role": "user", "content": item["q"]},
                        {"role": "assistant", "content": r["answer"]},
                    ]
                out["questions"].append(
                    {
                        "conversation": conv["id"],
                        "run": run_index,
                        "question": item["q"],
                        **r,
                        "rules": verdict["rules"],
                        "grade": verdict["grade"],
                        "reason": verdict["reason"],
                        "judge_raw": verdict["raw"],
                    }
                )
    return out
