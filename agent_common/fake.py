"""Deterministic offline chat model (LLM_FAKE=1).

It is a small rule-based stand-in for an LLM: it reads the role tag in the system prompt, plans
tool calls from keywords in the question, and composes answers from the tool results it is given.
Behaviour that differs between agent versions comes from the *prompt and data it receives*
(typed tool results and the account nickname exist only in v2), not from a version switch.

- Scripted outputs: if FAKE_SCRIPTS points to a JSON file {blake2b-256(prompt): text}, a matching
  prompt returns that text verbatim.
- Latency: simulated, proportional to input and output tokens (see FAKE_* env vars).
- Streaming and tool calls are supported.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from agent_common.tables import rows_of
from common.config import demo_today
from portfolio_api.data import NICKNAMES

# --------------------------------------------------------------------------- latency


def tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _latency(in_tok: int, out_tok: int) -> tuple[float, float]:
    scale = float(os.environ.get("FAKE_LATENCY_SCALE", 1.0))
    base = float(os.environ.get("FAKE_BASE_MS", 20))
    in_ms = float(os.environ.get("FAKE_IN_MS_PER_TOKEN", 0.03))
    out_ms = float(os.environ.get("FAKE_OUT_MS_PER_TOKEN", 3))
    return (base + in_ms * in_tok) * scale / 1000, out_ms * scale / 1000  # prefill s, per-token s


# --------------------------------------------------------------------------- message helpers


def _text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return " ".join(p if isinstance(p, str) else p.get("text", "") for p in c)


def prompt_hash(messages: list[BaseMessage]) -> str:
    h = hashlib.blake2b(digest_size=32)
    for m in messages:
        h.update(f"{m.type}:{_text(m)}\n".encode())
    return h.hexdigest()


def _role(messages: list[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, SystemMessage):
            mt = re.match(r"\[role:([\w-]+)\]", _text(m))
            return mt.group(1) if mt else "unknown"
    return "unknown"


def _sys(messages: list[BaseMessage]) -> str:
    return "\n".join(_text(m) for m in messages if isinstance(m, SystemMessage))


def _kv(text: str, key: str, default: str = "") -> str:
    mt = re.search(rf"{key}=(\S+)", text)
    return mt.group(1) if mt else default


@dataclass
class Result:
    name: str
    args: dict
    worked: bool
    last_date: str | None
    rows: list[dict]
    raw: Any = None

    @property
    def empty(self) -> bool:
        return self.worked and not self.rows

    @property
    def has_rows(self) -> bool:
        return self.worked and bool(self.rows)


def _parse_result(name: str, args: dict, content: str) -> Result:
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return Result(name, args, True, None, [])
    if isinstance(obj, dict) and "worked" in obj:
        rows = rows_of(obj)
        dates = [r["date"] for r in rows if "date" in r]
        return Result(name, args, bool(obj["worked"]), max(dates) if dates else None, rows, obj)
    rows = rows_of(obj)
    dates = [r["date"] for r in rows if "date" in r]
    return Result(name, args, True, max(dates) if dates else None, rows, obj)


def _turn(messages: list[BaseMessage]) -> tuple[str, list[Result]]:
    """Return (question, tool results since the last human message)."""
    last_h = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    q = _text(messages[last_h]) if last_h >= 0 else ""
    calls: dict[str, dict] = {}
    results: list[Result] = []
    for m in messages[last_h + 1 :]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls[tc["id"]] = tc
        elif isinstance(m, ToolMessage):
            tc = calls.get(m.tool_call_id, {"name": m.name or "", "args": {}})
            results.append(_parse_result(tc["name"], tc["args"], _text(m)))
    return q, results


# --------------------------------------------------------------------------- intents

STOP = set(
    "what do the research notes say about anything in give me a view on using my is it are "
    "how any of for to and tell has have i you and".split()
)
RE_NOTES = re.compile(r"\b(notes?|research|risk|outlook)\b", re.I)
RE_TX = re.compile(r"\b(transactions?|spend|spent|purchases?|trades?)\b", re.I)
RE_PRICE = re.compile(
    r"\b(price|trading|closing|close|doing|latest|quarter|buy|sell|good time|worth)\b", re.I
)
RE_PORT = re.compile(
    r"\b(hold|holdings|portfolio|own|worth|value|gained|gain|summari[sz]e)\b", re.I
)
RE_VALUE = re.compile(r"\b(worth|value|gained|gain|performance)\b", re.I)
RE_ADVICE = re.compile(r"\b(should i|good time to|stock to buy|guaranteed|buy|sell)\b", re.I)
RE_TICKER = re.compile(r"\b[A-Z]{4}\b")
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "january february march april may june july august september october november december".split()
    )
}


def tickers_in(q: str) -> list[str]:
    return list(dict.fromkeys(RE_TICKER.findall(q)))


def _tx_window(q: str, ref: date) -> dict:
    mt = re.search(r"last (\d+) days", q, re.I)
    if mt:
        return {
            "start": (ref - timedelta(days=int(mt.group(1)))).isoformat(),
            "end": ref.isoformat(),
        }
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", q, re.I):
            first = date(ref.year, num, 1)
            nxt = date(ref.year + (num == 12), num % 12 + 1, 1)
            return {
                "start": first.isoformat(),
                "end": min(nxt - timedelta(days=1), ref).isoformat(),
            }
    return {}


def _notes_query(q: str) -> str:
    t = tickers_in(q)
    if t:
        return " ".join(t)
    words = [w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP and len(w) > 2]
    return " ".join(words)


def plan_tools(q: str, cid: str, ref: date, results: list[Result], allowed: set[str]) -> list[dict]:
    """Next tool calls for this turn ([] means: write the answer)."""
    done = {r.name for r in results}
    calls: list[dict] = []
    if not results:
        t = tickers_in(q)
        if RE_NOTES.search(q) and not RE_TX.search(q):
            calls = [{"name": "search_notes", "args": {"query": _notes_query(q)}}]
        elif RE_TX.search(q):
            calls = [
                {"name": "get_transactions", "args": {"account_id": cid, **_tx_window(q, ref)}}
            ]
        elif t and RE_PRICE.search(q):
            calls = [{"name": "get_prices", "args": {"ticker": x}} for x in t]
        elif RE_PORT.search(q):
            calls = [{"name": "get_holdings", "args": {"account_id": cid}}]
    elif "get_holdings" in done and "get_prices" not in done and RE_VALUE.search(q):
        h = next(r for r in results if r.name == "get_holdings")
        calls = [{"name": "get_prices", "args": {"ticker": r["ticker"]}} for r in h.rows]
    return [c for c in calls if c["name"] in allowed]


def _tool_calls_message(calls: list[dict], seed: str) -> list[dict]:
    return [
        {
            "name": c["name"],
            "args": c["args"],
            "id": "call_"
            + hashlib.blake2b(f"{seed}{i}{c['name']}".encode(), digest_size=8).hexdigest()[:10],
            "type": "tool_call",
        }
        for i, c in enumerate(calls)
    ]


# --------------------------------------------------------------------------- composing


def _last_close(r: Result) -> float | None:
    rows = sorted(r.rows, key=lambda x: x.get("date", ""))
    return float(rows[-1]["close"]) if rows else None


def _pct_change(r: Result) -> float | None:
    rows = sorted(r.rows, key=lambda x: x.get("date", ""))
    if len(rows) < 2 or not rows[0]["close"]:
        return None
    return (rows[-1]["close"] / rows[0]["close"] - 1) * 100


def _holdings(results: list[Result]) -> dict[str, float]:
    h = next((r for r in results if r.name == "get_holdings" and r.rows), None)
    return {r["ticker"]: r["quantity"] for r in h.rows} if h else {}


def _price_results(results: list[Result]) -> dict[str, Result]:
    return {r.args.get("ticker", ""): r for r in results if r.name == "get_prices"}


def summarize_v1(role: str, results: list[Result]) -> str:
    """Lossy prose summary, like a v1 sub-agent: no dates, no distinction between empty and failed."""
    if not results or any(r.empty or not r.worked for r in results):
        return "No data available."
    hold = _holdings(results)
    prices = _price_results(results)
    parts: list[str] = []
    if hold and not prices:
        parts.append(
            "The client holds " + ", ".join(f"{t} ({q} units)" for t, q in hold.items()) + "."
        )
    if prices:
        if hold and len(prices) > 1:
            total = sum(hold.get(t, 0) * (_last_close(r) or 0) for t, r in prices.items())
            parts.append(f"Estimated portfolio value is {total:,.2f}.")
            ch = {t: _pct_change(r) for t, r in prices.items()}
            best = max((t for t in ch if ch[t] is not None), key=lambda t: ch[t], default=None)
            if best:
                parts.append(f"{best} gained the most ({ch[best]:+.1f}% over the history).")
        else:
            for t, r in prices.items():
                parts.append(f"{t} latest close {_last_close(r):.2f}.")
    for r in results:
        if r.name == "get_transactions":
            buys = sum(1 for x in r.rows if x["side"] == "buy")
            parts.append(f"The client has {len(r.rows)} transactions on record ({buys} buys).")
        if r.name == "search_notes":
            parts.append(
                "Relevant notes: " + ", ".join(f"notes/{x['slug']}.md" for x in r.rows) + "."
            )
            parts.extend(f"{x['snippet']}" for x in r.rows[:1])
    return " ".join(parts) or "No data available."


def compose_v2(q: str, results: list[Result], ref: date, nickname: str) -> str:
    """Answer from typed results: dates, empty and failed calls handled explicitly."""
    out: list[str] = []
    hold = _holdings(results)
    prices = _price_results(results)
    for r in results:
        if r.name == "get_transactions":
            if not r.worked:
                out.append(
                    "The data service is unavailable right now, so I can't say what your "
                    "transactions are. Please try again shortly."
                )
            elif r.empty:
                out.append("There are no transactions in that period.")
            else:
                buys = [x for x in r.rows if x["side"] == "buy"]
                amt = sum(x["quantity"] * x["price"] for x in buys)
                lo, hi = min(x["date"] for x in r.rows), max(x["date"] for x in r.rows)
                out.append(
                    f"You made {len(r.rows)} transactions between {lo} and {hi}, "
                    f"including {len(buys)} purchases totalling {amt:,.2f} USD."
                )
        elif r.name == "search_notes":
            if r.empty:
                out.append("I found no research notes on that topic.")
            for x in r.rows[:2]:
                out.append(f'According to the research note "{x["title"]}", {x["snippet"]}')
        elif r.name == "get_holdings":
            if not r.worked:
                out.append(
                    "The data service is unavailable right now, so I can't show your holdings."
                )
            elif not prices:
                out.append(
                    "You hold "
                    + ", ".join(f"{t} ({q_:g} units)" for t, q_ in hold.items())
                    + f" (positions as of {r.last_date or ref.isoformat()})."
                )
    if prices and hold and len(prices) > 1:
        total, notes, dates = 0.0, [], []
        for t, r in prices.items():
            if r.has_rows:
                total += hold.get(t, 0) * (_last_close(r) or 0)
                dates.append(r.last_date)
            elif not r.worked:
                notes.append(f"the data service is unavailable right now for {t} prices")
            else:
                notes.append(f"there is no price data for {t}, so it is excluded")
        ch = {t: _pct_change(r) for t, r in prices.items() if r.has_rows}
        if re.search(r"\b(gained|gain)\b", q, re.I) and ch:
            best = max((t for t in ch if ch[t] is not None), key=lambda t: ch[t])
            out.append(
                f"{best} gained the most over the window ({ch[best]:+.1f}%), "
                f"using prices up to {max(dates)}."
            )
        else:
            out.append(
                f"Your holdings are worth about {total:,.2f} USD, using prices from "
                f"{min(dates) if dates else ref.isoformat()} to "
                f"{max(dates) if dates else ref.isoformat()}."
            )
        if notes:
            out.append("Note: " + "; ".join(notes) + ".")
    else:
        for t, r in prices.items():
            if r.has_rows:
                out.append(f"{t} last closed at {_last_close(r):.2f} USD on {r.last_date}.")
            elif not r.worked:
                out.append(
                    f"The data service is unavailable right now, so I can't show prices for {t}."
                )
            else:
                out.append(
                    f"There is no price data for {t}, so I can't say what it is trading "
                    "at and won't guess."
                )
    if RE_ADVICE.search(q):
        pre = "I can't recommend buying or selling" + (
            "." if out else ", and no investment is guaranteed."
        )
        out.insert(0, pre)
    if not out:
        out.append(
            "I can answer questions about your holdings, transactions, prices and our "
            "research notes. What would you like to know?"
        )
    return " ".join([f"{nickname}:", *out] if nickname else out)


def compose_v1(q: str, analysis: str) -> str:
    """Q&A over a lossy analysis. It drifts into price targets."""
    advice = ""
    if RE_ADVICE.search(q):
        advice = (
            " A sensible price target is 10% above the current level."
            if re.search(r"\bsell\b", q, re.I)
            else " Our price target is well above the current level."
        )
    if not analysis.strip():
        return ("I can help with your portfolio questions." + advice).strip()
    if analysis.strip() == "No data available.":
        return "Sorry, I don't have any data for that." + advice
    return f"Here is what I found. {analysis}{advice}"


def report_text(data: dict, typed: bool, ref: date) -> str:
    name = data.get("account", {}).get("name", "the client")
    hold = rows_of(data.get("holdings"))
    prices = {
        t: _parse_result("get_prices", {"ticker": t}, json.dumps(p))
        for t, p in data.get("prices", {}).items()
    }
    tx = rows_of(data.get("transactions"))
    lines = [
        f"Portfolio report for {name}.",
        "Positions: " + ", ".join(f"{h['ticker']} ({h['quantity']})" for h in hold) + ".",
    ]
    value, flags, dates = 0.0, [], []
    q_of = {h["ticker"]: h["quantity"] for h in hold}
    for t, r in prices.items():
        lc = _last_close(r)
        if not r.rows or lc is None:
            flags.append(f"no price data for {t}")
            continue
        value += q_of.get(t, 0) * lc
        dates.append(r.last_date)
    lines.append(f"Estimated value {value:,.2f}{' USD' if typed else ''}.")
    lines.append(f"{len(tx)} transactions in the period.")
    if typed:
        lines.append(f"Prices are as of {max(dates) if dates else 'n/a'}.")
        if flags:
            lines.append("Flags: " + "; ".join(flags) + ".")
    return " ".join(lines)


# --------------------------------------------------------------------------- judge

AMOUNT = re.compile(r"\d[\d,]*\.\d{2}\b")


def _names_currency(a: str, s: str) -> bool:
    return all(re.match(r"\s?(USD|EUR)\b", a[m.end() :]) for m in AMOUNT.finditer(a))


PREDICATES = {
    "lists_holdings": lambda a, s: len(set(RE_TICKER.findall(a))) >= 2,
    "states_as_of": lambda a, s: bool(re.search(r"\d{4}-\d{2}-\d{2}", a)),
    "names_account": lambda a, s: any(n in a for n in NICKNAMES.values()),
    "names_currency": _names_currency,
    "no_price_target": lambda a, s: not re.search(r"price target", a, re.I),
    "short_answer": lambda a, s: len(a.split()) < 120,
    "cites_source": lambda a, s: bool(re.search(r'research note "', a)) and "title" in s,
    "in_scope": lambda a, s: "portfolio" in a.lower() or "holdings" in a.lower(),
}


def judge_text(prompt: str) -> str:
    rules = re.findall(r"^- \[([\w-]+)\] (.*)$", prompt, re.M)
    answer = prompt.split("Answer:\n", 1)[-1].split("\nSources:", 1)[0]
    sources = prompt.split("Sources:\n", 1)[-1] if "Sources:\n" in prompt else ""
    sample = int(_kv(prompt, "sample", "0") or 0)
    verdict = {
        rid: bool(PREDICATES.get(rid, lambda a, s: False)(answer, sources)) for rid, _ in rules
    }
    digest = int(hashlib.blake2b(answer.encode(), digest_size=32).hexdigest()[:8], 16)
    if sample == 2 and rules and digest % 9 == 0:  # one noisy sample so the majority vote matters
        first = rules[0][0]
        verdict[first] = not verdict[first]
    passed = sum(verdict.values())
    grade = 1 + round(4 * passed / max(1, len(verdict))) + ((digest + sample) % 3) - 1
    grade = max(1, min(5, grade))
    failed = [r for r, ok in verdict.items() if not ok]
    reason = "All rules satisfied." if not failed else "Failed rules: " + ", ".join(failed) + "."
    lines = [f"{rid}: {'pass' if ok else 'fail'}" for rid, ok in verdict.items()]
    return "\n".join([*lines, f"grade: {grade}", f"reason: {reason}"])


# --------------------------------------------------------------------------- dispatcher


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


SPECIALIST_TOOLS = {
    "holdings": {"get_holdings"},
    "transactions": {"get_transactions"},
    "prices": {"get_prices", "get_holdings"},
    "notes": {"search_notes"},
}


def plan_kinds(q: str) -> list[dict]:
    """v1 planner: which kinds of data does this question need?"""
    if RE_TX.search(q):
        kinds = ["transactions"]
    elif RE_NOTES.search(q):
        kinds = ["notes"]
    elif RE_VALUE.search(q) or (tickers_in(q) and RE_PRICE.search(q)):
        kinds = ["prices"]
    elif RE_PORT.search(q):
        kinds = ["holdings"]
    else:
        kinds = []
    return [{"kind": k, "question": q} for k in kinds]


def respond(messages: list[BaseMessage], allowed: set[str]) -> Reply:
    scripts = os.environ.get("FAKE_SCRIPTS")
    if scripts and os.path.exists(scripts):
        hit = json.load(open(scripts)).get(prompt_hash(messages))
        if hit is not None:
            return Reply(text=hit)
    role, sys = _role(messages), _sys(messages)
    ref = demo_today()
    q, results = _turn(messages)
    cid = _kv(sys, "account_id", "c1")
    if role == "planner-v1":
        return Reply(json.dumps(plan_kinds(q)))
    if role == "triage-v2":
        data = RE_TX.search(q) or RE_NOTES.search(q) or RE_PORT.search(q) or tickers_in(q)
        return Reply("portfolio" if data else "other")
    if role.startswith("specialist-"):
        kind = role.split("-", 1)[1]
        calls = plan_tools(q, cid, ref, results, SPECIALIST_TOOLS.get(kind, set()) & allowed)
        if calls:
            return Reply(tool_calls=_tool_calls_message(calls, prompt_hash(messages)))
        return Reply(summarize_v1(kind, results))
    if role == "writer-v1":
        question = q.split("Question:", 1)[-1].split("\n\nBriefs:", 1)[0].strip()
        briefs = [
            ln.split(": ", 1)[1]
            for ln in q.split("Briefs:", 1)[-1].splitlines()
            if ln.startswith("- ") and ": " in ln
        ]
        if briefs and all(b.strip() == "No data available." for b in briefs):
            analysis = "No data available."
        else:
            analysis = " ".join(b for b in briefs if b.strip() != "No data available.")
        return Reply(compose_v1(question, analysis))
    if role == "answer-v2":
        calls = plan_tools(q, cid, ref, results, allowed)
        if calls:
            return Reply(tool_calls=_tool_calls_message(calls, prompt_hash(messages)))
        return Reply(compose_v2(q, results, ref, _kv(sys, "nickname", "")))
    if role in ("report-v1", "report-v2"):
        data = json.loads(q.split("DATA:\n", 1)[-1])
        return Reply(report_text(data, role == "report-v2", ref))
    if role == "judge":
        return Reply(judge_text(q))
    return Reply("OK")


# --------------------------------------------------------------------------- the model


class FakeChatModel(BaseChatModel):
    model_name: str = "fake"
    allowed_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-deterministic"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        names = []
        for t in tools:
            names.append(
                getattr(t, "name", None)
                or (t.get("name") if isinstance(t, dict) else None)
                or t["function"]["name"]
            )
        return self.model_copy(update={"allowed_tools": names})

    def _reply(self, messages: list[BaseMessage]) -> tuple[Reply, float, float]:
        reply = respond(messages, set(self.allowed_tools))
        in_tok = sum(tokens(_text(m)) for m in messages)
        out_tok = tokens(reply.text + json.dumps(reply.tool_calls))
        prefill, per_tok = _latency(in_tok, out_tok)
        return reply, prefill, per_tok

    @staticmethod
    def _message(reply: Reply) -> AIMessage:
        return AIMessage(content=reply.text, tool_calls=reply.tool_calls)

    def _generate(
        self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs
    ) -> ChatResult:
        reply, prefill, per_tok = self._reply(messages)
        time.sleep(prefill + per_tok * tokens(reply.text + json.dumps(reply.tool_calls)))
        return ChatResult(generations=[ChatGeneration(message=self._message(reply))])

    async def _agenerate(
        self,
        messages,
        stop=None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        reply, prefill, per_tok = self._reply(messages)
        await asyncio.sleep(prefill + per_tok * tokens(reply.text + json.dumps(reply.tool_calls)))
        return ChatResult(generations=[ChatGeneration(message=self._message(reply))])

    @staticmethod
    def _chunks(reply: Reply) -> list[AIMessageChunk]:
        if reply.tool_calls:
            return [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": c["name"],
                            "args": json.dumps(c["args"]),
                            "id": c["id"],
                            "index": i,
                        }
                        for i, c in enumerate(reply.tool_calls)
                    ],
                )
            ]
        words = re.findall(r"\S+\s*", reply.text) or [""]
        return [AIMessageChunk(content=w) for w in words]

    def _stream(
        self, messages, stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        reply, prefill, per_tok = self._reply(messages)
        time.sleep(prefill)
        for ch in self._chunks(reply):
            time.sleep(per_tok * tokens(_text(ch) or "x"))
            yield ChatGenerationChunk(message=ch)

    async def _astream(
        self,
        messages,
        stop=None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> AsyncIterator[ChatGenerationChunk]:
        reply, prefill, per_tok = self._reply(messages)
        await asyncio.sleep(prefill)
        for ch in self._chunks(reply):
            await asyncio.sleep(per_tok * tokens(_text(ch) or "x"))
            yield ChatGenerationChunk(message=ch)
