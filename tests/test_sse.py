import json

import httpx

from agent_v2.agent import AgentV2
from agent_v2.cache import ReportCache
from agent_v2.flags import Flags
from client.sse import iter_events
from server.app import create_app
from tests.conftest import make_api


async def stream(app, path, body):
    out = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as c:
        async with c.stream("POST", path, json=body) as r:
            assert "text/event-stream" in r.headers["content-type"]
            async for ev, data in iter_events(r):
                out.append((ev, data))
    return out


def assert_order(events, closing="done"):
    """working* / token* ... cited (once) ... then done | failed, nothing after."""
    names = [e for e, _ in events]
    assert set(names) <= {"working", "token", "cited", "done", "failed"}
    assert names[-1] == closing and names.count(closing) == 1
    if closing == "done":
        assert names[-2] == "cited" and names.count("cited") == 1
        assert all(n in ("working", "token") for n in names[:-2])
    else:
        assert "cited" not in names and "done" not in names
        assert all(n in ("working", "token") for n in names[:-1])
    assert not any("ok" in d for _, d in events)


async def test_ask_event_order_and_done_payload(settings):
    agent = AgentV2(Flags(), settings, make_api(), ReportCache(":memory:"))
    app = create_app("v2", Flags(), settings, agent)
    events = await stream(
        app, "/ask", {"account_id": "c2", "message": "What do the research notes say about KELP?"}
    )
    assert_order(events)
    assert events[0][0] == "working" and "label" in events[0][1]
    texts = [d["text"] for e, d in events if e == "token"]
    cited, done = events[-2][1], events[-1][1]
    assert "".join(texts).strip() == done["answer"] and set(done) == {"answer"}
    assert cited["items"] and set(cited["items"][0]) == {"title", "href"}


async def test_failure_ends_with_failed_and_message(settings):
    agent = AgentV2(Flags(), settings, make_api(), ReportCache(":memory:"))

    class Boom:
        async def ainvoke(self, *_a, **_k):
            raise RuntimeError("model down")

    agent.triage_model = Boom()
    events = await stream(
        create_app("v2", Flags(), settings, agent),
        "/ask",
        {"account_id": "c1", "message": "What do I hold?"},
    )
    assert_order(events, closing="failed")
    assert "model down" in events[-1][1]["message"] and set(events[-1][1]) == {"message"}


async def test_open_stream_and_non_streaming_fallback(settings):
    agent = AgentV2(Flags(), settings, make_api(), ReportCache(":memory:"))
    events = await stream(
        create_app("v2", Flags(), settings, agent), "/briefing", {"account_id": "c1"}
    )
    assert_order(events)
    assert events[-1][0] == "done" and "answer" in events[-1][1]

    flags = Flags(streaming=False)
    agent2 = AgentV2(flags, settings, make_api(), ReportCache(":memory:"))
    app = create_app("v2", flags, settings, agent2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as c:
        r = await c.post("/ask", json={"account_id": "c1", "message": "What do I hold?"})
    assert r.headers["content-type"].startswith("application/json")
    assert "answer" in r.json() and r.json()["cited"] == []


def test_event_json_roundtrip():
    assert json.loads(json.dumps({"answer": "a"}))["answer"] == "a"
