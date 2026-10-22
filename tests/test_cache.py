import asyncio

import pytest

from agent_v2.agent import AgentV2
from agent_v2.cache import ReportCache
from agent_v2.flags import Flags
from tests.conftest import make_api


async def test_single_flight_computes_once():
    cache, calls = ReportCache(":memory:"), 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "report"

    results = await asyncio.gather(*(cache.get_or_compute("k", compute) for _ in range(8)))
    assert calls == 1
    assert [v for v, _ in results] == ["report"] * 8
    assert sum(1 for _, cached in results if not cached) == 1


async def test_failed_or_empty_compute_is_never_cached():
    cache = ReportCache(":memory:")

    async def boom():
        raise RuntimeError("model down")

    with pytest.raises(RuntimeError):
        await cache.get_or_compute("k", boom)
    assert cache.get("k") is None

    async def empty():
        return "  "

    await cache.get_or_compute("k", empty)
    assert cache.get("k") is None

    async def good():
        return "ok"

    assert await cache.get_or_compute("k", good) == ("ok", False)
    assert await cache.get_or_compute("k", good) == ("ok", True)


async def test_report_cached_only_after_done(settings):
    cache = ReportCache(":memory:")
    agent = AgentV2(Flags(), settings, make_api(), cache)
    first = await agent.open_report("c1")
    second = await agent.open_report("c1")
    assert first["cached"] is False
    assert second["cached"] is True and second["answer"] == first["answer"]

    class Boom:
        async def ainvoke(self, *_a, **_k):
            raise RuntimeError("model down")

    cache2 = ReportCache(":memory:")
    broken = AgentV2(Flags(), settings, make_api(), cache2)
    broken.answer_model = Boom()
    events = [e async for e in broken.open_events("c1")]
    assert events[-1]["event"] == "failed" and "model down" in events[-1]["data"]["message"]
    assert "done" not in [e["event"] for e in events]
    assert cache2.db.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 0


async def test_cache_flag_off_never_reports_cached(settings):
    agent = AgentV2(Flags(cache=False), settings, make_api())
    a, b = await agent.open_report("c1"), await agent.open_report("c1")
    assert a["cached"] is False and b["cached"] is False
