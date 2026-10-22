from agent_v1.graph import AgentV1, parse_plan
from agent_v2.agent import AgentV2
from agent_v2.cache import ReportCache
from agent_v2.flags import Flags
from tests.conftest import make_api


async def v2_answer(agent, cid, q):
    res = await agent.chat(cid, q)
    assert "answer" in res, res
    return res


async def test_planner_parse_is_defensive():
    assert parse_plan(
        '[{"kind": "prices", "question": "x"}, {"kind": "bogus"}, {"kind": "prices"}]'
    ) == [{"kind": "prices", "question": "x"}]
    assert parse_plan("not json") == []


async def test_v1_flattens_failure_into_empty_v2_does_not(settings):
    v1 = AgentV1(settings, make_api())
    v2 = AgentV2(Flags(), settings, make_api(), ReportCache(":memory:"))
    q = "How much did I spend on purchases in September?"
    a1 = (await v1.chat("c6", q))["answer"]
    a2 = (await v2_answer(v2, "c6", q))["answer"]
    assert "unavailable right now" not in a1 and "unavailable right now" in a2


async def test_v2_states_price_date_and_says_missing(settings):
    v2 = AgentV2(Flags(), settings, make_api(), ReportCache(":memory:"))
    dated = (await v2_answer(v2, "c1", "What was the most recent closing price of KELP?"))["answer"]
    assert "2026-09-30" in dated
    missing = (await v2_answer(v2, "c5", "What is TIDE trading at?"))["answer"]
    assert "no price data" in missing


async def test_sources_v1_paths_v2_titled(settings):
    q = "What do the research notes say about KELP?"
    s1 = (await AgentV1(settings, make_api()).chat("c2", q))["cited"]
    s2 = (
        await v2_answer(AgentV2(Flags(), settings, make_api(), ReportCache(":memory:")), "c2", q)
    )["cited"]
    assert s1 and all(isinstance(s, str) for s in s1)
    assert s2 and all({"title", "href"} == set(s) for s in s2)


async def test_price_target_only_without_tool_result_reading(settings):
    q = "Should I sell NORD now?"
    a1 = (await AgentV1(settings, make_api()).chat("c1", q))["answer"]
    a2 = (
        await v2_answer(AgentV2(Flags(), settings, make_api(), ReportCache(":memory:")), "c1", q)
    )["answer"]
    assert "price target" in a1 and "price target" not in a2 and "can't recommend" in a2


async def test_every_flag_can_be_ablated(settings):
    for name in ("cache", "direct_tools", "compact", "model_per_role", "streaming"):
        agent = AgentV2(Flags(**{name: False}), settings, make_api(), ReportCache(":memory:"))
        res = await agent.chat("c1", "What do I hold?")
        assert "answer" in res, name
        assert "answer" in await agent.open_report("c1"), name


def test_flags_from_cli():
    f = Flags.from_cli("CACHE=0,COMPACT=0")
    assert not f.cache and not f.compact and f.direct_tools
    assert f.label() == "off:CACHE,COMPACT"
