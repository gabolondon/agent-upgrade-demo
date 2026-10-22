import json

from agent_common.tools import VAGUE_EMPTY, make_tools
from tests.conftest import make_api


async def call(tools, name, **args):
    return await tools[name].ainvoke(args)


async def test_typed_results_worked_or_not():
    api = make_api(fail_rate=1.0)
    tools = make_tools(api, compact=True, typed=True)
    ok = json.loads(await call(tools, "get_prices", ticker="KELP"))
    empty = json.loads(await call(tools, "get_prices", ticker="TIDE"))
    failed = json.loads(await call(tools, "get_transactions", account_id="c6"))
    assert ok["worked"] is True and ok["data"]["header"]
    assert empty["worked"] is True and not empty["data"]["table"]
    assert failed["worked"] is False and failed["reason"]


async def test_typed_works_on_verbose_payloads_too():
    tools = make_tools(make_api(), compact=False, typed=True)
    out = json.loads(await call(tools, "get_holdings", account_id="c1"))
    assert out["worked"] is True and isinstance(out["data"], list)


async def test_untyped_collapses_failure_and_empty_into_one_string():
    tools = make_tools(make_api(fail_rate=1.0), compact=False, typed=False)
    assert await call(tools, "get_prices", ticker="TIDE") == VAGUE_EMPTY
    assert await call(tools, "get_transactions", account_id="c6") == VAGUE_EMPTY
