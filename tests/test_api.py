import httpx
import pytest

from portfolio_api.app import create_app


def client(**kw):
    app = create_app(overhead_ms=0, ms_per_1k_rows=0, **kw)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


async def test_compact_payload_shape_and_default_window():
    async with client(fail_rate=0) as c:
        body = (await c.get("/v2/prices/KELP")).json()
    assert set(body) == {"header", "table"}
    assert body["header"] == ["date", "close", "volume"]
    assert all(len(r) == 3 for r in body["table"])
    dates = [r[0] for r in body["table"]]
    assert dates == sorted(dates)
    assert 88 <= len(dates) <= 92  # default last 90 days


async def test_compact_window_is_bounded():
    async with client(fail_rate=0) as c:
        ok = await c.get(
            "/v2/prices/KELP", params={"window_start": "2026-01-01", "window_end": "2026-12-31"}
        )
        too_wide = await c.get(
            "/v2/prices/KELP", params={"window_start": "2024-01-01", "window_end": "2026-09-30"}
        )
        inverted = await c.get(
            "/v2/prices/KELP", params={"window_start": "2026-09-30", "window_end": "2026-01-01"}
        )
    assert ok.status_code == 200
    assert too_wide.status_code == 422
    assert inverted.status_code == 422


async def test_v1_order_is_unstable_v2_is_stable():
    async with client(fail_rate=0) as c:
        a = (await c.get("/v1/prices/KELP")).json()
        b = (await c.get("/v1/prices/KELP")).json()
        v2a = (await c.get("/v2/prices/KELP")).json()
        v2b = (await c.get("/v2/prices/KELP")).json()
    assert [r["date"] for r in a] != [r["date"] for r in b]
    assert {r["date"] for r in a} == {r["date"] for r in b}
    assert v2a == v2b
    assert len(a) > 1000  # whole history


async def test_seeded_gaps_exist():
    async with client(fail_rate=1.0) as c:
        missing = (await c.get("/v1/prices/TIDE")).json()
        flaky = await c.get("/v1/accounts/c6/transactions")
        fine = await c.get("/v1/accounts/c1/transactions")
    assert missing == []
    assert flaky.status_code == 503
    assert fine.status_code == 200


async def test_unknown_account_404():
    async with client() as c:
        assert (await c.get("/v1/accounts/zz/holdings")).status_code == 404


@pytest.mark.parametrize("path", ["/v1/notes?q=rates", "/v2/notes?q=rates"])
async def test_notes_search(path):
    async with client() as c:
        assert (await c.get(path)).status_code == 200
