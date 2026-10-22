import subprocess
from pathlib import Path

import httpx
import pytest

from bench import security
from common.metrics import COUNTER
from deploy.edge_rules import decide
from gateway.app import create_app as create_gateway
from server.app import create_app as create_server

ROOT = Path(__file__).resolve().parent.parent
BROKEN = ROOT / "deploy/edge/start-edge.sh"
FIXED = ROOT / "deploy/edge/start-edge.fixed.sh"
OUTSIDE, INSIDE = "203.0.113.9", "10.77.0.5"


def render(script: Path, tmp_path: Path, **env) -> Path:
    out = tmp_path / "conf.d" / "advisor.conf"
    subprocess.run(
        ["sh", str(script), str(out)],
        env={"PATH": "/usr/bin:/bin", **env},
        check=True,
    )
    return out


def test_broken_env_allows_everyone(tmp_path):
    conf = render(BROKEN, tmp_path, EDGE_ACCESS="internal", OPEN_DEBUG="1")
    assert decide(OUTSIDE, conf.read_text()) == "allow"
    assert decide(INSIDE, conf.read_text()) == "allow"
    text = conf.read_text()
    assert text.index("allow all;") < text.index("deny all;")


def test_compose_env_alone_is_silently_open(tmp_path):
    """Exactly what compose sets: EDGE_ACCESS=internal, nothing else. The script still opens."""
    conf = render(BROKEN, tmp_path / "a", EDGE_ACCESS="internal")
    assert decide(OUTSIDE, conf.read_text()) == "allow"
    assert decide(INSIDE, conf.read_text()) == "allow"
    fixed = render(FIXED, tmp_path / "b", EDGE_ACCESS="internal")
    assert decide(OUTSIDE, fixed.read_text()) == "deny"
    assert decide(INSIDE, fixed.read_text()) == "allow"


def test_compose_files_never_mention_open_debug():
    for name in ("docker-compose.yml", "docker-compose.fixed.yml"):
        assert "OPEN_DEBUG" not in (ROOT / "deploy" / name).read_text()


def test_fixed_script_ignores_legacy_switch(tmp_path):
    conf = render(FIXED, tmp_path, EDGE_ACCESS="internal", OPEN_DEBUG="1")
    assert "allow all" not in conf.read_text()
    assert decide(OUTSIDE, conf.read_text()) == "deny"
    assert decide(INSIDE, conf.read_text()) == "allow"


def test_decide_first_match_and_default():
    assert decide("1.2.3.4", "allow 10.0.0.0/8;\ndeny all;") == "deny"
    assert decide("10.1.2.3", "deny 10.0.0.0/8;\nallow all;") == "deny"
    assert decide("1.2.3.4", "") == "allow"


def test_bench_security(tmp_path, capsys):
    broken = render(BROKEN, tmp_path / "a", EDGE_ACCESS="internal")
    fixed = render(FIXED, tmp_path / "b", EDGE_ACCESS="internal")
    assert security.run(edge_conf=str(broken)) == 1
    assert "FAIL" in capsys.readouterr().out
    assert security.run(edge_conf=str(fixed)) == 0
    assert "FAIL" not in capsys.readouterr().out
    assert security.run() == 2


async def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_debug_counter_increments_per_call():
    c = await client_for(create_server(version="v2", agent=object(), debug=True))
    before = (await c.get("/dev-console/model-calls")).json()["calls"]
    for _ in range(3):
        r = await c.post("/dev-console/ask", json={"prompt": "hello"})
        assert r.status_code == 200
    after = (await c.get("/dev-console/model-calls")).json()
    assert after["calls"] == before + 3
    assert after["est_cost_usd"] == pytest.approx(after["calls"] * 0.002)
    assert COUNTER.calls == after["calls"]


async def test_debug_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ADVISOR_DEBUG", raising=False)
    c = await client_for(create_server(version="v2", agent=object()))
    assert (await c.post("/dev-console/ask", json={"prompt": "x"})).status_code == 404
    monkeypatch.setenv("ADVISOR_DEBUG", "1")
    c = await client_for(create_server(version="v2", agent=object()))
    assert (await c.get("/dev-console")).status_code == 200


async def test_gateway_rejects_missing_or_wrong_token():
    c = await client_for(create_gateway(token="s3cret", advisor_url="http://advisor:1"))
    body = {"account_id": "c1", "message": "hi"}
    assert (await c.post("/ask", json=body)).status_code == 401
    assert (await c.post("/ask", json=body, headers={"X-Demo-Token": "nope"})).status_code == 401
    assert (await c.post("/briefing", json=body)).status_code == 401


async def test_gateway_forwards_with_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"answer": "hi"})

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://advisor")
    c = await client_for(create_gateway(token="s3cret", client=upstream))
    r = await c.post(
        "/ask", json={"account_id": "c1", "message": "hi"}, headers={"X-Demo-Token": "s3cret"}
    )
    assert r.status_code == 200 and r.json() == {"answer": "hi"}
    assert seen["path"] == "/ask"
