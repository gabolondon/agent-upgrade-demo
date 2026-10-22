import random

from agent_v2.fingerprint import fingerprint_key, raw_digest, stable_form

ROWS = [
    {"date": f"2026-01-{d:02d}", "close": 10.0 + d / 7, "retrieved_on": "x", "trace_id": "t"}
    for d in range(1, 20)
]


def test_stable_under_row_order_and_volatile_keys():
    shuffled = ROWS[:]
    random.Random(1).shuffle(shuffled)
    for r in shuffled:
        r["trace_id"] = "other"
    assert stable_form(ROWS) == stable_form(shuffled)


def test_float_noise_is_normalised_but_real_changes_are_not():
    a = [{"v": 0.1 + 0.2}]
    b = [{"v": 0.3}]
    assert stable_form(a) == stable_form(b)
    assert stable_form([{"v": 0.3}]) != stable_form([{"v": 0.31}])
    assert stable_form([{"v": -0.0}]) == stable_form([{"v": 0.0}])
    assert stable_form([{"v": 120.0}]) == stable_form([{"v": 120}])


SCOPE = {"currency": "USD", "locale": "en"}


def test_key_depends_on_generator_and_scope_and_is_256_bit():
    k = fingerprint_key(ROWS, "stamp-a", SCOPE)
    assert len(k) == 64
    assert k != fingerprint_key(ROWS, "stamp-b", SCOPE)
    assert k != fingerprint_key(ROWS, "stamp-c", SCOPE)
    assert k != fingerprint_key(ROWS, "stamp-a", {**SCOPE, "currency": "EUR"})
    assert k == fingerprint_key(list(reversed(ROWS)), "stamp-a", dict(reversed(SCOPE.items())))


def test_raw_digest_is_order_sensitive():
    assert raw_digest("[1,2]") != raw_digest("[2,1]")


async def test_live_v1_payloads_raw_unstable_stable_form_constant():
    import json

    from tests.conftest import make_api

    api = make_api()
    texts = [(await api.client.get("/v1/prices/KELP")).text for _ in range(6)]
    assert len({raw_digest(t) for t in texts}) == 6
    assert len({raw_digest(stable_form(json.loads(t))) for t in texts}) == 1
