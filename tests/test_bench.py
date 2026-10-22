import pytest

from bench.judge import aggregate, build_prompt, parse_verdict
from bench.report import build_report, delta_rows, percentile
from bench.runner import read_scenario_file


def vote(rules, grade, reason="r"):
    return {"rules": rules, "grade": grade, "reason": reason}


def test_majority_vote_and_median_grade():
    votes = [
        vote({"a": True, "b": False}, 5),
        vote({"a": True, "b": True}, 3),
        vote({"a": False, "b": False}, 4),
    ]
    out = aggregate(votes)
    assert out["rules"] == {"a": True, "b": False}
    assert out["grade"] == 4


def test_even_sample_count_is_rejected():
    with pytest.raises(ValueError):
        aggregate([vote({"a": True}, 1), vote({"a": False}, 3)])


def test_parse_verdict_handles_noise_and_garbage():
    ok = parse_verdict("Sure!\na: pass\nb: FAIL\ngrade: 4\nreason: fine", ["a", "b", "c"])
    assert ok["rules"] == {"a": True, "b": False, "c": False}
    assert ok["grade"] == 4 and ok["reason"] == "fine"
    bad = parse_verdict("nope", ["a"])
    assert bad["rules"] == {"a": False} and bad["grade"] == 1
    assert parse_verdict("a: pass\ngrade: 9", ["a"])["grade"] == 1  # out of range


def test_prompt_lists_rules_and_sample():
    p = build_prompt("q?", "a.", [], {"r1": "does x"}, 2)
    assert "- [r1] does x" in p and p.endswith("sample=2")


def test_percentiles():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 50) == 3
    assert percentile(xs, 0) == 1 and percentile(xs, 100) == 5
    assert percentile(xs, 95) == pytest.approx(4.8)
    assert percentile([], 50) == 0.0
    assert percentile([7], 95) == 7


def q(conv, rules, total, cited=(), timeout=False, error=False, grade=5):
    return {
        "conversation": conv,
        "run": 0,
        "rules": rules,
        "total_s": total,
        "first_event_s": total / 2,
        "cited": list(cited),
        "timeout": timeout,
        "error": error,
        "grade": grade,
    }


def test_report_metrics():
    runs = [
        {
            "questions": [
                q("c1", {"a": True}, 1.0, [{"title": "T", "href": "u"}]),
                q("c1", {"a": True, "b": False}, 2.0, ["notes/x.md"], grade=2),
                q("c2", {"a": True}, 3.0),
                q("c2", {"a": False}, 0.0, timeout=True, grade=0),
            ],
            "opens": [{"kind": "open", "total_s": 4.0}, {"kind": "reopen", "total_s": 0.1}],
        }
    ]
    rep = build_report("x", runs)
    assert rep["questions_total"] == 4 and rep["correct_answers"] == 2
    assert rep["conversations_total"] == 2 and rep["conversations_fully_passing"] == 0
    assert rep["failed_rules"] == 2 and rep["questions_with_failed_rules"] == 2
    assert rep["answers_with_titled_sources"] == 1
    assert rep["latency_max_s"] == 3.0 and rep["timeouts"] == 1
    assert rep["mean_grade"] == 3.0
    assert rep["open_p50_s"] == 4.0 and rep["reopen_p50_s"] == 0.1


def test_delta_rows():
    a = build_report("a", [{"questions": [q("c", {"r": False}, 2.0)], "opens": []}])
    b = build_report("b", [{"questions": [q("c", {"r": True}, 1.0)], "opens": []}])
    rows = {r[0]: r for r in delta_rows(a, b)}
    assert rows["Correct answers"][3] == "+1" and rows["Latency p50 (s)"][3] == "-1"


def test_scenarios_shape():
    sc = read_scenario_file()
    assert len(sc["conversations"]) == 5
    qs = [x for c in sc["conversations"] for x in c["questions"]]
    assert len(qs) == 20
    assert all(1 <= len(x["rules"]) <= 4 and set(x["rules"]) <= set(sc["rules"]) for x in qs)


RULES = ["lists_holdings", "states_as_of"]


@pytest.mark.parametrize(
    "reply",
    [
        "lists_holdings: pass\nstates_as_of: fail\ngrade: 4\nreason: ok",
        "[lists_holdings]: pass\n[states_as_of]: fail\ngrade: 4\nreason: ok",
        "- lists_holdings: pass\n- states_as_of: fail\n- grade: 4\n- reason: ok",
        "**lists_holdings**: PASS\n**states_as_of**: FAIL\n**grade**: 4\n**reason**: ok",
        "`lists_holdings`: `pass`\n`states_as_of`: `fail`\ngrade: 4/5\nreason: ok",
        "1. Lists_Holdings: ✅ Pass\n2. States_As_Of: ❌ Fail\nGrade: 4\nReason: ok",
        "* [lists_holdings]: Yes\n* [states_as_of]: No\n* grade: **4**\n* reason: ok",
        '"lists_holdings": "pass"\n"states_as_of": "fail"\ngrade: 4\nreason: ok',
        "**lists_holdings: pass**\n**states_as_of: fail**\n**grade: 4**\n**reason: ok**",
    ],
)
def test_parse_verdict_tolerates_markdown_formats(reply):
    v = parse_verdict(reply, RULES)
    assert v["rules"] == {"lists_holdings": True, "states_as_of": False}
    assert v["grade"] == 4 and v["reason"] == "ok"


def test_parse_verdict_tick_mark_and_unknown_lines():
    v = parse_verdict("lists_holdings: ✓\nother_rule: pass\nstates_as_of: passed", RULES)
    assert v["rules"] == {"lists_holdings": True, "states_as_of": True}
    assert parse_verdict("lists_holdings: not passing", RULES)["rules"]["lists_holdings"] is False


async def test_judge_answer_keeps_truncated_raw_of_first_sample():
    from bench.judge import judge_answer

    class Model:
        n = 0

        async def ainvoke(self, _msgs):
            Model.n += 1
            return type("M", (), {"content": "a: pass\ngrade: 5\nreason: " + "x" * 900})()

    out = await judge_answer(Model(), "q", "ans", [], {"a": "rule a"}, samples=3)
    assert out["rules"] == {"a": True} and out["grade"] == 5
    assert len(out["raw"]) == 600 and out["raw"].startswith("a: pass")


async def test_rejudge_rebuilds_report_from_stored_answers(tmp_path, monkeypatch):
    import json

    from bench.cli import rejudge

    def rec(**kw):
        base = {
            "conversation": "c", "run": 0, "question": "What do I hold?", "answer": "KELP and NORD.",
            "cited": [], "rules": {"lists_holdings": False}, "grade": 1, "reason": "old",
            "total_s": 2.5, "first_event_s": 0.5, "timeout": False, "error": False,
        }  # fmt: skip
        return {**base, **kw}

    old = {
        "label": "v2", "meta": {"flags": "x"},
        "runs": [{"questions": [rec(), rec(timeout=True, answer="", total_s=60.0)], "opens": []}],
    }  # fmt: skip
    src, dst = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps(old))
    rep = await rejudge(str(src), str(dst), samples=1)
    assert rep["label"] == "v2 (rejudged)" and rep["meta"] == {"flags": "x"}
    qs = rep["runs"][0]["questions"]
    assert qs[0]["rules"] == {"lists_holdings": True} and qs[0]["judge_raw"]
    assert qs[1]["timeout"] and qs[1]["rules"] == {"lists_holdings": False}  # untouched
    assert rep["latency_max_s"] == 2.5 and rep["timeouts"] == 1
    assert (
        json.loads(dst.read_text())["label"] == "v2 (rejudged)" and dst.with_suffix(".md").exists()
    )
