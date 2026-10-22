"""LLM judge: per-rule pass/fail, a 1-5 grade and a reason; sampled N times and aggregated."""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

from agent_common import prompts


def build_prompt(
    question: str, answer: str, cited: list, rules: dict[str, str], sample: int
) -> str:
    rule_lines = "\n".join(f"- [{rid}] {text}" for rid, text in rules.items())
    return (
        f"Question: {question}\nRules:\n{rule_lines}\nAnswer:\n{answer}\n"
        f"Sources:\n{json.dumps(cited)}\nsample={sample}"
    )


_MARKER = re.compile(r"^(?:[-*\u2022]+\s+|\d+[.)]\s+)")
_JUNK = " \t[]*`\"'"
PASS_PREFIXES = ("pass", "yes", "\u2705", "\u2713", "\u2714")


def _norm(text: str) -> str:
    """Strip list markers, brackets, bold, backticks and quotes; lowercase."""
    return _MARKER.sub("", text.strip()).strip(_JUNK).lower()


def parse_verdict(text: str, rule_ids: list[str]) -> dict:
    """Parse one judge reply: a pass/fail line per rule id, `grade: N`, `reason: ...`.

    Keys and values tolerate markdown (`- **rule**: PASS`, `[rule]: pass`, backticks, a tick mark).
    A rule with no verdict line counts as failed; a missing or out-of-range grade counts as 1.
    """
    by_norm = {rid.lower(): rid for rid in rule_ids}
    rules = dict.fromkeys(rule_ids, False)
    grade, reason = 1, ""
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key = _norm(key)
        if key in by_norm:
            rules[by_norm[key]] = _norm(val).startswith(PASS_PREFIXES)
        elif key == "grade":
            m = re.match(r"[1-5]\b", _norm(val))
            grade = int(m.group(0)) if m else 1
        elif key == "reason":
            reason = val.strip().strip("*`").strip()
    return {"rules": rules, "grade": grade, "reason": reason}


def aggregate(votes: list[dict]) -> dict:
    """Majority vote per rule and median grade. An odd sample count means there is always a majority."""
    n = len(votes)
    if n % 2 == 0:
        raise ValueError("use an odd number of judge samples")
    rule_ids = list(votes[0]["rules"])
    rules = {rid: Counter(v["rules"][rid] for v in votes).most_common(1)[0][0] for rid in rule_ids}
    reasons = [v["reason"] for v in votes]
    return {
        "rules": rules,
        "grade": statistics.median(v["grade"] for v in votes),
        "reason": reasons[len(reasons) // 2],
    }


async def judge_answer(
    model, question: str, answer: str, cited: list, rules: dict[str, str], samples: int = 3
) -> dict:
    async def one(i: int) -> tuple[dict, str]:
        msg = await model.ainvoke(
            [
                SystemMessage(prompts.JUDGE),
                HumanMessage(build_prompt(question, answer, cited, rules, i)),
            ]
        )
        raw = str(msg.content)
        return parse_verdict(raw, list(rules)), raw

    pairs = await asyncio.gather(*(one(i) for i in range(samples)))
    out = aggregate([v for v, _ in pairs])
    out["raw"] = pairs[0][1][:600]  # first sample, kept so parsing can be audited later
    return out
