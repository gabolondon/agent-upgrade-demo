"""Aggregate benchmark records into metrics, markdown, and a delta table."""

from __future__ import annotations

import statistics


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def titled(cited: list) -> bool:
    return bool(cited) and all(isinstance(c, dict) and c.get("title") for c in cited)


def build_report(label: str, runs: list[dict], meta: dict | None = None) -> dict:
    """`runs` is a list of suite runs; each has `questions` and `opens` record lists."""
    qs = [q for r in runs for q in r["questions"]]
    opens = [o for r in runs for o in r["opens"]]
    ok_qs = [q for q in qs if not q["timeout"] and not q["error"]]
    correct = sum(1 for q in qs if q["rules"] and all(q["rules"].values()))
    failed_rules = sum(sum(1 for v in q["rules"].values() if not v) for q in qs)
    with_fail = sum(1 for q in qs if any(not v for v in q["rules"].values()))
    conv: dict[tuple, list[bool]] = {}
    for q in qs:
        conv.setdefault((q["run"], q["conversation"]), []).append(
            bool(q["rules"]) and all(q["rules"].values())
        )
    totals = [q["total_s"] for q in ok_qs]
    return {
        "label": label,
        "meta": meta or {},
        "questions_total": len(qs),
        "correct_answers": correct,
        "conversations_total": len(conv),
        "conversations_fully_passing": sum(all(v) for v in conv.values()),
        "mean_grade": round(statistics.mean(q["grade"] for q in qs), 2) if qs else 0.0,
        "failed_rules": failed_rules,
        "questions_with_failed_rules": with_fail,
        "answers_with_titled_sources": sum(1 for q in qs if titled(q["cited"])),
        "latency_p50_s": round(percentile(totals, 50), 3),
        "latency_p95_s": round(percentile(totals, 95), 3),
        "latency_max_s": round(max(totals, default=0.0), 3),
        "first_event_p50_s": round(percentile([q["first_event_s"] for q in ok_qs], 50), 3),
        "open_p50_s": round(
            percentile([o["total_s"] for o in opens if o["kind"] == "open"], 50), 3
        ),
        "reopen_p50_s": round(
            percentile([o["total_s"] for o in opens if o["kind"] == "reopen"], 50), 3
        ),
        "timeouts": sum(1 for q in qs if q["timeout"]),
        "errors": sum(1 for q in qs if q["error"]),
        "runs": runs,
    }


METRICS = [  # (key, label, higher_is_better)
    ("correct_answers", "Correct answers", True),
    ("conversations_fully_passing", "Conversations fully passing", True),
    ("mean_grade", "Mean judge grade (1-5, 0 = no answer)", True),
    ("failed_rules", "Failed rules", False),
    ("questions_with_failed_rules", "Questions with a failed rule", False),
    ("answers_with_titled_sources", "Answers with titled sources", True),
    ("latency_p50_s", "Latency p50 (s)", False),
    ("latency_p95_s", "Latency p95 (s)", False),
    ("latency_max_s", "Latency max (s)", False),
    ("first_event_p50_s", "First event p50 (s)", False),
    ("open_p50_s", "Open report p50 (s)", False),
    ("reopen_p50_s", "Reopen report p50 (s)", False),
    ("timeouts", "Timeouts", False),
    ("errors", "Errors", False),
]


def delta_rows(a: dict, b: dict) -> list[tuple[str, str, str, str]]:
    rows = []
    for key, label, _ in METRICS:
        va, vb = a[key], b[key]
        d = vb - va
        rows.append((label, f"{va:g}", f"{vb:g}", f"{d:+g}"))
    return rows


def delta_markdown(a: dict, b: dict) -> str:
    out = [f"| Metric | {a['label']} | {b['label']} | Delta |", "|---|---:|---:|---:|"]
    out += [f"| {' | '.join(r)} |" for r in delta_rows(a, b)]
    return "\n".join(out)


def to_markdown(rep: dict) -> str:
    n = rep["questions_total"]
    lines = [f"# Benchmark: {rep['label']}", "", f"Questions: {n}", ""]
    lines += ["| Metric | Value |", "|---|---:|"]
    lines += [f"| {label} | {rep[key]:g} |" for key, label, _ in METRICS]
    return "\n".join(lines) + "\n"
