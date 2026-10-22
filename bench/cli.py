"""Benchmark CLI: `run` a suite against a URL, `compare` two result files."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_common.llm import make_chat_model
from bench.judge import judge_answer
from bench.report import build_report, delta_markdown, delta_rows, to_markdown
from bench.runner import read_scenario_file, run_suite
from common.config import Settings, load_dotenv


async def run_benchmark(
    url: str,
    label: str,
    out: str,
    *,
    repeat: int = 1,
    timeout: float = 60.0,
    judge_samples: int = 3,
    scenarios: str | None = None,
    meta: dict | None = None,
) -> dict:
    settings = Settings.from_env()
    judge = make_chat_model(settings.judge_model, settings)
    sc = read_scenario_file(scenarios) if scenarios else read_scenario_file()
    runs = [
        await run_suite(url, sc, judge, timeout=timeout, judge_samples=judge_samples, run_index=i)
        for i in range(repeat)
    ]
    rep = build_report(label, runs, meta)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=2))
    p.with_suffix(".md").write_text(to_markdown(rep))
    return rep


async def rejudge(in_path: str, out_path: str, samples: int = 3) -> dict:
    """Re-run only the judge on the answers stored in a result file; rebuild the report."""
    old = json.loads(Path(in_path).read_text())
    settings = Settings.from_env()
    judge = make_chat_model(settings.judge_model, settings)
    rule_texts = read_scenario_file()["rules"]
    for run in old["runs"]:
        for q in run["questions"]:
            if q["timeout"] or q["error"]:
                continue  # nothing to judge: keep the original no-answer record
            rules = {rid: rule_texts[rid] for rid in q["rules"]}
            v = await judge_answer(judge, q["question"], q["answer"], q["cited"], rules, samples)
            q.update(rules=v["rules"], grade=v["grade"], reason=v["reason"], judge_raw=v["raw"])
    rep = build_report(f"{old['label']} (rejudged)", old["runs"], old.get("meta"))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=2))
    p.with_suffix(".md").write_text(to_markdown(rep))
    return rep


def compare(a_path: str, b_path: str) -> None:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    table = Table(title=f"{a['label']} -> {b['label']}")
    for col, just in (
        ("Metric", "left"),
        (a["label"], "right"),
        (b["label"], "right"),
        ("Delta", "right"),
    ):
        table.add_column(col, justify=just)
    for row in delta_rows(a, b):
        table.add_row(*row)
    Console(width=100).print(table)
    print("\n" + delta_markdown(a, b))


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--timeout", type=float, default=60.0)
    r.add_argument("--judge-samples", type=int, default=3)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    rj = sub.add_parser("rejudge", help="re-run only the judge on a stored result file")
    rj.add_argument("inp")
    rj.add_argument("out")
    rj.add_argument("--samples", type=int, default=3)
    sec = sub.add_parser("security", help="check that outside callers cannot reach the advisor")
    sec.add_argument("--edge-conf", help="rendered advisor.conf (offline, no Docker)")
    sec.add_argument("--url", help="live edge, e.g. http://localhost:8088")
    args = ap.parse_args()
    if args.cmd == "compare":
        compare(args.a, args.b)
    elif args.cmd == "rejudge":
        print(to_markdown(asyncio.run(rejudge(args.inp, args.out, args.samples))))
    elif args.cmd == "security":
        from bench.security import run

        raise SystemExit(run(args.edge_conf, args.url))
    else:
        rep = asyncio.run(
            run_benchmark(
                args.url,
                args.label,
                args.out,
                repeat=args.repeat,
                timeout=args.timeout,
                judge_samples=args.judge_samples,
            )
        )
        print(to_markdown(rep))


if __name__ == "__main__":
    main()
