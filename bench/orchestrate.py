"""Start the API and both agent servers, run the suite against v1 and v2, shut everything down.

uv run python -m bench.orchestrate --out results [--v2-flags "CACHE=0,COMPACT=0"] [--repeat 2]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from agent_v2.flags import Flags
from bench.cli import run_benchmark
from common.config import load_dotenv


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server for {url} exited early")
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise TimeoutError(url)


@contextmanager
def server(module: str, port: int, env: dict):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module,
            "--factory",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env={**os.environ, **env},
    )
    try:
        wait_ready(f"http://127.0.0.1:{port}", proc)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--v2-flags", default="", help='e.g. "CACHE=0,COMPACT=0"')
    ap.add_argument("--agent", choices=["v1", "v2", "both"], default="both")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--ports", default="8210,8101,8102")
    args = ap.parse_args()
    api_port, v1_port, v2_port = (int(p) for p in args.ports.split(","))
    flags = Flags.from_cli(args.v2_flags)
    cache_dir = tempfile.mkdtemp(prefix="aud-cache-")
    base = {
        "PORTFOLIO_API_URL": f"http://127.0.0.1:{api_port}",
        # Reproducible bench: the flaky ledger always fails, so failed tool calls are reproducible.
        "FAIL_RATE": os.environ.get("FAIL_RATE", "1.0"),
        "REPORT_CACHE_PATH": str(Path(cache_dir) / "reports.sqlite"),
    }
    flag_env = {
        n: ("1" if getattr(flags, n.lower()) else "0")
        for n in ("CACHE", "DIRECT_TOOLS", "COMPACT", "MODEL_PER_ROLE", "STREAMING")
    }
    out = Path(args.out)
    with server("portfolio_api.app:app_from_env", api_port, base):
        with (
            server("server.app:app_from_env", v1_port, {**base, "AGENT_VERSION": "v1"}) as v1,
            server(
                "server.app:app_from_env", v2_port, {**base, "AGENT_VERSION": "v2", **flag_env}
            ) as v2,
        ):
            v2_label = "v2" if flags.label() == "all-on" else f"v2 ({flags.label()})"
            targets = [
                (url, label, name)
                for url, label, name in ((v1, "v1", "v1"), (v2, v2_label, "v2"))
                if args.agent in ("both", name)
            ]
            # One event loop for every target: the OpenAI client keeps a shared async HTTP
            # connection pool, and a second asyncio.run() would find it bound to a closed loop.
            asyncio.run(bench_targets(targets, args, out, flags))


async def bench_targets(targets, args, out: Path, flags) -> None:
    for url, label, name in targets:
        print(f"== benchmarking {label} at {url}", flush=True)
        rep = await run_benchmark(
            url,
            label,
            str(out / f"{name}.json"),
            repeat=args.repeat,
            timeout=args.timeout,
            meta={"flags": flags.label() if name == "v2" else None},
        )
        print(
            f"   {rep['correct_answers']}/{rep['questions_total']} correct, "
            f"p50 {rep['latency_p50_s']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
