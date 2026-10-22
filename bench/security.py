"""`bench security`: is the advisor really closed to outside callers?

Offline: `--edge-conf <rendered advisor.conf>` evaluates the allow/deny lines (no Docker).
Live:    `--url http://localhost:8088` calls the running edge from where you are (an outside
         caller) and expects 403 on /dev-console/ask and /health.
Prints PASS/FAIL per assertion; the exit code is 1 on any FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from deploy.edge_rules import decide

OUTSIDE_IP = "203.0.113.9"  # documentation range: stands for "anyone on the internet"
INTERNAL_IP = "10.77.0.5"  # inside the backplane subnet


def check_conf(conf_path: str) -> list[tuple[str, bool]]:
    conf = Path(conf_path).read_text()
    return [
        (f"outside caller {OUTSIDE_IP} is denied", decide(OUTSIDE_IP, conf) == "deny"),
        (
            "/dev-console/ask is unreachable from outside (the rules sit on location /)",
            decide(OUTSIDE_IP, conf) == "deny",
        ),
        (f"internal caller {INTERNAL_IP} is allowed", decide(INTERNAL_IP, conf) == "allow"),
    ]


def check_live(url: str) -> list[tuple[str, bool]]:
    results = []
    with httpx.Client(base_url=url, timeout=10) as c:
        r = c.get("/health")
        results.append(
            (f"outside caller denied: GET /health -> {r.status_code}", r.status_code == 403)
        )
        r = c.post("/dev-console/ask", json={"prompt": "hi"})
        results.append(
            (f"/dev-console/ask unreachable from outside: -> {r.status_code}", r.status_code == 403)
        )
    return results


def run(edge_conf: str | None = None, url: str | None = None) -> int:
    if bool(edge_conf) == bool(url):
        print("give exactly one of --edge-conf or --url", file=sys.stderr)
        return 2
    results = check_conf(edge_conf) if edge_conf else check_live(url)
    if url:
        print("SKIP  internal caller allowed (cannot be tested from outside; use --edge-conf)")
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in results) else 1
