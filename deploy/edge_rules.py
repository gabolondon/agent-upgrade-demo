"""Offline evaluator for the allow/deny lines of a rendered nginx conf.

nginx walks allow/deny directives in order and stops at the first one that matches the client
address. If nothing matches, access is allowed. That is the whole trick of the demo: an
`allow all;` ahead of `deny all;` makes everything after it dead code.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path

_RULE = re.compile(r"^\s*(allow|deny)\s+([^\s;]+)\s*;", re.MULTILINE)


def parse_rules(conf: str) -> list[tuple[str, str]]:
    return [(verb, target) for verb, target in _RULE.findall(conf)]


def decide(ip: str, conf: str | Path) -> str:
    """First-match evaluation. `conf` is the conf text or a path to it."""
    text = Path(conf).read_text() if isinstance(conf, Path) else str(conf)
    addr = ipaddress.ip_address(ip)
    for verb, target in parse_rules(text):
        if target == "all" or addr in ipaddress.ip_network(target, strict=False):
            return "allow" if verb == "allow" else "deny"
    return "allow"
