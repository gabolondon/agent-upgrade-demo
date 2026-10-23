"""Seeded example portfolio data. Nothing here comes from a real system."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from common.config import demo_today

SEED = 42
YEARS = 3

INSTRUMENTS = {
    "KELP": ("Kelp Harvest Foods", 48.0, 0.016),
    "NORD": ("Nordlys Grid Partners", 112.0, 0.011),
    "SOLR": ("Solrise Panels", 27.5, 0.024),
    "TIDE": ("Tidewater Logistics", 63.0, 0.014),
    "BRIX": ("Brixton Modular Homes", 85.0, 0.013),
    "OAKB": ("Oakbridge Utilities", 39.0, 0.007),
    "FERN": ("Fernhill Biotech", 154.0, 0.021),
    "LUMA": ("Lumabyte Software", 201.0, 0.018),
}
MISSING_TICKER = "TIDE"  # no prices at all
FLAKY_ACCOUNT = "c6"  # its transactions endpoint fails at FAIL_RATE

ACCOUNTS = {
    "c1": "Marisol Quade",
    "c2": "Tobias Renner",
    "c3": "Anika Valdis",
    "c4": "Joss Pemberton",
    "c5": "Ilse Marchetti",
    "c6": "Dario Okonkwo",
}
NICKNAMES = {
    "c1": "Maple",
    "c2": "Heron",
    "c3": "Juniper",
    "c4": "Cobalt",
    "c5": "Saffron",
    "c6": "Willow",
}
HOLDINGS = {
    "c1": {"KELP": 120, "NORD": 40, "BRIX": 25, "OAKB": 300},
    "c2": {"FERN": 15, "LUMA": 30, "KELP": 200},
    "c3": {"OAKB": 500, "LUMA": 12, "NORD": 60},
    "c4": {"SOLR": 800, "KELP": 90, "NORD": 20},
    "c5": {"TIDE": 150, "BRIX": 35, "FERN": 10},
    "c6": {"BRIX": 50, "NORD": 75, "FERN": 22},
}

NOTES = [
    (
        "rate-outlook",
        "Rate Outlook: A Slower Glide",
        "Central banks signal a slower glide in rates; "
        "utilities and long-duration growth names are the most rate sensitive. Risk: a pause in cuts.",
    ),
    (
        "kelp-supply-chain",
        "KELP Supply Chain Review",
        "Kelp Harvest Foods reduced cold-storage "
        "costs this year. Margins improved, but harvest volatility remains a risk for KELP.",
    ),
    (
        "grid-capex",
        "Grid Capex Cycle",
        "Nordlys Grid Partners benefits from a multi-year grid "
        "capex cycle. NORD earnings depend on permitting speed. Rate risk is moderate.",
    ),
    (
        "solar-pricing",
        "Solar Panel Pricing Pressure",
        "Panel prices fell again. Solrise Panels "
        "faces margin pressure; SOLR is volatile and sensitive to subsidy news.",
    ),
    (
        "logistics-slowdown",
        "Logistics Slowdown Watch",
        "Freight volumes are soft. Tidewater "
        "Logistics reports are irregular; TIDE disclosures lag peers.",
    ),
    (
        "housing-modular",
        "Modular Housing Demand",
        "Demand for modular homes is steady. Brixton "
        "Modular Homes shows rising backlog; BRIX is exposed to mortgage rates.",
    ),
    (
        "utility-yield",
        "Utility Yield Primer",
        "Oakbridge Utilities pays a stable yield. OAKB is "
        "defensive; concentration risk applies if it exceeds a third of a portfolio.",
    ),
    (
        "biotech-trials",
        "Biotech Trial Calendar",
        "Fernhill Biotech has two trial readouts pending. "
        "FERN is high variance; outcomes are binary. Market risk is elevated.",
    ),
    (
        "software-multiples",
        "Software Multiples Check",
        "Lumabyte Software trades at a premium. "
        "LUMA is sensitive to rates; earnings growth must justify the multiple.",
    ),
    (
        "diversification",
        "Diversification Basics",
        "Spreading exposure across sectors reduces "
        "single-name risk. Review concentration at least quarterly. General market risk notes.",
    ),
    (
        "fx-and-costs",
        "Costs and Currency Notes",
        "Trading costs and currency moves can erode returns. Check fees before rebalancing.",
    ),
    (
        "data-quality",
        "Reading Price Data Carefully",
        "Prices can be missing or lag behind. Always check "
        "the date of the latest price and say which date it is.",
    ),
]


@dataclass
class World:
    ref: date
    prices: dict[str, list[tuple[str, float, int]]] = field(default_factory=dict)
    transactions: dict[str, list[tuple[str, str, str, int, float]]] = field(default_factory=dict)


def build_world(ref: date | None = None) -> World:
    ref = ref or demo_today()
    rng = random.Random(SEED)
    w = World(ref=ref)
    start = ref - timedelta(days=365 * YEARS - 1)
    for ticker, (_, p0, vol) in INSTRUMENTS.items():
        if ticker == MISSING_TICKER:
            w.prices[ticker] = []
            continue
        rows, price, d = [], p0, start
        while d <= ref:
            price = max(1.0, price * (1 + rng.gauss(0.0003, vol)))
            rows.append((d.isoformat(), round(price, 4), rng.randint(2_000, 90_000)))
            d += timedelta(days=1)
        w.prices[ticker] = rows
    for cid, held in HOLDINGS.items():
        tx = []
        for _ in range(42):
            ticker = rng.choice(sorted(held))
            day = ref - timedelta(days=rng.randint(0, 365 * YEARS - 1))
            if len(tx) < 6:  # guarantee recent activity
                day = ref - timedelta(days=rng.randint(0, 28))
            ref_price = INSTRUMENTS[ticker][1]
            tx.append(
                (
                    day.isoformat(),
                    ticker,
                    rng.choice(["buy", "sell"]),
                    rng.randint(1, 40),
                    round(ref_price * rng.uniform(0.8, 1.3), 2),
                )
            )
        w.transactions[cid] = sorted(tx)
    return w
