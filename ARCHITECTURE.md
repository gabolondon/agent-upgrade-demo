# Architecture

Components: `portfolio_api/` (example data, `/v1` verbose and `/v2` compact), `agent_v1/`,
`agent_v2/`, `server/` (FastAPI, `/ask` and `/briefing`), `client/` (rich CLI), `bench/`.

## Data API

`/v1`: lists of row dicts (field names repeated per row), whole history, rows shuffled on every
request, volatile fields (`retrieved_on`, `trace_id`). `/v2`: `{"header": [...], "table": [[...]]}`,
stable sort, `window_start` / `window_end` (default last 90 days, max 366). Seeded gaps: one client
holds an instrument with no prices; the transactions endpoint of one client fails at `FAIL_RATE`.

## Agent v1

```mermaid
flowchart LR
  Q[question] --> P[planner]
  P --> S1[holdings specialist]
  P --> S2[transactions specialist]
  P --> S3[prices specialist]
  P --> S4[notes specialist]
  S1 & S2 & S3 & S4 -->|/v1 tools, prose briefs| W[writer]
  W --> A[answer, raw note paths]
  O[open chat] -->|all data, every time| R[report LLM call]
```

## Agent v2

```mermaid
flowchart LR
  Q[question] --> T{triage<br/>TRIAGE_MODEL}
  T -->|small talk / off-topic| A
  T -->|portfolio| A[answer agent<br/>ANSWER_MODEL]
  A <-->|typed results: worked + data / reason| D[(/v2 compact, bounded)]
  A -->|working, token, cited, done / failed| SSE[SSE stream]
  O[open chat] --> F[fetch /v2 data] --> K[fingerprint_key:<br/>stable form of the data,<br/>stamped with the generator]
  K -->|hit| C[(sqlite cache)]
  K -->|miss, single-flight lock| G[report LLM call] -->|only after done| C
```

## Fingerprint

`stable_form()` sorts keys and rows, rounds floats, and drops `VOLATILE_KEYS`
(`retrieved_on`, `trace_id`). `fingerprint_key()` is a blake2b fingerprint of that stable form,
stamped with what generated the report. `scripts/fingerprint_stability.py` shows 25 raw payload
hashes collapsing to 1.
