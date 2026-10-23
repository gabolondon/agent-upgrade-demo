# agent-upgrade-demo

Run a slow LLM agent and an upgraded one side by side, then measure each improvement with a
benchmark and an LLM judge. The example domain is a personal-finance portfolio app: the assistant
answers a client's questions and opens every chat with a short report.

> This is example code: it shows the pattern I used, not the production implementation.
> Names, data, rules and prices are example values.

- **v1, the slow one**: planner -> pool of specialist sub-agents (fetch via verbose `/v1`, write prose
  briefs) -> writer. Opening report regenerated on every open. No streaming.
- **v2, the upgraded one**: triage on a light model -> ONE answer agent calling typed data tools
  directly over compact, bounded `/v2`; content-fingerprinted report cache; SSE streaming.
  Every improvement is a flag, so you can ablate one at a time.

## Quickstart (fake mode, no key, offline)

```bash
make install
make test && make lint
LLM_FAKE=1 make bench      # starts the API and both servers, runs the suite on v1 and v2
make compare               # delta table between results/v1.json and results/v2.json
uv run python scripts/fingerprint_stability.py   # raw: 25 distinct hashes -> stable_form: 1
```

`LLM_FAKE=1` swaps in a deterministic rule-based chat model with simulated latency proportional
to tokens, streaming and tool calls. It is a stand-in: it behaves differently for v1 and v2 only
because of the prompts and data each receives (typed tool results, the account nickname), so the
numbers show the *mechanics* of the benchmark, not real model quality.

Chat by hand (three terminals): `LLM_FAKE=1 make api`, `LLM_FAKE=1 make serve-v2`, then `make chat`
(`make chat AGENT=v1` after `make serve-v1`; `make bench AGENT=v1` benchmarks one agent only). Add `--pace 1.5` to the client to pause before a
cached report is revealed.

## Real models

Copy `.env.example` to `.env` and pick a preset. Variables: `LLM_BASE_URL`, `LLM_API_KEY`,
`TRIAGE_MODEL` (light, used by v2 triage), `ANSWER_MODEL`, `JUDGE_MODEL`.

| Preset | `LLM_BASE_URL` | key |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | your OpenRouter key |
| Local Ollama | `http://localhost:11434/v1` | any non-empty string |

Then run the same commands without `LLM_FAKE=1`. The tests always force fake mode.

## Measure each layer

v2 flags (env or `FLAGS=...`): `CACHE`, `DIRECT_TOOLS`, `COMPACT`, `MODEL_PER_ROLE`, `STREAMING`,
all on by default. `make bench` always runs v1 first, then v2 with the flags you give:

```bash
make bench FLAGS="CACHE=0"                 # reopen is slow again
make bench FLAGS="COMPACT=0"               # big payloads come back
make bench FLAGS="DIRECT_TOOLS=0"          # v1-style specialists behind the v2 server
make bench FLAGS="MODEL_PER_ROLE=0"        # triage on the answer model
make bench FLAGS="STREAMING=0"             # first event = whole answer
make bench BENCH_ARGS="--repeat 3"
make compare
```

The benchmark (`bench/`): 5 conversations / 20 questions in `bench/scenarios.yaml`, each with 1-4
short rules written in everyday language. The judge returns a per-rule pass/fail verdict, one 1-5
grade and a reason, sampled 3 times (majority vote per rule, median grade). The report lists
correct answers (all rules pass), fully passing conversations, mean grade (0 counts a missing
answer), failed rules, answers with titled sources, latency p50/p95/max, first-event time, report
open/reopen time, timeouts and errors. `make bench` sets `FAIL_RATE=1.0` so the flaky ledger
endpoint fails reproducibly.

## Wire contract (v2 streaming)

`POST /ask` and `POST /briefing` stream Server-Sent Events: any number of `working` and `token`
events, one `cited`, then either `done` or `failed`.

| event | data |
|---|---|
| `working` | `{"label": "Fetching prices"}` |
| `token` | `{"text": "..."}` |
| `cited` | `{"items": [{"title", "href"}]}`, sent once, before the close |
| `done` | `{"answer": "..."}` (`/briefing` adds `"cached"`) |
| `failed` | `{"message": "..."}` |

A turn is complete only with `done`, and only a completed briefing is ever cached. With
`STREAMING=0` the same content comes back as one JSON body (`answer`, `cited`; a failure is a 502
with `message`). Tool results are `{"worked": true, "data": ...}` or `{"worked": false, "reason": "..."}`;
empty data is a call that worked.

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md). License: MIT.

## The open door

The compose file says the advisor is internal-only (`EDGE_ACCESS: internal`, nothing else). The
edge's start script silently opens it anyway: a legacy "temporary: open for early testing" block,
on by default, writes `allow all;` before the restrictive rule, and nginx takes the first match, so
the internal-only rule is dead code. Nothing logs it. Behind the edge, a dev console calls the
LLM with no token. Everything here is example code and runs in fake LLM mode; it shows the pattern, not how any
real system was configured.

```text
$ docker compose -f deploy/docker-compose.yml config | grep EDGE_ACCESS            # closed
      EDGE_ACCESS: internal
$ docker compose -f deploy/docker-compose.yml exec edge grep -e allow -e deny <advisor.conf>  # open
    allow all;
    allow 10.77.0.0/24;
    deny all;
```

```bash
make demo-open    # broken: three curls from the host to :8088/dev-console/ask succeed, model-calls rises
make demo-fixed   # fixed script (legacy block removed): the same curls return 403
make demo-down
make security                              # offline: renders the conf, checks it (FAIL)
make security SCRIPT=start-edge.fixed.sh   # PASS
```

Layout: `deploy/docker-compose.yml` (broken) + `deploy/docker-compose.fixed.yml`, `deploy/edge/`
(`start-edge.sh` with the legacy block, `start-edge.fixed.sh` without it; the fix is in the script, not a compose variable), `deploy/edge_rules.py`
(first-match allow/deny evaluator used by tests and `bench security`), `gateway/` (token-checked
front door on :8080), `server/dev_console.py` (`ADVISOR_DEBUG=1`).

The billing point: an unauthenticated endpoint that reaches the LLM is a bill anyone can run up.
`/dev-console/model-calls` (an in-process counter) reports `est_cost_usd` using an illustrative `COST_PER_CALL_USD=0.002`; it is not
a real price.
