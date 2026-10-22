.PHONY: install api serve-v1 serve-v2 chat bench compare test lint rejudge demo-open demo-fixed demo-down security

API_PORT ?= 8210
V1_PORT ?= 8101
V2_PORT ?= 8102
RESULTS ?= results

install:
	uv sync

api:
	uv run uvicorn portfolio_api.app:app_from_env --factory --port $(API_PORT)

serve-v1:
	AGENT_VERSION=v1 uv run uvicorn server.app:app_from_env --factory --port $(V1_PORT)

serve-v2:
	AGENT_VERSION=v2 uv run uvicorn server.app:app_from_env --factory --port $(V2_PORT)

# Needs `make api` and a serve-* target running (or set LLM_FAKE=1 in .env).
chat:
	uv run python -m client.chat --url http://localhost:$(if $(filter v1,$(AGENT)),$(V1_PORT),$(V2_PORT)) --account c1

# Starts the API and both servers itself, runs the suite against v1 and v2, tears everything down.
# Override v2 flags with e.g. FLAGS="CACHE=0,COMPACT=0".
bench:
	uv run python -m bench.orchestrate --out $(RESULTS) $(if $(AGENT),--agent $(AGENT)) $(if $(FLAGS),--v2-flags "$(FLAGS)") $(BENCH_ARGS)

# Re-run only the judge on stored answers: make rejudge IN=results/v2.json OUT=results/v2-rejudged.json
rejudge:
	uv run python -m bench.cli rejudge $(IN) $(OUT) $(if $(SAMPLES),--samples $(SAMPLES))

compare:
	uv run python -m bench.cli compare $(RESULTS)/v1.json $(RESULTS)/v2.json

test:
	LLM_FAKE=1 uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

# --- The open door (deploy/) ---
COMPOSE = docker compose -f deploy/docker-compose.yml
EDGE_URL ?= http://localhost:8088

# Readiness: wait until the edge answers with anything but "refused" (000) or a bad gateway
# (502/503/504: nginx is up, the advisor is not yet). 200 = open edge, 403 = closed edge.
define wait_edge
	@for i in $$(seq 60); do \
	  c=$$(curl -s -o /dev/null -w '%{http_code}' $(EDGE_URL)/health); \
	  case $$c in 000|502|503|504) sleep 1;; *) echo "edge ready (HTTP $$c)"; break;; esac; \
	done
endef

define poke_edge
	$(wait_edge)
	@echo "--- from the host (outside caller)"
	@for i in 1 2 3; do curl -s -o /dev/null -w "POST /dev-console/ask -> HTTP %{http_code}\n" \
	  -X POST $(EDGE_URL)/dev-console/ask -H 'content-type: application/json' -d '{"prompt":"hello"}'; done
	@curl -s -w ' -> HTTP %{http_code}\n' $(EDGE_URL)/dev-console/model-calls | tail -c 120
	@echo "--- from web-gateway (internal caller) -> edge:80"
	@$(COMPOSE) exec -T web-gateway python scripts/internal_probe.py
endef

demo-open:
	$(COMPOSE) up -d --build
	$(poke_edge)

demo-fixed:
	$(COMPOSE) down
	$(COMPOSE) -f deploy/docker-compose.fixed.yml up -d --build
	$(poke_edge)

demo-down:
	$(COMPOSE) -f deploy/docker-compose.fixed.yml down

# Offline: render the conf with the compose env (EDGE_ACCESS=internal only), then check it.
# SCRIPT=start-edge.fixed.sh checks the fixed state. FAIL on the broken one.
security:
	EDGE_ACCESS=internal sh deploy/edge/$(or $(SCRIPT),start-edge.sh) results/advisor.conf
	uv run python -m bench.cli security --edge-conf results/advisor.conf
