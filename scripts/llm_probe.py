"""One-shot LLM connectivity check: lists models, then sends one chat turn through the
project's own model factory. Reads LLM_BASE_URL / LLM_API_KEY / ANSWER_MODEL from the env.

    uv run python scripts/llm_probe.py
"""

import json
import sys
import urllib.error
import urllib.request

from agent_common.llm import make_chat_model
from common.config import Settings, load_dotenv

load_dotenv()
s = Settings.from_env()
print(f"base_url : {s.llm_base_url}")
print(
    f"api_key  : {s.llm_api_key[:6]}…{s.llm_api_key[-4:]}"
    if len(s.llm_api_key) > 12
    else "api_key  : (short/none)"
)
print(f"model    : {s.answer_model}")

req = urllib.request.Request(
    s.llm_base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {s.llm_api_key}"}
)
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    ids = [m.get("id") for m in data.get("data", [])]
    print(f"GET /models -> {len(ids)} models: {', '.join(ids[:20])}")
except urllib.error.HTTPError as e:
    print(f"GET /models -> HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
    sys.exit(1)

resp = make_chat_model(s.answer_model, s).invoke("Reply with exactly: pong")
print("chat     ->", resp.content)
