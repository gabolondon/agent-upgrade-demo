"""Dev console, enabled by ADVISOR_DEBUG=1.

A leftover from early testing: it calls the answer model directly, with no session and no
gateway token. Never enable it on anything reachable from outside.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent_common.llm import make_chat_model
from common.config import Settings
from common.metrics import COUNTER

PAGE = """<!doctype html><title>advisor debug</title>
<h1>advisor dev console</h1>
<form onsubmit="go(event)"><input id=p size=60 placeholder="prompt"><button>ask</button></form>
<pre id=o></pre>
<script>async function go(e){e.preventDefault();
const r=await fetch('/dev-console/ask',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({prompt:document.getElementById('p').value})});
document.getElementById('o').textContent=await r.text();}</script>
"""


class AskIn(BaseModel):
    prompt: str


def debug_router(settings: Settings | None = None) -> APIRouter:
    router = APIRouter(prefix="/dev-console")

    @router.get("", response_class=HTMLResponse)
    async def page():
        return PAGE

    @router.post("/ask")
    async def ask(body: AskIn):
        s = settings or Settings.from_env()
        model = make_chat_model(s.answer_model, s)
        msg = await model.ainvoke([HumanMessage(content=body.prompt)])
        return {"answer": msg.content}

    @router.get("/model-calls")
    async def llm_calls():
        # est_cost_usd is illustrative (COST_PER_CALL_USD), not a real price.
        return COUNTER.snapshot()

    return router
