"""Agent v1, "the slow one".

planner (splits the question into sub-questions, one per kind of data) -> a pool of specialist
ReAct sub-agents (holdings, transactions, prices, notes) that each fetch through the /v1 tools and
write a short prose brief -> writer composes the answer from the briefs.
The opening report is regenerated from all the data on every chat open. No streaming.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agent_common import prompts
from agent_common.llm import make_chat_model
from agent_common.report import gather_report_data
from agent_common.tables import dumps, rows_of
from agent_common.tools import ApiClient, make_tools
from common.config import Settings, demo_today

SPECIALIST_TOOLS = {
    "holdings": ["get_holdings"],
    "transactions": ["get_transactions"],
    "prices": ["get_prices", "get_holdings"],
    "notes": ["search_notes"],
}


class State(TypedDict, total=False):
    account_id: str
    question: str
    plan: list[dict]
    briefs: list[dict]
    notes: list[dict]
    answer: str


def parse_plan(text: str) -> list[dict]:
    """Planner reply -> list of {"kind", "question"}; unknown kinds dropped, duplicates merged."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    seen, out = set(), []
    for it in items:
        if isinstance(it, dict) and it.get("kind") in SPECIALIST_TOOLS and it["kind"] not in seen:
            seen.add(it["kind"])
            out.append({"kind": it["kind"], "question": str(it.get("question", ""))})
    return out


def build_graph(model, tools: dict, today: str):
    async def planner(state: State) -> State:
        msg = await model.ainvoke(
            [SystemMessage(prompts.PLANNER_V1), HumanMessage(state["question"])]
        )
        return {"plan": parse_plan(str(msg.content))}

    async def run_specialist(kind: str, question: str, account_id: str) -> tuple[dict, list[dict]]:
        agent = create_react_agent(
            model,
            [tools[n] for n in SPECIALIST_TOOLS[kind]],
            prompt=prompts.SPECIALIST.format(kind=kind, account_id=account_id, today=today),
        )
        out = await agent.ainvoke({"messages": [HumanMessage(question)]})
        notes: list[dict] = []
        for m in out["messages"]:
            if isinstance(m, ToolMessage) and m.name == "search_notes":
                try:
                    notes += [
                        {"slug": r["slug"], "title": r["title"]}
                        for r in rows_of(json.loads(m.content))
                    ]
                except (json.JSONDecodeError, KeyError):
                    pass
        return {"kind": kind, "text": str(out["messages"][-1].content)}, notes

    async def specialists(state: State) -> State:
        done = await asyncio.gather(
            *(run_specialist(p["kind"], p["question"], state["account_id"]) for p in state["plan"])
        )
        return {"briefs": [b for b, _ in done], "notes": [n for _, ns in done for n in ns]}

    async def writer(state: State) -> State:
        sys = prompts.WRITER_V1.format(account_id=state["account_id"], today=today)
        briefs = "\n".join(f"- {b['kind']}: {b['text']}" for b in state.get("briefs", []))
        human = f"Question: {state['question']}\n\nBriefs:\n{briefs}"
        msg = await model.ainvoke([SystemMessage(sys), HumanMessage(human)])
        return {"answer": str(msg.content)}

    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("specialists", specialists)
    g.add_node("writer", writer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "specialists")
    g.add_edge("specialists", "writer")
    g.add_edge("writer", END)
    return g.compile()


class AgentV1:
    """The v1 agent. `compact`/`typed` exist only so v2 can reuse this graph when DIRECT_TOOLS=0."""

    def __init__(
        self,
        settings: Settings | None = None,
        api: ApiClient | None = None,
        *,
        compact: bool = False,
        typed: bool = False,
        answer_model=None,
    ):
        self.settings = settings or Settings.from_env()
        self.api = api or ApiClient(self.settings.api_url)
        self.compact = compact
        self.answer_model = answer_model or make_chat_model(
            self.settings.answer_model, self.settings
        )
        self.tools = make_tools(self.api, compact=compact, typed=typed)
        self.graph = build_graph(self.answer_model, self.tools, demo_today().isoformat())

    async def run(self, account_id: str, message: str, history: list[dict] | None = None) -> State:
        # v1 is stateless per question: history is accepted but not used.
        return await self.graph.ainvoke({"account_id": account_id, "question": message})

    async def chat(self, account_id: str, message: str, history: list[dict] | None = None) -> dict:
        st = await self.run(account_id, message, history)
        return {
            "answer": st["answer"],
            "cited": [f"notes/{n['slug']}.md" for n in st.get("notes", [])],
        }

    async def open_report(self, account_id: str) -> dict:
        data = await gather_report_data(self.api, account_id, self.compact)
        msg = await self.answer_model.ainvoke(
            [SystemMessage(prompts.REPORT_V1), HumanMessage("DATA:\n" + dumps(data))]
        )
        return {"report": str(msg.content), "cached": False}
