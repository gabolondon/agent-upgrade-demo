"""In-process LLM call counter, fed by a LangChain callback attached in make_chat_model."""

from __future__ import annotations

import os
import threading

from langchain_core.callbacks import BaseCallbackHandler

# Illustrative only: not a real price. It exists so the demo can show a bill rising.
COST_PER_CALL_USD = float(os.environ.get("COST_PER_CALL_USD", "0.002"))


class CallCounter:
    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()

    def incr(self) -> None:
        with self._lock:
            self._n += 1

    @property
    def calls(self) -> int:
        return self._n

    def reset(self) -> None:
        with self._lock:
            self._n = 0

    def snapshot(self) -> dict:
        calls = self.calls
        # est_cost_usd is ILLUSTRATIVE: calls * COST_PER_CALL_USD (default 0.002).
        return {"calls": calls, "est_cost_usd": round(calls * COST_PER_CALL_USD, 6)}


COUNTER = CallCounter()


class CountingHandler(BaseCallbackHandler):
    """Counts one per chat-model (or plain LLM) invocation."""

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        COUNTER.incr()

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        COUNTER.incr()


HANDLER = CountingHandler()
