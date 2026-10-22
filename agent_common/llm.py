"""Model factory: ChatOpenAI against any OpenAI-compatible endpoint, or the offline fake."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from common.config import Settings
from common.metrics import HANDLER


def make_chat_model(model: str, settings: Settings | None = None, **kwargs) -> BaseChatModel:
    s = settings or Settings.from_env()
    if s.fake:
        from agent_common.fake import FakeChatModel

        return FakeChatModel(model_name=model, callbacks=[HANDLER])
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=0,
        callbacks=[HANDLER],
        **kwargs,
    )
