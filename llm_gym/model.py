from functools import cache

from langchain_ollama import ChatOllama

from llm_gym.config import CONTEXT_TOKENS
from llm_gym.models import MODELS
from llm_gym.tools import tools

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


@cache
def get_model(model_id: str) -> ChatOllama:
    """One model's client, built on first use and kept for the process.

    Tools are bound here rather than by the caller: bind_tools returns a new
    object, so a client built outside this function carries none.
    """
    if model_id not in MODELS:
        raise KeyError(f"Unknown model {model_id!r}. Known: {', '.join(MODELS)}")

    return ChatOllama(
        model=model_id,
        base_url=OLLAMA_BASE_URL,
        num_ctx=CONTEXT_TOKENS,
    ).bind_tools(tools)
