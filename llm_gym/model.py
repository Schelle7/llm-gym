from functools import lru_cache

from langchain_openai import ChatOpenAI

from llm_gym.models import MODELS
from llm_gym.tools import tools

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"


@lru_cache(maxsize=None)
def get_model(model_id: str) -> ChatOpenAI:
    """One model's client, built on first use and kept for the process.

    Tools are bound here rather than by the caller: bind_tools returns a new
    object, so a client built outside this function carries none.
    """
    if model_id not in MODELS:
        raise KeyError(f"Unknown model {model_id!r}. Known: {', '.join(MODELS)}")

    return ChatOpenAI(
        model=model_id,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        # A streamed response carries no token counts unless this is asked for,
        # and the node streams. Without it every checkpoint records usage of None.
        stream_usage=True,
    ).bind_tools(tools)