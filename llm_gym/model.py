from langchain_openai import ChatOpenAI

from llm_gym.config import MODEL
from llm_gym.tools import tools


model = ChatOpenAI(
    model=MODEL,
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
).bind_tools(tools)