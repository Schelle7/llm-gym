from langchain_openai import ChatOpenAI

from config import MODEL
from langgraph_tools import langgraph_tools


model = ChatOpenAI(
    model=MODEL,
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
).bind_tools(langgraph_tools)
