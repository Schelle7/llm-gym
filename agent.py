import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime, timezone
from uuid import uuid4

from config import SYSTEM_PROMPT
from graph import build_graph


def run_agent(graph, user_prompt, thread_id):
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        },
        {"configurable": {"thread_id": thread_id}},
    )
    print("result-message content:", result["messages"][-1].content)


def main():
    with SqliteSaver.from_conn_string("checkpoints/checkpoints.db") as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S") + f"_{uuid4()}"
        print(f"LangSmith tracing: {os.getenv('LANGSMITH_TRACING', '<unset>')}")
        print(f"LangSmith project: {os.getenv('LANGSMITH_PROJECT', '<unset>')}")
        print(f"LangSmith API key set: {'yes' if os.getenv('LANGSMITH_API_KEY') else 'no'}")
        print(f"Started thread: {thread_id}")
        while True:
            user_prompt = input("Write prompt: ")
            if user_prompt == "exit":
                break
            run_agent(graph, user_prompt, thread_id)


if __name__ == "__main__":
    main()
