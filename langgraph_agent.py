from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import SYSTEM_PROMPT
from langgraph_model import model
from langgraph_tools import langgraph_tools


def call_model(state):
    response = model.invoke(state["messages"])
    print(f"Model response: {response}")
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(langgraph_tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()


def run_agent(user_prompt):
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        }
    )
    print("result-message content:", result["messages"][-1].content)


def main():
    while True:
        user_prompt = input("Write prompt: ")
        run_agent(user_prompt)


if __name__ == "__main__":
    main()