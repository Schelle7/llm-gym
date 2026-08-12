from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from logconf import log
from model import model
from tools import tools


def call_model(state):
    response = model.invoke(state["messages"])

    if response.tool_calls:
        for call in response.tool_calls:
            log.info("model -> tool call: %s(%s)", call["name"], call["args"])
    if response.content:
        log.info("model -> text: %r", response.content)
    if response.invalid_tool_calls:
        log.warning("model -> INVALID tool calls: %s", response.invalid_tool_calls)

    return {"messages": [response]}


def build_graph(checkpointer: BaseCheckpointSaver):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)