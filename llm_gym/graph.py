from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm_gym.logconf import log
from llm_gym.model import get_model
from llm_gym.tools import tools


async def call_model(state, config):
    # A second parameter is what makes LangGraph pass the caller's config, so
    # which model answers belongs to the call, not to the compiled graph.
    model = get_model(config["configurable"]["model"])

    # Streamed rather than invoked, so the tokens reach the browser while the
    # model is still writing. Summing the chunks rebuilds a single message:
    # AIMessageChunk.__add__ merges the content and stitches back together
    # tool calls, whose name and arguments arrive split across chunks.
    response = None
    async for chunk in model.astream(state["messages"]):
        response = chunk if response is None else response + chunk

    # Strict here, tolerant in history.render_messages: this is the last point
    # at which a reply missing its own name is still fixable.
    name = response.response_metadata["model_name"]

    if response.tool_calls:
        for call in response.tool_calls:
            log.info("%s -> tool call: %s(%s)", name, call["name"], call["args"])
    if response.content:
        log.info("%s -> text: %r", name, response.content)
    if response.invalid_tool_calls:
        log.warning("%s -> INVALID tool calls: %s", name, response.invalid_tool_calls)

    return {"messages": [response]}


def build_graph(checkpointer: BaseCheckpointSaver):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    return builder.compile(checkpointer=checkpointer)
