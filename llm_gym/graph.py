from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from ollama import ResponseError

from llm_gym.logconf import log
from llm_gym.model import get_model
from llm_gym.provenance import stamp
from llm_gym.tools import tools


async def call_model(state, config):
    # A second parameter is what makes LangGraph pass the caller's config, so
    # which model answers belongs to the call, not to the compiled graph.
    model_id = config["configurable"]["model"]
    model = get_model(model_id)

    # Off the client that answered, not from config: get_model is cached.
    settings = {
        "num_ctx": model.num_ctx,
        "reasoning": model.reasoning,
        "tools": [tool["function"]["name"] for tool in model.kwargs["tools"]],
        **stamp(),
    }

    # Streamed rather than invoked, so the tokens reach the browser while the
    # model is still writing. Summing the chunks rebuilds a single message:
    # AIMessageChunk.__add__ merges the content and stitches back together
    # tool calls, whose name and arguments arrive split across chunks.
    response = None
    try:
        async for chunk in model.astream(_filter_agent_input(state["messages"])):
            response = chunk if response is None else response + chunk
    except ResponseError as error:
        # Ollama refusing the request is a fact about the conversation, so it
        # is returned as state. A notice would not survive a reload.
        log.error("%s -> refused: %s", model_id, error)
        refusal = AIMessage(content="", response_metadata={"model": model_id, "error": str(error), **settings})
        return {"messages": [refusal]}

    response.response_metadata.update(settings)

    name = response.response_metadata["model"]

    if response.tool_calls:
        for call in response.tool_calls:
            log.info("%s -> tool call: %s(%s)", name, call["name"], call["args"])
    if response.content:
        log.info("%s -> text: %r", name, response.content)
    if response.invalid_tool_calls:
        log.warning("%s -> INVALID tool calls: %s", name, response.invalid_tool_calls)

    return {"messages": [response]}


def _filter_agent_input(messages):
    """Everything the model may be shown. A refusal is kept as a message so the
    transcript survives a reload, but replaying it would feed back its own error."""
    return [message for message in messages if is_agent_input(message)]


def is_agent_input(message) -> bool:
    # Reasoning needs no check here: ollama never takes it back in, so it goes away
    # whether or not the message carrying it is sent.
    return "error" not in message.response_metadata


def build_graph(checkpointer: BaseCheckpointSaver):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
