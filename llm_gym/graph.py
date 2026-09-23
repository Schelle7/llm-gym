import json
from dataclasses import asdict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from ollama import ResponseError

from llm_gym.logconf import log
from llm_gym.model import get_model
from llm_gym.provenance import stamp
from llm_gym.snapshots import snapshots
from llm_gym.tools import tools
from llm_gym.workspace_changes import WorkspaceChangedError, require_unchanged


async def call_model(state, config):
    # A second parameter is what makes LangGraph pass the caller's config, so
    # which model answers belongs to the call, not to the compiled graph.
    model_id = config["configurable"]["model"]
    thread_id = config["configurable"]["thread_id"]
    previous = state["messages"][-1].response_metadata["workspace_snapshot_id"]
    contents = snapshots.workspace.capture()
    snapshot_id = snapshots.save(thread_id, contents)
    try:
        require_unchanged(snapshots.load(thread_id, previous), contents, "before model call; operation cancelled")
    except WorkspaceChangedError as error:
        return {
            "messages": [
                AIMessage(
                    content=str(error),
                    response_metadata={
                        "error": str(error),
                        "workspace_snapshot_id": snapshot_id,
                        "workspace_before_snapshot_id": previous,
                        "workspace_changes": asdict(error.changes),
                    },
                )
            ]
        }
    model = get_model(model_id)

    # Off the client that answered, not from config: get_model is cached.
    settings = {
        "num_ctx": model.num_ctx,
        "reasoning": model.reasoning,
        "tools": [tool["function"]["name"] for tool in model.kwargs["tools"]],
        **stamp(),
        "workspace_snapshot_id": snapshot_id,
    }

    # Streamed rather than invoked, so the tokens reach the browser while the
    # model is still writing. Summing the chunks rebuilds a single message:
    # AIMessageChunk.__add__ merges the content and stitches back together
    # tool calls, whose name and arguments arrive split across chunks.
    response = None
    try:
        async for chunk in model.astream(state["messages"]):
            response = chunk if response is None else response + chunk
    except ResponseError as error:
        # Ollama refusing the request is a fact about the conversation, so it
        # is returned as state. A notice would not survive a reload.
        log.error("%s -> refused: %s", model_id, error)
        refusal = AIMessage(
            content="Agent call failed",
            response_metadata={"model": model_id, "error": str(error), **settings},
        )
        return {"messages": [refusal]}

    response.response_metadata.update(settings)
    after = snapshots.workspace.capture()
    try:
        require_unchanged(contents, after, "during model call; response invalidated")
    except WorkspaceChangedError as error:
        return {
            "messages": [
                AIMessage(
                    content=str(error),
                    response_metadata={
                        **settings,
                        "model": model_id,
                        "error": str(error),
                        "workspace_snapshot_id": snapshots.save(thread_id, after),
                        "workspace_before_snapshot_id": snapshot_id,
                        "workspace_changes": asdict(error.changes),
                        "discarded_response": response.model_dump(mode="json"),
                    },
                )
            ]
        }

    name = response.response_metadata["model"]

    if response.tool_calls:
        for call in response.tool_calls:
            log.info("%s -> tool call: %s(%s)", name, call["name"], call["args"])
    if response.content:
        log.info("%s -> text: %r", name, response.content)
    if response.invalid_tool_calls:
        log.warning("%s -> INVALID tool calls: %s", name, response.invalid_tool_calls)

    return {"messages": [response]}


tool_node = ToolNode(tools)


async def call_tools(state, config):
    thread_id = config["configurable"]["thread_id"]
    request = state["messages"][-1]
    snapshot_id = request.response_metadata["workspace_snapshot_id"]
    before = snapshots.workspace.capture()
    before_id = snapshots.save(thread_id, before)
    approval_tools = {"create_file", "propose_edit", "run_python"}
    try:
        require_unchanged(snapshots.load(thread_id, snapshot_id), before, "since the tool request; operation cancelled")
    except WorkspaceChangedError as error:
        return {
            "messages": [
                ToolMessage(
                    content=json.dumps({"status": "failed", "detail": str(error)}),
                    tool_call_id=call["id"],
                    name=call["name"],
                    response_metadata={
                        "workspace_snapshot_id": before_id,
                        "workspace_before_snapshot_id": snapshot_id,
                        "workspace_changed": True,
                        "workspace_changes": asdict(error.changes),
                    },
                )
                for call in request.tool_calls
            ]
        }

    if len(request.tool_calls) > 1:
        detail = (
            "Agent tried multiple tool calls at once. That is not supported. Agent must make one tool call at a time."
        )
        return {
            "messages": [
                ToolMessage(
                    content=json.dumps({"status": "failed", "detail": detail}),
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                    response_metadata={
                        "workspace_snapshot_id": before_id,
                        "workspace_before_snapshot_id": snapshot_id,
                        "workspace_changed": False,
                    },
                )
                for call in request.tool_calls
            ]
        }

    update = await tool_node.ainvoke(state, config)
    after = snapshots.workspace.capture()
    after_id = snapshots.save(thread_id, after)
    expected = dict(before)
    changed = False
    for message, call in zip(update["messages"], request.tool_calls, strict=True):
        if call["name"] in approval_tools and message.status != "error":
            result = json.loads(message.content)
            if "workspace_changed" in result:
                changed = changed or result["workspace_changed"]
            if result["status"] == "accepted":
                argument = "content" if call["name"] == "create_file" else "modified"
                expected[result["path"]] = call["args"][argument].encode("utf-8")
    try:
        require_unchanged(expected, after, "during tool execution; result invalidated")
    except WorkspaceChangedError as error:
        changed = True
        for message in update["messages"]:
            message.response_metadata["workspace_changes"] = asdict(error.changes)
            message.content = json.dumps(
                {
                    "status": "failed",
                    "detail": str(error),
                    "operation_result": message.content,
                }
            )
    for message in update["messages"]:
        message.response_metadata.update(
            {
                "workspace_snapshot_id": after_id,
                "workspace_before_snapshot_id": before_id,
                "workspace_changed": changed,
            }
        )
    return update


def after_tools(state):
    if state["messages"][-1].response_metadata["workspace_changed"]:
        return END
    return "agent"


def build_graph(checkpointer: BaseCheckpointSaver):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_conditional_edges("tools", after_tools)

    return builder.compile(checkpointer=checkpointer)
