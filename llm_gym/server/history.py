"""Projecting a thread's messages onto what the browser shows.

The message list inside a checkpoint is the durable record of a conversation;
the stream only ever shows it arriving. This module turns that list into chat
lines, and both paths use it: run_agent renders the messages a node just
returned, and GET /api/state renders the whole history. One function, so a
rebuilt conversation cannot disagree with the one you watched happen.

Deliberately pure -- no disk, no network, no clock -- which is what makes
those two renderings identical.
"""

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from llm_gym.server.schemas import ChatItem


def render_messages(messages: list[BaseMessage]) -> list[ChatItem]:
    """The chat lines a run of messages is worth, in order."""
    items: list[ChatItem] = []
    for message in messages:
        items.extend(_items_for(message))
    return items


def tool_result(message: ToolMessage) -> dict[str, Any]:
    """Parse a tool's return value back out of its ToolMessage.

    LangGraph hands tool results over as ToolMessage.content, a string. Our
    tools all return dicts, so this decodes one. Returns {} when the content
    is not a JSON object, which is what a tool that raised looks like -- the
    ToolNode replaces its result with a plain error string.
    """
    try:
        parsed = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _items_for(message: BaseMessage) -> list[ChatItem]:
    """The lines one message is worth.

    Nothing is dropped for being unrecognised. An unknown message class, an
    unfamiliar tool result, a reply with neither text nor calls -- each still
    gets a row, because a transcript that quietly omits whatever it could not
    parse looks correct precisely when it is not.

    A message with no content at all is the one exception: there is nothing
    being hidden.
    """
    if isinstance(message, SystemMessage):
        text = _text(message)
        # One row with the whole prompt behind it: it opens the conversation
        # and shapes every reply, so leaving it out misrepresents the thread.
        return (
            [ChatItem(role="system_prompt", text=_one_line(text), detail=text)]
            if text
            else []
        )

    if isinstance(message, HumanMessage):
        text = _text(message)
        return [ChatItem(role="user", text=text)] if text else []

    if isinstance(message, ToolMessage):
        return [_tool_item(message)]

    if isinstance(message, AIMessage):
        items = []
        # Tolerant here, strict in graph.call_model: a reply already stored
        # without a name can never gain one.
        model = message.response_metadata.get("model_name")
        text = _text(message)
        if text:
            items.append(ChatItem(role="agent", text=text, model=model))
        # The request itself, so a call stays visible even when its result is
        # a shape _tool_item has no phrasing for. The row is clipped to one
        # line, so the unclipped arguments go behind it: that is the only
        # place a proposed file body can be read before it is applied.
        for call in message.tool_calls:
            items.append(
                ChatItem(
                    role="tool_call",
                    text=f"{call['name']}({_call_args(call['args'])})",
                    detail=_call_detail(call),
                    model=model,
                )
            )
        if not items:
            items.append(
                _diagnostic("The model replied with neither text nor a tool call.")
            )
        return items

    return [_diagnostic(f"{type(message).__name__}: {_one_line(_text(message))}")]


def _tool_item(message: ToolMessage) -> ChatItem:
    """What a tool did, phrased for a reader rather than for the model.

    Always returns a row. A result this does not recognise is shown raw
    rather than skipped: an unfamiliar shape is the thing most worth seeing.
    """
    result = tool_result(message)
    name = message.name

    if not result:
        # Not a dict, so the tool raised and ToolNode replaced its result
        # with a plain error string.
        return ChatItem(
            role="tool_result",
            text=f"{name}: {_one_line(str(message.content))}",
            detail=str(message.content),
        )

    path = result.get("path", "")
    status = result.get("status")

    if status == "failed":
        return _result(f"{name} failed: {result.get('detail', 'unknown error')}")
    if status == "rejected":
        return _result(f"Rejected the change to {path}")
    if status == "accepted":
        return _result(f"{'Created' if name == 'create_file' else 'Edited'} {path}")
    if name == "read_file":
        return _result(f"Read {path}")
    if name == "ls":
        return _result(f"Listed {path or '.'}")

    # A tool, or a result shape, that arrived without a line being added here.
    return ChatItem(
        role="tool_result",
        text=f"{name} returned {_one_line(json.dumps(result))}",
        detail=json.dumps(result, indent=2),
    )


def _call_args(args: dict[str, Any]) -> str:
    """Tool arguments on one line, clipped hard -- propose_edit carries an
    entire file in `modified`."""
    return ", ".join(f"{key}={_one_line(str(value), 40)}" for key, value in args.items())


def _call_detail(call: dict[str, Any]) -> str:
    """Every argument in full, one block each, for the row's hover text."""
    blocks = [f"{key}:\n{value}" for key, value in call["args"].items()]
    return "\n\n".join([call["name"], *blocks])


def _result(text: str) -> ChatItem:
    """A line the tools node produced."""
    return ChatItem(role="tool_result", text=text)


def _diagnostic(text: str) -> ChatItem:
    """A line for something that fit none of the expected shapes."""
    return ChatItem(role="system", text=text)


def _text(message: BaseMessage) -> str:
    """The readable content of a message.

    Content is a plain string for an ordinary reply, but a list of typed
    blocks once a model sends anything richer, so pull the text out of those
    rather than rendering their repr.
    """
    if isinstance(message.content, str):
        return message.content.strip()

    return "".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _one_line(text: str, limit: int = 90) -> str:
    """Collapse a body to one line, for a row that carries the rest on hover."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"