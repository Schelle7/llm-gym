"""Projecting a thread's message history onto what the browser shows.

The message list inside a checkpoint is the durable record of a conversation:
the stream only ever shows it arriving. This module turns that list into chat
items, and is deliberately pure -- no disk, no network, no clock -- so that
replaying an old thread gives the same answer every time, and so the live path
and the reload path can eventually share it.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_gym.server.schemas import ChatItem


def render_messages(messages: list[BaseMessage]) -> list[ChatItem]:
    """The human's prompts and the agent's replies, in order.

    Everything else is skipped for now: the system prompt is setup rather
    than conversation, tool results want their own rendering, and an
    AIMessage carrying only tool calls has no text to show.
    """
    items: list[ChatItem] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        text = _text(message)
        if text:
            items.append(ChatItem(role=role, text=text))
    return items


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