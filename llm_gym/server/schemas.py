from typing import Literal

from pydantic import BaseModel


class ChatItem(BaseModel):
    """One line of conversation, rebuilt from a message in the checkpoint."""

    # A row is named after the graph node that produced it, not after the
    # message type -- hence "agent" over "assistant", and tool_call (from the
    # agent node's tool_calls) split from tool_result (from the tools node).
    # "system" is the leftover: something that fit no expected shape.
    role: Literal[
        "system_prompt", "user", "agent", "tool_call", "tool_result", "system"
    ]
    text: str
    # The full body behind a shortened row, shown on hover. Set wherever the
    # text was clipped, and nowhere else, so a tooltip always means there is
    # more to see.
    detail: str | None = None
    # None on rows no model produced, and on replies stored without a name.
    model: str | None = None


class ProposalPayload(BaseModel):
    """The file change a paused run is waiting on, as the interrupt stored it."""

    path: str
    content_hash: str
    original: str
    modified: str


class ThreadState(BaseModel):
    """A thread as it stands right now, for a browser that was not streaming.

    `pending_proposal` is what keeps a reload mid-review from stranding the
    run: without it the graph stays suspended with nothing able to answer it.
    """

    chat: list[ChatItem]
    pending_proposal: ProposalPayload | None


class ThreadSummary(BaseModel):
    id: str
    # Parsed back out of the id, which is timestamp-prefixed. None when the id
    # did not come from new_thread_id(). The browser renders it in local time.
    created_at: str | None


class ModelList(BaseModel):
    """What the picker offers, and which entry it opens on."""

    models: list[str]
    default: str


class RunRequest(BaseModel):
    thread_id: str
    prompt: str
    # Per message, not per thread: a conversation can change hands partway.
    model: str


class SaveRequest(BaseModel):
    path: str
    expected_hash: str
    content: str


class DecisionRequest(BaseModel):
    thread_id: str
    # The proposed content already lives inside the paused graph run, so the
    # decision is all the server needs to resume it.
    decision: Literal["accept", "reject"]
    # A resumed run still gives the agent a turn, and it need not be the model
    # that proposed the edit.
    model: str