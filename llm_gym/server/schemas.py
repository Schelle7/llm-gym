from typing import Literal

from pydantic import BaseModel


class ChatItem(BaseModel):
    """One line of conversation, rebuilt from a message in the checkpoint."""

    role: Literal["user", "assistant"]
    text: str


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


class RunRequest(BaseModel):
    thread_id: str
    prompt: str


class SaveRequest(BaseModel):
    path: str
    expected_hash: str
    content: str


class DecisionRequest(BaseModel):
    thread_id: str
    # The proposed content already lives inside the paused graph run, so the
    # decision is all the server needs to resume it.
    decision: Literal["accept", "reject"]