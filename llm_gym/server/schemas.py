from typing import Literal

from pydantic import BaseModel


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