from typing import Literal

from pydantic import BaseModel


class RunRequest(BaseModel):
    prompt: str


class SaveRequest(BaseModel):
    path: str
    expected_hash: str
    content: str


class DecisionRequest(BaseModel):
    # The proposed content already lives inside the paused graph run, so the
    # decision is all the server needs to resume it.
    decision: Literal["accept", "reject"]