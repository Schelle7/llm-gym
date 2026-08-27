from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel


class ChatItem(BaseModel):
    """One line of conversation, rebuilt from a message in the checkpoint."""

    # A row is named after the graph node that produced it, not after the
    # message type -- hence "agent" over "assistant", and tool_call (from the
    # agent node's tool_calls) split from tool_result (from the tools node).
    # "system" is the leftover: something that fit no expected shape.
    role: Literal["system_prompt", "user", "thinking", "agent", "tool_call", "tool_result", "error", "system"]
    text: str
    # The full body behind a shortened row. Thinking expands inline; other
    # detailed rows expose it on hover.
    detail: str | None = None
    is_agent_input: bool = True
    # None on rows no model produced, and on replies stored without a name.
    model: str | None = None
    num_ctx: int | None = None
    reasoning: bool | None = None
    tools: list[str] | None = None
    commit: str | None = None
    dirty: bool | None = None
    called_at: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class AgentDelta(BaseModel):
    type: Literal["agent_delta"]
    text: str


class ReasoningDelta(BaseModel):
    type: Literal["reasoning_delta"]
    text: str


class ChatItemEvent(ChatItem):
    type: Literal["chat_item"]


class ProposalReady(BaseModel):
    type: Literal["proposal_ready"]
    kind: str
    path: str
    content_hash: str
    original: str
    modified: str


class ApprovalRequired(BaseModel):
    type: Literal["approval_required"]


class FileSaved(BaseModel):
    type: Literal["file_saved"]
    path: str
    content: str
    content_hash: str


class AgentError(BaseModel):
    type: Literal["agent_error"]
    detail: str


class RunFinished(BaseModel):
    type: Literal["run_finished"]


EventPayload = Annotated[
    AgentDelta
    | ReasoningDelta
    | ChatItemEvent
    | ProposalReady
    | ApprovalRequired
    | FileSaved
    | AgentError
    | RunFinished,
    Field(discriminator="type"),
]


class AgentEvent(RootModel[EventPayload]):
    pass


class FileSnapshot(BaseModel):
    path: str
    content: str
    content_hash: str


class ProposalPayload(BaseModel):
    """What a paused run is waiting on, as the interrupt stored it.

    `kind` is the tool that raised it, and is what decides the wording of the
    decision: accepting an edit writes a file, accepting a run executes one.
    A plain str rather than a Literal, because this renders a checkpoint and a
    kind nobody anticipated should still open the thread.
    """

    kind: str
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
    # What ChatItem.input_tokens is measured against. Sent with the catalogue
    # because the browser needs it before the first reply gives it a number.
    context_tokens: int


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
