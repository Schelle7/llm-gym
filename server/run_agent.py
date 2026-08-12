import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from config import SYSTEM_PROMPT
from logconf import log
from workspace import Workspace


def new_thread_id() -> str:
    """One thread per server process, matching agent.py's naming."""
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    return f"{stamp}_{uuid4()}"


def sse(event: dict[str, Any]) -> str:
    """Encode one event as an SSE frame, the format api.ts parses."""
    return f"data: {json.dumps(event)}\n\n"


class AgentRunner:
    """Drives the LangGraph agent and translates its output into SSE events.

    The browser speaks the event vocabulary declared in frontend/src/api.ts;
    this class is the only place that knows how to map graph output onto it.
    """

    def __init__(self, workspace: Workspace, graph, thread_id: str) -> None:
        self.workspace = workspace
        self.graph = graph
        self.thread_id = thread_id

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    async def run(self, prompt: str) -> AsyncIterator[str]:
        """Start a turn from a user prompt."""
        state = await self.graph.aget_state(self.config)
        messages = []
        # The system prompt belongs to the thread, not the turn -- adding it
        # every time would stack duplicates in the message history.
        if not state.values.get("messages"):
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        messages.append(HumanMessage(content=prompt))

        log.info("run: %r (thread %s)", prompt, self.thread_id)
        async for frame in self._stream({"messages": messages}):
            yield frame

    async def resume(self, decision: str) -> AsyncIterator[str]:
        """Resume a run paused at propose_edit's interrupt().

        The graph picks up inside the tool, which writes the file (or not)
        and returns a result the model sees on its next turn -- so the agent
        can still speak after the decision.
        """
        log.info("resume: %s", decision)
        async for frame in self._stream(Command(resume=decision)):
            yield frame

    async def _stream(self, payload: Any) -> AsyncIterator[str]:
        async for mode, chunk in self.graph.astream(
            payload, self.config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                message, metadata = chunk
                text = getattr(message, "content", "")
                if text and metadata.get("langgraph_node") == "agent":
                    yield sse({"type": "assistant_delta", "text": text})
            elif mode == "updates":
                for node, update in chunk.items():
                    if node != "tools" or not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        name = getattr(message, "name", None)
                        log.info("tool <- %s returned %s", name, message.content)

                        if name == "read_file":
                            yield sse({"type": "file_read", "path": self._path_of(message)})

                        # propose_edit has returned, so the write (if any) is
                        # already on disk. Report it now rather than after the
                        # model's follow-up turn, which takes seconds.
                        elif name == "propose_edit":
                            snapshot = self.workspace.read_snapshot()
                            yield sse(
                                {
                                    "type": "file_saved",
                                    "path": snapshot.path,
                                    "content": snapshot.content,
                                    "content_hash": snapshot.content_hash,
                                }
                            )

        # The stream ends either because the graph finished or because a tool
        # called interrupt(). Asking for the state is a reliable way to tell
        # the two apart, without depending on how interrupts appear mid-stream.
        state = await self.graph.aget_state(self.config)

        if state.interrupts:
            proposal = state.interrupts[0].value
            yield sse(
                {
                    "type": "proposal_ready",
                    "path": proposal["path"],
                    "content_hash": proposal["content_hash"],
                    "original": proposal["original"],
                    "modified": proposal["modified"],
                }
            )
            yield sse({"type": "approval_required"})
        else:
            yield sse({"type": "run_finished"})

    def _path_of(self, message) -> str:
        """Tool results arrive as JSON strings; fall back to the known path."""
        try:
            return json.loads(message.content).get("path", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return self.workspace.relative_path.as_posix()