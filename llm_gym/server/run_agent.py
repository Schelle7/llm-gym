import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from llm_gym.config import SYSTEM_PROMPT
from llm_gym.logconf import log
from llm_gym.workspace import Workspace, WorkspaceError


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

                        result = self._tool_result(message)
                        path = result.get("path", "")

                        if name == "read_file" and path:
                            yield sse({"type": "file_read", "path": path})

                        elif name in ("propose_edit", "create_file"):
                            status = result.get("status")

                            # The tool has returned, so an accepted change is
                            # already on disk. Report it now rather than after
                            # the model's follow-up turn, which takes seconds.
                            if status == "accepted":
                                snapshot = self.workspace.read_snapshot(path)
                                yield sse(
                                    {
                                        "type": "file_saved",
                                        "path": snapshot.path,
                                        "content": snapshot.content,
                                        "content_hash": snapshot.content_hash,
                                    }
                                )
                            elif status == "failed":
                                yield sse(
                                    {
                                        "type": "agent_error",
                                        "detail": result.get("detail", "Unknown error"),
                                    }
                                )
                            elif status == "rejected":
                                # No event needed: the browser sent this
                                # decision itself and nothing on disk changed.
                                continue
                            else:
                                # Either the tool raised and ToolNode replaced
                                # its result with an error string, or a new
                                # status was added without updating this
                                # branch. Report it before raising, so the
                                # browser is told rather than left hanging on
                                # a truncated stream.
                                log.error("unhandled %s result: %s", name, result)
                                yield sse(
                                    {
                                        "type": "agent_error",
                                        "detail": (
                                            f"{name} returned an unexpected "
                                            f"status: {status!r}"
                                        ),
                                    }
                                )
                                raise RuntimeError(
                                    f"Unexpected {name} result: {result!r} "
                                    f"(raw: {message.content!r})"
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

    def _tool_result(self, message) -> dict[str, Any]:
        """Parse a tool's return value back out of its ToolMessage.

        LangGraph hands tool results over as ToolMessage.content, a string.
        Our tools all return dicts, so this decodes one to build frontend
        events from -- the model gets the tool's value directly and never
        sees this. Returns {} when the content is not a JSON object.
        """
        try:
            parsed = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}