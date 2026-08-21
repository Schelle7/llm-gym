import json
import sqlite3
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from llm_gym.config import SYSTEM_PROMPT
from llm_gym.logconf import log
from llm_gym.workspace import Workspace, WorkspaceError

# Thread ids start with a UTC timestamp in this format, so sorting ids
# lexicographically sorts threads by age. thread_created_at() reads it back.
THREAD_STAMP = "%Y_%m_%d_%H_%M_%S"


def new_thread_id() -> str:
    """Mint a thread id. Nobody owns this but the browser that asked for it."""
    stamp = datetime.now(timezone.utc).strftime(THREAD_STAMP)
    return f"{stamp}_{uuid4()}"


def thread_created_at(thread_id: str) -> str | None:
    """Recover the creation time that new_thread_id() encoded in the id.

    Returns an ISO-8601 UTC string, which the browser renders in local time.
    Returns None for any id this server did not mint, so a hand-written or
    legacy thread id still lists -- just under its raw name.
    """
    stamp = "_".join(thread_id.split("_")[:6])
    try:
        created = datetime.strptime(stamp, THREAD_STAMP)
    except ValueError:
        return None
    return created.replace(tzinfo=timezone.utc).isoformat()


def list_thread_ids(db_path: str) -> list[str]:
    """Every thread the checkpointer has written to.

    This reads the checkpoint table directly. LangGraph's saver API walks
    checkpoints *within* one thread and offers no way to enumerate threads.

    Ordering here is only a stable default -- ids this server minted sort
    chronologically, but ids from elsewhere do not, so the caller sorts by
    the parsed timestamp instead.

    A thread that was minted but never run has no rows yet and so does not
    appear here.
    """
    if not Path(db_path).exists():
        return []

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC"
        ).fetchall()
    except sqlite3.OperationalError:
        # The file exists but no checkpoint table does: nothing has run yet.
        return []
    finally:
        connection.close()
    return [row[0] for row in rows]


def sse(event: dict[str, Any]) -> str:
    """Encode one event as an SSE frame, the format api.ts parses."""
    return f"data: {json.dumps(event)}\n\n"


class AgentRunner:
    """Drives the LangGraph agent and translates its output into SSE events.

    The browser speaks the event vocabulary declared in frontend/src/api.ts;
    this class is the only place that knows how to map graph output onto it.
    """

    def __init__(self, workspace: Workspace, graph) -> None:
        self.workspace = workspace
        self.graph = graph

    def _config(self, thread_id: str) -> dict[str, Any]:
        """Which conversation a call belongs to. The runner holds no thread of
        its own -- every request names the one it wants."""
        return {"configurable": {"thread_id": thread_id}}

    async def run(self, thread_id: str, prompt: str) -> AsyncIterator[str]:
        """Start a turn from a user prompt."""
        config = self._config(thread_id)
        state = await self.graph.aget_state(config)
        messages = []
        # The system prompt belongs to the thread, not the turn -- adding it
        # every time would stack duplicates in the message history.
        if not state.values.get("messages"):
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        messages.append(HumanMessage(content=prompt))

        log.info("run: %r (thread %s)", prompt, thread_id)
        async for frame in self._stream(config, {"messages": messages}):
            yield frame

    async def resume(self, thread_id: str, decision: str) -> AsyncIterator[str]:
        """Resume a run paused at propose_edit's interrupt().

        The graph picks up inside the tool, which writes the file (or not)
        and returns a result the model sees on its next turn -- so the agent
        can still speak after the decision.
        """
        log.info("resume: %s (thread %s)", decision, thread_id)
        async for frame in self._stream(self._config(thread_id), Command(resume=decision)):
            yield frame

    async def _stream(self, config: dict[str, Any], payload: Any) -> AsyncIterator[str]:
        async for mode, chunk in self.graph.astream(
            payload, config, stream_mode=["updates", "messages"]
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
                                # The tool already wrote and re-read the file,
                                # so this read should not fail. Guard it
                                # anyway: an exception here escapes the
                                # generator and truncates the SSE response,
                                # leaving the browser stuck mid-run.
                                try:
                                    snapshot = self.workspace.read_snapshot(path)
                                except WorkspaceError as error:
                                    log.error("read-back of %s failed: %s", path, error)
                                    yield sse(
                                        {
                                            "type": "agent_error",
                                            "detail": (
                                                f"Saved {path}, but could not "
                                                f"read it back: {error}"
                                            ),
                                        }
                                    )
                                    continue
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
        state = await self.graph.aget_state(config)

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