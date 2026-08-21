import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from llm_gym.config import SYSTEM_PROMPT
from llm_gym.logconf import log
from llm_gym.server.history import render_messages, tool_result
from llm_gym.server.schemas import ProposalPayload, ThreadState
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


def chat_frames(messages: list[BaseMessage]) -> Iterator[str]:
    """Chat rows for a run of messages, through the projection a reload uses,
    so what you watch and what you come back to cannot drift."""
    for item in render_messages(messages):
        yield sse(
            {
                "type": "chat_item",
                "role": item.role,
                "text": item.text,
                "detail": item.detail,
                "model": item.model,
            }
        )


def chunk_text(message: Any) -> str:
    """Everything the model just emitted, as raw text.

    Content is empty for the whole of a tool call, so the name and argument
    fragments are included: without them a tool-calling turn streams as
    nothing at all. Both are None on most chunks, since only the first
    carries a name and the arguments arrive a few characters at a time.
    """
    parts = [message.content]
    for fragment in message.tool_call_chunks:
        parts.append(fragment["name"] or "")
        parts.append(fragment["args"] or "")
    return "".join(parts)


class AgentRunner:
    """Drives the LangGraph agent and translates its output into SSE events.

    The browser speaks the event vocabulary declared in frontend/src/api.ts;
    this class is the only place that knows how to map graph output onto it.
    """

    def __init__(self, workspace: Workspace, graph, checkpointer) -> None:
        self.workspace = workspace
        self.graph = graph
        self.checkpointer = checkpointer

    def _config(self, thread_id: str) -> dict[str, Any]:
        """Which conversation a call belongs to. The runner holds no thread of
        its own -- every request names the one it wants."""
        return {"configurable": {"thread_id": thread_id}}

    def _run_config(self, thread_id: str, model: str) -> dict[str, Any]:
        """The same, for a call that will run the agent node and so needs a
        model. Reading state does not, which is why these are separate."""
        return {"configurable": {"thread_id": thread_id, "model": model}}

    async def delete_thread(self, thread_id: str) -> None:
        """Discard a conversation and every checkpoint behind it.

        Permanent, and the only operation here that destroys history rather
        than appending to it. Files the agent wrote are untouched: they live
        in the workspace, not in the checkpoint.
        """
        log.info("delete thread %s", thread_id)
        await self.checkpointer.adelete_thread(thread_id)

    async def thread_state(self, thread_id: str) -> ThreadState:
        """A thread as it stands, for a browser that is not streaming it.

        This is the reload path, and the only place the checkpoint is read to
        build chat: during a run the stream already carries every message as
        the engine produces it, so re-reading state per update would fetch
        what we are already holding.

        An unknown thread id is not an error -- aget_state returns an empty
        snapshot, which renders as an empty conversation.
        """
        state = await self.graph.aget_state(self._config(thread_id))

        pending = None
        if state.interrupts:
            proposal = state.interrupts[0].value
            pending = ProposalPayload(
                path=proposal["path"],
                content_hash=proposal["content_hash"],
                original=proposal["original"],
                modified=proposal["modified"],
            )

        return ThreadState(
            chat=render_messages(state.values.get("messages", [])),
            pending_proposal=pending,
        )

    async def run(self, thread_id: str, model: str, prompt: str) -> AsyncIterator[str]:
        """Start a turn from a user prompt."""
        config = self._run_config(thread_id, model)
        state = await self.graph.aget_state(config)
        messages = []
        if not state.values.get("messages"):
            system = SystemMessage(content=SYSTEM_PROMPT)
            messages.append(system)
            # Graph input rather than a node's output, so the stream below
            # never carries it. The browser adds the user's own row itself.
            for frame in chat_frames([system]):
                yield frame
        messages.append(HumanMessage(content=prompt))

        log.info("run: %r (thread %s, model %s)", prompt, thread_id, model)
        async for frame in self._stream(config, {"messages": messages}):
            yield frame

    async def resume(self, thread_id: str, model: str, decision: str) -> AsyncIterator[str]:
        """Resume a run paused at propose_edit's interrupt().

        The graph picks up inside the tool, which writes the file (or not)
        and returns a result the model sees on its next turn -- so the agent
        can still speak after the decision.
        """
        log.info("resume: %s (thread %s, model %s)", decision, thread_id, model)
        config = self._run_config(thread_id, model)
        async for frame in self._stream(config, Command(resume=decision)):
            yield frame

    async def _stream(self, config: dict[str, Any], payload: Any) -> AsyncIterator[str]:
        async for mode, chunk in self.graph.astream(
            payload, config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                message, metadata = chunk
                if metadata["langgraph_node"] != "agent":
                    continue
                text = chunk_text(message)
                if text:
                    yield sse({"type": "agent_delta", "text": text})
            elif mode == "updates":
                for node, update in chunk.items():
                    # updates also carries engine keys such as "__interrupt__",
                    # whose value is a tuple rather than a node's state.
                    if node.startswith("__"):
                        continue
                    messages = update["messages"]

                    # Both nodes go through it: the tool_call rows live in the
                    # agent node's message, the results in the tools node's.
                    for frame in chat_frames(messages):
                        yield frame

                    if node != "tools":
                        continue
                    for message in messages:
                        log.info("tool <- %s returned %s", message.name, message.content)

                    # A saved file is a command rather than a line of
                    # transcript: the browser has to swap its editor, and the
                    # content for that is on disk, not in any message. So it
                    # stays hand-built here.
                    for message in messages:
                        result = tool_result(message)
                        if result.get("status") != "accepted":
                            continue

                        # The tool has returned, so the change is already on
                        # disk. Report it now rather than after the model's
                        # follow-up turn, which takes seconds.
                        path = result.get("path", "")
                        try:
                            snapshot = self.workspace.read_snapshot(path)
                        except WorkspaceError as error:
                            log.error("read-back of %s failed: %s", path, error)
                            yield sse(
                                {
                                    "type": "agent_error",
                                    "detail": (
                                        f"Saved {path}, but could not read it "
                                        f"back: {error}"
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
