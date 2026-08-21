from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from llm_gym.config import CHECKPOINT_DB, WORKSPACE_FILE, WORKSPACE_ROOT
from llm_gym.graph import build_graph
from llm_gym.logconf import setup_logging
from llm_gym.server.run_agent import (
    AgentRunner,
    list_thread_ids,
    new_thread_id,
    thread_created_at,
)
from llm_gym.server.schemas import (
    DecisionRequest,
    RunRequest,
    SaveRequest,
    ThreadState,
    ThreadSummary,
)
from llm_gym.workspace import FileSnapshot, Workspace


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

workspace = Workspace(root=WORKSPACE_ROOT)
# Which file the editor opens on load. A UI concern, not a workspace one.
DEFAULT_FILE = WORKSPACE_FILE.as_posix()
runner: AgentRunner | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Hold one checkpointer and one graph open for the process lifetime.

    The checkpointer is what lets a run paused at propose_edit's interrupt()
    be resumed by a later, separate request.

    Deliberately no thread: conversation identity belongs to the browser, so
    that a restart -- including the hot reload that fires on every save under
    llm_gym/ -- cannot silently strand the conversation you were having.
    """
    global runner
    setup_logging()
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        runner = AgentRunner(workspace, build_graph(checkpointer), checkpointer)
        yield
        runner = None


def _summary(thread_id: str) -> ThreadSummary:
    return ThreadSummary(id=thread_id, created_at=thread_created_at(thread_id))


app = FastAPI(title="LLM Gym API", lifespan=lifespan)


@app.get("/api/threads")
def get_threads() -> list[ThreadSummary]:
    """Every thread with at least one checkpoint, newest first.

    A thread minted by POST /api/threads shows up here only once something
    has run in it, since an unused thread has written no checkpoint yet.
    """
    summaries = [_summary(thread_id) for thread_id in list_thread_ids(CHECKPOINT_DB)]
    # Two stable passes: newest first, then push undatable ids to the bottom.
    # Anything the debug scripts wrote (debug/interrupt_demo.py uses the id
    # "demo-thread") has no timestamp and would otherwise sort by name.
    summaries.sort(key=lambda summary: summary.created_at or "", reverse=True)
    summaries.sort(key=lambda summary: summary.created_at is None)
    return summaries


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str) -> None:
    """Discard a thread for good. Nothing the agent wrote to disk is removed."""
    await runner.delete_thread(thread_id)


@app.get("/api/state")
async def get_thread_state(thread_id: str) -> ThreadState:
    """Rebuild a thread from its checkpoint. Called on load and when switching
    threads -- never during a run, which is fed by the SSE stream instead."""
    return await runner.thread_state(thread_id)


@app.post("/api/threads")
def create_thread() -> ThreadSummary:
    """Mint a thread id for the browser to hold. Nothing is written here --
    the first run against this id is what creates it in the database."""
    return _summary(new_thread_id())


@app.get("/api/files")
def list_files() -> list[str]:
    return workspace.list_files()


@app.get("/api/file")
def read_file(path: str = DEFAULT_FILE) -> FileSnapshot:
    return workspace.read_snapshot(path)


@app.put("/api/file")
def save_file(request: SaveRequest) -> FileSnapshot:
    return workspace.apply_change(
        path=request.path,
        expected_hash=request.expected_hash,
        modified=request.content,
    )


@app.post("/api/run")
async def start_run(request: RunRequest) -> StreamingResponse:
    return StreamingResponse(
        runner.run(request.thread_id, request.prompt),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/decision")
async def submit_decision(request: DecisionRequest) -> StreamingResponse:
    """Resume the paused graph. The agent may still have something to say,
    so this streams like a run instead of returning a single snapshot.
    """
    return StreamingResponse(
        runner.resume(request.thread_id, request.decision),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )