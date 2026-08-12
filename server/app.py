from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import CHECKPOINT_DB, WORKSPACE_FILE, WORKSPACE_ROOT
from graph import build_graph
from logconf import setup_logging
from server.run_agent import AgentRunner, new_thread_id
from server.schemas import DecisionRequest, RunRequest, SaveRequest
from workspace import FileSnapshot, Workspace


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

workspace = Workspace(root=WORKSPACE_ROOT, relative_path=WORKSPACE_FILE)
runner: AgentRunner | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Hold one checkpointer and one graph open for the process lifetime.

    The checkpointer is what lets a run paused at propose_edit's interrupt()
    be resumed by a later, separate request.
    """
    global runner
    setup_logging()
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        runner = AgentRunner(workspace, build_graph(checkpointer), new_thread_id())
        print(f"Agent thread: {runner.thread_id}")
        yield
        runner = None


app = FastAPI(title="LLM Gym API", lifespan=lifespan)


@app.get("/api/file")
def read_file() -> FileSnapshot:
    return workspace.read_snapshot()


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
        runner.run(request.prompt),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/api/decision")
async def submit_decision(request: DecisionRequest) -> StreamingResponse:
    """Resume the paused graph. The agent may still have something to say,
    so this streams like a run instead of returning a single snapshot.
    """
    return StreamingResponse(
        runner.resume(request.decision),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )