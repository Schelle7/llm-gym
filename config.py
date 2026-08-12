from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent / "test_workspace"
WORKSPACE_FILE = Path("hello.py")
CHECKPOINT_DB = str(Path(__file__).resolve().parent / "checkpoints" / "checkpoints.db")

MODEL = "gemma4:e2b"
SYSTEM_PROMPT = (
    "You are a local assistant. Use only the tools provided. "
    "Do not invent tool results. If no tool is needed, answer directly."
)