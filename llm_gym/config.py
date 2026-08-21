from pathlib import Path

# Repo root: this file lives in llm_gym/, so go up one level.
ROOT = Path(__file__).resolve().parents[1]

WORKSPACE_ROOT = ROOT / "test_workspace"
WORKSPACE_FILE = Path("hello.py")
CHECKPOINT_DB = str(ROOT / "checkpoints" / "checkpoints.db")

# The catalogue this is chosen from lives in models.py.
DEFAULT_MODEL = "gemma4:e2b"
SYSTEM_PROMPT = (
    "You are a local assistant. Use only the tools provided. "
    "Do not invent tool results. If no tool is needed, answer directly."
)