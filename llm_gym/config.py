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

# A context budget, not a quality knob: every result stays in the transcript
# and is re-sent to the model on every later turn of the conversation.
SEARCH_MAX_RESULTS = 4

CONTEXT_TOKENS = 8192

SANDBOX_SYSTEM_PATHS = ("/usr", "/bin", "/lib", "/lib64")

SANDBOX_TIMEOUT_SECONDS = 10.0
SANDBOX_CPU_SECONDS = 10
SANDBOX_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
SANDBOX_OUTPUT_BYTES = 4000
