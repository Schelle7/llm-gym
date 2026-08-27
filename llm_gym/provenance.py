import subprocess
from datetime import UTC, datetime

from llm_gym.config import ROOT


def stamp() -> dict[str, str | bool]:
    """What produced a reply besides the model: the code, and the moment."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain", "--untracked-files=no") != "",
        "called_at": datetime.now(UTC).isoformat(),
    }


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()
