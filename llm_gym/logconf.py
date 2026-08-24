"""Logging for the agent.

Everything the agent does goes to logs/agent.log; the console stays quiet so
`make dev` output remains readable. Follow along in a second terminal with:

    tail -f logs/agent.log
"""

import logging

from llm_gym.config import ROOT

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "agent.log"

log = logging.getLogger("llm_gym")


def setup_logging(console_level: int = logging.WARNING) -> None:
    if log.handlers:
        return

    LOG_DIR.mkdir(exist_ok=True)
    log.setLevel(logging.DEBUG)
    # Don't hand records to uvicorn's root logger as well.
    log.propagate = False

    to_file = logging.FileHandler(LOG_FILE, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(to_file)

    to_console = logging.StreamHandler()
    to_console.setLevel(console_level)
    to_console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    log.addHandler(to_console)
