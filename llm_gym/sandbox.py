"""Running a workspace Python file with the filesystem and the network taken away.

The boundary is a bubblewrap namespace, not a path check. `workspace.resolve()`
decides which file may be *named* and stops mattering the moment python starts;
everything after that is this module's job.

Nothing inside is writable. Printing still works because stdout and stderr are
inherited pipes rather than files, and they are the whole return channel: a
script cannot leave a result behind on disk for someone to pick up later. That
is also why `_drain` caps output as it reads rather than afterwards. The cap
exists so a runaway script never occupies server memory in the first place,
which is not something truncating a finished buffer can do.

A shared kernel is still a shared kernel. This stops a script from reading the
home directory, reaching the network, or filling the disk. It is not a boundary
against an exploit written to break out of one.
"""

import os
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_gym.config import (
    SANDBOX_ADDRESS_SPACE_BYTES,
    SANDBOX_CPU_SECONDS,
    SANDBOX_OUTPUT_BYTES,
    SANDBOX_SYSTEM_PATHS,
    SANDBOX_TIMEOUT_SECONDS,
    WORKSPACE_ROOT,
)

# Resolved once at import, so a machine without bubblewrap stops the server
# rather than surfacing as a failed tool call halfway through a conversation.
BWRAP = shutil.which("bwrap")
if BWRAP is None:
    raise RuntimeError("bubblewrap (bwrap) is not installed, so run_python has no sandbox to run in.")

_MISSING = [path for path in SANDBOX_SYSTEM_PATHS if not Path(path).exists()]
if _MISSING:
    raise RuntimeError(f"SANDBOX_SYSTEM_PATHS names paths that do not exist: {_MISSING}")

# Where the workspace appears inside the sandbox. Not configuration: nothing
# outside this module can see it, since the script path stays relative.
_WORK = "/work"
_READ_BYTES = 65536


@dataclass(frozen=True)
class RunResult:
    stdout: str
    stderr: str
    # Negative when a signal ended the process, which is what a timeout looks
    # like from here.
    exit_code: int
    # "completed" covers a script that raised: a traceback and a non-zero exit
    # are things a script produced, not failures of the runner.
    outcome: Literal["completed", "timeout"]
    truncated: bool
    seconds: float


def run(script: str) -> RunResult:
    """Run one workspace-relative path and collect what it printed.

    Never raises on anything the script does. A syntax error, an import of
    something not installed, an exit code of 3 and a script killed at the
    deadline are all ordinary results here, because the caller's job is to
    show the model what happened rather than to decide it went wrong.
    """
    started = time.monotonic()
    deadline = started + SANDBOX_TIMEOUT_SECONDS
    process = subprocess.Popen(
        _argv(script),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Nothing is going to answer a prompt, and an inherited stdin would be
        # the server's own.
        stdin=subprocess.DEVNULL,
        preexec_fn=_limits,  # noqa: PLW1509
        # Makes the child a session leader, so _kill has a group to signal.
        start_new_session=True,
    )

    collected, truncated, timed_out = _drain(process, deadline)
    if not timed_out:
        # Both pipes reaching EOF is not the same as the process having ended:
        # a script can close them and keep running.
        try:
            process.wait(timeout=max(deadline - time.monotonic(), 0.0))
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        _kill(process)
    process.wait()

    return RunResult(
        stdout=_decode(collected["stdout"]),
        stderr=_decode(collected["stderr"]),
        exit_code=process.returncode,
        outcome="timeout" if timed_out else "completed",
        truncated=truncated,
        seconds=time.monotonic() - started,
    )


def _argv(script: str) -> list[str]:
    """The sandbox, as bubblewrap flags.

    Order is load-bearing twice over: --clearenv wipes the environment, so the
    --setenv pairs have to follow it, and --ro-bind of the workspace has to
    follow the system paths it sits under. --unshare-all covers the network,
    which is why no separate flag says so.
    """
    binds: list[str] = []
    for path in SANDBOX_SYSTEM_PATHS:
        binds += ["--ro-bind", path, path]
    # sys.prefix rather than a written-down path: a script has to run under the
    # same interpreter and the same installed packages as the server itself.
    binds += ["--ro-bind", sys.prefix, sys.prefix]
    binds += ["--ro-bind", str(WORKSPACE_ROOT), _WORK]

    return [
        BWRAP,
        *binds,
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        _WORK,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        # No writable path exists anywhere inside, so a library reaching for a
        # home directory should fail where it asks rather than quietly landing
        # somewhere unexpected.
        "--setenv",
        "HOME",
        "/nonexistent",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        # Without this, a script killed at the deadline loses whatever was
        # still sitting in python's block buffer, which is usually everything
        # it printed and the only clue as to where it got stuck.
        "--setenv",
        "PYTHONUNBUFFERED",
        "1",
        sys.executable,
        script,
    ]


def _limits() -> None:
    """Ceilings a namespace cannot impose: it isolates, it does not ration.

    Runs in the child between fork and exec, so it does nothing but syscalls.
    Allocating here can deadlock against a lock some other thread held at the
    moment of the fork, and the server forks from a threadpool.
    """
    resource.setrlimit(resource.RLIMIT_AS, (SANDBOX_ADDRESS_SPACE_BYTES, SANDBOX_ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (SANDBOX_CPU_SECONDS, SANDBOX_CPU_SECONDS))
    # Nothing is writable, so this changes no outcome today. It is the policy
    # written where it still holds if a writable mount is ever added.
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    # RLIMIT_NPROC deliberately absent. It counts against the real uid in the
    # namespace this runs in, which is still the desktop session's -- a cap
    # low enough to stop a fork bomb is below the process count the user
    # already has, so bwrap fails to start at all. A real cap needs a cgroup;
    # what bounds a fork bomb here is the pid namespace, which the kernel
    # empties when _kill takes bwrap down at the deadline.


def _drain(process: subprocess.Popen, deadline: float) -> tuple[dict[str, bytearray], bool, bool]:
    """Read both pipes until they close or the clock runs out.

    Both at once, because a script filling one pipe while the parent blocks on
    the other deadlocks with neither side able to move.

    Past the cap the bytes are read and dropped rather than the script being
    killed. Draining keeps the pipe moving, so a chatty script still reaches
    its own end and still reports an exit code, which is often the only part
    worth having.
    """
    collected = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False

    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return collected, truncated, True
            for key, _ in selector.select(timeout=remaining):
                chunk = os.read(key.fileobj.fileno(), _READ_BYTES)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = collected[key.data]
                room = SANDBOX_OUTPUT_BYTES - len(buffer)
                if len(chunk) > room:
                    truncated = True
                buffer += chunk[:room]

    return collected, truncated, False


def _kill(process: subprocess.Popen) -> None:
    """Take down the sandbox and everything inside it.

    bwrap is PID 1 of its own namespace, so killing it makes the kernel reap
    every process in there. The signal goes to the process group rather than
    the pid because a survivor holding a pipe open would hang the wait that
    follows.
    """
    os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def _decode(raw: bytearray) -> str:
    """Whatever the script actually emitted.

    Undecodable bytes are replaced rather than raised on. This is the one
    thing the tool exists to show, and a script printing broken UTF-8 has
    still told you something.
    """
    return raw.decode("utf-8", errors="replace")
