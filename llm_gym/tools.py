from langchain_core.tools import tool
from langchain_tavily import TavilyExtract, TavilySearch
from langgraph.types import interrupt

from llm_gym import sandbox
from llm_gym.config import SEARCH_MAX_RESULTS, WORKSPACE_ROOT
from llm_gym.workspace import Workspace, WorkspaceError

workspace = Workspace(root=WORKSPACE_ROOT)

# Reads TAVILY_API_KEY and raises here if it is absent, so a missing key stops
# the server rather than surfacing as a failed tool call mid-conversation.
web_search = TavilySearch(max_results=SEARCH_MAX_RESULTS)

# Whole pages, so this is the tool that fills a context window. Nothing here
# caps it: the header gauge is where that shows up, and hiding the cost would
# defeat the point of watching it.
fetch_page = TavilyExtract(extract_depth="basic")


@tool
def ls(path: str = ".", include_hidden: bool = False):
    """List a directory in the workspace. `path` defaults to the workspace root.

    Directories are shown with a trailing slash.
    """
    try:
        return {"path": path, "entries": workspace.list_dir(path, include_hidden)}
    except WorkspaceError as error:
        return {"status": "failed", "detail": str(error)}


@tool
def read_file(path: str):
    """Read a file in the workspace. Returns its path, content and content hash.

    Pass the content_hash back when proposing an edit to that file, so the
    edit is refused if the file changed in the meantime.
    """
    try:
        snapshot = workspace.read_snapshot(path)
    except WorkspaceError as error:
        return {"status": "failed", "detail": str(error)}
    return {
        "path": snapshot.path,
        "content": snapshot.content,
        "content_hash": snapshot.content_hash,
    }


@tool
def create_file(path: str, content: str):
    """Propose creating a NEW file, for human approval.

    Fails if the file already exists -- use propose_edit for that. Missing
    parent directories are created. `content` is written verbatim, so use
    real line breaks; a backslash followed by n puts those two characters
    into the file instead of starting a new line.
    """
    try:
        workspace.resolve(path)
    except WorkspaceError as error:
        return {"status": "failed", "detail": str(error)}

    decision = interrupt(
        {
            "kind": "create_file",
            "path": path,
            "content_hash": "",
            "original": "",
            "modified": content,
        }
    )

    if decision != "accept":
        return {
            "status": "rejected",
            "path": path,
            "detail": (
                "The human rejected this file. Nothing was created. Do not "
                "propose the same content again -- ask what they want instead."
            ),
        }

    try:
        created = workspace.create(path, content)
    except WorkspaceError as error:
        return {"status": "failed", "path": path, "detail": str(error)}

    return {
        "status": "accepted",
        "path": created.path,
        "content_hash": created.content_hash,
    }


@tool
def propose_edit(path: str, expected_hash: str, modified: str):
    """Propose replacing an existing file's content, for human approval.

    Pass the complete new file content as `modified` -- not a diff, not a
    fragment. Pass the `content_hash` you got from read_file as
    `expected_hash`; the edit is refused if the file changed since then.

    `modified` is written to the file verbatim, so separate lines with real
    line breaks. Writing a backslash followed by n puts those two characters
    into the file instead of starting a new line.

    The edit is shown to the human, who accepts or rejects it. Only an
    accepted edit is written to disk. The result tells you what they decided.
    """
    try:
        snapshot = workspace.read_snapshot(path)
    except WorkspaceError as error:
        return {"status": "failed", "detail": str(error)}

    # Pausing point. This unwinds the entire graph run, so nothing below here
    # runs until the thread is resumed with Command(resume=...). On resume
    # this function re-executes FROM THE TOP and interrupt() returns the
    # resume value instead of unwinding.
    #
    # That replay is why the write has to stay *below* this call: anything
    # above it happens once per pass. read_snapshot() above is safe only
    # because it is a pure read.
    decision = interrupt(
        {
            "kind": "propose_edit",
            "path": snapshot.path,
            "content_hash": snapshot.content_hash,
            "original": snapshot.content,
            "modified": modified,
        }
    )

    if decision != "accept":
        return {
            "status": "rejected",
            "path": snapshot.path,
            "rejected_content": modified,
            "detail": (
                "The human rejected this edit and the file is unchanged. "
                "Do not propose the same content again -- ask what they "
                "would prefer instead."
            ),
        }

    try:
        updated = workspace.apply_change(
            path=snapshot.path,
            expected_hash=expected_hash,
            modified=modified,
        )
    except WorkspaceError as error:
        return {"status": "failed", "path": snapshot.path, "detail": str(error)}

    return {
        "status": "accepted",
        "path": updated.path,
        "content_hash": updated.content_hash,
    }


@tool
def run_python(path: str):
    """Run a Python file that already exists in the workspace, for human approval.

    Write the file with create_file first, then run it by path. The human is
    shown the code and decides; nothing executes until they accept.

    It runs with no network and no writable filesystem, so it cannot install
    packages, download anything, or save its results to a file. Only the
    standard library and packages already installed can be imported. Print
    what you want to see: stdout and stderr are the only results that come
    back, and they are truncated if the script is very chatty.

    If someone selects dont run, then don try it again but stop instead.
    """
    try:
        snapshot = workspace.read_snapshot(path)
    except WorkspaceError as error:
        return {"status": "failed", "detail": str(error)}

    if not snapshot.path.endswith(".py"):
        return {
            "status": "failed",
            "path": snapshot.path,
            "detail": "Only .py files can be run.",
        }

    # The same pausing point propose_edit documents, and the same constraint:
    # everything below re-runs from the top on resume, so the read above is
    # safe only because it is pure and the sandbox call must stay under here.
    decision = interrupt(
        {
            "kind": "run_python",
            "path": snapshot.path,
            "content_hash": snapshot.content_hash,
            # No before-and-after to show. The browser reads this kind as code
            # about to run rather than as a diff against an empty file.
            "original": "",
            "modified": snapshot.content,
        }
    )

    if decision != "accept":
        return {
            "status": "rejected",
            "path": snapshot.path,
            "detail": (
                "The human refused to run this file. Nothing was executed. "
                "Ask what they would rather you did than proposing it again."
            ),
        }

    result = sandbox.run(snapshot.path)
    # A traceback, a non-zero exit and a timeout are all reported as a run that
    # happened. Calling them failures would tell the model the tool broke, when
    # what broke is the code it wrote.
    return {
        "status": "ran",
        "path": snapshot.path,
        "exit_code": result.exit_code,
        "outcome": result.outcome,
        "truncated": result.truncated,
        "seconds": round(result.seconds, 2),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


tools = [ls, read_file, create_file, propose_edit, run_python, web_search, fetch_page]
