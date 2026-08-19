from langchain_core.tools import tool
from langgraph.types import interrupt

from llm_gym.config import WORKSPACE_ROOT
from llm_gym.workspace import Workspace, WorkspaceError

workspace = Workspace(root=WORKSPACE_ROOT)


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


tools = [ls, read_file, create_file, propose_edit]