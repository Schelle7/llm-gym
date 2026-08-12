from langchain_core.tools import tool
from langgraph.types import interrupt

from config import WORKSPACE_FILE, WORKSPACE_ROOT
from workspace import Workspace


workspace = Workspace(root=WORKSPACE_ROOT, relative_path=WORKSPACE_FILE)


@tool
def ls(include_hidden: bool):
    """List the files in the workspace directory."""
    entries = [entry.name for entry in workspace.root.iterdir()]
    if not include_hidden:
        entries = [entry for entry in entries if not entry.startswith(".")]
    return {"entries": sorted(entries)}


@tool
def read_file():
    """Read the workspace file. Returns its path, content, and content hash.

    The content_hash must be passed back when proposing an edit, so the edit
    can be rejected if the file changed in the meantime.
    """
    snapshot = workspace.read_snapshot()
    return {
        "path": snapshot.path,
        "content": snapshot.content,
        "content_hash": snapshot.content_hash,
    }


@tool
def propose_edit(expected_hash: str, modified: str):
    """Propose replacing the workspace file with new content, for human approval.

    Pass the complete new file content as `modified` -- not a diff, not a
    fragment. Pass the `content_hash` you got from read_file as
    `expected_hash`; the edit is refused if the file changed since then.

    `modified` is written to the file verbatim, so separate lines with real
    line breaks. Writing a backslash followed by n puts those two characters
    into the file instead of starting a new line.

    The edit is shown to the human, who accepts or rejects it. Only an
    accepted edit is written to disk. The result tells you what they decided.
    """
    snapshot = workspace.read_snapshot()

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
    except (ValueError, RuntimeError) as error:
        return {"status": "failed", "path": snapshot.path, "detail": str(error)}

    return {
        "status": "accepted",
        "path": updated.path,
        "content_hash": updated.content_hash,
    }


tools = [ls, read_file, propose_edit]