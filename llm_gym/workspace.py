from hashlib import sha256
from pathlib import Path

from llm_gym.server.schemas import FileSnapshot

# One crude cap on the whole workspace. Deliberately simple: if the agent
# makes a mess of many small files, that is cheap to notice and delete.
MAX_WORKSPACE_BYTES = 1_000_000


class WorkspaceError(Exception):
    """A tool asked for something the workspace will not allow."""


def _hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


class Workspace:
    """Every path the agent supplies passes through resolve() before use."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        """Turn an agent-supplied path into an absolute path inside the root.

        Rejects absolute paths and '..' up front so the model gets a clear
        error, but the guarantee comes from the resolve/relative_to pair:
        .resolve() collapses '..' AND follows symlinks, so relative_to()
        is comparing the real destination.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            raise WorkspaceError(f"Path must be relative to the workspace: {path!r}")
        if ".." in candidate.parts:
            raise WorkspaceError(f"Path may not contain '..': {path!r}")
        if ".git" in candidate.parts:
            raise WorkspaceError(f"Path may not touch the git directory: {path!r}")

        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(f"Path escapes the workspace: {path!r}") from None
        return resolved

    def relative(self, resolved: Path) -> str:
        return resolved.relative_to(self.root).as_posix()

    def _check_size(self, incoming: str, replacing: Path) -> None:
        total = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if replacing.is_file():
            total -= replacing.stat().st_size
        total += len(incoming.encode("utf-8"))
        if total > MAX_WORKSPACE_BYTES:
            raise WorkspaceError(
                f"Workspace would exceed {MAX_WORKSPACE_BYTES} bytes ({total} bytes). Delete something first."
            )

    def list_dir(self, path: str = ".", include_hidden: bool = False) -> list[str]:
        resolved = self.resolve(path)
        if not resolved.is_dir():
            raise WorkspaceError(f"Not a directory: {path!r}")
        entries = [entry.name + ("/" if entry.is_dir() else "") for entry in resolved.iterdir()]
        if not include_hidden:
            entries = [entry for entry in entries if not entry.startswith(".")]
        return sorted(entries)

    def list_files(self) -> list[str]:
        """Every visible file in the workspace, as relative posix paths."""
        return sorted(
            self.relative(found)
            for found in self.root.rglob("*")
            if found.is_file() and not any(part.startswith(".") for part in found.relative_to(self.root).parts)
        )

    def read_snapshot(self, path: str) -> FileSnapshot:
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise WorkspaceError(f"No such file: {path!r}")
        content = resolved.read_text(encoding="utf-8")
        return FileSnapshot(
            path=self.relative(resolved),
            content=content,
            content_hash=_hash(content),
        )

    def create(self, path: str, content: str) -> FileSnapshot:
        resolved = self.resolve(path)
        if resolved.exists():
            raise WorkspaceError(f"Already exists: {path!r}. Use an edit instead.")
        self._check_size(content, resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return self.read_snapshot(self.relative(resolved))

    def apply_change(self, path: str, expected_hash: str, modified: str) -> FileSnapshot:
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise WorkspaceError(f"No such file: {path!r}")

        current = self.read_snapshot(self.relative(resolved))
        if current.content_hash != expected_hash:
            raise WorkspaceError(f"File changed after it was read: {path!r}")

        self._check_size(modified, resolved)
        resolved.write_text(modified, encoding="utf-8")
        return self.read_snapshot(self.relative(resolved))
