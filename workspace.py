from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    content: str
    content_hash: str


class Workspace:
    def __init__(self, root: Path, relative_path: Path) -> None:
        self.root = root.resolve()
        self.relative_path = relative_path
        self.file_path = (self.root / relative_path).resolve()
        self.file_path.relative_to(self.root)

    def read_snapshot(self) -> FileSnapshot:
        content = self.file_path.read_text(encoding="utf-8")
        return FileSnapshot(
            path=self.relative_path.as_posix(),
            content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )

    def apply_change(self, path: str, expected_hash: str, modified: str) -> FileSnapshot:
        if path != self.relative_path.as_posix():
            raise ValueError(f"Unsupported workspace path: {path}")

        current = self.read_snapshot()
        if current.content_hash != expected_hash:
            raise RuntimeError(f"File changed after proposal generation: {path}")

        self.file_path.write_text(modified, encoding="utf-8")
        return self.read_snapshot()
