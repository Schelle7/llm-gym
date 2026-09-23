from dataclasses import dataclass

from llm_gym.workspace import WorkspaceError


@dataclass(frozen=True)
class WorkspaceChanges:
    added: list[str]
    deleted: list[str]
    modified: list[str]

    @property
    def count(self) -> int:
        return len(self.added) + len(self.deleted) + len(self.modified)

    def summary(self) -> str:
        return (
            f"{self.count} files affected "
            f"({len(self.added)} added, {len(self.deleted)} deleted, {len(self.modified)} modified)"
        )


def compare_workspaces(before: dict[str, bytes], after: dict[str, bytes]) -> WorkspaceChanges:
    return WorkspaceChanges(
        added=sorted(after.keys() - before.keys()),
        deleted=sorted(before.keys() - after.keys()),
        modified=sorted(path for path in before.keys() & after.keys() if before[path] != after[path]),
    )


class WorkspaceChangedError(WorkspaceError):
    def __init__(self, changes: WorkspaceChanges, context: str):
        self.changes = changes
        super().__init__(f"Workspace changed {context}: {changes.summary()}.")


def require_unchanged(before: dict[str, bytes], after: dict[str, bytes], context: str) -> None:
    changes = compare_workspaces(before, after)
    if changes.count:
        raise WorkspaceChangedError(changes, context)
