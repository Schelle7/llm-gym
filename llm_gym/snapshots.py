import json
import sqlite3
from base64 import b64decode, b64encode
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from llm_gym.config import SNAPSHOT_DB, WORKSPACE_ROOT
from llm_gym.workspace import Workspace


class SnapshotStore:
    def __init__(self, db_path: str, workspace: Workspace):
        self.db_path = db_path
        self.workspace = workspace

    def setup(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS workspace_snapshots "
                "(id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, contents TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS workspace_snapshots_thread ON workspace_snapshots(thread_id)"
            )

    def save(self, thread_id: str, contents: dict[str, bytes]) -> str:
        snapshot_id = str(uuid4())
        encoded = {
            path: {"kind": "file", "content": b64encode(content).decode("ascii")} for path, content in contents.items()
        }
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "INSERT INTO workspace_snapshots VALUES (?, ?, ?)",
                (snapshot_id, thread_id, json.dumps(encoded, sort_keys=True)),
            )
        return snapshot_id

    def capture(self, thread_id: str) -> str:
        return self.save(thread_id, self.workspace.capture())

    def load(self, thread_id: str, snapshot_id: str) -> dict[str, bytes]:
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT contents FROM workspace_snapshots WHERE thread_id = ? AND id = ?",
                (thread_id, snapshot_id),
            ).fetchone()
        return {path: b64decode(entry["content"], validate=True) for path, entry in json.loads(row[0]).items()}

    def delete_thread(self, thread_id: str) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("DELETE FROM workspace_snapshots WHERE thread_id = ?", (thread_id,))


snapshots = SnapshotStore(SNAPSHOT_DB, Workspace(WORKSPACE_ROOT))
