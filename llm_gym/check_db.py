import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Interrupt

from llm_gym.config import CHECKPOINT_DB, SNAPSHOT_DB


def snapshot_references(value: object) -> Iterator[str]:
    if isinstance(value, BaseMessage):
        yield from snapshot_references(value.response_metadata)
    elif isinstance(value, Interrupt):
        yield from snapshot_references(value.value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"workspace_snapshot_id", "workspace_before_snapshot_id"}:
                yield item
            else:
                yield from snapshot_references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from snapshot_references(item)


def main() -> int:
    print("Checking database references. Run with the server stopped.")
    with (
        closing(sqlite3.connect(Path(CHECKPOINT_DB).as_uri() + "?mode=ro", uri=True)) as checkpoints,
        closing(sqlite3.connect(Path(SNAPSHOT_DB).as_uri() + "?mode=ro", uri=True)) as snapshots,
    ):
        threads = {row[0] for row in checkpoints.execute("SELECT DISTINCT thread_id FROM checkpoints")}
        stored = set(snapshots.execute("SELECT thread_id, id FROM workspace_snapshots"))
        referenced = set()
        serializer = JsonPlusSerializer()
        for query in (
            "SELECT thread_id, type, checkpoint FROM checkpoints",
            "SELECT thread_id, type, value FROM writes",
        ):
            for thread_id, kind, payload in checkpoints.execute(query):
                value = serializer.loads_typed((kind, payload))
                referenced.update((thread_id, snapshot_id) for snapshot_id in snapshot_references(value))

    orphans = sorted((thread_id, snapshot_id) for thread_id, snapshot_id in stored if thread_id not in threads)
    missing = sorted(referenced - stored)
    for thread_id, snapshot_id in orphans:
        print(f"Orphan snapshot: thread={thread_id} snapshot={snapshot_id}")
    for thread_id, snapshot_id in missing:
        print(f"Missing snapshot: thread={thread_id} snapshot={snapshot_id}")
    if orphans or missing:
        print(f"Found {len(orphans)} orphan snapshots and {len(missing)} missing snapshots.")
        return 1
    print("OK: no orphan threads' snapshots or missing snapshot references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
