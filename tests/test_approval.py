import importlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from llm_gym.snapshots import SnapshotStore
from llm_gym.workspace import Workspace


class ApprovalTests(IsolatedAsyncioTestCase):
    def setUp(self):
        # Tool construction requires a key even though these tests never call Tavily.
        with patch.dict(os.environ, {"TAVILY_API_KEY": "unused-test-key"}):
            self.graph_module = importlib.import_module("llm_gym.graph")
            self.tools_module = importlib.import_module("llm_gym.tools")

        root = Path(self.enterContext(TemporaryDirectory()))
        workspace_root = root / "workspace"
        workspace_root.mkdir()
        self.workspace = Workspace(workspace_root)
        self.original = "print('original')\n"
        self.modified = "print('approved')\n"
        self.file = workspace_root / "example.py"
        self.file.write_text(self.original, encoding="utf-8")
        self.snapshots = SnapshotStore(str(root / "snapshots.db"), self.workspace)
        self.snapshots.setup()
        self.checkpoint_path = str(root / "checkpoints.db")
        self.config = {"configurable": {"thread_id": "approval-test", "model": "scripted-model"}}
        self.model_inputs = []
        self.responses = iter(())

        model = SimpleNamespace(
            num_ctx=8192,
            reasoning=False,
            kwargs={"tools": [{"function": {"name": "propose_edit"}}]},
            astream=self._stream_model,
        )
        self.enterContext(patch.object(self.graph_module, "get_model", return_value=model))
        self.enterContext(
            patch.object(
                self.graph_module,
                "stamp",
                return_value={"commit": "test-commit", "dirty": False, "called_at": "2026-01-01T00:00:00+00:00"},
            )
        )
        self.enterContext(patch.object(self.graph_module, "snapshots", self.snapshots))
        self.enterContext(patch.object(self.tools_module, "snapshots", self.snapshots))
        self.enterContext(patch.object(self.tools_module, "workspace", self.workspace))

    async def _stream_model(self, messages):
        self.model_inputs.append(list(messages))
        yield next(self.responses)

    @asynccontextmanager
    async def _open_graph(self):
        async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as checkpointer:
            yield self.graph_module.build_graph(checkpointer)

    async def _pause_edit(self, graph, expected_hash):
        self.responses = iter(
            [
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "propose_edit",
                            "args": {
                                "path": "example.py",
                                "expected_hash": expected_hash,
                                "modified": self.modified,
                            },
                            "id": "edit-call",
                            "type": "tool_call",
                        }
                    ],
                    response_metadata={"model": "scripted-model"},
                ),
                AIMessageChunk(content="Decision received.", response_metadata={"model": "scripted-model"}),
            ]
        )
        before = self.workspace.capture()
        snapshot_id = self.snapshots.capture("approval-test")
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Edit example.py", response_metadata={"workspace_snapshot_id": snapshot_id})
                ]
            },
            self.config,
        )
        state = await graph.aget_state(self.config)
        self.assertEqual(len(state.interrupts), 1)
        proposal = state.interrupts[0].value
        self.assertEqual(proposal["kind"], "propose_edit")
        self.assertEqual(proposal["original"], before["example.py"].decode("utf-8"))
        self.assertEqual(proposal["modified"], self.modified)
        self.assertEqual(self.workspace.capture(), before)
        self.assertEqual(len(self.model_inputs), 1)
        return proposal

    async def _resume(self, graph, proposal, decision):
        result = await graph.ainvoke(
            Command(resume={"decision": decision, "workspace_snapshot_id": proposal["workspace_snapshot_id"]}),
            self.config,
        )
        state = await graph.aget_state(self.config)
        self.assertEqual(state.interrupts, ())
        self.assertEqual(state.next, ())
        tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "edit-call")
        return tool_messages[0]

    async def test_rejection_preserves_file_and_returns_decision_to_model(self):
        expected_hash = self.workspace.read_snapshot("example.py").content_hash
        async with self._open_graph() as graph:
            proposal = await self._pause_edit(graph, expected_hash)
            message = await self._resume(graph, proposal, "reject")

        self.assertEqual(self.file.read_text(encoding="utf-8"), self.original)
        result = json.loads(message.content)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["rejected_content"], self.modified)
        self.assertEqual(len(self.model_inputs), 2)
        self.assertEqual(self.model_inputs[1][-1].content, message.content)

    async def test_acceptance_writes_proposed_contents_and_records_snapshot(self):
        expected_hash = self.workspace.read_snapshot("example.py").content_hash
        async with self._open_graph() as graph:
            proposal = await self._pause_edit(graph, expected_hash)
            message = await self._resume(graph, proposal, "accept")

        self.assertEqual(self.file.read_text(encoding="utf-8"), self.modified)
        self.assertEqual(json.loads(message.content)["status"], "accepted")
        before = self.snapshots.load("approval-test", message.response_metadata["workspace_before_snapshot_id"])
        after = self.snapshots.load("approval-test", message.response_metadata["workspace_snapshot_id"])
        self.assertEqual(before["example.py"], self.original.encode("utf-8"))
        self.assertEqual(after["example.py"], self.modified.encode("utf-8"))
        self.assertEqual(len(self.model_inputs), 2)

    async def test_stale_hash_preserves_newer_contents(self):
        stale_hash = self.workspace.read_snapshot("example.py").content_hash
        external_content = "print('external edit')\n"
        self.file.write_text(external_content, encoding="utf-8")
        async with self._open_graph() as graph:
            proposal = await self._pause_edit(graph, stale_hash)
            message = await self._resume(graph, proposal, "accept")

        self.assertEqual(self.file.read_text(encoding="utf-8"), external_content)
        self.assertEqual(json.loads(message.content)["status"], "failed")
        self.assertIn("changed on disk", json.loads(message.content)["detail"])
        self.assertEqual(len(self.model_inputs), 2)

    async def _assert_pending_change_refused(self, changed_file):
        expected_hash = self.workspace.read_snapshot("example.py").content_hash
        async with self._open_graph() as graph:
            proposal = await self._pause_edit(graph, expected_hash)
            changed_file.write_text("external change\n", encoding="utf-8")
            before_resume = self.workspace.capture()
            message = await self._resume(graph, proposal, "accept")

        self.assertEqual(self.workspace.capture(), before_resume)
        self.assertEqual(json.loads(message.content)["status"], "failed")
        self.assertTrue(message.response_metadata["workspace_changed"])
        self.assertEqual(len(self.model_inputs), 1)

    async def test_target_changed_while_approval_pending_prevents_write(self):
        await self._assert_pending_change_refused(self.file)

    async def test_unrelated_file_changed_while_approval_pending_prevents_write(self):
        other_file = self.workspace.root / "other.txt"
        other_file.write_text("original\n", encoding="utf-8")
        await self._assert_pending_change_refused(other_file)

    async def test_approval_survives_reopening_sqlite_and_rebuilding_graph(self):
        expected_hash = self.workspace.read_snapshot("example.py").content_hash
        async with self._open_graph() as graph:
            proposal = await self._pause_edit(graph, expected_hash)

        async with self._open_graph() as restarted_graph:
            state = await restarted_graph.aget_state(self.config)
            self.assertEqual(len(state.interrupts), 1)
            self.assertEqual(state.interrupts[0].value, proposal)
            self.assertEqual(self.file.read_text(encoding="utf-8"), self.original)
            message = await self._resume(restarted_graph, state.interrupts[0].value, "accept")

        self.assertEqual(json.loads(message.content)["status"], "accepted")
        self.assertEqual(self.file.read_text(encoding="utf-8"), self.modified)
        self.assertEqual(len(self.model_inputs), 2)
