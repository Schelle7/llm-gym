# llm-gym
A LangGraph agent that edits files in `test_workspace/` under human approval,
with a FastAPI + React workbench around it. Built to make LangGraph's execution
and checkpointing legible, so favour clarity over cleverness.

## Running things

There is no usable system interpreter, and no venv is ever created. Two cases,
written differently:

- **Commands for the user to run**: their conda env is already active, so no
  prefix. `python -m uvicorn llm_gym.server.app:app --reload`
- **Commands the agent runs**: a separate shell, so prefix them.
  `conda run -n llm-gym python ...`

`make dev` runs both halves (`--jobs=2`). Vite is ready in ~150ms and uvicorn
takes a second or two, so a page loaded in that gap gets a refused connection;
reload it. `--reload --reload-dir llm_gym` means every save under `llm_gym/`
respawns the worker and kills any in-flight SSE response.

## Failure

One rule. Which half applies depends on whether the failure is fixable from
where it is raised.

**A fixable failure crashes immediately.** No `try/except` around logic errors,
no `if x is not None` before using `x`, no `hasattr`, no fallback default that
lets a broken assumption keep running. `message.name or "tool"` hides exactly
the bug it was written for. A missing attribute should raise `AttributeError`.
One config path, loaded directly: if it is wrong, crash. Never search relative,
then absolute, then CWD.

**An unfixable failure is made visible instead.** Model output, tool results,
the network, the browser. Crashing here only hides the problem behind a
truncated response, so show what actually arrived. `_tool_item` in
`llm_gym/server/history.py` always returns a row, rendering an unrecognised
tool result raw rather than dropping it. The `read_snapshot` guard in
`run_agent._stream` is the same idea: an exception escaping an SSE generator
strands the browser mid-run with no error shown at all.

Ownership is only a proxy for fixability, and two things override it. A crash
must land somewhere it can be read: inside an SSE generator it is invisible.
And be strict where data is created, tolerant where it is replayed: a
checkpoint is a permanent record, so crashing while rendering one makes a
thread that can never be opened again.

Neither half permits a program that looks like it is working when it is not.

## Conventions

**The chat shows what is in the checkpoint, and nothing else.** Anything the
graph never saw (a failed request, a save conflict, the backend being down)
goes to the notice strip. This is what keeps a live conversation and a reloaded
one identical.

**One projection, both paths.** `history.render_messages` renders both the
messages a node just returned (live, in `run_agent._stream`) and a whole
history (`GET /api/state`). It is pure: no disk, no network, no clock, which is
what makes those two renderings agree.

**Behavioural configuration lives in `config.py`**, not in default arguments.
Formatting constants inside private helpers are not configuration, and neither
are defaults in `@tool` signatures: those are how an argument is made optional
to the model, so removing them changes what the LLM can call.

**Comments explain why, never what, and one per decision beats one per
line.** A comment that restates the code, repeats a rule already written down
here, or runs longer than the code it sits above should be deleted. If the
reasoning genuinely takes a paragraph, it belongs in this file, not in the
source.

## Layout

- `llm_gym/graph.py` -- the graph: `agent` <-> `tools`, over `MessagesState`
- `llm_gym/tools.py` -- workspace tools. `propose_edit`/`create_file` call
  `interrupt()` for approval, and the write must stay *below* that call: on
  resume the function re-runs from the top.
- `llm_gym/workspace.py` -- every agent-supplied path passes `resolve()`
- `llm_gym/sandbox.py` -- `run_python`'s bubblewrap namespace. `resolve()`
  decides which file may be *named*; this decides what it can reach once
  python starts, and the two are not the same boundary.
- `llm_gym/server/run_agent.py` -- thread identity, SSE translation
- `llm_gym/server/history.py` -- messages to chat lines
- `frontend/src/App.tsx` -- the browser owns `thread_id` (localStorage), so a
  server restart cannot strand a conversation