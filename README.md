# llm-gym

A LangGraph agent that edits files under human approval. It runs against local
models through Ollama, so nothing leaves the machine.

![Reviewing a file the agent proposed](docs/approval.png)

Nothing reaches disk until you accept the diff.

## Notable

- A paused approval is a checkpoint, so a server restart mid-decision loses nothing.
- `read_file` returns a content hash that `propose_edit` must hand back, so an edit
  built against a stale version is refused instead of overwriting your work.
- Every agent-supplied path goes through one `resolve()`, confining the agent to
  `test_workspace/`.
- Threads live in SQLite and the browser owns the thread id.
- Any local Ollama model, switchable per message, and each chat line records which
  one wrote it.
- The top bar shows how full the context window is. Ollama drops the oldest
  messages past it silently, so without that the failure looks like the model
  getting worse.

The status pill in the top right is the graph's state.

| `IDLE` | `STREAMING` |
|---|---|
| ![Idle](docs/idle.png) | ![Streaming](docs/streaming.png) |

## Running it

Needs [Ollama](https://ollama.com) running, the models listed in
`llm_gym/models.py` pulled, and a [Tavily](https://tavily.com) API key for the
search tools. The key is read at import, so the server will not start without
it.

```
export TAVILY_API_KEY=tvly-...
ollama pull qwen3.5:4b
make install
make dev
```

Then http://localhost:5173.

The context size is configured by `CONTEXT_TOKENS` in `llm_gym/config.py` and
sent to Ollama as `num_ctx` on every model request. The gauge in the top bar
shows how close the latest request came to that limit.

## Not built yet

- BC / RL
- Target task: read a CV, find matching jobs, filter on location, skill overlap and
  seniority, and output a table of links with no duplicates
