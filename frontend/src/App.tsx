import { DiffEditor, Editor } from "@monaco-editor/react";
import { Check, Circle, FileCode2, MessageSquare, Play, Plus, RotateCcw, Save, Square, Terminal, Trash2, X } from "lucide-react";
import { SyntheticEvent, useEffect, useRef, useState } from "react";
import { AgentEvent, ChatItem, Proposal, ThreadSummary, createThread, deleteThread, fetchFile, fetchFiles, fetchModels, fetchThreadState, fetchThreads, putFile, streamDecision, streamRun } from "./api";

type RunState = "loading" | "idle" | "saving" | "streaming" | "awaiting_approval" | "stopped" | "failed";

// The chat shows the conversation as the checkpoint records it, and nothing
// else. Anything the graph never saw -- a failed request, a save conflict,
// the backend being down -- goes to the notice strip instead, so that what
// you watch and what a reload rebuilds are the same thing.
type Message = ChatItem;

// Rows that show the graph working rather than someone speaking: one line
// each, wrench-marked, sharing the agent's lane down the left.
const MACHINERY_ROLES = new Set<ChatItem["role"]>([
  "tool_call",
  "tool_result",
  "system",
]);

// The browser owns which conversation it is in, so a server restart -- or the
// hot reload that fires on every save under llm_gym/ -- cannot change it.
const THREAD_KEY = "llm-gym.thread";
const MODEL_KEY = "llm-gym.model";

function loadStoredThread(): ThreadSummary | null {
  const raw = localStorage.getItem(THREAD_KEY);
  if (raw === null) {
    return null;
  }
  try {
    return JSON.parse(raw) as ThreadSummary;
  } catch {
    return null;
  }
}

function tokenLabel(count: number): string {
  return count < 1000 ? String(count) : `${(count / 1000).toFixed(1)}k`;
}

/** How alarming a context is, given how full it is. Ollama drops the oldest
 *  messages past num_ctx silently rather than failing, so this is the only
 *  warning a run gets before its earliest turns start disappearing. */
function contextLevel(used: number, limit: number): "ok" | "warn" | "full" {
  const share = used / limit;
  if (share >= 0.9) {
    return "full";
  }
  return share >= 0.7 ? "warn" : "ok";
}

function reviewLabel(proposal: Proposal): string {
  if (proposal.kind === "run_python") {
    return "Review code to run";
  }
  return proposal.original === "" ? "Review new file" : "Review proposal";
}

/** Thread ids are unreadable, so list them by the moment they were created. */
function threadLabel(entry: ThreadSummary): string {
  if (entry.created_at === null) {
    return entry.id;
  }
  const created = new Date(entry.created_at);
  if (Number.isNaN(created.getTime())) {
    return entry.id;
  }
  return created.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function App() {
  const [path, setPath] = useState("");
  const [content, setContent] = useState("");
  const [savedContent, setSavedContent] = useState("");
  const [contentHash, setContentHash] = useState("");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [runState, setRunState] = useState<RunState>("loading");
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  // Empty until /api/models answers, which is what settles the selection too:
  // the server owns the catalogue, so the browser never invents an id.
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [contextTokens, setContextTokens] = useState(0);
  const [thread, setThread] = useState<ThreadSummary | null>(loadStoredThread);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  // Transient trouble that never reached the graph, so has no place in the
  // transcript. Cleared whenever the next attempt starts.
  const [notice, setNotice] = useState<string | null>(null);
  // Raw tokens from the agent node, shown while they arrive and thrown away
  // once the finished message lands.
  const [streaming, setStreaming] = useState("");
  const [thinking, setThinking] = useState("");
  const abortController = useRef<AbortController | null>(null);
  const streamBox = useRef<HTMLDivElement | null>(null);
  const thinkingBox = useRef<HTMLDivElement | null>(null);
  const transcript = useRef<HTMLDivElement | null>(null);
  // Whether the transcript was sitting at the bottom when the last message
  // arrived. Held in a ref rather than state because it has to be read as it
  // was *before* the new content rendered: measured afterwards it is always
  // false, and the view would never follow.
  const atBottom = useRef(true);

  useEffect(() => {
    // Follow the tail: the box is capped at about five lines, so the top
    // scrolls away rather than the newest tokens being hidden below.
    if (streamBox.current !== null) {
      streamBox.current.scrollTop = streamBox.current.scrollHeight;
    }
    if (thinkingBox.current !== null) {
      thinkingBox.current.scrollTop = thinkingBox.current.scrollHeight;
    }
  }, [streaming, thinking]);

  useEffect(() => {
    // `streaming` belongs here even though the box sits outside the
    // transcript: it appears and disappears below the pane, which changes the
    // pane's height and pushes the newest row out of view.
    if (transcript.current !== null && atBottom.current) {
      transcript.current.scrollTop = transcript.current.scrollHeight;
    }
  }, [messages, streaming, thinking]);

  function trackScroll() {
    const pane = transcript.current;
    if (pane === null) {
      return;
    }
    // A threshold rather than equality: scroll heights are fractional, and a
    // reader two pixels off the end still means "at the bottom".
    atBottom.current = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
  }

  useEffect(() => {
    void bootstrap();
  }, []);

  async function bootstrap() {
    try {
      void fetchFiles().then(setFiles);
      void fetchThreads().then(setThreads);
      // Awaited, not fired off: a run started before this lands would have no
      // model to send.
      const catalogue = await fetchModels();
      setModels(catalogue.models);
      setContextTokens(catalogue.context_tokens);
      const storedModel = localStorage.getItem(MODEL_KEY);
      setModel(
        storedModel !== null && catalogue.models.includes(storedModel)
          ? storedModel
          : catalogue.default,
      );
      showFile(await fetchFile());

      // The stored thread may still be mid-review on the server. loadThread
      // decides the run state, so it has to settle after the file arrives
      // rather than racing it.
      const stored = loadStoredThread();
      if (stored === null) {
        // A first-ever load. Minting one here rather than leaving the
        // composer disabled until the human finds their way to "+ NEW".
        await newThread();
        return;
      }
      await loadThread(stored);
    } catch (error) {
      // `make dev` starts both halves at once and Vite wins by a second or
      // so, so a page loaded in that gap gets refused by a backend still
      // booting. Say so, rather than sitting on "loading" forever.
      setNotice(`Backend not reachable: ${error}. Reload the page.`);
      setRunState("failed");
    }
  }

  /** Rebuild the chat for a thread from its checkpoint, and pick up any
   *  proposal it is still waiting on. */
  async function loadThread(entry: ThreadSummary) {
    setNotice(null);
    // Otherwise a thread you had scrolled up in leaves the next one opening
    // part-way through its history.
    atBottom.current = true;
    try {
      const state = await fetchThreadState(entry.id);
      setMessages(state.chat);

      if (state.pending_proposal === null) {
        setProposal(null);
        setRunState("idle");
        return;
      }
      setProposal({
        kind: state.pending_proposal.kind,
        path: state.pending_proposal.path,
        content_hash: state.pending_proposal.content_hash,
        original: state.pending_proposal.original,
        modified: state.pending_proposal.modified,
      });
      setRunState("awaiting_approval");
    } catch (error) {
      setMessages([]);
      setNotice(`Could not load that thread: ${error}`);
      setProposal(null);
      setRunState("idle");
    }
  }

  function adoptThread(next: ThreadSummary) {
    localStorage.setItem(THREAD_KEY, JSON.stringify(next));
    setThread(next);
  }

  async function selectThread(entry: ThreadSummary) {
    if (entry.id === thread?.id || runState !== "idle") {
      return;
    }
    adoptThread(entry);
    await loadThread(entry);
  }

  async function removeThread(entry: ThreadSummary) {
    if (runState !== "idle") {
      return;
    }
    // Nothing here is recoverable, and the checkpoints are the only record
    // a conversation ever had.
    if (!window.confirm(`Delete the thread from ${threadLabel(entry)}? This cannot be undone.`)) {
      return;
    }

    setNotice(null);
    try {
      await deleteThread(entry.id);
      setThreads(await fetchThreads());
    } catch (error) {
      setNotice(`Could not delete that thread: ${error}`);
      return;
    }

    if (entry.id === thread?.id) {
      localStorage.removeItem(THREAD_KEY);
      setThread(null);
      setMessages([]);
      setProposal(null);
    }
  }

  /** Mint a thread and open it. Raises, because the two callers report a
   *  failure differently: one is a notice, the other is the page failing to
   *  start at all. */
  async function newThread() {
    const created = await createThread();
    adoptThread(created);
    // Empty, but it still has to clear whatever the last thread left up.
    await loadThread(created);
  }

  async function startThread() {
    if (runState !== "idle") {
      return;
    }
    try {
      await newThread();
    } catch (error) {
      setNotice(`Could not start a thread: ${error}`);
    }
  }

  function showFile(file: { path: string; content: string; content_hash: string }) {
    setPath(file.path);
    setContent(file.content);
    setSavedContent(file.content);
    setContentHash(file.content_hash);
  }

  async function openFile(wanted: string) {
    if (wanted === path || isDirty || runState !== "idle") {
      return;
    }
    showFile(await fetchFile(wanted));
  }

  function handleAgentEvent(event: AgentEvent) {
    if (event.type === "reasoning_delta") {
      setThinking((current) => current + event.text);
    }
    if (event.type === "agent_delta") {
      // Provisional, and deliberately kept out of the message list. Because
      // it never becomes a chat row there is nothing to reconcile later: the
      // finished message simply arrives as its own chat_item.
      setStreaming((current) => current + event.text);
    }
    if (event.type === "chat_item") {
      // The node finished, so whatever the box was showing is now superseded
      // by this. Cleared every time round the agent -> tools loop, not once
      // at the end of the run.
      setStreaming("");
      setThinking("");
      const { type: _type, ...item } = event;
      // The system prompt opens a thread, but it reaches us after the browser
      // has already put the user's own row up, so appending would invert them.
      setMessages((current) =>
        item.role === "system_prompt" ? [item, ...current] : [...current, item],
      );
    }
    if (event.type === "proposal_ready") {
      setProposal({
        kind: event.kind,
        path: event.path,
        content_hash: event.content_hash,
        original: event.original,
        modified: event.modified,
      });
    }
    if (event.type === "approval_required") {
      setRunState("awaiting_approval");
    }
    if (event.type === "agent_error") {
      // A tool that merely returned a failure is already in the transcript
      // via chat_item. This is the other kind: the server could not build an
      // event it wanted to send, which the checkpoint knows nothing about.
      setNotice(event.detail);
    }
    if (event.type === "file_saved") {
      // Follow whatever file the agent just touched, and pick up any file
      // it created.
      showFile(event);
      void fetchFiles().then(setFiles);
    }
  }

  async function startRun(event: SyntheticEvent) {
    event.preventDefault();
    await submitPrompt();
  }

  async function submitPrompt() {
    const submittedPrompt = prompt.trim();
    if (submittedPrompt.length === 0 || runState !== "idle" || content !== savedContent) {
      return;
    }
    if (thread === null) {
      return;
    }

    const controller = new AbortController();
    abortController.current = controller;
    setMessages((current) => [...current, { role: "user", text: submittedPrompt }]);
    setPrompt("");
    setNotice(null);
    setRunState("streaming");
    setProposal(null);
    setStreaming("");
    setThinking("");

    try {
      await streamRun(thread.id, model, submittedPrompt, controller.signal, handleAgentEvent);
      setRunState((current) => current === "streaming" ? "idle" : current);
      // This run may be what wrote the thread's first checkpoint, which is
      // what makes it appear in the list at all.
      void fetchThreads().then(setThreads);
    } catch (error) {
      if (controller.signal.aborted) {
        // The status pill already reads "stopped".
        setRunState("stopped");
        setProposal(null);
        return;
      }
      setNotice(`The run failed: ${error}`);
      setRunState("failed");
    } finally {
      // Half-written tokens were never committed to the transcript, so a run
      // that stopped or failed leaves nothing behind.
      setStreaming("");
      setThinking("");
    }
  }

  function stopRun() {
    abortController.current!.abort();
  }

  async function saveFile() {
    if (content === savedContent || runState !== "idle") {
      return;
    }

    setNotice(null);
    setRunState("saving");
    try {
      // No chat line: your save never reaches the graph, so it would vanish
      // on reload. The Save button going quiet is the confirmation.
      showFile(await putFile(path, contentHash, content));
      setRunState("idle");
    } catch (error) {
      // Usually the hash conflict from workspace.apply_change. Stay idle so
      // the edit is still there to retry or reconcile.
      setNotice(String(error));
      setRunState("idle");
    }
  }

  async function decide(decision: "accept" | "reject") {
    if (thread === null) {
      return;
    }
    const controller = new AbortController();
    abortController.current = controller;
    // No line announcing the decision: the resumed tool returns its own
    // "accepted"/"rejected" result, which is what lands in the checkpoint
    // and comes back through the projection a moment later.
    setProposal(null);
    setNotice(null);
    setRunState("streaming");

    // Resuming the paused run: the agent sees the decision as a tool result
    // and may keep going, so this streams just like starting a run does.
    try {
      await streamDecision(thread.id, model, decision, controller.signal, handleAgentEvent);
      setRunState((current) => (current === "streaming" ? "idle" : current));
    } catch (error) {
      if (controller.signal.aborted) {
        setRunState("stopped");
        return;
      }
      setNotice(`Could not submit the decision: ${error}`);
      setRunState("failed");
    } finally {
      setStreaming("");
      setThinking("");
    }
  }

  function resetRun() {
    setNotice(null);
    setStreaming("");
    setThinking("");
    setRunState("idle");
    setProposal(null);
  }

  const isRunning = runState === "streaming";
  const isReviewing = runState === "awaiting_approval" && proposal !== null;
  const isRunReview = isReviewing && proposal !== null && proposal.kind === "run_python";
  const isDirty = content !== savedContent;
  // Every call re-sends the whole conversation, so the newest reply's count is
  // how full the context is now. Rows the model did not produce carry none,
  // and neither do replies checkpointed before usage was recorded -- those say
  // so rather than reading as an empty context.
  const lastModelRow = [...messages]
    .reverse()
    .find((item) => item.role === "agent" || item.role === "tool_call");
  const usedTokens = lastModelRow?.input_tokens ?? null;
  // A brand new thread has no checkpoint yet, so the server cannot list it.
  // Show it anyway -- it is the one you are talking to.
  const threadList =
    thread !== null && !threads.some((entry) => entry.id === thread.id)
      ? [thread, ...threads]
      : threads;

  useEffect(() => {
    function handleSaveShortcut(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void saveFile();
      }
    }

    window.addEventListener("keydown", handleSaveShortcut);
    return () => window.removeEventListener("keydown", handleSaveShortcut);
  }, [content, contentHash, path, runState, savedContent]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">LG</span>
          <div>
            <strong>LLM Gym</strong>
            <span>Agent workbench</span>
          </div>
        </div>
        <div className="topbar-right">
          {lastModelRow !== undefined && contextTokens > 0 && (
            <div
              className={`context context-${usedTokens === null ? "unknown" : contextLevel(usedTokens, contextTokens)}`}
              title="How much of the context window the last reply used. Past it, the oldest messages are dropped."
            >
              context {usedTokens === null ? "missing" : tokenLabel(usedTokens)} / {tokenLabel(contextTokens)}
            </div>
          )}
          <div className={`status status-${runState}`}>
            <Circle size={8} fill="currentColor" />
            {runState.replace("_", " ")}
          </div>
        </div>
      </header>

      <section className="workspace">
        <nav className="file-pane">
          <div className="pane-header">
            <strong>Files</strong>
            <span>{files.length}</span>
          </div>
          {isDirty && (
            <p className="pane-hint">
              Save <strong>{path}</strong> before opening another file.
            </p>
          )}
          <ul className="pane-list">
            {files.map((entry) => (
              <li key={entry}>
                <button
                  type="button"
                  className={`file-item${entry === path ? " file-item-active" : ""}`}
                  onClick={() => void openFile(entry)}
                  disabled={isDirty || runState !== "idle"}
                  title={isDirty ? "Save your changes before switching files" : entry}
                >
                  <FileCode2 size={13} />
                  <span>{entry}</span>
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <section className="editor-pane">
          <div className="pane-header">
            <div className="file-name">
              <FileCode2 size={16} />
              {/* While reviewing, name the file the proposal touches -- it is
                  not necessarily the one currently open in the editor. */}
              {isReviewing ? proposal.path : path || "Loading file..."}
            </div>
            {isReviewing ? (
              <span>{reviewLabel(proposal)}</span>
            ) : (
              <button
                className={`button-save${isDirty ? " button-save-dirty" : ""}`}
                type="button"
                onClick={saveFile}
                disabled={!isDirty || runState !== "idle"}
                title={isDirty ? "Save file (Ctrl+S)" : "No unsaved changes"}
              >
                <Save size={14} />
                {runState === "saving" ? "Saving" : isDirty ? "Save" : "Saved"}
              </button>
            )}
          </div>
          {notice !== null && (
            <div className="notice">
              <span>{notice}</span>
              <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">
                <X size={13} />
              </button>
            </div>
          )}
          <div className="editor-surface">
            {isRunReview ? (
              <Editor
                language="python"
                value={proposal!.modified}
                theme="vs-dark"
                options={{
                  automaticLayout: true,
                  fontSize: 14,
                  minimap: { enabled: false },
                  padding: { top: 16 },
                  readOnly: true,
                  scrollBeyondLastLine: false,
                }}
              />
            ) : isReviewing ? (
              <DiffEditor
                language="python"
                original={proposal.original}
                modified={proposal.modified}
                theme="vs-dark"
                options={{ automaticLayout: true, readOnly: true, renderSideBySide: true }}
              />
            ) : (
              <Editor
                language="python"
                value={content}
                onChange={(value) => setContent(value!)}
                theme="vs-dark"
                options={{
                  automaticLayout: true,
                  fontSize: 14,
                  minimap: { enabled: false },
                  padding: { top: 16 },
                  readOnly: runState !== "idle",
                  scrollBeyondLastLine: false,
                }}
              />
            )}
          </div>
          {isReviewing && (
            <div className={`review-bar${isRunReview ? " review-bar-run" : ""}`}>
              <span>
                {isRunReview
                  ? `Run ${proposal.path}? It gets no network and nothing writable, and is killed if it runs long.`
                  : "One file change requires your decision."}
              </span>
              <div className="review-actions">
                <button className="button-secondary" onClick={() => decide("reject")}>
                  <X size={16} />
                  {isRunReview ? "Don't run" : "Reject"}
                </button>
                <button className="button-primary" onClick={() => decide("accept")}>
                  {isRunReview ? <Terminal size={16} /> : <Check size={16} />}
                  {isRunReview ? "Run" : "Accept"}
                </button>
              </div>
            </div>
          )}
        </section>

        <aside className="chat-pane">
          <div className="pane-header">
            <strong>Conversation</strong>
            <span>Agent</span>
          </div>
          <div className="messages" ref={transcript} onScroll={trackScroll}>
            {messages.map((message, index) => {
              if (message.role === "reasoning") {
                return (
                  <details className="message message-reasoning" key={`reasoning-${index}`}>
                    <summary>
                      <span>
                        thinking
                        {message.model && <em className="message-model">{message.model}</em>}
                      </span>
                      <p>{message.text}</p>
                    </summary>
                    <pre>{message.detail}</pre>
                  </details>
                );
              }
              const machinery = MACHINERY_ROLES.has(message.role);
              return (
                <article
                  // detail is set only where the server clipped the text, so
                  // the help cursor never promises more than there is.
                  className={[
                    "message",
                    `message-${message.role}`,
                    machinery ? "message-machinery" : "",
                    message.detail ? "message-detailed" : "",
                  ].join(" ")}
                  key={`${message.role}-${index}`}
                  title={message.detail ?? undefined}
                >
                  <span>
                    {machinery ? "🔧" : message.role}
                    {message.model && <em className="message-model">{message.model}</em>}
                  </span>
                  <p>{message.text}</p>
                </article>
              );
            })}
          </div>
          {thinking !== "" && (
            <div className="stream-box stream-box-thinking" ref={thinkingBox}>
              <span>Thinking</span>
              <pre>{thinking}</pre>
            </div>
          )}
          {streaming !== "" && (
            <div className="stream-box" ref={streamBox}>
              <span>Answer</span>
              <pre>{streaming}</pre>
            </div>
          )}
          <form className="composer" onSubmit={startRun}>
            <textarea
              aria-label="Message"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitPrompt();
                }
              }}
              placeholder={thread === null ? "Start a thread to send a message." : "Ask for a change. Enter to send, Shift+Enter for a new line."}
              disabled={isRunning || isReviewing || thread === null}
              rows={3}
            />
            <div className="composer-footer">
              {/* Enabled while a proposal is under review, unlike everything
                  else here: handing another model the decision is the point. */}
              <select
                aria-label="Model"
                className="model-picker"
                value={model}
                onChange={(event) => {
                  setModel(event.target.value);
                  localStorage.setItem(MODEL_KEY, event.target.value);
                }}
                disabled={models.length === 0 || isRunning}
              >
                {models.map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
              {isRunning ? (
                <button className="button-stop" type="button" onClick={stopRun} title="Stop run"><Square size={15} />Stop</button>
              ) : runState === "failed" || runState === "stopped" ? (
                <button className="button-secondary" type="button" onClick={resetRun}><RotateCcw size={15} />Reset</button>
              ) : (
                <button className="button-primary" type="submit" disabled={runState !== "idle" || isDirty || thread === null}><Play size={15} fill="currentColor" />Send</button>
              )}
            </div>
          </form>
        </aside>

        <nav className="thread-pane">
          <div className="pane-header">
            <strong>Threads</strong>
            <button
              type="button"
              className="button-new-thread"
              onClick={() => void startThread()}
              disabled={runState !== "idle"}
              title="Start a new thread"
            >
              <Plus size={14} />
              New
            </button>
          </div>
          {thread === null && (
            <p className="pane-hint">
              No thread yet. Press <strong>New</strong> to start one.
            </p>
          )}
          <ul className="pane-list">
            {threadList.map((entry) => (
              <li
                key={entry.id}
                className={`thread-row${entry.id === thread?.id ? " thread-row-active" : ""}`}
              >
                <button
                  type="button"
                  className={`thread-item${entry.id === thread?.id ? " thread-item-active" : ""}`}
                  onClick={() => void selectThread(entry)}
                  disabled={runState !== "idle"}
                  title={runState === "idle" ? entry.id : "Finish the current run before switching threads"}
                >
                  <MessageSquare size={13} />
                  <span>{threadLabel(entry)}</span>
                </button>
                <button
                  type="button"
                  className="thread-delete"
                  onClick={() => void removeThread(entry)}
                  disabled={runState !== "idle"}
                  title="Delete this thread"
                  aria-label={`Delete the thread from ${threadLabel(entry)}`}
                >
                  <Trash2 size={12} />
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </section>
    </main>
  );
}

export default App;