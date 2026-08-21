import { DiffEditor, Editor } from "@monaco-editor/react";
import { Check, Circle, FileCode2, MessageSquare, Play, Plus, RotateCcw, Save, Square, X } from "lucide-react";
import { SyntheticEvent, useEffect, useRef, useState } from "react";
import { AgentEvent, Proposal, ThreadSummary, createThread, fetchFile, fetchFiles, fetchThreads, putFile, streamDecision, streamRun } from "./api";

type RunState = "loading" | "idle" | "saving" | "streaming" | "awaiting_approval" | "stopped" | "failed";

interface Message {
  role: "user" | "assistant" | "system";
  text: string;
}

const INITIAL_MESSAGES: Message[] = [
  {
    role: "system",
    text: "Ask for a change to run the agent through the file-read, streaming, and approval flow.",
  },
];

// The browser owns which conversation it is in, so a server restart -- or the
// hot reload that fires on every save under llm_gym/ -- cannot change it.
const THREAD_KEY = "llm-gym.thread";

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
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [runState, setRunState] = useState<RunState>("loading");
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [files, setFiles] = useState<string[]>([]);
  const [thread, setThread] = useState<ThreadSummary | null>(loadStoredThread);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const abortController = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchFiles().then(setFiles);
    fetchThreads().then(setThreads);
    fetchFile().then((file) => {
      showFile(file);
      setRunState("idle");
    });
  }, []);

  function adoptThread(next: ThreadSummary) {
    localStorage.setItem(THREAD_KEY, JSON.stringify(next));
    setThread(next);
  }

  async function startThread() {
    if (runState !== "idle") {
      return;
    }
    adoptThread(await createThread());
    setMessages(INITIAL_MESSAGES);
    setProposal(null);
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
    if (event.type === "assistant_delta") {
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last.role === "assistant") {
          return [...current.slice(0, -1), { ...last, text: last.text + event.text }];
        }
        return [...current, { role: "assistant", text: event.text }];
      });
    }
    if (event.type === "file_read") {
      setMessages((current) => [...current, { role: "system", text: `Read ${event.path}` }]);
    }
    if (event.type === "proposal_ready") {
      setProposal({
        path: event.path,
        contentHash: event.content_hash,
        original: event.original,
        modified: event.modified,
      });
    }
    if (event.type === "approval_required") {
      setRunState("awaiting_approval");
    }
    if (event.type === "agent_error") {
      setMessages((current) => [
        ...current,
        { role: "system", text: `Tool failed: ${event.detail}` },
      ]);
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
    setRunState("streaming");
    setProposal(null);

    try {
      await streamRun(thread.id, submittedPrompt, controller.signal, handleAgentEvent);
      setRunState((current) => current === "streaming" ? "idle" : current);
      // This run may be what wrote the thread's first checkpoint, which is
      // what makes it appear in the list at all.
      void fetchThreads().then(setThreads);
    } catch (error) {
      if (controller.signal.aborted) {
        setMessages((current) => [...current, { role: "system", text: "Run stopped." }]);
        setRunState("stopped");
        setProposal(null);
        return;
      }
      setMessages((current) => [...current, { role: "system", text: String(error) }]);
      setRunState("failed");
    }
  }

  function stopRun() {
    abortController.current!.abort();
  }

  async function saveFile() {
    if (content === savedContent || runState !== "idle") {
      return;
    }

    setRunState("saving");
    try {
      const file = await putFile(path, contentHash, content);
      showFile(file);
      setMessages((current) => [...current, { role: "system", text: `${file.path} saved to disk.` }]);
      setRunState("idle");
    } catch (error) {
      setMessages((current) => [...current, { role: "system", text: String(error) }]);
      setRunState("failed");
    }
  }

  async function decide(decision: "accept" | "reject") {
    if (thread === null) {
      return;
    }
    const controller = new AbortController();
    abortController.current = controller;
    setMessages((current) => [
      ...current,
      {
        role: "system",
        text: decision === "accept"
          ? "Proposal accepted and written to disk."
          : "Proposal rejected. The file was not changed.",
      },
    ]);
    setProposal(null);
    setRunState("streaming");

    // Resuming the paused run: the agent sees the decision as a tool result
    // and may keep going, so this streams just like starting a run does.
    try {
      await streamDecision(thread.id, decision, controller.signal, handleAgentEvent);
      setRunState((current) => (current === "streaming" ? "idle" : current));
    } catch (error) {
      if (controller.signal.aborted) {
        setMessages((current) => [...current, { role: "system", text: "Run stopped." }]);
        setRunState("stopped");
        return;
      }
      setMessages((current) => [...current, { role: "system", text: String(error) }]);
      setRunState("failed");
    }
  }

  function resetRun() {
    setRunState("idle");
    setProposal(null);
  }

  const isRunning = runState === "streaming";
  const isReviewing = runState === "awaiting_approval" && proposal !== null;
  const isDirty = content !== savedContent;
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
        <div className={`status status-${runState}`}>
          <Circle size={8} fill="currentColor" />
          {runState.replace("_", " ")}
        </div>
      </header>

      <section className="workspace">
        <nav className="file-pane">
          <div className="pane-header">
            <strong>Files</strong>
            <span>{files.length}</span>
          </div>
          {isDirty && (
            <p className="file-hint">
              Save <strong>{path}</strong> before opening another file.
            </p>
          )}
          <ul className="file-list">
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
              <span>{proposal.original === "" ? "Review new file" : "Review proposal"}</span>
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
          <div className="editor-surface">
            {isReviewing ? (
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
            <div className="review-bar">
              <span>One file change requires your decision.</span>
              <div className="review-actions">
                <button className="button-secondary" onClick={() => decide("reject")}><X size={16} />Reject</button>
                <button className="button-primary" onClick={() => decide("accept")}><Check size={16} />Accept</button>
              </div>
            </div>
          )}
        </section>

        <aside className="chat-pane">
          <div className="pane-header">
            <strong>Conversation</strong>
            <span>Agent</span>
          </div>
          <div className="messages">
            {messages.map((message, index) => (
              <article className={`message message-${message.role}`} key={`${message.role}-${index}`}>
                <span>{message.role}</span>
                <p>{message.text}</p>
              </article>
            ))}
          </div>
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
              <span>Local agent</span>
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
            <p className="file-hint">
              No thread yet. Press <strong>New</strong> to start one.
            </p>
          )}
          <ul className="file-list">
            {threadList.map((entry) => (
              <li key={entry.id}>
                {/* Not selectable yet: this pane only shows what exists. */}
                <span
                  className={`thread-item${entry.id === thread?.id ? " thread-item-active" : ""}`}
                  title={entry.id}
                >
                  <MessageSquare size={13} />
                  <span>{threadLabel(entry)}</span>
                </span>
              </li>
            ))}
          </ul>
        </nav>
      </section>
    </main>
  );
}

export default App;