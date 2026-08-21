export type AgentEvent =
  | { type: "agent_delta"; text: string }
  // A finished line of transcript, rendered server-side by the same
  // projection that rebuilds a thread on reload.
  | {
      type: "chat_item";
      role: ChatItem["role"];
      text: string;
      detail: string | null;
      model: string | null;
    }
  | {
      type: "proposal_ready";
      path: string;
      content_hash: string;
      original: string;
      modified: string;
    }
  | { type: "approval_required" }
  | {
      type: "file_saved";
      path: string;
      content: string;
      content_hash: string;
    }
  | { type: "agent_error"; detail: string }
  | { type: "run_finished" };

export interface FileSnapshot {
  path: string;
  content: string;
  content_hash: string;
}

export interface Proposal {
  path: string;
  contentHash: string;
  original: string;
  modified: string;
}

/** A conversation. `created_at` is ISO-8601 UTC, or null for an id the
 *  server did not mint and therefore cannot date. */
export interface ThreadSummary {
  id: string;
  created_at: string | null;
}

/** One line of transcript. `detail` carries the full body behind a shortened
 *  row -- so far only the system prompt uses it. `model` is null on rows no
 *  model produced. */
export interface ChatItem {
  role: "system_prompt" | "user" | "agent" | "tool_call" | "tool_result" | "system";
  text: string;
  detail?: string | null;
  model?: string | null;
}

/** The models the picker offers, and which one it opens on. */
export interface ModelList {
  models: string[];
  default: string;
}

export interface ThreadState {
  chat: ChatItem[];
  pending_proposal: {
    path: string;
    content_hash: string;
    original: string;
    modified: string;
  } | null;
}

/** Rebuild a thread from its checkpoint. For loading and switching threads;
 *  a running thread is fed by the stream instead. */
export async function fetchThreadState(threadId: string): Promise<ThreadState> {
  const query = `?thread_id=${encodeURIComponent(threadId)}`;
  const response = await fetch(`/api/state${query}`);
  if (!response.ok) {
    throw new Error(`Unable to load thread: ${response.status}`);
  }
  return response.json() as Promise<ThreadState>;
}

export async function fetchThreads(): Promise<ThreadSummary[]> {
  const response = await fetch("/api/threads");
  if (!response.ok) {
    throw new Error(`Unable to list threads: ${response.status}`);
  }
  return response.json() as Promise<ThreadSummary[]>;
}

/** Discard a thread and its checkpoints. Permanent, and does not touch any
 *  file the agent wrote. */
export async function deleteThread(threadId: string): Promise<void> {
  const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Unable to delete thread: ${response.status}`);
  }
}

/** Mint a new thread. The server stores nothing until the first run, so the
 *  browser is what keeps this id alive. */
export async function createThread(): Promise<ThreadSummary> {
  const response = await fetch("/api/threads", { method: "POST" });
  if (!response.ok) {
    throw new Error(`Unable to create thread: ${response.status}`);
  }
  return response.json() as Promise<ThreadSummary>;
}

export async function fetchModels(): Promise<ModelList> {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error(`Unable to list models: ${response.status}`);
  }
  return response.json() as Promise<ModelList>;
}

export async function fetchFiles(): Promise<string[]> {
  const response = await fetch("/api/files");
  if (!response.ok) {
    throw new Error(`Unable to list files: ${response.status}`);
  }
  return response.json() as Promise<string[]>;
}

/** Load a file. Omit `path` to get the workspace's default file. */
export async function fetchFile(path?: string): Promise<FileSnapshot> {
  const query = path === undefined ? "" : `?path=${encodeURIComponent(path)}`;
  const response = await fetch(`/api/file${query}`);
  if (!response.ok) {
    throw new Error(`Unable to load file: ${response.status}`);
  }
  return response.json() as Promise<FileSnapshot>;
}

export async function putFile(
  path: string,
  expectedHash: string,
  content: string,
): Promise<FileSnapshot> {
  const response = await fetch("/api/file", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, expected_hash: expectedHash, content }),
  });
  if (!response.ok) {
    throw new Error(`Unable to save file: ${response.status}`);
  }
  return response.json() as Promise<FileSnapshot>;
}

/** Read an SSE body, handing each parsed event to onEvent as it arrives. */
async function readEventStream(
  response: Response,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  if (!response.ok || response.body === null) {
    throw new Error(`Stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const result = await reader.read();
    if (result.done) {
      break;
    }
    buffer += decoder.decode(result.value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop()!;
    for (const frame of frames) {
      const data = frame.split("\n").find((line) => line.startsWith("data: "))!;
      onEvent(JSON.parse(data.slice(6)) as AgentEvent);
    }
  }
}

export async function streamRun(
  threadId: string,
  model: string,
  prompt: string,
  signal: AbortSignal,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const response = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, model, prompt }),
    signal,
  });
  await readEventStream(response, onEvent);
}

/**
 * Submit the human's decision. This resumes the paused agent run, so the
 * agent can keep talking afterwards -- hence a stream, not a single reply.
 * `model` is whichever is selected now, which need not be the one that
 * proposed the edit.
 */
export async function streamDecision(
  threadId: string,
  model: string,
  decision: "accept" | "reject",
  signal: AbortSignal,
  onEvent: (event: AgentEvent) => void,
): Promise<void> {
  const response = await fetch("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, model, decision }),
    signal,
  });
  await readEventStream(response, onEvent);
}