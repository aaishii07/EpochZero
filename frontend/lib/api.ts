import type { AgentEvent, AnalyzeRequest, Example, Patch } from "./types";

// Set NEXT_PUBLIC_API_URL to the deployed backend (see .env.example). An empty value falls back to the local one.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export interface Health {
  status: string;
  llm: "live" | "mock";
  sandbox: "ready" | "starting";
}

// Free hosting sleeps when idle; the first request wakes it. Resolves with the health payload, or null on failure.
export async function checkHealth(timeoutMs = 8000): Promise<Health | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}/health`, { signal: controller.signal, cache: "no-store" });
    return res.ok ? ((await res.json()) as Health) : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// The backend answered, but with an error (for example 429 when the per-visitor limit is reached).
export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(detail || `Backend returned ${status}`);
  }
}

// Streams NDJSON events from POST /analyze, calling onEvent once per line.
export async function analyze(
  request: AnalyzeRequest,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    let detail = text.slice(0, 200);
    try {
      detail = String(JSON.parse(text).detail ?? detail);
    } catch {
      // not JSON: keep the raw text
    }
    throw new ApiError(res.status, detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line) as AgentEvent);
      newline = buffer.indexOf("\n");
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as AgentEvent);
}

export interface VerifyResult {
  accepted: string[];
  rejected: { id: string; reason: string }[];
  summary: string;
}

// Checks fixes together on the backend: each is kept only if the combined script gets no worse.
export async function verifyFixes(body: {
  code: string;
  goal: string;
  input_shape: number[] | null;
  answers: Record<string, string>;
  patches: { id: string; patch: Patch }[];
  applied_ids: string[];
}): Promise<VerifyResult> {
  const res = await fetch(`${API_URL}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Backend returned ${res.status}`);
  return res.json();
}

export async function fetchExamples(): Promise<Example[]> {
  const res = await fetch(`${API_URL}/examples`);
  if (!res.ok) return [];
  return res.json();
}
