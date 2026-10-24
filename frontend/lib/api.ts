import type { AgentEvent, AnalyzeRequest, Example, Patch } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
    const detail = await res.text().catch(() => "");
    throw new Error(`Backend returned ${res.status}${detail ? `: ${detail.slice(0, 200)}` : ""}`);
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
