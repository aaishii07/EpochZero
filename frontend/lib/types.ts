// Mirrors backend/app/schemas.py field for field.

export type Severity = "error" | "warning" | "info";
export type Tier = "verified" | "probed" | "advisory";
export type Impact = "crash" | "silent" | "risk";
export type VerifyStatus = "resolved" | "unresolved" | "inconclusive" | "not_attempted";

export interface AnalyzeRequest {
  code: string;
  goal: string;
  input_shape: number[] | null;
  answers: Record<string, string>;
}

export interface Evidence {
  tool: string;
  summary: string;
  lines: number[];
  data: Record<string, unknown>;
}

export interface DocRef {
  title: string;
  url: string;
  snippet: string;
}

export interface Finding {
  id: string;
  rule_id: string;
  title: string;
  message: string;
  severity: Severity;
  tier: Tier;
  impact: Impact;
  line: number;
  end_line: number;
  evidence: Evidence[];
  patchable: boolean;
  docs: DocRef | null;
}

export interface Patch {
  start_line: number;
  end_line: number;
  replacement: string;
}

export interface Suggestion {
  finding: Finding;
  explanation: string;
  patch: Patch | null;
  verify_status: VerifyStatus;
  verify_detail: string;
  attempts: number;
}

export interface Question {
  id: string;
  prompt: string;
  options: string[] | null;
}

export interface SkippedCheck {
  check: string;
  reason: string;
}

export interface Coverage {
  checked: string[];
  skipped: SkippedCheck[];
}

export type Phase = "parse" | "context" | "baseline" | "plan" | "repair" | "verify" | "report";

export type AgentEvent =
  | { type: "step"; phase: Phase; message: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown>; reason: string }
  | { type: "tool_result"; tool: string; ok: boolean; summary: string; finding_ids: string[] }
  | { type: "finding"; finding: Finding }
  | { type: "attempt"; finding_id: string; attempt: number; status: VerifyStatus; detail: string }
  | { type: "suggestion"; suggestion: Suggestion }
  | { type: "needs_input"; question: Question }
  | { type: "coverage"; coverage: Coverage }
  | { type: "done"; elapsed_ms: number }
  | { type: "error"; message: string; line: number | null };

export interface Example {
  id: string;
  title: string;
  goal: string;
  code: string;
}
