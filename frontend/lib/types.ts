// Mirrors backend/app/schemas.py field for field.

export type Severity = "error" | "warning" | "info";
export type Tier = "verified" | "probed" | "advisory";
export type Impact = "crash" | "silent" | "risk";
export type VerifyStatus = "resolved" | "unresolved" | "inconclusive" | "not_attempted";
export type Category =
  | "syntax" | "python" | "data" | "model" | "pretrained" | "loss" | "optimization" | "training_loop" | "evaluation"
  | "device" | "checkpointing" | "other";

export interface AnalyzeRequest {
  code: string;
  goal: string;
  input_shape: number[] | null;
  answers: Record<string, string>;
  // The log's memory: fixes already applied (oldest first) and the previous check's unfixed findings.
  history: { title: string; rule_id: string; check: number; line: number; before: string[]; after: string[] }[];
  previous_open: { rule_id: string; title: string; line: number }[];
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
  category: Category;
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

export interface GoalProfile {
  task: string;
  task_name: string;
  num_classes: number | null;
  modality: string;
  notes: string;
}

export type Phase = "parse" | "context" | "baseline" | "plan" | "review" | "repair" | "verify" | "report";

export type AgentEvent =
  | { type: "step"; phase: Phase; message: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown>; reason: string }
  | { type: "tool_result"; tool: string; ok: boolean; summary: string; finding_ids: string[] }
  | { type: "goal"; profile: GoalProfile; inferred: boolean; summary: string }
  | { type: "agent"; kind: "thought"; text: string }
  | { type: "report"; text: string; mode: "llm" | "fallback" | "offline" }
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
