"""Data contracts shared by every backend module. Mirrored in frontend/lib/types.ts."""

from typing import Literal, Union

from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]
Tier = Literal["verified", "probed", "advisory"]
Impact = Literal["crash", "silent", "risk"]
VerifyStatus = Literal["resolved", "unresolved", "inconclusive", "not_attempted"]
TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "regression",
    "segmentation",
    "unknown",
]
Modality = Literal["image", "text", "tabular", "audio", "unknown"]


class AnalyzeRequest(BaseModel):
    code: str
    goal: str = ""  # optional; inferred from the code when empty
    input_shape: list[int] | None = None
    answers: dict[str, str] = Field(default_factory=dict)


class GoalProfile(BaseModel):
    task: TaskType = "unknown"
    num_classes: int | None = None
    modality: Modality = "unknown"
    notes: str = ""


class Evidence(BaseModel):
    tool: str
    summary: str
    lines: list[int] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)


class DocRef(BaseModel):
    title: str
    url: str
    snippet: str


class Finding(BaseModel):
    id: str = ""
    rule_id: str
    title: str
    message: str
    severity: Severity
    tier: Tier
    impact: Impact
    line: int
    end_line: int
    evidence: list[Evidence]
    patchable: bool
    docs: DocRef | None = None


class Patch(BaseModel):
    start_line: int
    end_line: int
    replacement: str


class FixProposal(BaseModel):
    explanation: str
    patch: Patch


class Suggestion(BaseModel):
    finding: Finding
    explanation: str
    patch: Patch | None
    verify_status: VerifyStatus
    verify_detail: str
    attempts: int


class Question(BaseModel):
    id: str
    prompt: str
    options: list[str] | None = None


class SkippedCheck(BaseModel):
    check: str
    reason: str


class Coverage(BaseModel):
    checked: list[str]
    skipped: list[SkippedCheck]


# Events streamed to the frontend as NDJSON, one per line.

Phase = Literal["parse", "context", "baseline", "plan", "repair", "verify", "report"]


class StepEvent(BaseModel):
    type: Literal["step"] = "step"
    phase: Phase
    message: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict
    reason: str


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool: str
    ok: bool
    summary: str
    finding_ids: list[str] = Field(default_factory=list)


class GoalEvent(BaseModel):
    type: Literal["goal"] = "goal"
    profile: GoalProfile
    inferred: bool  # true when no goal was given and the task was inferred from the code
    summary: str  # one line, usable as a goal text


class FindingEvent(BaseModel):
    type: Literal["finding"] = "finding"
    finding: Finding


class AttemptEvent(BaseModel):
    type: Literal["attempt"] = "attempt"
    finding_id: str
    attempt: int
    status: VerifyStatus
    detail: str


class SuggestionEvent(BaseModel):
    type: Literal["suggestion"] = "suggestion"
    suggestion: Suggestion


class NeedsInputEvent(BaseModel):
    type: Literal["needs_input"] = "needs_input"
    question: Question


class CoverageEvent(BaseModel):
    type: Literal["coverage"] = "coverage"
    coverage: Coverage


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    elapsed_ms: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    line: int | None = None


Event = Union[
    StepEvent,
    ToolCallEvent,
    ToolResultEvent,
    GoalEvent,
    FindingEvent,
    AttemptEvent,
    SuggestionEvent,
    NeedsInputEvent,
    CoverageEvent,
    DoneEvent,
    ErrorEvent,
]
