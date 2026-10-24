# EpochZero

An agentic pre-training validation and repair system for PyTorch training scripts.

EpochZero orchestrates static analysis and controlled execution checks, uses their evidence to propose targeted code repairs, and reruns the relevant checks before presenting a verified patch to the developer. It does not replace linters or shape checkers. It automates the workflow between detection, diagnosis, repair, and verification, and it judges code against the developer's stated goal, which generic tools cannot do.

This README is the single source of truth for the build. Any change to scope, contracts, or design is recorded in the Decision Log (section 17) as it happens.

---

## Contents

1. Positioning and claims
2. What EpochZero catches (and what it does not)
3. How it works
4. Tech stack and model choice
5. Repository layout
6. Data contracts
7. Backend modules in detail
8. Rule catalogue
9. Agent loop specification
10. API specification
11. Frontend specification
12. Setup and bootstrap
13. Running and testing
14. Build plan (2 people, 8 hours)
15. Demo scripts
16. Working agreements
17. Decision log
18. Stretch goals and roadmap

---

## 1. Positioning and claims

**One-liner:** Linters check that your code is valid. EpochZero checks that it is right for your goal, fixes it, and proves the fix works before you spend GPU time.

**Claims we make:**
- Catches a defined set of common PyTorch training bugs before the first epoch (see section 2).
- Every patch for a rule-based finding is re-checked by the same tool that found the problem, and the result is shown as resolved, unresolved, or inconclusive.
- An LLM code reviewer also reads the whole script for problems no rule covers. Its findings are labelled Advisory and its patches Unverified, because no deterministic check proves them.
- When the correct fix is ambiguous, EpochZero asks the developer instead of guessing.
- When a check cannot run, EpochZero says so and says why.
- No change is applied without explicit human approval.

**Claims we never make:**
- "100% accurate", "eliminates compute waste", "guarantees convergence".
- That a passing forward pass means the model will train well.
- That no data leakage exists because a pattern check passed.

---

## 2. What EpochZero catches (and what it does not)

### Core (demoed)

| ID | Bug class | Detection | Tier | Impact |
|---|---|---|---|---|
| EZ001 | Tensor shape mismatch between layers | Meta-tensor forward pass with hooks | verified | crash |
| EZ002 | Softmax applied before `CrossEntropyLoss` | AST | verified | silent |
| EZ003 | Loss / output head wrong for the stated goal | Goal rules (facts vs expectations table) | verified | silent |
| EZ004 | Class count inconsistent across head, constants, class list, loss weights, goal | `check_class_count` + shape probe | verified | silent or crash |
| EZ005 | Random augmentation in validation/test transforms | AST | verified | silent |
| EZ006 | Missing `optimizer.zero_grad()` (accumulation-aware) | AST | verified | silent |
| EZ007 | `zero_grad()` placed between `backward()` and `step()` | AST | verified | silent |

### Cheap extras (included in MVP)

| ID | Bug class | Detection | Tier | Impact |
|---|---|---|---|---|
| EZ008 | `total += loss` without `.item()` (graph kept alive, memory growth) | AST | verified | risk |
| EZ009 | Validation loop without `model.eval()` or `torch.no_grad()` | AST | verified | silent |
| EZ010 | Targets cast to float and passed to `CrossEntropyLoss` | AST | verified | crash |
| EZ011 | Suspicious hyperparameters (learning rate, batch size with BatchNorm, weight decay) | Heuristics | advisory | risk |
| EZ000 | Syntax errors (all of them, not only the first) | `ast` | verified | crash |
| EZ012 | Undefined names and statements invalid where they appear | pyflakes | verified | crash |
| EZ013 | Legal but almost always mistaken patterns (redefined before use, `is` with a literal) | pyflakes | verified | risk |
| EZ100 | Anything else the LLM code reviewer finds: pretrained loading misuse, backbone output types, device handling, data pipeline, checkpointing | LLM review | advisory | any |

### Stretch (only if ahead at hour 6)

| ID | Bug class | Detection | Tier |
|---|---|---|---|
| EZ101 | NaN or inf loss within a few steps | Dry run on synthetic batch | probed |
| EZ102 | Parameters that never receive gradients | Dry run | probed |
| EZ103 | Cannot overfit a single repeated batch | Dry run | probed |

### Not caught (internal reference only)

This list is for the team. It is never shown in the application, the pitch, or any judge-facing material.

Data leakage beyond recognizable transform patterns, label noise, distribution shift, hyperparameter quality on real data, distributed / multi-GPU bugs, custom CUDA kernels, models whose forward pass branches on tensor values, and anything outside the extracted model / loss / optimizer / transform code.

### Tiers, impact, and verification status

These are three separate axes and the UI shows all three.

- **Tier** (how the problem was detected)
  - `verified`: a deterministic tool proved it (AST pattern, measured shape, table lookup).
  - `probed`: observed in a controlled run on synthetic data. Real data may behave differently.
  - `advisory`: reasoned heuristic, no proof. Never auto-patched.
- **Impact** (what happens if ignored), used for sorting: `crash` > `silent` > `risk`.
- **Verify status** (what happened to the proposed fix)
  - `resolved`: the same check was re-run on the patched code and the finding is gone with no new findings.
  - `unresolved`: the finding persists after 3 repair attempts. Shown with the last verifier message.
  - `inconclusive`: the check could not run on the patched code (sandbox error, unknown shape, timeout). Patch is shown labelled "unverified" with the reason.
  - `not_attempted`: advisory finding or finding marked not patchable.

---

## 3. How it works

### 3.1 The big picture

There is one agent. It is a Python loop (the orchestrator) that calls one LLM in three roles. Everything that detects or verifies is plain deterministic Python. No models are trained.

```
FRONTEND (Next.js)
  GoalInput, CodeEditor (Monaco), FindingsPanel, PatchDiff, AgentTrace, QuestionCard, CoveragePanel
        |  POST /analyze  {code, goal, input_shape?, answers}
        v  NDJSON event stream (one JSON object per line)
BACKEND (FastAPI)
  api/routes.py
        |
  agent/orchestrator.py  -- the loop, yields events
        |-- analysis/facts.py         code -> CodeFacts (AST, no LLM)
        |-- agent/goal_reader.py      goal text -> GoalProfile (LLM, low effort)
        |-- tools/registry.py         baseline static checks
        |-- agent/planner.py          LLM chooses next tool / asks user / finishes
        |       '-- tools/registry.py -> static rules | class count | shape probe | (dry run)
        |                                         '-- sandbox/ -> subprocess running torch on meta device
        |-- agent/fixer.py            finding + evidence + docs -> explanation + patch (LLM)
        '-- agent/verifier.py         apply patch, rerun the same tool, decide status (no LLM)
  llm/client.py  -- the only file that talks to the LLM provider (Groq)
```

The LLM decides and writes. The tools detect and prove. No finding reaches the user without tool evidence attached, except advisory findings, which are labelled as such.

### 3.2 One request, end to end

1. **Parse.** `facts.py` parses the code once with `ast` and builds `CodeFacts`: model classes, layers with arguments and line numbers, final activation, loss, optimizer, training and validation loops, transforms, class-count sources, data hints. A `SyntaxError` ends the request with an `error` event pointing to the line.
2. **Context.** `goal_reader.py` turns the free-text goal into a `GoalProfile` (task type, number of classes if stated, modality). Values are validated against a fixed enum; anything unknown becomes `null`, never invented.
3. **Baseline.** The orchestrator always runs the cheap static checks (`run_static_rules`, `check_goal_alignment`). These are deterministic and cost nothing.
4. **Investigate.** The planner receives facts, profile, baseline findings, and the tool menu. It chooses tools one at a time (shape probe, class-count check, and so on), reads each result, and decides what to do next. It can call `ask_user` when required information is missing or the evidence conflicts. Budget: 6 tool calls, 60 seconds.
5. **Prioritize.** Findings are de-duplicated (same rule, same line) and sorted by impact, then tier, then line.
6. **Repair.** For each patchable finding, the fixer writes a minimal line-range patch plus a short explanation, using the evidence and the rule's documentation snippet.
7. **Verify.** The verifier applies the patch to the original code, checks it still parses, reruns the tool(s) that produced the finding, and returns `resolved`, `unresolved`, or `inconclusive`. On `unresolved`, the verifier's message goes back to the fixer. Maximum 3 attempts.
8. **Report.** Each result streams to the frontend as a `suggestion` event, followed by a `coverage` event (what was checked, what was skipped and why) and `done`.
9. **Approve.** Clicking a finding card opens its patch as a diff (Accept, Reject, or Back to editor); the Apply fix button on the card applies it directly. Fixes can be applied one at a time without a re-check: the other fixes and markers move with the code, applied fixes can be undone, and fixes that overlap an applied one are marked. The script is re-checked automatically once every suggested fix is applied or after Apply all; otherwise press Check when ready.

### 3.3 Where the agent actually makes decisions

These are the points where the LLM's choice changes what happens next. Each one is visible in the Agent Trace panel.

| Decision | Outcomes |
|---|---|
| Which checks to run | Run the shape probe only when a shape is known or inferable, otherwise ask. Run the class-count check only when a classification or segmentation head exists. Skip checks irrelevant to the code. |
| Is the evidence sufficient? | Confirmed failure: patch. Risk only: warn. Conflicting sources: ask the user. |
| What is the minimal correct patch? | Change the component that disagrees with the majority of grounded sources, not the others. Never change more lines than needed. |
| What did verification say? | Resolved: present. Unresolved: revise with feedback, then stop after 3. Inconclusive: present as unverified with the reason. |

### 3.4 Stateless clarification

The backend keeps no session state. When the planner calls `ask_user`, the orchestrator emits a `needs_input` event with a question id and stops. The frontend shows the question, and the user's answer is sent back in `answers` on a new `/analyze` request with the same code and goal. Answers are injected into the planner context and into the class-count check as a source with the highest weight.

---

## 4. Tech stack and model choice

| Layer | Choice |
|---|---|
| Frontend | Next.js (App Router, TypeScript), Tailwind CSS, `@monaco-editor/react` |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic v2 |
| Static analysis | Python `ast` module |
| Controlled execution | PyTorch (CPU build), `meta` device, forward hooks; `torch.fx` as stretch |
| LLM | Groq API via the official `groq` Python SDK (OpenAI-compatible chat completions) |
| Tests | pytest |

### LLM configuration

| Role | Model | Effort | Output method |
|---|---|---|---|
| Goal reader | `openai/gpt-oss-20b` | `low` | JSON object mode, schema in the system prompt, validated with Pydantic, one retry with the validation error |
| Planner | `openai/gpt-oss-20b` | `low` | `chat.completions.create(tools=..., tool_choice="auto")`, tool results sent back as `role: "tool"` messages |
| Fixer | `openai/gpt-oss-120b` | `medium` | Same as the goal reader, with `FixProposal` |

Notes:
- The model is set with `EPOCHZERO_MODEL` in `backend/.env`; any Groq chat model with tool calling works. `reasoning_effort` is sent only to `openai/gpt-oss-*` models.
- Rate limits and API errors raise `LLMError`; every role then falls back to its offline version, so a run never fails because of the LLM.
- Planner prompts are kept small (facts capped at 5000 characters) to stay inside Groq rate limits.
- `EPOCHZERO_LLM=mock` swaps in canned responses so the full pipeline runs offline. This is used for tests, frontend development, and as the demo fallback.

### Why no trained models

There is no dataset of buggy scripts and correct fixes, a fine-tuned small model would reason about code worse than a frontier model, and a learned detector gives a probability rather than a proof. Accuracy comes from deterministic tools. PyTorch is used only as a measuring instrument.

---

## 5. Repository layout

```
EpochZero/
  README.md                       this file
  epochzero_idea_overview.md      original concept note
  .gitignore
  backend/
    requirements.txt
    .env                          GROQ_API_KEY, EPOCHZERO_MODEL, EPOCHZERO_LLM (gitignored)
    .env.example
    app/
      __init__.py
      main.py                     FastAPI app, CORS, router mount
      config.py                   settings from env (model id, budgets, timeouts, mock flag)
      schemas.py                  all data contracts (section 6)
      api/
        __init__.py
        routes.py                 POST /analyze, GET /health, GET /rules
      analysis/
        __init__.py
        facts.py                  code -> CodeFacts
      goals/
        __init__.py
        expectations.py           task -> expected loss / head / activation (plain dict)
      knowledge/
        __init__.py
        docs.py                   rule_id -> short PyTorch docs snippet + URL
      tools/
        __init__.py
        base.py                   Tool protocol, ToolResult
        registry.py               tool table, as_llm_tools(), run_tool()
        static/
          __init__.py
          ast_rules.py            EZ002, EZ005-EZ010
          goal_rules.py           EZ003
          hyperparams.py          EZ011
          class_count.py          EZ004 source gathering and consistency
        dynamic/
          __init__.py
          shape_probe.py          EZ001
          dry_run.py              EZ101-EZ103 (stretch)
      sandbox/
        __init__.py
        extract.py                build the safe subset of user code
        harness.py                code template executed inside the subprocess
        runner.py                 subprocess execution, timeout, JSON result
      agent/
        __init__.py
        orchestrator.py           the loop (section 9), yields events
        goal_reader.py            goal text -> GoalProfile
        planner.py                tool-use loop with budget
        fixer.py                  finding -> FixProposal
        verifier.py               patch -> VerifyStatus
        patching.py               line-range patch application and validation
        prompts.py                system prompts for the three LLM roles
      llm/
        __init__.py
        client.py                 structured(), tool_turn(), refusal handling
        mock.py                   canned responses for EPOCHZERO_LLM=mock
    tests/
      __init__.py
      test_rules.py               static rules, goal rules, class count, patching (inline snippets)
      test_demos.py               full agent loop on every demo script, offline mode
  frontend/
    package.json
    app/
      layout.tsx                  fonts, theme tokens, page shell
      page.tsx                    owns all state, wires components
      globals.css                 design tokens (section 11.3)
    components/
      TopBar.tsx
      GoalInput.tsx
      CodeEditor.tsx
      PatchDiff.tsx
      FindingsPanel.tsx
      FindingCard.tsx
      HistoryPanel.tsx            applied fixes for the active log, newest first
      Workspace.tsx               the whole app (state, logs, fixes); rendered client-side only
      AgentTrace.tsx
      QuestionCard.tsx
      CoveragePanel.tsx
      StatusBadge.tsx
    lib/
      types.ts                    mirrors backend/app/schemas.py exactly
      api.ts                      analyze(request, onEvent) NDJSON reader
      patch.ts                    applyPatch(code, patch) (same logic as backend)
      monacoTheme.ts              flat editor theme matching the design tokens
      logs.ts                     log and history storage in localStorage
  demo/
    01_shape_mismatch.py
    02_segmentation_classes_grounded.py
    02b_segmentation_classes_ambiguous.py
    03_val_leak_and_lr.py
```

### Dependency direction

`api -> agent -> {tools, analysis, goals, knowledge, llm, sandbox} -> schemas`

- Only `llm/client.py` imports `groq`.
- Only `sandbox/` executes user code.
- `tools/` never imports from `agent/`.
- `verifier.py` never calls the LLM.
- `schemas.py` imports nothing from the project.

---

## 6. Data contracts

`backend/app/schemas.py` is the contract. `frontend/lib/types.ts` mirrors it field for field. A change to either file requires both people to agree and a Decision Log entry.

```python
from typing import Literal
from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]
Tier = Literal["verified", "probed", "advisory"]
Impact = Literal["crash", "silent", "risk"]
VerifyStatus = Literal["resolved", "unresolved", "inconclusive", "not_attempted"]
TaskType = Literal[
    "binary_classification", "multiclass_classification", "multilabel_classification",
    "regression", "segmentation", "unknown",
]

class AnalyzeRequest(BaseModel):
    code: str
    goal: str = ""                               # optional; inferred from the code when empty
    input_shape: list[int] | None = None        # e.g. [1, 3, 224, 224]
    answers: dict[str, str] = {}                # question_id -> answer
    history: list[HistoryItem] = []             # fixes applied earlier in this log (title, rule, check, before, after)
    previous_open: list[OpenFinding] = []       # the previous check's unfixed findings

class GoalProfile(BaseModel):
    task: TaskType
    num_classes: int | None = None
    modality: Literal["image", "text", "tabular", "audio", "unknown"] = "unknown"
    notes: str = ""

class Evidence(BaseModel):
    tool: str                                   # e.g. "shape_probe"
    summary: str                                # one human-readable sentence
    lines: list[int] = []
    data: dict = {}                             # raw measurements

class DocRef(BaseModel):
    title: str
    url: str
    snippet: str

class Finding(BaseModel):
    id: str                                     # "f1", "f2", ... unique per analysis
    rule_id: str                                # "EZ001" ...
    title: str                                  # short, e.g. "Linear input size mismatch"
    message: str                                # deterministic, factual
    severity: Severity
    tier: Tier
    impact: Impact
    line: int
    end_line: int
    evidence: list[Evidence]
    patchable: bool
    docs: DocRef | None = None

class Patch(BaseModel):
    start_line: int                             # 1-indexed, inclusive
    end_line: int                               # inclusive
    replacement: str                            # full replacement text for those lines

class FixProposal(BaseModel):                   # what the fixer LLM returns
    explanation: str                            # at most 3 sentences
    patch: Patch

class Suggestion(BaseModel):
    finding: Finding
    explanation: str
    patch: Patch | None
    verify_status: VerifyStatus
    verify_detail: str                          # why that status, one sentence
    attempts: int

class Question(BaseModel):
    id: str                                     # stable, e.g. "num_classes"
    prompt: str
    options: list[str] | None = None

class SkippedCheck(BaseModel):
    check: str
    reason: str

class Coverage(BaseModel):
    checked: list[str]
    skipped: list[SkippedCheck]
```

### Events (NDJSON, one per line)

```python
class StepEvent(BaseModel):       type: Literal["step"]; phase: Literal["parse","context","baseline","plan","repair","verify","report"]; message: str
class ToolCallEvent(BaseModel):   type: Literal["tool_call"]; tool: str; args: dict; reason: str
class ToolResultEvent(BaseModel): type: Literal["tool_result"]; tool: str; ok: bool; summary: str; finding_ids: list[str]
class GoalEvent(BaseModel):       type: Literal["goal"]; profile: GoalProfile; inferred: bool; summary: str
class FindingEvent(BaseModel):    type: Literal["finding"]; finding: Finding
class AttemptEvent(BaseModel):    type: Literal["attempt"]; finding_id: str; attempt: int; status: VerifyStatus; detail: str
class SuggestionEvent(BaseModel): type: Literal["suggestion"]; suggestion: Suggestion
class NeedsInputEvent(BaseModel): type: Literal["needs_input"]; question: Question
class CoverageEvent(BaseModel):   type: Literal["coverage"]; coverage: Coverage
class DoneEvent(BaseModel):       type: Literal["done"]; elapsed_ms: int
class ErrorEvent(BaseModel):      type: Literal["error"]; message: str; line: int | None = None
```

Order guarantee: `step(parse)` first, then any mix of `step`, `tool_call`, `tool_result`, `finding`, `attempt`, `suggestion`, and finally exactly one of `needs_input`, `done`, or `error`. `coverage` is sent immediately before `done` or `needs_input`.

---

## 7. Backend modules in detail

### 7.1 `analysis/facts.py`

Input: code string. Output: `CodeFacts`. No LLM, no execution.

What it records (each item carries its source line):
- `imports`: aliases for `torch`, `torch.nn`, `torch.nn.functional`, `torchvision.transforms`.
- `model_classes`: classes subclassing `nn.Module` (resolved through import aliases), their `__init__` and `forward` line ranges.
- `layers`: every `self.<attr> = nn.<Type>(args)` in `__init__`, with literal argument values where they are constants, and names where they are variables. Includes layers inside `nn.Sequential(...)` with index-qualified names (`features.0`).
- `model_instantiations`: expressions like `model = Net(num_classes=5)`, so the sandbox can build the model the same way.
- `final_activation`: the last activation applied in `forward` before return (`softmax`, `sigmoid`, `relu`, or none).
- `loss`: constructor type and keyword arguments (`weight`, `ignore_index`, `pos_weight`), plus where it is called and with what arguments.
- `optimizer`: type, `lr`, `weight_decay`, `momentum`, which parameters it receives.
- `loops`: each `for`/`while` loop with the ordered method calls inside it (innermost loop owns the call), whether it looks like training (`backward` + `step`) or validation (`no_grad`, `eval`, or a loader named `val*`/`test*`), and any modulo condition guarding `step()`.
- `transforms`: each `transforms.Compose([...])` assignment, the variable name, the transform types inside, and which dataset or loader uses it.
- `class_count_sources`: every place a class count is stated (constant named like `NUM_CLASSES`/`n_classes`, `num_classes=` keyword, length of a list named like `classes`/`labels`/`CLASS_NAMES`, length of loss `weight`, final layer `out_features`/`out_channels`).
- `data_hints`: `Resize`/`CenterCrop`/`RandomResizedCrop` sizes, first layer type and input channels/features, `Embedding` presence, `batch_size` from loaders.
- `layer_lines`: map from layer qualified name to source line, used to point runtime errors at code.

### 7.2 `goals/expectations.py`

A plain dict. Example entries:

| Task | Expected loss | Head output | Final activation in model |
|---|---|---|---|
| binary_classification | `BCEWithLogitsLoss` (or `BCELoss` only with sigmoid) | 1 | none |
| multiclass_classification | `CrossEntropyLoss` | `num_classes` | none |
| multilabel_classification | `BCEWithLogitsLoss` | `num_classes` | none |
| regression | `MSELoss`, `L1Loss`, `SmoothL1Loss`, `HuberLoss` | target dimension | none |
| segmentation | `CrossEntropyLoss` (per pixel), Dice variants | `num_classes` channels, spatial size preserved | none |

### 7.3 `knowledge/docs.py`

For each rule: a short title, the relevant PyTorch documentation URL, and a two to four sentence snippet. The snippet is passed to the fixer and shown in the finding card. No retrieval system.

### 7.4 `tools/static/*`

Each rule is a function `(facts, profile) -> list[Finding]`. Rules never raise; malformed input yields no findings. Details per rule are in section 8.

### 7.5 `sandbox/`

- `extract.py`: builds a reduced script from the user's code containing only imports, constant assignments at module level, `nn.Module` class definitions, helper functions those classes call, the model instantiation expression, and loss/optimizer construction. It drops training and validation loops, `if __name__ == "__main__"` blocks, file and network I/O (`open`, `torch.load`, `requests`, `urllib`, dataset downloads), and `DataLoader` construction.
- `harness.py`: a template appended to the extracted code. It receives a JSON job (`{"kind": "shape_probe", "input_shape": [...], ...}`), runs it, and prints one JSON result to stdout.
- `worker.py`: one long-lived `python -I` process that imports torch once at backend startup and runs each job in a fresh namespace, reading jobs from stdin and writing results to stdout.
- `runner.py`: sends extracted code plus harness to the worker with a 15 second timeout and `CUDA_VISIBLE_DEVICES=""`. If a job hangs, the worker is killed and restarted; if it crashes the worker, the job is retried once in a fresh worker. Results are cached by script hash.

The sandbox is a crash and hang boundary for a demo, not a security boundary (internal note, not shown in the app).

### 7.6 `tools/dynamic/shape_probe.py`

Input shape resolution order:
1. `request.input_shape` if provided.
2. Inferred from facts: first layer `Conv2d(c, ...)` plus resize hint `s` gives `(1, c, s, s)`; `Conv2d` without resize hint gives `(1, c, 224, 224)` and is reported as an assumption; first layer `Linear(n, ...)` gives `(1, n)`; first layer `Embedding` gives `(1, 16)` of `torch.long`; `Conv1d(c, ...)` gives `(1, c, 128)`.
3. Otherwise the tool returns `needs_shape`, and the planner calls `ask_user`.

Inside the sandbox:
1. Instantiate the model using the user's instantiation expression, or a no-argument constructor. If construction fails, return `inconclusive` with the error.
2. Move the model to `device="meta"` and register forward hooks on every submodule that record input and output shapes.
3. Run one forward pass with a meta tensor of the resolved shape.
4. On success, return the per-layer shape table and the output shape. Downstream checks (class count, goal alignment) use the output shape as measured evidence.
5. On failure, the last hook entered identifies the failing module. `layer_lines` maps it to a source line. The error message is parsed for the two incompatible sizes (for example `mat1 and mat2 shapes cannot be multiplied (1x2048 and 512x10)`), and the finding reads "fc1 expects 512 input features but receives 2048".
6. If the failure is a data-dependent branch (bool of a meta tensor), return a coverage skip: "shape check stopped at line N: forward pass branches on tensor values".

### 7.7 `tools/static/class_count.py`

Gathers every class-count source from `facts.class_count_sources`, the goal profile, measured output shape (if the shape probe ran), and `answers["num_classes"]` (if the user answered). Each source is `{value, line, kind, weight}` where user answer > goal > constants and lists > loss weight > head. Returns:
- `consistent`: all sources agree. No finding.
- `grounded_mismatch`: one source disagrees with all others, and at least two independent sources agree. Finding EZ004, patchable, with the disagreeing line as the target.
- `ambiguous`: sources conflict without a clear majority, or only the head and one other source exist and they disagree. Finding EZ004, not patchable, with a suggested question for the planner.

### 7.8 `tools/registry.py`

A table of tools with name, description, JSON input schema, and function. Two entry points:
- `as_llm_tools()`: tool definitions for the planner, each with `strict: true` and `additionalProperties: false`.
- `run_tool(name, ctx, **args) -> ToolResult`: used by both the planner and the verifier.

`ToolResult` contains `ok`, `summary`, `findings`, `data`, and optionally `skipped` (for coverage).

### 7.9 `agent/patching.py`

`apply_patch(code, patch) -> str` replaces lines `start_line..end_line` with `replacement`, preserving the file's line endings. Rejects patches whose range is outside the file, whose start is after end, or that span more than 15 lines. The frontend `lib/patch.ts` implements identical logic.

### 7.10 `agent/verifier.py`

`verify(original, patch, finding, ctx) -> (VerifyStatus, detail)`:
1. Apply the patch. Invalid range: `unresolved`, "patch range invalid".
2. `ast.parse` the result. Failure: `unresolved` with the syntax error.
3. Rebuild facts for the patched code.
4. Rerun the tool(s) listed in the finding's evidence.
5. If any rerun tool reports it could not run: `inconclusive` with the reason.
6. Compare rule-id counts before and after (counts, not line numbers, because patches shift lines). Finding's rule count must drop and no rule may increase. Otherwise `unresolved` with which rule persisted or appeared.
7. Else `resolved`.

---

## 8. Rule catalogue

| ID | Name | Fires when | Does not fire when | Patchable | Verified by |
|---|---|---|---|---|---|
| EZ001 | shape-mismatch | Forward pass on meta tensors raises a shape error at a module | Shape unknown and not inferable (planner asks instead) | yes | shape_probe |
| EZ002 | double-softmax | `forward` ends in `softmax`/`nn.Softmax` (or `log_softmax`) and the loss is `CrossEntropyLoss` | Softmax is only applied under `torch.no_grad()` for inference | yes | static rules |
| EZ003 | loss-goal-mismatch | Loss type or final activation contradicts `expectations[profile.task]` | `profile.task == "unknown"` | yes | goal rules |
| EZ004 | class-count-mismatch | `class_count` returns `grounded_mismatch` or `ambiguous` | All sources agree | only if grounded | class_count, shape_probe |
| EZ005 | eval-augmentation-leak | A `Compose` used by a val/test dataset contains `Random*`, `ColorJitter`, `RandomErasing`, `AutoAugment`, `TrivialAugmentWide`, or the train transform object is reused for val/test | Val/test transforms are deterministic | yes | static rules |
| EZ006 | missing-zero-grad | Innermost training loop calls `backward` and `step` but never `zero_grad` | `step` is guarded by a modulo condition or the loss is divided by a constant (reported as info: "looks like intentional gradient accumulation") | yes | static rules |
| EZ007 | misplaced-zero-grad | `zero_grad` occurs between `backward` and `step` in the same loop | Order is zero_grad, backward, step | yes | static rules |
| EZ008 | loss-accumulation-leak | `x += loss` or `x = x + loss` where `loss` is the backward-called tensor, without `.item()` or `.detach()` | `.item()`/`.detach()` present | yes | static rules |
| EZ009 | eval-mode-missing | Validation loop lacks `torch.no_grad()`/`torch.inference_mode()`, or the code never calls `model.eval()` before it | Both present | yes | static rules |
| EZ010 | ce-float-targets | Targets passed to `CrossEntropyLoss` were cast with `.float()` in the visible code and the task is not soft-label | No cast visible | yes | static rules |
| EZ011 | suspicious-hyperparameter | Adam/AdamW `lr > 1e-2`; SGD `lr > 1.0`; `weight_decay > 0.1`; `batch_size < 2` with BatchNorm layers present | Within ranges | no (advisory) | n/a |

Each rule has a buggy and a fixed snippet in `backend/tests/test_rules.py`, plus edge cases (gradient accumulation, scheduler `step()`). The demo scripts are run end to end in `backend/tests/test_demos.py`.

---

## 9. Agent loop specification

### 9.1 Orchestrator (pseudocode)

```
yield step(parse)
facts = build_facts(code)                         # SyntaxError -> yield error, return
yield step(context)
profile = goal_reader(goal, answers)
yield step(baseline)
baseline = run_tool("run_static_rules") + run_tool("check_goal_alignment")
emit tool_result and finding events
yield step(plan)
plan_result = planner(facts, profile, baseline, answers)   # yields tool_call / tool_result / finding
if plan_result.question:
    yield coverage; yield needs_input(question); return
findings = dedupe_and_sort(baseline + plan_result.findings)
yield step(repair)
for f in findings:
    if not f.patchable: yield suggestion(f, patch=None, status=not_attempted); continue
    feedback = None
    for attempt in 1..3:
        proposal = fixer(code, goal, profile, f, feedback)
        status, detail = verifier(code, proposal.patch, f, ctx)
        yield attempt(f.id, attempt, status, detail)
        if status in (resolved, inconclusive): break
        feedback = detail
    yield suggestion(f, proposal, status, detail, attempt)
yield step(report); yield coverage; yield done
```

### 9.2 Planner tools

| Tool | Arguments | Purpose |
|---|---|---|
| `run_static_rules` | none | Rerun static rules (normally already in baseline) |
| `check_goal_alignment` | none | Goal rules |
| `probe_shapes` | `input_shape: int[] or null` | Shape probe; returns per-layer table or failure |
| `check_class_count` | none | Gather and compare class-count sources |
| `inspect_lines` | `start: int, end: int` | Return source lines so the planner can read code it is reasoning about |
| `dry_run` | `steps: int` | Stretch only; hidden from the menu until implemented |
| `ask_user` | `question_id: string, prompt: string, options: string[] or null` | Ends the investigation and asks the developer |
| `finish` | `summary: string` | Ends the investigation |

Budget: at most 6 tool calls and 60 seconds wall clock. On budget exhaustion the orchestrator proceeds to repair with what it has and records "investigation budget reached" in coverage.

### 9.3 Prompts (outline, full text in `agent/prompts.py`)

- **Goal reader:** extract task type, number of classes, and modality from the goal text. Use only what the text states. Unknown values are `null` or `"unknown"`.
- **Planner:** you are investigating a PyTorch training script for problems that would waste a training run. You see the code facts, the goal profile, and baseline findings. Choose the next tool that would most change your confidence. Do not claim a problem that no tool has shown. If information required for a check is missing, or sources disagree and the right fix would be a guess, call `ask_user`. Call `finish` when further tools would not change the outcome.
- **Fixer:** you are given one finding with evidence and a documentation snippet. Produce the smallest line-range patch that resolves it, touching only the lines that must change, preserving the user's style and variable names. Do not fix other problems. Explain in at most three sentences: what was used, why it is wrong for the stated goal, what the patch does. If the previous attempt failed, the verifier's message is included; address it directly.

### 9.4 Determinism and failure handling

- Tools are deterministic. The same code, goal, shape, and answers produce the same findings.
- LLM failures (network, refusal, validation error) are retried once, then the finding is reported with `verify_status = not_attempted` and detail "repair unavailable".
- The orchestrator never lets an exception escape: any unexpected error becomes an `error` event.

---

## 10. API specification

Base URL: `http://localhost:8000`. CORS allows `http://localhost:3000`.

### `POST /analyze`

Request body: `AnalyzeRequest` (section 6).
Response: `200`, `Content-Type: application/x-ndjson`, a stream of events (section 6).
Validation errors: `422` with the standard FastAPI body. Code larger than 200 KB: `413`.

Example:

```
POST /analyze
{"code": "...", "goal": "5-class road scene segmentation", "input_shape": null, "answers": {}}

{"type":"step","phase":"parse","message":"Parsed 84 lines, 1 model class, 1 training loop"}
{"type":"step","phase":"context","message":"Goal: segmentation, 5 classes, image"}
{"type":"tool_result","tool":"run_static_rules","ok":true,"summary":"No static issues","finding_ids":[]}
{"type":"tool_call","tool":"probe_shapes","args":{"input_shape":null},"reason":"Need measured head output to compare class counts"}
{"type":"tool_result","tool":"probe_shapes","ok":true,"summary":"Output (1, 7, 256, 256); input shape inferred from Resize(256)","finding_ids":[]}
{"type":"tool_call","tool":"check_class_count","args":{},"reason":"Head has 7 channels, goal says 5"}
{"type":"finding","finding":{...EZ004...}}
{"type":"attempt","finding_id":"f1","attempt":1,"status":"resolved","detail":"Head now outputs 5 channels, matching 4 sources"}
{"type":"suggestion","suggestion":{...}}
{"type":"coverage","coverage":{"checked":["static rules","goal alignment","shape probe","class count"],"skipped":[]}}
{"type":"done","elapsed_ms":14210}
```

### `GET /health`

Returns `{"status": "ok", "llm": "live" | "mock", "torch": "<version>"}`.

### `GET /rules`

Returns the rule catalogue (id, name, description, tier, docs URL) so the frontend can render the coverage panel and help text without hardcoding.

### `GET /examples`

Returns the scripts in `demo/` as `{id, title, goal, code}` for the Load example picker. Each demo file starts with a `# Goal: ...` line, which becomes `goal` and is stripped from `code`.

---

## 11. Frontend specification

### 11.1 Hard rules

- No gradients anywhere: not on backgrounds, buttons, borders, text, badges, scrollbars, loaders, or charts. Flat solid fills only.
- No emojis anywhere: not in UI copy, labels, placeholders, empty states, toasts, code comments, commit messages, or this README.
- No decorative glows, blurs, glassmorphism, or drop shadows beyond a single 1px border to separate surfaces.
- No icon fonts or emoji substitutes for status. Status is communicated with text labels plus a solid color dot or a solid left border.
- No animations beyond a 150ms opacity/color transition and a simple indeterminate progress bar (solid color block moving, no gradient).

### 11.2 Layout

Desktop first, minimum supported width 1280px. Below 1024px the two columns stack vertically.

```
+--------------------------------------------------------------------------------+
| TopBar: "EpochZero"                         [status text]   [Check  Ctrl+Enter] |
+-----------------------------------------------+--------------------------------+
| GoalInput                                     | Tabs: Findings | Trace | Coverage |
|   Goal  [ textarea, 2 rows               ]    |                                |
|   Input shape [ 1,3,224,224 ] (optional)      |  FindingsPanel                 |
+-----------------------------------------------+    grouped: Crash / Silent /    |
| CodeEditor (Monaco)                           |    Risk; each a FindingCard    |
|   or PatchDiff when a suggestion is open      |                                |
|                                               |  QuestionCard (when asked)     |
|                                               |  pinned at top of panel        |
|                                               |                                |
+-----------------------------------------------+--------------------------------+
  left column 62%                                 right column 38%, min 420px
```

### 11.3 Visual design tokens

Defined once in `app/globals.css` as CSS variables and mapped in Tailwind. The app uses a dark theme to match Monaco `vs-dark`.

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0E1116` | page background |
| `--surface` | `#161B22` | panels, cards, top bar |
| `--surface-raised` | `#1C2230` | hovered card, active tab |
| `--border` | `#2A313C` | all 1px borders |
| `--text` | `#E6EDF3` | primary text |
| `--text-muted` | `#8B949E` | secondary text, labels |
| `--accent` | `#4C8DFF` | primary button fill, focus ring, links |
| `--accent-text` | `#FFFFFF` | text on accent |
| `--crash` | `#F85149` | impact crash, error markers |
| `--silent` | `#D29922` | impact silent, warning markers |
| `--risk` | `#8B949E` | impact risk, advisory |
| `--resolved` | `#3FB950` | verify status resolved |
| `--unresolved` | `#F85149` | verify status unresolved |
| `--inconclusive` | `#D29922` | verify status inconclusive |

Typography (loaded with `next/font/google`):
- UI: Inter, 14px base, 1.5 line height. Headings 16px semibold. Labels 12px medium uppercase with 0.04em letter spacing.
- Code and numbers: JetBrains Mono, 13px in the editor and inline code.

Spacing: 4px base grid. Panel padding 16px. Card padding 12px. Gap between cards 8px. Border radius 6px on cards, inputs, and buttons; 4px on badges.

Buttons:
- Primary: solid `--accent` fill, `--accent-text`, no border. Hover: same color at 90% opacity. Disabled: `--surface-raised` fill with `--text-muted`.
- Secondary: transparent fill, 1px `--border`, `--text`. Hover: `--surface-raised` fill.
- Danger (Reject): transparent fill, 1px `--crash` border, `--crash` text.

Badges: 1px border in the status color, text in the status color, transparent fill. Labels are words: `Verified`, `Probed`, `Advisory`, `Resolved`, `Unresolved`, `Inconclusive`, `Not attempted`, `Crash`, `Silent`, `Risk`.

### 11.4 Components

| Component | Props / state | Behaviour |
|---|---|---|
| `page.tsx` | owns `code`, `goal`, `inputShape`, `answers`, `events[]`, `findings`, `suggestions`, `question`, `coverage`, `status` (`idle`/`analyzing`/`needs_input`/`done`/`error`), `selectedFindingId`, `editorMode` (`edit`/`diff`) | Single source of state. Derives findings and suggestions from events. |
| `TopBar` | `status`, `onCheck`, `disabled` | Product name as plain text, status text ("Analyzing", "Waiting for your answer", "3 findings, 2 resolved"), Check button. Ctrl+Enter triggers Check. |
| `GoalInput` | `goal`, `inputShape`, setters, `inferred`, `onUseInferred` | Goal textarea (optional). When it is empty, the task inferred from the code is shown under it with a Use as goal button; the developer can correct the goal and press Check again. Input shape text field accepting comma-separated integers, validated on blur, error text below in `--crash`. |
| `CodeEditor` | `code`, `onChange`, `findings`, `selectedFindingId`, `onSelectFinding` | Monaco, language python, theme `vs-dark`, font JetBrains Mono 13px, minimap off. Markers via `setModelMarkers`: crash as Error, silent as Warning, risk as Info. Hover message shows finding title and message. Clicking a marked line selects the finding. Selected finding's lines get a solid `--surface-raised` line background decoration. |
| `PatchDiff` | `original`, `suggestion`, `onAccept`, `onReject` | Monaco `DiffEditor`, side by side, read-only, scrolled to the patched range. Header shows finding title, verify status badge, attempts. Accept (primary) applies the patch to `code`, returns to edit mode, and starts a new analysis. Reject (danger) closes the diff and marks the suggestion dismissed for this session. Accept is disabled for `unresolved` suggestions and shows the text "This patch did not pass verification". |
| `FindingsPanel` | `suggestions`, `findings`, `selectedFindingId`, `onSelect` | Groups by impact in order Crash, Silent, Risk. Group headers show count. Empty state after a run: "No issues found in the checks that ran. See Coverage for what was checked." Empty state before a run: "Paste a training script, describe your goal, and press Check." |
| `FindingCard` | `suggestion` or `finding` | Solid 3px left border in impact color. Line 1: title and line number in mono. Line 2: tier badge and verify status badge. Body: explanation (or message if no explanation yet). Evidence list (tool name in mono, summary). Docs link. "View patch" secondary button when a patch exists. While repair is in progress, shows "Repairing, attempt N of 3". |
| `AgentTrace` | `events` | Vertical list, one row per event: monospaced timestamp offset, phase or tool name, message or reason. Tool calls show arguments in a collapsible mono block. Attempt rows show the status badge. Auto-scrolls to the newest row while analyzing unless the user has scrolled up. |
| `QuestionCard` | `question`, `onAnswer` | Pinned at the top of the right panel when `status = needs_input`. 1px `--accent` border. Shows the prompt; if `options` exist, renders them as secondary buttons; always offers a free-text field. Submit adds the answer to `answers` and reruns analysis with the same code and goal. |
| `CoveragePanel` | `coverage`, rule catalogue from `/rules` | Two lists: Checked, and Skipped with reasons (only checks that were attempted for this script). No list of unsupported bug classes is shown. |
| `StatusBadge` | `kind`, `value` | Renders the text badge described in 11.3. |

### 11.5 Interaction flow

1. User pastes code, writes a goal, optionally sets input shape, presses Check.
2. Check is disabled, top bar shows "Analyzing", the indeterminate bar runs under the top bar, the Trace tab is auto-selected.
3. Findings appear in the editor as markers and in the Findings tab as they stream in. When the first suggestion arrives, the Findings tab is auto-selected once.
4. If a `needs_input` event arrives, the QuestionCard appears, status becomes "Waiting for your answer", and the answer triggers a new run.
5. On `done`, status shows a summary. Clicking a finding with a patch opens PatchDiff in the left column.
6. Accept updates the code and reruns. Reject closes the diff.
7. On `error`, a solid `--surface` banner with a 1px `--crash` border appears above the editor with the message and line, and Check is re-enabled.

### 11.6 Accessibility and polish

- All interactive elements reachable by keyboard; visible 2px solid `--accent` focus outline.
- Color is never the only signal: every status has a text label.
- Text contrast at least 4.5:1 against its background.
- Copy is short and literal. No exclamation marks, no marketing language inside the app.

---

## 12. Setup and bootstrap

Prerequisites on this machine: Python 3.12 (`py -3.12`), Node 24, Git.

Recommended: develop from a folder outside OneDrive (for example `C:\dev\EpochZero`). OneDrive tries to sync `node_modules` and `.venv` (over 1 GB together), which slows installs and can lock files. If you stay in OneDrive, pause syncing while installing.

Run from the repository root in PowerShell:

```powershell
git init

# .gitignore
@'
node_modules/
.next/
.venv/
__pycache__/
.pytest_cache/
.env
'@ | Out-File .gitignore -Encoding utf8

# Frontend
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm --yes
cd frontend
npm install @monaco-editor/react
cd ..

# Backend
mkdir backend
cd backend
py -3.12 -m venv .venv
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi "uvicorn[standard]" pydantic groq python-dotenv pytest httpx
pip freeze > requirements.txt
"GROQ_API_KEY=`nEPOCHZERO_MODEL=`nEPOCHZERO_LLM=live" | Out-File .env.example -Encoding utf8
Copy-Item .env.example .env      # then put the real key in .env
cd ..
```

`requirements.txt` already contains the PyTorch CPU index line, so `pip install -r requirements.txt` installs the CPU build. A teammate cloning the repo later runs:

```powershell
cd backend; py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; cd ..
cd frontend; npm install; cd ..
```

---

## 13. Running and testing

Two terminals:

```powershell
# Terminal 1
cd backend; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend; npm run dev
```

Open `http://localhost:3000`.

Offline mode: set `EPOCHZERO_LLM=mock` in `backend/.env`. If `GROQ_API_KEY` is empty the backend switches to offline mode automatically, and `GET /health` reports `"llm": "mock"`. In offline mode the goal reader is a keyword parser, the planner runs a fixed order (shape probe, then class count), and the fixer has deterministic repairs for EZ001, EZ004, EZ005, EZ006, and EZ008. Live mode falls back to these when an API call fails.

Tests:

```powershell
cd backend; .\.venv\Scripts\Activate.ps1; pytest -q
```

Definition of done for any rule: the buggy snippet fires exactly that rule, the fixed snippet fires nothing, and the verifier reports `resolved` for the repair.

---

## 14. Build plan (2 people, 8 hours)

Member A owns the deterministic side. Member B owns the agent and the frontend. Both are blocked on nothing after hour 0:30.

| Time | Member A (deterministic) | Member B (agent and UI) | Exit criterion |
|---|---|---|---|
| 0:00 - 0:30 | Bootstrap together. Write `schemas.py` and `types.ts` together. | same | Both files committed, both people agree |
| 0:30 - 2:30 | `facts.py`; rules EZ002, EZ005 - EZ010 with fixtures and tests | `llm/client.py`, `llm/mock.py`, `prompts.py`, `goal_reader.py`, `fixer.py` tested against hand-written findings | `pytest` green for static rules; fixer returns valid `FixProposal` in live and mock mode |
| 2:30 - 4:30 | `sandbox/` (extract, harness, runner); `shape_probe.py` (EZ001); `class_count.py` (EZ004); `goal_rules.py` (EZ003); `patching.py`, `verifier.py` | `registry.py` wiring, `planner.py`, `orchestrator.py`, `api/routes.py`; end-to-end in mock mode from curl | `curl` to `/analyze` on each demo script streams the expected events |
| 4:30 - 6:00 | `hyperparams.py` (EZ011); `knowledge/docs.py`; the four demo scripts; edge-case fixtures | Frontend: layout, tokens, GoalInput, CodeEditor with markers, FindingsPanel, PatchDiff, AgentTrace, QuestionCard, CoveragePanel, against `mockEvents.ts` first, then the live API | Each demo script runs end to end in the browser |
| 6:00 - 7:00 | Integration and bug fixing together; tighten prompts on real outputs | same | All three demos pass three times in a row in live mode |
| 7:00 - 8:00 | Feature freeze. Rehearse the pitch, record a backup video, capture mock-mode event logs of each demo as the fallback | same | Video saved, fallback tested with network off |

Cut order if behind: EZ011, then `inspect_lines`, then CoveragePanel detail (plain text instead), then EZ010. Never cut: the verifier and its three outcomes, the `ask_user` path, tier and status labels.

### Progress checklist

Phase 0
- [x] Repository bootstrapped
- [x] `schemas.py` and `types.ts` agreed

Phase 1
- [x] `facts.py`
- [x] EZ002, EZ005, EZ006, EZ007, EZ008, EZ009, EZ010 with tests
- [x] `llm/client.py`, `llm/mock.py`
- [x] `goal_reader.py`, `fixer.py`

Phase 2
- [x] Sandbox
- [x] EZ001 shape probe
- [x] EZ003 goal rules
- [x] EZ004 class count
- [x] Patching and verifier
- [x] Registry, planner, orchestrator, API

Phase 3
- [x] EZ011, docs snippets, demo scripts
- [x] Frontend components
- [x] End-to-end in offline mode (all four demos pass in `test_demos.py` and over HTTP)
- [ ] End-to-end in live mode (needs `GROQ_API_KEY`; not yet run)
- [ ] Visual check of the UI in a browser

Phase 4
- [ ] Three demos pass three times in a row in live mode
- [ ] Backup video
- [x] Offline fallback

---

## 15. Demo scripts

Run in this order. Each takes under a minute.

1. **`01_shape_mismatch.py`**: a CNN whose `fc1 = nn.Linear(512, 10)` follows a flatten that produces 2048 features. Goal: "10-class image classification on 32x32 images". Shows: shape inferred from code, EZ001 found at the right line, patch to `Linear(2048, 10)`, verifier reruns the probe, status resolved.
2. **`02_segmentation_classes_grounded.py`**: segmentation head with `out_channels=7`, while `NUM_CLASSES = 5`, the class list has 5 names, and the goal says "5-class road scene segmentation". Shows: measured output of 7 channels, four sources agree on 5, minimal patch on the head line only, verification resolved.
   **`02b_segmentation_classes_ambiguous.py`**: same model, but the goal says nothing about class count and the class list has 7 names while `NUM_CLASSES = 5`. Shows: the agent refuses to guess and asks "The head outputs 7 channels, NUM_CLASSES is 5, and the class list has 7 names. How many classes are you segmenting?" The answer triggers a rerun and a verified patch.
3. **`03_val_leak_and_lr.py`**: validation transform contains `RandomHorizontalFlip` and `ColorJitter`, Adam with `lr=0.1`. Shows: a verified finding with a resolved patch next to an advisory warning with no patch, demonstrating honest tiers.

Close on the Coverage tab: what was checked and what was skipped for that script.

---

## 16. Working agreements

- Branches: `main` is always runnable. Work on short branches (`a/facts`, `b/planner`) and merge at least every hour.
- Commits: small, present tense, prefixed with the area (`backend:`, `frontend:`, `docs:`). No emojis.
- Contract changes (`schemas.py`, `types.ts`, event order) need both people to agree and a Decision Log entry before merging.
- Every rule ships with fixtures and tests in the same commit.
- Secrets live only in `backend/.env`, never committed.
- Keep `EPOCHZERO_LLM=mock` working at all times; it is the demo fallback.

---

## 17. Decision log

Record every change to scope, contracts, or design here, newest last.

| # | Date | Decision | Reason |
|---|---|---|---|
| D1 | 2026-10-22 | One agent (orchestrator loop) calling one LLM in three roles; no multi-agent framework | Simpler, debuggable in 8 hours, and the decision points are explicit |
| D2 | 2026-10-22 | No trained models | No dataset, worse than a frontier model at code reasoning, and deterministic tools give proof rather than probability |
| D3 | 2026-10-22 | Meta-tensor forward pass with hooks is the primary shape check; `torch.fx` is stretch | Works on more models including Python control flow; simpler to build |
| D4 | 2026-10-22 | Stateless backend; clarification answers are sent back in `answers` | No session store needed |
| D5 | 2026-10-22 | Structured LLM output via `messages.parse`; planner uses `tool_choice: auto` with strict tools | Forced `tool_choice` is rejected by `claude-opus-5-5` |
| D6 | 2026-10-22 | Patches are line-range replacements, each applied independently to the original code | More reliable than unified diffs from an LLM; independent Accept per finding |
| D7 | 2026-10-22 | Verification has three outcomes: resolved, unresolved, inconclusive (plus not_attempted) | Honest reporting when a check cannot run |
| D8 | 2026-10-22 | Dry-run probe (EZ101 - EZ103) moved to stretch | Two-person team; core checks first |
| D9 | 2026-10-22 | Class-count mismatch detected by cross-checking sources, not by shape alone | `CrossEntropyLoss` with more channels than label values does not crash, so shape alone misses it |
| D10 | 2026-10-22 | Missing `zero_grad` downgraded to info when gradient accumulation is detected | Accumulation is a valid pattern |
| D11 | 2026-10-22 | Frontend: flat dark theme, no gradients, no emojis | Team requirement |
| D12 | 2026-10-23 | The "Not caught" list is internal only; never shown in the app, pitch, or judge-facing material | Team decision |
| D13 | 2026-10-23 | `CodeFacts` lives in `analysis/facts.py` as dataclasses, not in `schemas.py` | It is internal to the backend and never sent to the frontend |
| D14 | 2026-10-23 | Backend switches to offline mode automatically when no API key is set; offline components are also the fallback when a live call fails | The demo must run even without network or credentials |
| D15 | 2026-10-23 | For class count, the stated goal or the developer's answer alone is enough to ground a fix; otherwise two agreeing sources are required | The goal is the developer's own statement of intent |
| D16 | 2026-10-23 | Every planner tool call carries a required `reason` argument, shown in the Agent trace | Makes each decision visible to the developer and the judges |
| D17 | 2026-10-23 | Shape mismatch sizes come from comparing the measured layer input with the size declared in the source, with error-message parsing as fallback | Error messages on the meta device differ from CPU messages |
| D18 | 2026-10-23 | Sandbox results are cached by script hash | The verifier reruns the probe on the original code; this halves probe time |
| D19 | 2026-10-23 | Added `GET /examples` and a Load example picker in the top bar | Fast, reliable demo flow |
| D20 | 2026-10-23 | Tests are inline snippets (`test_rules.py`) plus full demo runs (`test_demos.py`) instead of per-rule fixture files | Same coverage, faster to write and read |
| D21 | 2026-10-23 | `mockEvents.ts` dropped; the frontend is developed against the backend in offline mode | One mock path instead of two |
| D22 | 2026-10-24 | LLM provider switched from Claude to Groq (`openai/gpt-oss-120b`); supersedes the provider part of D5 | No budget for a paid API key; Groq has a free tier |
| D23 | 2026-10-24 | Structured output via JSON object mode plus Pydantic validation and one corrective retry | Works on every Groq model, not only those with strict schema support |
| D24 | 2026-10-24 | One-click Apply fix on each finding card and Apply all for verified fixes; patches are applied bottom-up to the analyzed code, overlapping ones skipped, then the script is re-checked | Faster flow; clicking Apply is the human approval |
| D25 | 2026-10-24 | Goal is optional. With no goal, the task is inferred from the code (dataset names, loss, head) and sent as a `goal` event; the UI shows it with Use as goal | Developers often do not state a goal; the inferred one is editable |
| D26 | 2026-10-24 | An inferred goal is not counted as a class-count source | It is derived from the code, so counting it would let the code vote for itself |
| D27 | 2026-10-24 | Verifier reruns the shape probe with the exact input shape the finding was measured on | Bug: the planner used 28x28, the verifier fell back to 224x224 and marked a correct patch unresolved |
| D28 | 2026-10-24 | Persistent sandbox worker instead of one process per check; planner effort set to low | Each fresh process spent about 6 s importing torch; demos dropped from about 25 s to about 2 s offline |
| D29 | 2026-10-24 | Unresolved patches can be applied with an Apply anyway button (outlined, red) | The developer has the final say; the re-check after applying shows the result |
| D30 | 2026-10-24 | Known datasets (MNIST, CIFAR-10/100, ImageNet and others) set the image size and class count hints | Better input-shape inference and goal inference |
| D31 | 2026-10-24 | Class count is consistent when every source agrees with the head, even with a single source | Bug: one agreeing source was treated as ambiguous and triggered a needless question |
| D32 | 2026-10-24 | Two Groq models: `openai/gpt-oss-20b` for goal reader and planner, `openai/gpt-oss-120b` for the fixer (`EPOCHZERO_FAST_MODEL`, `EPOCHZERO_MODEL`) | Speed, and Groq rate limits are per model so the free-tier budget is spread |
| D33 | 2026-10-24 | Rate limits: wait at most 8 s for a reset, otherwise use the offline fallback for that step; every call is logged with time and tokens | Measured cause of a 62 s run: 429 responses with 5 to 13 s waits, retried by the SDK |
| D34 | 2026-10-24 | Planner receives a compact script summary and reads source with inspect_lines; fixer receives a windowed excerpt for scripts over 120 lines | Cuts tokens per request roughly in half; measured live runs 4 to 10 s |
| D35 | 2026-10-24 | The planner may only ask a question a check reported it needs; a coverage guard runs the shape probe and class count if the planner skipped them | Observed: the planner asked for a class count before running any check and missed a real shape bug |
| D36 | 2026-10-24 | "Don't know" is a valid answer everywhere: I don't know button on every question, unknown text in the shape box, no repeated questions; unknown class count leaves the finding unpatched | Users may not know these details |
| D37 | 2026-10-24 | A 1-channel Conv2d input defaults to 28x28; any shape finding based on an assumed input size is tier probed, with the assumption stated | Real data may differ from the assumption, so it is not claimed as verified |
| D38 | 2026-10-24 | `task_name` free-text field on GoalProfile and new task types (image_to_image, object_detection, language_modeling, generation, other); summaries use the name when the type is coarse | Bug: a correctly inferred image restoration task was shown as "unknown task" because the type list had no slot for it |
| D39 | 2026-10-24 | LLM code review phase (EZ100) after the planner. It may raise findings, each with an optional patch; they are tier advisory, and their patches are verified only to parse and introduce no new rule findings (status inconclusive). Supersedes "the LLM never finds bugs" from D1 | 11 fixed rules found 0 of 20+ real issues in a DINOv3 segmentation script; most real bugs are semantic (pretrained loading, model output types) |
| D40 | 2026-10-24 | pyflakes added to the static rules (EZ012, EZ013) | Undefined names and similar errors crash a run and are deterministic to detect |
| D41 | 2026-10-24 | Syntax errors become EZ000 findings with fixes instead of an error banner; all errors are listed by blanking each reported line and parsing again | Bug: only the first error was reported and nothing could be applied |
| D42 | 2026-10-24 | The sandbox blocks torch.hub.load, torch.load, and from_pretrained while building the model, and reports a clear skip reason; numpy added to the backend environment | Bug: the shape probe failed on a missing numpy, and pretrained downloads would hang until the timeout |
| D43 | 2026-10-24 | Rule-based findings try the built-in deterministic repair first and call the LLM only if it does not apply or fails verification (EZ001, EZ002, EZ004, EZ005, EZ006, EZ007, EZ008, EZ010) | Saves the free-tier token budget for the code review; repairs are instant and verified the same way |
| D44 | 2026-10-24 | On a long rate-limit wait, a call retries on the fast model at low effort before falling back offline; reviewer findings get at most two repair attempts | Groq limits are per model; observed waits of 13 to 28 s on the larger model |
| D45 | 2026-10-24 | All LLM prose is passed through a plain-text filter that maps typographic punctuation to ASCII and drops emojis and symbol characters | Team rule: no emojis anywhere in the app |
| D46 | 2026-10-24 | Apply all covers verified and unverified fixes and states how many are unverified; fixes that failed verification are excluded | With a code review, a script can have a dozen unverified fixes; one click with an honest count |
| D47 | 2026-10-24 | Reviewer entries flagged `is_real_problem: false`, or phrased as "no issue" or "keep as is", are dropped; patches that change nothing are dropped | Observed: a card reading "No issue; os is already imported" |
| D48 | 2026-10-24 | Clicking a finding card opens its patch; the View patch button is removed | One click instead of two; clicking already moved to the patch |
| D49 | 2026-10-24 | Applying a fix no longer re-checks. Applied fixes are tracked as a set and composed onto the analyzed code; remaining patches, markers, and line numbers are shifted to match; overlapping fixes are marked; each applied fix can be undone. The script is re-checked automatically when every suggested fix is applied or after Apply all, otherwise when the developer presses Check. A hand edit pauses the fixes until the next check. Supersedes the re-check part of D24 | Re-checking after every single fix was slow and discarded the other suggestions |
| D50 | 2026-10-24 | Maximum patch size raised from 15 to 40 lines | Real fixes (for example a dataset `__getitem__`) were rejected only for their size |
| D51 | 2026-10-24 | Every LLM patch is re-indented against the code it replaces (first line lost its indent, whole block shifted, tabs), keeping the first candidate that parses; prompts require full absolute indentation | Observed: applied fixes repeatedly landed with wrong indentation |
| D52 | 2026-10-24 | No size limit on patches (supersedes D50) | A correct fix should never be rejected for its length |
| D53 | 2026-10-24 | Applied fixes stay openable: clicking an applied card opens a diff of what it changed (Undo or Back to editor) | Observed: once applied, the patch could no longer be reviewed |
| D54 | 2026-10-24 | After applying or closing a diff, the editor scrolls to the changed lines and marks every applied change with a green line tint and gutter bar; hovering shows which fix made it | Observed: the editor jumped back to line 1 after every fix |
| D55 | 2026-10-24 | History tab: every applied fix with time, check number, status, and its removed and added lines; fixes from the current check can be opened or undone | Developers need to see what was changed and when |
| D56 | 2026-10-24 | Logs: a log holds one script's code, goal, input shape, check count, and history, saved in the browser (localStorage, guarded). Re-checking stays in the log and keeps its history; New log starts empty; Load example opens a new log; logs can be switched from the top bar. The first check of a log starts automatically once a script of at least 5 lines is pasted and typing pauses; later checks are manual | Requested: history that persists within a session, and a clean start for new code |
| D57 | 2026-10-24 | The page body moved to `components/Workspace.tsx`, rendered in the browser only | It reads saved logs at startup, which is not possible during server rendering |
| D58 | 2026-10-24 | Every finding has a pipeline area (`category`: syntax, python, data, pretrained, model, loss, optimization, training_loop, evaluation, device, checkpointing, other); rules map to a fixed area and the reviewer picks one; the Findings tab groups by severity or by area; the reviewer gets precise crash, silent, and risk definitions | Observed: a shape error and a missing checkpoint folder were grouped together as "Crash" with nothing else to tell them apart |
| D59 | 2026-10-24 | Code-review fixes that pass the safety check are labelled Unverified (not Inconclusive), with a reason that says no automated check covers that kind of issue | The old text claimed a check could not run, which was not true |
| D60 | 2026-10-24 | Dry run (EZ101 to EZ103, previously stretch): two real training steps on CPU with a random batch, synthetic targets shaped for the task, and the script's own loss. Reports loss and target mismatches, backward errors, non-finite losses, and layers without gradients. Skipped for models over 150M parameters, when weights must be downloaded, or when the forward pass already fails | Most fix-induced breakage is only visible when the loss and backward actually run |
| D61 | 2026-10-24 | Code-review fixes are verified with static rules, the shape probe, and the dry run; a fix that makes any of them worse (new finding, or a check that can no longer run) is unresolved | "It parses" was the only check before |
| D62 | 2026-10-24 | `POST /verify`: fixes are checked together before they are applied. Starting from the already applied fixes, each candidate is added in turn and kept only if the combined script gets no worse; held-back fixes show the reason and can still be applied with Apply anyway. Apply all re-checks automatically only if nothing was held back | Each fix was verified alone, so fixes could contradict each other once combined |
| D63 | 2026-10-24 | Re-check labels: each finding in a re-check is marked "From your fix" (on lines an applied fix changed), "Still present", or "New in this check", with a summary line | Makes clear which problems the fixes caused and which were there before |
| D64 | 2026-10-24 | The fixer is told about the other problems being fixed in the same check, so its patch stays consistent with them | Fixes were written in isolation |
| D65 | 2026-10-24 | Every check sends the log's history (applied fixes with their removed and added lines) and the previous check's unfixed findings. The reviewer is told which changes were deliberate, must not re-report or undo them, and must report problems they introduced; the fixer is told not to undo them; the planner sees them. Reviewer issues that repeat a fix still present in the code are dropped, unless they say the fix caused them | The backend is stateless, so without this the agent forgot every earlier fix on each check |
| D66 | 2026-10-24 | The code review waits up to 30 s for rate-limit resets (other steps 8 s) and switches model before waiting; if it still cannot run, the app says the result is not complete instead of "No issues found". Reviewer filters tightened: the "not a problem" filter only reads the title and the start of the explanation, and the "already fixed" filter only compares with code-review fixes, needs two shared content words, and ignores generic words such as "error". Dropped reviewer issues are logged | Observed: after a syntax fix the automatic re-check hit the rate limit, the review was skipped silently, and the app showed "No issues found" for a script with many problems |

---

## 18. Stretch goals and roadmap

Hackathon stretch (in order):
1. Dry-run probe: EZ101 NaN/inf loss, EZ102 dead gradients, EZ103 cannot overfit one batch.
2. `torch.fx` symbolic trace as an additional shape source with per-node shapes.
3. "Apply all resolved patches" with sequential re-verification.

Roadmap (pitch only, not built):
- Mine public PyTorch training scripts to discover new bug patterns and turn them into rules.
- Learn a ranking of findings by estimated wasted compute.
- IDE extension and CI check that run the same backend.
- Support for PyTorch Lightning and Hugging Face `Trainer` scripts.
