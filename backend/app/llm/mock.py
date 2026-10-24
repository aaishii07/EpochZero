"""Deterministic stand-ins for the three LLM roles.

Used when EPOCHZERO_LLM=mock or no API key is set, and as the fallback when a live call fails, so the full
pipeline always runs end to end.
"""

from __future__ import annotations

import re

from app.schemas import Finding, FixProposal, GoalProfile, Patch

NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def read_goal(goal: str) -> GoalProfile:
    g = goal.lower()
    if re.search(r"segment", g):
        task = "segmentation"
    elif re.search(r"multi-?label", g):
        task = "multilabel_classification"
    elif re.search(r"binary|\bvs\.?\b|versus|yes or no|or not\b|two-class|2-class", g):
        task = "binary_classification"
    elif re.search(r"regress|predict (the )?(price|value|age|temperature|score)|continuous", g):
        task = "regression"
    elif re.search(r"classif|class\b|classes|categor", g):
        task = "multiclass_classification"
    else:
        task = "unknown"
    num = None
    m = re.search(r"(\d+)[\s-]*(class|classes|categories|labels)", g) or re.search(r"(\w+)[\s-]+(class|classes)\b", g)
    if m:
        token = m.group(1)
        num = int(token) if token.isdigit() else NUMBER_WORDS.get(token)
    if task == "binary_classification":
        num = 2
    elif task == "multiclass_classification" and num == 2:
        task = "binary_classification"
    modality = "image" if re.search(r"image|photo|picture|pixel|scene|cifar|mnist|vision", g) else \
        "text" if re.search(r"text|sentence|token|review|language", g) else \
        "tabular" if re.search(r"tabular|csv|feature", g) else \
        "audio" if re.search(r"audio|speech|sound", g) else "unknown"
    return GoalProfile(task=task, num_classes=num, modality=modality)


def infer_goal(facts) -> GoalProfile:
    """Offline guess at the task from the code alone (used when no goal is given)."""
    loss = facts.loss.type if facts.loss else None
    dataset = facts.data_hints.get("dataset") or {}
    head = facts.head
    spatial_head = head is not None and head.type in {"Conv2d", "ConvTranspose2d", "Conv3d"}
    evidence = []
    if loss:
        evidence.append(loss)
    if dataset:
        evidence.append(f"{dataset['name']} dataset")
    if loss in {"CrossEntropyLoss", "NLLLoss"}:
        task = "segmentation" if spatial_head else "multiclass_classification"
    elif loss in {"BCEWithLogitsLoss", "BCELoss"}:
        task = "binary_classification" if facts.head_out in (None, 1) else "multilabel_classification"
    elif loss in {"MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss"}:
        task = "image_to_image" if spatial_head else "regression"
    else:
        task = "unknown"
    if spatial_head and task in {"segmentation", "image_to_image"}:
        evidence.append("convolutional output head")
    code_text = "\n".join(facts.lines).lower()
    keyword_names = [("restor", "image restoration"), ("denois", "image denoising"), ("super_res", "super-resolution"),
                     ("superres", "super-resolution"), ("deblur", "image deblurring"), ("detect", "object detection")]
    task_name = next((name for key, name in keyword_names if key in code_text), "")
    if task == "unknown" and task_name:
        task = "image_to_image" if task_name != "object detection" else "object_detection"
        evidence.append(f"'{task_name}' in the code")
    if not task_name and task == "image_to_image":
        task_name = "image-to-image regression"
    first = (facts.data_hints.get("first_layer") or {}).get("type", "")
    if first.startswith("Conv") or dataset:
        modality = "image"
    elif first in {"Embedding", "LSTM", "GRU"}:
        modality = "text"
    else:
        modality = "tabular" if first == "Linear" else "unknown"
    num = dataset.get("num_classes") if task in {"multiclass_classification", "segmentation"} else None
    if task == "binary_classification":
        num = 2
    notes = ("Inferred from " + ", ".join(evidence)) if evidence else "No clear signal in the code"
    return GoalProfile(task=task, task_name=task_name, num_classes=num, modality=modality, notes=notes)


def plan_steps(has_model: bool, has_head: bool, task: str) -> list[str]:
    """Fixed investigation order used when the planner LLM is unavailable."""
    steps = []
    if has_model:
        steps += ["probe_shapes", "dry_run"]
    if has_head and task in {"multiclass_classification", "multilabel_classification", "segmentation", "unknown"}:
        steps.append("check_class_count")
    return steps


def fix(code: str, finding: Finding) -> FixProposal:
    lines = code.splitlines()
    ev = finding.evidence[0].data if finding.evidence else {}

    def line(n):
        return lines[n - 1]

    def indent(text):
        return text[: len(text) - len(text.lstrip())]

    if finding.rule_id == "EZ001" and "expected" in ev and "received" in ev:
        src = line(finding.line)
        new = re.sub(rf"\b{ev['expected']}\b", str(ev["received"]), src, count=1)
        if new != src:
            return FixProposal(
                explanation=f"The layer was built for {ev['expected']} inputs, but the tensor reaching it has {ev['received']}. "
                            f"The patch sets the layer's input size to {ev['received']}.",
                patch=Patch(start_line=finding.line, end_line=finding.line, replacement=new))
    if finding.rule_id == "EZ004" and "target_value" in ev:
        src = line(finding.line)
        target = ev["target_value"]
        current = next((s["value"] for s in ev.get("sources", []) if s["line"] == finding.line), None)
        const = next((name for name, v in _constants(code).items() if v == target and re.search(r"class", name, re.I)), None)
        if current is not None:
            new = re.sub(rf"\b{current}\b", const or str(target), src, count=1)
            if new != src:
                return FixProposal(
                    explanation=f"This line uses {current} classes while the rest of the code and the goal use {target}. "
                                f"The patch changes it to {const or target}.",
                    patch=Patch(start_line=finding.line, end_line=finding.line, replacement=new))
    if finding.rule_id == "EZ005":
        doomed = set(range(finding.line, finding.end_line + 1)) & set(finding.evidence[0].lines)
        start, end = min(doomed), max(doomed)
        kept = [l for i, l in enumerate(lines[start - 1:end], start) if i not in doomed]
        return FixProposal(
            explanation="Random augmentations make validation results change between runs. The patch removes them from "
                        "the evaluation transform and keeps the deterministic steps.",
            patch=Patch(start_line=start, end_line=end, replacement="\n".join(kept)))
    if finding.rule_id == "EZ006":
        src = line(finding.line)
        opt = re.search(r"(\w+)\.step\(", "\n".join(lines))
        name = opt.group(1) if opt else "optimizer"
        return FixProposal(
            explanation="backward() adds to gradients left over from the previous batch. The patch resets them right "
                        "before backward() on every step.",
            patch=Patch(start_line=finding.line, end_line=finding.line, replacement=f"{indent(src)}{name}.zero_grad()\n{src}"))
    if finding.rule_id == "EZ008":
        src = line(finding.line)
        m = re.search(r"(\w+)\s*(\+=|=.*\+)\s*(\w+)\s*$", src)
        if m:
            new = src.rstrip() + ".item()"
            return FixProposal(
                explanation="Adding the loss tensor keeps its graph in memory every step. The patch accumulates a number instead.",
                patch=Patch(start_line=finding.line, end_line=finding.line, replacement=new))
    if finding.rule_id == "EZ010":
        src = line(finding.line)
        if ".float()" in src:
            return FixProposal(
                explanation="CrossEntropyLoss needs class indices of dtype long, not floats. The patch casts the targets "
                            "with .long() instead.",
                patch=Patch(start_line=finding.line, end_line=finding.line, replacement=src.replace(".float()", ".long()", 1)))
    if finding.rule_id == "EZ002":
        src = line(finding.line)
        new = re.sub(r"\b(?:F|torch|nn\.functional|torch\.nn\.functional)\.(?:log_)?softmax\((.+?),\s*dim\s*=\s*-?\d+\)",
                     r"\1", src, count=1)
        if new == src:
            new = re.sub(r"\b(?:torch\.)?sigmoid\((.+)\)", r"\1", src, count=1)
        if new != src:
            return FixProposal(
                explanation="The loss already applies this activation internally, so the model applied it twice. The patch "
                            "returns the raw logits.",
                patch=Patch(start_line=finding.line, end_line=finding.line, replacement=new))
    if finding.rule_id == "EZ007" and len(finding.evidence[0].lines) == 3:
        backward_line, zero_line, _ = finding.evidence[0].lines
        if backward_line < zero_line:
            block = lines[backward_line - 1:zero_line - 1]
            zero = lines[zero_line - 1]
            moved = indent(block[0]) + zero.strip()
            return FixProposal(
                explanation="zero_grad() erased the gradients before step() could use them. The patch moves it in front "
                            "of backward().",
                patch=Patch(start_line=backward_line, end_line=zero_line, replacement="\n".join([moved, *block])))
    raise ValueError(f"no offline repair available for {finding.rule_id}")


# Rules whose built-in repair is reliable; the fixer tries these first and only calls the LLM if they do not apply.
DETERMINISTIC_FIXES = {"EZ001", "EZ002", "EZ004", "EZ005", "EZ006", "EZ007", "EZ008", "EZ010"}


def _constants(code: str) -> dict[str, int]:
    out = {}
    for m in re.finditer(r"^([A-Za-z_]\w*)\s*=\s*(\d+)\s*$", code, re.M):
        out[m.group(1)] = int(m.group(2))
    return out
