import type { Category, Finding, Suggestion } from "./types";

// Display order follows the pipeline, from the file itself to the outputs it writes.
export const CATEGORY_ORDER: Category[] = [
  "syntax", "python", "data", "pretrained", "model", "loss", "optimization", "training_loop", "evaluation", "device",
  "checkpointing", "other",
];

export const CATEGORY_LABEL: Record<Category, string> = {
  syntax: "Syntax",
  python: "Python errors",
  data: "Data pipeline",
  pretrained: "Pretrained weights",
  model: "Model",
  loss: "Loss and targets",
  optimization: "Optimization",
  training_loop: "Training loop",
  evaluation: "Evaluation",
  device: "Device",
  checkpointing: "Checkpointing",
  other: "Other",
};

// A code-review fix that "passed" only means it parses and breaks no rule: call it unverified, not inconclusive.
export function displayStatus(finding: Finding, suggestion: Suggestion) {
  return finding.tier === "advisory" && suggestion.verify_status === "inconclusive" ? "unverified" : suggestion.verify_status;
}
