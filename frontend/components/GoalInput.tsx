"use client";

import { useState } from "react";

interface Props {
  goal: string;
  inputShape: string;
  onGoalChange: (value: string) => void;
  onInputShapeChange: (value: string) => void;
}

export function parseShape(text: string): number[] | null | "invalid" {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const parts = trimmed.split(/[\s,x]+/).filter(Boolean);
  const dims = parts.map(Number);
  if (!dims.length || dims.some((d) => !Number.isInteger(d) || d <= 0)) return "invalid";
  return dims;
}

const label = "mb-1 block text-[12px] font-medium uppercase tracking-[0.04em] text-muted";
const field =
  "w-full rounded-md border border-line bg-canvas px-3 py-2 text-sm text-ink placeholder:text-muted/70 hover:border-muted/60";

export default function GoalInput({ goal, inputShape, onGoalChange, onInputShapeChange }: Props) {
  const [shapeError, setShapeError] = useState(false);
  return (
    <div className="flex gap-3 border-b border-line bg-surface p-4">
      <div className="flex-1">
        <label htmlFor="goal" className={label}>
          Goal
        </label>
        <textarea
          id="goal"
          rows={2}
          value={goal}
          onChange={(e) => onGoalChange(e.target.value)}
          placeholder="Describe what this model should do, for example 5-class road scene segmentation"
          className={`${field} resize-none`}
        />
      </div>
      <div className="w-52 shrink-0">
        <label htmlFor="shape" className={label}>
          Input shape <span className="normal-case tracking-normal">(optional)</span>
        </label>
        <input
          id="shape"
          value={inputShape}
          onChange={(e) => {
            onInputShapeChange(e.target.value);
            if (shapeError) setShapeError(false);
          }}
          onBlur={() => setShapeError(parseShape(inputShape) === "invalid")}
          placeholder="1,3,224,224"
          className={`${field} font-mono ${shapeError ? "border-crash" : ""}`}
        />
        {shapeError && <p className="mt-1 text-xs text-crash">Use positive integers separated by commas.</p>}
      </div>
    </div>
  );
}
