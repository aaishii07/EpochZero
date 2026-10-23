"use client";

import { useState } from "react";

interface Props {
  goal: string;
  inputShape: string;
  onGoalChange: (value: string) => void;
  onInputShapeChange: (value: string) => void;
  inferred: { summary: string; notes: string } | null;
  onUseInferred: () => void;
}

// Words such as "don't know" mean the shape is unknown, which is a valid answer: EpochZero then infers or assumes one.
export function shapeIsUnknownText(text: string): boolean {
  return /[a-z]/i.test(text.replace(/x/gi, ""));
}

export function parseShape(text: string): number[] | null | "invalid" {
  const trimmed = text.trim();
  if (!trimmed || shapeIsUnknownText(trimmed)) return null;
  const parts = trimmed.split(/[\s,x]+/).filter(Boolean);
  const dims = parts.map(Number);
  if (!dims.length || dims.some((d) => !Number.isInteger(d) || d <= 0)) return "invalid";
  return dims;
}

const label = "mb-1 block text-[12px] font-medium uppercase tracking-[0.04em] text-muted";
const field =
  "w-full rounded-md border border-line bg-canvas px-3 py-2 text-sm text-ink placeholder:text-muted/70 hover:border-muted/60";

export default function GoalInput({ goal, inputShape, onGoalChange, onInputShapeChange, inferred, onUseInferred }: Props) {
  const [shapeError, setShapeError] = useState(false);
  return (
    <div className="flex gap-3 border-b border-line bg-surface p-4">
      <div className="flex-1">
        <label htmlFor="goal" className={label}>
          Goal <span className="normal-case tracking-normal">(optional)</span>
        </label>
        <textarea
          id="goal"
          rows={2}
          value={goal}
          onChange={(e) => onGoalChange(e.target.value)}
          placeholder="What should this model do? Leave empty and EpochZero infers it from the code"
          className={`${field} resize-none`}
        />
        {inferred && (
          <div className="mt-2 flex items-start gap-3 rounded-md border border-line border-l-[3px] border-l-accent bg-canvas px-3 py-2">
            <p className="flex-1 text-[13px] leading-5">
              <span className="text-muted">Inferred from the code: </span>
              {inferred.summary}
              {inferred.notes && <span className="block text-xs text-muted">{inferred.notes}</span>}
              <span className="block text-xs text-muted">If this is wrong, type the real goal above and press Check.</span>
            </p>
            <button
              type="button"
              onClick={onUseInferred}
              className="h-7 shrink-0 rounded-md border border-line px-2.5 text-xs text-ink hover:bg-raised"
            >
              Use as goal
            </button>
          </div>
        )}
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
        {shapeError && <p className="mt-1 text-xs text-crash">Use positive integers separated by commas, or leave it empty.</p>}
        {!shapeError && inputShape.trim() !== "" && shapeIsUnknownText(inputShape) && (
          <p className="mt-1 text-xs text-muted">Unknown is fine. EpochZero will infer the shape or state its assumption.</p>
        )}
      </div>
    </div>
  );
}
