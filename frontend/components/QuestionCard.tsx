"use client";

import { useState } from "react";

import type { Question } from "@/lib/types";

export default function QuestionCard({ question, onAnswer }: { question: Question; onAnswer: (value: string) => void }) {
  const [value, setValue] = useState("");
  const submit = (v: string) => {
    if (v.trim()) onAnswer(v.trim());
  };
  return (
    <div className="m-4 rounded-md border border-accent bg-surface p-4">
      <p className="text-[12px] font-medium uppercase tracking-[0.04em] text-accent">Your input is needed</p>
      <p className="mt-2 text-[13px] leading-5">{question.prompt}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {(question.options ?? [])
          .filter((opt) => opt.trim().toLowerCase() !== "other")
          .map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => submit(opt)}
              className="h-8 min-w-10 rounded-md border border-line px-3 font-mono text-sm hover:bg-raised"
            >
              {opt}
            </button>
          ))}
        <button
          type="button"
          onClick={() => submit("dont know")}
          className="h-8 rounded-md border border-line px-3 text-sm text-muted hover:bg-raised hover:text-ink"
        >
          I don&apos;t know
        </button>
      </div>
      <form
        className="mt-3 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
      >
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Or type an answer"
          aria-label="Answer"
          className="h-8 flex-1 rounded-md border border-line bg-canvas px-3 text-sm placeholder:text-muted/70"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="h-8 rounded-md bg-accent px-3 text-sm font-medium text-on-accent hover:opacity-90 disabled:bg-raised disabled:text-muted"
        >
          Submit
        </button>
      </form>
    </div>
  );
}
