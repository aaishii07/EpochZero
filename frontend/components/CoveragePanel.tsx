import type { Coverage } from "@/lib/types";

export default function CoveragePanel({ coverage }: { coverage: Coverage | null }) {
  if (!coverage) {
    return <p className="p-6 text-sm text-muted">Coverage for the script appears here after a check.</p>;
  }
  return (
    <div className="space-y-5 p-4">
      <section>
        <h2 className="mb-2 text-[12px] font-medium uppercase tracking-[0.04em] text-muted">Checked</h2>
        {coverage.checked.length ? (
          <ul className="space-y-1">
            {coverage.checked.map((c) => (
              <li key={c} className="flex items-center gap-2 text-[13px]">
                <span className="h-2 w-2 rounded-full bg-resolved" aria-hidden />
                {c}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[13px] text-muted">No checks completed.</p>
        )}
      </section>
      {coverage.skipped.length > 0 && (
        <section>
          <h2 className="mb-2 text-[12px] font-medium uppercase tracking-[0.04em] text-muted">Skipped</h2>
          <ul className="space-y-1.5">
            {coverage.skipped.map((s, i) => (
              <li key={i} className="flex items-start gap-2 text-[13px]">
                <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-inconclusive" aria-hidden />
                <span>
                  {s.check}
                  <span className="text-muted">: {s.reason}</span>
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
