import type { Impact, Tier, VerifyStatus } from "@/lib/types";

type Props =
  | { kind: "tier"; value: Tier }
  | { kind: "impact"; value: Impact }
  | { kind: "status"; value: VerifyStatus | "repairing" };

const LABELS: Record<string, string> = {
  verified: "Verified",
  probed: "Probed",
  advisory: "Advisory",
  crash: "Crash",
  silent: "Silent",
  risk: "Risk",
  resolved: "Resolved",
  unresolved: "Unresolved",
  inconclusive: "Inconclusive",
  not_attempted: "Not attempted",
  repairing: "Repairing",
};

const COLORS: Record<string, string> = {
  verified: "border-ink text-ink",
  probed: "border-silent text-silent",
  advisory: "border-risk text-risk",
  crash: "border-crash text-crash",
  silent: "border-silent text-silent",
  risk: "border-risk text-risk",
  resolved: "border-resolved text-resolved",
  unresolved: "border-unresolved text-unresolved",
  inconclusive: "border-inconclusive text-inconclusive",
  not_attempted: "border-line text-muted",
  repairing: "border-accent text-accent",
};

export default function StatusBadge(props: Props) {
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-px text-[11px] font-medium leading-4 ${COLORS[props.value]}`}
    >
      {LABELS[props.value]}
    </span>
  );
}
