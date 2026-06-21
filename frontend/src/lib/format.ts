import type { ViolationType, ConfidenceBand, ViolationStatus } from "./types";

export const VIOLATION_LABELS: Record<ViolationType, string> = {
  helmet: "No Helmet",
  triple_riding: "Triple Riding",
  seatbelt: "No Seatbelt",
  wrong_side: "Wrong Side",
  stop_line: "Stop Line",
  red_light: "Red Light",
  illegal_parking: "Illegal Parking",
};

export const BAND_STYLE: Record<ConfidenceBand, string> = {
  auto_confirm: "bg-emerald-50 text-emerald-700 border border-emerald-200",
  human_review: "bg-amber-50 text-amber-700 border border-amber-200",
  discard: "bg-slate-100 text-slate-500 border border-slate-200",
};

export const BAND_LABELS: Record<ConfidenceBand, string> = {
  auto_confirm: "Auto-confirm",
  human_review: "Review",
  discard: "Discard",
};

export const STATUS_STYLE: Record<ViolationStatus, string> = {
  pending: "text-amber-600",
  confirmed: "text-emerald-600",
  rejected: "text-rose-600",
  challan_issued: "text-sky-600",
};

export function timeAgo(iso: string): string {
  const s = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function pct(v: number | null): string {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}
