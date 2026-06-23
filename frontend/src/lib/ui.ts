import type { ViolationType, ViolationStatus } from "./types";

/** Font-family CSS-var shorthands (set in app/layout.tsx). */
export const FONT = {
  sans: "var(--font-instrument-sans), system-ui, sans-serif",
  serif: "var(--font-instrument-serif), Georgia, serif",
  body: "var(--font-inter), system-ui, sans-serif",
  mono: "var(--font-mono), ui-monospace, monospace",
};

export type Severity = "Critical" | "High" | "Medium" | "Low";

/** Severity badge colours — matches the design canvas SC map. */
export const SEVERITY_COLOR: Record<Severity, { c: string; b: string }> = {
  Critical: { c: "#EF4444", b: "#FEF2F2" },
  High: { c: "#F59E0B", b: "#FFFBEB" },
  Medium: { c: "#0EA5E9", b: "#F0F9FF" },
  Low: { c: "#10B981", b: "#ECFDF5" },
};

/** Enforcement severity is a property of the violation class. */
const SEVERITY_BY_TYPE: Record<ViolationType, Severity> = {
  helmet: "Critical",
  red_light: "Critical",
  wrong_side: "Critical",
  triple_riding: "High",
  seatbelt: "High",
  stop_line: "Medium",
  illegal_parking: "Medium",
};

export function severityFor(type: ViolationType | string): Severity {
  return SEVERITY_BY_TYPE[type as ViolationType] ?? "Medium";
}

/** Real violation_status → design badge {label, colour}. */
export const STATUS_META: Record<ViolationStatus, { label: string; c: string; b: string }> = {
  pending: { label: "Pending", c: "#F59E0B", b: "#FFFBEB" },
  confirmed: { label: "Confirmed", c: "#10B981", b: "#ECFDF5" },
  challan_issued: { label: "Issued", c: "#10B981", b: "#ECFDF5" },
  rejected: { label: "Rejected", c: "#EF4444", b: "#FEF2F2" },
};

export function statusMeta(status: ViolationStatus | string) {
  return STATUS_META[status as ViolationStatus] ?? { label: String(status), c: "#52525B", b: "#F4F4F5" };
}

/** Colour a raw detection class label for its bounding box. */
export function boxColor(label: string): string {
  const l = label.toLowerCase();
  if (/(plate|licen[cs]e|number|anpr)/.test(l)) return "#8B5CF6"; // violet
  if (/(helmet|no_?helmet)/.test(l)) return "#EF4444"; // red
  if (/(person|rider|driver|pedestrian|passenger|pillion)/.test(l)) return "#0EA5E9"; // sky
  if (/(car|motorcycle|motorbike|bike|truck|bus|auto|vehicle|scooter)/.test(l)) return "#4F46E5"; // indigo
  if (/(light|signal|traffic_?light)/.test(l)) return "#F59E0B"; // amber
  return "#6366F1";
}

/** Human-readable detection label for chips (number_plate → PLATE). */
export function prettyClass(label: string): string {
  return label.replace(/[_-]+/g, " ").trim().toUpperCase();
}

/** Shared palette pulled from the canvas. */
export const COLORS = {
  bg: "#FAFAFA",
  ink: "#18181B",
  sub: "#52525B",
  muted: "#6B7280",
  faint: "#9CA3AF",
  border: "#ECECEC",
  hair: "#F4F4F5",
  indigo: "#4F46E5",
  indigo2: "#6366F1",
  sky: "#0EA5E9",
  violet: "#8B5CF6",
  green: "#10B981",
  red: "#EF4444",
  amber: "#F59E0B",
};
