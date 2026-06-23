export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Detection {
  class_label: string;
  confidence: number | null;
  bbox: BBox;
  track_id: number | null;
}

export interface Stage {
  key: string;
  label: string;
  ran: boolean;
  detail: string;
}

export interface ProcessResult {
  job_id: string | null;
  status: string;
  processing_ms: number;
  persisted: boolean;
  model_version: string | null;
  annotated_image_url: string | null;
  detections: Detection[];
  stages: Stage[];
  violations: Array<{
    id: string | null;
    violation_type: string;
    confidence: number | null;
    confidence_band: string;
    annotated_image_url: string | null;
    vlm_caption: string | null;
    plate: { plate_text: string; state_code: string | null } | null;
    evidence: Record<string, unknown>;
  }>;
}

export async function processFile(file: File): Promise<ProcessResult> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BACKEND_URL}/api/process`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Process failed (${res.status})`);
  return res.json();
}

export async function reviewViolation(
  id: string,
  action: "confirm" | "reject" | "escalate"
): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/api/violations/${id}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, reviewer_label: "operator" }),
  });
  if (!res.ok) throw new Error(`Review failed (${res.status})`);
}
