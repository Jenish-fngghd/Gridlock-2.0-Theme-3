export type ViolationType =
  | "helmet" | "triple_riding" | "seatbelt"
  | "wrong_side" | "stop_line" | "red_light" | "illegal_parking";

export type ConfidenceBand = "auto_confirm" | "human_review" | "discard";
export type ViolationStatus = "pending" | "confirmed" | "rejected" | "challan_issued";

export interface Plate {
  plate_text: string;
  plate_normalized: string;
  state_code: string | null;
  is_valid_format: boolean | null;
  ocr_confidence: number | null;
}

export interface Violation {
  id: string;
  job_id: string;
  camera_id: string | null;
  violation_type: ViolationType;
  confidence: number | null;
  confidence_band: ConfidenceBand;
  status: ViolationStatus;
  detected_at: string;
  annotated_image_path: string | null;
  evidence: Record<string, unknown>;
  sha256_evidence: string | null;
  vlm_caption: string | null;
  plate_id: string | null;
  created_at: string;
  plates?: Plate | null;
}
