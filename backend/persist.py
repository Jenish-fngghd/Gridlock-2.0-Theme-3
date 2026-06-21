"""Write a pipeline result into Supabase: storage + jobs + detections + plates + violations + audit."""
from __future__ import annotations
import hashlib
import uuid

from .supa import get_client
from .storage import upload_bytes
from .audit import chain_hash
from .schemas import ViolationOut, PlateOut
from inference.types import PipelineResult, PlateResult


def persist(
    image_bytes: bytes,
    filename: str | None,
    camera_id: str,
    media_type: str,
    result: PipelineResult,
    elapsed_ms: int,
) -> tuple[str, list[ViolationOut]]:
    sb = get_client()
    sha = hashlib.sha256(image_bytes).hexdigest()
    job_id = str(uuid.uuid4())

    # 1) original image
    orig_path = f"{job_id}/{filename or 'upload.jpg'}"
    upload_bytes("originals", orig_path, image_bytes)

    # 2) job row
    sb.table("ingestion_jobs").insert(
        {
            "id": job_id,
            "camera_id": camera_id,
            "media_type": media_type,
            "storage_path": orig_path,
            "sha256": sha,
            "status": "processing",
            "model_version": result.model_version,
        }
    ).execute()

    # 3) detections
    det_rows = [
        {
            "job_id": job_id,
            "class_label": d.class_label,
            "confidence": d.confidence,
            "bbox": d.bbox,
            "track_id": d.track_id,
            "attributes": d.attributes,
        }
        for d in result.detections
    ]
    if det_rows:
        sb.table("detections").insert(det_rows).execute()

    # 4) annotated image (optional)
    annotated_path = None
    annotated_url = None
    if result.annotated_image:
        annotated_path = f"{job_id}/annotated.jpg"
        annotated_url = upload_bytes("annotated", annotated_path, result.annotated_image)

    # 5) violations (+ plate upsert + genesis audit row)
    out: list[ViolationOut] = []
    for v in result.violations:
        plate_id = None
        plate_out = None
        if v.plate:
            plate_id = _upsert_plate(sb, v.plate)
            plate_out = PlateOut(**v.plate.__dict__)

        vid = str(uuid.uuid4())
        ev_hash = chain_hash(
            {"violation_id": vid, "type": v.violation_type, "evidence": v.evidence}, None
        )
        sb.table("violations").insert(
            {
                "id": vid,
                "job_id": job_id,
                "camera_id": camera_id,
                "violation_type": v.violation_type,
                "confidence": v.confidence,
                "confidence_band": v.confidence_band,
                "annotated_image_path": annotated_path,
                "evidence": v.evidence,
                "vlm_caption": v.vlm_caption,
                "model_module": v.model_module,
                "model_version": v.model_version,
                "sha256_evidence": ev_hash,
                "plate_id": plate_id,
            }
        ).execute()
        sb.table("evidence_audit").insert(
            {
                "violation_id": vid,
                "event_type": "created",
                "payload": {"evidence": v.evidence},
                "sha256": ev_hash,
                "prev_hash": None,
            }
        ).execute()

        out.append(
            ViolationOut(
                id=vid,
                violation_type=v.violation_type,
                confidence=v.confidence,
                confidence_band=v.confidence_band,
                annotated_image_url=annotated_url,
                vlm_caption=v.vlm_caption,
                plate=plate_out,
                evidence=v.evidence,
            )
        )

    # 6) close the job
    sb.table("ingestion_jobs").update(
        {"status": "done", "processing_ms": elapsed_ms}
    ).eq("id", job_id).execute()

    return job_id, out


def _upsert_plate(sb, p: PlateResult) -> str:
    existing = (
        sb.table("plates").select("id").eq("plate_normalized", p.plate_normalized).execute()
    )
    if existing.data:
        pid = existing.data[0]["id"]
        sb.table("plates").update({"last_seen_at": "now()"}).eq("id", pid).execute()
        return pid
    pid = str(uuid.uuid4())
    sb.table("plates").insert(
        {
            "id": pid,
            "plate_text": p.plate_text,
            "plate_normalized": p.plate_normalized,
            "state_code": p.state_code,
            "is_valid_format": p.is_valid_format,
            "ocr_confidence": p.ocr_confidence,
        }
    ).execute()
    return pid
