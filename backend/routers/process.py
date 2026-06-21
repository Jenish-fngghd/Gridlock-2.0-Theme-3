"""POST /api/process — upload an image/clip, run the pipeline, persist evidence."""
from __future__ import annotations
import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..config import settings
from ..schemas import ProcessOut, ViolationOut, PlateOut
from inference.service import run_pipeline

router = APIRouter(prefix="/api", tags=["process"])

UPLOADS_CAMERA = "00000000-0000-0000-0000-000000000001"  # seeded "Uploads" camera


@router.post("/process", response_model=ProcessOut)
async def process(
    file: UploadFile = File(...),
    camera_id: str | None = Form(None),
    media_type: str = Form("image"),
):
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")

    t0 = time.time()
    result = run_pipeline(data, media_type)
    elapsed = int((time.time() - t0) * 1000)

    if settings.supabase_configured:
        from ..persist import persist  # imported lazily so dev mode needs no supabase pkg

        job_id, violations = persist(
            data, file.filename, camera_id or UPLOADS_CAMERA, media_type, result, elapsed
        )
        return ProcessOut(
            job_id=job_id, status="done", processing_ms=elapsed,
            persisted=True, violations=violations,
        )

    # Dev mode (no keys): return results without persisting.
    violations = [
        ViolationOut(
            violation_type=v.violation_type,
            confidence=v.confidence,
            confidence_band=v.confidence_band,
            vlm_caption=v.vlm_caption,
            evidence=v.evidence,
            plate=PlateOut(**v.plate.__dict__) if v.plate else None,
        )
        for v in result.violations
    ]
    return ProcessOut(
        job_id=None, status="done", processing_ms=elapsed,
        persisted=False, violations=violations,
    )
