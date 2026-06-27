"""POST /api/process — upload an image/clip, run the pipeline, persist evidence."""
from __future__ import annotations
import base64
import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..config import settings
from ..schemas import ProcessOut, ViolationOut, PlateOut, DetectionOut, StageOut
from inference.service import run_pipeline
from inference.types import PipelineResult

router = APIRouter(prefix="/api", tags=["process"])

UPLOADS_CAMERA = "00000000-0000-0000-0000-000000000001"  # seeded "Uploads" camera

# Keyword buckets for classifying raw detection labels into the rail's stages.
_VEHICLE_KW = ("car", "motorcycle", "motorbike", "bike", "truck", "bus", "auto", "vehicle", "scooter")
_PERSON_KW = ("person", "rider", "driver", "pedestrian", "passenger", "pillion")
_PLATE_KW = ("plate", "license", "licence", "anpr", "number")


def _bucket(label: str) -> str:
    low = label.lower()
    if any(k in low for k in _PLATE_KW):
        return "plate"
    if any(k in low for k in _PERSON_KW):
        return "person"
    if any(k in low for k in _VEHICLE_KW):
        return "vehicle"
    return "other"


def _detections_out(result: PipelineResult) -> list[DetectionOut]:
    return [
        DetectionOut(
            class_label=d.class_label,
            confidence=d.confidence,
            bbox=d.bbox,
            track_id=d.track_id,
        )
        for d in result.detections
        if isinstance(d.bbox, dict) and {"x", "y", "w", "h"} <= set(d.bbox)
    ]


def _build_stages(result: PipelineResult, n_bytes: int) -> list[StageOut]:
    """Reflect the real pipeline stages and what each actually produced."""
    dets = result.detections
    counts = {"vehicle": 0, "person": 0, "plate": 0, "other": 0}
    for d in dets:
        counts[_bucket(d.class_label)] += 1

    vtypes = {v.violation_type for v in result.violations}
    plate_txt = next((v.plate.plate_text for v in result.violations if v.plate), None)

    def has(*types: str) -> list[str]:
        return [t for t in types if t in vtypes]

    rider_hits = has("helmet", "triple_riding")
    scene_hits = has("wrong_side", "red_light", "stop_line", "illegal_parking")

    kb = max(1, round(n_bytes / 1024))
    return [
        StageOut(key="upload", label="Uploading", ran=True, detail=f"{kb} KB"),
        StageOut(key="detect", label="Object detection", ran=True,
                 detail=f"{counts['vehicle']} vehicles · {counts['person']} people"),
        StageOut(key="rider", label="Rider / helmet & triple-riding", ran=True,
                 detail=", ".join(rider_hits) or "none flagged"),
        StageOut(key="seatbelt", label="Seatbelt classifier", ran=True,
                 detail="no seatbelt" if "seatbelt" in vtypes else "none flagged"),
        StageOut(key="scene", label="Signal / wrong-side / red-light", ran=True,
                 detail=", ".join(scene_hits) or "none flagged"),
        StageOut(key="plate", label="Plate detection & OCR", ran=True,
                 detail=plate_txt or ("plate region found" if counts["plate"] else "no plate read")),
        StageOut(key="report", label="Compiling evidence", ran=True,
                 detail=f"{len(result.violations)} violation(s)"),
    ]


@router.post("/process", response_model=ProcessOut)
async def process(
    file: UploadFile = File(...),
    camera_id: str | None = Form(None),
    media_type: str = Form("image"),
):
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large")

    # run_pipeline (CPU/torch + VLM I/O) and persist (Supabase HTTP) are BLOCKING. Run them in a
    # worker thread so the single-worker event loop stays free — otherwise one in-flight upload
    # freezes every other request (health checks, other users' uploads) until it completes.
    t0 = time.time()
    result = await run_in_threadpool(run_pipeline, data, media_type)
    elapsed = int((time.time() - t0) * 1000)

    detections = _detections_out(result)
    stages = _build_stages(result, len(data))

    if settings.supabase_configured:
        from ..persist import persist  # imported lazily so dev mode needs no supabase pkg

        job_id, violations = await run_in_threadpool(
            persist, data, file.filename, camera_id or UPLOADS_CAMERA, media_type, result, elapsed
        )
        # Surface the stored annotated frame (if the pipeline produced one) at the top level.
        annotated_url = next((v.annotated_image_url for v in violations if v.annotated_image_url), None)
        return ProcessOut(
            job_id=job_id, status="done", processing_ms=elapsed, persisted=True,
            model_version=result.model_version, annotated_image_url=annotated_url,
            detections=detections, stages=stages, violations=violations,
        )

    # Dev mode (no keys): return results without persisting. If the pipeline produced an
    # annotated frame, inline it as a data URL so the console can still display it.
    annotated_url = None
    if result.annotated_image:
        b64 = base64.b64encode(result.annotated_image).decode("ascii")
        annotated_url = f"data:image/jpeg;base64,{b64}"

    violations = [
        ViolationOut(
            violation_type=v.violation_type,
            confidence=v.confidence,
            confidence_band=v.confidence_band,
            annotated_image_url=annotated_url,
            vlm_caption=v.vlm_caption,
            evidence=v.evidence,
            plate=PlateOut(**v.plate.__dict__) if v.plate else None,
        )
        for v in result.violations
    ]
    return ProcessOut(
        job_id=None, status="done", processing_ms=elapsed, persisted=False,
        model_version=result.model_version, annotated_image_url=annotated_url,
        detections=detections, stages=stages, violations=violations,
    )
