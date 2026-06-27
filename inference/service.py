"""Inference entry point. Dispatches to mock (dev) or real (GPU box) per INFERENCE_MODE."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from .types import Detection, PipelineResult, PlateResult, ViolationResult
from . import mock

_ML_ROOT = Path(__file__).resolve().parents[1] / "ml"

# Maps the pipeline's internal violation labels -> the Supabase `violation_type` enum values.
_TYPE_MAP = {
    "no_helmet": "helmet",
    "no_seatbelt": "seatbelt",
    "wrong_side": "wrong_side",
    "triple_riding": "triple_riding",
    "stop_line": "stop_line",
    "red_light": "red_light",
    "illegal_parking": "illegal_parking",
}


def run_pipeline(image_bytes: bytes, media_type: str = "image") -> PipelineResult:
    mode = os.getenv("INFERENCE_MODE", "mock").lower()
    if mode == "real":
        return _run_real(image_bytes, media_type)
    return mock.run(image_bytes, media_type)


@lru_cache(maxsize=1)
def _get_real_pipeline():
    """Lazily build the ml/ Pipeline ONCE per process — loads RF-DETR/SAM-3/TrOCR/etc."""
    if str(_ML_ROOT) not in sys.path:
        sys.path.insert(0, str(_ML_ROOT))
    from src.pipeline import Pipeline, PipelineConfig  # noqa: E402

    cfg = PipelineConfig(
        output_dir=os.getenv("PIPELINE_OUTPUT_DIR", str(_ML_ROOT / "outputs" / "api")),
        variant=os.getenv("DETECTION_VARIANT", "large"),
        tier=os.getenv("PIPELINE_TIER", "cloud_required"),
        threshold=float(os.getenv("DETECTION_THRESHOLD", "0.3")),
        use_sam3=os.getenv("PIPELINE_USE_SAM3", "true").lower() != "false",
        use_vlm=os.getenv("PIPELINE_USE_VLM", "true").lower() != "false",
        use_roboflow=os.getenv("PIPELINE_USE_ROBOFLOW", "true").lower() != "false",
    )
    return Pipeline(cfg)


def _run_real(image_bytes: bytes, media_type: str) -> PipelineResult:
    import cv2
    import numpy as np

    pipe = _get_real_pipeline()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return PipelineResult(model_version="real-error: unreadable image")

    out = pipe.process_array(img)

    detections = [
        Detection(class_label=d["class_name"], confidence=d["confidence"],
                  bbox=_xyxy_to_norm(d["xyxy"], img.shape))
        for d in out.get("detection_list", [])
    ]

    violations: list[ViolationResult] = []
    for v in out.get("violations", []):
        if v.get("band") == "discard":
            continue  # discard band = not worth persisting as a violation record
        plate_result = None
        plate = out.get("plate", {}).get("plate")
        if plate and plate.get("text"):
            plate_result = PlateResult(
                plate_text=plate["text"], plate_normalized=plate["text"],
                state_code=plate.get("indian_validation", {}).get("normalized", "")[:2] or None,
                is_valid_format=plate.get("indian_validation", {}).get("format_valid"),
                ocr_confidence=plate.get("confidence"),
            )
        vlm = v.get("vlm") or {}
        violations.append(ViolationResult(
            violation_type=_TYPE_MAP.get(v["type"], v["type"]),
            confidence=v.get("calibrated_confidence", v.get("confidence", 0.0)),
            confidence_band=v.get("band", "human_review"),
            evidence={"bbox": v.get("bbox"), "vehicle_bbox": v.get("vehicle_bbox"),
                      "evidence_chain": v.get("evidence_chain"), "basis": v.get("basis")},
            vlm_caption=vlm.get("caption"),
            model_module=v["type"],
            model_version=out.get("image", "")[:0] or "pipeline-v1",
            plate=plate_result,
        ))

    annotated_bytes = None
    img_path = out.get("evidence_image_path")
    if img_path and Path(img_path).exists():
        annotated_bytes = Path(img_path).read_bytes()

    return PipelineResult(detections=detections, violations=violations,
                          annotated_image=annotated_bytes, model_version="pipeline-v1")


def _xyxy_to_norm(xyxy: list[float], shape) -> dict:
    h, w = shape[0], shape[1]
    x1, y1, x2, y2 = xyxy
    return {"x": round(x1 / w, 4), "y": round(y1 / h, 4),
            "w": round((x2 - x1) / w, 4), "h": round((y2 - y1) / h, 4)}
