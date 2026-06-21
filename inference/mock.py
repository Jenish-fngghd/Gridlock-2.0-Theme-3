"""Deterministic fake pipeline output for local development.

Seeded by the image hash so the same upload always yields the same result —
lets the frontend/backend be built and demoed without GPU or model checkpoints.
"""
from __future__ import annotations
import hashlib
from .types import Detection, ViolationResult, PlateResult, PipelineResult

_VTYPES = ["helmet", "triple_riding", "seatbelt", "wrong_side", "red_light"]


def _band(conf: float) -> str:
    if conf >= 0.80:
        return "auto_confirm"
    if conf >= 0.50:
        return "human_review"
    return "discard"


def run(image_bytes: bytes, media_type: str = "image") -> PipelineResult:
    seed = int(hashlib.sha256(image_bytes).hexdigest(), 16)

    def nxt(n: int) -> int:
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
        return seed % n

    dets = [
        Detection("motorcycle", 0.91, {"x": 0.30, "y": 0.40, "w": 0.22, "h": 0.34}),
        Detection("person", 0.88, {"x": 0.33, "y": 0.18, "w": 0.11, "h": 0.26}),
    ]
    vtype = _VTYPES[nxt(len(_VTYPES))]
    conf = round(0.55 + nxt(40) / 100.0, 3)
    plate = PlateResult("MH12AB1234", "MH12AB1234", "MH", True, 0.82)

    viol = ViolationResult(
        violation_type=vtype,
        confidence=conf,
        confidence_band=_band(conf),
        evidence={"detection_idx": [0, 1], "note": "mock"},
        vlm_caption=f"[mock] {vtype.replace('_', ' ')} detected",
        model_module=vtype,
        model_version="mock-0.1",
        plate=plate,
    )
    return PipelineResult(detections=dets, violations=[viol], model_version="mock-0.1")
