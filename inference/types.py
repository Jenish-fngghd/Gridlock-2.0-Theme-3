"""Structured pipeline I/O shared by the mock and real inference paths."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Detection:
    class_label: str
    confidence: float
    bbox: dict          # {x, y, w, h} (normalized 0..1)
    track_id: int | None = None
    attributes: dict = field(default_factory=dict)


@dataclass
class PlateResult:
    plate_text: str
    plate_normalized: str
    state_code: str | None = None
    is_valid_format: bool | None = None
    ocr_confidence: float | None = None


@dataclass
class ViolationResult:
    violation_type: str           # must be a value of the violation_type enum
    confidence: float
    confidence_band: str          # auto_confirm | human_review | discard
    evidence: dict = field(default_factory=dict)
    vlm_caption: str | None = None
    model_module: str | None = None
    model_version: str | None = None
    plate: PlateResult | None = None


@dataclass
class PipelineResult:
    detections: list[Detection] = field(default_factory=list)
    violations: list[ViolationResult] = field(default_factory=list)
    annotated_image: bytes | None = None
    model_version: str = "mock-0.1"
