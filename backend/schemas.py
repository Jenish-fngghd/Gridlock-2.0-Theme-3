"""Pydantic response models for the API."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class PlateOut(BaseModel):
    plate_text: str
    plate_normalized: str
    state_code: str | None = None
    is_valid_format: bool | None = None
    ocr_confidence: float | None = None


class ViolationOut(BaseModel):
    id: str | None = None
    violation_type: str
    confidence: float | None = None
    confidence_band: str
    annotated_image_url: str | None = None
    vlm_caption: str | None = None
    plate: PlateOut | None = None
    evidence: dict[str, Any] = {}


class DetectionOut(BaseModel):
    class_label: str
    confidence: float | None = None
    # bbox is normalized 0..1 — {x, y, w, h} with (x,y) the top-left corner.
    bbox: dict[str, float]
    track_id: int | None = None


class StageOut(BaseModel):
    """One real pipeline stage and what it produced — drives the Detect step rail."""
    key: str
    label: str
    ran: bool
    detail: str = ""


class ProcessOut(BaseModel):
    job_id: str | None = None
    status: str
    processing_ms: int
    persisted: bool
    model_version: str | None = None
    annotated_image_url: str | None = None
    detections: list[DetectionOut] = []
    stages: list[StageOut] = []
    violations: list[ViolationOut] = []
