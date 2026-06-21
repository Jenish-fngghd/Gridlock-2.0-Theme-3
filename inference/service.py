"""Inference entry point. Dispatches to mock (dev) or real (GPU box) per INFERENCE_MODE."""
from __future__ import annotations
import os
from .types import PipelineResult
from . import mock


def run_pipeline(image_bytes: bytes, media_type: str = "image") -> PipelineResult:
    mode = os.getenv("INFERENCE_MODE", "mock").lower()
    if mode == "real":
        return _run_real(image_bytes, media_type)
    return mock.run(image_bytes, media_type)


def _run_real(image_bytes: bytes, media_type: str) -> PipelineResult:
    # TODO: wire ml/src/modules/pipeline.py here once running on the AWS GPU box.
    #   - load checkpoints once at process start (module-level singletons)
    #   - decode image_bytes -> ndarray, run detect -> per-paradigm modules -> ANPR
    #   - draw annotations -> encode annotated_image bytes
    #   - map outputs into Detection / ViolationResult / PlateResult
    raise NotImplementedError(
        "real inference not wired yet; run with INFERENCE_MODE=mock for development"
    )
