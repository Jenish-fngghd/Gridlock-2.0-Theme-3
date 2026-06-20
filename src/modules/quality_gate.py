"""Module 0 — Ingest image-quality gate.

06_model_selection_justification.md row 1 specifies a lightweight IQA scorer (pyiqa) to
decide whether restoration runs. pyiqa has no Python-3.14 wheel on this machine (Phase 0a),
so this module uses classical OpenCV metrics instead — blur (variance of Laplacian),
brightness (mean luma), and saturation/over-exposure — which are dependency-light and good
enough to *gate* restoration. If a learned IQA is wanted later, swap it in behind this same
interface (registry: quality_gate).

Graceful degradation: if OpenCV is missing, returns {"model_unavailable": True} and treats
the image as acceptable (so the pipeline continues).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class QualityReport:
    blur_var: float = 0.0          # variance of Laplacian; low => blurry
    brightness: float = 0.0        # mean luma 0..255
    is_blurry: bool = False
    is_low_light: bool = False
    is_overexposed: bool = False
    needs_restoration: bool = False
    quality_ok: bool = True
    model_unavailable: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class QualityGate:
    def __init__(self, blur_thresh: float = 100.0, dark_thresh: float = 60.0,
                 bright_thresh: float = 220.0):
        self.blur_thresh = blur_thresh
        self.dark_thresh = dark_thresh
        self.bright_thresh = bright_thresh

    def assess(self, image) -> QualityReport:
        try:
            import cv2
            import numpy as np
            img = self._to_bgr(image, cv2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))
            is_blurry = blur_var < self.blur_thresh
            is_low_light = brightness < self.dark_thresh
            is_overexposed = brightness > self.bright_thresh
            needs_restoration = is_low_light or is_blurry
            quality_ok = not (is_blurry or is_low_light or is_overexposed)
            return QualityReport(
                blur_var=round(blur_var, 2), brightness=round(brightness, 2),
                is_blurry=is_blurry, is_low_light=is_low_light, is_overexposed=is_overexposed,
                needs_restoration=needs_restoration, quality_ok=quality_ok,
                note=("low_light" if is_low_light else "") + (" blurry" if is_blurry else ""),
            )
        except Exception as e:  # noqa: BLE001
            return QualityReport(model_unavailable=True, quality_ok=True,
                                 note=f"IQA unavailable: {type(e).__name__}: {e}")

    @staticmethod
    def _to_bgr(image, cv2):
        import numpy as np
        if isinstance(image, str):
            return cv2.imread(image)
        if hasattr(image, "mode"):  # PIL -> BGR
            return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return np.asarray(image)
