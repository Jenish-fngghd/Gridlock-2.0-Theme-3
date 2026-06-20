"""Module 1 — Preprocessing (conditional, quality-gated).

00_master_design.md §3.1 / 06_... rows 2–3 specify Retinexformer (low-light) and OneRestore
(weather). Neither has an installable Python-3.14 wheel + fitting weights on the 4 GB tier, so
they are reported `model_unavailable` and we fall back to classical, dependency-light
enhancement (CLAHE + gamma for low-light, mild unsharp for blur). This is enough to *not* hurt
the detector and to demonstrate the quality-gated branch; the learned restorers swap in behind
this same `restore()` interface on the cloud.

Restoration runs ONLY when the quality gate flags a problem (avoids artifacts + wasted compute,
per the design's "skip if good" rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreprocessResult:
    applied: list = field(default_factory=list)
    learned_restorer_available: bool = False
    note: str = ""
    image = None  # restored image (numpy BGR) or original

    def to_dict(self) -> dict:
        return {"applied": self.applied, "learned_restorer_available": self.learned_restorer_available,
                "note": self.note}


class Preprocessor:
    def __init__(self, enable: bool = True):
        self.enable = enable
        # learned restorers (Retinexformer/OneRestore) not available on this tier
        self.learned_available = False

    def restore(self, image, quality) -> PreprocessResult:
        """image: numpy/PIL/path. quality: QualityReport. Returns PreprocessResult (.image set)."""
        res = PreprocessResult(learned_restorer_available=self.learned_available)
        try:
            import cv2
            import numpy as np
            img = self._to_bgr(image, cv2, np)
            res.image = img
            if not self.enable or quality is None or not getattr(quality, "needs_restoration", False):
                res.note = "skipped (quality ok or disabled)"
                return res
            out = img
            if getattr(quality, "is_low_light", False):
                out = self._clahe_gamma(out, cv2, np)
                res.applied.append("clahe+gamma (fallback for Retinexformer)")
            if getattr(quality, "is_blurry", False):
                out = self._unsharp(out, cv2)
                res.applied.append("unsharp (fallback for deblur)")
            res.image = out
            res.note = "model_unavailable: Retinexformer/OneRestore not installed → classical fallback"
            return res
        except Exception as e:  # noqa: BLE001
            res.note = f"preprocess error: {type(e).__name__}: {e}"
            return res

    @staticmethod
    def _to_bgr(image, cv2, np):
        if isinstance(image, str):
            return cv2.imread(image)
        if hasattr(image, "mode"):
            return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return np.asarray(image).copy()

    @staticmethod
    def _clahe_gamma(img, cv2, np, gamma: float = 1.4):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img2 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        inv = 1.0 / gamma
        table = (np.array([((i / 255.0) ** inv) * 255 for i in range(256)])).astype("uint8")
        return cv2.LUT(img2, table)

    @staticmethod
    def _unsharp(img, cv2):
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)
