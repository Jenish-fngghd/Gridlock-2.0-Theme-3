"""Module 2 — Detection (RF-DETR family, Apache-2.0).

Primary detector per 06_model_selection_justification.md (rows 4–5): the RF-DETR family is
the license-clean spine for both stages. On the local `cloud_required` tier (4 GB VRAM) we
default to the Nano variant and a small inference size; RF-DETR-L / Co-DETR Stage-2 are not
loaded locally (stubbed) and belong on the cloud H200.

Zero-shot, the model emits COCO classes — so it covers car/motorcycle/bus/truck/person/
bicycle/traffic-light but NOT Indian-specific classes (auto-rickshaw, cart, animal-cart).
That structural gap is the domain gap §6/§7 and the reason the data-engine + fine-tune exist.

Graceful degradation: if rfdetr/torch are unavailable or weights fail to load, detect()
returns an empty result tagged {"model_unavailable": True} and never raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import log

# RF-DETR emits 1-indexed COCO ids ({1:person, 2:bicycle, 3:car, 4:motorcycle, ... 8:truck,
# 10:traffic light}). Use rfdetr's own mapping when available; fall back to the correct
# 1-indexed table (NOT 0-indexed — that off-by-one silently mislabels every class).
try:
    from rfdetr.util.coco_classes import COCO_CLASSES as _RF_COCO  # type: ignore
    COCO_CLASSES = dict(_RF_COCO)
except Exception:  # noqa: BLE001
    COCO_CLASSES = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
        7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
        13: "stop sign", 14: "parking meter", 15: "bench", 17: "cat", 18: "dog",
        19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    }
# Traffic-relevant subset we keep from the COCO head.
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
ROADUSER_CLASSES = VEHICLE_CLASSES | {"person", "traffic light"}

# RF-DETR variant by hardware tier (06_... + Phase 0a).
TIER_VARIANT = {
    "cloud_required": "nano",
    "low": "nano",
    "mid": "small",
    "high": "large",
}


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    model_unavailable: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_unavailable": self.model_unavailable,
            "note": self.note,
            "detections": [
                {"xyxy": [round(v, 2) for v in d.xyxy], "confidence": round(d.confidence, 4),
                 "class_id": d.class_id, "class_name": d.class_name}
                for d in self.detections
            ],
        }


class VehicleDetector:
    """RF-DETR detector wrapper with graceful degradation."""

    def __init__(self, variant: str = "nano", threshold: float = 0.5,
                 keep_classes: set[str] | None = None, device: str | None = None):
        self.variant = variant
        self.threshold = threshold
        self.keep_classes = keep_classes if keep_classes is not None else ROADUSER_CLASSES
        self.device = device
        self.model = None
        self._unavailable_reason = ""
        self._load()

    @classmethod
    def for_tier(cls, tier: str, **kw) -> "VehicleDetector":
        return cls(variant=TIER_VARIANT.get(tier, "nano"), **kw)

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from rfdetr import (RFDETRBase, RFDETRLarge, RFDETRMedium,  # noqa: F401
                                RFDETRNano, RFDETRSmall)
            variants = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium,
                        "base": RFDETRBase, "large": RFDETRLarge}
            ctor = variants.get(self.variant, RFDETRNano)
            self.model = ctor()
            log(f"[detection] RF-DETR-{self.variant} loaded.")
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:120]}"
            self.model = None
            log(f"[detection] model_unavailable: {self._unavailable_reason}")

    def detect(self, image) -> DetectionResult:
        """image: PIL.Image | np.ndarray (BGR or RGB) | path str. Never raises."""
        if self.model is None:
            return DetectionResult(model_unavailable=True,
                                   note=f"RF-DETR not loaded: {self._unavailable_reason}")
        try:
            pil = self._to_pil(image)
            preds = self.model.predict(pil, threshold=self.threshold)
            dets = self._from_supervision(preds)
            if self.keep_classes is not None:
                dets = [d for d in dets if d.class_name in self.keep_classes]
            return DetectionResult(detections=dets)
        except Exception as e:  # noqa: BLE001
            return DetectionResult(model_unavailable=True, note=f"inference error: {type(e).__name__}: {e}")

    # --- helpers ---
    @staticmethod
    def _to_pil(image):
        from PIL import Image
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if hasattr(image, "mode"):  # already PIL
            return image.convert("RGB")
        # numpy array
        import numpy as np
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            # assume BGR (opencv) -> RGB
            arr = arr[:, :, ::-1]
        return Image.fromarray(arr)

    @staticmethod
    def _from_supervision(preds) -> list[Detection]:
        out: list[Detection] = []
        xyxy = getattr(preds, "xyxy", None)
        conf = getattr(preds, "confidence", None)
        cid = getattr(preds, "class_id", None)
        if xyxy is None:
            return out
        for i in range(len(xyxy)):
            c = int(cid[i]) if cid is not None else -1
            out.append(Detection(
                xyxy=tuple(float(v) for v in xyxy[i]),
                confidence=float(conf[i]) if conf is not None else 0.0,
                class_id=c,
                class_name=COCO_CLASSES.get(c, str(c)),
            ))
        return out
