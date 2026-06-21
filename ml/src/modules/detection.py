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
                 keep_classes: set[str] | None = None, device: str | None = None,
                 resolution: int | None = None):
        self.variant = variant
        self.threshold = threshold
        self.keep_classes = keep_classes if keep_classes is not None else ROADUSER_CLASSES
        self.device = device
        self.resolution = resolution  # None = model default (640); 1280 for §3.4 high-res helmet
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
            kwargs = {}
            if self.resolution is not None:
                kwargs["resolution"] = self.resolution
            self.model = ctor(**kwargs)
            res_str = f"@{self.resolution}px" if self.resolution else ""
            log(f"[detection] RF-DETR-{self.variant}{res_str} loaded.")
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


# ---------------------------------------------------------------------------------------------
# Documented two-stage detection cascade (00_master_design.md §3.2):
#   Stage-1 fast SCREEN  = RF-DETR-N/S  (high recall, low threshold)
#   Stage-2 CONFIRM      = RF-DETR-L (default) OR Co-DETR (documented alternative, mmdet)
# The screen proposes candidates cheaply; the confirm pass re-checks candidate regions with the
# stronger model and keeps its refined box+score → speed (cheap screen) + precision (strong
# confirm). Both shippable RF-DETR variants are Apache-2.0.
# ---------------------------------------------------------------------------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class TwoStageDetector:
    """Documented Stage-1 screen -> Stage-2 confirm cascade (§3.2).

    stage2_model: "rfdetr-large" (default, Apache, runs on 4 GB) or "codetr" (documented
    alternative; requires mmdet/mmcv — flagged model_unavailable if not installed, never
    silently swapped).
    """

    def __init__(self, stage1_variant: str = "small", stage2_model: str = "rfdetr-large",
                 screen_threshold: float = 0.25, confirm_threshold: float = 0.5,
                 confirm_iou: float = 0.45, keep_classes: set[str] | None = None):
        self.screen_threshold = screen_threshold
        self.confirm_threshold = confirm_threshold
        self.confirm_iou = confirm_iou
        self.keep_classes = keep_classes if keep_classes is not None else ROADUSER_CLASSES
        self.stage2_kind = stage2_model
        # Stage-1: RF-DETR-N/S screen (low threshold for high recall)
        self.stage1 = VehicleDetector(variant=stage1_variant, threshold=screen_threshold,
                                      keep_classes=self.keep_classes)
        # Stage-2: RF-DETR-L confirm (or Co-DETR)
        self.stage2 = None
        self._stage2_reason = ""
        self._load_stage2(stage2_model, confirm_threshold)

    def _load_stage2(self, kind: str, threshold: float) -> None:
        if kind in ("rfdetr-large", "rfdetr-l", "large"):
            self.stage2 = VehicleDetector(variant="large", threshold=threshold,
                                          keep_classes=self.keep_classes)
            self.stage2_kind = "rfdetr-large"
        elif kind in ("codetr", "co-detr"):
            try:
                self.stage2 = CoDETRDetector(threshold=threshold, keep_classes=self.keep_classes)
                if self.stage2.model is None:
                    self._stage2_reason = self.stage2._unavailable_reason
                    self.stage2 = None
                self.stage2_kind = "codetr"
            except Exception as e:  # noqa: BLE001
                self._stage2_reason = f"{type(e).__name__}: {e}"
                self.stage2 = None
        else:
            self.stage2 = VehicleDetector(variant="large", threshold=threshold,
                                          keep_classes=self.keep_classes)
            self.stage2_kind = "rfdetr-large"

    def detect(self, image) -> DetectionResult:
        """Run the screen->confirm cascade. Never raises."""
        screen = self.stage1.detect(image)
        if screen.model_unavailable:
            return DetectionResult(model_unavailable=True, note=f"stage1 unavailable: {screen.note}")
        # No Stage-2 -> return the screen (graceful, flagged), do NOT pretend it's confirmed
        if self.stage2 is None:
            screen.note = (f"stage2 ({self.stage2_kind}) unavailable: {self._stage2_reason} "
                           f"-> returning Stage-1 screen only")
            return screen
        confirm = self.stage2.detect(image)
        if confirm.model_unavailable:
            screen.note = f"stage2 inference failed: {confirm.note} -> Stage-1 screen only"
            return screen
        # Confirm: keep a screened candidate only if Stage-2 agrees (same-class IoU match);
        # adopt Stage-2's refined box + score (higher precision).
        confirmed: list[Detection] = []
        for c in confirm.detections:
            if c.confidence < self.confirm_threshold:
                continue
            # require the screen to have proposed something overlapping (cascade gating)
            if any(s.class_name == c.class_name and _iou(s.xyxy, c.xyxy) >= self.confirm_iou
                   for s in screen.detections):
                confirmed.append(c)
        return DetectionResult(detections=confirmed,
                               note=f"two-stage: RF-DETR-{self.stage1.variant} screen "
                                    f"-> {self.stage2_kind} confirm "
                                    f"({len(screen.detections)} screened -> {len(confirmed)} confirmed)")


class CoDETRDetector:
    """Documented Stage-2 alternative — Co-DETR (ICCV'23, AICC helmet rank-1 backbone).

    Implemented via mmdetection. Heavy (mmdet/mmcv); if unavailable we report model_unavailable
    rather than substituting another model. COCO-class output, mapped to our names.
    """

    def __init__(self, threshold: float = 0.5, keep_classes: set[str] | None = None,
                 config: str | None = None, checkpoint: str | None = None):
        self.threshold = threshold
        self.keep_classes = keep_classes if keep_classes is not None else ROADUSER_CLASSES
        self.model = None
        self._unavailable_reason = ""
        try:
            from mmdet.apis import init_detector  # noqa: F401
            if not (config and checkpoint):
                raise RuntimeError("Co-DETR needs a config + checkpoint (mmdet). Provide --codetr-config/--codetr-ckpt.")
            from mmdet.apis import init_detector as _init
            self.model = _init(config, checkpoint, device="cuda:0")
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:140]}"
            self.model = None

    def detect(self, image) -> DetectionResult:
        if self.model is None:
            return DetectionResult(model_unavailable=True,
                                   note=f"Co-DETR unavailable: {self._unavailable_reason}")
        try:
            from mmdet.apis import inference_detector
            res = inference_detector(self.model, image if isinstance(image, str) else _to_numpy(image))
            dets = _codetr_to_detections(res, self.threshold, self.keep_classes)
            return DetectionResult(detections=dets, note="co-detr")
        except Exception as e:  # noqa: BLE001
            return DetectionResult(model_unavailable=True, note=f"co-detr inference error: {e}")


def _to_numpy(image):
    import numpy as np
    if hasattr(image, "mode"):
        return np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB->BGR for mmdet
    return np.asarray(image)


def _codetr_to_detections(res, threshold, keep_classes) -> list[Detection]:
    out: list[Detection] = []
    inst = getattr(res, "pred_instances", None)
    if inst is None:
        return out
    bboxes = inst.bboxes.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    labels = inst.labels.cpu().numpy()
    for b, s, lbl in zip(bboxes, scores, labels):
        if s < threshold:
            continue
        name = COCO_CLASSES.get(int(lbl) + 1, str(int(lbl)))  # mmdet labels 0-indexed
        if keep_classes is not None and name not in keep_classes:
            continue
        out.append(Detection(xyxy=tuple(float(v) for v in b[:4]), confidence=float(s),
                             class_id=int(lbl), class_name=name))
    return out
