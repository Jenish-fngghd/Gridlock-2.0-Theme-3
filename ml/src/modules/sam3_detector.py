"""SAM-3 open-vocabulary detector (facebook/sam3) — fills the detection gap on weak/India classes.

The documented RF-DETR two-stage cascade is strong on COCO-canonical classes (car/truck/person)
but scores ~0 on the India-specific / out-of-COCO classes (auto-rickshaw, traffic-sign, animal,
vehicle-fallback) and is weak on traffic-light/bicycle. SAM-3 takes a text concept and returns
boxes + scores for ANY class, so we use it as an open-vocab detector to cover exactly those gaps
(00_master_design.md §3.10 data-engine / open-vocab role).

Gated model — needs an approved HF token in env var HF_TOKEN (never hard-coded here).
Weights ~3.2 GB: tries GPU (bf16) first, falls back to CPU if VRAM is insufficient (honest,
no silent failure).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.modules.detection import Detection, DetectionResult

MODEL_ID = "facebook/sam3"


class SAM3Detector:
    def __init__(self, threshold: float = 0.3, device: str | None = None):
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.device = device
        self._unavailable_reason = ""
        self._load()

    def _load(self) -> None:
        try:
            import os
            import torch
            from transformers import Sam3Model, Sam3Processor
            token = os.environ.get("HF_TOKEN")  # must be set by the caller; not stored
            self.processor = Sam3Processor.from_pretrained(MODEL_ID, token=token)
            want = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if want == "cuda":
                try:
                    self.model = Sam3Model.from_pretrained(MODEL_ID, token=token,
                                                           dtype=torch.bfloat16).cuda()
                    self.device = "cuda"
                except Exception as e:  # noqa: BLE001 - VRAM too small -> CPU fallback
                    import gc
                    self.model = None
                    gc.collect(); torch.cuda.empty_cache()
                    self.model = Sam3Model.from_pretrained(MODEL_ID, token=token)
                    self.device = "cpu"
                    self._unavailable_reason = f"GPU load failed ({type(e).__name__}); using CPU"
            else:
                self.model = Sam3Model.from_pretrained(MODEL_ID, token=token)
                self.device = "cpu"
            self.model.eval()
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = f"{type(e).__name__}: {str(e)[:140]}"
            self.model = None

    def detect_concept(self, image, concept: str, threshold: float | None = None) -> DetectionResult:
        """Run text-prompted detection for one concept. Returns DetectionResult (never raises)."""
        if self.model is None:
            return DetectionResult(model_unavailable=True,
                                   note=f"SAM-3 unavailable: {self._unavailable_reason}")
        try:
            import torch
            pil = self._to_pil(image)
            thr = self.threshold if threshold is None else threshold
            inputs = self.processor(images=pil, text=concept, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            res = self.processor.post_process_object_detection(
                outputs, threshold=thr, target_sizes=[(pil.height, pil.width)])[0]
            dets = []
            boxes = res.get("boxes"); scores = res.get("scores")
            if boxes is not None:
                for b, s in zip(boxes.tolist(), scores.tolist()):
                    dets.append(Detection(xyxy=tuple(float(v) for v in b[:4]),
                                          confidence=float(s), class_id=-1, class_name=concept))
            return DetectionResult(detections=dets, note=f"sam3:{concept} ({self.device})")
        except Exception as e:  # noqa: BLE001
            return DetectionResult(model_unavailable=True,
                                   note=f"sam3 inference error: {type(e).__name__}: {e}")

    @staticmethod
    def _to_pil(image):
        from PIL import Image
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if hasattr(image, "mode"):
            return image.convert("RGB")
        import numpy as np
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        return Image.fromarray(arr)
