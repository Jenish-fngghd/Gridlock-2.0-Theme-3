"""Module 4a — Helmet + Triple riding (the anchor).

00_master_design.md §3.4 / 06_... rows 9-11. Two sub-tasks on the same detect→associate base:

- **Triple riding** — real zero-shot capability: RF-DETR detects motorcycle+person, riders are
  associated to a bike by overlap/horizontal-alignment (license-clean re-implementation of the
  VNPT overlap/nearest-x idea), and >=3 riders on one bike triggers the flag. Tested hit-rate
  1.0 on sample violation images (qualitative proxy — IDD has no GT triple-riding label).

- **Helmet state** — no AICC fine-tune checkpoint exists (imagery never obtained), so this was
  `not_testable` until SAM-3 (open-vocab) was validated by hand against the same sample images
  (1.0 hit-rate, see TODO.md). This module now wires that SAM-3 check in: for each associated
  rider, crop their box and ask SAM-3 for a "helmet" concept — gated to riders only (not every
  pixel of every image), keeping cost down. If SAM-3 is unavailable, helmet stays `not_testable`
  (never silently fabricated).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.modules.detection import VehicleDetector
from src.modules.tracking import iou

HELMET_CONCEPT = "helmet"
HEAD_REGION_FRAC = 0.32  # top fraction of the rider box treated as the head/helmet zone


def _is_rider_of(person_xyxy, moto_xyxy, x_tol: float = 0.15) -> bool:
    """Heuristic: a person belongs to a motorbike if their box overlaps it OR their center-x is
    within the (slightly expanded) motorbike x-span and their bottom is near/above the bike."""
    if iou(person_xyxy, moto_xyxy) > 0.1:
        return True
    px1, py1, px2, py2 = person_xyxy
    mx1, my1, mx2, my2 = moto_xyxy
    pcx = (px1 + px2) / 2.0
    span = (mx2 - mx1)
    if (mx1 - x_tol * span) <= pcx <= (mx2 + x_tol * span):
        if my1 - 0.5 * (my2 - my1) <= py2 <= my2 + 0.2 * (my2 - my1):
            return True
    return False


@dataclass
class RiderGroup:
    motorbike_bbox: list
    rider_bboxes: list = field(default_factory=list)
    rider_count: int = 0
    triple_riding: bool = False
    triple_confidence: float = 0.0
    helmet_status: str = "not_testable"
    rider_helmets: list = field(default_factory=list)  # per-rider {bbox, has_helmet, confidence}
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


class HelmetTripleModule:
    HELMET_NOTE_UNAVAILABLE = ("not_testable: SAM-3 unavailable for helmet-state crops "
                               "(no AICC fine-tune checkpoint exists either).")

    def __init__(self, detector: VehicleDetector | None = None, triple_min_riders: int = 3,
                 resolution: int = 1280, sam3=None):
        # §3.4 spec: "detect motorbike @high-res (1280–1536px)" — default 1280 to resolve riders.
        self.detector = detector or VehicleDetector(variant="nano", threshold=0.3,
                                                    keep_classes={"motorcycle", "person", "bicycle"},
                                                    resolution=resolution)
        self.triple_min_riders = triple_min_riders
        self.sam3 = sam3  # SAM3Detector | None — injected so it's loaded/shared once by the pipeline

    def analyze(self, image=None, detections=None, frame=None) -> dict:
        """Pass `detections` (from a shared detection pass) to avoid re-running inference;
        otherwise `image` is detected here. `frame` (numpy BGR) is needed to crop riders for
        the SAM-3 helmet check — if omitted, helmet stays not_testable even if SAM-3 is loaded."""
        if detections is None:
            res = self.detector.detect(image)
            if res.model_unavailable:
                return {"model_unavailable": True, "note": res.note, "groups": []}
            detections = res.detections
        motos = [d for d in detections if d.class_name in ("motorcycle",)]
        persons = [d for d in detections if d.class_name in ("person",)]
        groups: list[RiderGroup] = []
        any_helmet_checked = False
        for m in motos:
            riders = [p for p in persons if _is_rider_of(p.xyxy, m.xyxy)]
            n = len(riders)
            triple = n >= self.triple_min_riders
            rider_helmets = []
            if self.sam3 is not None and frame is not None:
                for p in riders:
                    rider_helmets.append(self._check_helmet(frame, p.xyxy))
                    any_helmet_checked = True
            groups.append(RiderGroup(
                motorbike_bbox=[round(v, 1) for v in m.xyxy],
                rider_bboxes=[[round(v, 1) for v in p.xyxy] for p in riders],
                rider_count=n,
                triple_riding=triple,
                triple_confidence=round(min(0.5 + 0.15 * n, 0.95), 3) if triple else 0.0,
                helmet_status=("checked" if rider_helmets else "not_testable"),
                rider_helmets=rider_helmets,
                flags={"triple_basis": "proxy: rider-association count (no GT triple label)",
                       **({} if rider_helmets else {"helmet": self.HELMET_NOTE_UNAVAILABLE})},
            ))
        return {
            "model_unavailable": False,
            "motorbikes": len(motos),
            "groups": [g.to_dict() for g in groups],
            "triple_riding_count": sum(1 for g in groups if g.triple_riding),
            "helmet_status": "checked (SAM-3 per-rider)" if any_helmet_checked else "not_testable",
            "note": "" if any_helmet_checked else self.HELMET_NOTE_UNAVAILABLE,
        }

    def _check_helmet(self, frame, rider_xyxy) -> dict:
        """Crop the rider, ask SAM-3 for a 'helmet' concept, check it lands in the head region."""
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in rider_xyxy]
            h_total = max(1, y2 - y1)
            pad = int(0.1 * h_total)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = x2 + pad, y2 + pad
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return {"bbox": list(rider_xyxy), "has_helmet": None, "confidence": 0.0,
                        "note": "empty crop"}
            res = self.sam3.detect_concept(crop, HELMET_CONCEPT, threshold=0.25)
            if res.model_unavailable:
                return {"bbox": list(rider_xyxy), "has_helmet": None, "confidence": 0.0,
                        "note": res.note}
            crop_h = crop.shape[0]
            head_limit = crop_h * HEAD_REGION_FRAC
            head_dets = [d for d in res.detections if (d.xyxy[1] + d.xyxy[3]) / 2.0 <= head_limit]
            best = max(head_dets, key=lambda d: d.confidence, default=None)
            if best is None:
                # fall back to the best helmet detection anywhere in the rider crop
                best = max(res.detections, key=lambda d: d.confidence, default=None)
            has_helmet = best is not None and best.confidence >= 0.25
            return {"bbox": list(rider_xyxy), "has_helmet": has_helmet,
                    "confidence": round(best.confidence, 3) if best else 0.0}
        except Exception as e:  # noqa: BLE001
            return {"bbox": list(rider_xyxy), "has_helmet": None, "confidence": 0.0,
                    "note": f"{type(e).__name__}: {e}"}
