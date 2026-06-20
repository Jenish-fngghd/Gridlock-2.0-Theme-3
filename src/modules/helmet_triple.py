"""Module 4a — Helmet + Triple riding (the anchor).

00_master_design.md §3.4 / 06_... rows 9–11. The design solves both violations with the AICC
Track-5 7-class scheme (per-rider helmet state + rider count) on a detect→crop→classify
pipeline, with SAM2+pose association.

Reality on this machine (Phase 0):
- **Helmet compliance is NOT testable zero-shot.** No public checkpoint does the AICC per-rider
  helmet-state classification, AND the AICC imagery is absent on disk (only the code repo). Per
  the goal, we DO NOT pass a generic detector off as a helmet result — helmet output is tagged
  `not_testable: requires AICC fine-tune`. (Tier 1.)
- **Triple riding runs as a proxy.** We CAN detect motorcycles + persons/riders zero-shot (RF-DETR
  COCO) and associate riders to a motorbike by overlap + horizontal alignment (license-clean
  re-implementation of the VNPT overlap/nearest-x idea), then flag ≥3 riders on one motorbike.
  IDD has `rider`+`motorcycle` so this is demonstrable, but there is no GT triple-riding label →
  reported as a qualitative proxy, never a fabricated metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.modules.detection import VehicleDetector
from src.modules.tracking import iou


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
        # rider sits above/on the bike: person bottom within bike vertical band
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
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


class HelmetTripleModule:
    HELMET_NOTE = ("not_testable: no zero-shot checkpoint for AICC per-rider helmet state; "
                   "AICC imagery absent on disk. Requires Phase-5 fine-tune (Tier 1).")

    def __init__(self, detector: VehicleDetector | None = None, triple_min_riders: int = 3):
        self.detector = detector or VehicleDetector(variant="nano", threshold=0.3,
                                                    keep_classes={"motorcycle", "person", "bicycle"})
        self.triple_min_riders = triple_min_riders

    def analyze(self, image=None, detections=None) -> dict:
        """Pass `detections` (from a shared detection pass) to avoid re-running inference;
        otherwise `image` is detected here."""
        if detections is None:
            res = self.detector.detect(image)
            if res.model_unavailable:
                return {"model_unavailable": True, "note": res.note, "groups": []}
            detections = res.detections
        motos = [d for d in detections if d.class_name in ("motorcycle",)]
        persons = [d for d in detections if d.class_name in ("person",)]
        groups: list[RiderGroup] = []
        for m in motos:
            riders = [p for p in persons if _is_rider_of(p.xyxy, m.xyxy)]
            n = len(riders)
            triple = n >= self.triple_min_riders
            groups.append(RiderGroup(
                motorbike_bbox=[round(v, 1) for v in m.xyxy],
                rider_bboxes=[[round(v, 1) for v in p.xyxy] for p in riders],
                rider_count=n,
                triple_riding=triple,
                triple_confidence=round(min(0.5 + 0.15 * n, 0.95), 3) if triple else 0.0,
                helmet_status="not_testable",
                flags={"helmet": self.HELMET_NOTE,
                       "triple_basis": "proxy: rider-association count (no GT triple label)"},
            ))
        return {
            "model_unavailable": False,
            "motorbikes": len(motos),
            "groups": [g.to_dict() for g in groups],
            "triple_riding_count": sum(1 for g in groups if g.triple_riding),
            "helmet_status": "not_testable",
            "note": self.HELMET_NOTE,
        }
