"""Module 5 — Violation classifier + confidence cascade.

00_master_design.md §3.8 / 06_... row 20: calibrate each violation's score (temperature
scaling + per-class thresholds) and sort into three bands — AUTO-CONFIRM (challan-eligible),
HUMAN-REVIEW (abstain), DISCARD. Low-confidence cases are flagged for VLM verification
(Module 6). This directly satisfies the brief's "assign confidence scores" requirement and
makes auto-challan defensible (admissibility, §11).

Pure-Python (no model deps). Temperature is optional and applied as a logit-space rescale of
a single probability; if no temperature is calibrated it is a no-op (T=1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Per-class operating thresholds (auto-confirm cutoff, review floor). Start values; these are
# the knobs Phase 4/5 tune against §7 operating points. review_floor < auto_confirm.
DEFAULT_THRESHOLDS = {
    # violation_type: (auto_confirm, review_floor)
    "no_helmet":        (0.85, 0.50),
    "triple_riding":    (0.80, 0.45),
    "no_seatbelt":      (0.85, 0.55),
    "red_light":        (0.90, 0.60),   # safety-critical -> stricter
    "wrong_side":       (0.85, 0.55),
    "stop_line":        (0.85, 0.55),
    "illegal_parking":  (0.80, 0.50),
}
AUTO_CONFIRM, HUMAN_REVIEW, DISCARD = "auto_confirm", "human_review", "discard"


@dataclass
class CascadeDecision:
    violation_type: str
    raw_confidence: float
    calibrated_confidence: float
    band: str
    needs_vlm: bool
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


class ConfidenceCascade:
    def __init__(self, thresholds: dict | None = None, temperatures: dict | None = None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.temperatures = temperatures or {}  # violation_type -> T (calibrated)

    def _calibrate(self, vtype: str, p: float) -> float:
        T = self.temperatures.get(vtype, 1.0)
        if T == 1.0 or p <= 0.0 or p >= 1.0:
            return p
        # temperature scaling on a single prob via its logit
        logit = math.log(p / (1.0 - p))
        z = logit / T
        return 1.0 / (1.0 + math.exp(-z))

    def decide(self, vtype: str, confidence: float) -> CascadeDecision:
        auto, floor = self.thresholds.get(vtype, (0.85, 0.50))
        cal = self._calibrate(vtype, float(confidence))
        if cal >= auto:
            # High model confidence — but still VLM-cross-check before auto-challan. The pipeline
            # only KEEPS auto_confirm if the VLM agrees (agreement-gate); on disagreement it drops
            # to human_review. We never auto-challan on a single model's say-so.
            band, needs_vlm = AUTO_CONFIRM, True
        elif cal >= floor:
            band, needs_vlm = HUMAN_REVIEW, True   # escalate uncertain cases to VLM (tiebreaker)
        else:
            band, needs_vlm = DISCARD, False
        return CascadeDecision(vtype, round(float(confidence), 4), round(cal, 4), band, needs_vlm,
                               note=f"auto>={auto},review>={floor}")

    def decide_many(self, violations: list[dict]) -> list[CascadeDecision]:
        """violations: [{'type':..., 'confidence':...}, ...]"""
        return [self.decide(v.get("type", ""), v.get("confidence", 0.0)) for v in violations]
