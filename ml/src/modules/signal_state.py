"""Signal-state classifier (red / yellow / green) — Tier 0, rule-based.

Feeds the geometry engine's red-light rule (Module 4b). Per 00_master_design.md §3.6 the
design names a learned signal-state classifier (LISA/BSTLD), but color-of-light is a problem
HSV thresholding solves well without training — so this is a **Tier 0** module (revised Phase 4:
"no fine-tuning by default"). If it clears its target zero-shot, it is never fine-tuned.

Given a cropped traffic-light ROI, count red/yellow/green pixels in HSV and return the
dominant color. Pure OpenCV; degrades to `model_unavailable` if cv2 is missing.
"""
from __future__ import annotations

# HSV ranges (OpenCV H in 0..179). Red wraps around 0.
_RED1 = ((0, 70, 50), (10, 255, 255))
_RED2 = ((160, 70, 50), (179, 255, 255))
_YELLOW = ((15, 70, 50), (35, 255, 255))
_GREEN = ((40, 40, 40), (90, 255, 255))


def classify_crop(crop) -> dict:
    """crop: numpy BGR image of the light ROI. Returns {state, scores, model_unavailable}."""
    try:
        import cv2
        import numpy as np
        if crop is None or getattr(crop, "size", 0) == 0:
            return {"state": "unknown", "model_unavailable": False, "note": "empty crop"}
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        def count(lo, hi):
            return int(cv2.inRange(hsv, np.array(lo), np.array(hi)).sum() // 255)

        red = count(*_RED1) + count(*_RED2)
        yellow = count(*_YELLOW)
        green = count(*_GREEN)
        scores = {"red": red, "yellow": yellow, "green": green}
        total = red + yellow + green
        if total == 0:
            return {"state": "unknown", "scores": scores, "model_unavailable": False}
        state = max(scores, key=scores.get)
        return {"state": state, "scores": scores, "confidence": round(scores[state] / total, 3),
                "model_unavailable": False}
    except Exception as e:  # noqa: BLE001
        return {"state": "unknown", "model_unavailable": True, "note": f"{type(e).__name__}: {e}"}


class SignalStateClassifier:
    """Crop a light box from a frame and classify its color."""

    def classify(self, image_bgr, box_xyxy) -> dict:
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
            crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
            return classify_crop(crop)
        except Exception as e:  # noqa: BLE001
            return {"state": "unknown", "model_unavailable": True, "note": str(e)}
