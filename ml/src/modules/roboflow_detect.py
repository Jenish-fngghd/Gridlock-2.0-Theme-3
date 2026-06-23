"""Roboflow hosted-inference client (prototype stage — §3.10 interim).

For the prototype we lean on Roboflow Universe hosted models for the violation classes where our
own single-image models don't yet generalise (wrong-side, illegal-parking). Later these get
replaced by models trained on the combined datasets. Each call is a serverless inference request
(image -> detections); graceful: missing key / network error -> model_unavailable, never raises.

Per-class model + decision rule is configured in _CLASS_CONFIG (overridable via the
ROBOFLOW_MODELS env var as JSON: {"wrong_side": "model/ver", ...}). Selected on the head-to-head
eval against the labelled sample folder (see RESULTS.md / ml docs).
"""
from __future__ import annotations

import base64
import json
import os

_BASE = "https://serverless.roboflow.com"

# violation_type -> {model, match (substring in class name, "" = any), min_conf}
# Only wrong-side: a dedicated heading classifier (outputs right-side/wrong-side), eval 2/2 + 0/7
# FP on samples — the single best detector we have. (The illegal-parking model was removed: it was
# scene-blind, firing on every parked car incl. legal ones. SAM-3 handles helmet/triple/red-light.)
_CLASS_CONFIG = {
    "wrong_side": {"model": "wrong-way-driving-detection/2", "match": "wrong", "min_conf": 0.5},
}

# Helmet has no usable "no-helmet" class on the available models, so it's an inference rule on a
# rider/helmet detector: a 'rider' with no 'with helmet' box over their head = no-helmet (interim,
# eval ~3/6 recall · 4/5 specificity; replaced by SAM-3 on GPU). Separate path from _CLASS_CONFIG.
_HELMET_MODEL = "helmet-violation-49gqq/1"
_HELMET_RIDER_CONF = 0.3
_HELMET_HELMET_CONF = 0.4


class RoboflowDetector:
    def __init__(self, api_key: str | None = None, timeout: float = 25.0):
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self.timeout = timeout
        self.config = dict(_CLASS_CONFIG)
        override = os.environ.get("ROBOFLOW_MODELS", "")
        if override:
            try:
                for vt, model in json.loads(override).items():
                    self.config.setdefault(vt, {"match": "", "min_conf": 0.5})["model"] = model
            except Exception:  # noqa: BLE001
                pass

    def available(self) -> bool:
        return bool(self.api_key)

    def supports(self, violation_type: str) -> bool:
        return violation_type in self.config

    def detect(self, image, violation_type: str) -> dict:
        """Run the configured Roboflow model for `violation_type` on a BGR/np image or path.
        Returns {model_unavailable, fired: bool, confidence: float, boxes: [...], note}."""
        cfg = self.config.get(violation_type)
        if not self.api_key or cfg is None:
            return {"model_unavailable": True, "fired": False, "confidence": 0.0, "boxes": [],
                    "note": "no api key" if not self.api_key else f"no model for {violation_type}"}
        try:
            import requests

            b64 = self._encode(image)
            url = f"{_BASE}/{cfg['model']}?api_key={self.api_key}&confidence=5"
            resp = requests.post(url, data=b64,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"},
                                 timeout=self.timeout)
            if resp.status_code != 200:
                return {"model_unavailable": True, "fired": False, "confidence": 0.0, "boxes": [],
                        "note": f"HTTP {resp.status_code}: {resp.text[:120]}"}
            preds = resp.json().get("predictions", [])
            match, minc = cfg.get("match", ""), cfg.get("min_conf", 0.5)
            hits = [p for p in preds
                    if p.get("confidence", 0) >= minc and (not match or match in p.get("class", "").lower())]
            best = max((p.get("confidence", 0.0) for p in hits), default=0.0)
            boxes = [{"class": p.get("class"), "confidence": round(p.get("confidence", 0), 4),
                      "xyxy": self._to_xyxy(p)} for p in hits]
            return {"model_unavailable": False, "fired": len(hits) > 0,
                    "confidence": round(float(best), 4), "boxes": boxes,
                    "model": cfg["model"], "note": ""}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "fired": False, "confidence": 0.0, "boxes": [],
                    "note": f"{type(e).__name__}: {str(e)[:120]}"}

    def detect_helmet(self, image) -> dict:
        """No-helmet via inference rule on the helmet detector: a 'rider' with no 'with helmet'
        box over their head = a no-helmet violation. Returns {model_unavailable, fired, boxes}."""
        if not self.api_key:
            return {"model_unavailable": True, "fired": False, "boxes": [], "note": "no api key"}
        try:
            import requests

            b64 = self._encode(image)
            url = f"{_BASE}/{_HELMET_MODEL}?api_key={self.api_key}&confidence=5"
            resp = requests.post(url, data=b64,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"},
                                 timeout=self.timeout)
            if resp.status_code != 200:
                return {"model_unavailable": True, "fired": False, "boxes": [],
                        "note": f"HTTP {resp.status_code}"}
            preds = resp.json().get("predictions", [])
            riders = [p for p in preds if p.get("class") == "rider"
                      and p.get("confidence", 0) >= _HELMET_RIDER_CONF]
            helmets = [p for p in preds if p.get("class") == "with helmet"
                       and p.get("confidence", 0) >= _HELMET_HELMET_CONF]
            boxes = []
            for r in riders:
                if not any(self._helmet_over_head(r, h) for h in helmets):
                    boxes.append({"class": "no_helmet", "confidence": round(r.get("confidence", 0), 4),
                                  "xyxy": self._to_xyxy(r)})
            return {"model_unavailable": False, "fired": len(boxes) > 0, "boxes": boxes,
                    "model": _HELMET_MODEL}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "fired": False, "boxes": [], "note": str(e)[:120]}

    @staticmethod
    def _helmet_over_head(rider: dict, helmet: dict) -> bool:
        def bx(p):
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        rx1, ry1, rx2, ry2 = bx(rider)
        hx1, hy1, hx2, hy2 = bx(helmet)
        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        return rx1 <= hcx <= rx2 and ry1 <= hcy <= (ry1 + 0.6 * (ry2 - ry1))

    @staticmethod
    def _encode(image) -> str:
        from pathlib import Path
        if isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        import cv2  # BGR np array -> JPEG bytes
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode()

    @staticmethod
    def _to_xyxy(p: dict) -> list[float]:
        # Roboflow gives center x,y + w,h
        x, y, w, h = p.get("x", 0), p.get("y", 0), p.get("width", 0), p.get("height", 0)
        return [round(x - w / 2, 1), round(y - h / 2, 1), round(x + w / 2, 1), round(y + h / 2, 1)]
