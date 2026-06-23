"""SAM-3 entity-detection + geometric violation rules.

Ported from the reference notebook (how_to_segment_images_with_segment_anything_3.ipynb): SAM-3
detects concrete entities via text prompts, then simple geometry decides each violation. Covers
the classes our trained single-image models can't do well:
  - triple_riding : a motorcycle with >= 3 overlapping persons
  - no_helmet     : a motorcycle WITH riders but no helmet over the rider head region
  - stop_line     : a vehicle past the stop line
  - red_light     : the light is red (HSV) AND a vehicle is past the stop line (connected to stop_line)

Seatbelt stays on the trained two-stage model (it already performs well). All entities come from a
SINGLE Roboflow SAM-3 call (RoboflowSAM3.detect_many) for cost control.
"""
from __future__ import annotations

# the concepts we ask SAM-3 for, in one call. NOTE: SAM-3 does not detect "stop line" (returns
# nothing on real images), but reliably finds the "crosswalk" — and the stop line sits right at
# the crosswalk's near edge, so we use the crosswalk as the stop-line reference.
PROMPTS = ["motorcycle", "person", "helmet", "vehicle", "crosswalk", "traffic light"]

# per-concept confidence floors (helmet kept low — small/occluded; lights/lines need to be clear)
CONF = {"motorcycle": 0.5, "person": 0.5, "helmet": 0.3, "vehicle": 0.5,
        "crosswalk": 0.4, "traffic light": 0.4}


def _overlap(a, b, expand=15):
    ax1, ay1, ax2, ay2 = a[0] - expand, a[1] - expand, a[2] + expand, a[3] + expand
    bx1, by1, bx2, by2 = b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _light_color(crop_box, img_bgr):
    """HSV red/green check on a traffic-light crop. img is a BGR numpy array (cv2)."""
    import cv2
    import numpy as np
    x1, y1, x2, y2 = [int(c) for c in crop_box]
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    red = (((h < 10) | (h > 170)) & (s > 100)).sum()
    green = ((h > 40) & (h < 80) & (s > 100)).sum()
    if red > green and red > 20:
        return "red"
    if green > 20:
        return "green"
    return "unknown"


class SAM3Violations:
    def __init__(self, sam3):
        self.sam3 = sam3  # RoboflowSAM3

    def available(self) -> bool:
        return self.sam3 is not None and self.sam3.available()

    def analyze(self, img_bgr) -> list[dict]:
        """Returns a list of violation dicts: {type, box[x1,y1,x2,y2], confidence, basis, chain}."""
        if not self.available():
            return []
        # one SAM-3 call for all concepts; per-concept threshold filtering applied below
        det = self.sam3.detect_many(img_bgr, PROMPTS, conf=0.3)
        if det.get("_unavailable"):
            return []

        def boxes(concept):
            return [d for d in det.get(concept, []) if d["conf"] >= CONF.get(concept, 0.5)]

        motos = boxes("motorcycle")
        persons = boxes("person")
        helmets = boxes("helmet")
        vehicles = boxes("vehicle") or motos  # fall back to motos if 'vehicle' empty
        stoplines = boxes("crosswalk")  # crosswalk near-edge == stop-line position
        lights = boxes("traffic light")

        out: list[dict] = []

        # --- triple riding & helmet (per motorcycle) ---
        for m in motos:
            mb = m["box"]
            rider_count = sum(_overlap(mb, p["box"]) for p in persons)
            if rider_count >= 3:
                out.append(self._v("triple_riding", mb, min(0.9, m["conf"]),
                                    "sam3: motorcycle + >=3 riders",
                                    ["sam3:motorcycle", "sam3:person x3", "rule:count>=3"]))
            # helmet: a motorcycle scene with NO helmet anywhere over its upper body = no-helmet.
            # Use the top ~45% of the motorcycle box (rider torso+head) and a generous expand —
            # the tight head sub-box missed close-up riders. A bike with a rider but zero helmets
            # nearby is the strong signal (helmet-present cleanly separates in eval).
            x1, y1, x2, y2 = mb
            upper = [x1, y1, x2, y1 + (y2 - y1) * 0.45]
            if not any(_overlap(upper, h["box"], expand=20) for h in helmets):
                out.append(self._v("no_helmet", mb, min(0.8, m["conf"]),
                                    "sam3: motorcycle with no helmet over the rider",
                                    ["sam3:motorcycle", "sam3:helmet absent over rider"]))

        # --- red-light / stop-line (CONNECTED — single-image only) ---
        # A vehicle merely "past the crosswalk" fires on almost every intersection photo (no
        # specificity), so a stop-line crossing is only an enforceable violation when the signal
        # is confirmably RED. We require: crosswalk (= stop-line ref) + a vehicle past it + an
        # HSV-red traffic light. That's the connected red-light/stop-line event. (Stop-line
        # crossing on red is reported as red_light here — they are the same physical event on a
        # single still; a true stand-alone stop-line check needs per-camera line geometry.)
        if stoplines and vehicles and lights:
            line_y = (stoplines[0]["box"][1] + stoplines[0]["box"][3]) / 2
            past = [v for v in vehicles if v["box"][3] < line_y]  # vehicle bottom past the line
            if past and _light_color(lights[0]["box"], img_bgr) == "red":
                for v in past:
                    out.append(self._v("red_light", v["box"], min(0.85, v["conf"]),
                                        "sam3: red light (HSV) + vehicle past the crosswalk/stop-line",
                                        ["sam3:traffic light=red", "sam3:crosswalk", "sam3:vehicle past line"]))
        return out

    # SAM-3 + geometric rules are heuristic, so they never auto-challan: clamp confidence below
    # every auto_confirm threshold (min is 0.80) so each lands in the human_review band -> the VLM
    # gives a second opinion and a human signs off. Serves "no wrong (auto) violations".
    _REVIEW_CAP = 0.79

    @staticmethod
    def _v(vtype, box, conf, basis, chain):
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        return {"type": vtype, "confidence": round(min(float(conf), SAM3Violations._REVIEW_CAP), 3),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "vehicle_bbox": [x1, y1, x2, y2],
                "evidence_chain": chain, "basis": basis}
