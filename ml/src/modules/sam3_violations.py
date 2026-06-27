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
# NOTE: "person riding motorcycle" (the reference notebook's rider-specific prompt) was tried
# here to filter out bystanders, but SAM-3 does not ground that compound/relational phrase at all
# — it returned zero boxes even on an unambiguous 4-rider test image. Reverted to plain "person";
# bystander filtering is done geometrically instead (see _is_rider below).
RIDER_PROMPT = "person"
PROMPTS = ["motorcycle", RIDER_PROMPT, "helmet", "vehicle", "autorickshaw",
           "crosswalk", "traffic light"]

# per-concept confidence floors (helmet kept low — small/occluded; lights/lines need to be clear)
CONF = {"motorcycle": 0.5, RIDER_PROMPT: 0.5, "helmet": 0.3, "vehicle": 0.5,
        "autorickshaw": 0.4, "crosswalk": 0.4, "traffic light": 0.4}

# SAM-3 concepts that map to detection class names (fed back into the main detection list)
_DET_PROMPTS = {
    "motorcycle": "motorcycle",
    "person": "person",
    "vehicle": "car",
    "autorickshaw": "autorickshaw",
}


def _overlap(a, b, expand=15):
    ax1, ay1, ax2, ay2 = a[0] - expand, a[1] - expand, a[2] + expand, a[3] + expand
    bx1, by1, bx2, by2 = b
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def _union_box(boxes):
    xs1 = [b[0] for b in boxes]; ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]; ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _is_rider(person_box, moto_box, x_margin_frac=0.15):
    """A person counts as ON this motorcycle only if their horizontal center falls within the
    bike's own x-span (+ a small margin) AND they vertically overlap it. This is stricter than a
    plain expanded-box overlap, which was also catching pedestrians/bystanders standing next to or
    behind the bike (e.g. on a sidewalk) — they touch an expanded box but their center is well
    outside the bike's footprint."""
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = moto_box
    pcx = (px1 + px2) / 2
    margin = (mx2 - mx1) * x_margin_frac
    if not (mx1 - margin <= pcx <= mx2 + margin):
        return False
    return py1 < my2 + 20 and py2 > my1 - 20  # some vertical overlap with the bike (+ small slack)


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

    def analyze(self, img_bgr) -> dict:
        """Returns {"violations": [...], "detections": [...]}.
        violations: standard violation dicts {type, confidence, bbox, ...}
        detections: SAM-3 detected objects merged back into the main detection list —
                    each is {class_name, xyxy, confidence} matching Detection dataclass fields.
        """
        empty = {"violations": [], "detections": []}
        if not self.available():
            return empty
        # one SAM-3 call for all concepts; per-concept threshold filtering applied below
        det = self.sam3.detect_many(img_bgr, PROMPTS, conf=0.3)
        if det.get("_unavailable"):
            return empty

        def boxes(concept):
            return [d for d in det.get(concept, []) if d["conf"] >= CONF.get(concept, 0.5)]

        motos = boxes("motorcycle")
        riders = boxes(RIDER_PROMPT)
        helmets = boxes("helmet")
        vehicles = boxes("vehicle") or motos  # fall back to motos if 'vehicle' empty
        stoplines = boxes("crosswalk")  # crosswalk near-edge == stop-line position
        lights = boxes("traffic light")
        autorickshaws = boxes("autorickshaw")

        out: list[dict] = []

        # --- triple riding & helmet (per motorcycle, using its associated riders) ---
        for m in motos:
            mb = m["box"]
            riders_on_bike = [r for r in riders if _is_rider(r["box"], mb)]
            if len(riders_on_bike) >= 3:
                # The motorcycle's own SAM-3 box is often just the bike frame and can be too tight
                # to include all riders (e.g. cuts off heads) — the VLM verification step crops to
                # vehicle_bbox, so a too-tight box shows it fewer people than were actually counted
                # and it correctly denies what it can't see. Use the union of the bike + every
                # counted rider so the evidence crop always contains everyone being claimed.
                union = _union_box([mb] + [r["box"] for r in riders_on_bike])
                out.append(self._v("triple_riding", mb, min(0.9, m["conf"]),
                                    "sam3: motorcycle + >=3 riders",
                                    ["sam3:motorcycle", "sam3:person x3 (on-bike)",
                                     "rule:count>=3"], vehicle_box=union))
            # helmet: check each ACTUAL RIDER's own head region (top ~35% of THEIR box), not a
            # region derived from the motorcycle's box. The motorcycle mask from SAM-3 is often
            # just the vehicle frame and doesn't extend up to head height, so deriving "head
            # region" from it was structurally wrong and flagged riders wearing visible helmets.
            # Using the rider's own (person) box fixes this — it reliably includes their head.
            no_helmet_riders = []
            for r in riders_on_bike:
                rx1, ry1, rx2, ry2 = r["box"]
                head = [rx1, ry1, rx2, ry1 + (ry2 - ry1) * 0.35]
                if not any(_overlap(head, h["box"], expand=15) for h in helmets):
                    no_helmet_riders.append(r)
            if no_helmet_riders:
                # one violation per bike (not per rider) to avoid duplicate near-identical boxes.
                # bbox = the rider's own box (accurate for the on-screen marker); vehicle_bbox =
                # union of bike + that rider (same tight-crop fix as triple_riding above — the
                # VLM verification step crops to vehicle_bbox, so it must actually contain the
                # rider's head, not just whatever the motorcycle's own mask happened to cover).
                worst = max(no_helmet_riders, key=lambda r: r["conf"])
                out.append(self._v("no_helmet", worst["box"], min(0.8, worst["conf"]),
                                    "sam3: rider with no helmet over their own head region",
                                    ["sam3:person (on-bike)", "sam3:helmet absent over rider head"],
                                    vehicle_box=_union_box([mb, worst["box"]])))

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
        # Build detection list from SAM-3 boxes — deduplicated by class, feeding back
        # detected objects (motorcycle, person, vehicle, autorickshaw) into the main pipeline.
        seen: set[tuple] = set()
        dets: list[dict] = []
        for prompt, class_name in _DET_PROMPTS.items():
            src = motos if prompt == "motorcycle" else \
                  riders if prompt == "person" else \
                  autorickshaws if prompt == "autorickshaw" else \
                  boxes("vehicle")
            for d in src:
                key = (class_name, round(d["box"][0]), round(d["box"][1]))
                if key not in seen:
                    seen.add(key)
                    dets.append({"class_name": class_name,
                                 "xyxy": tuple(d["box"]),
                                 "confidence": d["conf"]})

        return {"violations": out, "detections": dets}

    # SAM-3 + geometric rules are heuristic, so they never auto-challan: clamp confidence below
    # every auto_confirm threshold (min is 0.80) so each lands in the human_review band -> the VLM
    # gives a second opinion and a human signs off. Serves "no wrong (auto) violations".
    _REVIEW_CAP = 0.79

    @staticmethod
    def _v(vtype, box, conf, basis, chain, vehicle_box=None):
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        vx1, vy1, vx2, vy2 = [int(round(c)) for c in (vehicle_box or box)]
        return {"type": vtype, "confidence": round(min(float(conf), SAM3Violations._REVIEW_CAP), 3),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "vehicle_bbox": [vx1, vy1, vx2, vy2],
                "evidence_chain": chain, "basis": basis}
