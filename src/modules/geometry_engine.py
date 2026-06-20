"""Module 4b — Geometry rule engine / Scene Context Model.

00_master_design.md §3.6 / 01_justifications.md J3 / 06_... row 13: a per-camera, annotated-
once Scene Context Model (stop-line, no-parking polygon, lane-direction vectors, signal ROI)
drives the temporal/scene-context violations — interpretable, auditable, court-admissible,
and needing no (nonexistent) labelled temporal dataset.

Handles: wrong-side, stop-line, red-light, illegal parking. Pure-Python geometry (no shapely
dep): point-in-polygon (ray casting), segment intersection, vector dot for lane direction.

Inputs per camera come from a config (see configs/camera_config.json). Tracks come from the
tracker (Module 3); a single still degrades gracefully (no motion -> rule abstains with a
`needs_clip`/`needs_signal_state` flag rather than guessing).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ----------------------------- geometry primitives -----------------------------
def point_in_polygon(pt, poly) -> bool:
    """Ray-casting point-in-polygon. poly: [(x,y),...]."""
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1p2 intersects p3p4."""
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def bbox_bottom_center(xyxy) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)  # ground-contact proxy


@dataclass
class SceneConfig:
    camera_id: str = "CAM_UNKNOWN"
    stop_line: list = field(default_factory=list)        # [(x,y),(x,y)] a segment
    no_parking_polygon: list = field(default_factory=list)  # [(x,y),...]
    lane_direction: tuple = (0.0, 1.0)                   # unit-ish vector of legal travel
    signal_roi: list = field(default_factory=list)       # polygon around the signal head
    parking_dwell_frames: int = 30                       # frames stationary => parking

    @classmethod
    def from_dict(cls, d: dict) -> "SceneConfig":
        def tup_list(v):
            return [tuple(p) for p in v] if v else []
        return cls(
            camera_id=d.get("camera_id", "CAM_UNKNOWN"),
            stop_line=tup_list(d.get("stop_line")),
            no_parking_polygon=tup_list(d.get("no_parking_polygon")),
            lane_direction=tuple(d.get("lane_direction", (0.0, 1.0))),
            signal_roi=tup_list(d.get("signal_roi")),
            parking_dwell_frames=int(d.get("parking_dwell_frames", 30)),
        )


@dataclass
class GeoViolation:
    type: str
    track_id: int
    confidence: float
    bbox: list
    band_hint: str = "human_review"   # geometry rules are corroborating evidence -> review by default
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


class GeometryEngine:
    def __init__(self, scene: SceneConfig):
        self.scene = scene

    # ---- per-violation rules. A "track" = {id, xyxy, history:[(x,y)...], velocity, frames_static} ----
    def check_wrong_side(self, track: dict) -> GeoViolation | None:
        hist = track.get("history", [])
        if len(hist) < 2:
            return None  # needs motion
        dx = hist[-1][0] - hist[0][0]
        dy = hist[-1][1] - hist[0][1]
        if dx == 0 and dy == 0:
            return None
        lx, ly = self.scene.lane_direction
        dot = dx * lx + dy * ly
        if dot < 0:  # moving opposite to legal lane direction
            mag = (dx * dx + dy * dy) ** 0.5
            conf = min(0.5 + 0.5 * (-dot) / (mag * ((lx*lx+ly*ly) ** 0.5) + 1e-9), 0.95)
            return GeoViolation("wrong_side", track.get("id", -1), round(conf, 3),
                                list(track.get("xyxy", [])), flags={"basis": "trajectory_vs_lane"})
        return None

    def check_stop_line(self, track: dict, permitted: bool = False) -> GeoViolation | None:
        if permitted or len(self.scene.stop_line) != 2:
            return None
        hist = track.get("history", [])
        if len(hist) < 2:
            return None
        p1, p2 = self.scene.stop_line
        if segments_intersect(hist[-2], hist[-1], p1, p2):
            return GeoViolation("stop_line", track.get("id", -1), 0.7,
                                list(track.get("xyxy", [])), flags={"basis": "crossed_stop_line"})
        return None

    def check_red_light(self, track: dict, signal_state: str | None) -> GeoViolation | None:
        if signal_state is None:
            # cannot judge without signal state -> abstain, flag for the signal classifier
            return None
        if signal_state.lower() != "red":
            return None
        crossed = self.check_stop_line(track, permitted=False)
        if crossed:
            return GeoViolation("red_light", track.get("id", -1), 0.85,
                                list(track.get("xyxy", [])),
                                flags={"basis": "red_AND_crossing", "signal": "red"})
        return None

    def check_illegal_parking(self, track: dict) -> GeoViolation | None:
        if len(self.scene.no_parking_polygon) < 3:
            return None
        bc = bbox_bottom_center(track.get("xyxy", (0, 0, 0, 0)))
        if not point_in_polygon(bc, self.scene.no_parking_polygon):
            return None
        if track.get("frames_static", 0) >= self.scene.parking_dwell_frames:
            return GeoViolation("illegal_parking", track.get("id", -1), 0.8,
                                list(track.get("xyxy", [])),
                                flags={"basis": "dwell_in_no_park", "frames_static": track.get("frames_static")})
        return None

    def evaluate(self, tracks: list[dict], signal_state: str | None = None) -> list[GeoViolation]:
        out: list[GeoViolation] = []
        for t in tracks:
            for v in (self.check_wrong_side(t), self.check_stop_line(t),
                      self.check_red_light(t, signal_state), self.check_illegal_parking(t)):
                if v is not None:
                    out.append(v)
        return out
