"""Module 3 — Tracking.

06_model_selection_justification.md row 8 specifies BoostTrack (via boxmot). boxmot is not
installed on this Python-3.14 machine (and BoostTrack needs torch-ReID weights that don't fit
the 4 GB tier), so this module provides a dependency-free IoU/centroid tracker as the graceful
fallback the design already names (BoT-SORT/ByteTrack → here a simple IoU tracker). It assigns
persistent IDs and maintains the per-track history the geometry engine (Module 4b) needs
(bottom-center trajectory + frames_static for dwell-time).

When boxmot/BoostTrack become available (cloud), swap behind this same `update()` interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    id: int
    xyxy: tuple
    class_name: str
    history: list = field(default_factory=list)   # [(bottom_cx, bottom_cy), ...]
    frames_static: int = 0
    misses: int = 0

    def as_dict(self) -> dict:
        return {"id": self.id, "xyxy": list(self.xyxy), "class_name": self.class_name,
                "history": self.history, "frames_static": self.frames_static}


class IoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_misses: int = 10, static_eps: float = 4.0):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.static_eps = static_eps
        self._next_id = 1
        self.tracks: list[Track] = []
        self.backend = "iou_fallback"  # registry: not BoostTrack on this tier

    @staticmethod
    def _bottom_center(xyxy):
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2.0, y2)

    def update(self, detections: list) -> list[Track]:
        """detections: objects with .xyxy and .class_name (e.g. Detection). Returns live tracks."""
        dets = [(d.xyxy, getattr(d, "class_name", "")) for d in detections]
        unmatched = set(range(len(dets)))
        # greedy IoU matching against existing tracks
        for tr in self.tracks:
            best_j, best_iou = -1, self.iou_thresh
            for j in unmatched:
                v = iou(tr.xyxy, dets[j][0])
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                new_box = dets[best_j][0]
                bc_old = tr.history[-1] if tr.history else self._bottom_center(tr.xyxy)
                bc_new = self._bottom_center(new_box)
                moved = ((bc_new[0] - bc_old[0]) ** 2 + (bc_new[1] - bc_old[1]) ** 2) ** 0.5
                tr.frames_static = tr.frames_static + 1 if moved < self.static_eps else 0
                tr.xyxy = new_box
                tr.history.append(bc_new)
                tr.misses = 0
                unmatched.discard(best_j)
            else:
                tr.misses += 1
        # spawn new tracks
        for j in unmatched:
            box, cls = dets[j]
            self.tracks.append(Track(id=self._next_id, xyxy=box, class_name=cls,
                                     history=[self._bottom_center(box)]))
            self._next_id += 1
        # cull dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.misses == 0]
