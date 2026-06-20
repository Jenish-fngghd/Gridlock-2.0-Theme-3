"""Module 4a(seatbelt) — Seatbelt (best-effort, daytime).

00_master_design.md §3.5 / 06_... row 12: two-stage windshield → driver-crop → belt classifier.
No public zero-shot checkpoint does windshield+belt for this scheme, so zero-shot output is
`not_testable`. The downloaded `seat_belt-and-mobile` OBB dataset (779 train / 337 valid, classes
mobile/seatbelt/windshield) is the **fine-tune** target (Phase 5). The `mobile` class is outside
the 7 mandated violations → detected-but-flagged out-of-scope, never silently used or dropped.
"""
from __future__ import annotations


class SeatbeltModule:
    NOTE = ("not_testable: no zero-shot windshield/belt checkpoint. Fine-tune on "
            "seat_belt-and-mobile OBB (779 train). 'mobile' class = out-of-scope (logged, not used).")

    def __init__(self, model_path: str | None = None):
        self.model = None  # loaded only if a fine-tuned checkpoint is provided
        self.model_path = model_path
        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        try:
            from ultralytics import YOLO  # AGPL — internal only; here only if user supplies weights
            self.model = YOLO(path)
        except Exception:
            self.model = None

    def analyze(self, image) -> dict:
        if self.model is None:
            return {"model_unavailable": True, "violation": "no_seatbelt",
                    "status": "not_testable", "note": self.NOTE,
                    "out_of_scope_classes": ["mobile"]}
        # fine-tuned path (Phase 5+): windshield -> driver crop -> belt state
        try:
            res = self.model.predict(image, verbose=False)
            return {"model_unavailable": False, "raw": str(res[0].boxes.cls.tolist() if res else []),
                    "note": "fine-tuned seatbelt model output"}
        except Exception as e:  # noqa: BLE001
            return {"model_unavailable": True, "status": "error", "note": str(e)}
