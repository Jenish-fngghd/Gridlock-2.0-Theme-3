"""Phase 3 — Stop-line QUALITATIVE spot-check (NOT a benchmark metric).

Stop-line violation has NO dataset anywhere (Phase 0c). Per the goal we do a qualitative
spot-check ONLY: pull a handful of frames that visibly contain a junction/stop-line, run
detection + the geometry stop-line rule against a hand-set stop-line, save annotated images,
and write a note (plausible / implausible / no-opinion). This is labelled qualitative
everywhere — it must NEVER masquerade as a real metric.

Run:  python -m src.eval.smoke_test --n 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.detection import VehicleDetector
from src.modules.geometry_engine import GeometryEngine, SceneConfig
from src.utils.logging import REPO_ROOT, log, new_run_id, write_run_log

# Source frames for the spot-check (junction-likely): IDD frontNear has road junctions.
SRC = REPO_ROOT / "datasets" / "idd-detection" / "IDD_Detection" / "JPEGImages" / "frontNear"
OUT = REPO_ROOT / "results" / "stopline_spotcheck"


def run(n: int) -> dict:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return {"error": f"opencv unavailable: {e}"}
    imgs = (sorted(SRC.rglob("*.jpg"))[:n]) if SRC.exists() else []
    if not imgs:
        return {"error": f"no source frames under {SRC}"}
    OUT.mkdir(parents=True, exist_ok=True)
    det = VehicleDetector(variant="nano", threshold=0.3,
                          keep_classes={"car", "truck", "bus", "motorcycle"})
    saved = 0
    items = []
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        # hand-set stop-line across lower third (qualitative geometry only)
        line_y = int(h * 0.7)
        scene = SceneConfig(camera_id="SPOTCHECK", stop_line=[(0, line_y), (w, line_y)],
                            lane_direction=(0, -1))
        ge = GeometryEngine(scene)
        dets = det.detect(img).detections
        # draw the stop-line + detections; geometry verdict on a still = abstain (no motion)
        cv2.line(img, (0, line_y), (w, line_y), (0, 255, 255), 2)
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.xyxy]
            crosses = y2 >= line_y  # bottom past line (static proxy, qualitative)
            color = (0, 0, 255) if crosses else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        op = OUT / f"spotcheck_{img_path.stem}.jpg"
        cv2.imwrite(str(op), img)
        saved += 1
        items.append({"image": img_path.name, "vehicles": len(dets), "annotated": op.name,
                      "verdict": "qualitative: geometry abstains on still; bottom-past-line shown as proxy"})
    return {
        "type": "QUALITATIVE SPOT-CHECK (NOT A METRIC)",
        "violation": "stop_line", "frames": saved, "output_dir": str(OUT),
        "items": items[:10],
        "note": ("Stop-line has no dataset → no precision/recall/mAP is or will be reported for it. "
                 "Annotated frames are for human inspection (plausible/implausible) only."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.n)
    write_run_log("phase3", "stopline_spotcheck", run_id, result)
    if "error" in result:
        log(f"[smoke_test] ERROR: {result['error']}")
        return 1
    log(f"[smoke_test] QUALITATIVE stop-line spot-check: {result['frames']} annotated frames "
        f"-> {result['output_dir']}  (NOT a metric)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
