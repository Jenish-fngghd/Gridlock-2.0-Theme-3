"""Phase 5 — Signal-state zero-shot on sample violation images (§3.6, documented model).

00_master_design.md §3.6: signal-state classifier (LISA/BSTLD). Tier-0 approach: HSV
color thresholding on the traffic-light ROI (`SignalStateClassifier`). No model download,
no fine-tune needed.

This script runs the documented HSV classifier on the "Red light" sample images.
Since these images are full frames (not pre-cropped light ROIs), Stage-1 uses RF-DETR
to detect traffic lights (COCO class 9), then the HSV classifier runs on each detected
light crop. Hit = light detected AND classified as "red".

Dataset: sample images of violations/Red light/ (7 images, all known red-light violations).
Metric:  hit-rate (capability signal — no per-image GT boxes, same framing as 07_trackB).

Run:  python -m src.eval.eval_signal_zeroshot
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.signal_state import SignalStateClassifier
from src.modules.detection import VehicleDetector
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

SAMPLE_ROOT = (REPO_ROOT / "datasets" / "Helmet & Triple Riding" /
               "sample images of violations" / "sample images of violations")
REDLIGHT_DIR = SAMPLE_ROOT / "Red light"
TL_CLASS = "traffic light"


def run(conf: float) -> dict:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return {"error": f"opencv unavailable: {e}"}
    if not REDLIGHT_DIR.exists():
        return {"error": f"Red light sample folder not found: {REDLIGHT_DIR}"}

    detector = VehicleDetector(variant="nano", threshold=conf,
                               keep_classes={"traffic light"})
    clf = SignalStateClassifier()

    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in sorted(REDLIGHT_DIR.iterdir()) if p.suffix.lower() in exts]
    hits = 0
    results = []
    for fp in images:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        d = detector.detect(img)
        if d.model_unavailable:
            return {"error": f"RF-DETR unavailable: {d.note}"}
        lights = [x for x in d.detections if x.class_name == "traffic light"]
        states = []
        for lt in lights:
            s = clf.classify(img, lt.xyxy)
            states.append(s["state"])
        red_seen = "red" in states
        hits += int(red_seen)
        results.append({
            "image": fp.name,
            "traffic_lights_detected": len(lights),
            "states": states,
            "red_classified": red_seen,
            "hit": red_seen,
        })

    return {
        "classifier": "SignalStateClassifier (HSV Tier-0, §3.6 documented zero-shot approach)",
        "stage1_detector": "RF-DETR-nano (COCO class=traffic light, Apache-2.0)",
        "dataset": "sample images of violations/Red light/ (all known red-light violations)",
        "total_images": len(images),
        "images_with_red_classified": hits,
        "hit_rate": round(hits / max(len(images), 1), 3),
        "per_image": results,
        "note": ("Hit = traffic-light detected AND HSV classifier returns 'red'. "
                 "Capability signal only — no per-image GT boxes. "
                 "Full LISA benchmark is in eval_signal.py."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_signal_zeroshot_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Signal-state zero-shot — §3.6 documented model (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Classifier: {result['classifier']}")
    L.append(f"- Stage-1: {result['stage1_detector']}")
    L.append(f"- Dataset: {result['dataset']}")
    L.append(f"- Images: {result['total_images']} · Red classified: "
             f"{result['images_with_red_classified']} · **Hit-rate: {result['hit_rate']}**\n")
    L.append("| Image | Lights detected | States | Red hit |")
    L.append("|---|---|---|---|")
    for r in result["per_image"]:
        L.append(f"| {r['image']} | {r['traffic_lights_detected']} | "
                 f"{r['states']} | {'✅' if r['hit'] else '❌'} |")
    L.append(f"\n> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.15,
                    help="RF-DETR confidence threshold (0.15 catches borderline lights, was 0.25)")
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.conf)
    write_run_log("phase5", "signal_zeroshot", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_signal_zeroshot] ERROR: {result['error']}")
        return 1
    log(f"[eval_signal_zeroshot] hit-rate={result['hit_rate']} "
        f"({result['images_with_red_classified']}/{result['total_images']}) | report: {rp.name}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "signal_zeroshot",
                        "dataset": "sample-violation-images(redlight)", "model": "HSV+RF-DETR",
                        "metric": "hit_rate_proxy", "value": result["hit_rate"],
                        "target": "qualitative (capability signal)", "pass_fail": "zeroshot",
                        "note": f"red_hits={result['images_with_red_classified']}/{result['total_images']}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
