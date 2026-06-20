"""Phase 3 — Wrong-side driving evaluation (Wrong-Way OBB dataset, frame-level).

The design produces a wrong-side verdict via the geometry rule engine (trajectory vs lane
direction). That needs MOTION, so on the single still frames in this dataset the geometry
engine necessarily abstains → the zero-shot baseline for the wrong-side *verdict* is
`not_testable` (Tier 2). Per the goal, that means: if this baseline misses §7, promote to a
direct learned classifier fine-tuned on this very dataset (Phase 5 / Paradigm-A pattern).

What we CAN measure now, to ground that escalation:
  * the dataset is OBB → exercise src/utils/obb_convert.py (polygon→AABB) and report angle stats
    (the rotation angle is a heading proxy that could supplement monocular-3D yaw, §0b OBB note);
  * whether the zero-shot detector even localizes the labeled vehicles (detection recall @IoU0.5
    vs the GT boxes, ignoring the right/wrong class) — confirms a working base for the classifier.

Run:  python -m src.eval.eval_wrongside --split valid --limit 91
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.detection import VehicleDetector
from src.modules.tracking import iou
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)
from src.utils.obb_convert import parse_obb_label_file

WW_ROOT = (REPO_ROOT / "datasets" / "Wrong Side Driving" /
           "Wrong Way Driving Detection.v1i.yolov8-obb")
NAMES = {0: "right-side", 1: "wrong-side"}


def load_split(split: str, limit: int):
    img_dir = WW_ROOT / split / "images"
    lbl_dir = WW_ROOT / split / "labels"
    out = []
    if not img_dir.exists():
        return out
    for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            out.append((img, lbl))
        if limit and len(out) >= limit:
            break
    return out


def run(split: str, limit: int) -> dict:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return {"error": f"opencv unavailable: {e}"}
    samples = load_split(split, limit)
    if not samples:
        return {"error": f"no Wrong-Way samples in split={split} under {WW_ROOT}"}
    log(f"[eval_wrongside] {len(samples)} frames (split={split})")

    det = VehicleDetector(variant="nano", threshold=0.3,
                          keep_classes={"car", "truck", "bus", "motorcycle", "bicycle"})
    gt_total = 0
    gt_by_class = {0: 0, 1: 0}
    matched = 0
    angles = []
    for img_path, lbl_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        recs = parse_obb_label_file(lbl_path)
        # GT AABB in pixels
        gts = []
        for r in recs:
            x1, y1, x2, y2 = r["aabb_xyxy"]
            gts.append((r["class_id"], [x1 * w, y1 * h, x2 * w, y2 * h]))
            gt_by_class[r["class_id"]] = gt_by_class.get(r["class_id"], 0) + 1
            angles.append(r["angle_deg"])
        gt_total += len(gts)
        # detection recall (class-agnostic) vs GT
        dts = [d.xyxy for d in det.detect(img).detections]
        for _cid, gbox in gts:
            if any(iou(gbox, db) >= 0.5 for db in dts):
                matched += 1

    det_recall = matched / gt_total if gt_total else 0.0
    angle_stats = {"min": round(min(angles), 1), "max": round(max(angles), 1),
                   "mean": round(sum(angles) / len(angles), 1)} if angles else {}
    return {
        "dataset": f"Wrong-Way OBB ({split})", "frames": len(samples),
        "gt_vehicles": gt_total, "gt_by_class": {NAMES[k]: v for k, v in gt_by_class.items()},
        "wrongside_verdict": "not_testable (geometry needs motion; single stills) → Tier 2",
        "detector_recall_classagnostic@0.5": round(det_recall, 4),
        "obb_angle_stats_deg": angle_stats,
        "tier": "tier_2 (promote to learned classifier on this dataset if baseline < §7)",
        "note": ("Zero-shot wrong-side verdict is not computable from a still (rule engine "
                 "abstains). Detector localizes vehicles at the recall above → solid base for the "
                 "Phase-5 learned-classifier escalation. OBB angle kept as a heading proxy."),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_wrongside_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Wrong-side eval — Wrong-Way OBB (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Frames: {result['frames']} · GT vehicles: {result['gt_vehicles']} "
             f"({result['gt_by_class']})")
    L.append(f"- **Wrong-side verdict (zero-shot): {result['wrongside_verdict']}**")
    L.append(f"- Detector recall (class-agnostic @IoU0.5): **{result['detector_recall_classagnostic@0.5']}**")
    L.append(f"- OBB angle stats (heading proxy): {result['obb_angle_stats_deg']}")
    L.append(f"- Tier: {result['tier']}\n")
    L.append(f"> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def run_learned(split: str, weights: str) -> dict:
    """Evaluate the Phase-5 learned wrong-side classifier on held-out crops."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch/torchvision unavailable: {e}"}
    from src.train.train_wrongside import CropDataset, evaluate_model

    ckpt = Path(weights)
    if not ckpt.exists():
        return {"error": f"checkpoint not found: {ckpt}"}
    blob = torch.load(ckpt, map_location="cpu")
    import torch.nn as nn
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((blob.get("input", 128),) * 2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = CropDataset(split, tfm)
    if len(ds) == 0:
        return {"error": f"no crops in split={split}"}
    counts = [0, 0]
    for _t, c in ds.items:
        counts[c] += 1
    metrics = evaluate_model(model, DataLoader(ds, batch_size=32), "cpu")
    return {
        "dataset": f"Wrong-Way OBB ({split}, held-out)", "model": blob.get("backbone", "?") + " (fine-tuned)",
        "instances": len(ds), "class_counts": {"right-side": counts[0], "wrong-side": counts[1]},
        "metrics": metrics,
        "baseline_geometry": "not_testable (abstains on stills)",
        "verdict": "LEARNED classifier replaces geometry for the wrong-side verdict (Tier 2 promoted)",
    }


def write_learned_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_wrongside_learned_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Wrong-side eval (LEARNED) — Wrong-Way OBB (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    m = result["metrics"]
    L.append(f"- Model: {result['model']}  ·  {result['dataset']}")
    L.append(f"- Instances: {result['instances']} {result['class_counts']}\n")
    L.append("## Baseline → Learned (the Phase-5 escalation)\n")
    L.append("| Approach | Wrong-side verdict |")
    L.append("|---|---|")
    L.append(f"| Geometry rule (zero-shot) | {result['baseline_geometry']} |")
    L.append(f"| **Learned classifier (fine-tuned)** | **acc {m['accuracy']} · "
             f"wrong-side P {m['wrongside_precision']} / R {m['wrongside_recall']} / "
             f"F1 {m['wrongside_f1']}** |")
    L.append(f"\n> {result['verdict']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=91)
    ap.add_argument("--weights", default=None,
                    help="if given, evaluate the LEARNED classifier on --split instead of the geometry baseline")
    args = ap.parse_args()
    run_id = new_run_id()

    if args.weights:
        result = run_learned(args.split, args.weights)
        write_run_log("phase5", "wrongside_eval", run_id, result)
        rp = write_learned_report(result, run_id)
        if "error" in result:
            log(f"[eval_wrongside:learned] ERROR: {result['error']}")
            return 1
        m = result["metrics"]
        log(f"[eval_wrongside:learned] {args.split} acc={m['accuracy']} "
            f"wrong-side F1={m['wrongside_f1']} (report: {rp.name})")
        append_run_history({"run_id": run_id, "phase": "phase5", "module": "wrongside",
                            "dataset": f"Wrong-Way({args.split})", "model": "mobilenetv3s-ft",
                            "metric": "wrongside_f1", "value": m["wrongside_f1"],
                            "target": "beat geometry abstain", "pass_fail": "finetuned",
                            "note": f"acc={m['accuracy']} P={m['wrongside_precision']} R={m['wrongside_recall']}"})
        return 0

    result = run(args.split, args.limit)
    write_run_log("phase3", "wrongside", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_wrongside] ERROR: {result['error']}")
        return 1
    log(f"[eval_wrongside] verdict=not_testable(Tier2) det_recall="
        f"{result['detector_recall_classagnostic@0.5']} (report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "wrongside",
                        "dataset": "Wrong-Way", "model": "geometry-rule", "metric": "verdict",
                        "value": "not_testable", "target": "§7 P/R/F1", "pass_fail": "tier2",
                        "note": f"det_recall={result['detector_recall_classagnostic@0.5']}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
