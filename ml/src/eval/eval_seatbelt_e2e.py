"""Phase 5 — Seatbelt end-to-end two-stage evaluation (§3.5, real pipeline, no GT-crop shortcut).

`train_seatbelt.py` fine-tuned the belt classifier on GROUND-TRUTH windshield crops; that gives
a classifier-only number (best F1=0.678) but skips the Stage-1 detector the design actually
names ("YOLOv11/windshield detector → driver crop → CNN/CNN-SVM belt classifier"). This script
chains the real Stage-1 (`train_windshield_detector.py` output) into Stage-2 (the existing belt
classifier) and scores the IoU-matched result against GT — the honest §3.5 pipeline number,
which will usually sit at or below the GT-crop number since it now also pays for Stage-1
localization error.

Run:  python -m src.eval.eval_seatbelt_e2e --det checkpoints/windshield/v1/weights/best.pt \
          --clf checkpoints/seatbelt/v2/model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)
from src.utils.obb_convert import parse_obb_label_file

SB_ROOT = (REPO_ROOT / "datasets" / "seat belt detection" /
           "seat_belt and mobile.v2i.yolov8-obb")
BELT = 1
WINDSHIELD = 2


def _contain_frac(inner, outer) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    x1, y1, x2, y2 = max(ix1, ox1), max(iy1, oy1), min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (ix2 - ix1) * (iy2 - iy1)
    return inter / a if a > 0 else 0.0


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def load_gt(split: str):
    """Per image: list of (windshield_xyxy_px, label) where label=1 is no_seatbelt (violation)."""
    import cv2
    img_dir = SB_ROOT / split / "images"
    lbl_dir = SB_ROOT / split / "labels"
    out = []
    if not img_dir.exists():
        return out
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        recs = parse_obb_label_file(lbl)
        winds = [r["aabb_xyxy"] for r in recs if r["class_id"] == WINDSHIELD]
        belts = [r["aabb_xyxy"] for r in recs if r["class_id"] == BELT]
        gts = []
        for x1, y1, x2, y2 in winds:
            has_belt = any(_contain_frac(b, (x1, y1, x2, y2)) > 0.5 for b in belts)
            label = 0 if has_belt else 1
            gts.append(((x1 * w, y1 * h, x2 * w, y2 * h), label))
        out.append((img_path, gts))
    return out


def prf1(preds, labels, positive=1) -> dict:
    tp = sum(1 for p, y in zip(preds, labels) if p == positive and y == positive)
    fp = sum(1 for p, y in zip(preds, labels) if p == positive and y != positive)
    fn = sum(1 for p, y in zip(preds, labels) if p != positive and y == positive)
    tn = sum(1 for p, y in zip(preds, labels) if p != positive and y != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(preds), 1)
    return {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def run(det_weights: str, clf_weights: str, split: str, conf: float, iou_thresh: float) -> dict:
    try:
        import cv2
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        return {"error": f"deps unavailable: {e}"}

    det_path = Path(det_weights)
    clf_path = Path(clf_weights)
    if not det_path.exists():
        return {"error": f"detector checkpoint not found: {det_path} (run train_windshield_detector.py first)"}
    if not clf_path.exists():
        return {"error": f"classifier checkpoint not found: {clf_path} (run train_seatbelt.py first)"}

    detector = YOLO(str(det_path))
    blob = torch.load(clf_path, map_location="cpu")
    backbone_key = blob.get("backbone", "mobilenet_v3_small")
    if "large" in backbone_key:
        clf = models.mobilenet_v3_large(weights=None)
    else:
        clf = models.mobilenet_v3_small(weights=None)
    clf.classifier[-1] = nn.Linear(clf.classifier[-1].in_features, 2)
    clf.load_state_dict(blob["state_dict"])
    clf.eval()
    input_size = blob.get("input", 128)
    tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    gt_by_image = load_gt(split)
    if not gt_by_image:
        return {"error": f"no GT windshield instances in split={split}"}
    log(f"[eval_seatbelt_e2e] {len(gt_by_image)} images (split={split})")

    preds, labels = [], []
    n_det_total = n_matched = n_gt_total = 0
    for img_path, gts in gt_by_image:
        n_gt_total += len(gts)
        img = cv2.imread(str(img_path))
        res = detector.predict(str(img_path), conf=conf, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().tolist()
        n_det_total += len(boxes)
        used_gt = set()
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            best_idx, best_iou = None, 0.0
            for gi, (gbox, _label) in enumerate(gts):
                if gi in used_gt:
                    continue
                v = _iou((x1, y1, x2, y2), gbox)
                if v > best_iou:
                    best_iou, best_idx = v, gi
            if best_idx is None or best_iou < iou_thresh:
                continue  # detection doesn't correspond to a labeled windshield — skip, don't guess
            used_gt.add(best_idx)
            n_matched += 1
            crop = Image.fromarray(img[y1:y2, x1:x2][:, :, ::-1])
            with torch.no_grad():
                pred = int(clf(tfm(crop).unsqueeze(0)).argmax(1).item())
            preds.append(pred)
            labels.append(gts[best_idx][1])

    metrics = prf1(preds, labels) if preds else {"error": "no IoU-matched detections"}
    return {
        "dataset": f"seat_belt-and-mobile OBB ({split})", "detector": str(det_path),
        "classifier": str(clf_path), "classifier_backbone": backbone_key,
        "gt_windshields": n_gt_total, "detections": n_det_total,
        "iou_matched": n_matched, "scored_pairs": len(preds),
        "end_to_end_metrics": metrics,
        "gt_crop_baseline_f1": 0.678,
        "note": ("End-to-end uses the detector's OWN boxes (conf>=%.2f), IoU>=%.2f matched to GT "
                 "windshields; unmatched detections are skipped, not guessed. Compare "
                 "end_to_end_metrics.f1 against the GT-crop baseline (0.678, classifier fed perfect "
                 "boxes) to see Stage-1 detector's contribution to error." % (conf, iou_thresh)),
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_seatbelt_e2e_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Seatbelt end-to-end two-stage eval (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    m = result["end_to_end_metrics"]
    L.append(f"- Dataset: {result['dataset']}")
    L.append(f"- Classifier backbone: {result.get('classifier_backbone', 'mobilenet_v3_small')}")
    L.append(f"- GT windshields: {result['gt_windshields']} · detector boxes: {result['detections']} "
             f"· IoU-matched: {result['iou_matched']} · scored pairs: {result['scored_pairs']}\n")
    L.append("## §3.5 two-stage — GT-crop baseline vs end-to-end\n")
    L.append("| Stage | no_seatbelt F1 |")
    L.append("|---|---|")
    L.append(f"| Classifier-only (GT crops, `train_seatbelt.py` v2) | {result['gt_crop_baseline_f1']} |")
    if "error" in m:
        L.append(f"| **End-to-end (real detector boxes)** | {m['error']} |")
    else:
        L.append(f"| **End-to-end (real detector boxes)** | acc {m['accuracy']} · "
                 f"P {m['precision']} / R {m['recall']} / **F1 {m['f1']}** |")
    L.append(f"\n> {result['note']}")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="windshield detector checkpoint (.pt)")
    ap.add_argument("--clf", default=str(REPO_ROOT / "checkpoints" / "seatbelt" / "v2" / "model.pt"))
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.det, args.clf, args.split, args.conf, args.iou)
    write_run_log("phase5", "seatbelt_e2e", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_seatbelt_e2e] ERROR: {result['error']}")
        return 1
    m = result["end_to_end_metrics"]
    if "error" in m:
        log(f"[eval_seatbelt_e2e] {m['error']} (report: {rp.name})")
        return 1
    log(f"[eval_seatbelt_e2e] {args.split} acc={m['accuracy']} F1={m['f1']} "
        f"(GT-crop baseline 0.678) matched={result['iou_matched']}/{result['gt_windshields']} "
        f"(report: {rp.name})")
    bk = result.get("classifier_backbone", "mobilenet_v3_small")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "seatbelt_e2e",
                        "dataset": f"seatbelt-OBB({args.split})", "model": f"yolo11n+{bk}",
                        "metric": "f1", "value": m["f1"], "target": "beat/match GT-crop F1=0.678",
                        "pass_fail": "finetuned", "note": f"acc={m['accuracy']} matched={result['iou_matched']}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
