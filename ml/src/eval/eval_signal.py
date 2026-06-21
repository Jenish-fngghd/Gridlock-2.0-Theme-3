"""Phase 3 — Signal-state evaluation (LISA, frame-level).

Evaluates the Tier-0 HSV signal-state classifier against LISA ground truth. LISA's
frameAnnotationsBOX.csv gives, per frame, the light bounding box + an annotation tag
(go/stop/warning/stopLeft/warningLeft) which we map to red/yellow/green:
    stop, stopLeft        -> red
    warning, warningLeft  -> yellow
    go                    -> green

Metric: per-frame accuracy + per-class precision/recall (00_master_design.md §7 "signal state:
accuracy"). This needs no model download and no fine-tune — if it clears a reasonable bar it
stays Tier 0.

Run:  python -m src.eval.eval_signal --limit 1500
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

from src.modules.signal_state import SignalStateClassifier
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)

LISA_ROOT = REPO_ROOT / "datasets" / "Red Light" / "LISA Traffic Light Dataset"
TAG_TO_COLOR = {"stop": "red", "stopLeft": "red", "warning": "yellow",
                "warningLeft": "yellow", "go": "green"}


def index_frames(root: Path) -> dict[str, str]:
    """basename(jpg) -> full path, built once."""
    idx = {}
    for dp, _d, files in os.walk(root):
        if os.path.basename(dp) != "frames":
            continue
        for f in files:
            if f.lower().endswith(".jpg"):
                idx[f] = os.path.join(dp, f)
    return idx


def iter_annotations(root: Path):
    ann_root = root / "Annotations" / "Annotations"
    for dp, _d, files in os.walk(ann_root):
        for f in files:
            if f == "frameAnnotationsBOX.csv":
                yield os.path.join(dp, f)


def run(limit: int) -> dict:
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        return {"error": f"opencv unavailable: {e}"}

    log("[eval_signal] indexing LISA frames...")
    frame_idx = index_frames(LISA_ROOT)
    if not frame_idx:
        return {"error": f"no LISA frames found under {LISA_ROOT}"}
    log(f"[eval_signal] {len(frame_idx)} frames indexed")

    clf = SignalStateClassifier()
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0
    n = 0
    img_cache = (None, None)  # (path, image)
    for csv_path in iter_annotations(LISA_ROOT):
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                tag = row[1].strip()
                color = TAG_TO_COLOR.get(tag)
                if color is None:
                    continue
                base = os.path.basename(row[0].strip())
                fp = frame_idx.get(base)
                if fp is None:
                    skipped += 1
                    continue
                if img_cache[0] != fp:
                    img_cache = (fp, cv2.imread(fp))
                img = img_cache[1]
                if img is None:
                    skipped += 1
                    continue
                box = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                pred = clf.classify(img, box)
                state = pred.get("state", "unknown")
                if state == "unknown":
                    skipped += 1
                    continue
                y_true.append(color)
                y_pred.append(state)
                n += 1
                if n % 500 == 0:
                    log(f"   ...{n} frames classified")
                if limit and n >= limit:
                    break
        if limit and n >= limit:
            break

    if not y_true:
        return {"error": "no usable LISA annotations matched to frames"}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)
    # per-class P/R
    classes = ["red", "yellow", "green"]
    tp = Counter(); fp = Counter(); fn = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    per_class = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(tp[c] + fn[c])}

    return {
        "dataset": "LISA (frame-level signal state)", "model": "HSV classifier (Tier 0, rule-based)",
        "frames_scored": len(y_true), "skipped": skipped,
        "accuracy": round(acc, 4), "per_class": per_class,
        "gt_distribution": dict(Counter(y_true)),
        "tier": "tier_0 (rule-based, no fine-tune)",
    }


def write_report(result: dict, run_id: str) -> Path:
    rp = REPO_ROOT / "results" / f"eval_signal_{run_id}.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# Signal-state eval — LISA (run {run_id})\n"]
    if "error" in result:
        L.append(f"**ERROR:** {result['error']}")
        rp.write_text("\n".join(L), encoding="utf-8")
        return rp
    L.append(f"- Model: {result['model']}  ·  Tier: {result['tier']}")
    L.append(f"- Frames scored: {result['frames_scored']} (skipped {result['skipped']})")
    L.append(f"- GT distribution: {result['gt_distribution']}\n")
    L.append("## Quantitative\n")
    L.append("| Metric | Value | §7 reference |")
    L.append("|---|---|---|")
    L.append(f"| Accuracy | **{result['accuracy']}** | LISA published baselines |")
    L.append("\n| Class | Precision | Recall | F1 | Support |")
    L.append("|---|---|---|---|---|")
    for c, m in result["per_class"].items():
        L.append(f"| {c} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")
    rp.write_text("\n".join(L), encoding="utf-8")
    return rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    args = ap.parse_args()
    run_id = new_run_id()
    result = run(args.limit)
    write_run_log("phase3", "signal_state", run_id, result)
    rp = write_report(result, run_id)
    if "error" in result:
        log(f"[eval_signal] ERROR: {result['error']}")
        return 1
    log(f"[eval_signal] accuracy={result['accuracy']} on {result['frames_scored']} frames "
        f"(report: {rp.name})")
    append_run_history({"run_id": run_id, "phase": "phase3", "module": "signal_state",
                        "dataset": "LISA", "model": "HSV-tier0", "metric": "accuracy",
                        "value": result["accuracy"], "target": "baseline",
                        "pass_fail": "tier0", "note": f"{result['frames_scored']} frames"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
