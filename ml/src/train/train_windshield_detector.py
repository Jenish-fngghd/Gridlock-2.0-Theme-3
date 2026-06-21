"""Phase 5 — Windshield detector (Tier 1, §3.5 Stage-1 of the documented two-stage seatbelt pipeline).

00_master_design.md §3.5: "Two-stage: YOLOv11/windshield detector → driver crop → CNN/CNN-SVM
belt classifier." `train_seatbelt.py` already does Stage-2 (the belt classifier) but fed it
ground-truth windshield crops, not a real detector — this script trains the actual Stage-1
detector named in the doc, so the pipeline can be evaluated end-to-end (see
`eval_seatbelt_e2e.py`) instead of taking the GT-crop shortcut.

**Ultralytics YOLO = AGPL-3.0** → internal pipeline-stage use, same license flag as the
detection module's YOLOv12/YOLOv11 benchmarking elsewhere (§12 licensing hygiene). Single class
("windshield"), small dataset → YOLOv11n (nano) is enough; feasible on the local 4 GB GPU.

Converts the seat_belt-and-mobile OBB labels (class_id==2 = windshield) to single-class YOLO
detection format (images symlinked/copied, no dataset duplication beyond labels).

Run:  python -m src.train.train_windshield_detector --epochs 60 --version v1
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)
from src.utils.obb_convert import parse_obb_label_file

SB_ROOT = (REPO_ROOT / "datasets" / "seat belt detection" /
           "seat_belt and mobile.v2i.yolov8-obb")
YOLO_DIR = REPO_ROOT / "datasets" / "windshield_yolo"
CKPT_DIR = REPO_ROOT / "checkpoints" / "windshield"
WINDSHIELD = 2


def build_split(split: str) -> int:
    img_dir = SB_ROOT / split / "images"
    lbl_dir = SB_ROOT / split / "labels"
    odir = YOLO_DIR / split
    (odir / "images").mkdir(parents=True, exist_ok=True)
    (odir / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    if not img_dir.exists():
        return n
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        winds = [r["aabb_xywh"] for r in parse_obb_label_file(lbl) if r["class_id"] == WINDSHIELD]
        if not winds:
            continue
        dst_img = odir / "images" / img_path.name
        if not dst_img.exists():
            try:
                import os
                os.symlink(img_path.resolve(), dst_img)
            except Exception:
                shutil.copy2(img_path, dst_img)
        lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in winds]
        (odir / "labels" / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
        n += 1
    return n


def prepare_dataset() -> dict:
    n_train = build_split("train")
    n_valid = build_split("valid")
    (YOLO_DIR / "data.yaml").write_text(
        f"train: {(YOLO_DIR / 'train' / 'images').as_posix()}\n"
        f"val: {(YOLO_DIR / 'valid' / 'images').as_posix()}\n"
        f"nc: 1\nnames:\n  0: windshield\n")
    return {"train_images": n_train, "valid_images": n_valid, "yaml": str(YOLO_DIR / "data.yaml")}


def train(epochs: int, version: str, batch: int, imgsz: int) -> dict:
    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        return {"error": f"ultralytics unavailable: {e}"}

    prep = prepare_dataset()
    if prep["train_images"] == 0:
        return {"error": "no windshield-labeled training images found", "prep": prep}
    log(f"[train_windshield_detector] dataset ready: {prep}")

    ckpt_dir = CKPT_DIR / version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=prep["yaml"], epochs=epochs, imgsz=imgsz, batch=batch,
        device=0, workers=2, patience=20, amp=False, cache=False,
        project=str(ckpt_dir.parent), name=version, exist_ok=True, plots=False, verbose=True)
    best = ckpt_dir / "weights" / "best.pt"
    minutes = round((time.time() - t0) / 60, 2)

    metrics = model.val(data=prep["yaml"], imgsz=imgsz, device=0)
    map50 = round(float(metrics.box.map50), 4)
    map5095 = round(float(metrics.box.map), 4)
    return {
        "model": "yolo11n (windshield, single-class)", "version": version, "epochs": epochs,
        "minutes": minutes, "dataset": prep, "checkpoint": str(best),
        "mAP50": map50, "mAP50_95": map5095,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=416)
    args = ap.parse_args()
    run_id = new_run_id()
    result = train(args.epochs, args.version, args.batch, args.imgsz)
    write_run_log("phase5", "windshield_detector_train", run_id, result)
    if "error" in result:
        log(f"[train_windshield_detector] ERROR: {result['error']}")
        return 1
    log(f"[train_windshield_detector] done {result['minutes']}min | mAP50={result['mAP50']} "
        f"mAP50-95={result['mAP50_95']} | ckpt {result['checkpoint']}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "windshield_detector",
                        "dataset": "seatbelt-OBB(windshield-only)", "model": "yolo11n-ft",
                        "metric": "mAP50", "value": result["mAP50"],
                        "target": "Stage-1 of §3.5 two-stage", "pass_fail": "finetuned",
                        "note": f"{result['epochs']}ep {result['minutes']}min"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
