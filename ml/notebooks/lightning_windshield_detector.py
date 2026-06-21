"""
Gridlock 2.0 — Windshield Detector (§3.5 Stage-1)
Lightning AI standalone training script

Upload to Lightning AI Studio:
  1. This file
  2. datasets/windshield_yolo/  (the entire folder — train/ valid/ data.yaml)

Then run:
  pip install ultralytics
  python lightning_windshield_detector.py

Model: YOLOv11n (Ultralytics, AGPL-3.0 — internal pipeline use only, §12 licensing hygiene)
Task:  single-class detection — "windshield" (Stage-1 of the documented §3.5 two-stage seatbelt pipeline)
Data:  779 train / 337 valid images (converted from seat_belt-and-mobile OBB dataset, class_id==2)
"""

import json
import time
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
DATA_YAML   = Path("windshield_yolo/data.yaml")   # adjust if you put the folder elsewhere
EPOCHS      = 80
BATCH       = 16       # safe on any Lightning GPU (16 GB+); bump to 32 on A100
IMGSZ       = 640
WORKERS     = 4
PATIENCE    = 20
DEVICE      = 0        # single GPU; change to "0,1" for multi-GPU studio
OUTPUT_DIR  = Path("checkpoints_windshield")
RUN_NAME    = "v1"
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO
    import torch
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}  {p.name}  {round(p.total_memory/1024**3, 1)} GB")

    assert DATA_YAML.exists(), f"data.yaml not found at {DATA_YAML} — did you upload windshield_yolo/?"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining YOLOv11n — {EPOCHS} epochs, batch {BATCH}, imgsz {IMGSZ}")
    t0 = time.time()
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        amp=True,
        cache=False,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    minutes = round((time.time() - t0) / 60, 1)
    print(f"\nTraining done in {minutes} min")

    best = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"Best checkpoint: {best}")

    # ── final val ──────────────────────────────────────────────────────────────
    print("\nRunning final val...")
    m = YOLO(str(best))
    metrics = m.val(data=str(DATA_YAML), imgsz=IMGSZ, device=DEVICE)
    map50    = round(float(metrics.box.map50), 4)
    map5095  = round(float(metrics.box.map),   4)

    result = {
        "model": "yolo11n-windshield-ft",
        "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ,
        "minutes": minutes,
        "mAP50": map50, "mAP50_95": map5095,
        "checkpoint": str(best),
        "note": "Stage-1 of §3.5 two-stage seatbelt pipeline (AGPL/benchmark-only per §12)",
    }
    out_json = OUTPUT_DIR / f"results_{RUN_NAME}.json"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n" + "="*60)
    print(f"  mAP@0.5      : {map50}")
    print(f"  mAP@0.5:0.95 : {map5095}")
    print(f"  checkpoint   : {best}")
    print(f"  results JSON : {out_json}")
    print("="*60)
    print("\nDownload best.pt and paste mAP50 into TODO.md as the Stage-1 windshield baseline.")
    print("Then run eval_seatbelt_e2e.py locally with --det <path/to/best.pt> to get the real")
    print("two-stage F1 vs the GT-crop baseline (F1=0.678).")


if __name__ == "__main__":
    main()
