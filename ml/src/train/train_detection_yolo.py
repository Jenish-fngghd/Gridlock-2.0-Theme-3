"""Detection fine-tune — YOLOv11 (Ultralytics) on IDD.

FAST local alternative to the RF-DETR-Large fine-tune for this 4 GB laptop GPU. Per the master
design this is the **internal-benchmark** detector role (J6 / §12): Ultralytics YOLO is AGPL-3.0,
so the *trained weights are NOT shippable* — RF-DETR (Apache) remains the deliverable, trained on
the cloud. This run answers "how much does an Indian fine-tune help, fast" and gives a strong
comparator number, including the India-specific classes (auto-rickshaw) zero-shot COCO can't see.

Variant: YOLOv11-l by default — the largest variant that still trains with a healthy batch on
4 GB (x would be forced to a tiny batch that hurts training). On a big GPU, use x.

Prereq:  python -m src.train.coco_to_yolo --coco datasets/idd_coco_sub --out datasets/idd_yolo
Run:     python -m src.train.train_detection_yolo --data datasets/idd_yolo/data.yaml \
             --variant l --epochs 80 --imgsz 640 --batch -1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging import REPO_ROOT, append_run_history, log, new_run_id, write_run_log


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv11 detection fine-tune (Ultralytics, AGPL — benchmark only)")
    ap.add_argument("--data", required=True, help="YOLO data.yaml")
    ap.add_argument("--variant", default="l", choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = auto (max that fits ~60% VRAM)")
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs)")
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers (low = less RAM)")
    ap.add_argument("--cache", default="False", help="False | ram | disk (disk speeds epochs)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="idd_yolo11")
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        log(f"[train_yolo] ERROR importing ultralytics/torch: {e}")
        return 1

    cuda = torch.cuda.is_available()
    log(f"[train_yolo] torch {torch.__version__} CUDA {cuda} "
        f"| device {torch.cuda.get_device_name(0) if cuda else 'CPU'}")
    log(f"[train_yolo] YOLOv11-{args.variant} | data={args.data} | imgsz={args.imgsz} "
        f"batch={args.batch} epochs={args.epochs}")

    run_id = new_run_id()
    out_dir = REPO_ROOT / "checkpoints" / "detection_yolo"
    model = YOLO(f"yolo11{args.variant}.pt")          # COCO-pretrained, auto-downloads
    cache = args.cache if args.cache in ("ram", "disk") else False
    batch = int(args.batch) if args.batch != -1 else -1  # ultralytics needs int batch (or -1 auto)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=batch,
        device=args.device, amp=True, patience=args.patience, workers=args.workers,
        cache=cache, project=str(out_dir), name=f"{args.name}_{args.variant}", exist_ok=True,
        plots=True, verbose=True,
    )
    # pull final metrics
    try:
        m = results.results_dict if hasattr(results, "results_dict") else {}
        map50 = float(m.get("metrics/mAP50(B)", 0.0))
        map5095 = float(m.get("metrics/mAP50-95(B)", 0.0))
    except Exception:
        map50 = map5095 = 0.0
    best = out_dir / f"{args.name}_{args.variant}" / "weights" / "best.pt"
    payload = {"model": f"yolo11{args.variant}", "data": args.data, "imgsz": args.imgsz,
               "epochs": args.epochs, "mAP50": round(map50, 4), "mAP50_95": round(map5095, 4),
               "best_weights": str(best),
               "license_note": "Ultralytics YOLO = AGPL-3.0 -> benchmark only, NOT shipped (J6/§12)"}
    write_run_log("phase5", "detection_yolo", run_id, payload)
    log(f"[train_yolo] DONE | mAP50={map50:.4f} mAP50-95={map5095:.4f} | best -> {best}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "detection_yolo",
                        "dataset": "IDD", "model": f"yolo11{args.variant}-ft",
                        "metric": "mAP50", "value": round(map50, 4), "target": 0.787,
                        "pass_fail": "benchmark", "note": "AGPL - benchmark only; vs zero-shot 0.418"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
