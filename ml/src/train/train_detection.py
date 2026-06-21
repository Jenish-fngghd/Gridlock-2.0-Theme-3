"""Phase 5 — Detection fine-tune (RF-DETR, Tier 1) — for the cloud GPU (Lightning H200).

This is the highest-ROI fine-tune (§8): it lifts the zero-shot detector (IDD mAP@0.5 ~0.42)
toward the DriveIndia ~0.787 reference AND teaches the India-specific classes COCO can't see
(auto-rickshaw, vehicle-fallback). Not runnable on the 4 GB laptop tier → run on Lightning.

Prereq: convert IDD to the Roboflow-COCO layout first:
    python -m src.train.prepare_idd_coco --idd-root <IDD_Detection> --out datasets/idd_coco

Then train:
    python -m src.train.train_detection --dataset-dir datasets/idd_coco --variant medium \
        --epochs 40 --batch 8 --grad-accum 2 --output checkpoints/detection/v1

RF-DETR infers the class set from the dataset's COCO categories. Apache-2.0 (license-clean,
shippable — unlike AGPL YOLO). Variant by GPU budget: H200 (141 GB) handles medium/large easily.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge"}


def main() -> int:
    ap = argparse.ArgumentParser(description="RF-DETR detection fine-tune (Lightning/H200)")
    ap.add_argument("--dataset-dir", required=True, help="Roboflow-COCO dir (train/valid[/test])")
    # Master design (§3.2 Stage-2 confirm / §8 primary fine-tune target) = RF-DETR-Large.
    ap.add_argument("--variant", default="large", choices=list(VARIANTS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None, help="override (multiple of 16; e.g. 512)")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=False,
                    help="trade compute for memory — needed to fit Large on a small GPU (4 GB)")
    ap.add_argument("--num-workers", type=int, default=2, help="0 = no worker processes (saves RAM)")
    ap.add_argument("--no-ema", action="store_true", default=False,
                    help="disable EMA — frees a full model copy of VRAM (helps fit a small GPU)")
    ap.add_argument("--output", default="checkpoints/detection/v1")
    ap.add_argument("--early-stopping", action="store_true", default=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    if not (ds / "train" / "_annotations.coco.json").exists():
        print(f"ERROR: {ds}/train/_annotations.coco.json not found. Run prepare_idd_coco.py first.")
        return 1
    cats = json.loads((ds / "train" / "_annotations.coco.json").read_text(encoding="utf-8")).get("categories", [])
    print(f"[train_detection] dataset={ds} | {len(cats)} classes: {[c['name'] for c in cats]}")

    import torch
    print(f"[train_detection] torch {torch.__version__} | CUDA {torch.cuda.is_available()} "
          f"| device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        print("[train_detection] WARNING: no CUDA — RF-DETR fine-tune is impractical on CPU. "
              "Run this on the Lightning H200.")

    import rfdetr
    ctor = getattr(rfdetr, VARIANTS[args.variant])
    model = ctor(resolution=args.resolution) if args.resolution else ctor()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    train_kwargs = dict(
        dataset_dir=str(ds), dataset_file="roboflow",
        epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
        lr=args.lr, output_dir=args.output, early_stopping=args.early_stopping,
        amp=True,  # mixed precision — on by default, kept explicit for the small-GPU case
        num_workers=args.num_workers,
    )
    if args.gradient_checkpointing:
        train_kwargs["gradient_checkpointing"] = True
    if args.no_ema:
        train_kwargs["use_ema"] = False
    if args.resume:
        train_kwargs["resume"] = args.resume
    print(f"[train_detection] starting: variant={args.variant} {train_kwargs}")
    model.train(**train_kwargs)
    print(f"[train_detection] DONE. checkpoints + metrics in {args.output}")
    print("[train_detection] best weights: look for checkpoint_best_*.pth / checkpoint*.pth in the output dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
