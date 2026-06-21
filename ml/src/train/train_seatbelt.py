"""Phase 5 — Seatbelt classifier (Tier 1, mandatory fine-tune).

00_master_design.md §3.5: two-stage windshield → driver crop → belt classifier. The downloaded
seat_belt-and-mobile dataset is OBB *detection* (windshield/seatbelt/mobile boxes); we reframe
it to the design's two-stage form for a locally-feasible instance classifier:

  * crop each **windshield** box (the "driver region"),
  * label it **no_seatbelt (violation)** if NO seatbelt box lies inside it, else **seatbelt (ok)**,
  * 2-class MobileNetV3 (pretrained) — small (default, v2, F1=0.678) or large (--backbone large,
    v3+, more capacity for the thin belt-strap feature). §3.5 says "CNN classifier"; both fit.

Honest label semantics: "no belt box inside the windshield" = no *visible/annotated* belt → treated
as the no-seatbelt class. This is supervision, not perfect ground truth (night/tint/occlusion are
real failure modes, §3.5). The `mobile` class is OUT OF SCOPE (outside the 7 mandated violations);
its instances are counted and logged, never used as a belt label nor silently dropped.

No test split ships → evaluate on the held-out **valid** split.

Run:  python -m src.train.train_seatbelt --epochs 20 --version v2          # small (baseline)
      python -m src.train.train_seatbelt --backbone large --epochs 20 --version v3 --input 224
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.train.train_wrongside import build_model, evaluate_model
from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)
from src.utils.obb_convert import parse_obb_label_file

SB_ROOT = (REPO_ROOT / "datasets" / "seat belt detection" /
           "seat_belt and mobile.v2i.yolov8-obb")
CKPT_DIR = REPO_ROOT / "checkpoints" / "seatbelt"
# label 0 = seatbelt present (compliant); label 1 = no_seatbelt (violation, positive class)
CLASS_NAMES = {0: "seatbelt", 1: "no_seatbelt"}


def _contain_frac(inner, outer) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    x1, y1, x2, y2 = max(ix1, ox1), max(iy1, oy1), min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (ix2 - ix1) * (iy2 - iy1)
    return inter / a if a > 0 else 0.0


def crop_windshields(split: str, margin: float = 0.05):
    """Yield (PIL windshield crop, label, mobile_count) per windshield box."""
    import cv2
    from PIL import Image
    img_dir = SB_ROOT / split / "images"
    lbl_dir = SB_ROOT / split / "labels"
    mobiles_seen = 0
    if not img_dir.exists():
        return
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        recs = parse_obb_label_file(lbl)
        winds = [r["aabb_xyxy"] for r in recs if r["class_id"] == 2]
        belts = [r["aabb_xyxy"] for r in recs if r["class_id"] == 1]
        mobiles_seen += sum(1 for r in recs if r["class_id"] == 0)
        for wd in winds:
            x1, y1, x2, y2 = wd
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            cx1 = int(max(0, x1 * w - margin * bw))
            cy1 = int(max(0, y1 * h - margin * bh))
            cx2 = int(min(w, x2 * w + margin * bw))
            cy2 = int(min(h, y2 * h + margin * bh))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            has_belt = any(_contain_frac(b, wd) > 0.5 for b in belts)
            label = 0 if has_belt else 1   # 1 = no_seatbelt (violation)
            crop = img[cy1:cy2, cx1:cx2][:, :, ::-1]
            yield Image.fromarray(crop), label
    # stash mobile count on the function for the caller (simple side channel)
    crop_windshields.last_mobiles = mobiles_seen


class WindshieldDataset:
    def __init__(self, split: str, tfm):
        self.items = []
        for pil, label in crop_windshields(split):
            self.items.append((tfm(pil), label))
        self.mobiles = getattr(crop_windshields, "last_mobiles", 0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train(epochs: int, version: str, batch: int, lr: float,
          input_size: int = 128, weight_decay: float = 0.0,
          freeze_backbone: bool = False, backbone: str = "small",
          scheduler: str = "step") -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch/torchvision unavailable: {e}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # h-flip is SAFE here (belt presence is flip-invariant) — useful augmentation.
    # Higher input_size helps capture the thin diagonal strap (a fine detail behind glass).
    tfm_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    log("[train_seatbelt] cropping windshields (train/valid)...")
    train_ds = WindshieldDataset("train", tfm_train)
    val_ds = WindshieldDataset("valid", tfm_eval)
    if len(train_ds) == 0:
        return {"error": "no windshield crops"}

    counts = [0, 0]
    for _t, c in train_ds.items:
        counts[c] += 1
    total = sum(counts)
    weights = torch.tensor([total / (2 * max(c, 1)) for c in counts], dtype=torch.float32)
    log(f"[train_seatbelt] train crops={len(train_ds)} counts(seatbelt,no_seatbelt)={counts} "
        f"weights={weights.tolist()} | mobile(out-of-scope)={train_ds.mobiles}")

    model, pretrained = build_model(num_classes=2, backbone=backbone)
    if freeze_backbone:
        frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False
                frozen += 1
        log(f"[train_seatbelt] froze {frozen} backbone params (training classifier head only)")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # Cosine annealing eliminates the late-epoch F1 oscillation seen in flat-LR runs.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
             if scheduler == "cosine" else None)
    crit = nn.CrossEntropyLoss(weight=weights.to(device))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)

    t0 = time.time()
    history = []
    best_f1 = -1.0
    ckpt_path = CKPT_DIR / version / "model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += float(loss) * len(xb)
        if sched is not None:
            sched.step()
        vm = evaluate_model(model, val_dl, device)  # positive class idx 1 = no_seatbelt
        history.append({"epoch": ep, "train_loss": round(run_loss / len(train_ds), 4), **vm})
        log(f"   epoch {ep}/{epochs} loss={run_loss/len(train_ds):.4f} "
            f"val_acc={vm['accuracy']} no_seatbelt_f1={vm['wrongside_f1']}")
        if vm["wrongside_f1"] >= best_f1:
            best_f1 = vm["wrongside_f1"]
            torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES,
                        "backbone": f"mobilenet_v3_{backbone}", "input": input_size,
                        "positive_class": "no_seatbelt"}, ckpt_path)
    return {
        "model": f"mobilenet_v3_{backbone} (pretrained={pretrained})",
        "scheduler": scheduler, "version": version,
        "epochs": epochs, "device": device, "train_crops": len(train_ds),
        "class_counts": {"seatbelt": counts[0], "no_seatbelt": counts[1]},
        "mobile_out_of_scope": train_ds.mobiles,
        "minutes": round((time.time() - t0) / 60, 2),
        "best_val_no_seatbelt_f1": round(best_f1, 4),
        "val_metrics_last": history[-1], "history": history, "checkpoint": str(ckpt_path),
        "note": ("metrics use positive=no_seatbelt (the violation). Label = windshield WITHOUT an "
                 "annotated belt inside. 'mobile' counted but out-of-scope (not a mandated violation)."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--input", type=int, default=128, help="input resolution (try 224 for the thin strap)")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay (regularization vs overfit)")
    ap.add_argument("--freeze", action="store_true", help="freeze backbone, train head only (small-data)")
    ap.add_argument("--backbone", choices=["small", "large"], default="small",
                    help="MobileNetV3 variant: small (~2.5M, fast) or large (~5.4M, more capacity for belt strap)")
    ap.add_argument("--scheduler", choices=["step", "cosine"], default="step",
                    help="cosine: CosineAnnealingLR (eliminates late-epoch F1 oscillation)")
    args = ap.parse_args()
    run_id = new_run_id()
    result = train(args.epochs, args.version, args.batch, args.lr,
                   input_size=args.input, weight_decay=args.wd, freeze_backbone=args.freeze,
                   backbone=args.backbone, scheduler=args.scheduler)
    write_run_log("phase5", "seatbelt_train", run_id, result)
    if "error" in result:
        log(f"[train_seatbelt] ERROR: {result['error']}")
        return 1
    vm = result["val_metrics_last"]
    log(f"[train_seatbelt] done {result['minutes']}min | held-out valid: acc={vm['accuracy']} "
        f"no_seatbelt P={vm['wrongside_precision']} R={vm['wrongside_recall']} F1={vm['wrongside_f1']} "
        f"| best F1={result['best_val_no_seatbelt_f1']} | ckpt {result['checkpoint']}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "seatbelt",
                        "dataset": "seatbelt-OBB(valid)", "model": f"mobilenetv3{args.backbone[0]}-ft",
                        "metric": "no_seatbelt_f1", "value": result["best_val_no_seatbelt_f1"],
                        "target": "from not_testable", "pass_fail": "finetuned",
                        "note": f"acc={vm['accuracy']} {result['epochs']}ep {result['minutes']}min"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
