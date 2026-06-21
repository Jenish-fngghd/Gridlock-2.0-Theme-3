"""Phase 5 — Wrong-side learned classifier (Tier 2 escalation).

The geometry rule engine abstains on single stills (no motion), so the zero-shot wrong-side
*verdict* is not_testable → it misses §7 → promoted to a learned classifier per the goal
("a direct learned classifier fine-tuned on the Wrong-Way-Driving-Detection dataset … same
Paradigm-A pattern as helmet/seatbelt").

Approach (feasible on the local cloud_required tier — 4 GB GPU / CPU):
  * crop each vehicle instance from the OBB labels (AABB + small context margin),
  * 2-class classifier (right-side / wrong-side) on a pretrained MobileNetV3-small backbone
    (~10 MB; light enough for CPU), fine-tuned with class-weighted loss (≈7:1 imbalance).

Graceful: if pretrained weights can't be fetched, falls back to random init (logged). Saves to
checkpoints/wrongside/<version>/model.pt. Capped at a few epochs (CPU budget).

Run:  python -m src.train.train_wrongside --epochs 8 --version v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.utils.logging import (REPO_ROOT, append_run_history, log, new_run_id,
                               write_run_log)
from src.utils.obb_convert import parse_obb_label_file

WW_ROOT = (REPO_ROOT / "datasets" / "Wrong Side Driving" /
           "Wrong Way Driving Detection.v1i.yolov8-obb")
CKPT_DIR = REPO_ROOT / "checkpoints" / "wrongside"
CLASS_NAMES = {0: "right-side", 1: "wrong-side"}


def crop_instances(split: str, margin: float = 0.12):
    """Yield (PIL.Image crop, class_id) for every labelled vehicle in a split."""
    import cv2
    from PIL import Image
    img_dir = WW_ROOT / split / "images"
    lbl_dir = WW_ROOT / split / "labels"
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
        for r in parse_obb_label_file(lbl):
            x1, y1, x2, y2 = r["aabb_xyxy"]
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            cx1 = int(max(0, (x1 * w) - margin * bw))
            cy1 = int(max(0, (y1 * h) - margin * bh))
            cx2 = int(min(w, (x2 * w) + margin * bw))
            cy2 = int(min(h, (y2 * h) + margin * bh))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crop = img[cy1:cy2, cx1:cx2][:, :, ::-1]  # BGR->RGB
            yield Image.fromarray(crop), r["class_id"]


class CropDataset:
    """Lazily-materialized in-memory dataset of (tensor, label). Small enough to hold in RAM."""

    def __init__(self, split: str, tfm):
        self.items = []
        for pil, cid in crop_instances(split):
            self.items.append((tfm(pil), cid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_model(num_classes: int = 2, backbone: str = "small"):
    """Build MobileNetV3 classifier. backbone='small' (~2.5M) or 'large' (~5.4M, §3.5 upgrade)."""
    from torchvision import models
    import torch.nn as nn
    if backbone == "large":
        try:
            w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            m = models.mobilenet_v3_large(weights=w)
            pretrained = True
        except Exception as e:  # noqa: BLE001
            log(f"[build_model] pretrained large weights unavailable ({e}); random init")
            m = models.mobilenet_v3_large(weights=None)
            pretrained = False
    else:
        try:
            w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            m = models.mobilenet_v3_small(weights=w)
            pretrained = True
        except Exception as e:  # noqa: BLE001
            log(f"[build_model] pretrained small weights unavailable ({e}); random init")
            m = models.mobilenet_v3_small(weights=None)
            pretrained = False
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m, pretrained


def train(epochs: int, version: str, batch: int, lr: float) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch/torchvision unavailable: {e}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # IMPORTANT: NO horizontal flip — heading/orientation IS the wrong-side signal; mirroring
    # would turn a right-side vehicle into a wrong-side-looking one and corrupt the label.
    # Use only direction-preserving augmentation (mild color/brightness jitter).
    tfm_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    log("[train_wrongside] cropping train/valid instances...")
    train_ds = CropDataset("train", tfm_train)
    val_ds = CropDataset("valid", tfm_eval)
    if len(train_ds) == 0:
        return {"error": "no training crops"}

    # class weights for imbalance
    counts = [0, 0]
    for _t, c in train_ds.items:
        counts[c] += 1
    total = sum(counts)
    weights = torch.tensor([total / (2 * max(c, 1)) for c in counts], dtype=torch.float32)
    log(f"[train_wrongside] train crops={len(train_ds)} counts={counts} weights={weights.tolist()}")

    model, pretrained = build_model()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
        # validate
        vm = evaluate_model(model, val_dl, device)
        history.append({"epoch": ep, "train_loss": round(run_loss / len(train_ds), 4), **vm})
        log(f"   epoch {ep}/{epochs} loss={run_loss/len(train_ds):.4f} "
            f"val_acc={vm['accuracy']} wrong_f1={vm['wrongside_f1']}")
        if vm["wrongside_f1"] >= best_f1:
            best_f1 = vm["wrongside_f1"]
            torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES,
                        "backbone": "mobilenet_v3_small", "input": 128}, ckpt_path)
    return {
        "model": "mobilenet_v3_small (pretrained=%s)" % pretrained,
        "version": version, "epochs": epochs, "device": device,
        "train_crops": len(train_ds), "class_counts": counts,
        "minutes": round((time.time() - t0) / 60, 2),
        "best_val_wrongside_f1": round(best_f1, 4),
        "history": history, "checkpoint": str(ckpt_path),
    }


def evaluate_model(model, dl, device) -> dict:
    import torch
    model.eval()
    tp = fp = fn = tn = 0
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu()
            for p, y in zip(pred.tolist(), yb.tolist()):
                total += 1
                correct += int(p == y)
                if y == 1 and p == 1:
                    tp += 1
                elif y == 0 and p == 1:
                    fp += 1
                elif y == 1 and p == 0:
                    fn += 1
                else:
                    tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": round(correct / max(total, 1), 4),
            "wrongside_precision": round(prec, 4), "wrongside_recall": round(rec, 4),
            "wrongside_f1": round(f1, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    run_id = new_run_id()
    result = train(args.epochs, args.version, args.batch, args.lr)
    write_run_log("phase5", "wrongside_train", run_id, result)
    if "error" in result:
        log(f"[train_wrongside] ERROR: {result['error']}")
        return 1
    log(f"[train_wrongside] done in {result['minutes']}min | best val wrong-side F1="
        f"{result['best_val_wrongside_f1']} | ckpt {result['checkpoint']}")
    append_run_history({"run_id": run_id, "phase": "phase5", "module": "wrongside",
                        "dataset": "Wrong-Way", "model": "mobilenetv3s-ft",
                        "metric": "val_wrongside_f1", "value": result["best_val_wrongside_f1"],
                        "target": "beat geometry abstain", "pass_fail": "finetune",
                        "note": f"{result['epochs']}ep {result['minutes']}min"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
