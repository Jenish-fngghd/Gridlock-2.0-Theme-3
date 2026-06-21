"""Fine-tune TrOCR on Indian-LP plates for §3.3 ANPR (Lightning AI / Kaggle T4).

Model: microsoft/trocr-base-printed (default) or trocr-large-printed
Dataset: datasets/Indian LP/sirishan  (1,741 images with VOC XML GT text)
Output: checkpoints/anpr/trocr_ft/

The recognition model is fine-tuned end-to-end (encoder + decoder) with a
seq2seq cross-entropy loss. The preprocessed plate crop (4× upscale + CLAHE +
sharpen) is the input; the GT plate string (alphanumeric, no spaces) is the target.

Expected time on H200: ~3–5 min (base) / ~8 min (large) for 30 epochs.
Expected time on T4:   ~10 min (base) / ~25 min (large).

Run:
  python -m src.train.train_anpr                              # base, 30ep
  python -m src.train.train_anpr --model large --epochs 20   # large model
  python -m src.train.train_anpr --model base --epochs 50 --batch 16
"""
from __future__ import annotations

import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from src.modules.anpr import preprocess_crop
from src.utils.logging import REPO_ROOT, append_run_history, log, new_run_id

INDIAN_ROOT = REPO_ROOT / "datasets" / "Indian LP" / "sirishan"
CKPT_BASE   = REPO_ROOT / "checkpoints" / "anpr"

HF_MODELS = {
    "base":  "microsoft/trocr-base-printed",
    "large": "microsoft/trocr-large-printed",
    "small": "microsoft/trocr-small-printed",
}


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


# ─── Dataset loading ────────────────────────────────────────────────────────

def load_pairs(limit: int | None = None):
    """Load (crop_np_bgr, gt_text) pairs from Indian-LP sirishan XML annotations."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("opencv-python required: pip install opencv-python") from e

    pairs = []
    for dp, _d, files in os.walk(INDIAN_ROOT):
        for f in sorted(files):
            if not f.endswith(".xml"):
                continue
            xml = os.path.join(dp, f)
            try:
                root = ET.parse(xml).getroot()
            except Exception:
                continue
            stem = f[:-4]
            img_path = None
            for cand in (stem, stem + ".jpeg", stem + ".jpg", stem + ".png"):
                p = os.path.join(dp, cand)
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                fn = root.findtext("filename")
                if fn and os.path.exists(os.path.join(dp, fn)):
                    img_path = os.path.join(dp, fn)
            if img_path is None:
                continue
            obj = root.find("object")
            if obj is None:
                continue
            gt = norm((obj.findtext("name") or "").strip())
            if not gt:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1 = int(float(bb.findtext("xmin", "0")))
            y1 = int(float(bb.findtext("ymin", "0")))
            x2 = int(float(bb.findtext("xmax", "0")))
            y2 = int(float(bb.findtext("ymax", "0")))
            img = cv2.imread(img_path)
            if img is None:
                continue
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            # Apply preprocessing before storing (saves time during training)
            crop = preprocess_crop(crop, upscale=4, clahe=True, sharpen=True)
            pairs.append((crop, gt))
            if limit and len(pairs) >= limit:
                return pairs
    return pairs


# ─── HuggingFace Dataset ────────────────────────────────────────────────────

def make_dataset(pairs, processor, max_target_length: int = 16, split_frac: float = 0.1):
    """Convert (crop_bgr, gt_text) pairs to HF Dataset with pixel_values + labels.
    Returns (train_ds, val_ds).
    """
    from datasets import Dataset
    import numpy as np
    from PIL import Image

    log(f"[train_anpr] encoding {len(pairs)} crops → pixel_values + labels ...")
    records = []
    for crop_bgr, gt in pairs:
        rgb = crop_bgr[:, :, ::-1]  # BGR → RGB
        pil = Image.fromarray(rgb.astype("uint8"))
        records.append({"pil": pil, "text": gt})

    n_val = max(1, int(len(records) * split_frac))
    n_train = len(records) - n_val
    log(f"[train_anpr] train={n_train}  val={n_val}")

    def _encode(batch):
        pv = processor(images=batch["pil"], return_tensors="pt").pixel_values
        enc = processor.tokenizer(
            batch["text"], padding="max_length", max_length=max_target_length,
            truncation=True, return_tensors="pt"
        )
        labels = enc.input_ids
        # Replace padding token id with -100 so loss ignores pad positions
        pad_id = processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        return {"pixel_values": list(pv.numpy()), "labels": list(labels.numpy())}

    raw = Dataset.from_list([{"pil": r["pil"], "text": r["text"]} for r in records])
    encoded = raw.map(_encode, batched=True, batch_size=32, remove_columns=["pil", "text"])
    train_ds = encoded.select(range(n_train))
    val_ds   = encoded.select(range(n_train, n_train + n_val))
    return train_ds, val_ds


# ─── Training ───────────────────────────────────────────────────────────────

def train(model_size: str = "base", epochs: int = 30, batch: int = 8,
          lr: float = 5e-5, limit: int | None = None, version: str = "v1") -> dict:
    import torch
    from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              EarlyStoppingCallback)
    import numpy as np

    hf_id = HF_MODELS[model_size]
    ckpt_dir = CKPT_BASE / f"trocr_ft_{model_size}_{version}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_final = CKPT_BASE / "trocr_ft"  # canonical path read by ANPRModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[train_anpr] model={hf_id}  device={device}  epochs={epochs}  batch={batch}")
    log(f"[train_anpr] loading processor + model ...")
    proc  = TrOCRProcessor.from_pretrained(hf_id)
    model = VisionEncoderDecoderModel.from_pretrained(hf_id)

    # Configure seq2seq decoding tokens
    model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.config.pad_token_id           = proc.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    # Load and encode dataset
    pairs = load_pairs(limit=limit)
    if not pairs:
        return {"error": f"no valid pairs found under {INDIAN_ROOT}"}
    log(f"[train_anpr] loaded {len(pairs)} plate crops")
    train_ds, val_ds = make_dataset(pairs, proc)

    # CER metric for validation
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids   = pred.predictions
        pred_ids[pred_ids == -100] = proc.tokenizer.pad_token_id
        labels_ids[labels_ids == -100] = proc.tokenizer.pad_token_id
        pred_str  = proc.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str = proc.batch_decode(labels_ids, skip_special_tokens=True)

        def cer(a, b):
            a, b = norm(a), norm(b)
            if not b:
                return 0.0
            prev = list(range(len(b) + 1))
            for ca in a:
                cur = [prev[0] + 1]
                for j, cb in enumerate(b, 1):
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
                prev = cur
            return prev[-1] / len(b)

        cers = [cer(p, l) for p, l in zip(pred_str, label_str)]
        exact = sum(norm(p) == norm(l) for p, l in zip(pred_str, label_str))
        return {"CER": round(float(np.mean(cers)), 4),
                "exact_acc": round(exact / max(len(pred_str), 1), 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        predict_with_generate=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        warmup_steps=max(1, len(train_ds) // batch // 4),
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_acc",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    log("[train_anpr] starting fine-tune ...")
    train_result = trainer.train()
    log(f"[train_anpr] train done: {train_result.metrics}")

    # Evaluate on val
    eval_result = trainer.evaluate()
    val_cer     = eval_result.get("eval_CER", -1)
    val_exact   = eval_result.get("eval_exact_acc", -1)
    log(f"[train_anpr] val exact_acc={val_exact}  CER={val_cer}")

    # Save final model + processor to the canonical path
    trainer.save_model(str(out_final))
    proc.save_pretrained(str(out_final))
    log(f"[train_anpr] saved → {out_final}")

    return {
        "model": hf_id, "version": version,
        "epochs": epochs, "batch": batch,
        "val_exact_acc": val_exact, "val_CER": val_cer,
        "train_pairs": len(pairs),
        "ckpt": str(out_final),
    }


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune TrOCR on Indian-LP plates")
    ap.add_argument("--model", choices=["small", "base", "large"], default="base",
                    help="TrOCR model size (base recommended; large for best quality on H200)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--lr",     type=float, default=5e-5)
    ap.add_argument("--limit",  type=int, default=None,
                    help="Cap samples (None = all 1741)")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    run_id = new_run_id()
    result = train(model_size=args.model, epochs=args.epochs, batch=args.batch,
                   lr=args.lr, limit=args.limit, version=args.version)

    if "error" in result:
        log(f"[train_anpr] ERROR: {result['error']}")
        return 1

    append_run_history({
        "run_id": run_id, "phase": "phase5", "module": "anpr",
        "dataset": "Indian-LP", "model": f"trocr-{args.model}-printed-ft",
        "metric": "val_exact_acc",
        "value": result["val_exact_acc"],
        "target": "beat 0.449 baseline", "pass_fail": "finetuned",
        "note": (f"TrOCR-{args.model} ft {args.epochs}ep "
                 f"CER={result['val_CER']} N={result['train_pairs']}"),
    })
    log(f"[train_anpr] DONE  val_exact_acc={result['val_exact_acc']}  "
        f"CER={result['val_CER']}  ckpt={result['ckpt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
