# Module 02 — ANPR / License Plate Recognition (§3.3)

## Summary
Two-stage pipeline: plate detection (zero-shot RF-DETR, no plate class — uses GT boxes for OCR eval) → OCR via TrOCR-base fine-tuned on Indian-LP dataset. Preprocessing pipeline (4× upscale + CLAHE + sharpening + auto-invert) applied to every crop before OCR. Syntax-aware character correction applied post-OCR.

## Model
| Item | Detail |
|---|---|
| Architecture | TrOCR-base-printed (microsoft/trocr-base-printed), fine-tuned |
| Base model | ViT-Base encoder + GPT-2-mini decoder (Seq2Seq OCR) |
| Fine-tuned on | Indian-LP sirishan (1741 images, 1567 train / 174 val) |
| Input preprocessing | 4× bicubic upscale → CLAHE → unsharp-mask sharpen → auto-invert dark plates |
| Post-processing | `correct_plate_text()` — position-aware char confusion fix (0↔O, 1↔I, 5↔S, etc.) |
| Syntax validator | `validate_indian()` — regex for old + HSRP + BH-series Indian plate formats |
| License | MIT (TrOCR) |

## Training
| Item | Detail |
|---|---|
| Base model | microsoft/trocr-base-printed |
| Epochs | 7 (early stopping, patience=5) |
| Best checkpoint | Step 192 (epoch 2) |
| Optimizer | AdamW, lr=5e-5, cosine decay, warmup |
| Batch size | 8 (fp16, H200) |
| Train time | ~15 min on Lightning H200 |

## Results
| Metric | Baseline (PaddleOCR, no preprocessing) | Fine-tuned TrOCR + preprocessing |
|---|---|---|
| Plate exact-match accuracy | 0.449 | **0.7811** |
| CER (Character Error Rate) | 0.196 | **0.0481** |
| Improvement | — | **+33.2 pp exact acc** |

## Model Location
`checkpoints/anpr/trocr_ft/` — full HuggingFace-format checkpoint (model.safetensors + tokenizer + configs)

## Configuration
```
src/modules/anpr.py — ANPRModule(ocr_engine="trocr_ft", preprocess=True, upscale=4)
src/train/train_anpr.py — python -m src.train.train_anpr --model base --epochs 30
src/eval/eval_anpr.py  — python -m src.eval.eval_anpr --config trocr_ft --limit 1000
```

## Files in This Folder
- `README.md` — this file
- `training_summary.json` — key training parameters and results
- `eval_baseline_log.json` — baseline PaddleOCR eval (N=1000, exact_acc=0.449)
- `model_location.txt` — checkpoint path details
