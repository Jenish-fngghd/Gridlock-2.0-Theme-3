# Gridlock 2.0 R2 — Final Deliverable

**Automated Photo Identification & Classification of Traffic Violations using Computer Vision**

---

## System Overview

Gridlock 2.0 R2 is a modular, paradigm-partitioned traffic violation detection pipeline. It processes photographic evidence and outputs annotated, structured violation records for 7 mandated violation classes + ANPR.

The system is organized into **three reasoning paradigms**:

| Paradigm | Violations | Core Mechanism |
|---|---|---|
| A — Instance-attribute | Helmet, Seatbelt | Fine-grained part-level classifier on attributed person |
| B — Multi-instance counting | Triple riding | Rider-to-bike association + count |
| C — Scene-context | Wrong-side, Red-light, Illegal parking | Geometry rule engine + signal state + trajectory |

---

## Module Summary & Results

| # | Module | Approach | Best Model | Key Metric | Value |
|---|---|---|---|---|---|
| 01 | **Detection** | Zero-shot | RF-DETR-large (COCO) | mAP@0.5 | **0.5216** |
| 02 | **ANPR** | Fine-tuned OCR | TrOCR-base-printed (ft) | Plate exact acc | **0.7811** |
| 03 | **Helmet** | Zero-shot detection | RF-DETR-nano @1280px | Hit-rate | **1.0** (11/11) |
| 03 | **Triple Riding** | Zero-shot counting | RF-DETR-nano @1280px | Triggered | **4/6** (67%) |
| 04 | **Seatbelt** | Two-stage fine-tune | YOLOv11n + MobileNetV3-L | E2E F1 | **0.8082** |
| 05 | **Signal State** | HSV rule + detector | HSV Tier-0 + RF-DETR-nano | Red hit-rate | **1.0** (7/7) |
| 06 | **Wrong-side** | Fine-tuned classifier | MobileNetV3-small | Test F1 | **0.9551** |
| 07 | **Red-light event** | LSTM trajectory | LSTM(hidden=48) | Test F1 | **0.9000** |
| 08 | **Illegal parking** | Dwell-time rule | — | — | BLOCKED (no GT) |

---

## Folder Structure

```
deliverable/
├── README.md                         ← this file
├── master_design_updated.md          ← updated §3.x with exact models + results
├── 01_detection/
│   ├── README.md                     ← RF-DETR-nano, mAP@0.5=0.4803
│   ├── eval_log.json
│   └── model_location.txt
├── 02_anpr/
│   ├── README.md                     ← TrOCR-base ft, exact_acc=0.7811
│   ├── training_summary.json
│   ├── eval_baseline_log.json
│   └── model_location.txt            → checkpoints/anpr/trocr_ft/
├── 03_helmet_triple_riding/
│   ├── README.md                     ← RF-DETR-nano @1280px, hit=1.0
│   ├── eval_log.json
│   └── model_location.txt
├── 04_seatbelt/
│   ├── README.md                     ← YOLOv11n + MobileNetV3-L, e2e F1=0.8082
│   ├── windshield_train_log.json
│   ├── eval_log.json
│   └── model_location.txt            → checkpoints/windshield/v1/ + checkpoints/seatbelt/v4/
├── 05_signal_state/
│   ├── README.md                     ← HSV + RF-DETR-nano, 7/7 red hits
│   ├── eval_log.json
│   └── model_location.txt
├── 06_wrong_side/
│   ├── README.md                     ← MobileNetV3-small, F1=0.9551
│   ├── train_log.json
│   ├── eval_log.json
│   └── model_location.txt            → checkpoints/wrongside/v1/model.pt
├── 07_red_light_event/
│   ├── README.md                     ← LSTM trajectory, F1=0.9000
│   ├── train_log.json
│   └── model_location.txt            → checkpoints/redlight/v1/model.pt
└── 08_illegal_parking/
    └── README.md                     ← BLOCKED — no GT labels
```

---

## Checkpoints Reference

| Module | Checkpoint | Size |
|---|---|---|
| ANPR | `checkpoints/anpr/trocr_ft/model.safetensors` | ~400 MB |
| Seatbelt — Windshield | `checkpoints/windshield/v1/weights/best.pt` | ~6 MB |
| Seatbelt — Classifier | `checkpoints/seatbelt/v4/model.pt` | ~10 MB |
| Wrong-side | `checkpoints/wrongside/v1/model.pt` | ~2 MB |
| Red-light event | `checkpoints/redlight/v1/model.pt` | <1 MB |
| Detection | RF-DETR-nano loaded from library at runtime | — |
| Signal state | HSV rule-based, no checkpoint | — |
| Helmet / Triple | RF-DETR-nano loaded from library at runtime | — |

---

## Evaluation Commands

```bash
# Detection (zero-shot, best: RF-DETR-large)
python -m src.eval.eval_detection --variant large --limit 5000 --threshold 0.3

# ANPR (fine-tuned TrOCR)
python -m src.eval.eval_anpr --config trocr_ft --limit 1000

# Helmet + Triple Riding
python -m src.eval.eval_helmet_zeroshot --imgsz 1280

# Seatbelt (end-to-end)
python -m src.eval.eval_seatbelt_e2e --clf checkpoints/seatbelt/v4/model.pt

# Signal state
python -m src.eval.eval_signal_zeroshot --conf 0.15

# Wrong-side
python -m src.eval.eval_wrongside --weights checkpoints/wrongside/v1/model.pt

# Full run history
cat results/run_history.csv
```

---

## Key Improvements vs. Baselines

| Module | Baseline | Final |
|---|---|---|
| ANPR | 0.449 exact acc (PaddleOCR) | **0.7811** (+33 pp) |
| Seatbelt | 0.678 F1 (GT-crop classifier) | **0.8082** (+13 pp) |
| Wrong-side | not_testable (geometry-only) | **0.9551 F1** |
| Red-light event | not_testable (rule-only) | **0.9000 F1** |
| Detection | 0.4803 mAP@0.5 (RF-DETR-nano, 1000-img) | **0.5216** (RF-DETR-large, 5000-img, +4.1 pp) |
