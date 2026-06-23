# Gridlock 2.0 — Final Results & Model Metrics

> Consolidated, **source-traceable** results for every violation class + ANPR + detection.
> Each number below comes from a real evaluation run logged in
> [`ml/results/run_history.csv`](results/run_history.csv) and the matching `ml/results/eval_*.md`
> file (run-id shown). **No numbers are estimated or fabricated** — where a task was not yet
> benchmarked against ground truth, it is labelled *qualitative* or *not yet tested*, not given a
> fake score (per the project's no-silent-substitution rule).

**Problem statement scope:** 7 violation types (helmet, triple-riding, seat-belt, wrong-side,
stop-line, red-light, illegal-parking) + ANPR on confirmed violators, producing auditable evidence
records. Source: `ml/docs/final/00_master_design.md`.

---

## 1. Headline numbers (benchmarked, real ground truth)

| Task | Model | Dataset | Key metric | Value | Run-id |
|---|---|---|---|---|---|
| **Detection (general)** | RF-DETR-large (zero-shot, COCO) | IDD, 5000 imgs | mAP@0.5 | **0.4876** | `20260621-041258` |
| | | | mAP@0.5:0.95 | **0.3341** | |
| **Wrong-side** | MobileNetV3-small (fine-tuned) | Wrong-Way OBB (held-out test) | F1 (wrong-side) | **0.9551** | `20260619-191336` |
| | | | accuracy | **0.9889** | |
| **Seat-belt (end-to-end)** | YOLOv11n + MobileNetV3-large | seat-belt OBB (valid) | F1 (no_seatbelt) | **0.8082** | `20260620-173851` |
| | | | accuracy | **0.8605** | |
| **Windshield detector** (seat-belt Stage-1) | YOLOv11n (fine-tuned) | seat-belt OBB (windshield) | mAP@0.5 | **0.995** | `20260620-165700` |
| | | | mAP@0.5:0.95 | **0.8576** | |
| **Red-light event** | LSTM trajectory classifier | RunningRedlight (split-by-video) | F1 (cross) | **0.8995** | `20260619-194422` |
| | | | accuracy | **0.9218** | |
| **Signal state (red detect)** | HSV rule (tier-0) | LISA, 1500 frames | accuracy | **0.9967** | `20260619-144952` |
| | | LISA, 8000 frames | accuracy | **0.9324** | `20260619-145022` |
| **ANPR (OCR on GT crops)** | PaddleOCR PP-OCRv5 + post-proc | Indian-LP, 1000 | plate exact-match | **0.449** | `20260619-203717` |
| | | | CER | **0.1962** | |
| | | | 1−NED | **0.8015** | |

---

## 2. Per-violation detail

### 2.1 Detection (the backbone for everything)
- **RF-DETR-large, zero-shot COCO weights**, threshold 0.4, 5000 IDD images.
- **mAP@0.5 = 0.4876**, mAP@0.5:0.95 = 0.3341.
- Per-class AP@0.5: car 0.681 · bus 0.567 · motorcycle 0.527 · person 0.547 · bicycle 0.482 ·
  truck 0.405 · traffic-light 0.205.
- **Variant comparison** (1000–5000 imgs, zero-shot): nano 0.4803 · **large 0.4876–0.5216** ·
  xlarge 0.4073. Large is the sweet spot; xlarge regressed.
- **Known structural gap (not scored):** IDD classes with **no COCO equivalent** —
  autorickshaw, traffic-sign, animal, vehicle-fallback — are undetectable zero-shot (~0 recall).
  Closing these needs the fine-tune / data-engine step (master design §3.10/§8). Target with
  fine-tune: ~0.787 (DriveIndia reference).

### 2.2 Wrong-side driving — **best-performing learned module**
- MobileNetV3-small fine-tuned on a single vehicle crop (no motion needed).
- Held-out test (720 instances): **acc 0.9889 · P 0.977 · R 0.9341 · F1 0.9551**.
- Replaced the zero-shot geometry rule, which **abstained** on stills (`not_testable`).

### 2.3 Seat-belt — two-stage (detect windshield → classify belt)
- **Stage 1:** YOLOv11n windshield detector — mAP@0.5 **0.995**, mAP@0.5:0.95 0.8576.
- **Stage 2:** MobileNetV3-large belt classifier (224px, 120ep cosine-LR, "v4").
- **End-to-end** (detector's own boxes, IoU≥0.5 matched): **acc 0.8605 · P 0.8534 · R 0.7674 ·
  F1 0.8082** — beats the GT-crop classifier-only baseline (F1 0.678).

### 2.4 Red-light running — temporal (needs a clip, not a still)
- LSTM trajectory classifier, split-by-video (21 videos, no leakage).
- **F1 (cross) 0.8995 · acc 0.9218.** Promoted from `not_testable` (rule engine had no scene config).
- Note: single-image uploads can't trigger this — it needs a multi-frame trajectory.

### 2.5 Signal state (is the light red?)
- HSV colour rule (interpretable, tier-0). LISA: **0.9967** on 1500 frames, **0.9324** on 8000.

### 2.6 Helmet & triple-riding — **qualitative capability only (no GT metric)**
- IDD has **no helmet/triple ground truth**, so these are honestly **not benchmarked**.
- **Helmet:** RF-DETR (rider) → helmet model on crop. On 50 IDD frames: ~41% rider-head recall;
  correctly flags clear bare-head riders. Cross-domain bias (PPE-helmet model under-recognises
  full-face motorcycle helmets) → needs a motorcycle-helmet fine-tune to trust.
- **Triple-riding:** rider-count proxy on the motorcycle crop — capability shown, not scored.
- **SAM-3 open-vocab** confirmed it *can* prompt `helmet`/`no helmet`/`triple_riding` on sample
  images (qualitative, logged `20260620-sam3-*`).

### 2.7 Illegal parking — **blocked (no ground truth on disk)**
- Geometry-dwell harness is ready, but ISLab-PVD event-level GT is absent → cannot score.
  Honestly `not_testable` until GT is acquired.

### 2.8 ANPR (plate recognition)
- **OCR engine comparison on GT crops** (N=300–1000, Indian-LP):
  PaddleOCR PP-OCRv5 **exact 0.449 / CER 0.196** (1000) — best of the three;
  EasyOCR 0.483 / CER 0.228 (300); paddle 0.587 / CER 0.174 (300, smaller set).
- A **TrOCR fine-tuned** checkpoint also ships (`checkpoints/anpr/trocr_ft/`) for the live pipeline.
- Scored on **GT plate crops** to isolate OCR from plate-detection (plate localisation is a
  separate step — SAM-3 "license plate" concept in the live pipeline).

---

## 3. Honest status summary

| Violation | Benchmarked? | Headline | Notes |
|---|---|---|---|
| Wrong-side | ✅ real GT | F1 0.955 | strongest module |
| Seat-belt | ✅ real GT | F1 0.808 (e2e) | two-stage, solid |
| Red-light | ✅ real GT | F1 0.900 | needs video clip |
| Signal-state | ✅ real GT | acc 0.93–0.997 | HSV rule |
| ANPR | ✅ real GT | exact 0.449, CER 0.196 | OCR on GT crops |
| Detection | ✅ real GT | mAP@0.5 0.488 | zero-shot; +fine-tune gap |
| Helmet | ⚠️ qualitative | ~41% recall, no GT | needs moto-helmet fine-tune |
| Triple-riding | ⚠️ qualitative | capability shown | rider-count proxy |
| Illegal-parking | ❌ blocked | — | GT absent on disk |

---

## 4b. Prototype: SAM-3 (Roboflow-hosted) entity-detection + geometric rules

The prototype now uses **Roboflow-hosted SAM-3** (`serverless.roboflow.com/sam3/concept_segment`)
as an open-vocabulary entity detector, with per-violation geometric rules ported from
`how_to_segment_images_with_segment_anything_3.ipynb` (`src/modules/sam3_violations.py`). SAM-3
segments concrete nouns only (no negation/attribute reasoning), so we prompt nouns and apply the
rules ourselves. One hosted call per image carries all concept prompts (~2-8s). SAM-3 heuristic
rules are **capped to the `human_review` band** (never auto-challan) and VLM-verified.

Scorecard on the labelled `sample images/` set (small; directional):

| Class | Rule / model | Recall | Clean (no-FP) |
|---|---|---|---|
| helmet | SAM-3: motorcycle present + no helmet over rider upper-body | **6/6** | 3/5 |
| triple-riding | SAM-3: motorcycle + ≥3 overlapping persons | **12/12** | 4/6 |
| red-light (+stop-line) | SAM-3: crosswalk (= stop-line ref) + vehicle past line + HSV-red | **5/5** | 3/5 |
| **wrong-side** | Roboflow `wrong-way-driving-detection/2` (dedicated heading model) | **2/2** | **7/7** |
| seatbelt | trained 2-stage, 4-wheeler-gated | (own testset F1 0.808) | no motorbike FP |

Wrong-side note: SAM-3 cannot detect it (a direction/behaviour attribute, not a noun — text prompts
like "oncoming vehicle" segment ALL cars; exemplar prompts need a within-image positive box we
can't generate). The dedicated Roboflow heading model is the best detector we have (perfect on
samples) and is **VLM-exempt** — the VLM mis-judges direction from a single frame (denied a real
wrong-side), so it would wrongly kill correct detections.

§4b-note: SAM-3 does **not** detect "stop line" (returns nothing), but reliably detects
"crosswalk" — the stop line sits at the crosswalk near-edge, so the crosswalk is used as the line
reference. Red-light additionally requires an HSV-red traffic-light crop. (Earlier "stop line"
prompt gave 0 recall; crosswalk-based numbers logged in run history.)

Reverted/abandoned: the per-class Roboflow detection models (§4 below) were scene-blind
(illegal-parking fired on a red-light scene; no-seatbelt on a motorbike) — replaced by the SAM-3
approach above. The local `.venv-sam3` subprocess (GPU/CUDA-only) is no longer needed.

## 4. (superseded) Per-class hosted-model head-to-head

For the deployed prototype we benchmarked our local models against **Roboflow Universe hosted
models** on a small labelled `sample images/` set (violation + no-violation per class). This is an
**interim** step — the plan is to combine datasets and train our own models to replace these. The
samples are tiny (2–12 imgs/class) so numbers are **directional, not final accuracy**.

Recall = caught violations · Specificity = correctly ignored no-violation images.

| Class | OURS (local) | ROBOFLOW | **Shipped in pipeline** |
|---|---|---|---|
| wrong-side | 0/2 · 5/7 | **2/2 · 7/7** | 🟢 Roboflow `wrong-way-driving-detection/2` |
| triple-riding | **12/12** · 3/6 | 8/12 · 2/6 | 🟢 Ours (local rider-count proxy) |
| illegal-parking | 0/12 · 5/5 | 3/12 · 3/5 (`detection/1`) · or 8/12 · 0/5 (`ltiwv/1`) | 🟡 Roboflow `illegal-parking-detection/1`, **review-only** (conf capped <0.80) |
| red-light | 0/5 · 5/5 | 1/5 · 4/5 | 🔴 none reliable — needs temporal (signal-state + stop-line + crossing) |
| stop-line | 0/5 · 5/5 | 0/5 · 5/5 | 🔴 none reliable — needs per-camera geometry |
| helmet | 0/6 (SAM-3 off) | direct models 0/6; **rule 2–3/6 · 4/5** | 🟡 Roboflow `helmet-violation-49gqq/1` + rider-without-helmet rule, **review-only** (SAM-3 on GPU is the upgrade) |

Helmet note: both direct Roboflow helmet models failed (no usable "no-helmet" class / biased to "No Violation"). The shipped interim is an inference rule — a `rider` with no `with helmet` box over their head — wired via `RoboflowDetector.detect_helmet()`; modest (2–3/6 recall, 4/5 specificity), human-review only, to be replaced by SAM-3 on the GPU box.

**Integration** (`src/modules/roboflow_detect.py` + `src/pipeline.py`): the per-class model map is
in `RoboflowDetector._CLASS_CONFIG` (env-overridable via `ROBOFLOW_MODELS`). wrong-side &
illegal-parking route through Roboflow; triple-riding stays local. Disable with
`PIPELINE_USE_ROBOFLOW=false`. illegal-parking is capped to the `human_review` band (never
auto-challan) because single-image models can't distinguish legal vs illegal parking without
no-parking-zone context.

**Result vs the detect-only baseline:** wrong-side 0/2 → **2/2 (+7/7 clean)**, illegal-parking
0/12 → **3/12 (functional)**, triple-riding **12/12** retained.

---

## 5. Reproducing these numbers

Each row's eval is re-runnable from `ml/src/eval/` (e.g. `python -m src.eval.eval_detection ...`,
`eval_seatbelt_e2e`, `eval_wrongside`, `eval_anpr`, `eval_redlight_sequence`, `eval_signal`).
Full per-run detail is in `ml/results/eval_<task>_<run-id>.md`. The consolidated log is
`ml/results/run_history.csv`.

_Last compiled: 2026-06-23._
