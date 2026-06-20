# Gridlock 2.0 R2 — Implementation Checklist

> Living progress tracker. Source of truth for architecture: `docs/final/00_master_design.md` (§2 flow, §3 modules, §7 eval, §8 fine-tune), defended in `01_justifications.md`, model picks in `06_model_selection_justification.md`, datasets in `04_datasets_acquisition_and_prep.md`. **Implement, do not redesign.**
>
> Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/not-ready · `[-]` skipped (by design)

---

## Phase 0 — Environment & Data Audit (MANDATORY, before any pipeline code)  ✅ DONE
- [x] **0a Hardware detection** — `src/hardware.py` → `logs/environment_report.md`.
  - **TIER = `cloud_required`**: RTX 3050 Laptop, **4 GB VRAM**, Ryzen 7 6800HS, 15 GB RAM, Python 3.14, torch 2.12+cu126 (CUDA available). 4 GB → RF-DETR-L / Co-DETR / VLM **NOT runnable locally** → stub locally, push to cloud (Lightning H200). §8 A100 budgets do not apply.
  - **⚠️ ML stack mostly NOT installed** (Py3.14): missing cv2, torchvision, rfdetr, ultralytics, supervision, paddleocr, paddle, easyocr, transformers, pycocotools, lxml. Present: numpy, PIL, pandas, yaml, torch, huggingface_hub. → Phase 2/3 cannot run locally until env is sorted (and several may lack 3.14 wheels).
- [x] **0b Dataset integrity audit** — `src/data_audit.py` → `logs/dataset_audit_report.md`. **7/11 ready.**
  - [x] 1 IDD ✅ **46,659 img / 41,857 XML**, train/val/test.txt present, VOC parses
  - [!] 2 BDD100K ❌ only 3 `.md5` files → NOT READY, skipped (no placeholder)
  - [x] 3 CCPD ✅ **11,776 img** (CCPD2020 green subset), filename GT decodes (7 fields, 8 LP indices)
  - [x] 4 LISA ✅ **44,075 frames**, 24× frameAnnotationsBOX.csv
  - [~] 5 Indian-LP 🟨 **only 1,741 img on disk** (sirishan 1,694 + dcl 47) vs documented 16,192 — **extraction truncated ~89%**; 1,740 XML sidecars present. Re-extract if more needed.
  - [~] 6 UA-DETRAC 🟨 **140,131 img** but **annotation_xml=0** (annos in non-XML form / different layout — inspect; optional anyway, IDD covers detection)
  - [!] 7 AICC ❌ **imagery ABSENT** — on disk = winning-solution CODE repo + 2 annotation CSVs only (labels {1,2,3,4,5,7}, ≤7-class). Helmet+triple **not_testable** until raw Track-5 data downloaded.
  - [x] 8 Seatbelt+Mobile ✅ OBB, train 779 / valid 337 (no test), 3 cls (mobile=out-of-scope)
  - [~] 9 ISLab-PVD 🟨 **16 .mp4, zero GT files** → quantitative event-eval impossible; qualitative only until annotated
  - [x] 10 RunningRedlight ✅ **1,331 clip JSONs / 15,839 frames**, `meta.cross` bool label (~130/70 in first 200)
  - [x] 11 Wrong-Way ✅ OBB, train 426 / valid 91 / test 91, 2 cls
- [x] **OBB utility** — `src/utils/obb_convert.py` written (polygon→AABB + angle/heading proxy + CLI). Both #8/#11 confirmed 9-token OBB.
- [x] **0c Violation coverage audit** — done (see report). Quantitative-ready: detection, ANPR, seatbelt, red-light signal (LISA), red-light event (RunningRedlight), wrong-side. **Blocked:** helmet/triple (no imagery), illegal parking (no GT), BDD100K (empty). **Qualitative only:** stop-line.

## Phase 1 — Project Restructuring  🟨 IN PROGRESS
- [x] Package skeleton created (`src/modules/`, `src/eval/`, `src/train/`, `src/utils/`, `configs/`, `checkpoints/`, `logs/`, `results/`) with `__init__.py`. No notebooks.
- [x] `src/utils/logging.py` (run JSON + run_history.csv), `src/utils/obb_convert.py`, `src/utils/registry.py`
- [ ] `configs/` YAML per module/dataset (pending env outcome to set model variants)
- [ ] Decision on IDD role: helmet=NOT possible (no helmet labels); IDD used for rider/motorcycle detection + triple-riding **proxy** (rule-based, qualitative — no GT triple label). data-cluster-labs Indian-LP = ~47 img tiny OCR test set.

## Phase 2 — Baseline Pipeline (zero-shot)  🟨 IN PROGRESS
- [x] Env: stack installs on **Python 3.14** (rfdetr 1.8.0, opencv 4.13, supervision 0.29, pycocotools, transformers). ⚠️ pip swapped CUDA torch → CPU; cu126 restore downloading (bg `blt1yte0a`).
- [x] `detection.py` (RF-DETR) runs — localization excellent (IoU 0.89 on a real IDD box), 0.64 s/img CPU.
- [x] `quality_gate.py` (cv2 IQA), `confidence_cascade.py`, `evidence.py` (sha256+annotate), `geometry_engine.py` (SCM rules).
- [x] ✅ **DETECTOR ISSUE RESOLVED** — was NOT a degraded model; rfdetr emits **1-indexed COCO** ids and my map was 0-indexed (person→bicycle etc.). Fixed to use rfdetr's own `COCO_CLASSES`. Re-eval: **mAP@0.5=0.404, mAP@0.5:0.95=0.227** on IDD (100 img, zero-shot) — sensible baseline vs DriveIndia 0.787 fine-tuned. Per-class: car .63, truck .70, person .49, bus .31, moto .29.
- [x] all modules written: `preprocessing.py` (CLAHE/gamma fallback), `tracking.py` (IoU tracker), `helmet_triple.py`, `seatbelt.py`, `anpr.py` (lazy OCR), `signal_state.py` (HSV Tier-0), `geometry_engine.py`, `pipeline.py` orchestrator.
- [x] **Orchestrator runs end-to-end** on real IDD (detect→helmet/triple→cascade→signed evidence); 40-img batch clean, evidence gated on violations, OCR lazy (no startup download).
- [x] **Violation paths unit-tested** (synthetic): triple-proxy fires at 3 riders, all 4 geometry rules fire, red-light abstains w/o signal, cascade bands correct, evidence sha256+annotate works. 8/8 PASS.
- [x] AICC helmet → `not_testable` (no checkpoint, no imagery); NOT passed off as a detector result. ✓

## Phase 3 — Baseline Evaluation  🟨 IN PROGRESS
- [x] **Detection (IDD)** — `eval_detection.py`: zero-shot RF-DETR-nano. 100-img: mAP@0.5=0.404. **600-img (stable): mAP@0.5=0.418, mAP@.5:.95=0.243** — truck .68/car .62/person .45/moto .35/bus .31/bicycle .10/traffic-light .00. Domain-gap (traffic-sign 156, vehicle-fallback 103, autorickshaw 50) reported separately, not scored. bicycle was a 100-img sampling artifact; traffic-light genuinely fails (too small).
- [x] **Signal-state (LISA)** — `eval_signal.py`: HSV Tier-0 **acc=0.932** on 8,000 frames (red recall .998 ✓, green .91, yellow .27 weak). Fit-for-purpose: red-light rule only needs red.
- [x] **ANPR (Indian-LP)** — `eval_anpr.py`, 300 plates, EasyOCR (Paddle has **no Py3.14 wheel** → design's named backup). Baseline **0.31** → **IMPROVED to 0.483 exact-match** (CER 0.31→0.228, 1-NED 0.69→0.772) via **(a) multi-line merge** (0.31→0.383, recovers bottom-row-only) **+ (b) syntax-aware correction** (0.383→0.483, fixes 0↔O). No fine-tune. Corrector unit-tested: fixes confusions, never breaks correct plates. Remaining: state-code misreads, row-order, missing chars → PP-OCRv5+fine-tune (cloud).
- [x] **Wrong-side (Wrong-Way OBB)** — `eval_wrongside.py`: verdict `not_testable` (Tier 2, geometry needs motion) but **det recall 0.608** on labeled vehicles → working base for learned-classifier escalation. OBB angle exercised.
- [x] **Red-light event (RunningRedlight)** — `eval_redlight_sequence.py`: 500 clips, `not_testable` Tier 2; agreement-rate harness ready (rule-vs-learned, J3).
- [x] **Seatbelt** — `eval_seatbelt.py`: confirmed `not_testable` Tier 1; fine-tune data counted (779/337 OBB).
- [x] **Illegal parking** — `eval_illegal_parking.py`: `not_testable` (16 videos, GT absent); event-level `event_pr()` harness ready for alt dataset.
- [x] **Stop-line** — `smoke_test.py`: 25 annotated qualitative frames → `results/stopline_spotcheck/` (clearly NOT a metric).
- [x] **Consolidated report** → `results/PHASE3_4_baseline_and_triage.md` (quantitative vs blocked vs qualitative).

## Phase 4 — Gap Analysis & Triage  ✅ DONE → `results/PHASE3_4_baseline_and_triage.md`
- [x] **Tier 0** (skip Phase 5): signal-state HSV (red recall .998 ✓), illegal-parking dwell rule.
- [x] **Tier 1** (mandatory fine-tune): helmet+triple (AICC — blocked on imagery), seatbelt (OBB ready), **detection promoted** (0.404→need 0.787, Indian classes; IDD/DriveIndia on disk = cheapest ROI).
- [x] **Tier 2** (conditional): wrong-side (promote→learned classifier on 608-img set), red-light event (rule-vs-learned agreement).
- [x] Priority queue set: detection → helmet → seatbelt → ANPR → wrong-side → redlight-event.

## Phase 5 — Targeted Fine-Tuning Loop (Tier 1 + promoted Tier 2 ONLY)  🟨 IN PROGRESS
**Hardware reality (recomputed per Phase 5 instruction):** 4 GB GPU + CPU-only torch → heavy Tier-1 fine-tunes (RF-DETR detection, helmet) are **cloud work**; helmet also data-blocked. Locally-feasible: small instance classifiers.
- [x] **Wrong-side (Tier 2 promoted) — DONE, target cleared in 1 run.** `train_wrongside.py`: MobileNetV3-small (pretrained, ~10MB), crops vehicle instances from OBB, class-weighted loss (7:1), **NO h-flip** (heading is the signal). 15 epochs, **3.8 min on CPU**. Held-out **TEST: acc 0.989, wrong-side P 0.977 / R 0.934 / F1 0.955** vs baseline `not_testable`. Checkpoint `checkpoints/wrongside/v1/model.pt`. `eval_wrongside.py --weights` added. **STOP (criterion 1: beats target).**
- [~] **Detection (Tier 1) — CLOUD SCRIPTS READY for Lightning H200.** `prepare_idd_coco.py` (VOC→Roboflow-COCO, smoke-tested: 15 classes incl. autorickshaw, valid COCO, integrity checks pass) + `train_detection.py` (RF-DETR fine-tune, Apache). Full convert = 31,569 train / 10,225 val (test unlabeled→skipped). Commands handed to user. Awaiting user's Lightning run.
- [x] **Seatbelt (Tier 1) — DONE (best-effort, 3-attempt cap reached).** `train_seatbelt.py`: reframed OBB→two-stage (crop windshield, label = belt-box-inside?). MobileNetV3-small. 3 attempts: v1 128px F1=0.635 → **v2 224px+wd F1=0.678 acc=0.766 (BEST)** → v3 frozen-backbone regressed 0.619. Held-out valid: no_seatbelt P=0.74/R=0.59/F1=0.678. Honest "best-effort daytime" (§3.5) — subtle strap + label noise + 780 crops. Canonical: `checkpoints/seatbelt/v2/`. Mobile class counted (349) but out-of-scope. **STOP (criterion 2: cap reached, v2 best).**
- [~] **Helmet+triple (Tier 1)** — zero-shot pretrained test DONE (Phase 6). NVIDIA TAO Indian model = paper-only (no weights); AICC repo = code only. Tested `leeyunjai/yolo11-helmet` (PPE-domain) via RF-DETR two-stage on 50 IDD moto-frames: **capability shown** (correctly flags clear no-helmet riders, e.g. 000492_r.jpg) but ~41% rider recall + cross-domain bias (under-calls motorcycle helmets) → not a metric, needs fine-tune. Findings: `results/helmet_zeroshot_findings.md`. Still BLOCKED for a real metric on AICC imagery; Roboflow motorcycle-helmet model (needs API key) is a better zero-shot upgrade.
- [x] **Track B scripts ready for Lightning** — `test_locateanything.py` (LocateAnything-3B phrase grounding) + `test_sam3.py` (SAM-3 concepts) + runbook `docs/final/07_trackB_foundation_models_lightning.md`. Compile-verified. Can't run on 4GB → H200.
- [x] **ANPR — PP-OCRv5/v6 primary NOW RUNNING locally.** Created **Python 3.12 venv** (`.venv-paddle`) since paddle has no Py3.14 wheel; installed paddlepaddle 3.3.1 + paddleocr 3.7.0. anpr.py updated for 3.x API (`.predict()`, `rec_texts/rec_scores/rec_polys`) + oneDNN disabled (FLAGS_use_mkldnn=0 — fixes paddle-3.x CPU PIR crash). **PP-OCR > EasyOCR on matched 300 (0.587 vs 0.483).** **Larger 1000-plate sample (representative): plate exact 0.449, CER 0.196, 1-NED 0.802** — the 300 was an optimistic/cleaner subset. Run: `.venv-paddle/Scripts/python -m src.eval.eval_anpr --ocr paddleocr --limit 1000`. NOTE: full 1692-run crashed ~img1000 (paddle native crash on some image; uncatchable segfault) — cap at 1000 or add subprocess isolation if full set needed.
- [ ] **ANPR fine-tune** (further) — Indian-plate fine-tune of PP-OCR rec model → cloud. Remaining errors: validity-date text appended on plate, occasional J↔I, partial reads.
- [x] **Red-light event (Tier 2) — DONE, target cleared in 1 run.** `train_redlight.py`: LSTM(6→48) over vehicle trajectory (cx,cy,w,h + velocity), resampled to T=32. **Split by VIDEO** (leakage-safe): 1071 train / 243 test from 21 held-out videos. 30 epochs, **3.6 sec**. Held-out **TEST: acc 0.922, cross P 0.914 / R 0.885 / F1 0.900** vs baseline not_testable. Checkpoint `checkpoints/redlight/v1/`. **STOP (criterion 1).** Note: rule-vs-learned agreement (J3) NOT computable — anonymized clips lack scene config/signal state; harness stays in `eval_redlight_sequence.py`.

## Phase 6 — Escalation: Model/Module Substitution (STRETCH — only after all Tier 1 resolved)
- [ ] Optional, time-permitting; NOT required for a working submission. Enter only once every Tier 1 (and promoted Tier 2) module has hit target or exhausted the Phase-5 cap.
- [ ] On plateau: search arXiv/GitHub/HF/web for an alternative (Twitter/LinkedIn best-effort via general web search). Wrong-side escalation = direct learned classifier on Wrong-Way dataset. License filter per §12 (Apache/MIT ship; AGPL/GPL benchmark-only). Log every candidate to `03_sota_registry.md`; update `06_...md` row, mark prior choice **superseded** (don't delete).

## Phase 7 — Final Report (tier table MANDATORY; full comparison is stretch)
- [ ] **Mandatory:** tier-by-tier outcome table from Phase 4/5 — Tier 0 pass/fail, Tier 1 baseline→fine-tuned, Tier 2 baseline→promoted?→fine-tuned. This alone confirms the system works end-to-end across all 7 violation types; required even if time runs out.
- [ ] **Stretch (if time):** fuller table baseline→fine-tuned→swapped vs §7 targets, each row tagged quantitative/qualitative. Include redlight rule-vs-learned agreement rate (#10). Updated coverage map (only stop-line qualitative).

## Documentation Sync
- [ ] Update `docs/final/04_datasets_acquisition_and_prep.md` with all 11 datasets (real on-disk structure from 0b, same table format). Mark original `seat-belt-detection-uhqwa` superseded by `seat_belt-and-mobile`.

## Cross-cutting
- [ ] Logging: every run → `logs/<phase>/<module>_<run_id>.json` (timestamp, module, ckpt ver, dataset, hardware, metrics, pass/fail vs §7) + one-line append to `results/run_history.csv`.
- [ ] Stopping criteria per module: (1) meets §7 target, OR (2) 3 fine-tunes + 1 swap w/o gain → `plateaued, likely data-limited`, OR (3) stop-line → qualitative only.

---
### Run log
- 2026-06-19: TODO.md created. Starting Phase 0.
- 2026-06-19: Phase 0 COMPLETE. `src/hardware.py` + `src/data_audit.py` written & run. Reports in `logs/`. Tier=cloud_required (4 GB GPU). 7/11 datasets ready. Blockers: AICC imagery, ISLab GT, BDD100K, Indian-LP truncation, local ML stack not installed. **Awaiting direction on execution environment before Phase 2.**

### ⚠️ ENVIRONMENT — THREE interpreters (use explicit paths!)
- **★ 3.12 system (CUDA)** = `C:\Users\sorat\AppData\Local\Programs\Python\Python312\python.exe` — **torch 2.7.1+cu128, CUDA TRUE** on RTX 3050 (4 GB). **Primary for GPU work** (detection fine-tune, GPU inference). Installing rfdetr+ultralytics+pycocotools here.
- **3.14 system** (`C:\Python314\python.exe`): CPU torch + rfdetr + ultralytics + easyocr. The CPU baseline env (Phases 2-5 ran here).
- **3.12 paddle venv** (`.venv-paddle\Scripts\python.exe`): paddleocr (PP-OCRv5), CPU. ANPR only (`FLAGS_use_mkldnn=0`).
- **GPU = 4 GB hard limit:** RF-DETR-nano fine-tune fits (small batch); LocateAnything-3B (~12 GB) does NOT fit → Track B big-VLM here will OOM unless quantized.
- **venv/PATH leak:** bare `python` is unreliable → **always call the explicit interpreter path**.

### Phase 0 blockers to resolve (need data/decisions)
- [ ] AICC raw Track-5 videos/frames (registration-gated) — required for helmet+triple
- [ ] ISLab-PVD event-level GT (or accept qualitative-only for illegal parking)
- [ ] BDD100K real images (or drop — IDD/UA-DETRAC cover detection)
- [ ] Indian-LP re-extraction (only 1,741/16,192 extracted)
- [ ] Execution env decision: local CPU baseline vs cloud-first (H200); Python 3.14 wheel availability for cv2/ultralytics/rfdetr/paddle
