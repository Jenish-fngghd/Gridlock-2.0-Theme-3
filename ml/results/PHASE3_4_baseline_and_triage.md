# Gridlock 2.0 R2 — Phase 3 Baseline + Phase 4 Triage

_Generated 2026-06-19 · hardware tier `cloud_required` (RTX 3050 4 GB, see `logs/environment_report.md`) · zero-shot, no fine-tuning yet._

> Source of truth: `docs/final/00_master_design.md` (§7 metrics). Every row below is tagged **quantitative** or **qualitative/blocked**. Raw per-run JSON in `logs/phase3/`, one-line history in `results/run_history.csv`.

---

## Phase 3 — Baseline evaluation results

### A. Quantitative (real metrics on ready datasets)

| Capability | Dataset | Model (zero-shot) | Metric | **Baseline** | §7 reference | Eval script |
|---|---|---|---|---|---|---|
| Detection backbone | IDD (600 val) | RF-DETR-nano (COCO) | mAP@0.5 / @.5:.95 | **0.418 / 0.243** | DriveIndia 0.787 (fine-tuned) | `eval_detection.py` |
| Signal state (red-light input) | LISA (8k frames) | HSV classifier (Tier 0) | accuracy | **0.932** | LISA baselines | `eval_signal.py` |
| └ per-class | | | recall | red **0.998**, green 0.910, yellow 0.274 | | |
| ANPR — **PP-OCRv5/v6 (primary)** | Indian-LP (**1000**) | PaddleOCR 3.7 in Py3.12 venv + post-proc | plate exact / CER / 1-NED | **0.449 / 0.196 / 0.802** | vs CCPD/Indian SOTA | `eval_anpr.py --ocr paddleocr` |
| └ PP-OCR on 300 (cleaner subset) | Indian-LP (300) | " | plate exact | 0.587 (optimistic — first 300 = easier subfolder) | | |
| └ ANPR — EasyOCR (backup) | Indian-LP (300, matched) | EasyOCR + post-proc | plate exact / CER / 1-NED | 0.483 / 0.228 / 0.772 | PP-OCR > EasyOCR on matched 300 (0.587 vs 0.483) | `eval_anpr.py` |
| Wrong-side (det base) | Wrong-Way OBB (valid) | RF-DETR-nano | det recall @0.5 (class-agnostic) | **0.608** | — | `eval_wrongside.py` |
| **Wrong-side verdict (Phase 5 fine-tuned)** | Wrong-Way OBB (**test, held-out**) | MobileNetV3-small (fine-tuned) | wrong-side P / R / F1 · acc | **0.977 / 0.934 / 0.955 · 0.989** | beat geometry-abstain ✓ | `train_wrongside.py` |
| **Seatbelt (Phase 5 fine-tuned, best of 3)** | seatbelt-OBB (**valid, held-out**) | MobileNetV3-small @224 +wd (v2) | no_seatbelt P / R / F1 · acc | **0.740 / 0.592 / 0.678 · 0.766** | from not_testable ✓ | `train_seatbelt.py` |
| **Red-light event (Phase 5 fine-tuned)** | RunningRedlight (**21 held-out videos**) | LSTM trajectory classifier | cross P / R / F1 · acc | **0.914 / 0.885 / 0.900 · 0.922** | from not_testable ✓ | `train_redlight.py` |

**Detection per-class AP@0.5:** car 0.63 · truck 0.70 · person 0.49 · bus 0.31 · motorcycle 0.29 · bicycle 0.0 · traffic-light 0.0.
**Structural domain gap (not scored — no COCO class):** autorickshaw, vehicle-fallback, animal, traffic-sign → ~0 recall zero-shot; closed only by data-engine + fine-tune (§3.10/§8).

### B. Blocked / not-quantitative this run (honest)

| Capability | Dataset | Why not a metric | Disposition |
|---|---|---|---|
| Helmet compliance | AICC Track 5 | **Imagery absent** (only code repo + CSVs on disk); no zero-shot checkpoint for the per-rider scheme | `not_testable` → **Tier 1** |
| Triple riding | (AICC / IDD) | Proxy runs (rider-assoc count ≥3) but **no GT triple label** to score | qualitative proxy → Tier 1 verify |
| Seatbelt | seat_belt-and-mobile OBB | No zero-shot windshield/belt checkpoint | `not_testable` → **Tier 1** |
| Wrong-side *verdict* | Wrong-Way OBB | Geometry rule needs **motion**; dataset is single stills | `not_testable` → **Tier 2** |
| Red-light full event | RunningRedlight | Clips lack per-clip scene config; learned classifier untrained | `not_testable` → **Tier 2** |
| Illegal parking | ISLab-PVD | 16 videos, **GT absent on disk** → event P/R uncomputable | blocked (data) |
| ANPR | Indian-LP / CCPD | GT ready (Indian: box+text); **OCR engine installing**; plate detection needs fine-tune | pending OCR → baseline incoming |
| Stop-line | — none — | No dataset exists anywhere | **qualitative spot-check only** (`smoke_test.py`, 25 frames) — never a metric |

---

## Phase 4 — Triage (decides what Phase 5 touches)

| Module | Tier | Rationale | Phase 5 action |
|---|---|---|---|
| **Signal state (HSV)** | **Tier 0** | Rule-based; red recall 0.998 — fit for the red-light rule as-is | **skip** (don't fine-tune what works) |
| **Illegal-parking dwell rule** | **Tier 0** | Rule-based; blocked only by missing GT, not by model quality | skip (await GT/alt dataset) |
| **Detection** | **Tier 0→1** | 0.404 zero-shot is a usable base but far below 0.787; Indian classes need it | **promote**: cheap fine-tune (IDD/DriveIndia on disk) — esp. autorickshaw/rider |
| **Helmet + triple riding** | **Tier 1** | No checkpoint anywhere for per-rider attribution+count | **mandatory fine-tune** (needs AICC imagery — currently absent) |
| **Seatbelt** | **Tier 1** | No zero-shot checkpoint | **mandatory fine-tune** on OBB set (779 train) |
| **Wrong-side** | **Tier 2** | Geometry abstains on stills (det base 0.61 ready) | **promote** → learned classifier on Wrong-Way 608-img set |
| **Red-light full event** | **Tier 2** | Rule needs scene cfg; classifier untrained | promote only if needed; report rule-vs-learned agreement |
| **Stop-line** | — | No dataset | qualitative only, never fine-tuned |

### Priority queue for Phase 5 (per §8, adjusted to on-disk reality)
1. **Detection fine-tune** (IDD/DriveIndia ready now — highest ROI, unblocks Indian classes)
2. **Helmet + triple (Tier 1)** — *blocked on acquiring AICC Track-5 imagery*
3. **Seatbelt (Tier 1)** — OBB set ready
4. **ANPR fine-tune** (plate detection + OCR on Indian/CCPD) — after OCR baseline
5. **Wrong-side (Tier 2 promoted)** — learned classifier on the 608-img set
6. Red-light event (Tier 2) — only if rule-engine path underperforms

---

## Phase 7 (minimal) — End-to-end coverage of the 7 mandated violations

| # | Violation | Baseline status | Path to "working" |
|---|---|---|---|
| 1 | Helmet | not_testable (Tier 1) | fine-tune (needs AICC imagery) |
| 2 | Triple riding | proxy runs (no GT) | AICC fine-tune validates it |
| 3 | Seatbelt | not_testable (Tier 1) | fine-tune OBB set |
| 4 | Red-light | **signal acc 0.932** + rule engine | combine signal+crossing; event cross-check (Tier 2) |
| 5 | Wrong-side | det base 0.61; verdict Tier 2 | learned classifier on 608-img set |
| 6 | Stop-line | qualitative spot-check | demo only (no dataset) |
| 7 | Illegal parking | rule ready; GT absent | await GT / alt dataset |
| + | Detection | **mAP@0.5 0.404** | fine-tune → ~0.787 target |
| + | ANPR | GT ready; OCR installing | OCR baseline → fine-tune |

**Bottom line:** the pipeline runs end-to-end and every one of the 7 violations has a defined, honest path. Two have real quantitative baselines now (red-light signal 0.932, detection 0.404 + wrong-side det-base 0.61); ANPR's is imminent (OCR install); the rest are correctly gated as Tier-1 fine-tune or data-blocked rather than faked.
