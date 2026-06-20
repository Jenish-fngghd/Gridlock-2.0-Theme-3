# Gridlock 2.0 — R2 · Model & Module Selection Justification Sheet
### Companion to `00_master_design.md` · deeper narrative defenses → `01_justifications.md` (J#)

**What this is:** a one-row-per-choice sheet for *every* model/module in the master-design pipeline — **what** we picked, **where** it runs, **what we picked it over**, and **why**. Where a *module/technique* was chosen rather than a single named model, the "Selected" cell names the module and the "Type" cell says `module`. The hardest "why not the obvious alternative" calls have a full narrative defense in `01_justifications.md` (referenced as **J#**).

> License legend: ✅ = shippable (Apache/MIT/permissive) · ⚠️ = copyleft/restricted (study-only, re-implement, or internal-benchmark-only).

---

## A. Perception spine (ingest → detect → track)

| # | Selected model / module | Type | Where used in pipeline | Alternatives considered | Why selected over the alternatives | License |
|---|---|---|---|---|---|---|
| 1 | **pyiqa / piqa** IQA scorer | model | §3.0 — Ingest quality gate (step 0) | classical BRISQUE/NIQE; no gate at all | Learned, lightweight blur/exposure/glare scoring; **gates** restoration so we don't waste compute or add artifacts on already-clean images. Classical metrics are noisier; "no gate" means restoration runs blindly. | ✅ |
| 2 | **Retinexformer** | model | §3.1 — Preprocessing, low-light branch | RetinexMamba, MambaLLIE, Zero-DCE++ | ICCV'23, **NTIRE 2024 runner-up** (verified); strong transformer LLIE with a public checkpoint. Used **only** when IQA flags low-light; kept behind an ablation (drop if it doesn't lift detector mAP). | ✅ |
| 3 | **OneRestore** | model | §3.1 — Preprocessing, weather branch | separate task-specific models (Rain100 / SOTS / GoPro) | One model restores **composite** rain/haze/blur instead of chaining three single-degradation models → simpler, fewer artifacts at boundaries. | ✅ |
| 4 | **RF-DETR-N/S** | model | §3.2 — Detection **Stage-1 fast screen** | YOLOv12, YOLOv8/11, RT-DETR | **Apache-2.0** (license-clean) with a **DINOv2** backbone → strong on small/occluded/dense Indian scenes and fine-tunes well; one family covers both stages. YOLO/YOLOv12 are **AGPL** → can't ship. **(J6)** | ✅ |
| 5 | **RF-DETR-L / Co-DETR** | model | §3.2 — Detection **Stage-2 confirm** | reuse Stage-1; single-stage detector | RF-DETR-L = first real-time **60+ mAP COCO** (verified); **Co-DETR** is the **AICC helmet rank-1 backbone** (verified) → highest-precision second opinion on candidate crops only. | ✅ / open |
| 6 | **YOLOv12** | model | §3.2 — **internal benchmarking only** (never shipped) | — | Strong/fast (40.6% mAP-N @1.64 ms T4, verified) so it's a useful *comparator*, but **AGPL-3.0** → kept out of the distributed submission. **(J6)** | ⚠️ AGPL |
| 7 | **YOLO-World / GroundingDINO** | model | §3.2 / §3.10 — cold-start & offline auto-label | manual labeling only | Open-vocab, label-free detection of rare Indian classes (auto-rickshaw, cart) for the **data engine**; not used as the latency-critical primary. **(J4)** | ✅ |
| 8 | **BoostTrack / BoostTrack++** | model | §3.3 — Tracking (persistent IDs) | ByteTrack, BoT-SORT, OC-SORT | **HOTA 69.25 > ByteTrack 67.68** on MOT17 (verified); plugged via **boxmot** so BoT-SORT/ByteTrack remain drop-in fallbacks. IDs are the prerequisite for every temporal violation. **(J5)** | ✅ |

## B. Paradigm A+B — helmet, triple-riding, seatbelt

| # | Selected model / module | Type | Where used in pipeline | Alternatives considered | Why selected over the alternatives | License |
|---|---|---|---|---|---|---|
| 9 | **AICC Track 5 seven-class scheme** | module | §3.4 — Helmet + Triple-riding (the anchor) | separate helmet-classifier + separate rider-counting head; pure-SAC/pose | One labelled scheme (`motorbike, D/P1/P2 ×Helmet/NoHelmet`) encodes **helmet compliance AND rider count together** → both violations from one model, with a **public dataset + proven winning recipes** to beat. **(J2)** | ✅ (re-impl.) |
| 10 | **Two-stage detect→crop→classify** | module | §3.4 — inside the 7-class module | single-stage whole-frame classification | All AICC winners converge here: detect motorbike @1280–1536px → crop → classify 7 states recovers small/distant riders a single pass misses. | ✅ |
| 11 | **SAM2 masks + RTMPose keypoints** (SAC-style) | model | §3.4 — rider↔bike **association** (primary) | overlap + nearest-x-center heuristic only | Learned masks+pose give robust head→rider→bike attribution in dense overlap; the **VNPT overlap(0.3)+nearest-x(0.6)** heuristic is **re-implemented as a license-clean fallback**, not the primary. | ✅ |
| 12 | **Windshield → driver-crop → belt CNN/CNN-SVM** | module | §3.5 — Seatbelt (daytime best-effort) | end-to-end whole-frame seatbelt classifier | Belt is a tiny ROI behind glass; the two-stage crop is the only thing that works. Honestly scoped (night/tint/glare = failure modes). | ✅ |

## C. Paradigm C — scene-context / temporal violations

| # | Selected model / module | Type | Where used in pipeline | Alternatives considered | Why selected over the alternatives | License |
|---|---|---|---|---|---|---|
| 13 | **Geometry-as-config Scene Context Model** | module | §3.6 — red-light / wrong-side / stop-line / parking | learned end-to-end temporal/video model | Cameras are **fixed** → annotate stop-line/lanes/no-park/signal-ROI **once** per camera. Interpretable, **auditable, court-admissible**, and needs **no** (nonexistent) labelled temporal-violation dataset. **(J3)** | ✅ |
| 14 | **Signal-state classifier (LISA/BSTLD-trained)** | model | §3.6 — red-light rule input | VLM read of the signal | A cheap deterministic classifier on the signal ROI beats paying a VLM per frame for a 3-state output; pairs with the geometry rule (red **AND** crossing). | ✅ |
| 15 | **LocateAnything3D** (CVPR'26, Chain-of-Sight) | model | §3.6 — single-frame monocular-3D | nothing (can't judge from a still otherwise) | Gives monocular **3D yaw / ground position** so wrong-side & stop-line can be decided from **one still** where no clip exists. Foundation-model role only (teacher/verifier), never the throughput path. **(J1)** | ⚠️ verify |

## D. ANPR (ROI-gated)

| # | Selected model / module | Type | Where used in pipeline | Alternatives considered | Why selected over the alternatives | License |
|---|---|---|---|---|---|---|
| 16 | **RF-DETR plate detector** | model | §3.7 — plate localization | separate YOLO plate detector | Reuses the Apache spine; no extra AGPL dependency for a single extra class. | ✅ |
| 17 | **PaddleOCR PP-OCRv5** | model | §3.7 — plate OCR (primary) | PP-OCRv4, PARSeq, EasyOCR, VLM-only | **+13pp E2E over v4** (verified), handles rotation/skew. **PP-OCRv4 kept as the high-throughput option** (v5's bigger dict is slower) and **EasyOCR as backup**. ANPR is ROI-gated, so accuracy wins. | ✅ |
| 18 | **Indian-format (HSRP) syntax validator** | module | §3.7 — post-OCR | accept raw OCR string | State-code + HSRP regex rejects impossible plates and corrects near-misses → higher effective plate accuracy. | ✅ |
| 19 | **Self-reflective VLM** (arXiv 2508.01387) | model | §3.7 — low-conf cross-check + make/model | OCR only | Independent OCR cross-check on low-confidence plates **and** adds make/model (UFPR-ALPR 83% plate / 61% make-model, verified) for richer challans + repeat-offender ReID. **Augments, never replaces** the OCR (its plate acc < a fine-tuned OCR). | ✅ |

## E. Decision, evidence, serving + offline

| # | Selected model / module | Type | Where used in pipeline | Alternatives considered | Why selected over the alternatives | License |
|---|---|---|---|---|---|---|
| 20 | **Temperature scaling + per-class thresholds + abstain band** | module | §3.8 — confidence cascade (step 5) | raw softmax confidence | Calibrated scores + 3 bands (auto-confirm / human-review / discard) satisfy the explicit "assign confidence scores" ask and make auto-challan defensible. | ✅ |
| 21 | **Qwen2.5-VL / InternVL3 / LocateAnything-3B** | model | §3.9 — VLM verify + caption (step 6) | per-frame VLM; closed GPT-4V | Open-source; run **only on low-confidence cases** (cascade) → precision + NL evidence without per-frame VLM cost. Per-frame use is ~200–1000× too expensive. **(J1)** | ✅ |
| 22 | **Signed JSON + SHA-256 + audit trail** | module | §4 — evidence generator (step 7) | plain record / image only | Tamper-evident, versioned, court-admissible — the differentiator vs a "helmet detector" demo. | ✅ |
| 23 | **SQLite/Postgres + object store + Streamlit/FastAPI** | module | §5 — store + analytics dashboard (8/9) | heavyweight data platform | Lightweight, searchable records + exportable reports + hotspot/trend views; fast to build for the MVP. | ✅ |
| 24 | **GroundingDINO / Grounded-SAM-2 / Florence-2 / LocateAnything-3B → distill** | module | §3.10 — offline data engine | hand-label everything | Open-vocab teachers auto-label rare Indian classes → human spot-check → **distill into Stage-1 RF-DETR**, slashing labeling cost. Runs offline, off the latency path. **(J1, J4)** | mixed (teacher-only) |

---

### Cross-references
- **J1** — why the VLM is a verifier/teacher, not the per-image screen (rows 7, 15, 21, 24).
- **J2** — AICC 7-class vs pure SAC/pose (row 9).
- **J3** — geometry-as-config vs learned temporal model (row 13).
- **J4** — why fine-tune vs pure zero-shot (rows 7, 24).
- **J5** — BoostTrack vs ByteTrack/BoT-SORT (row 8).
- **J6** — RF-DETR (Apache) vs YOLOv12 (AGPL) (rows 4, 6).
- Per-model datasets, GPU-time and fine-tuning priority → `02_comparison_merge_finetuning.md` §6.
