# Automated Traffic Violation Detection — Architecture & Solution Framework

**Hackathon:** Flipkart Gridlock 2.0 — Phase 2 (idea submission / solution framework)
**Problem:** Automatically process traffic images → detect vehicles & road users → identify
& classify violations → recognize license plates → generate annotated evidence + metadata →
analytics, robust to weather/light/density/quality.
**Data stance:** No proprietary data → train/evaluate on **public benchmarks**; demo on
benchmark samples + a few real-world stress images.
**This doc** is the single source of truth, consolidating: PLAN, RESEARCH_FINDINGS,
AICITY_INDIAN_DATASETS, REPOS_REVIEW, GEOMETRY_VIOLATIONS. Metrics marked **VERIFY** need
source confirmation before quoting to judges.

---

## 1. Executive Summary
We propose a **modular, geometry-aware, confidence-cascaded** traffic-violation pipeline.
A robust detector + tracker feeds (a) a **two-stage helmet/rider module** (adopts the AI
City Challenge 7-class scheme that encodes helmet compliance *and* rider count in one), (b)
a **geometry + signal rule engine** for red-light/wrong-side/stop-line/parking (annotated
once per fixed camera — interpretable, not black-box), (c) an **ANPR module** (plate
detection + PaddleOCR), and (d) a **VLM verification & evidence-captioning layer** that
checks low-confidence cases and writes human-readable evidence. Everything lands in a
structured **evidence store** powering a **search + analytics dashboard**.

**Three novelty hooks** (what makes us stand out):
1. **Confidence cascade + human-in-the-loop** — cheap models first; a VLM verifies only
   uncertain detections → high precision without per-frame VLM cost.
2. **Geometry-as-config** — fixed-camera zones/lines/lane-directions make hard temporal
   violations feasible, interpretable, and auditable (court-admissible evidence trail).
3. **Indian-context grounding** — fine-tune on IDD / DriveIndia / Indian plates; honest
   handling of dense two-wheeler, mixed traffic, varied plate fonts.

---

## 2. Design Principles
- **Two-stage detect→crop→classify** for small/distant riders (all AICC winners converge here).
- **Geometry + time over pure ML** for temporal violations (cameras are fixed → annotate once).
- **Cascade by cost & confidence**: detector → rule engine → VLM only when uncertain.
- **Honest scope**: separate "demonstrated on benchmark" from "designed / needs video".
- **License-clean**: build on Apache/MIT models; re-implement ideas from GPL/unlicensed repos.
- **Evidence-first**: every flag yields an annotated image + structured, auditable record.

---

## 3. End-to-End Architecture
```
                         ┌──────────────────────────────────────────────┐
   Traffic image/        │  0. INGEST  (image or sampled video frame)     │
   sampled frame  ─────► │     + image-quality assessment (pyiqa)         │
                         └───────────────┬──────────────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────────┐
        │ 1. PREPROCESSING (conditional, quality-gated)               │
        │    low-light→Retinexformer | weather→OneRestore | else skip │
        └───────────────┬────────────────────────────────────────────┘
                        ▼
        ┌────────────────────────────────────────────────────────────┐
        │ 2. DETECTION  (RF-DETR / YOLOv12)                           │
        │    vehicles, riders, pedestrians, plates, traffic lights    │
        └──────┬───────────────────────────────┬─────────────────────┘
               ▼                                ▼
   ┌───────────────────────┐      ┌──────────────────────────────────┐
   │ 3. TRACKER            │      │ 4a. HELMET/RIDER MODULE           │
   │  BoT-SORT / ByteTrack │      │  detect motorbike → crop →        │
   │  persistent IDs       │      │  7-class (D/P1/P2 ×helmet)        │
   └─────────┬─────────────┘      │  → assoc head→rider→bike          │
             │                    │  → helmet-violation + triple-ride │
             ▼                    └──────────────┬───────────────────┘
   ┌───────────────────────────┐                 │
   │ 4b. GEOMETRY RULE ENGINE  │                 │
   │  per-camera config:       │                 │
   │   stop-line, no-park,     │                 │
   │   lane-dir, signal ROI    │                 │
   │  → red-light / wrong-side │                 │
   │    stop-line / parking    │                 │
   └─────────┬─────────────────┘                 │
             │            ┌────────────────────────────────────┐
             │            │ 4c. ANPR  plate det → PaddleOCR     │
             │            └──────────────┬─────────────────────┘
             ▼                           ▼
        ┌──────────────────────────────────────────────────────────┐
        │ 5. VIOLATION CLASSIFIER + CONFIDENCE CASCADE              │
        │   per-class thresholds → calibrate → if low-conf:        │
        │   6. VLM VERIFIER (Qwen2.5-VL/InternVL3) confirm+caption  │
        └───────────────┬──────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │ 7. EVIDENCE GENERATOR  annotated image + JSON record      │
        │    + timestamp + plate + confidence + audit trail         │
        └───────────────┬──────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │ 8. STORE (DB)  →  9. ANALYTICS & SEARCH DASHBOARD         │
        │    stats, trends, hotspots, searchable records, reports   │
        └──────────────────────────────────────────────────────────┘
```

---

## 4. Component-by-Component Spec (model picks + datasets)

### 4.0 Ingest + Image-Quality Gate
- Lightweight IQA (pyiqa/piqa) scores blur/exposure → decides whether preprocessing runs.
- Avoids wasting compute and prevents restoration artifacts on already-good images.

### 4.1 Preprocessing (conditional)
| Need | Primary | Backup | Dataset |
|------|---------|--------|---------|
| Low-light | **Retinexformer** (ICCV'23, NTIRE winner*) | RetinexMamba/MambaLLIE | LOL-v1/v2, ExDark |
| Composite weather (rain/haze/blur) | **OneRestore** (ECCV'24) | task-specific (Rain100/SOTS/GoPro) | Rain100, SOTS, GoPro |
- **Ablation planned:** detector mAP with vs without restoration on **BDD100K** night/rain.
  *(If restoration doesn't help, drop it and train a robust detector — report either way.)*

### 4.2 Detection (core)
| Model | License | Why |
|-------|---------|-----|
| **RF-DETR** (2025) | Apache-2.0 | best occlusion/domain-shift; precision-critical |
| **YOLOv12** (2025) | check repo | fast (40.5% mAP @1.62ms T4); strong alt |
| YOLO-World / GroundingDINO | — | open-vocab cold-start for rare classes |
- **Datasets:** COCO, **BDD100K**, **IDD-Detection** (40k), **DriveIndia** (66,986 imgs, 24
  classes, YOLO format, baseline mAP50 **78.7%**), UA-DETRAC, KITTI.
- ⚠️ Ultralytics YOLO = **AGPL-3.0** (matters if distributing).

### 4.3 Tracking
- **BoT-SORT** (camera-motion compensation, suits CCTV) → **ByteTrack** backup. Both ship in
  Ultralytics. Enables wrong-side/red-light/parking (persistent IDs across frames).
- **Datasets:** UA-DETRAC (140k frames, 1.21M boxes), CityFlow (multi-cam ReID), BDD100K-MOT.

### 4.4a Helmet + Triple Riding (the anchor — adopt AICC scheme)
- **7-class scheme** (one model, two violations):
  `motorbike, DHelmet, DNoHelmet, P1Helmet, P1NoHelmet, P2Helmet, P2NoHelmet`
  - Helmet non-compliance = any `*NoHelmet`; Triple riding = D+P1+P2 on one bike.
- **Pipeline (re-implemented, license-clean):** detect motorbike @high-res → crop →
  classify 7 states → **associate** head→rider→bike (overlap + nearest-x-center heuristic,
  re-written from VNPT idea) → emit violations.
- **Per-class confidence defaults** (from VNPT): motorbike 0.35, D/P1 0.32, **P2 0.20**.
- **Dataset:** **AI City Challenge Track 5** (100 videos, 20s, 10fps, 1080p, track_ids).
- **Reality:** winners sit at **mAP ~0.49–0.70** (year/test-set dependent) — not 99%.

### 4.4b Geometry / Temporal Violations (rule engine)
Per-camera **scene config** (annotated once): stop-line polygon, no-parking polygon, lane
direction vectors, signal-light ROI. Rule engine over tracks:
| Violation | Rule | Dataset / source |
|-----------|------|------------------|
| Red-light | track crosses stop-line zone while signal=red | LISA (43k, 7 states), BSTLD, DriveU, DualCam |
| Wrong-side | trajectory vector opposes lane direction | UA-DETRAC/CityFlow + per-ROI direction (no benchmark) |
| Stop-line | bbox crosses line when not permitted | geometry; IDD-Seg lane masks; annotate per camera |
| Illegal parking | dwell-time in no-park polygon > threshold | i-LIDS + small sets |
- **Honest framing:** these need a short clip or signal state — demoed on sampled video frames.

### 4.4c License Plate Recognition (ANPR)
- Plate detect (YOLO/RF-DETR) → **PaddleOCR PP-OCRv5** (Apache, +13pp over v4, handles
  skew/rotation; 37-language support). Backup OCR: EasyOCR.
- **Datasets:** **CCPD** (250k, pretrain) → fine-tune on **Indian plates in the wild**
  (16,192 imgs / 21,683 plates, 10 states). UFPR-ALPR, AOLP.

### 4.4d Seatbelt (best-effort, daytime)
- Two-stage: YOLOv11 windshield → driver crop → CNN/CNN-SVM belt classifier (~122 FPS).
- **Datasets:** ~12k windshield + ~10k belt; **AICC Track 3** (594 clips, 90 hrs, 99 drivers).
- **Scope honestly:** night/tint/glare/rear-occupant are failure modes.

### 4.5 Violation Classification + Confidence Cascade
- Map detector/rule outputs → predefined classes with **calibrated** confidence
  (temperature scaling + per-class thresholds). Low-confidence → route to VLM (4.6) and/or
  human review queue. Directly satisfies the "assign confidence scores" requirement.

### 4.6 VLM Verification & Evidence Captioning (novelty layer)
- **Qwen2.5-VL** / **InternVL3** (open-source). Used selectively (cascade), NOT per frame
  (7B ≈ 8.5s & ~18GB / 10 frames — too heavy for primary).
- Roles: (a) verify uncertain violations, (b) generate NL evidence captions, (c) zero-shot
  fallback for rare violations. Precedent: **AICC 2025 Track 2** (Traffic Video QA).

---

## 5. Evidence Generation — Schema (was under-attended → now specified)
Each confirmed violation produces an **annotated image** + this record:
```json
{
  "violation_id": "uuid",
  "timestamp": "2026-06-17T14:32:07+05:30",
  "camera_id": "CAM_MG_ROAD_07",
  "frame_ref": "path/or/hash",
  "violations": [
    {"type": "no_helmet", "role": "driver", "confidence": 0.91,
     "bbox": [x,y,w,h], "verified_by": "vlm"}
  ],
  "vehicle": {
    "type": "motorcycle", "track_id": 142,
    "plate": {"text": "MH12AB1234", "confidence": 0.84, "ocr": "PP-OCRv5"}
  },
  "evidence_image": "annotated/uuid.jpg",
  "vlm_caption": "Motorcycle driver without helmet crossing MG Road at 14:32.",
  "audit": {"model_versions": {...}, "scene_config": "CAM_07_v3", "review_status": "pending"}
}
```
- **Audit trail** (model versions, scene config, review status) → court-admissible, tamper-evident.
- **Privacy:** plate/owner linkage gated by access control; configurable retention; hash PII.

## 6. Analytics & Reporting (was under-attended → now specified)
- **Store:** SQLite/Postgres for records + object storage for evidence images.
- **Dashboard (Streamlit/FastAPI+React):** violation counts by type/time/camera; hotspot
  map; trend charts; **searchable records** (by plate, type, date, camera); exportable
  summary reports (PDF/CSV). Repeat-offender view (optional, via CityFlow-style ReID).

---

## 7. Tech Stack
- **Models:** RF-DETR/YOLOv12, BoT-SORT/ByteTrack, PaddleOCR PP-OCRv5, Retinexformer/OneRestore, Qwen2.5-VL.
- **Frameworks:** PyTorch, Ultralytics (note AGPL), OpenCV, Hugging Face, pyiqa, filterpy.
- **Serving/UI:** FastAPI + Streamlit dashboard; SQLite/Postgres; object storage.
- **Deployment:** edge (lightweight detector + rules) → cloud (VLM verification + analytics).
  Batch image processing; throughput estimate to be measured in eval.

## 8. Evaluation Protocol (lock before building)
| Component | Metric | Benchmark | Target/Reference (VERIFY) |
|-----------|--------|-----------|---------------------------|
| Detection | mAP@.5, mAP@.5:.95 | BDD100K / DriveIndia | DriveIndia baseline mAP50 78.7% |
| Helmet/triple | mAP, P/R/F1 | AICC Track 5 | ~0.49–0.70 mAP band |
| Plate OCR | full-plate acc, CER | CCPD / Indian set | report vs CCPD SOTA |
| Tracking | MOTA, IDF1 | UA-DETRAC | published baselines |
| Restoration (ablation) | downstream det mAP Δ | BDD night/rain | with vs without |
| Signal state | accuracy/mAP | LISA | published baselines |
| Efficiency | FPS, latency, throughput, memory | all | single mid-range GPU |
- Always report a number **with its dataset + year**; expect real-world helmet mAP ~0.5–0.7.

## 9. Implementation Roadmap (time-boxed)
| Phase | Output |
|-------|--------|
| M0 | Framing + this architecture (DONE) |
| M1 | Research + dataset map (DONE — see other docs) |
| M2 | Prototype: detect→track→helmet+triple→plate OCR→**evidence JSON**→annotated image |
| M3 | Geometry rule engine (1–2 violations) + scene-config annotator |
| M4 | Confidence cascade + VLM verification + captioning |
| M5 | Analytics dashboard + searchable records |
| M6 | Benchmark eval table + ablations + submission packaging |
**MVP demo:** helmet + triple-riding end-to-end with plate OCR, annotated evidence, JSON
record, on AICC/benchmark samples. **Stubbed/designed:** seatbelt, geometry violations,
analytics — interfaces visible.

## 10. Risk Register
| Risk | Mitigation |
|------|------------|
| No public dataset (stop-line, wrong-side, seatbelt) | per-camera ROI annotation; proxies; honest scoping |
| Geometry violations infeasible from stills | demo on sampled video frames; "needs clip" framing |
| Overclaiming accuracy | report metric+dataset+year; expect 0.5–0.7 helmet mAP |
| License contamination (GPL/unlicensed repos) | re-implement ideas; build on Apache/MIT |
| VLM cost/latency | cascade — VLM only on low-confidence cases |
| Domain shift (night/rain/India) | fine-tune on IDD/DriveIndia; restoration ablation |
| Privacy/legal (plate→owner) | access control, retention policy, PII hashing, audit trail |

## 11. What sets us apart (vs a plain helmet-detector submission)
- Covers **all 7 violation types** + ANPR + evidence + analytics (AICC repos do helmet only).
- **Geometry-as-config** rule engine → interpretable, scalable, court-admissible.
- **Confidence cascade + VLM** → precision + explainability without per-frame VLM cost.
- **Indian-context datasets** → real-world credibility.
- **Honest engineering** → explicit about what needs video and what's daytime-only.

## 12. Reference Index (companion docs)
- `RESEARCH_FINDINGS.md` — full per-component SOTA survey + sources.
- `AICITY_INDIAN_DATASETS.md` — AICC Track 5 deep dive + Indian datasets.
- `REPOS_REVIEW.md` — winning-repo code review + license notes + reusable IP.
- `GEOMETRY_VIOLATIONS.md` — temporal/geometry violations + dataset gaps + full gap audit.
- `PLAN.md` / `RESEARCH_PLAN.md` — project & research plans. `PROMPT.md` — Sonnet driver prompt.

> **Standing VERIFY list:** Co-DETR mAP 0.4860; Retinexformer NTIRE wins; DriveIndia 78.7%;
> seatbelt ~99%; all licenses (RF-DETR Apache, Ultralytics AGPL, YOLOv12 repo). Confirm at
> primary source before quoting in the final submission.
