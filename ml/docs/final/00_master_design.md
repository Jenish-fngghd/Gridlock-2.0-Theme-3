# Gridlock 2.0 — R2 · Master Solution Design (Canonical)
## Automated Photo Identification & Classification of Traffic Violations using Computer Vision
### Merged design: paradigm-partitioned reasoning (novelty) on an AICC-anchored, geometry-aware, build-ready spine

> **This is the canonical submission document.** It supersedes `../plan_a/09_plan_a_solution_design_SUPERSEDED.md` (Plan A) and consolidates `../plan_b/` (Plan B). Design *defenses* live in `01_justifications.md` (J#); the SOTA registry in `03_sota_registry.md`; the plan comparison + fine-tuning/GPU plan in `02_comparison_merge_finetuning.md`.
> **VERIFY discipline:** every quoted metric is tagged where it needs primary-source confirmation before being shown to judges.

---

## 0. Executive Summary

We propose a **modular, paradigm-partitioned, geometry-aware, confidence-cascaded** system that ingests **photographic evidence at scale** (image-first, cloud-native; edge optional) and outputs **annotated, auditable, court-admissible violation records** for all seven mandated violations + ANPR + analytics.

The organizing insight is that the seven violations fall into **three reasoning paradigms**, each handled by a purpose-built module on a shared perception backbone:

| Paradigm | Violations | Core mechanism |
|---|---|---|
| **A · Instance-attribute** | Helmet, Seatbelt | fine-grained part-level classification on an *attributed* person |
| **B · Multi-instance counting** | Triple riding | count riders co-located on one two-wheeler |
| **C · Scene-context** | Wrong-side, Stop-line, Red-light, Illegal parking | geometry + signal-state + motion, grounded by a per-camera Scene Context Model |

**The anchor:** helmet **and** triple-riding are solved together by the **NVIDIA AI City Challenge Track 5 seven-class scheme** (one model encodes per-rider helmet compliance *and* rider count), upgraded with **SAM2 + pose association** for robust rider↔bike attribution. Scene-context violations use a **geometry-as-config rule engine** (annotate a fixed camera once) — interpretable and auditable — strengthened by **single-frame monocular-3D** where only a still exists. A **confidence cascade** routes only low-confidence cases to a **VLM verifier/captioner**, and a **foundation-model data engine** auto-labels Indian footage to cut fine-tuning cost.

**Novelty hooks (what sets us apart):**
1. **Paradigm-partitioned reasoning** — the architecture is organized around the *nature of evidence* each violation needs, not one monolithic head.
2. **Geometry-as-config + Scene Context Model** — hard temporal violations become feasible, interpretable, court-admissible (annotate once per fixed camera).
3. **Confidence cascade + VLM-in-the-loop** — cheap models first; a VLM verifies only uncertain cases and writes human-readable evidence → precision without per-frame VLM cost.
4. **Foundation-model data engine** — open-vocab teachers auto-label rare Indian classes and distill into the fast detector.
5. **Indian-context grounding + admissibility** — DriveIndia/IDD/AICC/Indian-plate fine-tuning; calibrated confidence, abstain band, tamper-evident signed evidence.

---

## 1. Problem Decomposition

### 1.1 The three paradigms (design spine)
*(see table in §0)* — **Design consequence:** A needs *pose-guided ROI + fine-grained classifier*; B needs *rider↔vehicle association + counting*; C needs a *scene prior + signal state + motion/3D*. One monolithic head cannot serve all three → **hybrid: shared backbone, specialized modules** (J-note: monolithic vs modular trade-off).

### 1.2 Task → sub-problem map (8 tasks)
1. **Preprocessing** — low-light/rain/shadow/blur; *quality-gated* selective restoration; restore the **LP crop** specifically; push robustness into training.
2. **Detection** — heterogeneous Indian classes (car, truck, bus, **auto-rickshaw**, two-wheeler, cycle, cart, pedestrian, animal); small/occluded/dense; domain shift.
3. **Violation detection** — per the 3 paradigms.
4. **Classification + confidence** — calibrated scores, per-class thresholds, **abstain/human-review band**, multi-evidence corroboration.
5. **LPR** — small/angled/blurred plate → rectify → OCR Indian formats (old + **HSRP**) → syntax-validate; **gate OCR to violators**.
6. **Evidence** — annotated image + structured metadata + timestamp + **tamper-evidence/audit trail**.
7. **Analytics** — stats, trends, hotspots, **searchable records**, reports, repeat-offender.
8. **Evaluation** — Accuracy/Precision/Recall/F1/mAP per module + attribution-correctness + calibration + efficiency/scalability.

---

## 2. End-to-End Merged Architecture

```
                          ┌────────────────────────────────────────────────────┐
   Traffic image /        │  0. INGEST  (image or sampled video frame)          │
   photographic    ─────► │     + image-quality assessment (pyiqa: blur/glare/  │
   evidence              │       exposure)   [image-first · batch + API · edge optional] │
                          └───────────────────────┬────────────────────────────┘
                                                  ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 1. PREPROCESSING  (conditional, quality-gated — only if IQA flags it)   │
        │    low-light → Retinexformer  |  rain/haze/blur → OneRestore  |  else skip │
        └───────────────────────────────┬────────────────────────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 2. DETECTION  (RF-DETR family, Apache-2.0 — one license-clean spine)    │
        │    Stage-1 fast SCREEN  (RF-DETR-N/S) ──► Stage-2 CONFIRM (RF-DETR-L /  │
        │    Co-DETR) on candidate crops                                          │
        │    detects: vehicles · riders · pedestrians · plates · traffic-light heads │
        └──────┬───────────────────────────────────────────────┬─────────────────┘
               ▼                                                ▼
   ┌───────────────────────────┐               ┌──────────────────────────────────────┐
   │ 3. TRACKING (multi-frame) │               │ 4a. HELMET + TRIPLE-RIDING  (anchor)  │
   │    BoostTrack (via boxmot)│               │  AICC 7-class · detect motorbike@hi-res│
   │    ▶ BoT-SORT/ByteTrack   │──persistent──►│  → crop → classify D/P1/P2 ×Helmet     │
   │      fallback             │   track IDs   │  → ASSOCIATE head→rider→bike (SAM2 +   │
   │  (needed by 4b: motion)   │               │     RTMPose ▶ overlap/nearest-x fallbk)│
   └─────────┬─────────────────┘               │  ⇒ helmet non-compliance + triple-ride │
             ▼                                  └──────────────────┬─────────────────────┘
   ┌────────────────────────────────────┐                         │
   │ 4b. GEOMETRY RULE ENGINE           │                         │
   │     Scene Context Model (per-camera,│                        │
   │     annotated ONCE):                │                        │
   │       stop-line · no-park polygon · │                        │
   │       lane-direction · signal ROI   │                        │
   │   + signal-state classifier         │                        │
   │   + single-frame monocular-3D       │                        │
   │     (LocateAnything3D) for a still  │                        │
   │   ⇒ red-light · wrong-side ·        │                        │
   │      stop-line · illegal parking    │                        │
   │   (4a-seatbelt: windshield→driver→belt)                      │
   └─────────┬──────────────────────────┘                        │
             │                                                    │
             ▼                                                    ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 5. VIOLATION CLASSIFIER + CONFIDENCE CASCADE                            │
        │    per-class thresholds → temperature-scaling calibration → 3 bands:    │
        │    AUTO-CONFIRM (challan)  |  HUMAN-REVIEW (abstain)  |  DISCARD         │
        │    if low-confidence ──────────────────────────────────┐               │
        └───────────────┬───────────────────────────────────────┼────────────────┘
                        │                                        ▼
                        │            ┌───────────────────────────────────────────┐
                        │            │ 6. VLM VERIFY + CAPTION  (low-conf ONLY)   │
                        │            │    Qwen2.5-VL / InternVL3 / LocateAnything-3B│
                        │            │    confirms violation + writes NL evidence │
                        │            └───────────────────────────┬───────────────┘
                        ▼                                        │
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 4c. ROI-GATED ANPR  (runs ONLY on confirmed violators — not every car) │
        │    plate detect (RF-DETR) → rectify/SR-restore crop → PaddleOCR PP-OCRv5│
        │    → Indian-format (HSRP) syntax validator → plate + confidence         │
        └───────────────────────────────┬────────────────────────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 7. EVIDENCE GENERATOR                                                   │
        │    annotated image + structured JSON (timestamp · camera · bbox ·       │
        │    evidence-chain · plate · VLM caption) + SHA-256 + signature + audit  │
        └───────────────────────────────┬────────────────────────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │ 8. STORE (DB + object store)  →  9. ANALYTICS & SEARCH DASHBOARD        │
        │    counts by type/time/camera · hotspot map · trends · searchable       │
        │    records · PDF/CSV reports · repeat-offender view                     │
        └────────────────────────────────────────────────────────────────────────┘

  ╔════════════════════════════════════════════════════════════════════════════╗
  ║ OFFLINE DATA-ENGINE (runs separately, feeds Stage-1):                        ║
  ║   GroundingDINO / Grounded-SAM-2 / LocateAnything-3B auto-label              ║
  ║   IDD / DriveIndia / local footage → human spot-check → DISTILL into the     ║
  ║   fast RF-DETR detector. Cuts manual Indian-class labeling sharply.          ║
  ╚════════════════════════════════════════════════════════════════════════════╝
```

**How to read the flow (step by step):**
1. **Ingest & gate (0):** every image (or sampled video frame) is scored for quality; a clean image skips restoration entirely.
2. **Restore only if needed (1):** Retinexformer for night/low-light, OneRestore for rain/haze/blur — never run blindly (avoids artifacts + wasted compute).
3. **Detect once, confirm twice (2):** a fast RF-DETR screen proposes everything; a heavier RF-DETR-L / Co-DETR pass re-checks only the candidate crops → speed *and* precision from one Apache-licensed family.
4. **Track for motion (3):** BoostTrack assigns persistent IDs — the prerequisite for every *temporal* judgement (crossing a line, opposing lane direction, dwell-time).
5. **Branch by paradigm (4a/4b):** the **same detections** feed two parallel reasoners — the **7-class helmet/rider module** (helmet + triple-riding in one shot) and the **geometry rule engine** (red-light / wrong-side / stop-line / parking, plus seatbelt). Neither blocks the other.
6. **Score, then escalate (5→6):** outputs are calibrated into auto-confirm / human-review / discard bands; **only** low-confidence cases pay for a VLM, which both verifies and writes a human-readable caption.
7. **Read the plate last (4c):** ANPR is **gated to confirmed violators** — we never OCR every vehicle, only the ones being challaned.
8. **Emit auditable evidence (7→9):** each confirmed violation becomes a signed, tamper-evident record + annotated image, stored and surfaced in the analytics/search dashboard.

**Architecture choice — hybrid (shared backbone + paradigm-specialized cascade), server-side two-stage compute cascade for scalability.** Rationale and the "why not one monolithic head / why not run a VLM on every image" defenses → `01_justifications.md` (J1).

---

## 3. Component Specifications (model · dataset · license · why)

### 3.0 Ingest + image-quality gate
- Lightweight IQA (**pyiqa/piqa**) scores blur/exposure → decides whether restoration runs (avoids artifacts + wasted compute on good images).

### 3.1 Preprocessing (conditional)
| Need | Primary | Backup | Dataset |
|---|---|---|---|
| Low-light | **Retinexformer** (ICCV'23; **NTIRE 2024 runner-up**, 2025 winner reported ✓) | RetinexMamba / Zero-DCE++ | LOL-v1/v2, ExDark |
| Composite weather | **OneRestore** (ECCV'24) | task-specific | Rain100, SOTS, GoPro |
- **Planned ablation:** detector mAP **with vs without** restoration on BDD100K night/rain — *if it doesn't help, drop it and rely on a degradation-augmented detector.* Heavy restoration + super-resolution is reserved for the **LP crop** (high payoff, small ROI).

### 3.2 Detection (core) — Stage-1 screen → Stage-2 confirm
| Role | Model | License | Why |
|---|---|---|---|
| Stage-1 fast screen | **RF-DETR-N/S** (Roboflow, 2025) | **Apache-2.0** ✓ | real-time, **DINOv2** backbone, fine-tunes well on small/occluded data |
| Stage-2 confirm | **RF-DETR-L** / **Co-DETR** (ICCV'23) | Apache / open | RF-DETR-L = first real-time **60+ mAP COCO** ✓; Co-DETR = **AICC helmet rank-1 backbone** ✓ |
| Benchmark-only alt | **YOLOv12** (NeurIPS'25) | ⚠️ **AGPL-3.0** ✓ | 40.6% mAP-N @1.64 ms T4 ✓; strong but copyleft → internal benchmarking, not shipped |
| Cold-start / auto-label | YOLO-World / GroundingDINO | — | open-vocab, label-free rare classes |
- **RF-DETR family (Nano→2XL, Apache-2.0, ICLR'26 ✓)** is our **license-clean spine for *both* stages** → no AGPL exposure in shipped code. **YOLOv12 and Ultralytics YOLO are AGPL-3.0** → use only for internal benchmarking/ablation, never in the distributed submission. (Defense → `01_justifications.md` J6.)
- **Datasets:** COCO/Objects365 (pretrain) → **DriveIndia** (66,986 imgs, 24 cls; YOLOv8 baseline **mAP50 78.7%** ✓) + **IDD-Detection** (40k) + BDD100K + UA-DETRAC + **FishEye8K** (if fisheye/wide-angle cams).

### 3.3 Tracking
- **BoostTrack/BoostTrack++** (2024) — HOTA 69.25 > ByteTrack 67.68 on MOT17 (VERIFY); via **boxmot** (pluggable) → **BoT-SORT/ByteTrack** fallback. Enables wrong-side / red-light / parking (persistent IDs).
- **Datasets:** UA-DETRAC (140k frames), CityFlow (multi-cam ReID), BDD100K-MOT.

### 3.4 Paradigm A+B — Helmet + Triple Riding (the anchor)
- **AICC Track 5 seven-class scheme** (one model, two violations):
  `motorbike, DHelmet, DNoHelmet, P1Helmet, P1NoHelmet, P2Helmet, P2NoHelmet`
  - Helmet non-compliance = any `*NoHelmet`; **Triple riding = D+P1+P2 on one motorbike**.
- **Two-stage pipeline** (proven by all AICC winners): detect motorbike @high-res (1280–1536px) → crop → classify 7 states → **associate head→rider→bike**.
- **Association — upgraded:** primary = **SAM2 masks + RTMPose keypoints** (SAC-style cross-association, learned/robust); fallback = re-implemented **overlap (0.3) + nearest-x-center (0.6)** heuristic with 5% combined-box expansion (VNPT idea, re-implemented license-clean).
- **Per-class thresholds (start values):** motorbike 0.35, D/P1 0.32, **P2 0.20** (rare class).
- **Class-imbalance handling:** minority oversampling / focal loss / copy-paste / **synthetic data** (Stable Diffusion for rare night/no-helmet).
- **Dataset:** AICC Track 5 (100 videos, 20s, 10fps, 1080p, track_ids → frames as stills + free temporal labels).
- **Realistic target:** winners sit at **mAP ~0.49–0.70** (year/test-set dependent). 2024 rank-1 = **Co-DETR mAP 0.4860** (Vo et al., *Robust Motorcycle Helmet Detection… Using Co-DETR*, CVPRW'24 ✓); 2023 best ~0.69. *Do not promise 99%.*

### 3.5 Paradigm A — Seatbelt (best-effort, daytime)
- Two-stage: **YOLOv11/windshield detector → driver crop → CNN/CNN-SVM belt classifier** (~122 FPS reported — VERIFY); glare on windshield ROI handled by the LP-style crop restoration.
- **Datasets:** ~12k windshield + ~10k belt; **AICC Track 3** (594 clips, 90 hrs, 99 drivers).
- **Honest scope:** night/tint/glare/rear-occupant are failure modes.

### 3.6 Paradigm C — Geometry / Temporal Violations (rule engine + SCM)
**Scene Context Model (per-camera config, annotated once):** stop-line polygon, no-parking polygon, lane-direction vectors, signal-light ROI, ground-plane homography. Semi-automatic: IDD-Seg lane/road masks propose geometry → human verifies once → stored/reused.

| Violation | Rule | Signal/3D | Dataset |
|---|---|---|---|
| **Red-light** | track crosses stop-line zone while signal=red | signal-state cls (LISA/BSTLD) | LISA (43k, 7 states), BSTLD, DriveU, DualCam |
| **Wrong-side** | trajectory vector (or **monocular-3D yaw**) opposes lane direction | LocateAnything3D single-frame | UA-DETRAC/CityFlow + per-ROI direction |
| **Stop-line** | bbox/ground-projection crosses line when not permitted | homography / 3D box | geometry; IDD-Seg lanes; per-camera annotation |
| **Illegal parking** | dwell-time in no-park polygon > threshold | tracklet velocity≈0 | i-LIDS + small sets |
- **Single-frame strengthening:** **LocateAnything3D** (CVPR'26, Chain-of-Sight) gives monocular 3D yaw/ground-position → wrong-side & stop-line from a *single* still where no clip exists.
- **Honest framing:** red-light/wrong-side/parking-duration are temporally ambiguous from one still → **photo-first, burst/clip-capable** (enforcement cameras shoot 2–3 frame bursts); demo on sampled video frames.

### 3.7 ROI-gated ANPR
- Runs **only on flagged violators**. plate detect (RF-DETR) → rectify (homography/STN) → SR-restore crop → **PaddleOCR PP-OCRv5** (Apache; rotation/skew; **+13pp E2E over v4** ✓) → **Indian-format syntax validator** (state-code + HSRP regex) → plate confidence.
  - *Throughput caveat (✓):* PP-OCRv5 uses a larger dictionary and is **slower than v4** — since ANPR is ROI-gated (only violators), accuracy wins here; keep **PP-OCRv4** as the high-throughput option if needed.
- **Alternatives:** PARSeq (syntax-aware), **Relaxed-Syntax Transformer (ICDAR'25)** for future-proof HSRP, EdgeFormer-LPR (edge); EasyOCR backup.
- **Evidence enrichment / cross-check (new):** **self-reflective VLM** (arXiv 2508.01387) reads **plate + vehicle make/model** under in-motion/wild conditions (UFPR-ALPR **83% plate / 61% make-model** ✓; self-reflection +5.7% make-model). Use it (a) as an **independent OCR cross-check** on low-confidence plates, and (b) to add **make/model** to the evidence record for richer challans and **repeat-offender ReID**. *VLM-only plate accuracy (~83%) < a fine-tuned OCR, so OCR stays primary; the VLM augments, it does not replace.*
- **Datasets:** **CCPD** (250k, pretrain) → fine-tune **Indian plates in the wild** (16,192 imgs / 21,683 plates, 10 states) + **synthetic HSRP**. UFPR-ALPR, AOLP.

### 3.8 Violation classification + confidence cascade
- Map module outputs → predefined classes with **per-class thresholds + temperature-scaling calibration**. Three bands: **auto-confirm** (high-precision, challan-eligible) / **human-review** / **discard**. Multi-evidence corroboration (red-light needs signal=red AND crossing). Satisfies the explicit "assign confidence scores" requirement.

### 3.9 VLM verification & evidence captioning (selective)
- **Qwen2.5-VL / InternVL3 / LocateAnything-3B**, used **only on low-confidence cases** (cascade), never per-frame (7B ≈ 8.5s & ~18GB / 10 frames — too heavy as primary — VERIFY).
- Roles: (a) verify uncertain violations, (b) generate NL **evidence captions**, (c) zero-shot fallback for rare violations, (d) optional **agentic** rule-compliance reasoning. Precedent: AICC 2025 Track 2 (Traffic Video QA).

### 3.10 Offline foundation-model data engine
- **GroundingDINO / Grounded-SAM-2 / LocateAnything-3B / Florence-2** auto-annotate IDD/DriveIndia/local footage (open-vocab + referring prompts incl. auto-rickshaw, "rider without helmet", "vehicle past stop line") → human-spot-check → **distill into the fast Stage-1 detector**. Cuts manual labeling sharply.

---

## 4. Evidence Generation — Schema
Each confirmed violation → annotated image + record:
```json
{
  "violation_id": "uuid",
  "timestamp": "2026-06-17T14:32:07+05:30",
  "camera_id": "CAM_MG_ROAD_07",
  "frame_ref": "path/or/hash",
  "violations": [
    {"type": "no_helmet", "role": "driver", "confidence": 0.91,
     "bbox": [x,y,w,h], "evidence_chain": ["detect","associate","classify"],
     "verified_by": "vlm"}
  ],
  "vehicle": {"type": "motorcycle", "track_id": 142,
    "plate": {"text": "MH12AB1234", "confidence": 0.84, "ocr": "PP-OCRv5"}},
  "evidence_image": "annotated/uuid.jpg",
  "vlm_caption": "Motorcycle driver without helmet crossing MG Road at 14:32.",
  "audit": {"model_versions": {...}, "scene_config": "CAM_07_v3",
            "sha256": "...", "signature": "...", "review_status": "pending"}
}
```
- **Audit trail + SHA-256 + signature** → court-admissible, tamper-evident. **Privacy:** plate→owner linkage gated by access control; configurable retention; PII hashing.

## 5. Analytics & Reporting
- **Store:** SQLite/Postgres (records) + object storage (evidence images) + search index.
- **Dashboard (Streamlit/FastAPI):** counts by type/time/camera; hotspot map; trend charts; **searchable records** (plate/type/date/camera); exportable PDF/CSV reports; **repeat-offender** view (optional CityFlow ReID).

---

## 6. Datasets Map (no proprietary data needed)
| Component | Datasets | India-relevant | License note |
|---|---|---|---|
| Detection | COCO, **BDD100K**, **IDD-Det (40k)**, **DriveIndia (67k/24cls)**, UA-DETRAC, **FishEye8K** (if fisheye cams) | ✅ IDD/DriveIndia | research-use; verify DriveIndia/TiHAN |
| Restoration | LOL-v1/v2, ExDark, Rain100, SOTS, GoPro | — | research |
| Helmet+triple | **AICC Track 5** (100 vids) | partial | challenge/CC; register |
| Seatbelt | ~12k/10k sets, **AICC Track 3** | — | research |
| Plates | **CCPD (250k)**, **Indian-in-the-wild (16k/21k)**, UFPR-ALPR, AOLP | ✅ Indian set | research |
| Signal state | **LISA (43k)**, BSTLD, DriveU, DualCam | needs Indian FT | research |
| Tracking | UA-DETRAC, CityFlow, BDD100K-MOT | — | research |

---

## 7. Evaluation Protocol (lock before building)
| Component | Metric | Benchmark | Target/Reference (VERIFY) |
|---|---|---|---|
| Detection | mAP@.5, mAP@.5:.95, small-obj AP | DriveIndia/BDD/IDD | DriveIndia mAP50 ~78.7% |
| Helmet/triple | mAP, P/R/F1, **attribution acc** | AICC Track 5 | ~0.49–0.70 mAP band |
| Plate OCR | full-plate acc, CER, 1-NED | CCPD/Indian | report vs CCPD SOTA |
| Signal state | accuracy/mAP | LISA | published baselines |
| Tracking | MOTA, IDF1, HOTA | UA-DETRAC | BoostTrack baselines |
| Restoration (ablation) | downstream det mAP Δ | BDD night/rain | with vs without |
| Confidence | **ECE / reliability**, P@operating-point | all | calibrated |
| Efficiency/scale | **images/sec/GPU, cost/1k images**, latency, memory | all | single mid GPU |
- **Custom metrics:** attribution-correctness (rider↔bike↔plate), event-level vs frame-level, robustness slices (night/rain/glare via IDD-AW), cross-camera generalization, separate operating points for **auto-challan vs human-review**.
- **Always report number + dataset + year.** Expect real-world helmet mAP ~0.5–0.7.

---

## 8. Fine-Tuning & Compute (summary)
Fine-tuning is **required** (transfer learning, no training from scratch). Full plan, per-model GPU-time, cost, and method → `02_comparison_merge_finetuning.md` §6.
- **Must fine-tune:** core detector (DriveIndia+IDD), helmet 7-class (AICC), plate det+OCR (Indian/CCPD), signal-state (LISA), seatbelt (light).
- **As-is:** preprocessing, trackers, pose, SAM2, VLM (optional QLoRA only).
- **Budget:** competitive MVP **~6–8 A100-GPU-days (~$260–700 spot)**; full **~11–16 GPU-days (~$450–1,300)**; ~3–4 days (MVP) / 6–8 days (full) wall-clock on 2× A100. *(RF-DETR-primary is ~1–2 GPU-days heavier than the YOLO alt.)*
- **Priority:** detector → helmet 7-class → plate det+OCR → signal+geometry → seatbelt → (skip VLM LoRA).

---

## 9. Implementation Roadmap
| Phase | Output |
|---|---|
| M0–M1 | Framing + research + dataset map (DONE) |
| M2 | MVP: detect → BoostTrack → 7-class helmet+triple → ANPR → **signed evidence JSON + annotated image** (AICC/DriveIndia samples) |
| M3 | Geometry rule engine (1–2 violations) + Scene-Config annotator |
| M4 | Confidence cascade + VLM verify/caption |
| M5 | Analytics dashboard + searchable records |
| M6 | Benchmark eval table + ablations (restoration, association) + submission packaging |
**MVP demo:** helmet + triple-riding end-to-end with ANPR, annotated evidence, JSON record. **Stubbed/designed:** seatbelt, geometry violations, analytics — interfaces visible.

## 10. Risk Register
| Risk | Mitigation |
|---|---|
| No public dataset (stop-line, wrong-side, seatbelt) | per-camera ROI annotation; proxies; honest scoping |
| Geometry violations infeasible from stills | demo on sampled video frames; single-frame-3D; "needs clip" framing |
| Overclaiming accuracy | report metric+dataset+year; expect 0.5–0.7 helmet mAP |
| License contamination (GPL/AGPL/unlicensed) | re-implement ideas; build on Apache/MIT (RF-DETR, PaddleOCR) |
| VLM cost/latency | cascade — VLM only on low-confidence cases |
| Domain shift (night/rain/India) | fine-tune on IDD/DriveIndia; restoration ablation |
| Foundation-model dependency/license (LocateAnything) | use as teacher/verifier only; YOLO student is the deployable spine |
| Privacy/legal (plate→owner) | access control, retention, PII hashing, audit trail |

---

## 11. Novelty Statement
Beyond "a recent YOLO + OCR": (1) **paradigm-partitioned reasoning**; (2) **geometry-as-config Scene Context Model** making temporal violations feasible/auditable; (3) **AICC 7-class + learned SAC association** solving "whose helmet / how many on the bike"; (4) **confidence cascade + VLM-in-the-loop** for precision + explainability without per-frame VLM cost; (5) **foundation-model data engine** (teacher→distilled student) for Indian-data scarcity; (6) **single-frame monocular-3D** to disambiguate scene violations from a still; (7) **admissibility-oriented** calibrated, tamper-evident evidence; (8) **Indian-domain grounding** throughout. Defenses of each non-obvious choice → `01_justifications.md`.

## 12. Licensing Hygiene
Build shippable code on **Apache/MIT** (RF-DETR, PaddleOCR, boxmot). Treat **Ultralytics YOLO = AGPL-3.0** and any **GPL** repo as study-only → **re-implement** logic. Confirm AICC/dataset research-use terms and the NVIDIA LocateAnything license before any deployment claim.

## 13. Companion Documents
- `01_justifications.md` — J1… narrative design defenses.
- `06_model_selection_justification.md` (+ `.csv`) — **per-model/-module selection sheet** (what · where · alternatives · why).
- `03_sota_registry.md` (+ `.csv`) — SOTA review + Paper/Source Registry.
- `02_comparison_merge_finetuning.md` — Plan A vs B, merge rationale, **fine-tuning + GPU/cost plan**.
- `04_datasets_acquisition_and_prep.md` — dataset acquisition + prep.
- `05_one_page_note.md` — one-page architecture summary.
- `07_trackB_foundation_models_lightning.md` — LocateAnything-3B / SAM-3 zero-shot violation tests for the H200.
- `../plan_b/` — Plan B source research (AICC deep-dive, repo review, geometry/dataset audits).

## 14. Verification Log (primary-source checked this cycle)

**✓ Verified — safe to quote (with the cited source):**
| Claim | Verified value | Source |
|---|---|---|
| AICC'24 Track 5 rank-1 | **Co-DETR mAP 0.4860** (Minority Optimizer + Virtual Expander) | Vo et al., *Robust Motorcycle Helmet Detection… Using Co-DETR*, CVPRW 2024 |
| DriveIndia | **66,986 imgs, 24 classes, 471k instances; YOLOv8 mAP50 78.7%** | arXiv 2507.19912 (ITSC 2025) |
| RF-DETR | **Apache-2.0**; first real-time **60+ mAP COCO**; Nano→2XL; ICLR 2026 | Roboflow rf-detr repo / blog |
| YOLOv12 | **AGPL-3.0**; YOLOv12-N **40.6% mAP @1.64 ms T4**; X 55.2% | sunsmarterjie/yolov12 (NeurIPS 2025) |
| PP-OCRv5 | **+13pp E2E over v4** (but larger dict → slower) | PaddleOCR PP-OCRv5 docs / 3.0 report (arXiv 2507.05595) |
| Retinexformer | ICCV 2023; **NTIRE 2024 runner-up** (2025 winner reported) | NTIRE 2024 LLIE report (arXiv 2404.14248) |
| BoostTrack | **HOTA 69.25 vs ByteTrack 67.68** (MOT17) | BoostTrack++ (arXiv 2408.13003) |
| Self-reflective VLM ANPR | UFPR-ALPR **83.05% plate / 61.07% make-model**; +5.72% w/ reflection | arXiv 2508.01387 |
| LocateAnything-3B | ~12–25 boxes/s/H100; Parallel Box Decoding | research.nvidia.com / HF |

**⧗ Still to verify before final submission:**
AICC 2023 best ~0.69 (IC_SmartVision exact); seatbelt ~99% (controlled-condition caveat); Indian-plate-in-the-wild exact counts (16,192/21,683); IDD-Detection 40k exact; Retinexformer NTIRE 2025/2026 win specifics; **license/access terms** for AICC Track 5 data, DriveIndia (TiHAN), IDD, and **NVIDIA LocateAnything**.
