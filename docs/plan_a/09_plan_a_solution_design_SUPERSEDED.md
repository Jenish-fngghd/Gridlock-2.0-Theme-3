# Gridlock 2.0 — Round 2 Solution Design
## Automated Photo Identification & Classification of Traffic Violations using Computer Vision
### A research-backed, paradigm-partitioned pipeline for unstructured Indian traffic

> ⚠️ **SUPERSEDED — this is "Plan A" (novelty-first draft).** The canonical merged submission document is **`../final/00_master_design.md`**, which folds this plan together with the `../plan_b/` (Plan B) research (AICC scheme, geometry rule engine, dataset map, license hygiene, realistic eval targets). Kept here for reference/history.

---

## 0. Executive Summary

The seven mandated violations are **not architecturally uniform**. Forcing them through one detection head is the central mistake to avoid. We partition them into **three reasoning paradigms** and build one **shared perception backbone** that feeds **three specialized reasoners**, grounded by a **per-camera Scene Context Model (SCM)** and finalized by an **admissibility-oriented confidence-fusion + ROI-gated LPR** stage.

Our design adds a **Foundation-Model Layer** built on NVIDIA **LocateAnything-3B** (open-vocabulary grounding VLM, 2026) and **LocateAnything3D / Chain-of-Sight** (CVPR 2026) used in three disciplined roles — *data-engine (teacher), open-vocab verifier, and single-frame 3D grounding* — **without** making a slow 3B VLM the throughput workhorse. The system is **image-first and cloud-scalable**: a fast detector (**YOLOv11/12 + ByteTrack**) screens the bulk of incoming images, a heavier model confirms only flagged candidates, and the VLM acts as offline teacher + confirmation-time verifier. (On-camera **edge** deployment is an *optional* profile, not the spine — see §0.1.)

> Headline novelty: **paradigm-partitioned reasoning + a reusable per-camera scene prior + a "Chain-of-Evidence" reasoning structure (borrowed from Chain-of-Sight) + a foundation-model teacher→efficient-student data engine**, all oriented toward *legally admissible* evidence rather than raw mAP.

### 0.1 Deployment scope: image-first & cloud-scalable (edge is optional)

The brief asks for a *"scalable AI-based traffic **image** analysis system ... from **photographic evidence**."* We therefore treat this as a **server/cloud-native batch + API pipeline that ingests photographic evidence at scale**, not a live on-camera video system. "Scalable" here means **throughput on large image volumes** (queue-based ingestion → autoscaling GPU workers → batched inference → ROI-gating → object store/DB), measured in **images/sec/GPU and cost per 1,000 images** — *not* edge-device FPS.

- **Two-stage compute cascade (server-side, for scalability):** a cheap detector screens *every* image; the expensive confirmation model + OCR run *only* on flagged candidates. This keeps cost-per-image low at volume — the same "filter→confirm" idea as before, but it is a **compute tier, not an edge tier**.
- **Photo-first, burst/clip-capable:** red-light *running*, wrong-side, and parking-*duration* are temporally ambiguous from a single still. Real enforcement cameras already capture **2–3 shot bursts** per trigger — still "photographic evidence," but enough temporal signal to decide these. Where only one image exists, the system uses single-frame 3D grounding (§3.3) and otherwise routes to human review rather than guessing.
- **Edge = optional profile.** The same models export to **Jetson/DeepStream** for sites that want on-camera pre-filtering to cut bandwidth — a deployment choice layered on top, not a requirement of the architecture.

---

## 1. Problem Decomposition

### 1.1 The three violation paradigms (the spine of the design)

| Paradigm | Violations | What it fundamentally requires | Failure mode if done naïvely |
|---|---|---|---|
| **A. Instance-level attribute** | Helmet non-compliance, Seatbelt non-compliance | Fine-grained, part-level reasoning on a *specific* detected human, correctly attributed to a *specific* vehicle | A "no-helmet" box with no idea which rider/bike it belongs to → unenforceable |
| **B. Multi-instance spatial counting** | Triple riding | Counting persons *co-located on one two-wheeler* under occlusion (pillion/child riders hidden) | Counting all nearby people, or missing occluded third rider |
| **C. Scene-context** | Wrong-side driving, Stop-line, Red-light, Illegal parking | Spatial/directional/zone/temporal reasoning about the *scene*, relative to static infrastructure (lane direction, stop line, signal head, no-parking polygon) | Single-frame guess with no scene geometry → ill-posed, high false-positive rate |

**Design consequence:** Paradigm A needs *pose-guided ROI + fine-grained classifiers*; Paradigm B needs *rider↔vehicle association + counting*; Paradigm C needs a *scene prior + temporal/3D state*. One monolithic head cannot serve all three → **hybrid (shared backbone, specialized heads)**.

### 1.2 Task-by-task technical sub-problems

**Task 1 — Image preprocessing.** Low light, rain, shadow, motion blur (two-wheelers move fast → plate blur). *Key insight:* naïve full-frame restoration often *hurts* detection and adds latency. Sub-problems: (a) no-reference quality assessment to decide *whether* to enhance; (b) selective enhancement; (c) **restoration spent on the LP crop specifically** (small ROI, high payoff); (d) train detectors with synthetic rain/fog/blur so robustness lives in the model, not only at test time.

**Task 2 — Vehicle & road-user detection.** Heterogeneous Indian classes (car, truck, bus, **auto-rickshaw**, two-wheeler, bicycle, cart, tractor, pedestrian, animal). Sub-problems: small/distant objects, dense occlusion, scale variance, non-lane discipline, class long-tail. COCO-trained detectors miss auto-rickshaws/carts → domain adaptation is mandatory.

**Task 3+4 — Violation detection & classification + confidence.** Per paradigm above. Plus: calibrated confidence, an *abstain/escalate band* for human review, and multi-evidence corroboration (a red-light call must combine signal=red AND crossing event).

**Task 5 — LPR.** Detect a small, angled, possibly blurred plate → rectify → OCR Indian formats (old black-on-white, new **HSRP**, state codes, bilingual/decorative fonts) → syntax-validate. *Bottleneck risk:* running OCR on every vehicle every frame. Sub-problem: **gate OCR to confirmed violators only**.

**Task 6 — Evidence generation.** Annotated composite (boxes, labels, confidence, zoomed insets for head/plate, signal state, timestamp, camera ID, geo) + structured metadata + **tamper-evidence (hash/signature) for chain-of-custody**.

**Task 7 — Analytics & reporting.** Aggregations, trends, hotspots, **search by plate/type/time/location**, repeat-offender lookup, per-camera health, e-challan export.

**Task 8 — Evaluation.** Map Accuracy/Precision/Recall/F1/mAP to each module; add attribution-correctness, calibration (ECE), event-level vs frame-level, robustness slices (night/rain/glare), cross-camera generalization, and compute/throughput/energy.

---

## 2. SOTA Research & Paper / Source Registry (2022–2026)

| Source / Paper | Venue & Year | Core Contribution | Module Borrowed | Why It Fits Our Problem |
|---|---|---|---|---|
| **IDD: India Driving Dataset** (Varma et al.) | WACV 2019 | 34-class unstructured Indian road dataset; fuzzy lanes, dense heterogeneous traffic | Backbone domain pretraining + seg priors | Cityscapes/BDD assume disciplined Western traffic; IDD captures Indian chaos |
| **IDD-AW** | WACV 2024 | 5k image pairs, pixel labels under rain/fog/low-light/snow | Robustness eval + preprocessing target | Directly stresses Task-1 adverse-condition robustness |
| **IDD-3D** | WACV 2023 | Multi-modal (cam+LiDAR) 3D Indian scenes | Monocular-3D fine-tune/eval reference | Grounds 3D scene-context reasoning in Indian data |
| **DriveIndia** (arXiv 2507.19912) | 2025 | 66,986 imgs, 24 classes, 471k instances, weather/lighting variety; YOLOv8 mAP@50 ≈ 78.7 | Detection fine-tuning + benchmark | Modern large Indian *detection* (not just seg) benchmark incl. auto-rickshaw |
| **DashCop** (arXiv 2503.00428) | 2025 | **Segmentation-and-Cross-Association (SAC)** rider↔motorcycle module; cross-association tracking; **RideSafe-400** dataset | Association substrate (Paradigm A & B) | Solves the exact "whose helmet / how many on this bike" attribution problem |
| **Helmet violation via NVIDIA TAO + YOLOv8** (Frontiers in AI) | 2025 | Indian smart-city helmet + plate; DetectNet_v2-ResNet18 triple-rider 91.42% | Helmet branch + TAO/DeepStream deployment path | Proven Indian-city deployment recipe on NVIDIA edge stack |
| **Optimized Mask R-CNN triple-rider** (Syst. Sci. & Control Eng.) | 2025 | Instance-seg approach to triple riding | Counting reasoner reference (Paradigm B) | Validates seg-based counting under occlusion |
| **YOLOv11** (Ultralytics) | 2024 | C3k2 blocks, spatial attention, ~22% fewer params vs v8m at higher mAP | Stage-1 fast screening detector (primary) | Best accuracy/throughput for screening large image volumes; edge-exportable |
| **YOLOv12** (arXiv) | Feb 2025 | Area-Attention (A²) + R-ELAN; attention-centric, SOTA mAP/latency | Real-time detector (alt/upgrade) | Attention gains for small/occluded objects at real-time speed |
| **RT-DETR** (Zhao et al.) | CVPR 2024 | Real-time end-to-end DETR; hybrid encoder; NMS-free | High-accuracy detector option | NMS-free helps dense, overlapping Indian traffic |
| **Co-DETR** (Zong et al.) | ICCV 2023 | Collaborative hybrid-assignment training; richer supervision | Cloud review-tier detector | Higher recall in cluttered scenes for evidence confirmation |
| **DINOv2** (Oquab et al.) | TMLR 2024 / arXiv 2023 | Self-supervised general visual features | Backbone initialization for label-efficient domain transfer | Strong features → adapts to Indian domain with less labeled data |
| **Segment Anything 2 (SAM 2)** (Meta) | 2024 | Promptable image/video segmentation + tracking | Masks for SAC association + scene-zone labeling | Generates instance masks to cross-associate riders↔bikes |
| **RTMPose** (Jiang et al., MMPose) | arXiv 2023 | Real-time multi-person 2D pose | Pose-guided ROI (helmet/seatbelt) + rider counting | Efficient in crowded scenes; localizes head/torso for attributes |
| **ViTPose** (Xu et al.) | NeurIPS 2022 | ViT-based SOTA pose under occlusion | High-accuracy pose alternative | Better under heavy occlusion for review tier |
| **PARSeq** (Bautista & Atienza) | ECCV 2022 | Permuted autoregressive STR; SOTA accuracy/latency; internal LM | LP OCR engine | Syntax-aware decoding suits structured Indian plate format |
| **PaddleOCR PP-OCRv4** (Baidu) | 2023 | DBNet detector + lightweight recognizer, deployable/multilingual | LP detect+OCR practical alt | CPU-friendly, production-proven fallback |
| **MMOCR** (OpenMMLab) | 2022– | DBNet/NRTR/ABINet toolkit | OCR experimentation/ablation | Swap recognizers to tune Indian-plate accuracy |
| **End-to-end ANPR for Indian datasets** (arXiv 2207.06657) | 2022 | Indian-specific ANPR network | Indian LP reference design | Targets Indian plate variability directly |
| **Zero-DCE++** (Guo et al.) | TPAMI 2021 | Zero-reference low-light enhancement, very fast | Selective low-light preprocessing | Unsupervised, real-time; no paired data needed |
| **ByteTrack** (Zhang et al.) | ECCV 2022 | Associates *every* detection box (incl. low-score) | MOT for temporal Paradigm-C violations | Robust tracklets in dense traffic for red-light/parking/wrong-way |
| **Red-light running via LSTM + rules** (Springer) | 2022 | Temporal modeling of RLR events | Scene-context temporal reasoner reference | Frames red-light as a temporal-event problem |
| **Wrong-way detection (YOLO + optical-flow / DeepSORT)** (MDPI; 2020/2024) | 2020–24 | Direction estimation for wrong-way | Wrong-side reasoner reference | Direction-vs-lane logic adapted to our SCM |
| **Seatbelt: windshield-YOLOv5s + belt model / 3D-pose + belt seg** (SIViP 2022; arXiv 2204.07946) | 2022 | Windshield→occupant→belt cascade; pose+seg | Seatbelt branch (Paradigm A) | Handles through-windshield glare/occlusion |
| **NVIDIA LocateAnything-3B** (NVIDIA Tech Report) | 2026 | Open-vocab grounding VLM (Moon-ViT+Qwen2.5); **Parallel Box Decoding**; detection+referring+point+OCR-localization; 12M img/138M queries | **Foundation-Model Layer**: data-engine teacher, cloud open-vocab verifier, plate-locate assist | Zero-shot Indian/long-tail classes + compositional scene queries; teacher for distillation |
| **NVIDIA LocateAnything3D — Chain-of-Sight** (arXiv 2511.20648) | CVPR 2026 | Open-vocab **monocular 3D** detection as next-token prediction; 2D→distance→size→pose curriculum; Omni3D +13.98 AP3D | **Chain-of-Evidence** reasoning structure + single-frame 3D heading/ground-position | Resolves single-frame ambiguity for wrong-side/stop-line; interpretable evidence chain |
| **DINO-X / Grounding DINO 1.5** (IDEA Research) | 2024–25 | Unified open-world detection; prompt-free; Grounding-100M | Auto-labeling teacher alternative | Bulk open-vocab annotation of Indian footage |
| **Grounded-SAM-2** (IDEA Research) | 2024 | Grounding-DINO + SAM2 detect-segment-track | Auto-label + association alt | Box→mask→track for weak supervision |
| **Florence-2** (Microsoft) | CVPR 2024 | Unified prompt-based vision (detect/caption/ground) | Teacher/aux annotation | Cheap multi-task pseudo-labels |

---

## 3. Proposed Architecture

### 3.1 Architecture choice: **Hybrid (shared backbone + paradigm-specialized cascade)**

- **Not monolithic multi-task:** the three paradigms consume different inputs (pose ROI vs association vs scene/temporal/3D) and different temporal scopes → a single head causes negative transfer.
- **Not fully modular (one model per violation):** wasteful (redundant backbones), slow, hard to maintain.
- **Hybrid wins:** one shared perception/foundation backbone → a small set of specialized reasoners → ROI-gated expensive modules (OCR, pose, seatbelt, VLM) run *only where needed*. Plus a **two-stage compute cascade** (server-side, for scalability): a fast detector screens every image → a heavier model confirms only flagged candidates.

> **Design-rationale note:** defenses of non-obvious choices (e.g. *"why not run LocateAnything-3B on every image instead of Stage-1 YOLO?"* → **J1**) live in the companion **`01_justifications.md`**, to keep this document focused on the architecture.

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ L0  Degradation-Aware Preprocessing (DAP)  ── quality gate → selective enhance │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L1  Shared Perception Backbone (server-side two-stage cascade)                 │
 │     Stage-1 fast screen: YOLOv11/12 + ByteTrack (batched; edge-exportable)     │
 │     Stage-2 confirm: Co-DETR / RT-DETR (DINOv2 init)  +  LocateAnything-3B VLM  │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L2  Association & Scene Grounding                                              │
 │     SAC rider↔vehicle (SAM2 masks) │ RTMPose keypoints │ per-camera SCM        │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L3  Paradigm-Specialized Reasoners                                            │
 │     A) Instance-Attribute (helmet, seatbelt)                                  │
 │     B) Multi-Instance Counting (triple riding)                                │
 │     C) Scene-Context (wrong-side, stop-line, red-light, parking) ← SCM + 3D   │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L4  Confidence Fusion + Abstain band  (+ optional VLM open-vocab verify)       │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L5  ROI-Gated LPR  (only on confirmed violators)                              │
 │     plate detect → rectify → SR/restore crop → PARSeq → Indian-format syntax  │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L6  Evidence Generation (annotated composite + signed metadata)               │
 ├──────────────────────────────────────────────────────────────────────────────┤
 │ L7  Analytics & Reporting (search, trends, hotspots, e-challan export)         │
 └──────────────────────────────────────────────────────────────────────────────┘
        ▲ Offline Data-Engine: LocateAnything-3B / DINO-X / Grounded-SAM-2
          auto-label IDD/DriveIndia/local footage → distill into Stage-1 YOLO
```

### 3.2 Module specifications (technique · source · why over alternatives)

**L0 — Degradation-Aware Preprocessing**
- *Technique:* No-reference quality score (BRISQUE/lightweight CNN) routes each image; **Zero-DCE++** for flagged low-light; light deblur; full-frame left largely untouched. Heavy restoration + super-resolution applied **only to the LP crop** in L5.
- *Source:* Zero-DCE++ (TPAMI 2021); IDD-AW (WACV 2024) for training augmentation realism.
- *Why over alternatives:* test-time full-frame deraining/deblurring frequently degrades detection and burns latency; we instead push robustness into training (synthetic rain/fog/blur) and reserve restoration for the high-payoff small ROI. Zero-DCE++ is zero-reference (no paired night data) and real-time.

**L1 — Shared Perception Backbone (server-side two-stage cascade)**
- *Stage-1 (fast screen):* **YOLOv11/YOLOv12** with a P2 small-object head + **ByteTrack**, run as batched GPU inference on autoscaling workers (and **TensorRT/DeepStream/Jetson**-exportable for optional edge pre-filtering).
- *Stage-2 (confirm):* **Co-DETR** (or **RT-DETR**) with a **DINOv2**-initialized backbone for high-recall confirmation on flagged candidates; **LocateAnything-3B** as an open-vocabulary grounding head.
- *Source:* YOLOv11 (Ultralytics 2024), YOLOv12 (2025), Co-DETR (ICCV 2023), RT-DETR (CVPR 2024), DINOv2 (TMLR 2024), DeepStream/TAO.
- *Why over alternatives:* one detector cannot be both cheap-at-volume and maximally accurate; the **screen → confirm** cascade gives throughput on large image volumes *and* the precision legal enforcement needs, spending the expensive model only where it matters. NMS-free DETR variants handle dense overlap; DINOv2 init reduces Indian-domain labeling burden.
- *Domain adaptation:* COCO pretrain → fine-tune on **IDD + DriveIndia + RideSafe-400** (adds auto-rickshaw, cart, two-wheeler-with-pillion). Heavy weather/blur augmentation from IDD-AW statistics.

**L2 — Association & Scene Grounding**
- *Rider↔vehicle association:* **SAC (Segmentation-and-Cross-Association)** from DashCop, using **SAM2** masks + geometric/appearance cross-association and cross-association tracking.
- *Pose:* **RTMPose** keypoints (head → helmet ROI; torso → seatbelt ROI; per-person centroids → rider count). ViTPose for the review tier under heavy occlusion.
- *Scene Context Model (SCM):* a **per-camera static prior** built once: stop-line polyline, **lane direction field**, no-parking polygons, signal-head ROI, drivable area + a ground-plane **homography**. Semi-automatic: IDD-trained segmentation + lane/road extraction proposes geometry; a human verifies once per camera; stored and reused.
- *Source:* DashCop (2025), SAM2 (2024), RTMPose (2023), homography calibration practice; IDD seg priors.
- *Why over alternatives:* attribution ("whose violation") is the difference between a demo and an enforceable system — plain detection ignores it. Decoupling **static scene geometry** (calibrated once) from **per-frame perception** converts ill-posed single-frame scene violations into well-posed grounded reasoning, and the SCM is reused across millions of frames at near-zero cost.

**L3 — Paradigm-Specialized Reasoners**
- **A. Instance-Attribute (helmet, seatbelt).** Pose-guided ROI → fine-grained binary classifier.
  - *Helmet:* RTMPose head keypoint → tight head crop on the *associated rider* → helmet/no-helmet CNN (EfficientNet/ConvNeXt-tiny). Avoids confusing pedestrians/bystanders.
  - *Seatbelt:* windshield localization → driver/front-passenger region → belt classifier/segmentation (handles glare via the L5 restoration trick on the windshield ROI).
  - *Source:* DashCop association; SIViP-2022 windshield cascade; arXiv 2204.07946 (pose+belt seg).
  - *Why:* fine-grained part-level decisions on a correctly attributed person beat a global "no-helmet" object class that can't be tied to a bike/plate.
- **B. Multi-Instance Counting (triple riding).** Count SAC-associated riders per two-wheeler (pose centroids + mask instances); occlusion-robust threshold ≥3 → violation; child-rider heuristic via keypoint scale.
  - *Source:* DashCop SAC; Mask R-CNN triple-rider (2025).
  - *Why:* counting *on a specific bike* (not all nearby people) is the only correct formulation; association makes occluded pillion handling tractable.
- **C. Scene-Context (wrong-side, stop-line, red-light, parking).** Consumes SCM + tracklets + (optional) monocular-3D.
  - *Signal state:* classifier on SCM signal-head ROI → red/amber/green (+ temporal smoothing).
  - *Wrong-side:* vehicle **heading** (tracklet displacement, or **single-frame 3D yaw from LocateAnything3D**) vs SCM lane-direction field → mismatch.
  - *Stop-line:* vehicle front (ground-plane projection via homography / 3D box) crosses stop-line polyline during red.
  - *Red-light:* enters intersection while signal=red — **temporal** (≥2 frames / short clip).
  - *Illegal parking:* tracklet velocity ≈ 0 inside no-parking polygon for > T seconds.
  - *Source:* SCM design; RLR-LSTM (2022); wrong-way optical-flow/DeepSORT refs; LocateAnything3D (CVPR 2026).
  - *Why:* these violations are *defined relative to infrastructure and time* — they are impossible without a scene prior, and several require temporal/3D evidence.

**L4 — Confidence Fusion + Abstain band**
- *Technique:* per-violation confidence = calibrated product/learned-fusion of (detector score × attribute/association confidence × scene-grounding certainty), with **temperature scaling** for calibration. **Multi-evidence corroboration** (e.g., red-light needs signal=red AND crossing). Three output bands: **auto-confirm** (high precision, eligible for challan), **human-review queue** (mid), **discard** (low). Optional **LocateAnything-3B open-vocabulary verification**: re-ground the flagged event with a referring query as an independent second opinion.
- *Source:* calibration literature; LocateAnything-3B (2026).
- *Why:* legal enforcement penalizes false positives heavily; corroboration + calibrated abstain + human-in-the-loop is what makes the system *deployable*, not just accurate.

**L5 — ROI-Gated LPR (no bottleneck)**
- *Technique:* runs **only on L4-confirmed violators**. plate detect (small YOLO head / detector class) → perspective rectification (STN/homography) → restoration + super-resolution on the *crop* → **PARSeq** OCR fine-tuned on Indian plates → **syntax validator** (regex over state codes + HSRP format) → plate confidence. NVIDIA **LPDNet/LPRNet** as a deploy-ready alternative; LocateAnything-3B text-localization as cross-check.
- *Source:* PARSeq (ECCV 2022); PaddleOCR (2023); Indian ANPR (arXiv 2207.06657); NVIDIA LPR.
- *Why over alternatives:* gating removes the dominant throughput cost (OCR on every vehicle); PARSeq's internal language model + an explicit Indian-format grammar beats generic OCR on decorative/HSRP plates; the syntax layer rejects impossible strings before they reach a challan.

**L6 — Evidence Generation**
- *Technique:* annotated composite (full frame + boxes + per-violation label + confidence + **zoomed insets** for head/plate, signal state, timestamp, camera ID, geo) + structured JSON metadata (violation_id, type, confidences, plate_text+conf, vehicle_class, bbox, tracklet_id, timestamp, camera/geo, **model_versions**, evidence_path) + **SHA-256 hash + digital signature** for tamper-evidence / chain-of-custody.
- *Why:* admissibility requires reproducible, auditable, tamper-evident artifacts — not just a labeled JPEG.

**L7 — Analytics & Reporting**
- *Technique:* time-series + search index (OpenSearch/Elastic) keyed by plate/type/location/time; dashboards for counts, trends, **hotspot maps**, repeat-offender lookup, per-camera health; summary reports + e-challan export API.
- *Why:* turns evidence into enforcement decisions and policy insight; directly serves the "searchable records / summary reports / trends" task.

**Offline Data-Engine (the multiplier).**
- *Technique:* **LocateAnything-3B / DINO-X / Grounded-SAM-2 / Florence-2** auto-annotate IDD/DriveIndia/local footage via open-vocab + referring prompts (e.g., "auto-rickshaw", "motorcyclist without helmet", "vehicle past the white stop line"); pseudo-labels are human-spot-checked, then **distilled into the fast Stage-1 YOLOv11/12 student**.
- *Why:* solves Indian-data scarcity cheaply and is the practical route to high accuracy on rare classes without massive manual labeling.

### 3.3 Handling single-frame ambiguity (explicitly)

Some violations are ill-posed from one static frame. We handle this on **three escalating levels**, choosing per violation:
1. **Single-frame, scene-grounded:** stop-line, illegal-parking-candidate, helmet, seatbelt, triple-riding — decidable from one frame + SCM + association.
2. **Single-frame 3D:** wrong-side and stop-line strengthened by **LocateAnything3D** monocular 3D yaw/ground-position (Chain-of-Sight) — heading vs lane-direction field without needing multiple frames.
3. **Short tracklet / clip (2–N frames):** red-light (crossing *during* red) and confirmed illegal parking (stationary > T) genuinely need time → ByteTrack tracklets over a short buffered clip. The system is **photo-first but clip-capable**: where a still is inherently ambiguous, it escalates to a short sequence rather than guessing.

---

## 4. Novelty Statement

This is meaningfully more than "a recent YOLO + an OCR module":

1. **Paradigm-partitioned reasoning.** The architecture is organized around the *nature of evidence* each violation needs — instance-attribute, multi-instance counting, scene-context — with three specialized reasoners instead of one monolithic head. This is the design principle the problem statement hints at, made concrete.
2. **Per-camera Scene Context Model (SCM)** as a first-class, reusable spatial prior. Decoupling static scene geometry (calibrated once) from per-frame perception turns ill-posed single-frame scene violations into well-posed grounded reasoning, reused across millions of frames at ~zero marginal cost.
3. **Association-first attribution.** A SAC-style rider↔vehicle cross-association (SAM2 masks + pose) is the shared substrate for *both* attribute attribution (whose helmet/seatbelt) and counting (how many on this bike) — solving the "whose violation" problem plain detection ignores, which is what makes evidence enforceable.
4. **Chain-of-Evidence reasoning** (borrowed from LocateAnything3D's *Chain-of-Sight*): each scene-context decision is produced as an ordered, interpretable chain — *locate vehicle → ground in scene (lane/zone, distance to stop-line) → infer state (heading/stopped) → decide* — yielding auditable evidence rather than an opaque score, which matters for legal review.
5. **Foundation-model teacher→efficient-student data engine + open-vocab cloud verifier.** LocateAnything-3B / DINO-X used to (a) auto-label scarce Indian data and distill into a fast edge model, and (b) provide an independent open-vocabulary second opinion at confirmation time — *without* putting a slow 3B VLM on the real-time path.
6. **Admissibility-oriented, scalable system design.** Calibrated multi-evidence fusion, an abstain/human-review band, a server-side screen→confirm compute cascade, and tamper-evident signed evidence — engineered for *deployable enforcement at image-volume scale*, optimizing false-positive cost, not just mAP.
7. **Indian-domain specialization throughout:** auto-rickshaw/cart classes, triple-riding as a first-class paradigm, HSRP/old-format plate grammar, IDD/IDD-AW/DriveIndia/RideSafe-400 adaptation.

---

## 5. Evaluation Plan

### 5.1 Metric mapping (per module/task)

| Module / Task | Primary metrics | Datasets |
|---|---|---|
| L1 Detection (Task 2) | **mAP@0.5, mAP@0.5:0.95**, per-class AP, **small-object AP** | IDD, DriveIndia, IDD-AW; COCO (pretrain), Cityscapes/BDD (cross-domain) |
| A. Helmet / Seatbelt (Task 3/4) | **Accuracy, Precision, Recall, F1** (per instance) + **attribution accuracy** | RideSafe-400, helmet sets; custom windshield/seatbelt set |
| B. Triple riding (Task 3/4) | Counting **MAE / accuracy**, violation **P/R/F1** | RideSafe-400 |
| C. Scene-context (Task 3/4) | **P/R/F1** per violation, **event-level** detection rate & **false-alarm rate**; signal-state **accuracy** | Custom SCM-annotated cameras; LISA/BSTLD (signal); wrong-way clips |
| L5 LPR (Task 5) | plate-detect **mAP**; OCR **char accuracy (CRR)**, **plate exact-match**, **1-NED**; end-to-end recognition rate | Indian ANPR sets, custom HSRP set; CCPD (cross-domain) |
| L4 Confidence | **ECE / reliability diagrams**, precision@operating-point | All violation sets |
| System (Task 8) | **images/sec/GPU, cost per 1,000 images**, batch throughput, autoscaling efficiency, end-to-end latency per image; (optional) edge FPS/energy | On-target benchmarking |

### 5.2 Datasets
- **Indian-primary:** IDD (seg/detection), **IDD-AW** (adverse weather), **DriveIndia** (24-class detection), **RideSafe-400** (helmet/triple riding), IDD-3D (3D/eval).
- **Cross-domain/standard:** COCO (pretrain), Cityscapes/BDD100K (generalization), CCPD (plates), LISA Traffic Light / BSTLD (signals), Omni3D (3D sanity).
- **Custom required:** per-camera SCM-annotated set (stop-line/lane/zone/signal), seatbelt-through-windshield set, Indian HSRP plate set.

### 5.3 Custom evaluation considerations (unique to this problem)
- **Operating-point separation:** report a high-precision threshold for *auto-challan* and a recall-leaning threshold for the *human-review queue*; use F-beta accordingly.
- **Attribution-correctness** as a distinct metric (correct rider↔bike↔plate linkage) — not captured by detection mAP.
- **Event-level vs frame-level** scoring for temporal violations (one violation event, not per-frame double counting).
- **Robustness slices:** report all metrics sliced by night / rain / fog / glare using IDD-AW-style conditions.
- **Cross-camera generalization:** train on a subset of cameras, test on unseen cameras (real deployment condition).
- **Calibration & abstain coverage:** ECE plus the precision/recall *within* the auto-confirm band and the fraction sent to human review.
- **Fairness/error analysis** across plate states and vehicle types to avoid systematic bias against specific vehicle classes.

---

## 6. Direct assessment — does LocateAnything-3B (and friends) help us?

**Short answer: yes, in three disciplined roles — but it does *not* replace the fast screening stack that gives us throughput at image-volume scale.** (The "GoodBye YOLO" framing is hype; here is the engineering reality.)

**What it is.** LocateAnything-3B (NVIDIA Tech Report, 2026) is an open-vocabulary grounding **VLM** (Moon-ViT + Qwen2.5) whose innovation is **Parallel Box Decoding** — predicting full boxes in one parallel step (~12.7 boxes/s on an H100, ~10× faster than Qwen3-VL, 2.5× vs Rex-Omni). It does detection, **referring-expression grounding**, point localization, and **text localization (OCR)**, trained on 12M images across natural/driving/GUI/document domains. **LocateAnything3D** (CVPR 2026) extends this to open-vocab **monocular 3D** via **Chain-of-Sight** (2D→distance→size→pose).

**Where it genuinely helps us:**
1. **Data engine (highest ROI).** Use it (with DINO-X / Grounded-SAM-2) to auto-label Indian footage by prompt — including long-tail classes (auto-rickshaw, cart) and *compositional* events ("motorcyclist without helmet", "car past the stop line") — then **distill into the fast Stage-1 YOLOv11/12 student**. This directly attacks Indian-data scarcity.
2. **Cloud open-vocabulary verifier (precision booster).** At L4, re-ground a flagged event with a referring query as an independent second opinion → corroboration → fewer false challans. Scene-context violations especially benefit from compositional grounding a closed-vocab detector can't express.
3. **Single-frame 3D for scene-context (LocateAnything3D).** Monocular 3D **yaw + ground position** helps wrong-side and stop-line *from a single frame*, partially resolving the static-frame ambiguity the brief calls out. And its **Chain-of-Sight** is the template for our interpretable **Chain-of-Evidence** reasoning.

**Why it can't be the workhorse (be honest in the pitch):**
- **Throughput/cost:** ~12 boxes/s on an **H100** is orders of magnitude too slow/expensive to run on *every* incoming image at scale. The batched **Stage-1 YOLOv11/12** screen stays the throughput workhorse; the VLM runs only on flagged candidates (and offline as a teacher).
- **Determinism & calibration:** generative VLM output can hallucinate and is harder to calibrate/audit than a fixed detector with known FP/FN — a liability for legally binding challans. Hence VLM = teacher/verifier, not sole evidence source.
- **Licensing:** confirm the NVIDIA model license (often research/eval terms) before any commercial deployment claim.

**Net:** LocateAnything-3B / LocateAnything3D slot cleanly into our **Foundation-Model Layer** (offline teacher + cloud verifier + single-frame-3D grounder) and the **Chain-of-Evidence** structure — strengthening both accuracy and the novelty story — while the scalable throughput spine remains the batched Stage-1 YOLOv11/12 + ByteTrack screen feeding the paradigm-specialized reasoners.

---

## 7. Sources
- LocateAnything-3B — research.nvidia.com/labs/lpr/locate-anything ; huggingface.co/nvidia/LocateAnything-3B
- LocateAnything3D (Chain-of-Sight, CVPR 2026) — arxiv.org/abs/2511.20648
- DashCop (RideSafe-400, SAC) — arxiv.org/abs/2503.00428
- DriveIndia — arxiv.org/abs/2507.19912
- IDD — arxiv.org/abs/1811.10200 ; IDD-AW — iddaw.github.io ; IDD-3D — idd3d.github.io
- Helmet via NVIDIA TAO+YOLOv8 — frontiersin.org (frai.2025.1582257)
- YOLOv12 — arXiv 2025 ; RT-DETR — arxiv.org/abs/2304.08069 ; Co-DETR (ICCV 2023)
- DINOv2 — arxiv.org/abs/2304.07193 ; DINO-X — arxiv.org/abs/2411.14347
- RTMPose — arxiv.org/abs/2303.07399 ; ViTPose (NeurIPS 2022)
- PARSeq — arxiv.org/abs/2207.06966 ; Indian end-to-end ANPR — arxiv.org/abs/2207.06657
- NVIDIA TAO Toolkit / DeepStream / TrafficCamNet / LPDNet / LPRNet — developer.nvidia.com/tao-toolkit
