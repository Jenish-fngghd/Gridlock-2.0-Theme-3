# Research Findings — SOTA Models, Papers & Datasets

**Status:** First pass complete (web search, June 2026). Numbers labeled **VERIFY** need
confirmation from the primary source before quoting to judges. Watch licenses —
Ultralytics YOLO is **AGPL-3.0** (matters if you distribute).

---

## 1. Image Preprocessing / Restoration
*Question: enhance images, or train a detector robust to degradation? Recommend: do BOTH
— light enhancement as optional front-end + train/eval detectors on weather benchmarks.*

| Method | Year/Venue | Task | Code | Note |
|--------|-----------|------|------|------|
| **Retinexformer** | ICCV 2023 | Low-light enhancement | open source | One-stage Retinex transformer; **won NTIRE 2025/2026** low-light challenges (VERIFY) |
| RetinexMamba / Retinexformer+ / MambaLLIE / ERetinex | 2024–2025 | Low-light | open | Follow-ups improving PSNR on LOL-v2; Mamba/SSM variants |
| **OneRestore** | ECCV 2024 | All-in-one composite degradation | open | Single model for haze/rain/low-light/blur combos — great fit for "one preprocessor" story |
| MS state-space restoration (MambaIR-style) | 2024 | General restoration | open | Efficient UHD restoration |

- **Primary pick:** Retinexformer (low-light) + **OneRestore** (composite weather) as an
  optional, confidence-gated front-end.
- **Backup:** skip restoration, instead train detector directly on BDD100K/IDD night+rain
  splits (often more robust than cascading a restorer).
- **Datasets:** LOL-v1/v2 (low-light), ExDark, Rain100/SOTS (derain/dehaze), GoPro (deblur).
- **Trade-off:** restoration adds latency and can hallucinate; only apply when an
  image-quality classifier flags degradation.

## 2. Vehicle & Road-User Detection (CORE detector)
| Model | Year | Code | License | Key metric (VERIFY) | Note |
|-------|------|------|---------|---------------------|------|
| **RF-DETR** (Roboflow) | 2025 | open | Apache-2.0 | Leads on occlusion & domain shift | Best when fine-tuning on custom/benchmark data; transformer DETR |
| **YOLOv12** | 2025 | [github](https://github.com/sunsmarterjie/yolov12) | check | YOLOv12-N 40.5% mAP COCO, 1.62ms T4 | Attention-centric; beats YOLOv10/11 at similar speed; X beats RT-DETRv2/v3 |
| RT-DETR / RT-DETRv2 | 2023–24 | open | Apache-2.0 | Real-time DETR | Strong, license-friendly transformer baseline |
| YOLO11 (Ultralytics) | 2024 | open | **AGPL-3.0** | — | Easiest tooling/ecosystem; license caveat |
| YOLO-World / GroundingDINO | 2024 | open | — | Open-vocabulary | Zero-shot detect "rider", "helmet", "no-helmet" without training |

- **Primary pick:** **RF-DETR** (precision + occlusion, Apache license) for vehicles/road
  users; **YOLOv12** as the fast alternative.
- **Backup / cold-start:** YOLO-World or GroundingDINO for open-vocabulary prototyping
  (label-free demo of rare classes).
- **Datasets:** COCO (general), **BDD100K** (100k imgs, weather/time-of-day), **IDD /
  DriveIndia** (Indian scenes), UA-DETRAC, KITTI, Cityscapes, Mapillary.
- **Trade-off:** DETR variants = better accuracy/occlusion but heavier; YOLO = faster +
  bigger ecosystem but AGPL for Ultralytics builds.

## 3. Violation Detection (per type)
| Violation | Approach | Evidence / model |
|-----------|----------|------------------|
| **Helmet non-compliance** | Detector fine-tuned on helmet data | AI City Challenge **Track 5**; YOLOv8/v11, Co-DETR; synthetic (Stable Diffusion) data hit mAP@.5 0.955 val / 0.822 test (VERIFY) |
| **Triple riding** | Occupant counting on two-wheeler | ResNet18 **DetectNet_v2** reported 91.42% accuracy (VERIFY) |
| **Seatbelt** | In-cabin via windshield crop + classifier | No clean public dataset → flag as proxy/weak-supervision; hard at night/tint |
| **Wrong-side / stop-line / red-light** | Geometry + tracking + signal-state, NOT single still | Needs lane/zone map + MOT (§6) + temporal; honest "needs video" framing |
| **Illegal parking** | No-parking zone map + dwell time | Detection + zone polygon + time threshold |

- **Best competition reference:** **NVIDIA AI City Challenge** (8th, 2024 / 9th, 2025) —
  Track 5 is *Motorcyclist Helmet Violation Detection*. Mine winning repos: few-shot
  YOLOv8, YOLOv5 ensembles, Co-DETR + minority-class enhancement. Public leaderboard
  scores ~0.52 mAP show how hard real-world helmet detection is (use to set expectations).
- **India-specific:** Frontiers 2025 paper — helmet violation + vehicle ID in Indian smart
  cities using **NVIDIA TAO toolkit + YOLOv8** (directly citable for your context).
- **Primary pick:** fine-tuned RF-DETR/YOLOv12 head per violation + rider–helmet
  association logic; geometry module for the temporal ones.

## 4. Violation Classification + Confidence
- Derived layer: take detector/tracker outputs → rule engine + lightweight classifier →
  predefined classes with calibrated confidence (temperature scaling / softmax).
- **Innovation hook:** cascaded confidence → low-confidence cases routed to VLM verifier
  (§7) and/or human-in-the-loop.

## 5. License Plate Detection + OCR (ANPR)
| Stage | Model | Year | License | Note |
|-------|-------|------|---------|------|
| Plate detection | YOLOv11/YOLOv10 | 2024–25 | AGPL (Ultralytics) | YOLOv10 ~99.16% detection acc reported (VERIFY) |
| Plate detection | RF-DETR / YOLOv12 | 2025 | Apache/check | License-friendlier alternative |
| OCR | **PaddleOCR PP-OCRv5** | 2025 | Apache-2.0 | +13pp E2E over v4; 3.1.0 adds 37 languages; handles rotated/curved/skewed text |
| OCR | EasyOCR | — | Apache-2.0 | Simple, good baseline |
| OCR | Tesseract | — | Apache-2.0 | Weakest on plates, easiest |

- **Primary pick:** YOLO/RF-DETR plate detector → **PaddleOCR PP-OCRv5** for recognition.
- **Backup:** EasyOCR for a fast prototype.
- **Datasets:** **CCPD** (250k+ Chinese plates, vertex annotations, ECCV 2018), **UFPR-ALPR**,
  **AOLP**, plus Indian sets — "Indian License Plate Dataset in the wild" (16,192 imgs /
  21,683 plates, 10 states), and arXiv Indian-plate recognition papers.
- **Trade-off:** CCPD is huge but Chinese-format; fine-tune on Indian set for real plates.

## 6. Multi-Object Tracking (enables geometry/temporal violations)
| Tracker | Note |
|---------|------|
| **ByteTrack** | SOTA tracking-by-detection; associates low-conf boxes; great under occlusion |
| **BoT-SORT** | Adds camera-motion compensation + appearance; better for moving/CCTV cameras |
| OC-SORT | Strong on non-linear motion |

- **Primary pick:** **BoT-SORT** (camera-motion comp suits surveillance) → **ByteTrack**
  backup. Both ship inside Ultralytics/YOLO frameworks → trivial to wire.
- **Why it matters:** wrong-side (direction of track), red-light/stop-line (cross line
  during red), illegal parking (dwell time) all need IDs persisted across frames.

## 7. VLM / Multimodal Verification Layer (the novelty differentiator)
| Model | Note |
|-------|------|
| **Qwen2.5-VL** (7B/72B) | Open-source, SOTA image understanding; 7B ~8.5s & ~18GB / 10 frames on L40S (heavy) |
| **InternVL3** | Native multimodal pretraining; strong grounding |
| TrafficInternVL (ICCV 2025) | VLM specialized for traffic scenario understanding |
| TrafficVLM | Controllable VLM for traffic video captioning |
| Cerberus (2025) | Cascaded VLMs for real-time video anomaly detection |

- **Use as:** (a) verification/explanation layer for low-confidence violations, (b)
  natural-language evidence captioning ("rider without helmet, plate XX, 14:32"),
  (c) zero-shot fallback for rare violation types.
- **Caveat:** too slow/expensive as primary detector → use selectively (cascade), not on
  every frame.
- **Benchmarks emerging:** RoadSafe365, RoadSafe-style VLM traffic-safety benchmarks (2026).

## 8. Benchmark Dataset Map (proves each component — no proprietary data needed)
| Component | Dataset(s) | Size | License | Leaderboard | India-relevant |
|-----------|-----------|------|---------|-------------|----------------|
| Detection (vehicles/road users) | COCO, **BDD100K**, IDD, **DriveIndia**, UA-DETRAC, KITTI | 100k+ | mostly research-use | Yes (PwC) | IDD/DriveIndia ✅ |
| Weather/quality robustness | LOL-v1/v2, ExDark, Rain100, SOTS, GoPro | varies | research | NTIRE | — |
| Helmet violation | **AI City Challenge Track 5**, Roboflow helmet sets | 100 videos | challenge/ CC | Yes (AI City) | partial |
| Triple riding | derive from helmet/AI City data + occupant count | — | — | — | ✅ |
| Seatbelt | **no clean public set** → proxy/weak-supervision | — | — | — | — |
| License plate | **CCPD**, UFPR-ALPR, AOLP, Indian-plate-in-the-wild | 250k / 16k | research | Yes | ✅ Indian set |
| OCR | CCPD + PP-OCRv5 eval sets | — | Apache (model) | — | — |
| Tracking | BDD100K MOT, UA-DETRAC | — | research | Yes | — |

---

## Open Questions / VERIFY before quoting
- Confirm exact mAP/PSNR numbers above against primary sources (several are search-summarized).
- Confirm licenses: YOLOv12 repo license, RF-DETR (Apache), Ultralytics (AGPL-3.0).
- Pull the **AI City Challenge 9th (2025)** Track 5 leaderboard + winning method repos.
- Check DriveIndia (arXiv 2507.19912) availability/license — newest Indian detection set.
- Decide: still-image-only vs allow short video clips (unlocks wrong-side/red-light/stop-line).

## Sources
- [Best Object Detection Models 2026 — Roboflow](https://blog.roboflow.com/best-object-detection-models/)
- [RF-DETR — Roboflow](https://blog.roboflow.com/rf-detr/)
- [YOLOv12 (OpenReview)](https://openreview.net/forum?id=gCvByDI4FN) · [code](https://github.com/sunsmarterjie/yolov12)
- [Retinexformer+ (CMC 2025)](https://www.techscience.com/cmc/v82n2/59451)
- [AI City Challenge helmet (arXiv 2304.08256)](https://arxiv.org/pdf/2304.08256) · [YOLOv5 ensemble (2304.09246)](https://arxiv.org/pdf/2304.09246)
- [Indian smart-city helmet detection — Frontiers 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1582257/full)
- [CCPD dataset (GitHub)](https://github.com/detectRecog/CCPD)
- [DriveIndia (arXiv 2507.19912)](https://arxiv.org/html/2507.19912v1) · [Indian plate in the wild (2111.06054)](https://arxiv.org/pdf/2111.06054)
- [PaddleOCR PP-OCRv5 docs](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) · [PaddleOCR 3.0 report (arXiv 2507.05595)](https://arxiv.org/html/2507.05595v1)
- [ByteTrack/BoT-SORT — Ultralytics docs](https://docs.ultralytics.com/modes/track) · [MOT review IET 2025](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.70010)
- [Qwen2.5-VL](https://www.kaggle.com/models/qwen-lm/qwen2.5-vl) · [Cerberus cascaded VLM (arXiv 2510.16290)](https://arxiv.org/pdf/2510.16290)
