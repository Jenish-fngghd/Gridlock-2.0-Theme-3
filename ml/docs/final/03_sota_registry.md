# Section 2 — SOTA Research Review & Paper / Source Registry (2022–2026)
### Gridlock 2.0 R2 · Automated Traffic-Violation Image Analysis
*(Standalone companion to the main Solution Design. A spreadsheet version is in `03_sota_registry.csv`.)*

---

## Search methodology

We searched 2022–2026 SOTA across the five technical axes mandated by the brief, plus four industry/open-source sources. Each result is mapped to (a) the axis it informs, (b) the **module** it contributes to our pipeline, and (c) **why** it beats alternatives for *unstructured Indian traffic*. Axis legend:

| Code | Axis |
|---|---|
| **A** | General object detection & multi-task / foundation detection |
| **B** | Traffic monitoring, violation detection & driver-behavior CV |
| **C** | Fine-grained / part-level / human pose & attribute / segmentation |
| **D** | License-plate detection & OCR (Indian formats) |
| **E** | Scene understanding — lane / signal-state / spatial-temporal context |
| **F** | Datasets, domain & applied/industry (NVIDIA, Meta/FAIR, OSS zoos) |
| **P** | Image preprocessing / restoration |

---

## Axis-by-axis synthesis

### A · General object detection & multi-task / foundation
The frontier has split into two families. **CNN-family YOLOv11 (2024)** and **attention-centric YOLOv12 (2025, Area-Attention + R-ELAN)** lead the real-time accuracy/latency trade-off; **RT-DETR (CVPR 2024)** and **Co-DETR (ICCV 2023)** lead NMS-free / high-recall detection for dense scenes. Above them sit **foundation** models: **DINOv2 (TMLR 2024)** features for label-efficient transfer, and **open-vocabulary grounding** — **DINO-X / Grounding-DINO 1.5**, **Florence-2 (CVPR 2024)**, and NVIDIA **LocateAnything-3B (2026)** — which enable zero-shot/long-tail detection and referring-expression queries. *Takeaway:* use YOLOv11/12 as the fast screening detector, DETR for confirmation, and the grounding VLMs as teacher + verifier (not the throughput workhorse).

### B · Traffic monitoring, violation detection & driver behavior
**DashCop (2025)** is the most relevant: its **Segmentation-and-Cross-Association (SAC)** module solves rider↔motorcycle attribution and ships the **RideSafe-400** helmet/triple-riding dataset. The **NVIDIA-TAO + YOLOv8 (Frontiers in AI, 2025)** Indian smart-city study reports helmet + plate + triple-rider (91.4%) on the production NVIDIA stack. **Optimized Mask R-CNN (2025)** validates segmentation-based triple-rider counting. *Takeaway:* attribution (whose violation) — not raw detection — is the hard part; SAC is our substrate for Paradigms A & B.

### C · Fine-grained / part-level / pose & attribute / segmentation
**RTMPose (2023)** gives real-time multi-person 2D pose; **ViTPose (NeurIPS 2022)** is the high-accuracy/occlusion option. Pose-guided ROI localizes the *head* (helmet) and *torso* (seatbelt) on the correct person, and per-person keypoints drive triple-rider counting. **SAM 2 (Meta, 2024)** supplies promptable masks for SAC association and scene-zone labeling. Seatbelt SOTA uses a **windshield→occupant→belt cascade (SIViP 2022)** and **3D-pose + belt segmentation (arXiv 2204.07946)** to fight glare/occlusion. *Takeaway:* fine-grained attributes must be reasoned on a *correctly attributed* person, not a global object class.

### D · License-plate detection & OCR (Indian formats)
**PARSeq (ECCV 2022)** — permuted-autoregressive STR with an internal language model — is the accuracy SOTA and its syntax modeling suits structured plates; **PaddleOCR PP-OCRv4 (DBNet + recognizer)** is the deployable/CPU-friendly fallback; **MMOCR** allows recognizer ablation. **End-to-end ANPR for Indian datasets (arXiv 2207.06657)** and **YOLO+EasyOCR Indian ANPR** target Indian variability (old black-plate vs new **HSRP**, state codes, decorative fonts). NVIDIA **LPDNet/LPRNet** are deploy-ready alternatives. *Takeaway:* pair PARSeq with an explicit Indian-format grammar, and **gate OCR to confirmed violators** to avoid a throughput bottleneck.

### E · Scene understanding — lane / signal / spatial-temporal context
Scene-context violations are *defined relative to infrastructure and time*. **Red-light-running via LSTM + rules (2022)** frames it as a temporal-event problem; **wrong-way detection (YOLO + optical-flow / DeepSORT, 2020–24)** supplies direction-vs-lane logic; **ByteTrack (ECCV 2022)** gives robust tracklets for crossing/parking-duration events. **LocateAnything3D — Chain-of-Sight (CVPR 2026)** adds open-vocab **monocular 3D** (2D→distance→size→pose), giving single-frame heading/ground-position for wrong-side and stop-line. *Takeaway:* encode static geometry once per camera (our **Scene Context Model**) and add temporal/3D evidence only where a still is ambiguous.

### F · Datasets, domain & applied/industry
**IDD (WACV 2019)** + **IDD-AW (WACV 2024, adverse weather)** + **IDD-3D (WACV 2023)** + **DriveIndia (2025; 66,986 imgs, 24 classes, YOLOv8 mAP@50≈78.7)** anchor Indian-domain training/eval; **RideSafe-400** covers helmet/triple-riding. Industry: **NVIDIA TAO Toolkit + DeepStream** (TrafficCamNet/DashCamNet/LPDNet/LPRNet) for deployment; **Meta/FAIR** (DINOv2, SAM 2); open-source zoos (**Ultralytics, OpenMMLab MMPose/MMOCR, PaddleOCR, Hugging Face**). *Takeaway:* Western benchmarks (Cityscapes/BDD) assume lane discipline — Indian datasets are mandatory, not optional.

### P · Image preprocessing / restoration
**Zero-DCE++ (TPAMI 2021)** — zero-reference, real-time low-light enhancement — applied *selectively* (quality-gated), with heavy restoration/super-resolution reserved for the **license-plate crop**. *Takeaway:* push robustness into training (synthetic rain/fog/blur from IDD-AW statistics); naïve full-frame restoration hurts detection and adds latency.

---

## Master Paper / Source Registry

| # | Axis | Source / Paper | Venue & Year | Core Contribution | Module Borrowed | Why It Fits Our Problem |
|---|---|---|---|---|---|---|
| 1 | F | **IDD: India Driving Dataset** (Varma et al.) | WACV 2019 | 34-class unstructured Indian road dataset; fuzzy lanes, dense heterogeneous traffic | Backbone domain pretraining + seg priors | Cityscapes/BDD assume disciplined Western traffic; IDD captures Indian chaos |
| 2 | F/P | **IDD-AW** | WACV 2024 | 5k image pairs, pixel labels under rain/fog/low-light/snow | Robustness eval + preprocessing target | Directly stresses adverse-condition robustness |
| 3 | F/E | **IDD-3D** | WACV 2023 | Multi-modal (cam+LiDAR) 3D Indian scenes | Monocular-3D fine-tune/eval reference | Grounds 3D scene-context reasoning in Indian data |
| 4 | F | **DriveIndia** (arXiv 2507.19912) | 2025 | 66,986 imgs, 24 classes, 471k instances; YOLOv8 mAP@50≈78.7 | Detection fine-tuning + benchmark | Modern large Indian *detection* benchmark incl. auto-rickshaw |
| 5 | B/C | **DashCop** (arXiv 2503.00428) | 2025 | **SAC** rider↔motorcycle module; cross-association tracking; **RideSafe-400** | Association substrate (Paradigm A & B) | Solves the exact "whose helmet / how many on this bike" problem |
| 6 | B/F | **Helmet via NVIDIA TAO + YOLOv8** (Frontiers in AI) | 2025 | Indian smart-city helmet + plate; triple-rider 91.4% | Helmet branch + TAO/DeepStream deploy path | Proven Indian-city deployment recipe on NVIDIA stack |
| 7 | B | **Optimized Mask R-CNN triple-rider** (Syst. Sci. Control Eng.) | 2025 | Instance-seg approach to triple riding | Counting reasoner reference (Paradigm B) | Validates seg-based counting under occlusion |
| 8 | A | **YOLOv11** (Ultralytics) | 2024 | C3k2 blocks, spatial attention, ~22% fewer params vs v8m | Stage-1 fast screening detector (primary) | Best accuracy/throughput at volume; edge-exportable |
| 9 | A | **YOLOv12** (arXiv) | Feb 2025 | Area-Attention (A²) + R-ELAN; attention-centric SOTA | Stage-1 detector (alt/upgrade) | Attention gains for small/occluded objects in real time |
| 10 | A | **RT-DETR** (Zhao et al.) | CVPR 2024 | Real-time end-to-end DETR; hybrid encoder; NMS-free | Stage-2 confirmation detector option | NMS-free helps dense overlapping Indian traffic |
| 11 | A | **Co-DETR** (Zong et al.) | ICCV 2023 | Collaborative hybrid-assignment training | Stage-2 confirmation detector | Higher recall in cluttered scenes for evidence |
| 12 | A | **DINOv2** (Oquab et al., FAIR) | TMLR 2024 / arXiv 2023 | Self-supervised general visual features | Backbone init for label-efficient transfer | Adapts to Indian domain with less labeled data |
| 13 | C | **Segment Anything 2 (SAM 2)** (Meta) | 2024 | Promptable image/video segmentation + tracking | Masks for SAC association + scene-zone labeling | Generates instance masks to cross-associate riders↔bikes |
| 14 | C | **RTMPose** (Jiang et al., MMPose) | arXiv 2023 | Real-time multi-person 2D pose | Pose-guided ROI (helmet/seatbelt) + rider count | Efficient in crowded scenes; localizes head/torso |
| 15 | C | **ViTPose** (Xu et al.) | NeurIPS 2022 | ViT-based SOTA pose under occlusion | High-accuracy pose alternative | Better under heavy occlusion for confirmation |
| 16 | D | **PARSeq** (Bautista & Atienza) | ECCV 2022 | Permuted-autoregressive STR; internal LM | LP OCR engine | Syntax-aware decoding suits structured Indian plates |
| 17 | D | **PaddleOCR PP-OCRv4** (Baidu) | 2023 | DBNet + lightweight recognizer; deployable | LP detect+OCR practical alt | CPU-friendly, production-proven fallback |
| 18 | D | **MMOCR** (OpenMMLab) | 2022– | DBNet/NRTR/ABINet toolkit | OCR experimentation/ablation | Swap recognizers to tune Indian-plate accuracy |
| 19 | D | **End-to-end ANPR for Indian datasets** (arXiv 2207.06657) | 2022 | Indian-specific ANPR network | Indian LP reference design | Targets Indian plate variability directly |
| 20 | P | **Zero-DCE++** (Guo et al.) | TPAMI 2021 | Zero-reference low-light enhancement, very fast | Selective low-light preprocessing | Unsupervised, real-time; no paired night data |
| 21 | E | **ByteTrack** (Zhang et al.) | ECCV 2022 | Associates *every* detection box (incl. low-score) | MOT for temporal Paradigm-C violations | Robust tracklets for red-light/parking/wrong-way |
| 22 | E | **Red-light running via LSTM + rules** (Springer) | 2022 | Temporal modeling of RLR events | Scene-context temporal reasoner reference | Frames red-light as a temporal-event problem |
| 23 | E | **Wrong-way detection (YOLO + optical-flow / DeepSORT)** (MDPI) | 2020–24 | Direction estimation for wrong-way | Wrong-side reasoner reference | Direction-vs-lane logic adapted to our SCM |
| 24 | C/B | **Seatbelt: windshield-YOLOv5s / 3D-pose + belt seg** (SIViP 2022; arXiv 2204.07946) | 2022 | Windshield→occupant→belt cascade; pose+seg | Seatbelt branch (Paradigm A) | Handles through-windshield glare/occlusion |
| 25 | A/F | **NVIDIA LocateAnything-3B** (NVIDIA Tech Report) | 2026 | Open-vocab grounding VLM (Moon-ViT+Qwen2.5); **Parallel Box Decoding**; detect+refer+point+OCR; 12M img/138M queries | **Foundation Layer**: data-engine teacher, open-vocab verifier, plate-locate assist | Zero-shot Indian/long-tail classes + compositional scene queries |
| 26 | E/F | **NVIDIA LocateAnything3D — Chain-of-Sight** (arXiv 2511.20648) | CVPR 2026 | Open-vocab **monocular 3D** as next-token prediction; 2D→distance→size→pose; Omni3D +13.98 AP3D | **Chain-of-Evidence** structure + single-frame 3D heading/position | Resolves single-frame ambiguity for wrong-side/stop-line |
| 27 | A | **DINO-X / Grounding-DINO 1.5** (IDEA Research) | 2024–25 | Unified open-world detection; prompt-free; Grounding-100M | Auto-labeling teacher alternative | Bulk open-vocab annotation of Indian footage |
| 28 | A/C | **Grounded-SAM-2** (IDEA Research) | 2024 | Grounding-DINO + SAM2 detect-segment-track | Auto-label + association alt | Box→mask→track for weak supervision |
| 29 | A | **Florence-2** (Microsoft) | CVPR 2024 | Unified prompt-based vision (detect/caption/ground) | Teacher / aux annotation | Cheap multi-task pseudo-labels |

---

## How each source maps to the 8 tasks

| Task | Primary sources |
|---|---|
| 1 · Preprocessing | Zero-DCE++ (20), IDD-AW (2) |
| 2 · Detection | YOLOv11/12 (8,9), RT-DETR/Co-DETR (10,11), DINOv2 (12), IDD/DriveIndia (1,4) |
| 3 · Violation detection | DashCop/SAC (5), TAO+YOLOv8 (6), Mask R-CNN (7), RTMPose/ViTPose (14,15), SAM2 (13), Seatbelt (24), RLR-LSTM (22), wrong-way (23), LocateAnything3D (26) |
| 4 · Violation classification + confidence | RTMPose (14), LocateAnything-3B verifier (25); calibration practice |
| 5 · LPR | PARSeq (16), PaddleOCR (17), MMOCR (18), Indian ANPR (19), NVIDIA LPDNet/LPRNet (F) |
| 6 · Evidence generation | (engineering — chain-of-custody; informed by NVIDIA DeepStream pipelines) |
| 7 · Analytics & reporting | (engineering — search/time-series; OpenSearch/Elastic) |
| 8 · Evaluation | All datasets (1–4), Omni3D, CCPD, LISA/BSTLD |

---

## Sources (URLs)
- LocateAnything-3B — research.nvidia.com/labs/lpr/locate-anything · huggingface.co/nvidia/LocateAnything-3B
- LocateAnything3D (CVPR 2026) — arxiv.org/abs/2511.20648
- DashCop — arxiv.org/abs/2503.00428 · DriveIndia — arxiv.org/abs/2507.19912
- IDD — arxiv.org/abs/1811.10200 · IDD-AW — iddaw.github.io · IDD-3D — idd3d.github.io
- Helmet via NVIDIA TAO+YOLOv8 — frontiersin.org (frai.2025.1582257)
- YOLOv12 — arXiv 2025 · RT-DETR — arxiv.org/abs/2304.08069 · Co-DETR — ICCV 2023
- DINOv2 — arxiv.org/abs/2304.07193 · DINO-X — arxiv.org/abs/2411.14347 · Florence-2 — CVPR 2024
- RTMPose — arxiv.org/abs/2303.07399 · ViTPose — NeurIPS 2022 · SAM 2 — ai.meta.com
- PARSeq — arxiv.org/abs/2207.06966 · Indian ANPR — arxiv.org/abs/2207.06657 · PaddleOCR / MMOCR — github
- Zero-DCE++ — TPAMI 2021 · ByteTrack — arxiv.org/abs/2110.06864
- NVIDIA TAO / DeepStream / TrafficCamNet / LPDNet / LPRNet — developer.nvidia.com/tao-toolkit
