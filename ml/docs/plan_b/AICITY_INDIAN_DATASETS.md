# AI City Challenge + Indian Datasets — Deep Dive

**Why this doc:** The NVIDIA AI City Challenge (AICC) **Track 5 — Helmet Violation
Detection for Motorcyclists** is the closest published competition to our problem
statement. Its data, class scheme, winning methods, and metric are directly reusable.
This doc captures what to copy, what to cite, and the best India-specific datasets.

**Compiled:** June 2026 (web search). Items marked **VERIFY** = confirm at primary source
before quoting in the submission.

---

## 1. Why AICC Track 5 is the perfect anchor for our project
- It is *exactly* "detect motorcyclists + helmet compliance from real traffic video."
- It uses a **single 7-class annotation scheme that simultaneously encodes (a) helmet
  compliance per person and (b) how many people are on the bike** → solves BOTH our
  "helmet non-compliance" AND "triple riding" violations with one model. This is a key
  design insight for our pipeline.
- Public papers + open GitHub repos from winners = ready-to-adapt baselines.
- Public leaderboard numbers (~0.48–0.53 mAP) let us set honest expectations and define a
  "SOTA number to beat."

### The 7-class scheme (AICC 2023/2024 Track 5) — ADOPT THIS
| Class | Meaning |
|-------|---------|
| `motorbike` | the motorcycle itself |
| `DHelmet` | Driver WITH helmet |
| `DNoHelmet` | Driver WITHOUT helmet → **violation** |
| `P1Helmet` | Passenger 1 WITH helmet |
| `P1NoHelmet` | Passenger 1 WITHOUT helmet → **violation** |
| `P2Helmet` | Passenger 2 WITH helmet |
| `P2NoHelmet` | Passenger 2 WITHOUT helmet → **violation** |

**Derived violations for free:**
- **Helmet non-compliance** = any `*NoHelmet` class present.
- **Triple riding** = driver + P1 + P2 all present on one `motorbike` (3 person-boxes).
- Per-rider association is built into the labels (D / P1 / P2 roles).

---

## 2. Dataset details (AICC Track 5)
| Property | Value |
|----------|-------|
| Train videos | 100 videos (groundtruth bboxes) |
| Duration | 20 s each |
| Frame rate | 10 fps |
| Resolution | 1920 × 1080, H.264 mp4, 16:9 |
| Max riders/bike | 3 (driver + 2 passengers) |
| GT format | one object/line: `video, frame, track_id, bbox(x,y,w,h), class` |
| Metric | **mAP** (mean average precision) |
| Access | Register per-track at aicitychallenge.org (research use; verify license terms) |

> Note: it's **video** (10 fps), not stills. Frames can be sampled as still images for our
> image-based pipeline, and the track_id column gives us free tracking labels if we want
> to demo the temporal violations.

---

## 3. Winning / notable methods (what to adapt)
| Year | Method | Core idea | Result (VERIFY) | Repo |
|------|--------|-----------|-----------------|------|
| **2024** | **Co-DETR + Minority Class Enhancement** (Vo et al., CVPRW'24) | Co-DETR detector; **Minority Optimizer** + **Virtual Expander** to fix class imbalance (rare passenger/no-helmet classes) | **Rank 1, mAP 0.4860** | CVF paper (PDF blocks scraping; on IEEE Xplore 10678200) |
| 2024 | SKKU AutoLab solution | 2-stage: YOLOv8x detects motorbike (1536px) → crop → YOLOv8x ensemble (320/448/512px) classifies 9 helmet states → K-means temporal smoothing | — | github.com/SKKUAutoLab/aicity_2024_helmet |
| 2023 | Few-Shot Sampling + YOLOv8 (arXiv 2304.08256) | Few-shot data sampling to handle rare classes | — | — |
| 2023 | YOLOv5 + Ensemble / Genetic-Algorithm-enhanced YOLOv5 (arXiv 2304.09246, 2304.09248) | Ensemble + GA hyperparam tuning, real-time | — | cmtsai2023/AICITY2023_Track5_DVHRM |
| 2023 | Custom Tracking Framework (Duong et al., VNPT AI) | Detection + custom tracker for rule violation | — | github.com/vnptai/AI-City-Challenge-2023 |

**Patterns that win this task (steal these):**
1. **Two-stage**: detect motorbike first → crop → classify rider/helmet states on the crop
   (small/distant riders are the hard part; cropping zooms in).
2. **Class-imbalance handling is decisive** — passenger + no-helmet classes are rare.
   Use minority oversampling, few-shot sampling, focal loss, or synthetic data.
3. **Temporal smoothing / tracking** to stabilize per-frame predictions (K-means,
   ByteTrack/BoT-SORT) — reduces flicker false positives.
4. **Ensembles + multi-resolution** inference squeeze the last mAP points.
5. **Synthetic data** (Stable Diffusion) for rare weather/night helmet cases (2025 work).

**Reality check for judges:** even winning solutions sit around **mAP ~0.49–0.53**. Real-
world helmet detection is HARD (occlusion, small riders, night, motion blur). Frame our
target accordingly — don't promise 99%.

---

## 4. The broader AI City Challenge (context + reusable tracks)
**9th AICC (2025, ICCV workshop)** — 245 teams, 15 countries; 4 tracks (arXiv 2508.13564):
- **Track 1:** Multi-camera 3D perception (people/robots/forklifts) — synthetic via Omniverse.
- **Track 2:** **Traffic Video Question Answering** — multi-camera incident understanding
  with 3D gaze; *directly relevant to our VLM verification/explanation layer.*
- **Track 3:** Warehouse spatial intelligence (~500k VQA pairs) — less relevant.
- **Track 4:** **Fisheye road object detection**, lightweight/edge — relevant if cameras
  are fisheye CCTV.
- Several 2025 datasets released publicly on **Hugging Face** (30k+ downloads).

**8th AICC (2024, CVPRW)** — included **Track 5 helmet violation** (our anchor). Earlier
editions (2023) also ran Track 5; that's where most open helmet repos come from.

> Takeaway: cite AICC as the authoritative benchmark lineage; use **Track 5 (2023/2024)**
> for helmet+triple-riding, and **Track 2 (2025)** as precedent for using VLMs to reason
> about traffic incidents.

---

## 5. Indian datasets (for domain relevance — key differentiator vs generic submissions)
| Dataset | Size | Annotation | Use in our pipeline | Access |
|---------|------|-----------|---------------------|--------|
| **DriveIndia** (ITSC 2025, arXiv 2507.19912) | **66,986 imgs, 24 classes**, YOLO format; fog/rain/night; 120+ hrs, 3,400+ km | Object detection | Primary Indian detector training/eval; baseline **mAP50 78.7%** | TiHAN-IIT Hyderabad: tihan.iith.ac.in/TiAND.html |
| **IDD — Detection** | 40,000 imgs, bbox | Detection | Vehicles/road users, unstructured Indian roads | idd.insaan.iiit.ac.in |
| **IDD — Segmentation** | 10,000 imgs, 34 classes (14k/2k/4k split) | Semantic seg | Drivable area / lane / road geometry for stop-line & wrong-side logic | same |
| **IDD — Multimodal** | stereo + 16-ch LIDAR + GPS + CAN | 3D/temporal | Future scope (geometry) | same |
| **Indian License Plate (in the wild)** (arXiv 2111.06054) | 16,192 imgs / 21,683 plates, 10 states | Plate bbox + text | Fine-tune ANPR for Indian plate formats/fonts/placement | per paper |
| **CCPD** (ECCV 2018) | 250k+ imgs, vertex annotations | Plate det+recog | Large-scale pretraining (Chinese format) → fine-tune on Indian set | github.com/detectRecog/CCPD |

**Why this matters:** generic COCO/BDD models miss Indian realities (mixed traffic, auto-
rickshaws, dense two-wheelers, varied plate fonts). Training/evaluating on **DriveIndia +
IDD + Indian plates** is a concrete, defensible novelty/credibility point for judges.

---

## 6. How this plugs into our pipeline (actionable mapping)
| Our component | Use from this research |
|---------------|------------------------|
| Vehicle/road-user detection | DriveIndia / IDD-Detection to fine-tune RF-DETR or YOLOv12 |
| Helmet non-compliance | AICC Track 5 7-class scheme + Co-DETR/YOLOv8x two-stage |
| Triple riding | Same 7-class model — count D+P1+P2 person boxes per motorbike |
| Class imbalance (rare no-helmet/passenger) | Minority Optimizer / few-shot sampling / synthetic data |
| Tracking (wrong-side, red-light, parking) | track_id labels in AICC + BoT-SORT/ByteTrack |
| Road geometry (stop-line, lanes) | IDD-Segmentation drivable-area/lane masks |
| License plate | CCPD pretrain → Indian-plate fine-tune → PaddleOCR PP-OCRv5 |
| VLM verification/captioning | Precedent: AICC 2025 Track 2 Traffic Video QA |
| Evaluation metric | mAP (matches AICC); report vs ~0.49 helmet baseline + DriveIndia 78.7% mAP50 |

---

## 7. Action items / VERIFY
- [ ] Register at aicitychallenge.org for Track 5 data; confirm research-use license terms.
- [ ] Confirm Co-DETR mAP 0.4860 / Rank 1 from CVPRW'24 paper (IEEE Xplore 10678200).
- [ ] Clone & inspect: SKKUAutoLab/aicity_2024_helmet, vnptai/AI-City-Challenge-2023,
      cmtsai2023/AICITY2023_Track5_DVHRM (check licenses before reuse).
- [ ] Request DriveIndia via TiHAN portal; confirm license + download.
- [ ] Verify IDD download terms (registration/academic use).
- [ ] Decide stills-only vs sampled-video (AICC is 10fps video — sampling gives us stills
      AND optional temporal labels).

## Sources
- [9th AI City Challenge (arXiv 2508.13564)](https://arxiv.org/abs/2508.13564) · [2025 page](https://www.aicitychallenge.org/2025-ai-city-challenge/)
- [8th AI City Challenge (CVPRW 2024 PDF)](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Wang_The_8th_AI_City_Challenge_CVPRW_2024_paper.pdf)
- [Co-DETR helmet winner (IEEE Xplore 10678200)](https://ieeexplore.ieee.org/document/10678200/)
- [SKKU 2024 helmet repo](https://github.com/SKKUAutoLab/aicity_2024_helmet) · [VNPT 2023 repo](https://github.com/vnptai/AI-City-Challenge-2023) · [DVHRM 2023 repo](https://github.com/cmtsai2023/AICITY2023_Track5_DVHRM)
- [Few-shot YOLOv8 (arXiv 2304.08256)](https://arxiv.org/pdf/2304.08256) · [GA-YOLOv5 (arXiv 2304.09248)](https://arxiv.org/abs/2304.09248)
- [DriveIndia (arXiv 2507.19912)](https://arxiv.org/abs/2507.19912) · [TiHAN portal](https://tihan.iith.ac.in/TiAND.html)
- [IDD (arXiv 1811.10200)](https://arxiv.org/pdf/1811.10200) · [IDD portal](https://idd.insaan.iiit.ac.in/) · [INSAAN datasets](https://insaan.iiit.ac.in/datasets/)
- [Indian plate in the wild (arXiv 2111.06054)](https://arxiv.org/pdf/2111.06054) · [CCPD](https://github.com/detectRecog/CCPD)
