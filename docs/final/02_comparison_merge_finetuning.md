# Plan A vs Plan B — Comparison, Best-of-Both Merge & Fine-Tuning Plan
### Gridlock 2.0 R2 · decision document

- **Plan A** = `../plan_a/09_plan_a_solution_design_SUPERSEDED.md` — *paradigm-partitioned, foundation-model, novelty-first.*
- **Plan B** = `../plan_b/` (ARCHITECTURE.md + 4 research docs) — *modular, geometry-aware, build-ready, honesty-first.*

> **Bottom line:** they are **complementary, not competing.** Plan B is the more *credible, feasible, dataset-grounded* spine; Plan A is the *novelty, SOTA, and architectural-framing* layer. The winning submission = **Plan B's grounded skeleton + Plan A's intellectual spine and selective SOTA upgrades.** Neither alone is optimal.

---

## 1. Pros & Cons

### Plan A (root — paradigm-partitioned / foundation-model)
**Pros**
- Strong **intellectual spine**: the 3-paradigm decomposition (instance-attribute / multi-instance-counting / scene-context) is a genuinely clarifying framing judges remember.
- **Cutting-edge SOTA**: LocateAnything-3B/3D (CVPR'26), DINOv2, SAM2, RTMPose, PARSeq, Chain-of-Evidence.
- **Foundation-model data engine** (teacher→distilled student) — a real answer to Indian-data scarcity.
- **Admissibility focus** (calibration, abstain band, tamper-evident, signed evidence) is a differentiator.
- Cleaner separation of concerns (Scene Context Model as reusable prior).

**Cons**
- **Less grounded/feasible**: light on concrete datasets, licenses, realistic accuracy numbers, and a build roadmap.
- **Omits the single most relevant benchmark** — NVIDIA AI City Challenge **Track 5** (helmet+rider 7-class scheme).
- Heavy reliance on a 3B VLM and newest models → **cost / licensing / availability risk** (NVIDIA research license, H100-class).
- Risk of sounding **aspirational** ("we'll distill a foundation model") without a demo path.
- No explicit honest scoping of what needs video vs a still.

### Plan B (another plan — modular / geometry-aware)
**Pros**
- **Build-ready & honest**: concrete model picks, license notes (AGPL/GPL traps), realistic numbers (helmet mAP ~0.5–0.7, not 99%), risk register, time-boxed roadmap, eval protocol table.
- **AICC Track 5 7-class scheme** = one model encoding *helmet compliance AND rider count* → solves helmet + triple-riding together. Excellent, concrete insight.
- **Re-implemented VNPT association heuristic** + per-class thresholds = working IP for "whose helmet / how many on the bike."
- **Geometry-as-config rule engine** = interpretable, auditable, scalable for the temporal violations (annotate fixed camera once).
- **Excellent dataset map** (DriveIndia, IDD, AICC, CCPD, Indian plates, LISA, UA-DETRAC) + repo review + license hygiene.
- Concrete **evidence JSON schema** + analytics/dashboard plan + gap audit.

**Cons**
- **Less novel-sounding**: "RF-DETR + rules + PaddleOCR + VLM verify" reads like solid engineering, not a research contribution.
- The 3-paradigm framing is *implicit*; intellectual story is weaker.
- Uses **2022-era trackers** (ByteTrack/BoT-SORT) and doesn't exploit the newest foundation models / single-frame 3D.
- Association heuristic is hand-tuned (overlap + nearest-center) — brittle vs a learned association (Plan A's SAC/SAM2).
- Doesn't fully answer single-frame ambiguity beyond "needs video."

---

## 2. Differences — Major & Minor

### Major
| Dimension | Plan A | Plan B |
|---|---|---|
| **Organizing principle** | 3 violation **paradigms** + specialized reasoners | Modular components + **geometry-as-config** rule engine |
| **Helmet + triple riding** | SAC association (SAM2 masks) + RTMPose pose, threshold ≥3 | **AICC Track 5 7-class** model + VNPT overlap/nearest-center association |
| **Scene-context violations** | Scene Context Model + single-frame 3D (LocateAnything3D) + tracklets | Per-camera config + **rule engine** over tracks (+signal state) |
| **VLM role** | LocateAnything-3B verifier + offline **teacher/distillation** | Qwen2.5-VL/InternVL3 **verifier + evidence captioner** |
| **Detector** | YOLOv11/12 screen → Co-DETR/RT-DETR (DINOv2) confirm | **RF-DETR** (Apache) / YOLOv12 primary |
| **Grounding maturity** | Vision/novelty, lighter on feasibility | Datasets, licenses, roadmap, realistic numbers |

### Minor
| Dimension | Plan A | Plan B |
|---|---|---|
| Preprocessing | Zero-DCE++ selective + LP-crop restore | Retinexformer + **OneRestore**, quality-gated (pyiqa), ablation planned |
| OCR | PARSeq + Indian grammar | **PaddleOCR PP-OCRv5** (+EasyOCR backup) |
| Tracker | ByteTrack | **BoT-SORT** → ByteTrack |
| Pose | RTMPose / ViTPose | (none) |
| Evidence | signed metadata + insets | concrete **JSON schema** + audit/privacy |
| Licensing | not discussed | **first-class** (AGPL/GPL warnings) |
| Confidence | temperature scaling + abstain band | temperature scaling + **per-class thresholds (VNPT values)** |

*The two are ~80% aligned at the block-diagram level (detect→track→violation-modules→OCR→evidence→analytics, VLM as verifier). They differ mainly in (a) novelty framing, (b) how helmet/triple-riding is implemented, and (c) feasibility grounding.*

---

## 3. Which gives better results? (probability of success)

| Criterion | Better plan | Why |
|---|---|---|
| **Demo-able in hackathon timeframe** | **B** | concrete datasets, repos, roadmap, working association IP |
| **Benchmark accuracy on helmet/triple** | **B** | AICC Track 5 scheme + winning patterns (two-stage, class-imbalance, ensembles) are proven |
| **Judge "wow"/novelty score** | **A** | paradigm framing + CVPR'26 SOTA + admissibility story |
| **Robustness to scrutiny / honesty** | **B** | realistic numbers, license hygiene, gap audit |
| **Scalability & interpretability narrative** | **tie** | A: data-engine + SCM; B: geometry-as-config |
| **Risk of failure** | **B lower** | A risks over-promising on foundation-model distillation |

**Verdict:** For *actually winning*, **Plan B is the safer, higher-probability spine** (it will demo and the numbers will hold up), and **Plan A supplies the differentiation** that lifts it above a "good helmet detector." A merge dominates either alone: highest expected score at acceptable risk.

---

## 4. Best-of-Both — the merged architecture (recommended)

Keep **Plan B's skeleton and discipline**; graft on **Plan A's framing and selective upgrades**.

**Adopt from B (the spine):**
1. **AICC Track 5 7-class helmet/rider model** as the helmet+triple-riding core (two-stage detect→crop→classify; per-class thresholds; class-imbalance handling).
2. **Geometry-as-config rule engine** for red-light / wrong-side / stop-line / parking (interpretable, auditable, fixed-camera annotate-once).
3. **Concrete dataset map + license hygiene + realistic eval table + roadmap + risk register.**
4. **PaddleOCR PP-OCRv5** ANPR (CCPD pretrain → Indian fine-tune), **evidence JSON schema**, **Streamlit/FastAPI analytics**.
5. **Honest scoping** (still vs needs-clip; daytime seatbelt) and the **VERIFY discipline**.

**Graft from A (the lift):**
6. **Reframe the whole thing with the 3-paradigm spine** — it makes B's components feel like a designed system, not a bag of models. (Instance-attribute = helmet/seatbelt; counting = triple riding; scene-context = the geometry rule engine.)
7. **Upgrade association**: replace the brittle overlap/nearest-center heuristic with **SAM2-mask + pose (RTMPose) SAC-style association** as the high-accuracy path; keep the heuristic as the fast fallback.
8. **Single-frame 3D (LocateAnything3D / monocular 3D yaw)** to *strengthen* wrong-side & stop-line where only a still exists — answers B's "needs video" caveat partially.
9. **Foundation-model data engine** (LocateAnything-3B / GroundingDINO / Grounded-SAM-2) to **auto-label** Indian footage for rare classes and **distill into the fast detector** — directly cuts fine-tuning data cost.
10. **Chain-of-Evidence + admissibility** (calibration, abstain band, tamper-evident signed evidence) layered on B's JSON schema.

**Merged pipeline (one line):**
`Ingest+IQA → (gated) restore → detector [YOLOv12/RF-DETR, Indian-FT] → BoostTrack → {7-class helmet/rider + SAC association | geometry rule engine + signal-state | windshield→seatbelt | ROI-gated ANPR} → confidence cascade (per-class + calibration + abstain) → VLM verify/caption (only low-conf) → signed evidence JSON + annotated image → store → analytics`

---

## 5. Architecture alternatives we (both plans) missed

| Alternative | What it is | Where it helps | Verdict |
|---|---|---|---|
| **BoostTrack / BoostTrack++** (2024, MVA/arXiv 2408.13003) | Boosts detection-confidence + similarity; **HOTA 69.25 > ByteTrack 67.68** on MOT17 | Drop-in tracker upgrade (via **boxmot**) for all temporal violations | **Adopt** — strictly better than ByteTrack/BoT-SORT, same interface |
| **boxmot** (mikel-brostrom) | Pluggable SOTA trackers (ByteTrack/BoT-SORT/BoostTrack/OC-SORT), OBB support | Swap trackers without rewrite | **Adopt** as the tracking abstraction |
| **FishEye8K + AICC 2025 Track 4** | Fisheye road-object detection (5 classes, 157k boxes) | Indian junction CCTV is often **fisheye/wide-angle** → train/eval here | **Add** to dataset map if any fisheye cameras |
| **Agentic / multi-agent VLM** (VLMLight, INSIGHT, traffic-QA 2025) | VLM → scene text; agents do mode-select + **rule-compliance verification** | An *alternative* to the rule engine for ambiguous scenes; explainability | **Selective** — use as the verifier's reasoning pattern, not the primary (cost) |
| **Self-reflective VLM for plate+make+model** (arXiv 2508.01387) | VLM reads plate **and** vehicle make/model with self-reflection | Richer vehicle attributes for evidence/ReID; OCR cross-check | **Optional** evidence enrichment |
| **End-to-end LPR**: YOLOv5s+LPRNet+Triplet-Attn (7.5M, 147 FPS), **EdgeFormer-LPR**, **Relaxed-Syntax Transformer** (ICDAR'25) | Lightweight / future-proof plate recognition incl. **new HSRP formats** | Faster ANPR + robustness to format changes; **synthetic LP-2025** data | **Consider** transformer-syntax model for HSRP generalization |
| **CityFlow ReID** | Multi-camera vehicle re-identification | **Repeat-offender** tracking across cameras (analytics novelty) | **Optional stretch** |
| **Synthetic data** (Stable Diffusion for rare helmet/night; LP synth) | Generate rare-class/edge-case training data | Class imbalance (no-helmet, night, rare plates) | **Adopt** for imbalance — AICC winners used it |
| **Copy-paste / minority oversampling, focal loss** | Class-imbalance training tricks | Rare classes (P2 passenger, no-helmet) | **Adopt** in fine-tuning |

---

## 6. Do we need to fine-tune? — analysis + GPU compute plan

**Short answer: yes, fine-tuning is required for competitive accuracy.** Zero-shot open-vocab models (YOLO-World/GroundingDINO/LocateAnything) are great for *cold-start, auto-labeling, and demo*, but will underperform on the hard cases (no-helmet, small/distant riders, Indian plates, Indian signals). **Everything is transfer learning from public checkpoints — no training from scratch.**

### 6.1 What needs fine-tuning vs used as-is

| Model | Fine-tune? | Data | Why |
|---|---|---|---|
| Core detector (YOLOv12 / RF-DETR) | ✅ **Yes (essential)** | DriveIndia 67k + IDD 40k | Indian classes (auto-rickshaw, cart), domain shift |
| Helmet/rider 7-class (2-stage) | ✅ **Yes (essential)** | AICC Track 5 | The core violation model; rare classes |
| Plate detector | ✅ Yes | Indian plates 16k | Indian plate size/placement |
| Plate OCR (PARSeq / PP-OCRv5 rec) | ✅ Yes | CCPD-pretrained → Indian 21k (+synthetic) | Indian charset/HSRP fonts |
| Signal-state classifier | ✅ Yes (light) | LISA 43k (+Indian signals) | Indian signal layouts/countdowns |
| Seatbelt (windshield + belt) | ✅ Yes (light) | ~12k/10k sets | Domain-specific, daytime |
| Preprocessing (Retinexformer/OneRestore/Zero-DCE) | ❌ Use pretrained | — | Generic restoration |
| Tracker (BoostTrack/ByteTrack) | ❌ No training | — | Detection-driven, no learning |
| Pose (RTMPose) | ❌ Pretrained (optional light FT) | COCO | Generic person keypoints |
| SAM2 | ❌ Use as-is | — | Promptable segmentation |
| VLM (Qwen2.5-VL / LocateAnything-3B) | ⚙️ **Optional LoRA** | ~2–5k instruction pairs | Only for evidence-caption style / verification tuning; zero-shot works |

### 6.2 GPU-compute estimate (transfer learning, mixed precision bf16)

*Assumptions: rented A100 80GB (or 4090 24GB for small models); pretrained checkpoints; numbers are engineering estimates, ±50%.*

| Model | Resolution | Epochs | GPU | Est. GPU-time | VRAM |
|---|---|---|---|---|---|
| Detector **RF-DETR-L** (Apache, primary) | 560–800 | 60–120 | 1× A100 | **2–4 days** | 40–80 GB |
| (AGPL benchmark-only alt) YOLOv12-L | 640–960 | 80–150 | 1× A100 | 1–2 days | 24–48 GB |
| Helmet Stage-1 (motorbike, 1-cls) | **1280–1536** | 50–100 | 1× A100 | **0.75–1.5 days** | 24–40 GB |
| Helmet Stage-2 (7-cls on crops) | 320–512 | 60–120 | 1× A100 | **0.5–1 day** | 16–24 GB |
| Plate detector | 640 | 50–100 | 1× 4090/A100 | 0.25–0.5 day | 12–24 GB |
| Plate OCR recognizer | 32×128 | 20–50 | 1× 4090/A100 | 0.5–1 day | 12–24 GB |
| Signal-state classifier | 224 / 640 | 30–50 | 1× 4090 | 0.25–0.5 day | 8–16 GB |
| Seatbelt (windshield+belt) | 640 / 224 | 30–50 | 1× 4090 | 0.25–0.5 day | 8–16 GB |
| (opt) VLM QLoRA (Qwen2.5-VL-7B) | — | 2–3 | 1× A100 80GB (or 24GB 4-bit) | 0.5–1 day | 24–48 GB |

**Totals (incl. HP tuning / restarts / ablation ≈ ×1.5–2):**
- **Competitive MVP** (RF-DETR detector + helmet ×2 + plate det + plate OCR): **~6–8 A100-GPU-days** *(RF-DETR primary adds ~1–2 days vs the YOLO alt)*.
- **Full system** (+ signal + seatbelt + optional VLM LoRA): **~11–16 A100-GPU-days**.
- **Wall-clock:** ~3–4 days (MVP) / ~6–8 days (full) on **2× A100** running in parallel.

**Cost (cloud):** A100 80GB ≈ $1.8–3.5/hr on-demand; **$1.0–1.8/hr spot** (RunPod/Lambda/Vast).
- MVP ≈ 6–8 GPU-days → **~$260–700**. Full ≈ 11–16 GPU-days → **~$450–1,300**.
- **Free-tier path:** Kaggle (2× T4, 30 GPU-h/week) + Colab can train the *small* models and a reduced MVP, ~3–5× slower; **rent A100s for the detector + helmet high-res** runs (the only expensive ones).

### 6.3 Fine-tuning method (per family)
- **Detectors:** load COCO/Objects365-pretrained → replace head for class count → train with strong aug (mosaic, mixup, copy-paste, HSV) **+ weather/blur aug** synthesized from IDD-AW stats; cosine LR + warmup + EMA; **focal loss / minority oversampling** for rare classes (no-helmet, P2). Validate mAP@.5 / .5:.95 per class.
- **OCR:** start from **CCPD-pretrained** recognizer → fine-tune on Indian 21k + **synthetic HSRP plates** (TextRecognitionDataGenerator); CTC or attention decoder; metric = full-plate accuracy + CER.
- **VLM (only if doing LoRA):** **QLoRA** 4-bit, rank 16–64, ~2–5k instruction/caption pairs; fits a 7B on 24 GB. Otherwise use zero-shot prompting — recommended for the hackathon.
- **Label efficiency:** use the **foundation-model data engine** (GroundingDINO / Grounded-SAM-2 / LocateAnything-3B) to auto-label rare/custom classes, human-spot-check, then fine-tune — cuts manual annotation sharply.

### 6.4 Priority order (if compute/time is tight)
1. **Detector on DriveIndia** (everything depends on it).
2. **Helmet/rider 7-class on AICC** (the headline demo + 2 violations).
3. **Plate detector + OCR (Indian)** (completes the e-challan story).
4. Signal-state + geometry rule engine (red-light/stop-line/wrong-side/parking — mostly config, light training).
5. Seatbelt (daytime, best-effort).
6. (Optional) VLM LoRA — skip if short on time; zero-shot is fine.

---

## 7. Recommended next actions
1. Adopt the **merged architecture (§4)** as the canonical design; refactor Plan A's doc to absorb B's AICC scheme, geometry rule engine, dataset map, license notes, and realistic eval table.
2. Lock the **fine-tuning priority (§6.4)** and reserve ~**2× A100 for ~1 week** (or staged spot rentals).
3. Run the **restoration ablation** (B) early — decide if preprocessing stays.
4. Build the **MVP demo**: detect → BoostTrack → 7-class helmet+triple → ANPR → signed evidence JSON + annotated image, on AICC/DriveIndia samples.
5. Keep the **VERIFY discipline** — confirm every quoted number at source before the submission.
