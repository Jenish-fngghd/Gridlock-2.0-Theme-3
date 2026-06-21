# Gridlock 2.0 — R2 · Design Justifications & Defenses
### Companion to `00_master_design.md` (canonical) — supersedes the Plan A companion `../plan_a/09_plan_a_solution_design_SUPERSEDED.md`

This file collects the **standalone rationales** for every non-obvious design choice — the "why did you do it this way and not the obvious alternative?" questions a judge will ask. Keeping them here keeps the main Solution Design focused on the architecture itself.

**Convention:** each justification is numbered `J#`, tagged with the design section it defends, and written as *Question → Answer*. New justifications are appended here (not in the main doc).

## Index

| ID | Defends | Question |
|---|---|---|
| **J1** | §3.1 Architecture choice | Why not just run LocateAnything-3B on every image (drop Stage-1 YOLO)? |
| **J2** | §3.4 Helmet + triple riding | Why anchor on the AICC 7-class scheme instead of pure SAC/pose? |
| **J3** | §3.6 Scene-context | Why a geometry-as-config rule engine instead of a learned end-to-end model? |
| **J4** | §8 Fine-tuning | Why fine-tune at all — can't we just use zero-shot open-vocab models? |
| **J5** | §3.3 Tracking | Why BoostTrack instead of ByteTrack / BoT-SORT? |
| **J6** | §3.2 Detection | Why RF-DETR (Apache) as the primary detector instead of YOLOv12? |

---

## J1 · Why not just run LocateAnything-3B on every image (drop Stage-1 YOLO)?
**Defends:** §3.1 — two-stage compute cascade · **Verdict:** No. The VLM is kept for the role it wins (teacher + verifier), **not** as the per-image screen.

1. **Economics break scalability (the decisive one).** LocateAnything-3B runs at ~12–25 *boxes*/s on an H100 → **~1 image/s** for a dense Indian scene; YOLOv11/12 runs at **~300–1000 images/s** (batched, TensorRT) → **~200–1000× cheaper per image**. At a realistic **1M candidate images/day**, the VLM ≈ 12 H100s 24/7 ≈ **$300–400k/yr**; at **10M/day** ≈ **$3–4M/yr**. YOLO does the same screen for a few dollars/day. "Scalable" in this brief *means* cost-per-image — the VLM blows it up.
2. **Cascade logic: most frames are negatives.** Stage-1 exists to cheaply discard the 90%+ of images with nothing actionable so the expensive model only sees candidates. Running the heavy model on everything throws away the very efficiency that makes the system scalable (same principle as Viola-Jones → modern two-stage detectors).
3. **Admissibility wants a deterministic detector.** A generative VLM can hallucinate and is harder to calibrate to a fixed PR operating point, version, and defend in court. A frozen YOLO checkpoint with a published PR curve is far more defensible for an auto-challan; the VLM is therefore a *second opinion*, not the sole arbiter.
4. **Wrong tool for fine-grained work.** The VLM emits boxes from prompts — it still needs RTMPose (pose/counting), PARSeq (HSRP OCR) and the helmet/seatbelt classifiers downstream, so it does not actually collapse the pipeline.

**Reconciliation — we *do* use it:** its open-vocab capability is **distilled into the Stage-1 YOLO student** (offline Data-Engine), so Stage-1 is effectively "LocateAnything-3B compressed, ~500× faster," and the VLM itself runs only on the **thousands** of flagged candidates/day (verifier), not the **millions** of raw images. *If* open-vocabulary behavior is wanted at Stage-1, use a lightweight open-vocab detector (**YOLO-World** / efficient RT-DETR) — not the 3B model.

*(Throughput/cost figures are order-of-magnitude estimates; the 2–3 orders-of-magnitude gap is robust to reasonable assumptions on hardware and scene density.)*

---

## J2 · Why anchor on the AICC 7-class scheme instead of pure SAC/pose?
**Defends:** §3.4 — helmet + triple riding · **Verdict:** Use the 7-class scheme as the **labelled training target**; use SAC/pose as the **association upgrade on top** — they are complementary, not alternatives.

1. **It solves two violations with one model.** `motorbike + D/P1/P2 ×{Helmet,NoHelmet}` encodes per-rider helmet compliance *and* rider count, so **helmet non-compliance and triple riding** fall out of a single detector — no separate counting head.
2. **It comes with the most relevant public dataset + proven recipes.** AICC Track 5 (100 annotated videos, track_ids) plus open winning methods (two-stage detect→crop→classify, per-class thresholds, minority-class handling) give us labelled data and a baseline to beat — Plan A's pure-SAC approach had *no* dataset anchor.
3. **Honest, defensible numbers.** AICC leaderboards (~0.49–0.70 mAP) let us set realistic expectations instead of overclaiming.
4. **SAC/pose still adds value** where the 7-class boxes are ambiguous (dense overlap, occluded pillion): SAM2 masks + RTMPose give a robust, learned rider↔bike↔head association, with the overlap/nearest-x heuristic as a cheap fallback. So we keep the best of both.

## J3 · Why a geometry-as-config rule engine instead of a learned end-to-end model?
**Defends:** §3.6 — scene-context violations · **Verdict:** Geometry + rules over a black-box, because enforcement cameras are **fixed**.

1. **Feasibility:** there is **no public dataset** for stop-line / wrong-side on Indian roads; a supervised end-to-end model would have nothing to train on. Geometry needs only detection + tracking + a one-time per-camera annotation.
2. **Interpretability & admissibility:** a rule ("vehicle crossed stop-line polygon while signal=red") yields an auditable, court-defensible evidence chain; a learned score does not.
3. **Scalability:** cameras are fixed → zones/lines/directions are **config, annotated once per camera**, reused across millions of frames at ~zero marginal cost.
4. **We still add ML where it helps:** signal-state classifier (LISA), and **single-frame monocular-3D** (LocateAnything3D) to disambiguate heading/position when only a still exists. So the engine is geometry-first but ML-assisted.

## J4 · Why fine-tune at all — can't we just use zero-shot open-vocab models?
**Defends:** §8 — fine-tuning · **Verdict:** Zero-shot for cold-start/auto-labeling/demo; **fine-tune for competitive accuracy.**

1. **Hard cases break zero-shot.** No-helmet vs helmet on a 20-px head, small/distant riders, Indian plate fonts/HSRP, Indian signal layouts — open-vocab detectors (YOLO-World/GroundingDINO/LocateAnything) underperform here; the benchmark numbers that judges score come from fine-tuned models.
2. **It's cheap (transfer learning, not from scratch).** All checkpoints are pretrained; fine-tuning the competitive MVP is **~5–7 A100-GPU-days (~$215–600 spot)**. (Full plan → `02_comparison_merge_finetuning.md` §6.)
3. **Zero-shot still earns its place** — as the **data engine**: auto-label rare/custom Indian classes, human-spot-check, then fine-tune/distill. So zero-shot *reduces* the labeling cost of fine-tuning rather than replacing it.

## J5 · Why BoostTrack instead of ByteTrack / BoT-SORT?
**Defends:** §3.3 — tracking · **Verdict:** Strictly-better drop-in; keep ByteTrack/BoT-SORT as fallback.

1. **Better accuracy, same interface.** BoostTrack/BoostTrack++ (2024) report **HOTA 69.25 vs ByteTrack 67.68** on MOT17 (VERIFY), with stronger ID consistency under occlusion — exactly the failure mode in dense two-wheeler traffic.
2. **No rewrite cost.** Via the **boxmot** library, trackers are pluggable, so we can swap/ablate BoostTrack ↔ BoT-SORT ↔ ByteTrack without changing the pipeline.
3. **ID stability matters downstream.** Wrong-side/red-light/parking all depend on persistent IDs across frames; fewer ID switches = fewer false violations.

## J6 · Why RF-DETR (Apache) as the primary detector instead of YOLOv12?
**Defends:** §3.2 — detection · **Verdict:** RF-DETR is the shippable spine; YOLOv12 is benchmark-only. The deciding factor is **license**, and the accuracy gap is negligible-to-favourable.

1. **License (decisive).** **YOLOv12 and Ultralytics YOLO are AGPL-3.0** (✓ verified) — copyleft that forces source disclosure if we distribute. **RF-DETR is Apache-2.0** (✓), so our submission/codebase stays clean for commercial or public release. An enforcement product that legally can't ship is worthless.
2. **Accuracy is not a sacrifice.** RF-DETR is the **first real-time model to pass 60+ mAP on COCO** (✓), with a **DINOv2** backbone that adapts well to small, occluded, domain-shifted data — exactly Indian traffic. It comes in 6 sizes (Nano→2XL), so one Apache architecture covers *both* the fast screen and the heavy confirm.
3. **YOLOv12 still earns a role.** It's strong (40.6% mAP-N @1.64 ms T4 ✓) and useful for **internal benchmarking/ablation** — just never inside the distributed artifact.
4. **Co-DETR remains the high-recall confirmer** (AICC helmet rank-1 backbone), used offline on candidates where recall matters most.
