# Winning Repos Review — AICC Track 5 Helmet Violation

**What I did:** Shallow-cloned the 3 open-source AI City Challenge Track-5 winning
solutions into `./repos/` and read their pipelines, configs, and core code to identify
**what we can reuse**.

**Cloned:**
- `repos/AI-City-Challenge-2023/` — **VNPT AI** (custom tracking + object association). ⭐ most reusable logic
- `repos/aicity_2024_helmet/` — **SKKU Automation Lab** (clean 2-stage YOLOv8 pipeline + config-driven)
- `repos/AICITY2023_Track5_DVHRM/` — **cmtsai** (YOLOv7-E6E + attention; ranked 4/5/6 in 2023)

> ⚠️ **LICENSE WARNING (read before reusing code):**
> - `AICITY2023_Track5_DVHRM` = **GPLv3** (it's a YOLOv7 fork) → copyleft; reuse forces GPL.
> - `AI-City-Challenge-2023` and `aicity_2024_helmet` = **no LICENSE file** → default "all
>   rights reserved." We may **study and re-implement the ideas**, but should NOT copy code
>   verbatim into our submission without permission. **Re-implement the logic ourselves.**
> - Safe path for us: learn the *approach*, write our own clean implementation on
>   Apache/MIT-licensed models (RF-DETR, PaddleOCR, Ultralytics-under-AGPL-with-care).

---

## 1. The shared winning pattern (all 3 converge on this)
```
Stage 1: DETECT motorbike(s)            (1-class detector, high-res ~1536–1920px)
Stage 2: CROP motorbike region          (zoom in on small/distant riders)
Stage 3: CLASSIFY/DETECT helmet states  (7–9 class detector on the crop)
Stage 4: ASSOCIATE head→rider→motorbike (geometry/overlap rules)  ⭐ key IP
Stage 5: TRACK + temporal smoothing     (stabilize across frames, drop flicker)
Stage 6: PER-CLASS confidence thresholds (tuned individually per class)
```
**Takeaways for our pipeline:**
- **Two-stage (detect→crop→classify) is essential** — riders are small; full-frame
  detection alone underperforms. We should adopt this.
- **Per-class confidence thresholds** matter a lot (see VNPT values below) — rare classes
  (P2) get *lower* thresholds so they aren't suppressed.
- **High input resolution** (1536–1920) is consistently used — these are HD traffic frames.

---

## 2. ⭐ Reusable IP: rider–helmet–motorbike association (VNPT)
Files: `repos/AI-City-Challenge-2023/object_association/{object_association.py, detection_object.py}`

This is the most valuable, hard-to-reinvent logic. Three object classes with overlap-based
attachment:

- **`Motor`** — has a `combined_box` that expands to envelop its attached riders.
- **`Human`** (driver/P1/P2) — `attach_motor_id()`: assigned to a motorbike if
  `overlap_ratio(human, motor) > 0.3`; expands the motor's `combined_box`.
- **`Head`** — `attach_head_id()`: nearest head to a human by x-center distance among
  overlapping heads → sets `human.wear_helmet = head.is_helmet`.

**Why it matters for us:** this solves "which helmet belongs to which rider, and which
rider belongs to which bike" — exactly the association needed for **helmet non-compliance**
AND **triple riding** (count Humans attached to one Motor). We should **re-implement this
overlap+nearest-center heuristic** (it's a simple, license-safe algorithm to rewrite).

Key constants observed:
- human→motor overlap threshold: **0.3**
- head→human overlap threshold: **0.6**
- combined_box expansion: **5%** of width/height

## 3. Per-class confidence thresholds (VNPT `infer.py`) — reuse as starting values
| Class | Threshold |
|-------|-----------|
| motorbike | 0.35 |
| DHelmet / DNoHelmet | 0.32 |
| P1Helmet / P1NoHelmet | 0.32 |
| P2Helmet / P2NoHelmet | **0.20** (lower — rarer class) |

→ Lesson: tune thresholds **per class**, lower for rare classes. Directly usable as our
defaults.

## 4. SKKU 2024 pipeline (cleanest architecture to imitate)
Files: `repos/aicity_2024_helmet/{main.py, configs/helmet.yaml}`
- **Config-driven** (YAML) design — nice template for our own modular config.
- Stage 1 detector: `yolov8x` @ **1536px, 1 class** (motorbike).
- Stage 2 identifier: **ensemble** of `yolov8x` @ 320/448/512px, **9 classes**, multi-group.
- **K-means clustering** post-step (`kmeans_cluster/kmeans_model.pkl`) for temporal
  consistency instead of a full tracker.
- Optional **face detector** (`yolov8l-face`) as a heuristic to confirm head/helmet region.
- Note: requirements include `pyiqa`/`piqa` (image-quality assessment) and `filterpy`
  (Kalman) — useful libs for our preprocessing-quality-gate and tracking ideas.

**Borrowable ideas (re-implement):** YAML-config modular stages; multi-resolution ensemble
for the classifier; image-quality assessment to gate preprocessing.

## 5. cmtsai DVHRM (2023, rank 4–6) — attention-augmented detector
File: `repos/AICITY2023_Track5_DVHRM/README.md`
- Trains **YOLOv7-E6E** + **CBAM** + **SimAM** attention variants as 7-class helmet
  detectors at 1280–1920px, then ensembles.
- Includes handy data utilities: `GTxywh2yolo.py` (convert AICC GT → YOLO format),
  `extract_*_frames.sh` (video→frames). **These conversion scripts are the practically
  useful part** (but GPLv3 — re-implement the trivial format conversion ourselves).
- Confirms the **7-class scheme** and that attention modules give measurable gains.

## 6. Headline numbers (set expectations)
| Solution | Year | Result (VERIFY) |
|----------|------|-----------------|
| IC_SmartVision (per VNPT repo leaderboard) | 2023 | **mAP 0.6997** |
| VNPT custom tracking | 2023 | top-tier |
| cmtsai DVHRM | 2023 | ranks 4/5/6 |
| Co-DETR + Minority Enhancement | 2024 | Rank 1, mAP 0.4860 |

> 2023 (0.69) vs 2024 (0.48) differ because the **2024 test set was harder** (different
> data/eval). Lesson for judges: report the metric *with its dataset/year context*, and
> expect real-world helmet mAP in the **~0.5–0.7** band, not 99%.

---

## 7. Our action plan from this review
1. **Adopt two-stage detect→crop→classify** as our helmet/triple-riding architecture.
2. **Re-implement the association heuristic** (Motor/Human/Head overlap + nearest-center)
   ourselves — license-safe, ~100 lines, high value.
3. **Use the per-class threshold table** as starting defaults.
4. **Use a YAML-config modular design** like SKKU for our prototype.
5. **Re-implement GT→YOLO conversion + frame extraction** (trivial, avoid GPL copy).
6. Build on **Apache/MIT models** (RF-DETR, PaddleOCR) to keep our submission license-clean;
   if using Ultralytics YOLO, note its AGPL-3.0.
7. Add our **novelty on top**: VLM verification layer + Indian-dataset fine-tuning +
   geometry module for the temporal violations (none of these repos do plate OCR, analytics,
   or non-helmet violations — that's our differentiation).

## 8. What these repos do NOT cover (our opportunity)
- ❌ License plate detection/OCR — none of them do ANPR.
- ❌ Other violations (seatbelt, wrong-side, stop-line, red-light, illegal parking).
- ❌ Preprocessing/restoration for weather/low-light.
- ❌ Evidence generation, metadata store, analytics/reporting dashboard.
- ❌ VLM-based verification/captioning.
→ These gaps = exactly where our solution adds value beyond a known helmet baseline.

---
*Repos live in `./repos/` (shallow clones). Treat as read-only reference. Re-implement,
don't copy, given the license situation in the warning above.*
