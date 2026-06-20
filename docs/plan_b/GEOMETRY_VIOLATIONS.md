# Geometry / Temporal Violations + Seatbelt — Research & Gap Audit

**Why this doc:** These are the violations our earlier research flagged as HARD from a
single still image — they need **scene geometry (zones/lines), traffic-signal state, and/or
motion across frames (tracking)**. None of the AICC helmet repos cover them, so this is our
thinnest area. This doc gives the approach, the (often missing) datasets, and a fallback,
then audits what's still under-attended across the whole pipeline.

**Compiled:** June 2026 (web search). Metrics = **VERIFY** at source before quoting.

> **Core honest message for judges:** still images alone cannot prove red-light / wrong-side
> / stop-line violations — you need either a **short video clip** or **camera geometry +
> signal state**. Our design treats these as a **geometry + tracking module** layered on the
> detector, and we demo them on sampled video frames (AICC / UA-DETRAC give us frames *with*
> tracking labels).

---

## 1. Red-Light Running Violation
- **Needs:** (a) traffic-light **state** (red/amber/green), (b) a **stop-line / crossing
  zone**, (c) vehicle **track** crossing the zone while signal = red.
- **Approach:** detect+classify signal state → detect+track vehicles (YOLO + SORT/ByteTrack)
  → trigger if a tracked vehicle crosses the stop-line polygon during a red frame.
- **Datasets (good news — these exist):**
  | Dataset | Content | Note |
  |---------|---------|------|
  | **LISA Traffic Light** | 43,007 frames, 113,888 lights, 7 states (go/warning/stop + left variants) | San Diego; the standard benchmark |
  | **Bosch Small Traffic Light (BSTLD)** | small/distant lights | good for hard small-light case |
  | **DriveU (DTLD)** | large German TL dataset | scale + diversity |
  | **DualCam** (arXiv 2209.01357) | fine-grained real-time TL benchmark | newer |
- **Pick:** signal-state classifier trained on **LISA** (+ BSTLD for small lights) → fuse
  with vehicle tracker + manually-annotated stop-line polygon per camera.
- **Caveat:** signal state must be from the **same scene/timestamp**; works on video, not a
  random still. Indian signals (different layouts, countdown timers) → fine-tune needed.

## 2. Wrong-Side / Wrong-Way Driving
- **Needs:** per-vehicle **direction of motion** vs the lane's **allowed direction**.
- **Approach:** (a) **tracking-based** — track vehicles (ByteTrack/BoT-SORT), compute
  trajectory vector, compare to allowed direction for that lane/ROI; OR (b) **optical-flow**
  — Lucas-Kanade/Farneback flow vs a learned "normal flow" model (Mixture-of-Gaussians).
  2024 SOTA = YOLOv4 + Lucas-Kanade optical flow.
- **Datasets:** **No dedicated public benchmark.** Approaches improvise:
  - Use **UA-DETRAC / CityFlow** (tracking benchmarks) + manually define allowed direction
    per ROI to derive wrong-way labels.
  - Some papers evaluate on **UCSD Ped1** (anomaly) — weak proxy.
- **Pick:** tracking-direction method on UA-DETRAC tracks + per-lane direction annotation.
  Optical-flow as backup for crowded scenes.
- **Caveat:** needs ≥2 frames (motion). A single still cannot establish direction → frame it
  as the "requires short clip" violation.

## 3. Stop-Line Violation (stopping past the line, esp. at red/crossing)
- **Needs:** the **stop-line geometry** + vehicle front-wheel/bbox position relative to it
  (+ usually signal state for "stopped past line on red").
- **Approach:** detect stop-line via **Hough Transform / RANSAC / least-squares line
  fitting** on edges (or annotate it once per fixed camera) → define a polygon zone → flag a
  tracked vehicle whose bbox crosses the line when it shouldn't.
- **Datasets:** **No dedicated public dataset.** It's geometry-driven: combine a vehicle
  detector + **per-camera annotated stop-line** (cameras are usually fixed → annotate once).
  IDD-Segmentation lane/road masks help localize road markings.
- **Pick:** fixed-camera annotated stop-line polygon + vehicle tracker + (optional) signal
  state. Pure-geometry, minimal ML beyond detection.

## 4. Illegal Parking
- **Needs:** a **no-parking zone polygon** + vehicle **dwell time** (temporal persistence).
- **Approach:** detect+track vehicles → if a track stays inside the no-parking polygon
  beyond a time threshold → flag. (Polygon zone + temporal-persistence criterion is the
  standard 2024 recipe.)
- **Datasets:**
  - **i-LIDS** (UK gov sterile-zone/parking benchmark) — includes a no-parking scenario.
  - Several small custom sets (~2,348 imgs in one paper).
  - Open GitHub baselines: Kevinjoythomas/Illegal-Parking-Detection,
    prakharninja0927/illegal-Parking-Detection, sakibreza/Traffic-Rules-Violation-Detection.
- **Pick:** detector + tracker + annotated no-parking polygon + dwell-time threshold. Easy
  to demo qualitatively even without a big dataset.

## 5. Seatbelt Non-Compliance (also "hard" — in-cabin visibility)
- **Needs:** see through **windshield** → locate **driver region** → classify belt on/off.
- **Approach (two-stage, standard):** YOLOv11/YOLOv7 detect windshield → crop driver area →
  CNN/CNN-SVM/DenseNet classify seatbelt. One pipeline: 8.2 ms/image (122 FPS).
- **Datasets (better than expected):**
  - One study: **~12,000 windshield** images + **~10,000 seatbelt** classification images.
  - **AICC 2024 Track 3** (Naturalistic Driving): 594 clips, ~90 hrs, 99 drivers — in-cabin
    behavior (distraction/phone/belt) — strong reference for driver-monitoring.
  - Reported accuracies up to ~99% (windshield) / ~99.1% (CNN-SVM belt) — **VERIFY**;
    these are often controlled conditions, not night/tinted-glass enforcement cameras.
- **Pick:** two-stage windshield→driver→belt classifier; flag honestly that **glare, tint,
  night, and rear occupants** are real failure modes. Hardest violation to do reliably from
  roadside images.

---

## 6. Supporting tracking/ReID datasets (enable §1–4)
| Dataset | Content | Use |
|---------|---------|-----|
| **UA-DETRAC** | 100 seqs, 140k+ frames, 1.21M boxes, weather/occlusion/vehicle-type | detection + MOT for direction/zone violations |
| **CityFlow** | 40 cameras, 10 intersections, 200k+ boxes, multi-cam ReID | multi-camera tracking; plate-independent vehicle ReID |
| **BDD100K (MOT)** | 100k clips, weather/time | tracking + robustness |
| **AICC Track 5** | 100 helmet videos w/ track_id | free tracking labels for our demo |

---

## 7. Design pattern that unifies ALL geometry violations (our framework)
```
Detector (vehicles, lights)  →  Tracker (ByteTrack/BoT-SORT, persistent IDs)
        │
        ├─ Scene config (annotated ONCE per fixed camera):
        │     • lane direction vectors   • stop-line polygon
        │     • no-parking polygon        • signal-light ROI
        │
        └─ Rule engine per track:
              red-light   = crosses stop-line zone while signal=red
              wrong-side  = trajectory vector opposes lane direction
              stop-line   = bbox crosses line when not permitted
              parking     = dwell-time in no-park polygon > threshold
```
**Key efficiency insight:** enforcement cameras are **fixed**, so zones/lines/directions are
annotated **once per camera** (config, not ML). This makes geometry violations very feasible
— the ML is just detection+tracking; the violation logic is geometry + time. Great story for
judges (scalable, interpretable, auditable).

---

## 8. ⚠️ GAP AUDIT — what we've UNDER-attended so far
Ranked by how much attention they still need:

| Area | Attention so far | Risk | Action |
|------|------------------|------|--------|
| **Evidence generation** (annotated image + metadata schema + audit trail) | 🔴 Low | It's a *required deliverable* and easy points; judges want to SEE evidence output | Define JSON schema + annotation renderer in prototype |
| **Analytics & reporting** (stats, trends, searchable records, dashboard) | 🔴 Low | Required deliverable; differentiator if done well | Plan a simple dashboard (Streamlit) + DB schema |
| **Stop-line & wrong-side** datasets | 🟡 Partial | No public benchmark → must self-annotate ROIs | Use fixed-camera per-ROI annotation; document honestly |
| **Seatbelt** reliability | 🟡 Partial | Overclaiming risk (night/tint) | Scope as "best-effort, daytime"; cite 12k/10k datasets |
| **Image preprocessing** real impact | 🟡 Partial | Restoration may not help detection; unproven in our context | Run ablation: detector w/ vs w/o restoration on BDD night/rain |
| **Performance evaluation protocol** (per-component metrics, efficiency) | 🟡 Partial | Judges score on rigor | Lock metric+dataset per component (table) before building |
| **Confidence calibration** (assign trustworthy confidence scores) | 🔴 Low | Problem statement explicitly asks for confidence scores | Temperature scaling / per-class thresholds (have VNPT values) |
| **License-plate → owner linkage / privacy** | 🔴 Low | Legal/ethical; data-protection angle impresses judges | Add privacy note: hashing, access control, retention |
| **Edge vs cloud deployment / scalability numbers** | 🟡 Partial | "Scalability" is in the objective | Give throughput estimate + edge/cloud split |
| **Multi-camera / vehicle ReID** (track same vehicle across cams) | 🔴 Low | Not required, but CityFlow enables a novelty | Optional stretch: cross-camera ReID for repeat offenders |

**Top 3 to fix next (biggest score-per-effort):**
1. **Evidence generation + metadata schema** — required, demo-able, low effort.
2. **Analytics/reporting dashboard** — required, visual, differentiator.
3. **Evaluation protocol table** — locks credibility; one table, high judge value.

---

## 9. Datasets summary for geometry violations
| Violation | Best dataset | Exists? | Fallback |
|-----------|-------------|---------|----------|
| Red-light | LISA / BSTLD / DriveU / DualCam (lights) | ✅ for lights | annotate stop-line per camera |
| Wrong-side | UA-DETRAC / CityFlow (tracks) | ⚠️ proxy | define lane direction per ROI |
| Stop-line | — (geometry) | ❌ | per-camera line annotation + IDD lanes |
| Illegal parking | i-LIDS + small sets | ⚠️ small | no-parking polygon + dwell time |
| Seatbelt | ~12k/10k sets, AICC Track 3 | ✅ partial | two-stage, daytime scope |

## Sources
- [LISA Traffic Light (Roboflow)](https://universe.roboflow.com/ithb-5ka4m/lisa-traffic-light-detection-8vuch) · [DualCam (arXiv 2209.01357)](https://arxiv.org/pdf/2209.01357)
- [Traffic Signal Violation Detection (ResearchGate)](https://www.researchgate.net/publication/379943035_Traffic_Signal_Violation_Detection_System_Using_Computer_Vision)
- [Wrong-Way Driving CV+ML (IJASEIT)](https://ijaseit.insightsociety.org/index.php/ijaseit/article/view/12376) · [Real-Time Wrong Direction (MDPI)](https://mdpi.com/2076-3417/10/7/2453/htm)
- [Illegal parking real-time (ACM 2024)](https://dl.acm.org/doi/10.1145/3691016.3691066) · [Illegal-Parking-Detection (GitHub)](https://github.com/Kevinjoythomas/Illegal-Parking-Detection)
- [Vehicle yielding to pedestrians (MDPI Sustainability)](https://www.mdpi.com/2071-1050/15/22/15714) · [RoadEye framework (ACM)](https://dl.acm.org/doi/10.1145/3700838.3703683)
- [Seatbelt CNN-SVM (ResearchGate)](https://www.researchgate.net/publication/381361078) · [Seatbelt YOLOv7 overhead (ResearchGate)](https://www.researchgate.net/publication/399255010)
- [UA-DETRAC (arXiv 1511.04136)](https://arxiv.org/abs/1511.04136) · [CityFlow (arXiv 1903.09254)](https://arxiv.org/abs/1903.09254)
- [Next-gen mobile traffic violation system (arXiv 2311.16179)](https://arxiv.org/pdf/2311.16179)
