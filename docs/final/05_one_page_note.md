# Gridlock 2.0 — R2 · Architecture Note
**Automated Photo Identification & Classification of Traffic Violations using Computer Vision**
---

### The core idea
A **modular, paradigm-partitioned, geometry-aware, confidence-cascaded** system that ingests **photographic evidence at scale** (image-first, cloud-native; edge optional) and outputs **annotated, auditable, court-admissible** records for all 7 violations + ANPR + analytics.

**Organizing insight — the 7 violations fall into 3 reasoning paradigms**, each handled by a purpose-built module on one shared perception backbone:

| Paradigm | Violations | Mechanism |
|---|---|---|
| **A · Instance-attribute** | Helmet, Seatbelt | part-level fine-grained classification on an *attributed* person |
| **B · Multi-instance counting** | Triple riding | count riders co-located on one two-wheeler |
| **C · Scene-context** | Wrong-side, Stop-line, Red-light, Illegal parking | geometry + signal-state + motion via a per-camera Scene Context Model |

---

### End-to-end architecture
```
            image / frame
                  │
                  ▼
       ┌─────────────────────┐
       │ 0 Ingest+QA          │   pyiqa blur/glare score
       └──────────┬───────────┘
                  ▼
       ┌─────────────────────┐
       │ 1 Preprocess         │   Retinexformer / OneRestore, quality-gated
       └──────────┬───────────┘
                  ▼
       ┌─────────────────────┐        distills into     ┌──────────────────────────┐
       │ 2 Detect             │◄───────────────────────  │ Offline Data Engine       │
       │   RF-DETR 2-stage     │                          │ GroundingDINO/SAM2/LocAny │
       └──────┬────────┬──────┘                          │ auto-label → distill      │
              ▼        ▼                                 └──────────────────────────┘
   ┌────────────────┐ ┌──────────────────────┐
   │ 3 Track         │ │ 4a Helmet+Triple      │
   │   BoostTrack    │ │   AICC 7-class+assoc  │
   └────────┬─────────┘ └──────────┬────────────┘
            ▼                      │
   ┌─────────────────────┐         │
   │ 4b Geometry Rules    │         │
   │   per-camera SCM     │         │
   │   + signal + 3D       │        │
   └────────┬──────────────┘        │
            └────────────┬──────────┘
                         ▼
            ┌─────────────────────────┐
            │ 5 Conf. Cascade          │   calibrate → auto/review/discard
            └────┬────────────────┬────┘
                 │         low-conf▼
                 │           ┌──────────────────────┐
                 │           │ 6 VLM Verify          │   Qwen2.5-VL/InternVL3, selective
                 │           └──────────┬─────────────┘
                 └─────────┬─────────────┘
                           ▼
                ┌─────────────────────┐
                │ 4c ANPR              │   PP-OCRv5, violators only
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ 7 Evidence           │   signed JSON + SHA-256 + audit
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ 8 Store              │   DB + object store
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │ 9 Analytics          │   dashboard, search, reports
                └─────────────────────┘
```

**Brief Overview of Architecture:**

1. **0 Ingest+QA** — Accepts each photo or sampled frame and scores its quality (blur, glare, exposure) with a lightweight IQA model (pyiqa). It exists to send only degraded images into restoration, saving compute and avoiding artifacts on already-clean images.
2. **1 Preprocess** — Conditionally restores degraded images with Retinexformer for low-light and OneRestore for rain/haze/blur, skipping clean ones. It runs only when the QA gate flags a problem, so the detector sees a cleaner image without blind over-processing.
3. **2 Detect** — Localizes vehicles, riders, pedestrians, plates and traffic-light heads using the RF-DETR family (Apache-2.0): a fast Stage-1 screen confirmed by a heavier RF-DETR-L/Co-DETR pass on candidate crops. It is the shared, license-clean perception backbone that every downstream module builds on.
4. **3 Track** — Assigns persistent IDs across frames using BoostTrack (via boxmot, with BoT-SORT/ByteTrack as fallback). It exists because every temporal violation — crossing a line, opposing lane direction, dwell time — requires following the same object over time.
5. **4a Helmet+Triple** — Detects each motorbike, crops it, and classifies the AICC seven-class scheme (driver/pillion × helmet) to flag both helmet non-compliance and triple-riding, associating heads to riders to bikes via SAM2 + RTMPose. One labelled scheme solves two violations at once, backed by a public dataset and proven recipes.
6. **4b Geometry Rules** — Applies a per-camera Scene Context Model (stop-line, no-park polygon, lane direction, signal ROI annotated once) plus a signal-state classifier and single-frame monocular-3D to decide red-light, wrong-side, stop-line and parking. It is rule-based rather than learned so decisions stay interpretable, auditable and court-admissible without a labelled temporal dataset.
7. **5 Conf. Cascade** — Calibrates each violation's score with temperature scaling and per-class thresholds, then sorts it into auto-confirm, human-review or discard. It exists to make auto-challan defensible and to decide which uncertain cases are worth escalating.
8. **6 VLM Verify** — Sends only low-confidence cases to an open-source VLM (Qwen2.5-VL / InternVL3 / LocateAnything-3B) that confirms the violation and writes a human-readable evidence caption. Running it selectively delivers precision and explainability without the prohibitive cost of a VLM on every frame.
9. **4c ANPR** — Reads the plate only on confirmed violators: detect plate → rectify/super-resolve → PaddleOCR PP-OCRv5 → Indian-format (HSRP) syntax validation. Gating OCR to violators saves compute, and the syntax check rejects impossible plates.
10. **7 Evidence** — Packages each confirmed violation into an annotated image plus a signed JSON record (timestamp, camera, bbox, plate, caption) with SHA-256 and an audit trail. This tamper-evident record is what makes the output usable as legal evidence rather than a demo.
11. **8 Store** — Persists records in SQLite/Postgres and evidence images in object storage behind a search index. It exists to make every violation queryable by plate, type, date or camera.
12. **9 Analytics** — Surfaces counts, trends, hotspot maps, searchable records, repeat-offender views and exportable reports through a Streamlit/FastAPI dashboard. This turns raw violations into the actionable enforcement insight the brief asks for.
13. **Offline Data Engine** — Separately from the live path, open-vocab teachers (GroundingDINO / Grounded-SAM-2 / LocateAnything-3B) auto-label Indian footage, which is spot-checked and distilled into the Stage-1 detector. It cuts manual labeling of rare Indian classes and keeps detection improving over time.

### Novelty:
1. **Paradigm-partitioned reasoning** — architecture organized around the *nature of evidence* each violation needs, not one monolithic head.
2. **Geometry-as-config + Scene Context Model** — annotate a fixed camera once (stop-line, lanes, no-park, signal ROI); makes hard temporal violations feasible, interpretable, auditable.
3. **Confidence cascade + VLM-in-the-loop** — cheap models first; a VLM verifies only uncertain cases and writes human-readable evidence → precision without per-frame VLM cost.
4. **Foundation-model data engine** — open-vocab teachers auto-label rare Indian classes (auto-rickshaw, "rider without helmet") and distill into the fast detector → cuts labeling cost.
5. **Admissibility + Indian grounding** — calibrated confidence, abstain/human-review band, tamper-evident signed evidence; fine-tuned on IDD/DriveIndia/AICC/Indian plates.
