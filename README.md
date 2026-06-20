# Gridlock 2.0 — Round 2

**Automated Photo Identification & Classification of Traffic Violations using Computer Vision.**
A paradigm-partitioned, geometry-aware, confidence-cascaded pipeline for unstructured Indian traffic, with a zero-shot baseline implementation and an evaluation harness for Western + Indian benchmarks.

---

## Repository structure

```
Gridlock2.0_R2/
├── README.md                         ← you are here (start with docs/final/00_master_design.md)
├── requirements.txt                  ← Python dependencies (for the implementation, in progress)
├── .gitignore
├── docs/                             ← design & research, grouped by plan
│   ├── final/                        ← ★ the canonical merged plan + companions (read in numeric order)
│   │   ├── 00_master_design.md       ← ★ CANONICAL submission design (single source of truth)
│   │   ├── 01_justifications.md      ← J1…J6: defenses of every non-obvious decision
│   │   ├── 02_comparison_merge_finetuning.md ← Plan A vs B, merge rationale, fine-tuning + GPU/cost plan
│   │   ├── 03_sota_registry.md / .csv ← SOTA review + Paper/Source Registry (2022–2026)
│   │   ├── 04_datasets_acquisition_and_prep.md ← dataset acquisition + prep guide
│   │   ├── 05_one_page_note.md       ← one-page architecture summary (diagram + node-by-node)
│   │   └── 06_model_selection_justification.md / .csv ← per-model/-module selection sheet
│   ├── plan_a/                       ← original Plan A draft (superseded, kept for history)
│   └── plan_b/                       ← Plan B source research (AICC deep-dive, repo & geometry audits)
├── configs/
│   └── camera_config.json            ← per-camera geometry config (Scene Context Model)
├── src/                              ← implementation (pipeline — in progress)
├── scripts/                          ← helper scripts
├── datasets/                         ← place benchmark datasets here (gitignored; see datasets/README.md)
└── outputs/                          ← annotated images, evidence JSON, benchmark results (gitignored)
```

## Where to start

1. **`docs/final/00_master_design.md`** — the canonical architecture (all module/model decisions, datasets, eval targets). Everything else supports this.
2. **`docs/final/05_one_page_note.md`** — the one-page summary: 3-paradigm framing, the end-to-end diagram, and a node-by-node walkthrough. Read this first if you want the gist fast.
3. **`docs/final/01_justifications.md`** — the "why this and not the obvious alternative" defenses (paradigm partitioning, RF-DETR over YOLOv12, geometry-as-config, why fine-tune, etc.).
4. **`docs/final/06_model_selection_justification.md`** (+ `.csv`) — the per-model/-module selection sheet (what · where · alternatives · why).
5. **`docs/final/02_comparison_merge_finetuning.md`** — Plan A vs Plan B comparison, the merged design, and the **fine-tuning plan with GPU-time & cost estimates**.
6. **`docs/final/03_sota_registry.md`** / **`04_datasets_acquisition_and_prep.md`** — SOTA registry and dataset acquisition guide.

> **Status:** design is finalized; the implementation (`src/pipeline.py`) is **in progress** — datasets are being acquired manually before the build starts. Numbers marked **VERIFY** in the docs should be confirmed at primary source before quoting to judges (see the Verification Log in `docs/final/00_master_design.md` §14).

---

## Quickstart (target interface — once `src/pipeline.py` lands)

> The pipeline is being (re)implemented after manual dataset acquisition; the commands below are the intended CLI.

```bash
# 1. install (minimal CPU set; see requirements.txt for the full list)
pip install numpy opencv-python ultralytics paddleocr paddlepaddle

# 2. run on an image or folder -> annotated images + evidence JSON in outputs/
python src/pipeline.py --mode demo --input path/to/image_or_folder --output outputs/

# 3. evaluate one dataset (place data under datasets/ first — see datasets/README.md)
python src/pipeline.py --mode eval --dataset idd --data-root ./datasets --limit 500

# 4. full zero-shot benchmark (Western + Indian) with auto domain-gap + fine-tuning priority
python src/pipeline.py --mode benchmark --data-root ./datasets --limit 500

# helper: emit a sample per-camera geometry config
python src/pipeline.py --write-sample-config configs/camera_config.json
```

Every module **degrades gracefully**: if a model weight or dependency is missing, it logs the gap, stubs the output with a `model_unavailable` / `needs_review` flag, and the pipeline continues. By design, **seatbelt, wrong-side, and illegal-parking are zero-shot stubs** — the benchmark's auto-generated *Fine-tuning Priority* ranks them first.

See `docs/final/02_comparison_merge_finetuning.md` §6 for the fine-tuning plan and GPU/cost budget.
