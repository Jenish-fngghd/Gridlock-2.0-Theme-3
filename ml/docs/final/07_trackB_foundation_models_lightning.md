# Track B — Foundation-Model Tests on Lightning (H200): test ALL tasks

Tests whether ONE open-vocabulary foundation model can detect **every mandated violation +
the India-specific detection classes + license plates**, zero-shot, vs. our specialized
modules. This is the §3.10 "foundation-model data engine" question and J1's teacher/verifier
role, made concrete and exhaustive.

Two models, both probed on the **same 12 tasks** (`src/eval/violation_prompts.py`):
helmet (no/yes) · seatbelt · triple-riding · red-light · wrong-side · stop-line · illegal-parking
· auto-rickshaw · cycle-rickshaw · vehicles · license-plate.

| Model | Repo | VRAM | License |
|---|---|---|---|
| **LocateAnything-3B** | `nvidia/LocateAnything-3B` | ~12 GB | non-commercial (teacher/baseline only) |
| **SAM 3** | `facebook/sam3` or ultralytics `sam3.pt` | H100/H200 class | check Meta terms |

Both need ≥12 GB → **the 4 GB dev laptop OOMs; run on the H200.**

---

## 1. What to upload to Lightning

Upload these (a zip of `src/` + a sample of images is easiest):

```
project_root/
├── src/                                  # the whole package (keep __init__.py files)
│   ├── __init__.py
│   └── eval/
│       ├── __init__.py
│       ├── violation_prompts.py          # the 12-task prompt set (shared)
│       ├── test_locateanything.py        # LocateAnything-3B runner
│       └── test_sam3.py                   # SAM 3 runner
└── <some images>                         # e.g. a few hundred IDD/Indian traffic frames
```

You do **not** need the datasets folder or the other modules — just `src/eval/` (+ its
`__init__.py` parents) and an image folder. Point `--images` at wherever you put the frames.

> Minimal upload tip: `zip -r trackB.zip src` locally, upload, `unzip trackB.zip` on Lightning,
> then upload/observe an image folder (a few hundred frames is plenty for a capability test).

---

## 2. Setup on Lightning (H200)

```bash
pip install "transformers>=4.49" accelerate pillow torch torchvision
pip install "ultralytics>=8.4"        # for the SAM3 ultralytics backend
```

## 3. Run — both models, all 12 tasks

```bash
# A) LocateAnything-3B  (phrase grounding -> boxes, per task)
python -m src.eval.test_locateanything --images <your_image_folder> --limit 200 \
    --out results/locateanything

# B) SAM 3  (open-vocab concept detect+segment; try ultralytics first, then hf)
python -m src.eval.test_sam3 --images <your_image_folder> --limit 200 \
    --out results/sam3 --backend ultralytics
# if the ultralytics SAM3 wrapper isn't available yet:
python -m src.eval.test_sam3 --images <your_image_folder> --limit 200 \
    --out results/sam3 --backend hf
```

`--limit` = how many images to test (start ~200; the H200 is fast). Bigger = firmer signal.

---

## 4. What you get (per model)

- **`SUMMARY.md`** ← the headline: a per-task **hit-rate table** (in how many images the model
  fired on each violation/class), with the 7 mandated violations starred.
- **`*_results.json`** ← per-image, per-task boxes/instances (machine-readable).
- **annotated images / mask overlays** ← to eyeball correctness.

## 5. How to read it (honest framing)

- **No ground truth here** → hit-rate is a **capability signal** ("does it fire on the right
  thing?"), NOT accuracy. Inspect the annotated images to judge quality.
- **The decisions this informs:**
  - High, clean hit-rate on **helmet** → unblocks our one no-result violation without AICC data.
  - High hit-rate on **auto-rickshaw / cycle-rickshaw** → use the model as an **auto-labeler**
    to fine-tune our fast RF-DETR on the India-specific classes (closes the §6/§7 detection gap).
  - Weak/noisy → confirms the specialized modules + targeted fine-tunes remain the right call,
    and these stay teacher-only tools.
- **License:** LocateAnything is non-commercial → never ship it; use it to *generate labels* or
  as a *baseline*. SAM 3 — verify Meta's terms before any deployment claim.

## 6. Known caveats (scripts handle them)
- **API drift:** these VLMs are brand-new; method names shift. `test_locateanything.py` tries
  several call patterns and tags `unsupported_api` in the json if none work — then open the HF
  model card's example and tweak `_ground()` (the prompt set stays). `test_sam3.py` has both an
  ultralytics and an HF backend for the same reason.
- **Phrases are the tuning surface.** Edit `src/eval/violation_prompts.py` to try better wording
  for any violation — that's the whole experiment for open-vocab models.
