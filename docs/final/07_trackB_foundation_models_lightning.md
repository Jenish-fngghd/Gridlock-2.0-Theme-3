# Track B — Foundation-Model Violation Tests on Lightning (H200)

Goal: test whether a single open-vocabulary foundation model can detect our traffic violations
**zero-shot** (no fine-tuning), versus our specialized per-violation modules. This is the
§3.10 "foundation-model data engine" question and J1's teacher/verifier role made concrete.

**Why Lightning, not local:** these models need ≥12 GB VRAM; the dev laptop has 4 GB. Run them
on the H200 you already have up for the detection fine-tune.

**License caution:** LocateAnything-3B is **NVIDIA non-commercial** → use as a teacher/baseline,
never in the shipped product (consistent with §12 / J1). SAM 3 — check Meta's license before any
deployment claim. For *measuring capability* both are fine.

---

## Models under test

| Model | Repo | What we probe it for | License |
|---|---|---|---|
| **LocateAnything-3B** | `nvidia/LocateAnything-3B` | open-vocab box grounding from phrases ("motorcyclist without helmet", "license plate", "three people on one motorcycle") | non-commercial |
| **SAM 3** | `facebook/sam3` (or `ultralytics` SAM3) | open-vocab concept detect+segment ("auto rickshaw", "motorcycle with three riders") + clean evidence masks | check Meta terms |

The two highest-value questions:
1. **Helmet** (our only no-result mandated violation) — can either model flag "motorcyclist
   without helmet" zero-shot, replacing the blocked AICC path?
2. **India-specific detection** (auto-rickshaw / cycle-rickshaw — the COCO-gap classes our RF-DETR
   scores ~0 on) — can SAM 3 / LocateAnything segment them from the noun phrase alone? If yes,
   they become an auto-labeler (data engine) to fine-tune our fast detector cheaply.

---

## Setup on Lightning

```bash
# from the project root (upload src/ + a sample of IDD frames)
pip install "transformers>=4.49" accelerate pillow torch torchvision
pip install "ultralytics>=8.4"          # for the SAM3 ultralytics backend
```

## Run

```bash
# 1) LocateAnything-3B — phrase grounding for every violation
python -m src.eval.test_locateanything \
    --images datasets/idd-detection/IDD_Detection/JPEGImages \
    --limit 40 --out results/locateanything

# 2) SAM 3 — open-vocab concepts (try ultralytics backend first, then hf)
python -m src.eval.test_sam3 \
    --images datasets/idd-detection/IDD_Detection/JPEGImages \
    --limit 40 --out results/sam3 --backend ultralytics
```

Outputs: annotated images + a `*_results.json` per model (per-image, per-phrase box/instance
counts), plus (SAM3) a concept hit-rate table.

---

## Reading the results / known caveats

- **API drift:** these are brand-new VLMs; their exact inference method names move. Both scripts
  try several call patterns and **degrade gracefully** (`unsupported_api` in the json). If you see
  that, open the model card's example snippet on HF and adjust the `_ground()` (LocateAnything) or
  the backend fn (SAM3) to its exact API — the *prompt list* is the part that stays.
- **Phrases are the tuning surface.** Edit `PROMPTS` / `CONCEPTS` in the scripts — open-vocab means
  the whole experiment is "which phrasing best isolates each violation."
- **No GT here** → these are **qualitative capability tests** (do they fire on the right thing?),
  not benchmark metrics. To turn a winner into a metric, use it to auto-label, then score the
  labels against a held-out set (the data-engine loop).

## Decision this informs
- If a foundation model nails **helmet** zero-shot → helmet is unblocked without AICC imagery.
- If it nails **auto-rickshaw/cycle-rickshaw** → use it as the **auto-labeler** to fine-tune
  RF-DETR on the India-specific classes (closes the structural detection gap, §6/§7).
- If neither → confirms the specialized modules + targeted fine-tunes remain the right call, and
  these stay teacher-only tools.
```
