# Tri-Paradigm Verdict Engine

**Confidence-Cascaded, VLM-Verified Traffic Violation Detection with Tamper-Evident Evidence**
*(built for Gridlock 2.0 — Automated Photo Identification & Classification of Traffic Violations using Computer Vision)*

Gridlock 2.0 ingests a traffic photo (or short clip), detects **7 violation types + ANPR**, and
produces an annotated, auditable evidence record — surfaced through a live web dashboard.

**Live demo:** https://gridlock-2-0-theme-3.vercel.app/
**Backend/API:** https://13-232-119-109.sslip.io (AWS EC2)

---

## What it does

| Violation | How |
|---|---|
| Helmet | SAM-3 open-vocabulary detection (motorcycle/person/helmet) + a geometric rule on rider upper-body |
| Triple riding | SAM-3 entity counting — motorcycle + ≥3 overlapping persons |
| Seatbelt | Two-stage trained pipeline: YOLOv11n windshield detector → MobileNetV3 belt classifier (4-wheelers only) |
| Wrong-side driving | Dedicated fine-tuned heading classifier (MobileNetV3-small) |
| Red-light / stop-line | SAM-3 (crosswalk + vehicle-past-line) + HSV traffic-light colour rule, connected into one decision |
| Illegal parking | Geometry-dwell rule engine (needs scene config / video; abstains safely on stills) |
| ANPR | Plate localization + PaddleOCR / fine-tuned TrOCR, run only on confirmed violators |

Every result lands in a **confidence cascade** — `auto_confirm` / `human_review` / `discard` — and
uncertain cases are escalated to a **VLM verifier** (NVIDIA NIM, vision-language model) which
re-checks the violation and writes a human-readable caption before anything is auto-confirmed.

Full metrics and evaluation methodology: [`ml/RESULTS.md`](ml/RESULTS.md).
Architecture rationale and novelty write-up: [`ml/docs/final/00_master_design.md`](ml/docs/final/00_master_design.md)
and [`ml/docs/final/05_one_page_note.md`](ml/docs/final/05_one_page_note.md).

---

## Architecture

```
                 ┌────────────────────────┐
                 │  Frontend (Next.js)    │   deployed: Vercel
                 │  upload · feed ·       │
                 │  search · analytics    │
                 └───────┬─────────┬──────┘
            writes (POST)│         │ reads (direct, supabase-js, realtime)
                         ▼         ▼
        ┌────────────────────────┐   ┌──────────────────────────────┐
        │ Backend + Inference    │   │ Supabase                      │
        │ (FastAPI, on AWS EC2)  │──▶│  Postgres + Storage + Realtime│
        │  detect → violation    │   │  analytics views, audit log   │
        │  modules → VLM verify  │   └──────────────────────────────┘
        │  → confidence cascade  │
        └────────────────────────┘
```

- **Detection backbone:** RF-DETR (two-stage cascade).
- **Entity detector for helmet/triple/red-light:** Roboflow-hosted **SAM-3** (open-vocabulary
  promptable segmentation) + geometric rules — used because it generalises to scenes our
  single-purpose models hadn't seen, without per-class retraining.
- **Wrong-side:** a dedicated fine-tuned model, not SAM-3 — direction is a behavioural attribute,
  not a segmentable noun, so SAM-3 cannot express it; this is documented in `ml/RESULTS.md`.
- **VLM verification is selective**, not run on every frame — only `human_review`-band cases — to
  keep latency and cost down while still gaining precision.
- **Evidence is auditable**: every confirmed violation is stored with its annotated image,
  structured JSON, model version, and a hash-chained audit trail (tamper-evident).

## Novelty

1. **Paradigm-partitioned reasoning** — helmet/seatbelt (instance-attribute), triple-riding
   (multi-instance counting), and wrong-side/red-light/stop-line/parking (scene-context geometry)
   are solved by purpose-built modules, not one monolithic head.
2. **Open-vocabulary entity detection (SAM-3) + explicit geometric rules** instead of training a
   bespoke detector per violation class — new violation logic is a rule change, not a retrain.
3. **Confidence cascade + VLM-in-the-loop** — cheap/fast models decide first; a VLM verifies only
   the uncertain band and writes a human-readable caption, giving precision without per-frame VLM
   cost. The VLM is prompted adversarially-skeptically (confirm only if unambiguous) to avoid
   rubber-stamping leading questions.
4. **Honest, source-traceable metrics** — every number in `ml/RESULTS.md` is tied to a real
   evaluation run against ground truth; tasks without ground truth are explicitly labelled
   qualitative/not-yet-tested rather than given a fabricated score.
5. **Tamper-evident evidence** — hash-chained audit log + signed evidence images, so output is
   usable as actual enforcement evidence, not just a demo prediction.

---

## Repository layout

```
ml/             CV pipeline — modules, training, evaluation, checkpoints, design docs, RESULTS.md
inference/      Model-serving layer: loads checkpoints once, wraps ml/src/pipeline.py
backend/        FastAPI app — /api/process, /api/violations/{id}/review, /api/analytics, /api/health
frontend/       Next.js dashboard (upload, live feed, search, analytics) — deployed on Vercel
supabase/       SQL migrations (schema, RLS, storage buckets) + seed data
infra/          Dockerfile, docker-compose, AWS deployment runbook (infra/DEPLOY.md)
sample images/  Labelled ground-truth samples (violation / no-violation) used for evaluation
```

---

## Running it

### Option A — use the live deployment
Just open https://gridlock-2-0-theme-3.vercel.app/ and upload an image. No setup needed.

### Option B — run locally (mock inference, no GPU/API keys needed)

```bash
# Backend (mock mode — deterministic fake results, for UI/API testing without models)
cd backend
python -m venv ../.venv-web && ../.venv-web/Scripts/activate   # Windows
pip install -r requirements.txt
copy ..\.env.example ..\.env        # fill in Supabase keys if you want persistence
set INFERENCE_MODE=mock
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev   # http://localhost:3000, set NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Option C — run the real pipeline (requires GPU + API keys)

Set `INFERENCE_MODE=real` and populate `.env` with `ROBOFLOW_API_KEY` and `NVIDIA_API_KEY` (VLM).
See `.env.example` for every variable and `infra/DEPLOY.md` for the full Docker/AWS build.

```bash
docker build -t gridlock-api .
docker run --gpus all -p 8000:8000 --env-file .env gridlock-api
```

---

## Tech stack

Detection: RF-DETR · Entity segmentation: SAM-3 (Roboflow-hosted) · Wrong-side: MobileNetV3-small ·
Seatbelt: YOLOv11n + MobileNetV3-large · Red-light (temporal): LSTM trajectory classifier ·
ANPR: PaddleOCR PP-OCRv5 / fine-tuned TrOCR · VLM verification: NVIDIA NIM (Llama-3.2-Vision) ·
Backend: FastAPI · Frontend: Next.js + Tailwind · DB/Storage: Supabase (Postgres + Realtime) ·
Deploy: Vercel (frontend) + AWS EC2 + Docker + Caddy/HTTPS (backend).
