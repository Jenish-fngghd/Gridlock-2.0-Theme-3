# Gridlock 2.0 — Web Platform Plan

> Turns the `ml/` CV pipeline into a **deployed web product**: upload traffic photos/clips →
> run the violation-detection pipeline → store auditable evidence → browse, search, and analyze
> in a dashboard. **No authentication** (open demo). DB + storage on **Supabase**; model inference
> on **AWS**; frontend on **Vercel**.

**Decisions (locked):** demo accepts **images + short clips** (full violation coverage) · inference on
**AWS EC2 GPU `g4dn.xlarge` (T4)** · frontend **Next.js** on Vercel · **combined** FastAPI service
(API + inference + Supabase writes) on the AWS box, with clean module boundaries to split later.

---

## 1. What the problem statement requires (from `ml/docs/final/00_master_design.md`)

The system must, for **7 violation types + ANPR**, produce *annotated, auditable, court-admissible*
records and surface them through an **analytics/search dashboard** (§ "8 Store → 9 Analytics"):

- **Violations:** helmet, triple-riding, seatbelt, wrong-side, stop-line, red-light, illegal parking.
- **ANPR:** plate read on confirmed violators only (Indian HSRP + old formats).
- **Evidence:** annotated image + structured JSON + timestamp + SHA-256 + audit trail (tamper-evident).
- **Confidence cascade:** every result lands in `auto-confirm | human-review | discard`.
- **Analytics:** counts by type/time/camera, hotspot map, trends, **searchable records**, repeat-offender view, CSV/PDF export.

This maps cleanly onto: a **DB schema** (records + evidence + audit), an **inference service** (the `ml/` models),
a **backend API** (orchestration + write path), and a **frontend** (dashboard + upload + search).

---

## 2. Target architecture

```
                 ┌────────────────────────┐
                 │  Frontend (Next.js)     │   deployed: Vercel  → the public link
                 │  - Upload page          │
                 │  - Live violations feed │
                 │  - Search / filters     │
                 │  - Analytics dashboard  │
                 │  - Evidence detail view │
                 └───────┬─────────┬───────┘
            writes (POST)│         │ reads (direct, supabase-js + anon key, realtime)
                         ▼         ▼
        ┌────────────────────────┐   ┌──────────────────────────────┐
        │ Backend + Inference     │   │ Supabase                      │
        │ (FastAPI, on AWS)       │──▶│  - Postgres (schema §4)       │
        │  - POST /process        │   │  - Storage (originals/        │
        │  - runs ml/ pipeline    │   │    annotated/ crops/)         │
        │  - writes via service   │   │  - Auto REST + Realtime       │
        │    key + uploads images │   │  - SQL views for analytics    │
        │  - POST /review         │   └──────────────────────────────┘
        │  - GET /analytics (RPC) │
        └────────────────────────┘
```

**Key design choices**
- **Reads bypass the backend.** The dashboard reads Supabase **directly** (supabase-js, anon key) and
  subscribes to **realtime** — new violations appear live with zero backend code. The custom backend is
  essentially just the **write path** (inference).
- **Images live in Supabase Storage**, not AWS S3 — one less service, public CDN URLs out of the box.
- **Service-role key stays server-side only** (in the FastAPI backend). The frontend only ever holds the
  **anon** key. (See §6 security.)
- **Clean module boundary** so backend + inference can later split into two services if needed; for the
  demo they run as one FastAPI app on the AWS box.

---

## 3. Repository layout (monorepo)

```
/
├── ml/                  # existing CV pipeline (unchanged) — research, training, eval, checkpoints
├── inference/           # model-serving layer: loads checkpoints once, wraps ml/src/modules/pipeline.py
│   ├── service.py       #   load_models() + run_pipeline(image) -> structured result
│   └── mock.py          #   deterministic fake results for local dev (real models need GPU/RAM)
├── backend/             # FastAPI app (API + orchestration + Supabase writes)
│   ├── main.py
│   ├── routers/         #   process.py, violations.py, analytics.py, health.py
│   ├── db.py            #   Supabase client (service key)
│   ├── storage.py       #   image upload helpers
│   └── schemas.py       #   pydantic request/response models
├── frontend/            # Next.js dashboard (deployed on Vercel)
├── supabase/
│   ├── migrations/      #   versioned SQL (schema §4)
│   └── seed.sql
├── infra/               # Dockerfile(s), docker-compose, AWS deploy notes
├── .env.example
└── PLAN.md
```

---

## 4. Database schema (Postgres / Supabase) — "strongest possible"

Normalized + auditable. Enums for closed sets; JSONB for flexible geometry/bbox; a **hash-chained audit
log** for tamper-evidence; a **model_registry** to honor the "no silent model substitution" rule.

### Enums
- `violation_type`: helmet | triple_riding | seatbelt | wrong_side | stop_line | red_light | illegal_parking
- `confidence_band`: auto_confirm | human_review | discard
- `violation_status`: pending | confirmed | rejected | challan_issued
- `job_status`: queued | processing | done | failed
- `media_type`: image | video | frame
- `review_action`: confirm | reject | escalate

### Tables

**cameras** — a source (physical camera or "uploads"). Holds the Scene Context Model.
| col | type | notes |
|---|---|---|
| id | uuid pk | |
| name | text | |
| location_name | text | |
| latitude / longitude | double | for hotspot map |
| scene_context | jsonb | stop_line polygon, no_park polygons, lane vectors, signal ROI, homography |
| is_active | bool | |
| created_at | timestamptz | |

**ingestion_jobs** — one uploaded image/video and its processing lifecycle.
| id uuid pk · camera_id fk→cameras · media_type · storage_path (original) · sha256 · captured_at · uploaded_at · status (job_status) · processing_ms · error · width · height · model_version · created_at |

**detections** — every object found in a job (vehicle/person/plate/light).
| id uuid pk · job_id fk→ingestion_jobs · class_label · confidence · bbox jsonb · track_id (nullable, video) · attributes jsonb · created_at |

**violations** — the core record (one per detected violation).
| id uuid pk · job_id fk · camera_id fk · violation_type · confidence · confidence_band · status (violation_status) · detected_at · annotated_image_path · evidence jsonb (bboxes, involved detection ids) · vlm_caption · model_module · model_version · sha256_evidence · signature · plate_id fk→plates (nullable) · metadata jsonb · created_at · updated_at |

**plates** — ANPR results, normalized for repeat-offender lookups.
| id uuid pk · plate_text (raw) · plate_normalized (unique) · state_code · is_valid_format · ocr_confidence · first_seen_at · last_seen_at · created_at |

**review_actions** — human review-band decisions (anonymous; no auth).
| id uuid pk · violation_id fk · action (review_action) · reviewer_label text · notes · created_at |

**evidence_audit** — append-only, **hash-chained** (`sha256 = H(payload + prev_hash)`) → tamper-evident trail.
| id uuid pk · violation_id fk · event_type · payload jsonb · sha256 · prev_hash · created_at |

**model_registry** — which model/version produced results (auditability + no-silent-substitution).
| id uuid pk · module · model_name · variant · version · metric_name · metric_value · checkpoint_ref · is_active · deployed_at |

### Analytics (SQL views / RPC)
`v_violations_by_type`, `v_violations_by_day`, `v_violations_by_camera`,
`v_hotspots` (lat/lng + count), `v_repeat_offenders` (plate_normalized + count + last_seen),
`v_confidence_bands`. Exposed to the frontend via Supabase **RPC** or direct view selects.

### Indexes
`violations(violation_type)`, `violations(detected_at desc)`, `violations(camera_id)`,
`violations(plate_id)`, `violations(confidence_band, status)`, `plates(plate_normalized) unique`,
`detections(job_id)`, `ingestion_jobs(status, created_at)`.

### Storage buckets
`originals/` (uploads) · `annotated/` (evidence images) · `crops/` (plate/violation crops).

---

## 5. Backend API (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/process` | multipart image/clip → store original → run pipeline → create job + detections + violations + plate + annotated image → return records |
| POST | `/api/violations/{id}/review` | record a human review action; append to audit log |
| GET | `/api/analytics/summary` | aggregates (or proxy to Supabase RPC) |
| GET | `/api/health` | model + DB readiness |

Reads for the feed/search/detail pages go **direct to Supabase** from the frontend (supabase-js), not through these endpoints. Large videos process **async** (background task / queue) so the request doesn't block.

---

## 6. Security (with auth OFF)

- **Service-role key**: backend only, via env var, never shipped to the browser.
- **Frontend**: anon key only. Lock Supabase RLS so anon is **read-only**; all writes go through the backend.
- **Backend**: CORS allow-list (Vercel domain + localhost), request size limit, basic rate limit on `/process`.
- **Password hygiene** — see correction C4 below.

---

## 7. Deployment

- **Frontend** → **Vercel** (Next.js): the public deployed link. Env: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_BACKEND_URL`.
- **DB + Storage** → **Supabase** (already created).
- **Backend + Inference** → **AWS**. Recommended simplest path: one **EC2** instance + Docker + uvicorn,
  Elastic IP, behind HTTPS. GPU (`g4dn.xlarge`, T4) if we want fast RF-DETR-large; CPU (`t3.xlarge`) is
  cheaper but slower. (Decision pending — see questions.) Start/stop or spot to control cost.

---

## 8. Corrections & things to flag (you asked me to)

- **C1 — You don't need AWS S3.** Supabase Storage already gives buckets + public CDN URLs for evidence
  images. Using S3 too would just add a second storage system to keep in sync. Dropped from the plan.
- **C2 — Most of the dashboard needs *no* backend.** Supabase auto-generates a REST/Realtime API. The
  frontend reads violations/analytics directly and gets **live updates** for free. Keeps the custom
  backend tiny (just the inference write-path).
- **C3 — Local dev can't run the real models.** Your local Windows box has the paging/VRAM limits noted in
  memory. So the backend ships with a **mock inference mode** for local development; real checkpoints load
  only on the AWS box. Plan accordingly (don't block backend work on local model loading).
- **C4 — The Supabase password has trailing spaces** (`"Gridlock 2.0  "`). That is fragile: it must be
  URL-encoded in any connection string and silently breaks copy/paste. **Strong recommendation: reset it**
  to a clean value with no spaces. Also, prefer the Supabase **client libraries + API keys** over a raw
  Postgres connection string wherever possible, so the password barely matters.
- **C5 — Temporal violations need motion.** red-light-running, illegal-parking (dwell), and true wrong-side
  need a **video/burst**, not a single still. From one photo we can do helmet, triple-riding, seatbelt,
  signal-state, and a single-frame wrong-side proxy. Decide whether the demo is image-only or supports clips.
- **C6 — Inference cost/ops.** A persistent AWS GPU box is ~$0.5/hr (~$380/mo if always on). If that's a
  concern, **Modal / Hugging Face Spaces** give a pay-per-use GPU endpoint with near-zero ops — but you said
  AWS, so AWS is the plan unless you want to revisit.
- **C7 — I still need the Supabase project URL + API keys.** The pasted "[22 lines]" didn't reach me — only
  the password did. To wire anything up I need: **Project URL** (`https://<ref>.supabase.co`), the **anon**
  public key, and the **service_role** key (paste the service_role privately — it's a secret).

---

## 9. TODO

### Phase A — Foundations
- [x] Confirm decisions (images+clips · AWS GPU g4dn.xlarge · Next.js · combined service)
- [ ] Get Supabase project URL + anon key + service_role key; reset password to a clean value
- [x] Create `.env.example`  (local `.env` pending keys)
- [~] Scaffold dirs (`supabase/` done; `backend/` `inference/` `frontend/` `infra/` pending)

### Phase B — Database (Supabase)
- [x] `supabase/migrations/0001_init.sql`: enums, 8 tables, indexes, trigger, analytics views (§4)
- [x] Storage buckets `originals`, `annotated`, `crops`  (`0002_rls_storage.sql`)
- [x] Analytics views (`v_violations_by_type/day/camera`, `v_hotspots`, `v_repeat_offenders`, `v_confidence_bands`)
- [x] RLS: anon read-only; service-role full (`0002_rls_storage.sql`)
- [x] `seed.sql`: "Uploads" camera + model_registry rows mirroring `ml/results/run_history.csv`
- [x] Applied migrations + seed via psql/IPv6; verified (8 tables, 6 enums, 6 views, RLS×8, 3 buckets, 7 model rows)

### Phase C — Backend + Inference
- [x] `inference/types.py` + `inference/mock.py`: deterministic fake results for local dev
- [x] `inference/service.py`: mock|real dispatch (real path = TODO: wrap `ml/src/modules/pipeline.py`)
- [x] `backend/`: FastAPI app, config, Supabase client, storage, audit hash-chain, pydantic schemas
- [x] `POST /api/process`: infer → (persist) store/job/detections/plate/violations/audit → return
- [x] `POST /api/violations/{id}/review` + chained audit append
- [x] `GET /api/analytics/summary`, `GET /api/health`
- [x] Smoke-tested in mock mode (`.venv-web`) — health + process + determinism pass
- [x] **Live end-to-end verified** against Supabase: process→storage+job+detections+plate+violation+audit,
      review (status flip + chained audit CHAIN OK), analytics aggregates. Keys in gitignored `.env`.
- [x] **Wired `real` inference** — rewrote `ml/src/pipeline.py` orchestrator: every violation
      module now calls an actual trained classifier/decision (was previously detect-only stubs
      for seatbelt/helmet). Added confidence-cascade VLM escalation (NVIDIA NIM,
      `human_review`-band only) and SAM-3 plumbing (helmet-state crops, plate localization) —
      SAM-3 itself segfaults locally on Windows (root-caused to the official package's text
      encoder construction; works on Colab per the user's own test) — deferred to AWS
      verification, code path is complete and degrades gracefully if unavailable. See
      `ml/TODO.md` Phase 8 for full detail.
      **Verified end-to-end through the real FastAPI backend**: upload → real RF-DETR + real
      wrong-side classifier + VLM verification → persisted to Supabase (job/detections/
      violations/signed evidence image) → confirmed via direct DB query → cleaned up.
- [ ] Dockerfile + local docker-compose (combined web+ML deps into one image for AWS)

### Phase D — Frontend (Next.js 16 + Tailwind v4 + motion)
- [x] Scaffold app, supabase-js client (anon), Tailwind; deps: motion, recharts
- [x] Design system: preloader, Space Grotesk display font, glassmorphism, mesh bg, motion entrances
- [x] Dashboard: animated stat cards, by-type bar + bands donut, deployed-models strip, realtime feed
- [x] Upload page → backend `/process`, drag-drop + animated result cards
- [x] Violations list (realtime) + filters (type/band) + plate search + detail drawer with review (confirm/reject)
- [x] Supabase realtime enabled on `violations`; production build passes (TS + lint clean)
- [ ] Stretch: hotspot map, CSV/PDF export, video/burst UI for temporal violations

### Phase E — Deploy
- [ ] Frontend → Vercel (env vars) → the public link
- [ ] Backend+inference → AWS (EC2 + Docker + HTTPS + Elastic IP)
- [ ] End-to-end smoke test against the deployed stack
