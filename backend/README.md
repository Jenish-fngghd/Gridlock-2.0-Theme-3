# Gridlock 2.0 — Backend + Inference

Combined FastAPI service: API + violation-detection pipeline + Supabase writes.

## Run locally (mock inference, no keys needed)

```bash
python -m venv .venv-web && source .venv-web/Scripts/activate   # Windows: .venv-web\Scripts\activate
pip install -r backend/requirements.txt
export INFERENCE_MODE=mock          # dev: deterministic fake results, no GPU/checkpoints
uvicorn backend.main:app --reload --port 8000
```

- `GET  /api/health` — readiness (inference mode + whether Supabase is configured)
- `POST /api/process` — multipart `file=@photo.jpg` → runs pipeline → returns violations
  (persists to Supabase only when `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set)
- `POST /api/violations/{id}/review` — `{action: confirm|reject|escalate}` → status + chained audit
- `GET  /api/analytics/summary` — dashboard aggregates from the SQL views

## Modes

| Var | Dev | Production (AWS GPU) |
|---|---|---|
| `INFERENCE_MODE` | `mock` | `real` (wires `ml/src/modules/pipeline.py`) |
| Supabase keys | unset → returns results un-persisted | set → persists evidence + images |

## Layout
```
backend/
  main.py        # FastAPI app + CORS + routers
  config.py      # env settings
  supa.py        # service-role Supabase client (lazy)
  storage.py     # bucket uploads
  audit.py       # evidence_audit hash chain
  persist.py     # write job/detections/plates/violations/audit
  schemas.py     # response models
  routers/       # health, process, violations, analytics
inference/
  service.py     # mock | real dispatch
  mock.py        # deterministic dev output
  types.py       # Detection / ViolationResult / PlateResult / PipelineResult
```
