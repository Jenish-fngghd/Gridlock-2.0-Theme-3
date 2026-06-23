"""Gridlock 2.0 — combined API + inference service (FastAPI)."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import health, process, violations, analytics

app = FastAPI(title="Gridlock 2.0 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (health, process, violations, analytics):
    app.include_router(r.router)


@app.on_event("startup")
def _prewarm_models() -> None:
    """In real mode, load the heavy pipeline (RF-DETR/TrOCR/classifiers) in a background thread
    at boot so the first user upload doesn't pay the ~20s cold-load. Health stays up immediately;
    warming proceeds in parallel. No-op in mock/dev mode."""
    import os
    import threading

    if os.getenv("INFERENCE_MODE", "mock").lower() != "real":
        return

    def _warm() -> None:
        try:
            from inference.service import _get_real_pipeline
            _get_real_pipeline()
        except Exception:  # noqa: BLE001 — warming is best-effort; first request will retry
            pass

    threading.Thread(target=_warm, daemon=True).start()


@app.get("/")
def root():
    return {"name": "Gridlock 2.0 API", "docs": "/docs", "health": "/api/health"}
