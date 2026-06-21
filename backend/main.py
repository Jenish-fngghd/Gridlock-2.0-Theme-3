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


@app.get("/")
def root():
    return {"name": "Gridlock 2.0 API", "docs": "/docs", "health": "/api/health"}
