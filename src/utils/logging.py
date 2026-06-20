"""Structured logging helpers shared across all phases.

Every run writes a JSON blob to logs/<phase>/<module>_<run_id>.json and appends a
one-line summary to results/run_history.csv so progress is greppable without opening
individual logs (see the goal's Logging section).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Repo root = two levels up from this file (src/utils/logging.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
RUN_HISTORY = RESULTS_DIR / "run_history.csv"

# Make Windows consoles tolerate non-ASCII (CCPD Chinese chars, plate text, arrows).
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def new_run_id() -> str:
    """Short, sortable run id: YYYYmmdd-HHMMSS-xxxx."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)


def write_run_log(phase: str, module: str, run_id: str, payload: dict) -> Path:
    """Write logs/<phase>/<module>_<run_id>.json. Returns the path."""
    out_dir = LOGS_DIR / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{module}_{run_id}.json"
    record = {"timestamp": utcnow(), "phase": phase, "module": module, "run_id": run_id, **payload}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_run_history(row: dict) -> None:
    """Append a one-line summary to results/run_history.csv (creates header if new)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", "run_id", "phase", "module", "dataset", "model",
            "metric", "value", "target", "pass_fail", "note"]
    new_file = not RUN_HISTORY.exists()
    with RUN_HISTORY.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({"timestamp": utcnow(), **row})
