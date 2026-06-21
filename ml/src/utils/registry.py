"""Model/decision registry — logs every model choice, swap, and paper consulted.

Phase 6 (escalation) requires logging every candidate considered — paper/source, what it
offers, why chosen or rejected — and updating the selection-justification doc. This util
appends structured entries to results/model_registry.jsonl (machine-readable) so those
updates are auditable and greppable.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging import RESULTS_DIR, utcnow

REGISTRY = RESULTS_DIR / "model_registry.jsonl"


def record(module: str, model: str, decision: str, *, source: str = "",
           offers: str = "", reason: str = "", supersedes: str = "") -> None:
    """Append one registry entry.

    decision: one of {selected, benchmarked, rejected, superseded, swapped_in}.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": utcnow(),
        "module": module,
        "model": model,
        "decision": decision,
        "source": source,      # arXiv id / repo / HF id / URL
        "offers": offers,      # what it provides
        "reason": reason,      # why chosen or rejected
        "supersedes": supersedes,  # prior model id this replaces (Phase 6)
    }
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_all() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()]
