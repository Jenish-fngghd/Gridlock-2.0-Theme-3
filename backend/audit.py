"""Hash-chain helper for the tamper-evident evidence_audit log."""
from __future__ import annotations
import hashlib
import json


def chain_hash(payload: dict, prev_hash: str | None) -> str:
    """sha256( canonical(payload) || prev_hash ) — links each audit row to the previous one."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")) + (prev_hash or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
