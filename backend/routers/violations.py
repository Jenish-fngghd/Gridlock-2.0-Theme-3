"""POST /api/violations/{id}/review — human review-band decision + chained audit entry."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..audit import chain_hash

router = APIRouter(prefix="/api", tags=["violations"])

_STATUS = {"confirm": "confirmed", "reject": "rejected", "escalate": "pending"}


class ReviewIn(BaseModel):
    action: str  # confirm | reject | escalate
    reviewer_label: str | None = None
    notes: str | None = None


@router.post("/violations/{violation_id}/review")
def review(violation_id: str, body: ReviewIn):
    if body.action not in _STATUS:
        raise HTTPException(status_code=422, detail="action must be confirm|reject|escalate")
    if not settings.supabase_configured:
        raise HTTPException(status_code=503, detail="supabase not configured")

    from ..supa import get_client

    sb = get_client()
    sb.table("review_actions").insert(
        {
            "violation_id": violation_id,
            "action": body.action,
            "reviewer_label": body.reviewer_label,
            "notes": body.notes,
        }
    ).execute()
    sb.table("violations").update({"status": _STATUS[body.action]}).eq("id", violation_id).execute()

    # append to the hash chain
    last = (
        sb.table("evidence_audit")
        .select("sha256")
        .eq("violation_id", violation_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    prev = last.data[0]["sha256"] if last.data else None
    payload = {"action": body.action, "notes": body.notes}
    h = chain_hash(payload, prev)
    sb.table("evidence_audit").insert(
        {
            "violation_id": violation_id,
            "event_type": "reviewed",
            "payload": payload,
            "sha256": h,
            "prev_hash": prev,
        }
    ).execute()

    return {"ok": True, "status": _STATUS[body.action]}
