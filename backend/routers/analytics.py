"""GET /api/analytics/summary — dashboard aggregates from the SQL views."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import settings

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/summary")
def summary():
    if not settings.supabase_configured:
        raise HTTPException(status_code=503, detail="supabase not configured")

    from ..supa import get_client

    sb = get_client()
    return {
        "by_type": sb.table("v_violations_by_type").select("*").execute().data,
        "bands": sb.table("v_confidence_bands").select("*").execute().data,
        "by_day": sb.table("v_violations_by_day").select("*").execute().data,
        "repeat_offenders": sb.table("v_repeat_offenders").select("*").limit(10).execute().data,
    }
