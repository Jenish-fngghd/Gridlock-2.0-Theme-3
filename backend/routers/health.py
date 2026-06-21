from fastapi import APIRouter
from ..config import settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "inference_mode": settings.inference_mode,
        "supabase_configured": settings.supabase_configured,
    }
