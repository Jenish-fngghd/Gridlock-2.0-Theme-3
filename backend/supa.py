"""Supabase client (service-role). Lazily created; never import keys into the frontend."""
from __future__ import annotations
from functools import lru_cache
from .config import settings


@lru_cache(maxsize=1)
def get_client():
    if not settings.supabase_configured:
        raise RuntimeError(
            "Supabase not configured: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
    from supabase import create_client
    return create_client(settings.supabase_url, settings.service_key)
