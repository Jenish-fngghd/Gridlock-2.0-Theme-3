"""Supabase Storage helpers for evidence images."""
from __future__ import annotations
from .supa import get_client


def upload_bytes(bucket: str, path: str, data: bytes, content_type: str = "image/jpeg") -> str:
    client = get_client()
    client.storage.from_(bucket).upload(
        path, data, {"content-type": content_type, "upsert": "true"}
    )
    return client.storage.from_(bucket).get_public_url(path)
