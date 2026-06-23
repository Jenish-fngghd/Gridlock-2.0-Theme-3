"""Runtime settings, read from environment (.env loaded by the process manager)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field

try:  # .env is the source of truth; override=True lets a redeployed .env update a running
      # container's settings (process env vars set at container creation otherwise win).
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass


@dataclass
class Settings:
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    service_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "50")))
    inference_mode: str = field(default_factory=lambda: os.getenv("INFERENCE_MODE", "mock"))
    allowed_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
            if o.strip()
        ]
    )

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.service_key)


settings = Settings()
