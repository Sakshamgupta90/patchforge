"""Settings, read from environment variables (and a .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    github_app_id: str = ""
    github_private_key: str = ""
    github_webhook_secret: str = ""
    sandbox: str = "docker"
    sandbox_image: str = "python:3.12-slim"
    agent: str = "stub"
    api_key: str = ""
    cors_origins: list[str] = field(default_factory=list)

    @property
    def github_configured(self) -> bool:
        return bool(self.github_app_id and self.github_private_key)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        key = os.getenv("GITHUB_PRIVATE_KEY", "")
        key_path = os.getenv("GITHUB_PRIVATE_KEY_PATH", "")
        if not key and key_path:
            path = Path(key_path).expanduser()
            if not path.exists():
                raise SystemExit(f"GITHUB_PRIVATE_KEY_PATH points at {path}, which doesn't exist")
            key = path.read_text()
        return cls(
            github_app_id=os.getenv("GITHUB_APP_ID", ""),
            github_private_key=key.replace("\\n", "\n"),
            github_webhook_secret=os.getenv("GITHUB_WEBHOOK_SECRET", ""),
            sandbox=os.getenv("PATCHFORGE_SANDBOX", "docker"),
            sandbox_image=os.getenv("SANDBOX_IMAGE", "python:3.12-slim"),
            agent=os.getenv("PATCHFORGE_AGENT", "stub"),
            api_key=os.getenv("PATCHFORGE_API_KEY", ""),
            cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()],
        )
