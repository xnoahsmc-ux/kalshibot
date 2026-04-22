from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    env: str
    api_key_id: str | None
    api_private_key_path: str | None
    email: str | None
    password: str | None
    dry_run: bool
    max_position_usd: float
    bankroll_usd: float
    min_edge: float
    poll_seconds: int
    base_url_override: str | None = None

    @property
    def base_url(self) -> str:
        if self.base_url_override:
            return self.base_url_override
        if self.env == "prod":
            return "https://api.elections.kalshi.com/trade-api/v2"
        return "https://demo-api.kalshi.co/trade-api/v2"


def load_config() -> Config:
    def _get(k: str, default: str | None = None) -> str | None:
        v = os.getenv(k, default)
        return v if v not in ("", None) else default

    return Config(
        env=_get("KALSHI_ENV", "demo") or "demo",
        api_key_id=_get("KALSHI_API_KEY_ID"),
        api_private_key_path=_get("KALSHI_API_PRIVATE_KEY_PATH"),
        email=_get("KALSHI_EMAIL"),
        password=_get("KALSHI_PASSWORD"),
        dry_run=(_get("KALSHIBOT_DRY_RUN", "true") or "true").lower() == "true",
        max_position_usd=float(_get("KALSHIBOT_MAX_POSITION_USD", "50") or 50),
        bankroll_usd=float(_get("KALSHIBOT_BANKROLL_USD", "1000") or 1000),
        min_edge=float(_get("KALSHIBOT_MIN_EDGE", "0.03") or 0.03),
        poll_seconds=int(_get("KALSHIBOT_POLL_SECONDS", "30") or 30),
        base_url_override=_get("KALSHI_BASE_URL"),
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
