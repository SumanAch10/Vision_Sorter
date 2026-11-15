"""Validated runtime settings; credentials are never included in model metadata."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    allow_anonymous: bool = False
    request_timeout_s: float = 15.0
    requests_per_minute: int = 60
    max_clients: int = 1024

    def __post_init__(self):
        if self.api_key and len(self.api_key) < 16:
            raise ValueError("SORTER_API_KEY must contain at least 16 characters")
        if (
            not 1 <= self.request_timeout_s <= 120
            or not 1 <= self.requests_per_minute <= 10000
            or not 1 <= self.max_clients <= 10000
        ):
            raise ValueError("Invalid timeout, rate limit, or client capacity")

    @classmethod
    def from_env(cls):
        key = os.getenv("SORTER_API_KEY", "")
        keyfile = os.getenv("SORTER_API_KEY_FILE")
        if keyfile:
            if key:
                raise ValueError("Configure API key or key file, not both")
            key = Path(keyfile).read_text().strip()
            if not key:
                raise ValueError("API key file is empty")
        return cls(
            api_key=key,
            allow_anonymous=os.getenv("SORTER_ALLOW_ANONYMOUS", "false").lower() == "true",
            request_timeout_s=float(os.getenv("SORTER_REQUEST_TIMEOUT", "15")),
            requests_per_minute=int(os.getenv("SORTER_RATE_LIMIT", "60")),
        )

    def validate_binding(self, host):
        if (
            host not in ("127.0.0.1", "localhost", "::1")
            and not self.api_key
            and not self.allow_anonymous
        ):
            raise ValueError(
                "External binding requires SORTER_API_KEY_FILE/SORTER_API_KEY or explicit SORTER_ALLOW_ANONYMOUS=true"
            )
