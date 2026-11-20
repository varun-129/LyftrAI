"""
config.py – reads environment variables via pydantic-settings.
Key values are NEVER logged or returned in responses.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Raw comma-separated string from env
    llm_api_keys: str = ""

    # Other settings
    database_url: str = "sqlite:////data/app.db"
    allowed_origins: str = "http://localhost:5173,http://localhost:3000,https://lyftr-ai-eosin.vercel.app"
    enable_failure_simulation: bool = False
    llm_model: str = "gemini/gemini-3.5-flash-lite"
    llm_api_base: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    default_system_prompt: str = "You are a helpful assistant."
    cache_dir: str = "/data/cache"
    mock_llm: bool = False

    # ------------------------------------------------------------------ #
    # Parsed derivatives (populated by validators)                         #
    # ------------------------------------------------------------------ #
    _key_pool: List[str] = []          # ordered list of raw key values
    _key_ids: List[str] = []           # ["key_1", "key_2", ...]
    _origins_list: List[str] = []      # parsed ALLOWED_ORIGINS

    @field_validator("enable_failure_simulation", mode="before")
    @classmethod
    def _parse_bool(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == "true"

    @field_validator("mock_llm", mode="before")
    @classmethod
    def _parse_mock_bool(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == "true"

    @model_validator(mode="after")
    def _parse_composites(self) -> "Settings":
        # Parse key pool – ignore empty entries, never store IDs in logs
        raw_keys = [k.strip() for k in self.llm_api_keys.split(",") if k.strip()]
        object.__setattr__(self, "_key_pool", raw_keys)
        object.__setattr__(self, "_key_ids", [f"key_{i + 1}" for i in range(len(raw_keys))])

        # Parse allowed origins (strip whitespace and any accidental trailing slashes)
        origins = [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]
        object.__setattr__(self, "_origins_list", origins)

        return self

    # ------------------------------------------------------------------ #
    # Public accessors (safe – never expose raw key values)               #
    # ------------------------------------------------------------------ #

    @property
    def key_pool(self) -> List[str]:
        """Ordered list of raw API key strings."""
        return list(self._key_pool)

    @property
    def key_ids(self) -> List[str]:
        """Ordered list of key identifiers: ['key_1', 'key_2', ...]."""
        return list(self._key_ids)

    @property
    def origins_list(self) -> List[str]:
        """Parsed list of allowed CORS origins."""
        return list(self._origins_list)

    def key_by_id(self, key_id: str) -> str | None:
        """Return the raw key value for a given key_id, or None if not found."""
        try:
            idx = self._key_ids.index(key_id)
            return self._key_pool[idx]
        except (ValueError, IndexError):
            return None

    def key_id_for_index(self, index: int) -> str | None:
        """Return key_id for a zero-based index, or None if out of range."""
        if 0 <= index < len(self._key_ids):
            return self._key_ids[index]
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
