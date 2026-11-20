"""
schemas.py – Pydantic request/response models.
Field names are the API contract; do not rename them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    status: str           # "ok" | "degraded"
    keys_configured: int  # number of keys loaded
    cache_enabled: bool   # always true when service is up


# --------------------------------------------------------------------------- #
# Chat – request                                                               #
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str         # stripped + length-validated below
    system_prompt: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("message", mode="before")
    @classmethod
    def _strip_and_validate_message(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("message must be a string")
        stripped = v.strip()
        if len(stripped) == 0:
            raise ValueError("message must not be empty or whitespace-only")
        if len(stripped) > 4000:
            raise ValueError(
                f"message must be at most 4000 characters after trimming "
                f"(got {len(stripped)})"
            )
        return stripped


# --------------------------------------------------------------------------- #
# Chat – response                                                              #
# --------------------------------------------------------------------------- #

class ChatMetadata(BaseModel):
    resolution_source: str       # "LIVE_INFERENCE" | "LLM_CACHE"
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]        # null on LLM_CACHE
    fallback_attempts: int       # 0 on LLM_CACHE


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


# --------------------------------------------------------------------------- #
# History                                                                      #
# --------------------------------------------------------------------------- #

class HistoryMessage(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None   # only on assistant messages


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[HistoryMessage]


# --------------------------------------------------------------------------- #
# Error (always {"detail": "..."})                                             #
# --------------------------------------------------------------------------- #

class ErrorDetail(BaseModel):
    detail: str
