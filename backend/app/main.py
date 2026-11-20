"""
main.py – FastAPI application entry-point.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import (
    get_history,
    get_session_system_prompt,
    init_db,
    ping_db,
    save_exchange,
    session_exists,
    set_session_system_prompt,
)
from .llm_service import AllProvidersUnavailable, get_completion, init_llm_service, parse_simulate_header
from .schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan                                                                     #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up – initialising database …")
    await init_db()
    logger.info("Database ready.")
    init_llm_service()
    logger.info("LLM service ready.")
    yield
    logger.info("Shutting down.")


# --------------------------------------------------------------------------- #
# App instance                                                                 #
# --------------------------------------------------------------------------- #

settings = get_settings()

app = FastAPI(
    title="LLM Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------- #
# CORS                                                                         #
# --------------------------------------------------------------------------- #

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-Simulate-Failure"],
    expose_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Exception handlers                                                           #
# --------------------------------------------------------------------------- #

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    detail = "; ".join(
        f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "internal server error"},
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "not found"},
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        content={"detail": "method not allowed"},
    )


# --------------------------------------------------------------------------- #
# GET /health                                                                   #
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["meta"])
async def health():
    cfg = get_settings()
    db_ok = await ping_db()
    keys_ok = len(cfg.key_pool) > 0 or cfg.mock_llm

    if not db_ok or not keys_ok:
        reason = []
        if not db_ok:
            reason.append("database unreachable")
        if not keys_ok:
            reason.append("no API keys configured")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "; ".join(reason)},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "keys_configured": len(cfg.key_pool),
            "cache_enabled": True,
        },
    )


# --------------------------------------------------------------------------- #
# POST /chat                                                                   #
# --------------------------------------------------------------------------- #

@app.post("/chat", tags=["chat"])
async def chat(
    body: ChatRequest,
    request: Request,
    x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
):
    start = time.perf_counter()
    cfg = get_settings()

    # ---- Failure simulation (only when ENABLE_FAILURE_SIMULATION=true) ----
    simulate: dict = {}
    if cfg.enable_failure_simulation and x_simulate_failure:
        simulate = parse_simulate_header(x_simulate_failure)

    # ---- Resolve system prompt ----
    if body.system_prompt is not None:
        # Caller supplied one → store it and use it
        await set_session_system_prompt(body.session_id, body.system_prompt)
        system_prompt = body.system_prompt
    else:
        stored = await get_session_system_prompt(body.session_id)
        system_prompt = stored if stored is not None else cfg.default_system_prompt

    # ---- Build message list: [system] + history + [new user message] ----
    history_rows = await get_history(body.session_id)
    messages = [{"role": "system", "content": system_prompt}]
    for row in history_rows:
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": body.message})

    # ---- Call LLM service ----
    try:
        result = await get_completion(messages, simulate=simulate)
    except AllProvidersUnavailable:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "all providers unavailable"},
        )
    except Exception as exc:
        logger.exception("Unexpected error in /chat for session %s", body.session_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    # ---- Persist (only on success) ----
    meta_dict = {
        "resolution_source": result.resolution_source,
        "model": result.model,
        "latency_ms": latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "key_id": result.key_id,
        "fallback_attempts": result.fallback_attempts,
    }
    await save_exchange(
        session_id=body.session_id,
        user_content=body.message,
        assistant_content=result.reply,
        metadata=meta_dict,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "session_id": body.session_id,
            "reply": result.reply,
            "metadata": meta_dict,
        },
    )


# --------------------------------------------------------------------------- #
# GET /sessions/{session_id}/messages                                          #
# --------------------------------------------------------------------------- #

@app.get("/sessions/{session_id}/messages", tags=["history"])
async def get_session_messages(session_id: str):
    # 404 if session was never created
    exists = await session_exists(session_id)
    if not exists:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "session not found"},
        )

    rows = await get_history(session_id)
    messages = []
    for row in rows:
        msg: dict = {
            "role": row["role"],
            "content": row["content"],
        }
        if row["role"] == "assistant" and row.get("metadata"):
            try:
                msg["metadata"] = json.loads(row["metadata"])
            except (TypeError, ValueError):
                msg["metadata"] = None
        else:
            msg["metadata"] = None
        messages.append(msg)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "session_id": session_id,
            "messages": messages,
        },
    )
