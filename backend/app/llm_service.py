"""
llm_service.py – LLM gateway with key failover, response caching, and failure simulation.

Architecture:
  - Cache: Our own disk cache layer (diskcache) keyed on sha256(model + messages).
           This gives full control over the key (api_key / session_id never affect it)
           and survives process restarts. litellm.Cache(type="local") is also set
           so litellm's internal helpers remain active.
  - Failover: Explicit ordered loop over key_1, key_2, … – no LiteLLM Router.
  - Simulation: X-Simulate-Failure header raises the matching litellm exception
                inside the loop so it flows through the same failover path.
  - Mock: MOCK_LLM=true passes mock_response= to litellm, no real API calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import diskcache
import litellm
import litellm.exceptions as llm_ex
from litellm.caching.caching import Cache as LiteLLMCache
from litellm import ModelResponse

from .config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once at startup via init_llm_service)
# ---------------------------------------------------------------------------
_disk_cache: Optional[diskcache.Cache] = None


def init_llm_service() -> None:
    """
    Called once from the FastAPI lifespan.
    Sets up litellm.cache (local, so litellm internals are happy) and opens
    the persistent diskcache store for our own cache layer.
    """
    global _disk_cache

    settings = get_settings()

    # litellm global cache – local in-memory (disk type not supported in 1.110.x)
    litellm.cache = LiteLLMCache(type="local")

    # Our persistent disk cache (survives restarts)
    cache_dir = settings.cache_dir
    try:
        os.makedirs(cache_dir, exist_ok=True)
        _disk_cache = diskcache.Cache(cache_dir)
    except (PermissionError, OSError):
        fallback_dir = "/tmp/data/cache"
        os.makedirs(fallback_dir, exist_ok=True)
        _disk_cache = diskcache.Cache(fallback_dir)
        logger.warning("Cache dir %s not writable, falling back to %s", cache_dir, fallback_dir)
    logger.info("LLM service initialised. Cache dir: %s", cache_dir)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Canonical, stable cache key: sha256(model + normalised messages list).
    api_key, session_id, timestamps – nothing else is mixed in.
    """
    raw = json.dumps({"model": model, "messages": messages}, sort_keys=True,
                     ensure_ascii=False)
    return "chatbot:" + hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(key: str) -> Optional[ModelResponse]:
    """Return a reconstructed ModelResponse from disk cache, or None."""
    if _disk_cache is None:
        return None
    raw = _disk_cache.get(key)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return ModelResponse(**data)
    except Exception:
        logger.warning("Cache entry corrupt for key %s – ignoring.", key[:20])
        return None


def _put_cached(key: str, response: ModelResponse) -> None:
    """Persist a ModelResponse to disk cache."""
    if _disk_cache is None:
        return
    try:
        _disk_cache[key] = json.dumps(response.model_dump())
    except Exception:
        logger.warning("Failed to write cache entry for key %s.", key[:20])


# ---------------------------------------------------------------------------
# Failure simulation parser
# ---------------------------------------------------------------------------

_SIM_EXCEPTIONS: Dict[int, type] = {
    429: llm_ex.RateLimitError,
    500: llm_ex.InternalServerError,
    502: llm_ex.InternalServerError,
    503: llm_ex.ServiceUnavailableError,
}


def parse_simulate_header(header_value: str) -> Dict[str, int]:
    """
    Parse "key_1=429,key_2=500,*=503" -> {"key_1": 429, "key_2": 500, "*": 503}.
    Silently ignores malformed or unknown pairs.
    """
    result: Dict[str, int] = {}
    if not header_value:
        return result
    for part in header_value.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        kid, _, code_str = part.partition("=")
        kid = kid.strip()
        code_str = code_str.strip()
        try:
            code = int(code_str)
        except ValueError:
            continue
        if code not in _SIM_EXCEPTIONS:
            continue
        result[kid] = code
    return result


def _raise_simulated(key_id: str, code: int, model: str) -> None:
    """Raise the litellm exception corresponding to the simulated status code."""
    exc_cls = _SIM_EXCEPTIONS[code]
    msg = f"[simulated] {key_id} returned HTTP {code}"
    if exc_cls is llm_ex.RateLimitError:
        raise exc_cls(message=msg, llm_provider="openai", model=model)
    elif exc_cls is llm_ex.ServiceUnavailableError:
        raise exc_cls(message=msg, llm_provider="openai", model=model)
    else:  # InternalServerError (covers 500 and 502)
        raise exc_cls(message=msg, llm_provider="openai", model=model)


# ---------------------------------------------------------------------------
# Failover-triggering exception predicate
# ---------------------------------------------------------------------------

_FAILOVER_TYPES = (
    llm_ex.RateLimitError,
    llm_ex.InternalServerError,
    llm_ex.ServiceUnavailableError,
    llm_ex.APIConnectionError,
    llm_ex.Timeout,
    llm_ex.AuthenticationError,
    llm_ex.BadRequestError,
)


def _is_failover_exc(exc: Exception) -> bool:
    if isinstance(exc, _FAILOVER_TYPES):
        return True
    code = getattr(exc, "status_code", None)
    if code is not None and (code in (400, 401, 403, 429) or code >= 500):
        return True
    return False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    reply: str
    resolution_source: str            # "LLM_CACHE" | "LIVE_INFERENCE"
    model: str
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]             # None on cache hit
    fallback_attempts: int            # keys that failed before success


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AllProvidersUnavailable(Exception):
    """Raised when every key in the pool fails."""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def get_completion(
    messages: List[Dict[str, str]],
    simulate: Dict[str, int],          # parsed from X-Simulate-Failure header
) -> CompletionResult:
    """
    1. Check disk cache (unless simulation is active -> bypass read).
    2. Loop over keys in order; raise simulated exceptions where instructed.
    3. Call litellm.acompletion with each key (num_retries=0, timeout=30).
    4. On success: write to cache, return LIVE_INFERENCE result.
    5. If all keys fail: raise AllProvidersUnavailable.
    """
    settings = get_settings()
    model = settings.llm_model
    key_pool = settings.key_pool      # raw values
    key_ids = settings.key_ids        # ["key_1", "key_2", ...]

    if not key_pool and not settings.mock_llm:
        raise AllProvidersUnavailable("No API keys configured.")

    # ---- Cache lookup (skipped when any simulation is active) ----
    skip_cache_read = bool(simulate)
    ck = _cache_key(model, messages)

    if not skip_cache_read:
        cached = _get_cached(ck)
        if cached is not None:
            logger.info("Cache HIT for key %s", ck[:20])
            usage = cached.usage
            return CompletionResult(
                reply=cached.choices[0].message.content or "",
                resolution_source="LLM_CACHE",
                model=cached.model or model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                key_id=None,
                fallback_attempts=0,
            )

    # ---- Key failover loop ----
    fallback_attempts = 0
    last_exc: Optional[Exception] = None

    # In mock mode with no real keys, synthesize a single-key pool
    effective_pool = list(zip(key_ids, key_pool))
    if settings.mock_llm and not effective_pool:
        effective_pool = [("key_1", "mock")]

    for key_id, raw_key in effective_pool:
        # Apply failure simulation
        sim_code = simulate.get(key_id) or simulate.get("*")
        if sim_code is not None:
            logger.info("Simulating HTTP %d for %s", sim_code, key_id)
            try:
                _raise_simulated(key_id, sim_code, model)
            except Exception as exc:
                if _is_failover_exc(exc):
                    last_exc = exc
                    fallback_attempts += 1
                    continue
                raise

        # Determine actual model and api_base for the provider call
        call_model = model
        call_api_base = getattr(settings, "llm_api_base", None)
        if call_api_base and (model.startswith("gemini/") or not model.startswith("openai/")):
            # Route to Google's OpenAI-compatible endpoint using Bearer auth
            clean_name = model.split("/", 1)[-1]
            call_model = f"openai/{clean_name}"

        # Build call kwargs
        call_kwargs: Dict[str, Any] = {
            "model": call_model,
            "messages": messages,
            "num_retries": 0,
            "timeout": 30,
        }
        if call_api_base:
            call_kwargs["api_base"] = call_api_base

        if settings.mock_llm:
            call_kwargs["mock_response"] = (
                f"[MOCK] I am a mock LLM. You said: {messages[-1]['content']}"
            )
        else:
            call_kwargs["api_key"] = raw_key

        try:
            response: ModelResponse = await litellm.acompletion(**call_kwargs)
        except llm_ex.NotFoundError as exc:
            # 404 model not found is a CONFIG error: log clearly on server and return 503
            logger.error("Configuration error: model '%s' not found or no longer available (HTTP 404): %s", model, exc)
            raise AllProvidersUnavailable(f"Model {model} not found or no longer available.") from exc
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if _is_failover_exc(exc):
                logger.warning("Key %s failed: status=%s (%s)", key_id, status_code, type(exc).__name__)
                last_exc = exc
                fallback_attempts += 1
                continue
            # Non-retriable error – bubble up
            raise

        reply_text = response.choices[0].message.content or ""
        # Empty reply from model -> treat as key failure and fail over
        if not reply_text.strip():
            logger.warning("Key %s returned empty reply, failing over", key_id)
            last_exc = RuntimeError(f"Key {key_id} returned empty reply")
            fallback_attempts += 1
            continue

        usage = response.usage
        resp_model = response.model or model

        if settings.mock_llm:
            # Compute realistic token counts proportional to actual content so that
            # follow-up turns (which include history) show MORE input_tokens than the
            # first turn.  chars/4 is a rough but deterministic approximation.
            input_tok = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)
            output_tok = max(1, len(reply_text) // 4)
            # Patch usage on the response so the cache stores accurate numbers
            if response.usage is not None:
                response.usage.prompt_tokens = input_tok
                response.usage.completion_tokens = output_tok
                response.usage.total_tokens = input_tok + output_tok
        else:
            input_tok = getattr(usage, "prompt_tokens", 0) or 0
            output_tok = getattr(usage, "completion_tokens", 0) or 0

        # Write to disk cache (even for simulated requests – real answer is cacheable)
        _put_cached(ck, response)
        logger.info("LIVE_INFERENCE via %s (fallbacks=%d)", key_id, fallback_attempts)

        return CompletionResult(
            reply=reply_text,
            resolution_source="LIVE_INFERENCE",
            model=resp_model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            key_id=key_id,
            fallback_attempts=fallback_attempts,
        )

    # All keys exhausted
    logger.error("All providers unavailable. Last error: %s", last_exc)
    raise AllProvidersUnavailable(str(last_exc) if last_exc else "All keys failed.")
