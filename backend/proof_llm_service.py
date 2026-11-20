#!/usr/bin/env python3
"""
proof_llm_service.py – Standalone proof script for llm_service.py

Demonstrates:
  1. LIVE_INFERENCE: first call returns real (mock) LLM answer
  2. LLM_CACHE:      identical second call returns cached reply (same text)
  3. KEY FAILOVER:   key_1=429 simulation → key_id=key_2, fallback_attempts=1
  4. ALL FAIL:       *=429 simulation → AllProvidersUnavailable (-> 503)

Run from /backend:
    PYTHONPATH=. MOCK_LLM=true LLM_API_KEYS=fake1,fake2 DATABASE_URL=sqlite:////tmp/proof.db \
    python proof_llm_service.py
"""
import asyncio
import os
import sys
import tempfile

# Minimal env so Settings doesn't complain
os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("LLM_API_KEYS", "fake_key_1,fake_key_2")
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/proof_llm.db")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")

# Use a fresh temp dir for each proof run so cache is clean
_tmpdir = tempfile.mkdtemp(prefix="proof_cache_")
os.environ["CACHE_DIR"] = _tmpdir


from app.llm_service import (
    AllProvidersUnavailable,
    CompletionResult,
    get_completion,
    init_llm_service,
    parse_simulate_header,
)


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def show(result: CompletionResult) -> None:
    print(f"  reply            : {result.reply[:80]!r}")
    print(f"  resolution_source: {result.resolution_source}")
    print(f"  model            : {result.model}")
    print(f"  key_id           : {result.key_id}")
    print(f"  fallback_attempts: {result.fallback_attempts}")
    print(f"  input_tokens     : {result.input_tokens}")
    print(f"  output_tokens    : {result.output_tokens}")


async def main() -> None:
    init_llm_service()
    print(f"Cache dir: {_tmpdir}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is 2 + 2?"},
    ]

    # ------------------------------------------------------------------ #
    # TEST 1: First call → LIVE_INFERENCE                                 #
    # ------------------------------------------------------------------ #
    banner("TEST 1 – LIVE_INFERENCE (first call)")
    r1 = await get_completion(messages, simulate={})
    show(r1)
    assert r1.resolution_source == "LIVE_INFERENCE", "Expected LIVE_INFERENCE"
    assert r1.key_id is not None, "Expected a key_id"
    assert r1.reply, "Expected non-empty reply"
    print("  ✅ PASS")

    # ------------------------------------------------------------------ #
    # TEST 2: Identical second call → LLM_CACHE                          #
    # ------------------------------------------------------------------ #
    banner("TEST 2 – LLM_CACHE (identical second call)")
    r2 = await get_completion(messages, simulate={})
    show(r2)
    assert r2.resolution_source == "LLM_CACHE", f"Expected LLM_CACHE, got {r2.resolution_source}"
    assert r2.reply == r1.reply, "Cached reply must match first reply"
    assert r2.key_id is None, "Cache hit should have key_id=None"
    assert r2.fallback_attempts == 0, "Cache hit should have fallback_attempts=0"
    print("  ✅ PASS")

    # ------------------------------------------------------------------ #
    # TEST 3: Different message → NOT from cache                          #
    # ------------------------------------------------------------------ #
    banner("TEST 3 – Different message → LIVE_INFERENCE")
    different = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the capital of France?"},
    ]
    r3 = await get_completion(different, simulate={})
    show(r3)
    assert r3.resolution_source == "LIVE_INFERENCE", "Different message must NOT come from cache"
    print("  ✅ PASS")

    # ------------------------------------------------------------------ #
    # TEST 4: key_1=429 → failover to key_2                              #
    # ------------------------------------------------------------------ #
    banner("TEST 4 – key_1=429 simulation → key_id=key_2, fallback_attempts=1")
    sim = parse_simulate_header("key_1=429")
    new_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is 3 + 3?"},
    ]
    r4 = await get_completion(new_messages, simulate=sim)
    show(r4)
    assert r4.resolution_source == "LIVE_INFERENCE", "Should get real answer via key_2"
    assert r4.key_id == "key_2", f"Expected key_2, got {r4.key_id!r}"
    assert r4.fallback_attempts == 1, f"Expected 1 fallback, got {r4.fallback_attempts}"
    assert r4.reply, "Reply must not be empty"
    print("  ✅ PASS")

    # ------------------------------------------------------------------ #
    # TEST 5: key_1=500 → failover to key_2                              #
    # ------------------------------------------------------------------ #
    banner("TEST 5 – key_1=500 simulation → key_id=key_2")
    sim2 = parse_simulate_header("key_1=500")
    new_messages2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Name any planet."},
    ]
    r5 = await get_completion(new_messages2, simulate=sim2)
    show(r5)
    assert r5.key_id == "key_2", f"Expected key_2, got {r5.key_id!r}"
    print("  ✅ PASS")

    # ------------------------------------------------------------------ #
    # TEST 6: *=429 (all keys fail) → AllProvidersUnavailable            #
    # ------------------------------------------------------------------ #
    banner("TEST 6 – *=429 (all keys) → AllProvidersUnavailable (503)")
    sim3 = parse_simulate_header("*=429")
    fail_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "This should 503."},
    ]
    try:
        await get_completion(fail_messages, simulate=sim3)
        print("  ❌ FAIL – expected AllProvidersUnavailable")
        sys.exit(1)
    except AllProvidersUnavailable as e:
        print(f"  Caught AllProvidersUnavailable: {e}")
        print("  ✅ PASS")

    # ------------------------------------------------------------------ #
    # TEST 7: parse_simulate_header edge cases                           #
    # ------------------------------------------------------------------ #
    banner("TEST 7 – parse_simulate_header edge cases")
    assert parse_simulate_header("") == {}
    assert parse_simulate_header("bad_pair") == {}
    assert parse_simulate_header("key_1=999") == {}      # 999 not in allowed
    assert parse_simulate_header("key_1=abc") == {}      # non-int
    assert parse_simulate_header("key_1=429,key_2=503,*=500") == {
        "key_1": 429, "key_2": 503, "*": 500
    }
    print("  ✅ PASS")

    banner("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
