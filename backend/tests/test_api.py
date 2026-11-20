"""
test_api.py – comprehensive pytest suite for the LLM Chatbot backend.

Covers every requirement listed in Part 4:
  - /health shape
  - Chat happy path (all metadata fields, types)
  - Cache hits (same message/session → LLM_CACHE; diff message → LIVE_INFERENCE)
  - Per-request token counts (not accumulated; follow-up > first)
  - Multi-turn memory, session isolation, history shape
  - Persistence across app restarts (same DB file)
  - Validation (422 for all bad inputs)
  - Failover (key_1=429/500 → key_2; all keys fail → 503)
  - Simulation ignored when ENABLE_FAILURE_SIMULATION=false
  - 404 for unknown sessions
  - System prompt changes cache key
  - API key values never appear in any response
"""
from __future__ import annotations

import json
import pytest
from httpx import ASGITransport, AsyncClient

# ===========================================================================
# /health
# ===========================================================================

class TestHealth:

    async def test_shape_200(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    async def test_all_fields_present(self, client):
        data = (await client.get("/health")).json()
        assert "status"          in data
        assert "keys_configured" in data
        assert "cache_enabled"   in data

    async def test_field_types_and_values(self, client):
        data = (await client.get("/health")).json()
        assert data["status"]           == "ok"
        assert isinstance(data["keys_configured"], int)
        assert data["keys_configured"]  == 2
        assert data["cache_enabled"]    is True


# ===========================================================================
# POST /chat – happy path
# ===========================================================================

class TestChatHappyPath:

    async def test_200_on_valid_request(self, client):
        r = await client.post("/chat", json={
            "session_id": "hp_1",
            "message":    "Hello there",
        })
        assert r.status_code == 200

    async def test_all_metadata_fields_present(self, client):
        data = (await client.post("/chat", json={
            "session_id": "hp_2",
            "message":    "What is 2+2?",
        })).json()

        assert "session_id" in data
        assert "reply"      in data
        assert "metadata"   in data

        meta = data["metadata"]
        for field in ("resolution_source", "model", "latency_ms",
                      "input_tokens", "output_tokens", "key_id", "fallback_attempts"):
            assert field in meta, f"missing metadata field: {field}"

    async def test_metadata_field_types(self, client):
        meta = (await client.post("/chat", json={
            "session_id": "hp_3",
            "message":    "Type check",
        })).json()["metadata"]

        assert meta["resolution_source"] in ("LIVE_INFERENCE", "LLM_CACHE")
        assert isinstance(meta["model"],             str) and meta["model"]
        assert isinstance(meta["latency_ms"],        int) and meta["latency_ms"] >= 0
        assert isinstance(meta["input_tokens"],      int) and meta["input_tokens"] >= 0
        assert isinstance(meta["output_tokens"],     int) and meta["output_tokens"] >= 0
        assert isinstance(meta["fallback_attempts"], int) and meta["fallback_attempts"] >= 0
        # key_id is str or None
        assert meta["key_id"] is None or isinstance(meta["key_id"], str)

    async def test_reply_non_empty(self, client):
        data = (await client.post("/chat", json={
            "session_id": "hp_4",
            "message":    "Any question",
        })).json()
        assert data["reply"] and len(data["reply"]) > 0

    async def test_session_id_echoed(self, client):
        data = (await client.post("/chat", json={
            "session_id": "echo_sess",
            "message":    "Echo test",
        })).json()
        assert data["session_id"] == "echo_sess"

    async def test_first_call_live_inference(self, client):
        data = (await client.post("/chat", json={
            "session_id":   "live_1",
            "message":      "First call unique xyz_live_1",
            "system_prompt": "You are a helpful assistant.",
        })).json()
        assert data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert data["metadata"]["key_id"] is not None


# ===========================================================================
# Caching
# ===========================================================================

class TestCache:

    async def test_same_message_same_system_prompt_new_session_cache_hit(self, client):
        """Second session with identical messages → LLM_CACHE."""
        payload = {
            "message":       "cache test unique message aabbcc112233",
            "system_prompt": "You are a helpful assistant.",
        }
        r1 = (await client.post("/chat", json={"session_id": "cache_a", **payload})).json()
        assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"

        r2 = (await client.post("/chat", json={"session_id": "cache_b", **payload})).json()
        assert r2["metadata"]["resolution_source"]    == "LLM_CACHE"
        assert r2["reply"]                             == r1["reply"],       "reply must be identical"
        assert r2["metadata"]["key_id"]                is None,              "key_id null on cache hit"
        assert r2["metadata"]["fallback_attempts"]     == 0
        assert r2["metadata"]["input_tokens"]          == r1["metadata"]["input_tokens"]
        assert r2["metadata"]["output_tokens"]         == r1["metadata"]["output_tokens"]

    async def test_cache_hit_faster_than_live(self, client):
        payload = {
            "message":       "latency comparison test qwerty987",
            "system_prompt": "You are a helpful assistant.",
        }
        r1 = (await client.post("/chat", json={"session_id": "lat_a", **payload})).json()
        r2 = (await client.post("/chat", json={"session_id": "lat_b", **payload})).json()
        assert r2["metadata"]["latency_ms"] <= r1["metadata"]["latency_ms"], \
            "cache hit should be at least as fast as live call"

    async def test_different_message_no_cache_hit(self, client):
        """Two different messages → both LIVE_INFERENCE."""
        sys_p = "You are a helpful assistant."
        await client.post("/chat", json={
            "session_id": "diff_a", "message": "Message A zxcvbn", "system_prompt": sys_p,
        })
        r2 = (await client.post("/chat", json={
            "session_id": "diff_b", "message": "Message B 123456", "system_prompt": sys_p,
        })).json()
        assert r2["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ===========================================================================
# Per-request token counts
# ===========================================================================

class TestTokenCounts:

    async def test_two_fresh_sessions_same_tokens(self, client):
        """Equal first messages in two new sessions → same input_tokens (not accumulated)."""
        payload = {
            "message":       "token count test equal message aaabbb",
            "system_prompt": "You are a helpful assistant.",
        }
        r1 = (await client.post("/chat", json={"session_id": "tok_a", **payload})).json()
        r2 = (await client.post("/chat", json={"session_id": "tok_b", **payload})).json()
        # r2 is a cache hit – tokens come from stored response, not accumulated
        assert r2["metadata"]["input_tokens"] == r1["metadata"]["input_tokens"], \
            "input_tokens must not accumulate across sessions"

    async def test_followup_has_more_input_tokens(self, client):
        """Follow-up turn includes history → more input_tokens than the first turn."""
        sys_p = "You are a helpful assistant."
        r1 = (await client.post("/chat", json={
            "session_id": "tok_multi", "message": "First message here", "system_prompt": sys_p,
        })).json()
        t1 = r1["metadata"]["input_tokens"]

        # Second message in same session: history (user+assistant) is now included
        r2 = (await client.post("/chat", json={
            "session_id": "tok_multi", "message": "Second message follows up",
        })).json()
        t2 = r2["metadata"]["input_tokens"]

        assert t2 > t1, f"follow-up must have more input_tokens than first: {t2} vs {t1}"


# ===========================================================================
# Multi-turn & history
# ===========================================================================

class TestMultiTurn:

    async def test_history_stored_oldest_first(self, client):
        await client.post("/chat", json={
            "session_id": "mt_1", "message": "My name is Varun",
            "system_prompt": "You are helpful.",
        })
        await client.post("/chat", json={
            "session_id": "mt_1", "message": "What is my name?",
        })

        r = await client.get("/sessions/mt_1/messages")
        assert r.status_code == 200
        msgs = r.json()["messages"]
        assert len(msgs) == 4
        assert msgs[0]["role"]    == "user"
        assert msgs[0]["content"] == "My name is Varun"
        assert msgs[1]["role"]    == "assistant"
        assert msgs[2]["role"]    == "user"
        assert msgs[2]["content"] == "What is my name?"
        assert msgs[3]["role"]    == "assistant"

    async def test_no_system_prompt_in_history(self, client):
        await client.post("/chat", json={
            "session_id": "mt_sp", "message": "Hi",
            "system_prompt": "You are a pirate.",
        })
        msgs = (await client.get("/sessions/mt_sp/messages")).json()["messages"]
        for m in msgs:
            assert m["role"] in ("user", "assistant")

    async def test_sessions_isolated(self, client):
        for sess, msg in [("iso_a", "Session A unique"), ("iso_b", "Session B unique")]:
            await client.post("/chat", json={
                "session_id": sess, "message": msg, "system_prompt": "Be helpful.",
            })

        msgs_a = (await client.get("/sessions/iso_a/messages")).json()["messages"]
        msgs_b = (await client.get("/sessions/iso_b/messages")).json()["messages"]

        assert len(msgs_a) == 2
        assert len(msgs_b) == 2
        assert msgs_a[0]["content"] == "Session A unique"
        assert msgs_b[0]["content"] == "Session B unique"
        assert all(m["content"] != "Session B unique" for m in msgs_a)
        assert all(m["content"] != "Session A unique" for m in msgs_b)

    async def test_assistant_metadata_in_history(self, client):
        await client.post("/chat", json={
            "session_id": "mt_meta", "message": "Test meta", "system_prompt": "Be helpful.",
        })
        msgs = (await client.get("/sessions/mt_meta/messages")).json()["messages"]
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert assistant_msgs
        for am in assistant_msgs:
            assert am.get("metadata") is not None
            meta = am["metadata"]
            assert "resolution_source" in meta
            assert "input_tokens"      in meta
            assert "output_tokens"     in meta

    async def test_user_messages_have_no_metadata(self, client):
        await client.post("/chat", json={
            "session_id": "mt_umeta", "message": "User no meta", "system_prompt": "Be helpful.",
        })
        msgs = (await client.get("/sessions/mt_umeta/messages")).json()["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        for um in user_msgs:
            assert um.get("metadata") is None

    async def test_history_session_id_field(self, client):
        await client.post("/chat", json={
            "session_id": "sid_check", "message": "Check session_id", "system_prompt": "Be helpful.",
        })
        data = (await client.get("/sessions/sid_check/messages")).json()
        assert data["session_id"] == "sid_check"


# ===========================================================================
# Persistence across app restarts (same DB file, two lifespan cycles)
# ===========================================================================

class TestPersistence:

    async def test_history_survives_restart(self, test_env):
        from app.main        import app as fastapi_app
        from app.config      import get_settings
        from app.db          import init_db
        from app.llm_service import init_llm_service
        import app.llm_service as svc

        # ---- First "process" ----
        await init_db()
        init_llm_service()
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://test"
        ) as ac:
            r = await ac.post("/chat", json={
                "session_id": "persist_sess",
                "message":    "Remember this message",
                "system_prompt": "You are helpful.",
            })
            assert r.status_code == 200

        # ---- Simulate restart: clear all in-process caches ----
        get_settings.cache_clear()
        svc._disk_cache = None

        # ---- Second "process" – same DB file (env vars unchanged) ----
        await init_db()
        init_llm_service()
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://test"
        ) as ac:
            r = await ac.get("/sessions/persist_sess/messages")
            assert r.status_code == 200
            msgs = r.json()["messages"]
            assert len(msgs) == 2
            assert msgs[0]["role"]    == "user"
            assert msgs[0]["content"] == "Remember this message"


# ===========================================================================
# Validation (all → 422 with "detail")
# ===========================================================================

class TestValidation:

    async def _expect_422(self, client, payload=None, *, raw: bytes | None = None):
        if raw is not None:
            r = await client.post("/chat", content=raw,
                                  headers={"Content-Type": "application/json"})
        else:
            r = await client.post("/chat", json=payload)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
        body = r.json()
        assert "detail" in body, f"'detail' missing from 422 body: {body}"
        return body

    async def test_missing_session_id(self, client):
        await self._expect_422(client, {"message": "hello"})

    async def test_missing_message(self, client):
        await self._expect_422(client, {"session_id": "s1"})

    async def test_empty_string_message(self, client):
        await self._expect_422(client, {"session_id": "s1", "message": ""})

    async def test_whitespace_only_message_space(self, client):
        await self._expect_422(client, {"session_id": "s1", "message": "   "})

    async def test_whitespace_only_message_mixed(self, client):
        await self._expect_422(client, {"session_id": "s1", "message": " \t\n\r "})

    async def test_message_4001_chars(self, client):
        await self._expect_422(client, {"session_id": "s1", "message": "a" * 4001})

    async def test_message_4000_chars_ok(self, client):
        r = await client.post("/chat", json={
            "session_id": "s1", "message": "a" * 4000,
        })
        assert r.status_code == 200

    async def test_malformed_json(self, client):
        await self._expect_422(client, raw=b"{bad json}")

    async def test_session_id_too_long(self, client):
        await self._expect_422(client, {
            "session_id": "x" * 129,
            "message": "hi",
        })

    async def test_session_id_empty(self, client):
        await self._expect_422(client, {"session_id": "", "message": "hi"})


# ===========================================================================
# Failover
# ===========================================================================

class TestFailover:

    async def test_key1_429_uses_key2_200(self, client):
        r = await client.post("/chat",
            json={"session_id": "fo_429", "message": "429 failover",
                  "system_prompt": "Be helpful."},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r.status_code == 200
        meta = r.json()["metadata"]
        assert meta["key_id"]           == "key_2"
        assert meta["fallback_attempts"] == 1

    async def test_key1_500_uses_key2_200(self, client):
        r = await client.post("/chat",
            json={"session_id": "fo_500", "message": "500 failover",
                  "system_prompt": "Be helpful."},
            headers={"X-Simulate-Failure": "key_1=500"},
        )
        assert r.status_code == 200
        meta = r.json()["metadata"]
        assert meta["key_id"]           == "key_2"
        assert meta["fallback_attempts"] == 1

    async def test_key1_502_uses_key2_200(self, client):
        r = await client.post("/chat",
            json={"session_id": "fo_502", "message": "502 failover",
                  "system_prompt": "Be helpful."},
            headers={"X-Simulate-Failure": "key_1=502"},
        )
        assert r.status_code == 200
        assert r.json()["metadata"]["key_id"] == "key_2"

    async def test_reply_is_real_on_failover(self, client):
        """Simulated failure must still return a real (non-empty) reply, not an error string."""
        r = await client.post("/chat",
            json={"session_id": "fo_real", "message": "Real answer needed",
                  "system_prompt": "Be helpful."},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r.status_code == 200
        reply = r.json()["reply"]
        assert reply and len(reply) > 0

    async def test_both_keys_fail_503(self, client):
        r = await client.post("/chat",
            json={"session_id": "fo_both", "message": "Both fail"},
            headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
        )
        assert r.status_code == 503
        assert r.json() == {"detail": "all providers unavailable"}

    async def test_wildcard_fail_503(self, client):
        r = await client.post("/chat",
            json={"session_id": "fo_wild", "message": "Wildcard fail"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert r.status_code == 503
        assert r.json()["detail"] == "all providers unavailable"

    async def test_503_not_saved_to_history(self, client):
        r = await client.post("/chat",
            json={"session_id": "fo_nosave", "message": "Should not be saved"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert r.status_code == 503

        # Session was never created → 404
        hist = await client.get("/sessions/fo_nosave/messages")
        assert hist.status_code == 404

    async def test_503_messages_empty_even_with_session_prompt(self, client):
        """
        If system_prompt is provided and fails, session row exists but messages table is empty.
        GET should return 200 with empty messages list (session WAS created by set_session_system_prompt).
        """
        r = await client.post("/chat",
            json={"session_id": "fo_sp_fail", "message": "Fail with prompt",
                  "system_prompt": "You are helpful."},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert r.status_code == 503

        # Session row exists (created by set_session_system_prompt) but no messages saved
        hist = await client.get("/sessions/fo_sp_fail/messages")
        assert hist.status_code == 200
        assert hist.json()["messages"] == []


# ===========================================================================
# Failure simulation disabled
# ===========================================================================

class TestSimulationDisabled:

    async def test_header_ignored_when_simulation_false(self, no_sim_client):
        """When ENABLE_FAILURE_SIMULATION=false, X-Simulate-Failure must be ignored."""
        r = await no_sim_client.post("/chat",
            json={"session_id": "nodis", "message": "Should use key_1",
                  "system_prompt": "Be helpful."},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r.status_code == 200
        meta = r.json()["metadata"]
        # key_1 must have been used (header ignored)
        assert meta["key_id"]           == "key_1"
        assert meta["fallback_attempts"] == 0


# ===========================================================================
# Unknown session → 404
# ===========================================================================

class TestUnknownSession:

    async def test_unknown_session_404(self, client):
        r = await client.get("/sessions/no_such_session_xyz_abc/messages")
        assert r.status_code == 404

    async def test_404_detail_field(self, client):
        r = await client.get("/sessions/ghost_session/messages")
        body = r.json()
        assert body == {"detail": "session not found"}


# ===========================================================================
# System prompt changes cache key
# ===========================================================================

class TestSystemPromptCacheKey:

    async def test_different_system_prompts_different_cache_keys(self, client):
        msg = "Tell me a unique fact about the ocean."

        r1 = (await client.post("/chat", json={
            "session_id": "sp_pirate",   "message": msg,
            "system_prompt": "You are a pirate.",
        })).json()
        assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"

        # Different system prompt → different messages list → different cache key
        r2 = (await client.post("/chat", json={
            "session_id": "sp_scientist", "message": msg,
            "system_prompt": "You are a scientist.",
        })).json()
        assert r2["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    async def test_same_system_prompt_cache_hit(self, client):
        msg = "What is the boiling point of water?"
        sys_p = "You are a chemistry teacher."

        r1 = (await client.post("/chat", json={
            "session_id": "sp_chem_a", "message": msg, "system_prompt": sys_p,
        })).json()
        r2 = (await client.post("/chat", json={
            "session_id": "sp_chem_b", "message": msg, "system_prompt": sys_p,
        })).json()

        assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert r2["metadata"]["resolution_source"] == "LLM_CACHE"


# ===========================================================================
# API key values never appear in any response
# ===========================================================================

class TestNoApiKeyLeak:

    _KEYS = ("fake_key_1", "fake_key_2")

    def _assert_no_key(self, text: str, label: str) -> None:
        for k in self._KEYS:
            assert k not in text, f"API key {k!r} leaked in {label}: {text[:200]}"

    async def test_no_key_in_chat_response(self, client):
        r = await client.post("/chat", json={
            "session_id": "kchk_1", "message": "Hello", "system_prompt": "Be helpful.",
        })
        self._assert_no_key(r.text, "/chat response")

    async def test_no_key_in_history(self, client):
        await client.post("/chat", json={
            "session_id": "kchk_2", "message": "Hello", "system_prompt": "Be helpful.",
        })
        r = await client.get("/sessions/kchk_2/messages")
        self._assert_no_key(r.text, "/sessions history")

    async def test_no_key_in_health(self, client):
        r = await client.get("/health")
        self._assert_no_key(r.text, "/health")

    async def test_no_key_in_validation_error(self, client):
        r = await client.post("/chat", json={"session_id": "kchk_3"})
        self._assert_no_key(r.text, "422 error")

    async def test_no_key_in_503(self, client):
        r = await client.post("/chat",
            json={"session_id": "kchk_4", "message": "fail"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        self._assert_no_key(r.text, "503 error")
