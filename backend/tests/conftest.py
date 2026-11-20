"""
conftest.py – shared fixtures for all backend tests.

Every test that uses `client` gets:
  - A fresh SQLite DB (tmp_path/test.db)
  - A fresh diskcache dir (tmp_path/cache)
  - MOCK_LLM=true  (no real API calls)
  - ENABLE_FAILURE_SIMULATION=true
  - LLM_API_KEYS=fake_key_1,fake_key_2
  - get_settings lru_cache cleared before and after
  - llm_service._disk_cache reset before and after
  - DB schema created explicitly (not relying on lifespan order)
"""
from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Base env fixture (function-scoped – each test gets its own DB + cache dir)
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_env(monkeypatch, tmp_path):
    db_file   = tmp_path / "test.db"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    monkeypatch.setenv("DATABASE_URL",               f"sqlite:///{db_file}")
    monkeypatch.setenv("CACHE_DIR",                  str(cache_dir))
    monkeypatch.setenv("MOCK_LLM",                   "true")
    monkeypatch.setenv("LLM_API_KEYS",               "fake_key_1,fake_key_2")
    monkeypatch.setenv("ALLOWED_ORIGINS",            "http://localhost:5173")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION",  "true")
    monkeypatch.setenv("LLM_MODEL",                  "gemini/gemini-2.0-flash")

    _reset_singletons()
    yield {"db_file": db_file, "cache_dir": cache_dir, "tmp_path": tmp_path}
    _reset_singletons()


@pytest.fixture()
def no_sim_env(monkeypatch, tmp_path):
    """Same as test_env but with ENABLE_FAILURE_SIMULATION=false."""
    db_file   = tmp_path / "nosim.db"
    cache_dir = tmp_path / "nosim_cache"
    cache_dir.mkdir()

    monkeypatch.setenv("DATABASE_URL",               f"sqlite:///{db_file}")
    monkeypatch.setenv("CACHE_DIR",                  str(cache_dir))
    monkeypatch.setenv("MOCK_LLM",                   "true")
    monkeypatch.setenv("LLM_API_KEYS",               "fake_key_1,fake_key_2")
    monkeypatch.setenv("ALLOWED_ORIGINS",            "http://localhost:5173")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION",  "false")
    monkeypatch.setenv("LLM_MODEL",                  "gemini/gemini-2.0-flash")

    _reset_singletons()
    yield {"db_file": db_file, "cache_dir": cache_dir}
    _reset_singletons()


# ---------------------------------------------------------------------------
# Async HTTP clients
# ---------------------------------------------------------------------------

@pytest.fixture()
async def client(test_env):
    # Explicitly initialise DB + LLM service AFTER env vars are set,
    # so we never depend on lifespan ordering within the async event loop.
    from app.db          import init_db
    from app.llm_service import init_llm_service
    from app.main        import app as fastapi_app

    await init_db()
    init_llm_service()

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture()
async def no_sim_client(no_sim_env):
    from app.db          import init_db
    from app.llm_service import init_llm_service
    from app.main        import app as fastapi_app

    await init_db()
    init_llm_service()

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _reset_singletons() -> None:
    """Clear cached settings and disk-cache singleton between tests."""
    try:
        from app.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass

    try:
        import app.llm_service as svc
        svc._disk_cache = None
    except Exception:
        pass
