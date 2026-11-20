"""
db.py – SQLite access layer.
Uses aiosqlite for async I/O with WAL mode enabled.
The DB file path is derived from DATABASE_URL.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from .config import get_settings

# --------------------------------------------------------------------------- #
# Path helpers                                                                 #
# --------------------------------------------------------------------------- #

def _db_path() -> str:
    """Extract the filesystem path from a sqlite:///... URL."""
    url = get_settings().database_url
    match = re.match(r"sqlite:///(.*)", url, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported DATABASE_URL format: {url!r}")
    raw_path = match.group(1)
    return _ensure_parent(raw_path)


def _ensure_parent(path: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
            return path
        except (PermissionError, OSError):
            fallback_parent = "/tmp/data"
            os.makedirs(fallback_parent, exist_ok=True)
            return os.path.join(fallback_parent, os.path.basename(path))
    return path


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    metadata     TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);
"""


# --------------------------------------------------------------------------- #
# Lifecycle                                                                    #
# --------------------------------------------------------------------------- #

async def init_db() -> None:
    """Create parent directories and initialise the schema."""
    path = _db_path()
    _ensure_parent(path)
    async with aiosqlite.connect(path) as db:
        await db.executescript(_DDL)
        await db.commit()


# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #

async def ping_db() -> bool:
    """Return True if the DB is reachable."""
    try:
        path = _db_path()
        async with aiosqlite.connect(path) as db:
            await db.execute("SELECT 1")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Sessions                                                                     #
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def session_exists(session_id: str) -> bool:
    path = _db_path()
    async with aiosqlite.connect(path) as db:
        async with db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ) as cur:
            return await cur.fetchone() is not None


async def get_session_system_prompt(session_id: str) -> Optional[str]:
    path = _db_path()
    async with aiosqlite.connect(path) as db:
        async with db.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None


async def set_session_system_prompt(session_id: str, system_prompt: str) -> None:
    path = _db_path()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """
            INSERT INTO sessions (session_id, system_prompt, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET system_prompt = excluded.system_prompt
            """,
            (session_id, system_prompt, _now_iso()),
        )
        await db.commit()


async def _ensure_session(db: aiosqlite.Connection, session_id: str) -> None:
    """Create a session row if it doesn't exist yet (no system_prompt)."""
    await db.execute(
        """
        INSERT OR IGNORE INTO sessions (session_id, system_prompt, created_at)
        VALUES (?, NULL, ?)
        """,
        (session_id, _now_iso()),
    )


# --------------------------------------------------------------------------- #
# History                                                                      #
# --------------------------------------------------------------------------- #

async def get_history(session_id: str) -> List[Dict[str, Any]]:
    """Return all messages for a session, oldest first."""
    path = _db_path()
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, session_id, role, content, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Save exchange (user + assistant in one transaction)                          #
# --------------------------------------------------------------------------- #

async def save_exchange(
    session_id: str,
    user_content: str,
    assistant_content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist the user message and the assistant reply atomically.
    Only called after a successful LLM response – failed requests are never saved.
    """
    path = _db_path()
    now = _now_iso()
    meta_str = json.dumps(metadata) if metadata else None

    async with aiosqlite.connect(path) as db:
        await _ensure_session(db, session_id)

        await db.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", user_content, None, now),
        )
        await db.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", assistant_content, meta_str, now),
        )
        await db.commit()
