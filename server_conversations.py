"""Simple SQLite-backed conversation store used by the FastAPI server.

Provides persistent storage for chat messages per `thread_id`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

DB_PATH = Path(__file__).resolve().parent / "data" / "server_conversations.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            thread_id TEXT,
            ts TEXT,
            role TEXT,
            content TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_thread ON messages(thread_id)")
    conn.commit()
    conn.close()


def save_message(thread_id: str, role: str, content: str, ts: str | None = None) -> None:
    ts = ts or datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO messages(thread_id, ts, role, content) VALUES (?, ?, ?, ?)", (thread_id, ts, role, content))
    conn.commit()
    conn.close()


def get_messages(thread_id: str, limit: int | None = None) -> List[Tuple[str, str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q = "SELECT ts, role, content FROM messages WHERE thread_id = ? ORDER BY ts ASC"
    if limit:
        q = q + " LIMIT ?"
        cur.execute(q, (thread_id, limit))
    else:
        cur.execute(q, (thread_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


# Initialize DB on import
init_db()
