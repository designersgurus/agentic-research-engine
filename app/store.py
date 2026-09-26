"""Tiny SQLite document store (swap for Postgres/Redis in production)."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional


class Store:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS docs (collection TEXT, id TEXT, data TEXT, updated REAL, "
                "PRIMARY KEY (collection, id))"
            )
            self._conn.commit()

    def put(self, collection: str, doc_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO docs VALUES (?, ?, ?, ?)",
                (collection, doc_id, json.dumps(data, default=str), time.time()),
            )
            self._conn.commit()

    def get(self, collection: str, doc_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM docs WHERE collection=? AND id=?", (collection, doc_id)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, collection: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM docs WHERE collection=? ORDER BY updated DESC LIMIT ?", (collection, limit)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def count_where_status(self, collection: str, statuses: tuple[str, ...]) -> int:
        marks = ",".join("?" * len(statuses))
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM docs WHERE collection=? AND json_extract(data, '$.status') IN ({marks})",
                (collection, *statuses),
            ).fetchone()
        return int(row[0])

    def purge_older_than(self, hours: float) -> int:
        cutoff = time.time() - hours * 3600
        with self._lock:
            cur = self._conn.execute("DELETE FROM docs WHERE updated < ?", (cutoff,))
            self._conn.commit()
        return cur.rowcount
