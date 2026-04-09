# services/frontend/src/app/db.py
"""Minimal psycopg2 helper — builds DSN from POSTGRES_* env vars."""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extensions
import psycopg2.extras


def get_conn() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection. Caller is responsible for closing."""
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def query(sql: str, params: tuple[object, ...] = ()) -> list[dict[str, object]]:
    """Execute a SELECT and return rows as a list of dicts."""
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
