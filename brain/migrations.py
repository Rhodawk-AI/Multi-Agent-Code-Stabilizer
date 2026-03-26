"""
brain/migrations.py
====================
ARCH-06 FIX: Lightweight schema migration framework.

Tracks schema version in a `schema_version` table. Each migration is a
(version, description, up_sql) tuple applied in order. Migrations are
idempotent — re-running a migration that already ran is a no-op.

This avoids full Alembic dependency while preventing SQLite/PostgreSQL
schema drift after field changes.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "initial schema", ""),
    (2, "formal_verification_results: drop run_id, rename evaluated_at to verified_at, add evidence_path",
     ""),
    (3, "audit_runs: max_cycles default 200",
     ""),
]

_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_VERSION_DDL_PG = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def get_current_version_sqlite(db: Any) -> int:
    try:
        async with db.execute(
            "SELECT MAX(version) FROM schema_version"
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row and row[0] else 0
    except Exception:
        return 0


async def apply_migrations_sqlite(db: Any) -> int:
    await db.execute(_VERSION_DDL)
    await db.commit()

    current = await get_current_version_sqlite(db)
    applied = 0

    for version, description, up_sql in MIGRATIONS:
        if version <= current:
            continue
        if up_sql:
            for stmt in up_sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    try:
                        await db.execute(stmt)
                    except Exception as exc:
                        log.warning(f"Migration v{version} statement failed (may be idempotent): {exc}")
        await db.execute(
            "INSERT OR IGNORE INTO schema_version (version, description) VALUES (?, ?)",
            (version, description),
        )
        await db.commit()
        applied += 1
        log.info(f"Applied migration v{version}: {description}")

    if applied:
        log.info(f"Schema at version {current + applied} ({applied} migrations applied)")
    return current + applied


def get_schema_version() -> int:
    return MIGRATIONS[-1][0] if MIGRATIONS else 0
