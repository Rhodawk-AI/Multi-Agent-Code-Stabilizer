"""
tests/unit/test_schema_sync.py
==============================
ARCH-06 FIX: Verify SQLite and PostgreSQL DDL table sets stay in sync.
"""
import re
import pytest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SQLITE_PATH = _ROOT / "brain" / "sqlite_storage.py"
_PG_PATH     = _ROOT / "brain" / "postgres_storage.py"

def _extract_tables(source: str) -> set[str]:
    return set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", source))


def test_sqlite_and_pg_share_same_tables():
    sqlite_src = _SQLITE_PATH.read_text()
    pg_src     = _PG_PATH.read_text()

    sqlite_tables = _extract_tables(sqlite_src)
    pg_tables     = _extract_tables(pg_src)

    pg_tables.discard("file_chunks_")

    sqlite_only = sqlite_tables - pg_tables
    pg_only     = pg_tables - sqlite_tables

    assert not sqlite_only, f"Tables in SQLite but missing from PG: {sqlite_only}"
    assert not pg_only, f"Tables in PG but missing from SQLite: {pg_only}"


def test_max_cycles_defaults_match():
    sqlite_src = _SQLITE_PATH.read_text()
    pg_src     = _PG_PATH.read_text()

    sqlite_default = re.search(r"max_cycles\s+INTEGER\s+DEFAULT\s+(\d+)", sqlite_src)
    pg_default     = re.search(r"max_cycles\s+INTEGER\s+DEFAULT\s+(\d+)", pg_src)

    assert sqlite_default and pg_default, "max_cycles DEFAULT not found in DDL"
    assert sqlite_default.group(1) == pg_default.group(1), (
        f"max_cycles default mismatch: SQLite={sqlite_default.group(1)} "
        f"PG={pg_default.group(1)}"
    )
