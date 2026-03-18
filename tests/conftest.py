"""
tests/conftest.py
=================
Shared pytest fixtures for the MACS test suite.

FIXES vs previous version
──────────────────────────
• ``event_loop_policy`` fixture used the old pytest-asyncio API which was
  deprecated in 0.21 and removed in 0.23.  Replaced with the correct
  ``asyncio_mode="auto"`` approach (already set in pyproject.toml) and a
  proper ``event_loop`` fixture override for session scope where needed.
• Added ``tmp_graph_engine`` fixture so unit tests can exercise the graph engine
  without hitting the DB.
• Added ``tmp_vector_brain`` fixture with an isolated temp directory so
  VectorBrain tests don't write to the repo root.
• Added ``sample_issues`` fixture used by consensus engine tests.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Event loop
# ──────────────────────────────────────────────────────────────────────────────
# asyncio_mode="auto" is set in pyproject.toml [tool.pytest.ini_options].
# The old `event_loop_policy` fixture approach is deprecated; just remove it.
# If you need session-scoped async fixtures, override event_loop at that scope.

@pytest.fixture(scope="session")
def event_loop():
    """
    Session-scoped event loop.
    Required for session-scoped async fixtures (e.g. shared DB).
    """
    policy = asyncio.DefaultEventLoopPolicy()
    loop   = policy.new_event_loop()
    yield loop
    loop.close()


# ──────────────────────────────────────────────────────────────────────────────
# Database fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_brain.db"


@pytest.fixture
async def storage(tmp_path: Path):
    """Isolated SQLiteBrainStorage for each test."""
    from brain.sqlite_storage import SQLiteBrainStorage
    db = SQLiteBrainStorage(tmp_path / "brain.db")
    await db.initialise()
    yield db
    await db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Graph engine fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_graph():
    """A freshly constructed (unbuilt) DependencyGraphEngine."""
    from brain.graph import DependencyGraphEngine
    return DependencyGraphEngine()


# ──────────────────────────────────────────────────────────────────────────────
# Vector brain fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_vector_brain(tmp_path: Path):
    """
    VectorBrain backed by a temp directory.
    Returns a stub (is_available=False) if chromadb is not installed.
    """
    from brain.vector_store import VectorBrain
    vb = VectorBrain(
        store_path=tmp_path / "vectors",
        embedding_model=None,   # use built-in hash embedder — no model download needed
    )
    vb.initialise()
    yield vb
    vb.close()


# ──────────────────────────────────────────────────────────────────────────────
# Sample data fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_issue():
    import uuid
    from brain.schemas import ExecutorType, Issue, Severity
    return Issue(
        run_id=str(uuid.uuid4()),
        severity=Severity.CRITICAL,
        file_path="agents/fixer.py",
        executor_type=ExecutorType.SECURITY,
        description="SQL injection in query builder",
        fingerprint="abc123def456",
        consensus_confidence=0.90,
    )


@pytest.fixture
def sample_issues():
    import uuid
    from brain.schemas import ExecutorType, Issue, Severity
    run_id = str(uuid.uuid4())
    return [
        Issue(
            run_id=run_id,
            severity=Severity.CRITICAL,
            file_path="agents/fixer.py",
            executor_type=ExecutorType.SECURITY,
            description="SQL injection",
            fingerprint="fp001",
            consensus_confidence=0.92,
        ),
        Issue(
            run_id=run_id,
            severity=Severity.CRITICAL,
            file_path="agents/fixer.py",
            executor_type=ExecutorType.ARCHITECTURE,
            description="SQL injection",  # same finding, different auditor
            fingerprint="fp001",
            consensus_confidence=0.88,
        ),
        Issue(
            run_id=run_id,
            severity=Severity.MAJOR,
            file_path="utils/chunking.py",
            executor_type=ExecutorType.STANDARDS,
            description="Missing type annotations",
            fingerprint="fp002",
            consensus_confidence=0.75,
        ),
        Issue(
            run_id=run_id,
            severity=Severity.MINOR,
            file_path="config/loader.py",
            executor_type=ExecutorType.STANDARDS,
            description="Magic number without constant",
            fingerprint="fp003",
            consensus_confidence=0.55,
        ),
    ]
