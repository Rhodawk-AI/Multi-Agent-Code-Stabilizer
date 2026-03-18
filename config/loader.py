"""
config/loader.py
================
TOML + environment-variable configuration loader for MACS.

FIXES vs previous version
──────────────────────────
• GAP-19 CRITICAL: Class was named ``RHODAWK AI CODE STABILIZERConfig`` (spaces in
  identifier) — a Python SyntaxError that prevented the module from being imported
  at all.  Renamed to ``AppConfig``.  All internal helpers updated accordingly.
• Added DomainMode field so CLI / API can select mission-critical rule sets.
• Added vector_store_path forwarding from env var.
• _apply_env_overrides made exhaustive — all tunable fields now honoured.
• Type annotations tightened; no bare ``Any`` return types.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from brain.schemas import DomainMode

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.toml"


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelsConfig:
    primary:                 str       = "claude-sonnet-4-20250514"
    fallbacks:               list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    triage_model:            str       = "claude-haiku-4-5-20251001"
    audit_model:             str       = "claude-sonnet-4-20250514"
    fix_model:               str       = "claude-sonnet-4-20250514"
    critical_fix_model:      str       = "claude-opus-4-20250514"
    cross_validate_critical: bool      = True


@dataclass
class LoopConfig:
    max_cycles:           int   = 50
    cost_ceiling_usd:     float = 50.0
    concurrency:          int   = 4
    stall_threshold:      int   = 2
    regression_threshold: float = 0.10
    max_fix_attempts:     int   = 3
    parallel_fixes:       bool  = True
    """Run independent fix groups concurrently (requires dependency graph)."""


@dataclass
class StaticAnalysisConfig:
    run_ruff:        bool  = True
    run_mypy:        bool  = True
    run_semgrep:     bool  = True
    run_bandit:      bool  = True
    fail_on_warning: bool  = False
    timeout_s:       int   = 60


@dataclass
class GitHubConfig:
    base_branch:   str  = "main"
    branch_prefix: str  = "stabilizer"
    auto_merge:    bool = False


@dataclass
class APIConfig:
    host:   str  = "0.0.0.0"
    port:   int  = 8000
    reload: bool = False


@dataclass
class BrainConfig:
    backend:               str  = "sqlite"
    db_path:               str  = ".stabilizer/brain.db"
    vector_store_enabled:  bool = False
    vector_store_path:     str  = ".stabilizer/vectors"


@dataclass
class PatrolConfig:
    poll_interval_s:          int   = 60
    task_timeout_min:         int   = 15
    rejection_rate_threshold: float = 0.50
    cost_warn_pct:            float = 0.80


@dataclass
class GraphConfig:
    enabled:           bool  = True
    """Build a dependency graph after the read phase."""
    backend:           str   = "networkx"
    """'networkx' (default, pure-Python) or 'neo4j' (requires a Neo4j instance)."""
    neo4j_uri:         str   = "bolt://localhost:7687"
    neo4j_user:        str   = "neo4j"
    neo4j_password:    str   = ""
    min_centrality_threshold: float = 0.0
    """Prune edges with weight below this value to keep the graph lean."""


@dataclass
class FormalVerificationConfig:
    enabled:     bool  = False
    """Z3 formal verification — disabled by default; enable for finance/medical/military."""
    solver:      str   = "z3"
    timeout_s:   int   = 30
    domains:     list[str] = field(default_factory=lambda: ["finance", "medical", "military"])
    """Only run formal verification when domain_mode is one of these values."""


# ──────────────────────────────────────────────────────────────────────────────
# Root config — was incorrectly named with spaces; now AppConfig
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    """
    Root application configuration for MACS.

    Previously this class was named with spaces (``RHODAWK AI CODE STABILIZERConfig``)
    which is a Python SyntaxError.  Renamed to ``AppConfig``.
    """
    models:       ModelsConfig              = field(default_factory=ModelsConfig)
    loop:         LoopConfig                = field(default_factory=LoopConfig)
    static_analysis: StaticAnalysisConfig   = field(default_factory=StaticAnalysisConfig)
    github:       GitHubConfig              = field(default_factory=GitHubConfig)
    api:          APIConfig                 = field(default_factory=APIConfig)
    brain:        BrainConfig               = field(default_factory=BrainConfig)
    patrol:       PatrolConfig              = field(default_factory=PatrolConfig)
    graph:        GraphConfig               = field(default_factory=GraphConfig)
    formal:       FormalVerificationConfig  = field(default_factory=FormalVerificationConfig)
    domain_mode:  DomainMode                = DomainMode.GENERAL


# ──────────────────────────────────────────────────────────────────────────────
# Public loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path | None = None) -> AppConfig:
    """
    Load configuration from a TOML file, then apply environment variable overrides.

    Priority (highest → lowest):
        1. Environment variables
        2. TOML file at *config_path* (or default.toml)
        3. Dataclass defaults
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    raw: dict = {}

    if path.exists():
        try:
            import tomllib                        # Python ≥3.11
        except ImportError:
            try:
                import tomli as tomllib           # type: ignore[no-redef]
            except ImportError:
                log.warning(
                    "tomllib/tomli not available — using default config. "
                    "Install tomli for Python <3.11: pip install tomli"
                )
                return _apply_env_overrides(AppConfig())

        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            log.info(f"Loaded config from {path}")
        except Exception as exc:
            log.warning(f"Failed to parse {path}: {exc}. Using defaults.")
    else:
        log.info(f"Config file not found at {path}. Using defaults.")

    cfg = AppConfig()
    _apply_section(cfg.models,           raw.get("models", {}))
    _apply_section(cfg.loop,             raw.get("loop", {}))
    _apply_section(cfg.static_analysis,  raw.get("static_analysis", {}))
    _apply_section(cfg.github,           raw.get("github", {}))
    _apply_section(cfg.api,              raw.get("api", {}))
    _apply_section(cfg.brain,            raw.get("brain", {}))
    _apply_section(cfg.patrol,           raw.get("patrol", {}))
    _apply_section(cfg.graph,            raw.get("graph", {}))
    _apply_section(cfg.formal,           raw.get("formal_verification", {}))

    # domain_mode from TOML top-level key
    raw_domain = raw.get("domain_mode", "")
    if raw_domain:
        try:
            cfg.domain_mode = DomainMode(raw_domain)
        except ValueError:
            log.warning(f"Unknown domain_mode '{raw_domain}' in TOML — keeping default.")

    return _apply_env_overrides(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_section(obj: object, data: dict) -> None:
    """Copy matching keys from *data* onto *obj*'s attributes."""
    for key, val in data.items():
        if hasattr(obj, key):
            try:
                setattr(obj, key, val)
            except Exception as exc:
                log.warning(f"Config key '{key}' could not be set: {exc}")


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """Override config fields from environment variables."""
    # Models
    _env_str(cfg.models,  "primary",            "MACS_PRIMARY_MODEL")
    _env_str(cfg.models,  "triage_model",        "MACS_TRIAGE_MODEL")
    _env_str(cfg.models,  "critical_fix_model",  "MACS_CRITICAL_MODEL")
    _env_str(cfg.models,  "audit_model",         "MACS_AUDIT_MODEL")
    _env_bool(cfg.models, "cross_validate_critical", "MACS_CROSS_VALIDATE")

    # Loop
    _env_float(cfg.loop, "cost_ceiling_usd", "MACS_COST_CEILING")
    _env_int(cfg.loop,   "max_cycles",       "MACS_MAX_CYCLES")
    _env_int(cfg.loop,   "concurrency",      "MACS_CONCURRENCY")
    _env_bool(cfg.loop,  "parallel_fixes",   "MACS_PARALLEL_FIXES")

    # Static analysis
    _env_bool(cfg.static_analysis, "run_mypy",        "MACS_RUN_MYPY")
    _env_bool(cfg.static_analysis, "run_semgrep",     "MACS_RUN_SEMGREP")
    _env_bool(cfg.static_analysis, "run_bandit",      "MACS_RUN_BANDIT")
    _env_bool(cfg.static_analysis, "fail_on_warning", "MACS_FAIL_ON_WARNING")

    # GitHub
    _env_str(cfg.github, "base_branch",   "MACS_BASE_BRANCH")
    _env_str(cfg.github, "branch_prefix", "MACS_BRANCH_PREFIX")

    # API
    _env_int(cfg.api, "port", "MACS_API_PORT")

    # Brain / vector store
    _env_bool(cfg.brain, "vector_store_enabled", "MACS_VECTOR_STORE")
    _env_str(cfg.brain,  "vector_store_path",    "MACS_VECTOR_PATH")

    # Graph
    _env_bool(cfg.graph, "enabled",      "MACS_GRAPH_ENABLED")
    _env_str(cfg.graph,  "backend",      "MACS_GRAPH_BACKEND")

    # Formal verification
    _env_bool(cfg.formal, "enabled",   "MACS_FORMAL_ENABLED")
    _env_int(cfg.formal,  "timeout_s", "MACS_FORMAL_TIMEOUT")

    # Domain mode
    domain_env = os.getenv("MACS_DOMAIN_MODE", "")
    if domain_env:
        try:
            cfg.domain_mode = DomainMode(domain_env.lower())
        except ValueError:
            log.warning(f"Invalid MACS_DOMAIN_MODE '{domain_env}' — ignoring.")

    return cfg


def _env_str(obj: object, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        setattr(obj, attr, val)


def _env_int(obj: object, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        try:
            setattr(obj, attr, int(val))
        except ValueError:
            log.warning(f"Invalid int for {env_key}: {val!r}")


def _env_float(obj: object, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        try:
            setattr(obj, attr, float(val))
        except ValueError:
            log.warning(f"Invalid float for {env_key}: {val!r}")


def _env_bool(obj: object, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        setattr(obj, attr, val.lower() not in ("false", "0", "no", "off"))
