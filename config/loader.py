from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.toml"


@dataclass
class ModelsConfig:
    primary: str = "claude-sonnet-4-20250514"
    fallbacks: list[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    triage_model: str = "claude-haiku-4-5-20251001"
    audit_model: str = "claude-sonnet-4-20250514"
    fix_model: str = "claude-sonnet-4-20250514"
    critical_fix_model: str = "claude-opus-4-20250514"
    cross_validate_critical: bool = True


@dataclass
class LoopConfig:
    max_cycles: int = 50
    cost_ceiling_usd: float = 50.0
    concurrency: int = 4
    stall_threshold: int = 2
    regression_threshold: float = 0.10
    max_fix_attempts: int = 3


@dataclass
class StaticAnalysisConfig:
    run_ruff: bool = True
    run_mypy: bool = True
    run_semgrep: bool = True
    run_bandit: bool = True
    fail_on_warning: bool = False
    timeout_s: int = 60


@dataclass
class GitHubConfig:
    base_branch: str = "main"
    branch_prefix: str = "stabilizer"
    auto_merge: bool = False


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass
class BrainConfig:
    backend: str = "sqlite"
    db_path: str = ".stabilizer/brain.db"
    vector_store_enabled: bool = False


@dataclass
class PatrolConfig:
    poll_interval_s: int = 60
    task_timeout_min: int = 15
    rejection_rate_threshold: float = 0.50
    cost_warn_pct: float = 0.80


@dataclass
class RHODAWK AI CODE STABILIZERConfig:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    static_analysis: StaticAnalysisConfig = field(default_factory=StaticAnalysisConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    api: APIConfig = field(default_factory=APIConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)


def load_config(config_path: Path | None = None) -> RHODAWK AI CODE STABILIZERConfig:
    path = config_path or _DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}

    if path.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                log.warning(
                    "tomllib/tomli not available — using default config. "
                    "Install tomli for Python <3.11: pip install tomli"
                )
                return _apply_env_overrides(RHODAWK AI CODE STABILIZERConfig())

        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
            log.info(f"Loaded config from {path}")
        except Exception as exc:
            log.warning(f"Failed to parse {path}: {exc}. Using defaults.")
    else:
        log.info(f"Config file not found at {path}. Using defaults.")

    cfg = RHODAWK AI CODE STABILIZERConfig()
    _apply_section(cfg.models, raw.get("models", {}))
    _apply_section(cfg.loop, raw.get("loop", {}))
    _apply_section(cfg.static_analysis, raw.get("static_analysis", {}))
    _apply_section(cfg.github, raw.get("github", {}))
    _apply_section(cfg.api, raw.get("api", {}))
    _apply_section(cfg.brain, raw.get("brain", {}))
    _apply_section(cfg.patrol, raw.get("patrol", {}))
    return _apply_env_overrides(cfg)


def _apply_section(obj: Any, data: dict) -> None:
    for key, val in data.items():
        if hasattr(obj, key):
            setattr(obj, key, val)


def _apply_env_overrides(cfg: RHODAWK AI CODE STABILIZERConfig) -> RHODAWK AI CODE STABILIZERConfig:
    _env_str(cfg.models, "primary", "RHODAWK_AI_CODE_STABILIZER_PRIMARY_MODEL")
    _env_str(cfg.models, "triage_model", "RHODAWK_AI_CODE_STABILIZER_TRIAGE_MODEL")
    _env_str(cfg.models, "critical_fix_model", "RHODAWK_AI_CODE_STABILIZER_CRITICAL_MODEL")
    _env_float(cfg.loop, "cost_ceiling_usd", "RHODAWK_AI_CODE_STABILIZER_COST_CEILING")
    _env_int(cfg.loop, "max_cycles", "RHODAWK_AI_CODE_STABILIZER_MAX_CYCLES")
    _env_int(cfg.loop, "concurrency", "RHODAWK_AI_CODE_STABILIZER_CONCURRENCY")
    _env_bool(cfg.static_analysis, "run_mypy", "RHODAWK_AI_CODE_STABILIZER_RUN_MYPY")
    _env_bool(cfg.static_analysis, "run_semgrep", "RHODAWK_AI_CODE_STABILIZER_RUN_SEMGREP")
    _env_str(cfg.github, "base_branch", "RHODAWK_AI_CODE_STABILIZER_BASE_BRANCH")
    _env_int(cfg.api, "port", "RHODAWK_AI_CODE_STABILIZER_API_PORT")
    return cfg


def _env_str(obj: Any, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        setattr(obj, attr, val)


def _env_int(obj: Any, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        try:
            setattr(obj, attr, int(val))
        except ValueError:
            log.warning(f"Invalid int for {env_key}: {val!r}")


def _env_float(obj: Any, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        try:
            setattr(obj, attr, float(val))
        except ValueError:
            log.warning(f"Invalid float for {env_key}: {val!r}")


def _env_bool(obj: Any, attr: str, env_key: str) -> None:
    val = os.getenv(env_key)
    if val:
        setattr(obj, attr, val.lower() not in ("false", "0", "no", "off"))
