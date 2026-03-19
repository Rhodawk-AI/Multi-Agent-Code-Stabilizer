"""
config/loader.py — Configuration loader for Rhodawk AI Code Stabilizer.
Loads StabilizerConfig from YAML/TOML/env with validation.
"""
from __future__ import annotations
import logging, os
from pathlib import Path
from typing import Any
from orchestrator.controller import StabilizerConfig, DomainMode, SoftwareLevel

log = logging.getLogger(__name__)


def load_config(config_path: str | Path | None = None, **overrides: Any) -> StabilizerConfig:
    """Load config from file and/or environment variables."""
    data: dict[str, Any] = {}

    # Load from file
    if config_path:
        p = Path(config_path)
        if p.exists():
            if p.suffix in (".yaml", ".yml"):
                try:
                    import yaml  # type: ignore
                    data = yaml.safe_load(p.read_text()) or {}
                except ImportError:
                    log.warning("PyYAML not installed — skipping YAML config")
            elif p.suffix == ".toml":
                try:
                    import tomllib  # type: ignore
                    data = tomllib.loads(p.read_text()) or {}
                except ImportError:
                    try:
                        import tomli as tomllib  # type: ignore
                        data = tomllib.loads(p.read_text()) or {}
                    except ImportError:
                        log.warning("tomllib/tomli not installed — skipping TOML config")

    # Environment variable overrides
    env_map = {
        "RHODAWK_REPO_URL":       "repo_url",
        "RHODAWK_GITHUB_TOKEN":   "github_token",
        "RHODAWK_PRIMARY_MODEL":  "primary_model",
        "RHODAWK_REVIEWER_MODEL": "reviewer_model",
        "RHODAWK_DOMAIN_MODE":    "domain_mode",
        "RHODAWK_SOFTWARE_LEVEL": "software_level",
        "RHODAWK_MAX_CYCLES":     "max_cycles",
        "RHODAWK_COST_CEILING":   "cost_ceiling_usd",
        "RHODAWK_PG_DSN":         "postgres_dsn",
        "RHODAWK_API_BASE_URL":   "api_base_url",
        "RHODAWK_AUTO_COMMIT":    "auto_commit",
        "RHODAWK_USE_SQLITE":     "use_sqlite",
    }
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if cfg_key in ("max_cycles", "cost_ceiling_usd"):
                try:
                    data[cfg_key] = float(val) if "." in val else int(val)
                except ValueError:
                    pass
            elif cfg_key in ("auto_commit", "use_sqlite"):
                data[cfg_key] = val.lower() in ("1", "true", "yes")
            elif cfg_key == "domain_mode":
                try:
                    data[cfg_key] = DomainMode(val.upper())
                except ValueError:
                    pass
            elif cfg_key == "software_level":
                try:
                    data[cfg_key] = SoftwareLevel(val.upper())
                except ValueError:
                    pass
            else:
                data[cfg_key] = val

    data.update(overrides)

    # repo_root defaults to cwd if not set
    if "repo_root" not in data:
        data["repo_root"] = Path(os.environ.get("RHODAWK_REPO_ROOT", "."))

    try:
        return StabilizerConfig(**data)
    except Exception as exc:
        log.error(f"Config validation failed: {exc}")
        raise
