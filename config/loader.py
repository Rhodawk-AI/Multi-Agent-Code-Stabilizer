"""
config/loader.py — B10 fix: fail fast if required secrets are missing.
"""
from __future__ import annotations
import logging, os, sys
from pathlib import Path

log = logging.getLogger(__name__)

# Required secrets — process exits if ANY of these are absent
_REQUIRED_SECRETS = ["RHODAWK_JWT_SECRET", "RHODAWK_AUDIT_SECRET"]


def _validate_secrets() -> None:
    missing = [k for k in _REQUIRED_SECRETS if not os.environ.get(k)]
    if missing:
        msg = (
            "FATAL: Required secrets not set: " + ", ".join(missing) + "\n"
            "Generate them:\n"
            "  python -c \"import secrets; print('RHODAWK_JWT_SECRET=' + secrets.token_hex(32))\"\n"
            "  python -c \"import secrets; print('RHODAWK_AUDIT_SECRET=' + secrets.token_hex(32))\""
        )
        log.critical(msg)
        sys.exit(1)


try:
    import tomllib as _toml
except ImportError:
    import tomli as _toml  # type: ignore[no-reattr]


def load_config(path: str | Path = "config/default.toml") -> dict:
    """Load TOML config and override with environment variables."""
    config_path = Path(path)
    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = _toml.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Failed to load config from {config_path}: {e}")

    # Validate secrets before returning config
    _validate_secrets()
    return cfg
