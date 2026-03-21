from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from orchestrator.controller import StabilizerConfig, DomainMode, SoftwareLevel

log = logging.getLogger(__name__)


def load_config(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> StabilizerConfig:
    """Load StabilizerConfig from TOML/YAML file then apply environment overrides."""
    data: dict[str, Any] = {}

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
                _tomllib = None
                try:
                    import tomllib as _tomllib  # type: ignore
                except ImportError:
                    try:
                        import tomli as _tomllib  # type: ignore
                    except ImportError:
                        log.warning("tomllib/tomli not installed — skipping TOML config")
                if _tomllib is not None:
                    raw = _tomllib.loads(p.read_text()) or {}
                    data = _flatten_toml(raw)

    _apply_env(data)
    data.update(overrides)

    if "repo_root" not in data:
        data["repo_root"] = Path(os.environ.get("RHODAWK_REPO_ROOT", "."))

    try:
        return StabilizerConfig(**data)
    except Exception as exc:
        log.error(f"Config validation failed: {exc}")
        raise


def _flatten_toml(raw: dict) -> dict:
    """Map nested TOML sections to flat StabilizerConfig field names."""
    out: dict[str, Any] = {}

    for k, v in raw.items():
        if not isinstance(v, dict):
            out[k] = v

    models = raw.get("models", {})
    if models.get("primary"):
        out["primary_model"] = models["primary"]
    for key in ("triage_model", "critical_fix_model", "reviewer_model"):
        if models.get(key):
            out[key] = models[key]
    if models.get("fallbacks"):
        out["fallback_models"] = models["fallbacks"]

    _map(out, raw.get("loop", {}), {
        "max_cycles":        "max_cycles",
        "cost_ceiling_usd":  "cost_ceiling_usd",
        "concurrency":       "concurrency",
        "chunk_concurrency": "chunk_concurrency",
    })

    brain = raw.get("brain", {})
    backend = brain.get("backend", "")
    if backend == "sqlite":
        out["use_sqlite"] = True
    elif backend == "postgres":
        out["use_sqlite"] = False
    _map(out, brain, {
        "vector_store_enabled": "vector_store_enabled",
        "vector_store_path":    "vector_store_path",
    })

    _map(out, raw.get("static_analysis", {}), {
        "run_ruff":    "run_ruff",
        "run_mypy":    "run_mypy",
        "run_semgrep": "run_semgrep",
        "run_bandit":  "run_bandit",
    })

    graph = raw.get("graph", {})
    if graph.get("enabled") is not None:
        out["graph_enabled"] = graph["enabled"]

    formal = raw.get("formal_verification", {})
    if formal.get("enabled") is not None:
        out["formal_verification"] = formal["enabled"]

    # Gap 1: [cpg] section
    _map(out, raw.get("cpg", {}), {
        "enabled":                "cpg_enabled",
        "joern_url":              "joern_url",
        "project_name":           "joern_project_name",
        "repo_path":              "joern_repo_path",
        "max_slice_nodes":        "cpg_max_slice_nodes",
        "max_files_in_slice":     "cpg_max_files_in_slice",
        "blast_radius_threshold": "cpg_blast_radius_threshold",
    })

    # Gap 2: [synthesis] section
    _map(out, raw.get("synthesis", {}), {
        "enabled":               "synthesis_enabled",
        "dedup_enabled":         "synthesis_dedup_enabled",
        "compound_enabled":      "synthesis_compound_enabled",
        "synthesis_model":       "synthesis_model",
        "max_compound_findings": "synthesis_max_compound",
    })

    # Gap 5: [gap5] section — adversarial BoBN ensemble
    _map(out, raw.get("gap5", {}), {
        "enabled":                  "gap5_enabled",
        "vllm_secondary_base_url":  "gap5_vllm_secondary_base_url",
        "vllm_secondary_model":     "gap5_vllm_secondary_model",
        "vllm_critic_base_url":     "gap5_vllm_critic_base_url",
        "vllm_critic_model":        "gap5_vllm_critic_model",
        # Synthesis model (Mistral/Devstral) — fourth independent family.
        # Must be set to a different family from both fixers AND the critic.
        # Leave vllm_synthesis_base_url blank to route through OpenRouter.
        "vllm_synthesis_base_url":  "gap5_vllm_synthesis_base_url",
        "vllm_synthesis_model":     "gap5_vllm_synthesis_model",
        "bobn_n_candidates":        "gap5_bobn_n_candidates",
        "bobn_fixer_a_count":       "gap5_bobn_fixer_a_count",
        "bobn_fixer_b_count":       "gap5_bobn_fixer_b_count",
    })

    # Gap 6: [gap6] section — federated anonymized pattern store
    _map(out, raw.get("gap6", {}), {
        "federation_enabled":  "gap6_federation_enabled",
        "contribute_patterns": "gap6_contribute_patterns",
        "receive_patterns":    "gap6_receive_patterns",
        "registry_url":        "gap6_registry_url",
        "extra_peer_urls":     "gap6_extra_peer_urls",
        "instance_id":         "gap6_instance_id",
        "min_complexity":      "gap6_min_complexity",
    })

    _map(out, raw.get("github", {}), {
        "base_branch":   "base_branch",
        "branch_prefix": "branch_prefix",
    })
    _map(out, raw.get("auditing", {}), {
        "validate_findings": "validate_findings",
    })

    return out


def _map(out: dict, section: dict, mapping: dict[str, str]) -> None:
    for src, dst in mapping.items():
        val = section.get(src)
        if val is not None:
            out[dst] = val


_ENV_MAP: dict[str, tuple[str, str]] = {
    "RHODAWK_REPO_URL":           ("repo_url",                    "str"),
    "RHODAWK_GITHUB_TOKEN":       ("github_token",                "str"),
    "RHODAWK_PRIMARY_MODEL":      ("primary_model",               "str"),
    "RHODAWK_REVIEWER_MODEL":     ("reviewer_model",              "str"),
    "RHODAWK_TRIAGE_MODEL":       ("triage_model",                "str"),
    "RHODAWK_DOMAIN_MODE":        ("domain_mode",                 "domain"),
    "RHODAWK_SOFTWARE_LEVEL":     ("software_level",              "software_level"),
    "RHODAWK_MAX_CYCLES":         ("max_cycles",                  "int"),
    "RHODAWK_COST_CEILING":       ("cost_ceiling_usd",            "float"),
    "RHODAWK_PG_DSN":             ("postgres_dsn",                "str"),
    "RHODAWK_API_BASE_URL":       ("api_base_url",                "str"),
    "RHODAWK_AUTO_COMMIT":        ("auto_commit",                 "bool"),
    "RHODAWK_USE_SQLITE":         ("use_sqlite",                  "bool"),
    "RHODAWK_QDRANT_URL":         ("qdrant_url",                  "str"),
    "RHODAWK_VLLM_BASE_URL":      ("vllm_base_url",               "str"),
    # Gap 1
    "JOERN_URL":                  ("joern_url",                   "str"),
    "JOERN_REPO_PATH":            ("joern_repo_path",             "str"),
    "JOERN_PROJECT_NAME":         ("joern_project_name",          "str"),
    "CPG_ENABLED":                ("cpg_enabled",                 "bool"),
    "CPG_BLAST_RADIUS_THRESHOLD": ("cpg_blast_radius_threshold",  "int"),
    "CPG_MAX_SLICE_NODES":        ("cpg_max_slice_nodes",         "int"),
    "CPG_MAX_FILES_IN_SLICE":     ("cpg_max_files_in_slice",      "int"),
    # Gap 2
    "RHODAWK_SYNTHESIS_MODEL":    ("synthesis_model",             "str"),
    "RHODAWK_SYNTHESIS_ENABLED":  ("synthesis_enabled",           "bool"),
    "RHODAWK_COMPOUND_ENABLED":   ("synthesis_compound_enabled",  "bool"),
    "RHODAWK_SYNTHESIS_MAX":      ("synthesis_max_compound",      "int"),
    # Gap 5: adversarial BoBN ensemble
    "RHODAWK_GAP5_ENABLED":       ("gap5_enabled",                "bool"),
    "VLLM_SECONDARY_BASE_URL":    ("gap5_vllm_secondary_base_url","str"),
    "VLLM_SECONDARY_MODEL":       ("gap5_vllm_secondary_model",   "str"),
    "VLLM_CRITIC_BASE_URL":       ("gap5_vllm_critic_base_url",   "str"),
    "VLLM_CRITIC_MODEL":          ("gap5_vllm_critic_model",      "str"),
    # Synthesis vLLM endpoint — Devstral/Mistral family (fourth independent family).
    # Set VLLM_SYNTHESIS_BASE_URL to route to a local vLLM server instead of OpenRouter.
    "VLLM_SYNTHESIS_BASE_URL":    ("gap5_vllm_synthesis_base_url","str"),
    "VLLM_SYNTHESIS_MODEL":       ("gap5_vllm_synthesis_model",   "str"),
    "RHODAWK_BOBN_CANDIDATES":    ("gap5_bobn_n_candidates",      "int"),
    "RHODAWK_BOBN_FIXER_A":       ("gap5_bobn_fixer_a_count",     "int"),
    "RHODAWK_BOBN_FIXER_B":       ("gap5_bobn_fixer_b_count",     "int"),
    # Gap 6: federated anonymized pattern store
    "RHODAWK_GAP6_ENABLED":       ("gap6_federation_enabled",     "bool"),
    "RHODAWK_FED_CONTRIBUTE":     ("gap6_contribute_patterns",    "bool"),
    "RHODAWK_FED_RECEIVE":        ("gap6_receive_patterns",       "bool"),
    "RHODAWK_FED_REGISTRY_URL":   ("gap6_registry_url",           "str"),
    "RHODAWK_FED_PEERS":          ("gap6_extra_peer_urls",        "str"),
    "RHODAWK_FED_INSTANCE_ID":    ("gap6_instance_id",            "str"),
    "RHODAWK_FED_MIN_COMPLEXITY": ("gap6_min_complexity",         "float"),
}


def _apply_env(data: dict) -> None:
    for env_key, (field, kind) in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        try:
            if kind == "int":
                data[field] = int(val)
            elif kind == "float":
                data[field] = float(val) if "." in val else int(val)
            elif kind == "bool":
                data[field] = val.strip().lower() in ("1", "true", "yes")
            elif kind == "domain":
                data[field] = DomainMode(val.upper())
            elif kind == "software_level":
                data[field] = SoftwareLevel(val.upper())
            else:
                data[field] = val
        except (ValueError, KeyError) as exc:
            log.warning(f"Ignoring invalid env {env_key}={val!r}: {exc}")
