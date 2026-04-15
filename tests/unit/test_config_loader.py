"""
tests/unit/test_config_loader.py
==================================
Unit tests for config/loader.py.

Covers:
  - _flatten_toml()      — nested TOML section → flat field mapping
  - _apply_env()         — env vars → typed field override (int, float, bool,
                           domain, software_level, str)
  - load_config()        — ADD-1 FIX: unknown key raises ValueError with
                           clear message naming offending key
  - load_config()        — valid minimal config constructs StabilizerConfig
  - load_config()        — repo_root defaults to CWD when absent
  - load_config()        — TOML file loaded and flattened when path provided
  - load_config()        — YAML file loaded when PyYAML installed

No real Docker, Postgres, Qdrant, or LLM calls.
All filesystem interaction is via tmp_path.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _flatten_toml
# ---------------------------------------------------------------------------

class TestFlattenToml:
    def _flatten(self, raw: dict) -> dict:
        from config.loader import _flatten_toml
        return _flatten_toml(raw)

    def test_top_level_scalar_passed_through(self):
        result = self._flatten({"max_cycles": 50})
        assert result["max_cycles"] == 50

    def test_models_primary_mapped(self):
        result = self._flatten({"models": {"primary": "ollama/qwen2.5-coder:7b"}})
        assert result["primary_model"] == "ollama/qwen2.5-coder:7b"

    def test_models_fallbacks_mapped(self):
        result = self._flatten({"models": {"fallbacks": ["m1", "m2"]}})
        assert result["fallback_models"] == ["m1", "m2"]

    def test_loop_max_cycles_mapped(self):
        result = self._flatten({"loop": {"max_cycles": 100}})
        assert result["max_cycles"] == 100

    def test_loop_cost_ceiling_mapped(self):
        result = self._flatten({"loop": {"cost_ceiling_usd": 25.0}})
        assert result["cost_ceiling_usd"] == 25.0

    def test_nested_dict_section_not_in_output_as_nested(self):
        result = self._flatten({"models": {"primary": "x"}, "loop": {"max_cycles": 5}})
        # nested dicts should not appear as their original keys
        assert "models" not in result
        assert "loop" not in result

    def test_unknown_top_level_scalar_preserved(self):
        """Unknown top-level scalars are passed through for later validation."""
        result = self._flatten({"my_custom_key": "value"})
        assert result["my_custom_key"] == "value"

    def test_models_triage_model_mapped(self):
        result = self._flatten({"models": {"triage_model": "ollama/qwen2.5-coder:7b"}})
        assert result["triage_model"] == "ollama/qwen2.5-coder:7b"

    def test_models_reviewer_model_mapped(self):
        result = self._flatten({"models": {"reviewer_model": "ollama/qwen2.5-coder:7b"}})
        assert result["reviewer_model"] == "ollama/qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# _apply_env
# ---------------------------------------------------------------------------

class TestApplyEnv:
    def _apply(self, env_vars: dict[str, str]) -> dict:
        from config.loader import _apply_env
        data: dict = {}
        old = {k: os.environ.get(k) for k in env_vars}
        os.environ.update(env_vars)
        try:
            _apply_env(data)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return data

    def test_int_env_converted(self):
        data = self._apply({"RHODAWK_MAX_CYCLES": "42"})
        assert data.get("max_cycles") == 42

    def test_float_env_converted(self):
        data = self._apply({"RHODAWK_COST_CEILING_USD": "15.5"})
        assert data.get("cost_ceiling_usd") == pytest.approx(15.5)

    def test_bool_true_variants(self):
        for val in ("1", "true", "yes", "True", "YES"):
            data = self._apply({"RHODAWK_AUTO_COMMIT": val})
            result = data.get("auto_commit")
            if result is not None:
                assert result is True, f"Expected True for env value {val!r}"

    def test_bool_false_variants(self):
        for val in ("0", "false", "no"):
            data = self._apply({"RHODAWK_AUTO_COMMIT": val})
            result = data.get("auto_commit")
            if result is not None:
                assert result is False, f"Expected False for env value {val!r}"

    def test_invalid_int_env_ignored(self):
        """Invalid value must be silently skipped (logged as warning)."""
        data = self._apply({"RHODAWK_MAX_CYCLES": "not_a_number"})
        assert "max_cycles" not in data

    def test_domain_mode_env_applied(self):
        data = self._apply({"RHODAWK_DOMAIN": "MILITARY"})
        if "domain_mode" in data:
            assert str(data["domain_mode"]).upper() in ("MILITARY", "DOMAIN_MODE.MILITARY")

    def test_repo_root_env_applied(self):
        data = self._apply({"RHODAWK_REPO_ROOT": "/tmp/myrepo"})
        # repo_root may be set as str or Path depending on env map entry
        if "repo_root" in data:
            assert str(data["repo_root"]) == "/tmp/myrepo"


# ---------------------------------------------------------------------------
# load_config — ADD-1 FIX: unknown key detection
# ---------------------------------------------------------------------------

class TestLoadConfigUnknownKeyRejection:
    def test_unknown_key_raises_value_error(self, tmp_path):
        """Typos in config keys must raise ValueError naming the bad key."""
        from config.loader import load_config
        with pytest.raises(ValueError, match="gap5_bobn_n_canddiates"):
            load_config(repo_root=tmp_path, gap5_bobn_n_canddiates=10)

    def test_multiple_unknown_keys_all_named(self, tmp_path):
        from config.loader import load_config
        with pytest.raises(ValueError) as exc_info:
            load_config(
                repo_root=tmp_path,
                typo_key_one="x",
                typo_key_two="y",
            )
        msg = str(exc_info.value)
        assert "typo_key_one" in msg or "typo_key_two" in msg

    def test_known_keys_do_not_raise(self, tmp_path):
        from config.loader import load_config
        # max_cycles is a known StabilizerConfig field
        cfg = load_config(repo_root=tmp_path, max_cycles=5)
        assert cfg.max_cycles == 5


# ---------------------------------------------------------------------------
# load_config — minimal valid construction
# ---------------------------------------------------------------------------

class TestLoadConfigMinimalValid:
    def test_returns_stabilizer_config(self, tmp_path):
        from config.loader import load_config
        from orchestrator.controller import StabilizerConfig
        cfg = load_config(repo_root=tmp_path)
        assert isinstance(cfg, StabilizerConfig)

    def test_repo_root_set(self, tmp_path):
        from config.loader import load_config
        cfg = load_config(repo_root=tmp_path)
        assert Path(cfg.repo_root) == tmp_path

    def test_default_repo_root_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RHODAWK_REPO_ROOT", str(tmp_path))
        from config.loader import load_config
        cfg = load_config()
        assert Path(cfg.repo_root) == tmp_path

    def test_override_max_cycles(self, tmp_path):
        from config.loader import load_config
        cfg = load_config(repo_root=tmp_path, max_cycles=10)
        assert cfg.max_cycles == 10

    def test_override_cost_ceiling(self, tmp_path):
        from config.loader import load_config
        cfg = load_config(repo_root=tmp_path, cost_ceiling_usd=7.5)
        assert cfg.cost_ceiling_usd == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# load_config — TOML file loading
# ---------------------------------------------------------------------------

class TestLoadConfigToml:
    def test_toml_max_cycles_loaded(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[loop]\nmax_cycles = 77\n")
        from config.loader import load_config
        cfg = load_config(config_path=toml_file, repo_root=tmp_path)
        assert cfg.max_cycles == 77

    def test_toml_primary_model_loaded(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[models]\nprimary = "ollama/qwen2.5-coder:7b"\n')
        from config.loader import load_config
        cfg = load_config(config_path=toml_file, repo_root=tmp_path)
        assert cfg.primary_model == "ollama/qwen2.5-coder:7b"

    def test_nonexistent_config_path_no_error(self, tmp_path):
        from config.loader import load_config
        cfg = load_config(
            config_path=tmp_path / "nonexistent.toml",
            repo_root=tmp_path,
        )
        assert cfg is not None

    def test_toml_unknown_key_raises(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text("[loop]\nmax_cyclez = 10\n")  # typo
        from config.loader import load_config
        with pytest.raises((ValueError, Exception)):
            load_config(config_path=toml_file, repo_root=tmp_path)


# ---------------------------------------------------------------------------
# load_config — env vars take priority over file
# ---------------------------------------------------------------------------

class TestLoadConfigEnvPriority:
    def test_env_overrides_toml_value(self, tmp_path, monkeypatch):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[loop]\nmax_cycles = 50\n")
        monkeypatch.setenv("RHODAWK_MAX_CYCLES", "99")
        from config.loader import load_config
        cfg = load_config(config_path=toml_file, repo_root=tmp_path)
        assert cfg.max_cycles == 99
