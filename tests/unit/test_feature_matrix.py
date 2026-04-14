"""tests/unit/test_feature_matrix.py — FeatureMatrix unit tests."""
from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest

from startup.feature_matrix import (
    Capability,
    CapabilityStatus,
    ConfigurationError,
    FeatureMatrix,
    is_available,
    require_capability,
    verify_startup,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset FeatureMatrix singleton between tests."""
    FeatureMatrix.reset()
    yield
    FeatureMatrix.reset()


# ── Capability dataclass ──────────────────────────────────────────────────────

class TestCapability:
    def test_default_status_unavailable(self):
        cap = Capability(
            name="test_cap",
            description="Test capability",
            check_type="import",
            check_target="nonexistent_pkg",
        )
        assert cap.status == CapabilityStatus.UNAVAILABLE

    def test_required_for_empty_by_default(self):
        cap = Capability(name="x", description="y", check_type="import", check_target="z")
        assert cap.required_for == set()


# ── CapabilityStatus enum ─────────────────────────────────────────────────────

class TestCapabilityStatus:
    def test_values(self):
        assert CapabilityStatus.AVAILABLE.value == "AVAILABLE"
        assert CapabilityStatus.UNAVAILABLE.value == "UNAVAILABLE"
        assert CapabilityStatus.DEGRADED.value == "DEGRADED"


# ── FeatureMatrix singleton ───────────────────────────────────────────────────

class TestFeatureMatrixSingleton:
    def test_get_returns_instance(self):
        fm = FeatureMatrix.get()
        assert isinstance(fm, FeatureMatrix)

    def test_get_returns_same_instance(self):
        fm1 = FeatureMatrix.get()
        fm2 = FeatureMatrix.get()
        assert fm1 is fm2

    def test_reset_clears_singleton(self):
        fm1 = FeatureMatrix.get()
        FeatureMatrix.reset()
        fm2 = FeatureMatrix.get()
        assert fm1 is not fm2


# ── FeatureMatrix.verify ──────────────────────────────────────────────────────

class TestFeatureMatrixVerify:
    def test_verify_returns_self(self):
        fm = FeatureMatrix()
        result = fm.verify(domain_mode="GENERAL", strict=False)
        assert result is fm

    def test_verify_sets_verified_flag(self):
        fm = FeatureMatrix()
        assert fm._verified is False
        fm.verify(domain_mode="GENERAL")
        assert fm._verified is True

    def test_military_mode_strict_by_default(self):
        """MILITARY domain forces strict=True even if caller passes strict=False."""
        fm = FeatureMatrix()
        # If any required-for-MILITARY cap is unavailable, strict mode raises.
        # We patch _check_capability to return all AVAILABLE to avoid false positives.
        with patch.object(fm, "_check_capability", side_effect=lambda c: _mark_available(c)):
            result = fm.verify(domain_mode="MILITARY", strict=False)
        assert result is fm  # no raise when all available

    def test_general_mode_missing_required_logs_not_raises(self):
        fm = FeatureMatrix()
        # All caps unavailable — GENERAL mode should warn but not raise
        with patch.object(fm, "_check_capability", side_effect=lambda c: _mark_unavailable(c)):
            fm.verify(domain_mode="GENERAL", strict=False)  # no raise

    def test_strict_mode_raises_on_missing_required(self):
        """Strict mode raises ConfigurationError when a required cap is missing."""
        fm = FeatureMatrix()
        # Inject a fake required capability that is unavailable
        fake_cap = Capability(
            name="critical_tool",
            description="Critical for MILITARY",
            check_type="executable",
            check_target="nonexistent_binary_xyz",
            required_for={"MILITARY"},
        )
        with patch("startup.feature_matrix._CAPABILITIES", [fake_cap]):
            with pytest.raises(ConfigurationError):
                fm.verify(domain_mode="MILITARY", strict=True)


# ── FeatureMatrix.is_available ────────────────────────────────────────────────

class TestFeatureMatrixIsAvailable:
    def test_available_cap_returns_true(self):
        fm = FeatureMatrix()
        fm._capabilities["test"] = Capability(
            name="test", description="", check_type="import", check_target="",
            status=CapabilityStatus.AVAILABLE,
        )
        assert fm.is_available("test") is True

    def test_degraded_cap_returns_true(self):
        fm = FeatureMatrix()
        fm._capabilities["test"] = Capability(
            name="test", description="", check_type="import", check_target="",
            status=CapabilityStatus.DEGRADED,
        )
        assert fm.is_available("test") is True

    def test_unavailable_cap_returns_false(self):
        fm = FeatureMatrix()
        fm._capabilities["test"] = Capability(
            name="test", description="", check_type="import", check_target="",
            status=CapabilityStatus.UNAVAILABLE,
        )
        assert fm.is_available("test") is False

    def test_unknown_cap_returns_false(self):
        fm = FeatureMatrix()
        assert fm.is_available("nonexistent_capability") is False


# ── FeatureMatrix.require ─────────────────────────────────────────────────────

class TestFeatureMatrixRequire:
    def test_require_available_no_raise(self):
        fm = FeatureMatrix()
        fm._capabilities["present"] = Capability(
            name="present", description="", check_type="import", check_target="",
            status=CapabilityStatus.AVAILABLE,
        )
        fm.require("present")  # no raise

    def test_require_unavailable_raises(self):
        fm = FeatureMatrix()
        fm._capabilities["missing"] = Capability(
            name="missing", description="", check_type="import", check_target="",
            status=CapabilityStatus.UNAVAILABLE,
        )
        with pytest.raises(ConfigurationError):
            fm.require("missing")

    def test_require_unknown_raises(self):
        fm = FeatureMatrix()
        with pytest.raises(ConfigurationError):
            fm.require("totally_unknown")


# ── FeatureMatrix.report ──────────────────────────────────────────────────────

class TestFeatureMatrixReport:
    def test_report_returns_dict(self):
        fm = FeatureMatrix()
        fm._capabilities["cap1"] = Capability(
            name="cap1", description="desc", check_type="import", check_target="",
            status=CapabilityStatus.AVAILABLE, version="1.0",
        )
        report = fm.report()
        assert isinstance(report, dict)
        assert "cap1" in report
        assert report["cap1"]["status"] == "AVAILABLE"

    def test_report_includes_version(self):
        fm = FeatureMatrix()
        fm._capabilities["cap1"] = Capability(
            name="cap1", description="", check_type="import", check_target="",
            status=CapabilityStatus.AVAILABLE, version="2.3.4",
        )
        report = fm.report()
        assert report["cap1"]["version"] == "2.3.4"


# ── Module-level helpers ──────────────────────────────────────────────────────

class TestModuleLevelHelpers:
    def test_verify_startup_returns_feature_matrix(self):
        result = verify_startup(domain_mode="GENERAL", strict=False)
        assert isinstance(result, FeatureMatrix)

    def test_is_available_delegates_to_singleton(self):
        fm = FeatureMatrix.get()
        fm._capabilities["known_cap"] = Capability(
            name="known_cap", description="", check_type="import", check_target="",
            status=CapabilityStatus.AVAILABLE,
        )
        assert is_available("known_cap") is True

    def test_require_capability_raises_on_missing(self):
        fm = FeatureMatrix.get()
        fm._capabilities["needed"] = Capability(
            name="needed", description="", check_type="import", check_target="",
            status=CapabilityStatus.UNAVAILABLE,
        )
        with pytest.raises(ConfigurationError):
            require_capability("needed")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mark_available(cap: Capability) -> Capability:
    cap.status = CapabilityStatus.AVAILABLE
    return cap


def _mark_unavailable(cap: Capability) -> Capability:
    cap.status = CapabilityStatus.UNAVAILABLE
    return cap
