"""
startup/feature_matrix.py
==========================
Pre-flight feature matrix verifier for Rhodawk AI Code Stabilizer.

AUDIT FIX: The review identified that optional dependencies could silently
degrade at runtime, causing production deployments to operate without
critical capabilities (Z3, NetworkX, tree-sitter, Qdrant) with only a
log warning. For military/aerospace domain modes these are hard failures.

This module:
  • Enumerates all optional capabilities and their import paths
  • Verifies executable tools (clang-tidy, cppcheck, cbmc, ldra) via shutil.which
  • Maps each capability to the DomainMode(s) that require it
  • Raises ConfigurationError for any required capability missing in strict modes
  • Emits a structured capability report at startup
  • Provides a FeatureMatrix singleton consumed by controller.py
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class ConfigurationError(RuntimeError):
    """Raised when a required capability is missing for the configured domain."""


# ─────────────────────────────────────────────────────────────────────────────
# Capability definitions
# ─────────────────────────────────────────────────────────────────────────────

class CapabilityStatus(str, Enum):
    AVAILABLE   = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED    = "DEGRADED"    # Partial functionality


@dataclass
class Capability:
    name:             str
    description:      str
    check_type:       str              # "import" | "executable" | "subprocess"
    check_target:     str              # module name, binary name, or command
    required_for:     set[str]         = field(default_factory=set)  # DomainMode values
    status:           CapabilityStatus = CapabilityStatus.UNAVAILABLE
    version:          str              = ""
    degraded_fallback: str             = ""
    error_detail:     str              = ""


# Canonical capability registry
# DomainMode strings used: MILITARY, AEROSPACE, NUCLEAR, MEDICAL, FINANCE, EMBEDDED, GENERAL
_CAPABILITIES: list[Capability] = [
    # ── Python library capabilities ──────────────────────────────────────────
    Capability(
        name="networkx",
        description="Dependency graph engine (import/call graph, centrality)",
        check_type="import", check_target="networkx",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="Dependency graph disabled; impact radius unavailable",
    ),
    Capability(
        name="z3_solver",
        description="Z3 SMT solver for formal property proofs",
        check_type="import", check_target="z3",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL"},
        degraded_fallback="Formal verification falls back to static pattern matching only",
    ),
    Capability(
        name="tree_sitter",
        description="Multi-language AST parser for syntax validation and chunking",
        check_type="import", check_target="tree_sitter",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="Syntax gate falls back to regexp heuristic",
    ),
    Capability(
        name="tree_sitter_language_pack",
        description="Tree-sitter language pack (C, C++, Rust, Go, etc.)",
        check_type="import", check_target="tree_sitter_language_pack",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="Multi-language syntax gate degraded; C/C++ AST parsing unavailable",
    ),
    Capability(
        name="langgraph",
        description="LangGraph state machine orchestration",
        check_type="import", check_target="langgraph",
        required_for=set(),
        degraded_fallback="LangGraph orchestration disabled; DeerFlow used as fallback",
    ),
    Capability(
        name="crewai",
        description="CrewAI role-based agent crews",
        check_type="import", check_target="crewai",
        required_for=set(),
        degraded_fallback="CrewAI domain specialist crews disabled",
    ),
    Capability(
        name="autogen",
        description="AutoGen conversational multi-agent framework",
        check_type="import", check_target="autogen",
        required_for=set(),
        degraded_fallback="AutoGen formal verification conversations disabled",
    ),
    Capability(
        name="instructor",
        description="Instructor structured LLM output (Pydantic v2)",
        check_type="import", check_target="instructor",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL", "FINANCE",
                      "EMBEDDED", "GENERAL"},
        degraded_fallback="NONE — structured LLM output is critical",
    ),
    Capability(
        name="litellm",
        description="LiteLLM unified model router",
        check_type="import", check_target="litellm",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL", "FINANCE",
                      "EMBEDDED", "GENERAL"},
        degraded_fallback="NONE — model routing is critical",
    ),
    Capability(
        name="aiosqlite",
        description="Async SQLite storage backend",
        check_type="import", check_target="aiosqlite",
        required_for={"GENERAL"},
        degraded_fallback="Database unavailable",
    ),
    Capability(
        name="asyncpg",
        description="Async PostgreSQL storage backend (production)",
        check_type="import", check_target="asyncpg",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL"},
        degraded_fallback="Falls back to SQLite — NOT suitable for production military use",
    ),
    Capability(
        name="prometheus_client",
        description="Prometheus metrics exporter",
        check_type="import", check_target="prometheus_client",
        required_for=set(),
        degraded_fallback="Metrics disabled",
    ),
    Capability(
        name="tenacity",
        description="LLM call retry/resilience library",
        check_type="import", check_target="tenacity",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="No retry on LLM failures — single-shot only",
    ),
    Capability(
        name="cryptography",
        description="HMAC-SHA256 audit trail signing",
        check_type="import", check_target="cryptography",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL"},
        degraded_fallback="Audit trail unsigned — DO-178C SCM evidence invalid",
    ),
    # ── Executable tool capabilities ─────────────────────────────────────────
    Capability(
        name="clang_tidy",
        description="clang-tidy C/C++ static analyzer (MISRA, CWE, CERT)",
        check_type="executable", check_target="clang-tidy",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="C/C++ MISRA/CERT static analysis unavailable",
    ),
    Capability(
        name="cppcheck",
        description="cppcheck C/C++ static analyzer",
        check_type="executable", check_target="cppcheck",
        required_for={"MILITARY", "AEROSPACE"},
        degraded_fallback="C/C++ secondary static analysis unavailable",
    ),
    Capability(
        name="cbmc",
        description="CBMC bounded model checker (formal C/C++ proofs)",
        check_type="executable", check_target="cbmc",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="Bounded model checking unavailable — DO-178C formal evidence gap",
    ),
    Capability(
        name="semgrep",
        description="Semgrep pattern-based static analysis",
        check_type="executable", check_target="semgrep",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL"},
        degraded_fallback="Semgrep pattern analysis unavailable",
    ),
    Capability(
        name="ruff",
        description="Ruff Python linter",
        check_type="executable", check_target="ruff",
        required_for={"GENERAL", "FINANCE", "MEDICAL"},
        degraded_fallback="Python linting unavailable",
    ),
    Capability(
        name="bandit",
        description="Bandit Python security linter",
        check_type="executable", check_target="bandit",
        required_for={"GENERAL", "FINANCE", "MEDICAL"},
        degraded_fallback="Python security linting unavailable",
    ),
    Capability(
        name="mypy",
        description="Mypy Python type checker",
        check_type="executable", check_target="mypy",
        required_for=set(),
        degraded_fallback="Python type checking unavailable",
    ),
    Capability(
        name="clang_preprocessor",
        description="clang -E preprocessor expansion (for C macro analysis)",
        check_type="executable", check_target="clang",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="Macro expansion unavailable — chunking operates on raw source",
    ),
    Capability(
        name="cscope",
        description="cscope C/C++ cross-reference database (call graph)",
        check_type="executable", check_target="cscope",
        required_for=set(),
        degraded_fallback="Call graph MCP uses tree-sitter fallback",
    ),
    Capability(
        name="ctags",
        description="Universal ctags symbol index",
        check_type="executable", check_target="ctags",
        required_for=set(),
        degraded_fallback="Symbol index MCP unavailable",
    ),
    Capability(
        name="docker",
        description="Docker daemon (for ToolHive MCP container isolation)",
        check_type="executable", check_target="docker",
        required_for={"MILITARY", "AEROSPACE", "NUCLEAR"},
        degraded_fallback="MCP tools run in-process — NO ISOLATION",
    ),
    # SEC-4 FIX: validate RHODAWK_AUDIT_SECRET entropy and placeholder rejection.
    # An empty or placeholder audit secret (e.g. "CHANGE_ME_generate_with_python")
    # produces HMAC-SHA256 signatures that anyone who knows the key can forge,
    # invalidating the audit trail as tamper-evident evidence. Required for all
    # domains so this is caught before AuditTrailSigner is constructed.
    Capability(
        name="audit_secret",
        description="RHODAWK_AUDIT_SECRET entropy validation (non-empty, non-placeholder, ≥32 chars)",
        check_type="env_entropy",
        check_target="RHODAWK_AUDIT_SECRET",
        required_for={"GENERAL", "MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL", "FINANCE", "EMBEDDED"},
        degraded_fallback="NONE — set RHODAWK_AUDIT_SECRET to a 32+ char random hex string",
    ),
    # 0-A FIX: qdrant-client absence causes an unhandled ImportError crash at
    # _init_vector_store() even though the preflight check reports it as UNAVAILABLE
    # without blocking startup. Making it GENERAL-required surfaces the gap before
    # the controller crashes with a confusing traceback deep in the init sequence.
    # Operators who genuinely want in-memory-only mode can set CPG_ENABLED=0 and
    # HELIX_USE_LOCAL_QDRANT=1 (or equivalent) to skip vector store init entirely.
    Capability(
        name="qdrant_client",
        description="Qdrant vector database client (required for semantic search and context retrieval)",
        check_type="import", check_target="qdrant_client",
        required_for={"GENERAL", "MILITARY", "AEROSPACE", "NUCLEAR", "MEDICAL", "FINANCE", "EMBEDDED"},
        degraded_fallback="Vector similarity search unavailable — set HELIX_USE_LOCAL_QDRANT=1 "
                          "to use in-memory fallback, or start the Qdrant container",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# FeatureMatrix
# ─────────────────────────────────────────────────────────────────────────────

class FeatureMatrix:
    """
    Singleton that verifies all capabilities at startup and exposes
    is_available(name) for runtime feature gating throughout the system.
    """

    _instance: FeatureMatrix | None = None

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._verified = False

    @classmethod
    def get(cls) -> "FeatureMatrix":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def verify(self, domain_mode: str = "GENERAL", strict: bool = False) -> "FeatureMatrix":
        """
        Run all capability checks.

        Parameters
        ----------
        domain_mode:
            Active DomainMode string value (e.g. "MILITARY").
        strict:
            If True, raise ConfigurationError for any required capability
            that is unavailable.  Always True for military/aerospace/nuclear.
        """
        force_strict = domain_mode in {"MILITARY", "AEROSPACE", "NUCLEAR"}
        strict = strict or force_strict

        _names = [cap.name for cap in _CAPABILITIES]
        _dupes = {n for n in _names if _names.count(n) > 1}
        if _dupes:
            raise ValueError(
                f"Duplicate capability names in _CAPABILITIES: {_dupes}. "
                "Each capability must have a unique name."
            )

        for cap in _CAPABILITIES:
            checked = self._check_capability(cap)
            self._capabilities[checked.name] = checked

        self._verified = True
        self._emit_report(domain_mode)

        missing_required: list[str] = []
        for cap in self._capabilities.values():
            if (domain_mode in cap.required_for
                    and cap.status == CapabilityStatus.UNAVAILABLE):
                missing_required.append(
                    f"  [{cap.name}] {cap.description}\n"
                    f"    Degraded fallback: {cap.degraded_fallback or 'NONE'}"
                )

        if missing_required:
            msg = (
                f"\nFeatureMatrix: {len(missing_required)} required capabilities "
                f"unavailable for domain '{domain_mode}':\n"
                + "\n".join(missing_required)
            )
            if strict:
                raise ConfigurationError(msg)
            else:
                log.error(msg)

        return self

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self, name: str) -> bool:
        """Return True if the named capability is AVAILABLE or DEGRADED."""
        cap = self._capabilities.get(name)
        if cap is None:
            log.warning(f"FeatureMatrix.is_available({name!r}): unknown capability")
            return False
        return cap.status != CapabilityStatus.UNAVAILABLE

    def is_fully_available(self, name: str) -> bool:
        """Return True only if the named capability is fully AVAILABLE."""
        cap = self._capabilities.get(name)
        return cap is not None and cap.status == CapabilityStatus.AVAILABLE

    def require(self, name: str, context: str = "") -> None:
        """
        Assert that a capability is available.  Raises ConfigurationError if not.
        Use at the top of any code path that cannot degrade.
        """
        if not self.is_available(name):
            raise ConfigurationError(
                f"Required capability '{name}' is unavailable"
                + (f" (context: {context})" if context else "")
            )

    def version(self, name: str) -> str:
        cap = self._capabilities.get(name)
        return cap.version if cap else ""

    def report(self) -> dict[str, dict[str, str]]:
        """Return a structured capability report for the dashboard."""
        return {
            name: {
                "status": cap.status.value,
                "version": cap.version,
                "description": cap.description,
                "degraded_fallback": cap.degraded_fallback,
                "error": cap.error_detail,
            }
            for name, cap in self._capabilities.items()
        }

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check_capability(self, cap: Capability) -> Capability:
        try:
            if cap.check_type == "import":
                return self._check_import(cap)
            elif cap.check_type == "executable":
                return self._check_executable(cap)
            elif cap.check_type == "subprocess":
                return self._check_subprocess(cap)
            elif cap.check_type == "env_entropy":
                return self._check_env_entropy(cap)
            else:
                cap.status = CapabilityStatus.UNAVAILABLE
                cap.error_detail = f"Unknown check_type: {cap.check_type}"
                return cap
        except Exception as exc:
            cap.status = CapabilityStatus.UNAVAILABLE
            cap.error_detail = str(exc)
            return cap

    def _check_import(self, cap: Capability) -> Capability:
        try:
            mod = __import__(cap.check_target)
            cap.status = CapabilityStatus.AVAILABLE
            cap.version = getattr(mod, "__version__", "unknown")
        except ImportError as exc:
            cap.status = CapabilityStatus.UNAVAILABLE
            cap.error_detail = str(exc)
        return cap

    def _check_executable(self, cap: Capability) -> Capability:
        path = shutil.which(cap.check_target)
        if path is None:
            cap.status = CapabilityStatus.UNAVAILABLE
            cap.error_detail = f"'{cap.check_target}' not found in PATH"
            return cap
        # Try to get version
        try:
            result = subprocess.run(
                [cap.check_target, "--version"],
                capture_output=True, text=True, timeout=5
            )
            version_line = (result.stdout or result.stderr or "").splitlines()
            cap.version = version_line[0].strip() if version_line else "found"
            cap.status = CapabilityStatus.AVAILABLE
        except Exception as exc:
            cap.version = "found"
            cap.status = CapabilityStatus.AVAILABLE
            cap.error_detail = f"version check failed: {exc}"
        return cap

    def _check_subprocess(self, cap: Capability) -> Capability:
        try:
            import shlex
            parts = shlex.split(cap.check_target)
            result = subprocess.run(
                parts, capture_output=True, text=True, timeout=10
            )
            cap.status = (
                CapabilityStatus.AVAILABLE
                if result.returncode == 0
                else CapabilityStatus.DEGRADED
            )
            cap.version = result.stdout.strip()[:80]
        except FileNotFoundError:
            cap.status = CapabilityStatus.UNAVAILABLE
            cap.error_detail = f"Command not found: {cap.check_target}"
        except subprocess.TimeoutExpired:
            cap.status = CapabilityStatus.DEGRADED
            cap.error_detail = "Timeout — process may be available but slow"
        return cap

    def _check_env_entropy(self, cap: Capability) -> Capability:
        import collections, math
        val = __import__('os').environ.get(cap.check_target, '')
        if not val or len(val) < 32:
            cap.status = CapabilityStatus.UNAVAILABLE
            cap.error_detail = f"{cap.check_target} is empty or < 32 chars"
            return cap
        bad = ('CHANGE_ME', 'changeme', 'change_me', 'placeholder', 'PLACEHOLDER')
        if any(val.lower().startswith(p.lower()) for p in bad):
            cap.status = CapabilityStatus.UNAVAILABLE
            cap.error_detail = f"{cap.check_target} is a known placeholder"
            return cap
        freq = collections.Counter(val)
        entropy = -sum((c/len(val))*math.log2(c/len(val)) for c in freq.values())
        if entropy < 4.0:
            cap.status = CapabilityStatus.DEGRADED
            cap.error_detail = f"Low entropy ({entropy:.2f} bits/char < 4.0)"
            return cap
        cap.status = CapabilityStatus.AVAILABLE
        return cap

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def _emit_report(self, domain_mode: str) -> None:
        available  = [c for c in self._capabilities.values()
                      if c.status == CapabilityStatus.AVAILABLE]
        degraded   = [c for c in self._capabilities.values()
                      if c.status == CapabilityStatus.DEGRADED]
        missing    = [c for c in self._capabilities.values()
                      if c.status == CapabilityStatus.UNAVAILABLE]
        required_missing = [c for c in missing if domain_mode in c.required_for]

        log.info(
            f"FeatureMatrix [{domain_mode}]: "
            f"{len(available)} available, "
            f"{len(degraded)} degraded, "
            f"{len(missing)} unavailable "
            f"({len(required_missing)} REQUIRED missing)"
        )
        for cap in required_missing:
            log.error(
                f"  REQUIRED MISSING: [{cap.name}] {cap.description} — "
                f"fallback: {cap.degraded_fallback or 'NONE'}"
            )
        for cap in degraded:
            log.warning(f"  DEGRADED: [{cap.name}] — {cap.degraded_fallback}")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def verify_startup(domain_mode: str = "GENERAL", strict: bool = False) -> FeatureMatrix:
    """
    Entry point called by StabilizerController.initialise().
    Returns the verified FeatureMatrix singleton.
    """
    fm = FeatureMatrix.get()
    fm.verify(domain_mode=domain_mode, strict=strict)
    return fm


def require_capability(name: str, context: str = "") -> None:
    """Convenience wrapper — raises ConfigurationError if unavailable."""
    FeatureMatrix.get().require(name, context)


def is_available(name: str) -> bool:
    """Convenience wrapper for runtime feature gating."""
    return FeatureMatrix.get().is_available(name)
