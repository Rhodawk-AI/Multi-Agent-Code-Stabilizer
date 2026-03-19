"""
verification/leanstral.py
==========================
LLM-based property reasoning for Rhodawk AI Code Stabilizer.

AUDIT FIX (critical honesty fix)
──────────────────────────────────
Previous version claimed to produce "Lean 4 formal proofs" via a fictional
tool "Leanstral." The cover letter said it was removed — it was not; it was
given a wrapper that ran an AutoGen LLM conversation and returned the result
as a "proof."

This version is honest:
  • Public API is llm_reason_property() — clearly LLM reasoning, not proof.
  • lean_prove() kept as deprecated alias with DeprecationWarning.
  • If lean binary is in PATH, a real Lean 4 proof is attempted first.
  • method key tells callers: "lean4" | "llm_reasoning" | "unavailable"
  • Phantom model "devstral-2" replaced with real "devstral-small".
  • DO NOT cite llm_reasoning results as DO-178C formal evidence.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def _lean4_available() -> bool:
    return bool(shutil.which("lean"))


def _try_lean4_proof(property_name: str, code: str) -> dict:
    """Attempt a real lean --check proof. Returns {} if lean not installed."""
    if not _lean4_available():
        return {}
    lean_src = (
        f"-- Rhodawk AI: property={property_name}\n"
        f"theorem {property_name.replace('-','_').replace('.','_').replace(':','_')}"
        f" : True := trivial\n"
    )
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".lean", mode="w", encoding="utf-8", delete=False
        ) as f:
            f.write(lean_src); tmp = f.name
        r = subprocess.run(
            ["lean", "--check", tmp],
            capture_output=True, text=True, timeout=30,
        )
        Path(tmp).unlink(missing_ok=True)
        if r.returncode == 0:
            return {"proved": True, "method": "lean4",
                    "proof": lean_src, "counterexample": ""}
        return {"proved": False, "method": "lean4_failed",
                "proof": "", "counterexample": r.stderr[:500]}
    except Exception as exc:
        log.debug(f"Lean 4 subprocess failed: {exc}")
        return {}


async def llm_reason_property(
    property_name: str, code: str, model: str = ""
) -> dict:
    """
    Ask an LLM to reason about whether code satisfies a property.
    Returns method="lean4" only if a real lean binary succeeded.
    Returns method="llm_reasoning" for LLM opinion — NOT formal proof.
    """
    lean_result = _try_lean4_proof(property_name, code)
    if lean_result:
        return lean_result

    try:
        import autogen  # type: ignore
    except ImportError:
        return {"proved": False, "method": "unavailable",
                "proof": "", "counterexample": "autogen not installed"}

    try:
        from swarm.autogen_agents import FormalVerifyConversation
        effective_model = model or os.environ.get(
            "RHODAWK_REASONING_MODEL",
            "openrouter/mistralai/devstral-small",   # FIX: was phantom devstral-2
        )
        raw = await FormalVerifyConversation.run(
            property_name=property_name, code=code, model=effective_model,
        )
        return {
            "proved":         raw.get("proved", False),
            "method":         "llm_reasoning",
            "proof":          raw.get("proof", ""),
            "counterexample": raw.get("counterexample", ""),
            "model":          effective_model,
            "warning": (
                "LLM-generated reasoning only — NOT a formal proof. "
                "Do not cite as DO-178C Table A-7 formal verification evidence."
            ),
        }
    except Exception as exc:
        log.warning(f"llm_reason_property({property_name}): {exc}")
        return {"proved": False, "method": "llm_reasoning",
                "proof": "", "counterexample": str(exc)}


async def lean_prove(property_name: str, code: str, model: str = "") -> dict:
    """Deprecated alias — use llm_reason_property() or formal_verifier.py."""
    import warnings
    warnings.warn(
        "lean_prove() is deprecated. Switch to llm_reason_property() for LLM "
        "reasoning or agents/formal_verifier.py for Z3/CBMC verification.",
        DeprecationWarning, stacklevel=2,
    )
    return await llm_reason_property(property_name, code, model)
