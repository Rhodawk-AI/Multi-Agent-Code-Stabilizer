"""
verification/leanstral.py
==========================
Lean 4 / Mistral formal proof assistant (AutoGen conversational mode).
Wraps AutoGen FormalVerifyConversation for DO-178C formal evidence.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)


async def lean_prove(property_name: str, code: str, model: str = "") -> dict:
    """
    Attempt to prove a property about code using Lean 4 via AutoGen.
    Returns dict with keys: proved (bool), proof (str), counterexample (str).
    """
    try:
        import autogen  # type: ignore
    except ImportError:
        log.info("autogen not installed — Lean 4 proof unavailable")
        return {"proved": False, "proof": "", "counterexample": "autogen not installed"}

    try:
        from swarm.autogen_agents import FormalVerifyConversation
        result = await FormalVerifyConversation.run(
            property_name=property_name,
            code=code,
            model=model or "openrouter/mistralai/devstral-2",
        )
        return result
    except Exception as exc:
        log.warning(f"lean_prove({property_name}): {exc}")
        return {"proved": False, "proof": "", "counterexample": str(exc)}
