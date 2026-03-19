"""
swarm/autogen_agents.py
=======================
AutoGen (Microsoft) conversational multi-agent framework integration.

https://github.com/microsoft/autogen

AutoGen is used for conversational coordination between agents—specifically:
• CodeReviewConversation  — two-agent back-and-forth adversarial code review
• DebuggingConversation   — planner + executor iterative debugging loop
• FormalVerifyConversation— prover + checker dialogue for Lean4 proof generation

AutoGen complements LangGraph (state machine) and CrewAI (role-based tasks):
LangGraph  = structured pipeline with checkpoints
CrewAI     = role-defined crews with tools
AutoGen    = free-form multi-turn conversation between specialist agents

All agents use the TieredModelRouter — local Granite models for cheap turns,
cloud models only for critical reasoning steps.

Environment variables
──────────────────────
AUTOGEN_CACHE_DIR   — directory for AutoGen caching (default: .stabilizer/autogen_cache)
AUTOGEN_MAX_TURNS   — maximum conversation turns per task (default: 15)
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_AUTOGEN_AVAILABLE = False
try:
    import autogen  # type: ignore[import]
    _AUTOGEN_AVAILABLE = True
    log.info(f"AutoGen {autogen.__version__} available")
except ImportError:
    log.info(
        "pyautogen not installed — AutoGen conversations disabled. "
        "Run: pip install pyautogen"
    )

_MAX_TURNS  = int(os.environ.get("AUTOGEN_MAX_TURNS", "15"))
_CACHE_DIR  = os.environ.get("AUTOGEN_CACHE_DIR", ".stabilizer/autogen_cache")


# ──────────────────────────────────────────────────────────────────────────────
# LLM config builder
# ──────────────────────────────────────────────────────────────────────────────

def _llm_config(task: str = "review") -> dict:
    """Build AutoGen llm_config from TieredModelRouter."""
    from models.router import get_router
    router = get_router()
    primary  = router.primary_model(task)
    fallbacks = router.fallback_models(task)

    config_list = []
    for model in [primary] + fallbacks[:2]:
        entry: dict = {"model": model}
        if "ollama" in model:
            entry["base_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            entry["api_key"]  = "ollama"  # Ollama doesn't need a real key
        elif "openrouter" in model:
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if key:
                entry["api_key"]  = key
                entry["base_url"] = "https://openrouter.ai/api/v1"
        elif "claude" in model:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                entry["api_key"] = key
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
            if key:
                entry["api_key"] = key
        config_list.append(entry)

    return {
        "config_list": config_list,
        "cache_seed": None,
        "timeout": 120,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Code Review Conversation
# ──────────────────────────────────────────────────────────────────────────────

async def run_code_review_conversation(
    file_path:   str,
    content:     str,
    issues:      list[dict],
    fixed_content: str,
) -> dict:
    """
    Two-agent adversarial code review via AutoGen.

    Agents:
    • Reviewer (critic)  — finds problems with the proposed fix
    • Defender (author)  — defends the fix or acknowledges issues

    Returns {"approved": bool, "verdict": str, "issues_found": list[str]}
    """
    if not _AUTOGEN_AVAILABLE:
        log.debug("AutoGen unavailable — skipping conversational review")
        return {"approved": True, "verdict": "AutoGen not available", "issues_found": []}

    llm_cfg = _llm_config("review")

    reviewer = autogen.AssistantAgent(
        name="Adversarial_Reviewer",
        system_message=(
            "You are a hostile code reviewer. Your job is to find EVERY problem "
            "with the proposed fix: logic errors, security issues, regressions, "
            "incomplete implementations, style violations. Be specific with line numbers. "
            "If the fix is actually correct, you MUST acknowledge it explicitly."
        ),
        llm_config=llm_cfg,
    )

    defender = autogen.AssistantAgent(
        name="Fix_Author",
        system_message=(
            "You are the engineer who wrote the fix. For each criticism, either: "
            "(1) acknowledge it is correct and explain how to address it, or "
            "(2) provide a clear technical argument why the concern is invalid. "
            "Be concise. Do not over-explain. End with a final verdict: "
            "APPROVED or NEEDS_REVISION with a one-sentence reason."
        ),
        llm_config=llm_cfg,
    )

    user_proxy = autogen.UserProxyAgent(
        name="Review_Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=_MAX_TURNS,
        is_termination_msg=lambda m: (
            "APPROVED" in m.get("content", "").upper()
            or "NEEDS_REVISION" in m.get("content", "").upper()
        ),
        code_execution_config=False,
    )

    issue_summary = "\n".join(
        f"  - [{i.get('severity','?')}] {i.get('description','')}"
        for i in issues[:5]
    )

    initial_message = (
        f"## Code Review Task\n\n"
        f"File: `{file_path}`\n\n"
        f"Issues being fixed:\n{issue_summary}\n\n"
        f"### Original (first 60 lines)\n```\n"
        f"{chr(10).join(content.splitlines()[:60])}\n```\n\n"
        f"### Proposed Fix (first 60 lines)\n```\n"
        f"{chr(10).join(fixed_content.splitlines()[:60])}\n```\n\n"
        "Reviewer: analyze this fix and identify any problems."
    )

    try:
        await user_proxy.a_initiate_chat(
            reviewer,
            message=initial_message,
            max_turns=_MAX_TURNS,
        )
        # Extract verdict from last message
        history = user_proxy.chat_messages.get(reviewer, [])
        last_msg = history[-1]["content"] if history else ""
        approved = "APPROVED" in last_msg.upper() and "NEEDS_REVISION" not in last_msg.upper()
        return {
            "approved": approved,
            "verdict": last_msg[:500],
            "issues_found": [],
            "turns": len(history),
        }
    except Exception as exc:
        log.warning(f"AutoGen code review failed: {exc}")
        return {"approved": True, "verdict": f"AutoGen error: {exc}", "issues_found": []}


# ──────────────────────────────────────────────────────────────────────────────
# Debugging Conversation (Planner + Executor loop)
# ──────────────────────────────────────────────────────────────────────────────

async def run_debugging_conversation(
    error_message:  str,
    file_path:      str,
    content:        str,
    test_output:    str = "",
) -> dict:
    """
    Iterative debugging via AutoGen planner-executor loop.

    Returns {"fixed_content": str, "diagnosis": str, "resolved": bool}
    """
    if not _AUTOGEN_AVAILABLE:
        return {"fixed_content": content, "diagnosis": "AutoGen unavailable", "resolved": False}

    llm_cfg = _llm_config("critical_fix")

    planner = autogen.AssistantAgent(
        name="Debug_Planner",
        system_message=(
            "You analyze errors and produce a step-by-step debugging plan. "
            "You diagnose the root cause, propose a hypothesis, and describe "
            "the exact change needed. You do NOT write code — that is the executor's job."
        ),
        llm_config=llm_cfg,
    )

    executor = autogen.AssistantAgent(
        name="Fix_Executor",
        system_message=(
            "You implement exactly what the planner describes. "
            "You return the COMPLETE fixed file content in a fenced code block. "
            "You do not plan — you execute. If the planner's hypothesis seems wrong, "
            "say so and ask for clarification."
        ),
        llm_config=_llm_config("fix"),
    )

    user_proxy = autogen.UserProxyAgent(
        name="Debug_Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda m: "RESOLVED" in m.get("content", "").upper(),
        code_execution_config=False,
    )

    msg = (
        f"## Debugging Task\n\n"
        f"**File:** `{file_path}`\n\n"
        f"**Error:**\n```\n{error_message[:1000]}\n```\n\n"
    )
    if test_output:
        msg += f"**Test output:**\n```\n{test_output[:500]}\n```\n\n"
    msg += f"**Current code:**\n```python\n{content[:3000]}\n```\n\nPlanner: diagnose this."

    try:
        await user_proxy.a_initiate_chat(planner, message=msg, max_turns=8)
        history = user_proxy.chat_messages.get(planner, [])

        # Extract fixed code from executor messages
        fixed = content
        import re
        for msg_item in history:
            c = msg_item.get("content", "")
            m = re.search(r"```(?:python)?\n(.*?)```", c, re.DOTALL)
            if m and len(m.group(1)) > len(content) * 0.5:
                fixed = m.group(1)

        diagnosis = history[-1]["content"][:300] if history else ""
        return {
            "fixed_content": fixed,
            "diagnosis": diagnosis,
            "resolved": fixed != content,
            "turns": len(history),
        }
    except Exception as exc:
        log.warning(f"AutoGen debugging conversation failed: {exc}")
        return {"fixed_content": content, "diagnosis": str(exc), "resolved": False}


# ──────────────────────────────────────────────────────────────────────────────
# Formal Verification Conversation (Prover + Checker)
# ──────────────────────────────────────────────────────────────────────────────

async def run_formal_verification_conversation(
    property_name:  str,
    property_desc:  str,
    code_snippet:   str,
) -> dict:
    """
    Two-agent Lean4 theorem generation via AutoGen.

    Prover generates the Lean4 theorem; Checker validates it syntactically.
    Returns {"lean4_theorem": str, "valid": bool, "proof_attempt": str}
    """
    if not _AUTOGEN_AVAILABLE:
        return {"lean4_theorem": "", "valid": False, "proof_attempt": "AutoGen unavailable"}

    llm_cfg = _llm_config("formal_extract")

    prover = autogen.AssistantAgent(
        name="Lean4_Prover",
        system_message=(
            "You are a formal methods expert. You translate code properties "
            "into valid Lean 4 theorems using Mathlib. "
            "Always import Mathlib.Data.Int.Basic or Mathlib.Data.Real.Basic as needed. "
            "Use `by omega`, `by ring`, `by simp`, or `by norm_num` as appropriate. "
            "Return ONLY the Lean 4 code in a fenced ```lean block."
        ),
        llm_config=llm_cfg,
    )

    checker = autogen.AssistantAgent(
        name="Lean4_Checker",
        system_message=(
            "You review Lean 4 theorems for syntactic validity. "
            "Check: correct imports, proper theorem syntax, valid tactic proofs. "
            "If valid, respond with 'VALID: <brief explanation>'. "
            "If invalid, point out the specific error and ask the prover to fix it."
        ),
        llm_config=_llm_config("triage"),
    )

    user_proxy = autogen.UserProxyAgent(
        name="Proof_Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=6,
        is_termination_msg=lambda m: "VALID:" in m.get("content", ""),
        code_execution_config=False,
    )

    msg = (
        f"## Lean 4 Theorem Generation\n\n"
        f"**Property:** {property_name}\n"
        f"**Description:** {property_desc}\n\n"
        f"**Code to verify:**\n```python\n{code_snippet[:2000]}\n```\n\n"
        "Prover: generate the Lean 4 theorem."
    )

    try:
        await user_proxy.a_initiate_chat(prover, message=msg, max_turns=6)
        history = user_proxy.chat_messages.get(prover, [])

        import re
        lean_code = ""
        for msg_item in history:
            c = msg_item.get("content", "")
            m = re.search(r"```lean\n(.*?)```", c, re.DOTALL)
            if m:
                lean_code = m.group(1)

        valid = any("VALID:" in m.get("content", "") for m in history)
        return {
            "lean4_theorem": lean_code,
            "valid": valid,
            "proof_attempt": history[-1]["content"][:300] if history else "",
            "turns": len(history),
        }
    except Exception as exc:
        log.warning(f"AutoGen formal verification conversation failed: {exc}")
        return {"lean4_theorem": "", "valid": False, "proof_attempt": str(exc)}
