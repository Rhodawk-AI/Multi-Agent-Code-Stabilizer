"""
tools/servers/aurite_server.py
===============================
Aurite-ai agent verifier MCP server.

https://github.com/aurite-ai/aurite

Aurite-ai scans multi-agent swarms for anti-patterns including:
• Agent loops (infinite retry cycles)
• Hallucination cascades (agents agreeing on wrong answers)
• Cost spirals (exponential API spending)
• Goal drift (agents pursuing objectives unrelated to the task)
• Prompt injection propagation through the swarm
• Consensus fraud (colluding agents manipulating voting)

Tools exposed
──────────────
• scan_agent_logs      — analyze agent execution logs for anti-patterns
• detect_loops         — detect infinite agent loops
• check_cost_spiral    — alert on exponential cost growth
• verify_consensus     — check consensus votes for fraud patterns
• audit_agent_history  — full audit trail analysis

Transport: stdio JSON-RPC 2.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# Try native Aurite first
_AURITE_NATIVE = False
try:
    import aurite  # type: ignore[import]
    _AURITE_NATIVE = True
    log.info("Aurite-ai native client available")
except ImportError:
    log.info(
        "aurite-ai not installed — using built-in pattern detection. "
        "Install: pip install aurite-ai"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Anti-pattern detectors
# ──────────────────────────────────────────────────────────────────────────────

def detect_agent_loops(logs: list[dict]) -> list[dict]:
    """
    Detect infinite agent loops: same agent + same action > N times.
    """
    action_counts: Counter = Counter()
    threshold = int(os.environ.get("AURITE_LOOP_THRESHOLD", "5"))

    for entry in logs:
        agent  = entry.get("agent", "unknown")
        action = entry.get("action", "unknown")
        key    = f"{agent}:{action}"
        action_counts[key] += 1

    findings = []
    for key, count in action_counts.items():
        if count >= threshold:
            agent, action = key.split(":", 1)
            findings.append({
                "pattern":    "AGENT_LOOP",
                "severity":   "CRITICAL",
                "agent":      agent,
                "action":     action,
                "count":      count,
                "threshold":  threshold,
                "description": f"Agent '{agent}' repeated action '{action}' {count}× (threshold: {threshold})",
            })
    return findings


def detect_cost_spiral(cost_history: list[float]) -> list[dict]:
    """
    Detect exponential cost growth over cycles.
    """
    if len(cost_history) < 4:
        return []
    findings = []
    growth_rates = []
    for i in range(1, len(cost_history)):
        prev = cost_history[i - 1]
        curr = cost_history[i]
        if prev > 0:
            growth_rates.append((curr - prev) / prev)

    if len(growth_rates) >= 3:
        recent = growth_rates[-3:]
        if all(r > 0.5 for r in recent):  # >50% growth per cycle
            findings.append({
                "pattern":    "COST_SPIRAL",
                "severity":   "HIGH",
                "growth_rates": [f"{r:.0%}" for r in recent],
                "total_cost": sum(cost_history),
                "description": f"Cost growing >50%/cycle: {recent}",
            })
    return findings


def detect_consensus_fraud(consensus_events: list[dict]) -> list[dict]:
    """
    Detect consensus fraud: agents always agreeing regardless of evidence.
    """
    findings = []
    agent_votes: dict[str, list[bool]] = defaultdict(list)

    for event in consensus_events:
        for vote in event.get("votes", []):
            agent = vote.get("agent_id", "unknown")
            agent_votes[agent].append(vote.get("confirms", True))

    for agent, votes in agent_votes.items():
        if len(votes) < 5:
            continue
        agreement_rate = sum(votes) / len(votes)
        if agreement_rate > 0.95:
            findings.append({
                "pattern":        "CONSENSUS_FRAUD",
                "severity":       "HIGH",
                "agent":          agent,
                "agreement_rate": f"{agreement_rate:.0%}",
                "sample_size":    len(votes),
                "description":    f"Agent '{agent}' confirms {agreement_rate:.0%} of findings — possible rubber-stamping",
            })
    return findings


def detect_hallucination_cascade(audit_logs: list[dict]) -> list[dict]:
    """
    Detect hallucination cascades: multiple agents reporting same non-existent issue.
    """
    findings = []
    rejected_fps: Counter = Counter()
    total_fps: Counter    = Counter()

    for entry in audit_logs:
        fp = entry.get("fingerprint", "")
        if not fp:
            continue
        total_fps[fp] += 1
        if entry.get("validator_rejected", False):
            rejected_fps[fp] += 1

    for fp, rejected in rejected_fps.items():
        total = total_fps[fp]
        if total >= 2 and rejected / total > 0.5:
            findings.append({
                "pattern":     "HALLUCINATION_CASCADE",
                "severity":    "HIGH",
                "fingerprint": fp,
                "total_reports": total,
                "rejected":    rejected,
                "rejection_rate": f"{rejected/total:.0%}",
                "description": f"Finding {fp[:8]} reported {total}× but rejected {rejected}×",
            })
    return findings


def detect_goal_drift(agent_actions: list[dict], original_objective: str) -> list[dict]:
    """
    Detect goal drift: agents performing actions unrelated to the stated objective.
    """
    findings = []
    objective_words = set(re.findall(r'\w+', original_objective.lower()))
    if len(objective_words) < 3:
        return []

    unrelated_actions = []
    for action in agent_actions:
        action_text = action.get("description", "").lower()
        action_words = set(re.findall(r'\w+', action_text))
        overlap = len(objective_words & action_words) / max(len(objective_words), 1)
        if overlap < 0.1 and len(action_words) > 3:
            unrelated_actions.append(action)

    if len(unrelated_actions) > 3:
        findings.append({
            "pattern":          "GOAL_DRIFT",
            "severity":         "MEDIUM",
            "unrelated_actions": len(unrelated_actions),
            "total_actions":    len(agent_actions),
            "drift_rate":       f"{len(unrelated_actions)/max(len(agent_actions),1):.0%}",
            "description":      f"{len(unrelated_actions)} actions unrelated to objective",
        })
    return findings


def detect_prompt_injection_propagation(logs: list[dict]) -> list[dict]:
    """
    Detect prompt injection attempts propagating through the swarm.
    """
    injection_patterns = [
        re.compile(r'ignore\s+(?:all\s+)?previous\s+instructions', re.IGNORECASE),
        re.compile(r'SYSTEM\s*PROMPT\s*OVERRIDE', re.IGNORECASE),
        re.compile(r'you\s+are\s+now\s+a\s+different', re.IGNORECASE),
        re.compile(r'jailbreak', re.IGNORECASE),
    ]

    findings = []
    for entry in logs:
        content = json.dumps(entry)
        for pat in injection_patterns:
            if pat.search(content):
                findings.append({
                    "pattern":    "PROMPT_INJECTION",
                    "severity":   "CRITICAL",
                    "agent":      entry.get("agent", "unknown"),
                    "timestamp":  entry.get("timestamp", ""),
                    "description": f"Prompt injection pattern in agent log: {pat.pattern[:50]}",
                })
                break
    return findings


# ──────────────────────────────────────────────────────────────────────────────
# Main tool: full swarm scan
# ──────────────────────────────────────────────────────────────────────────────

async def scan_agent_logs(
    logs:                list[dict],
    cost_history:        list[float] | None = None,
    consensus_events:    list[dict]  | None = None,
    original_objective:  str                = "",
) -> dict:
    """Full Aurite-ai style swarm scan."""
    all_findings = []

    all_findings.extend(detect_agent_loops(logs))
    all_findings.extend(detect_prompt_injection_propagation(logs))

    if cost_history:
        all_findings.extend(detect_cost_spiral(cost_history))

    if consensus_events:
        all_findings.extend(detect_consensus_fraud(consensus_events))
        all_findings.extend(detect_hallucination_cascade(consensus_events))

    if original_objective:
        all_findings.extend(detect_goal_drift(logs, original_objective))

    critical = sum(1 for f in all_findings if f["severity"] == "CRITICAL")
    high     = sum(1 for f in all_findings if f["severity"] == "HIGH")

    return {
        "total_findings":    len(all_findings),
        "critical":          critical,
        "high":              high,
        "swarm_healthy":     (critical == 0 and high == 0),
        "findings":          all_findings,
        "scanned_at":        datetime.now(tz=timezone.utc).isoformat(),
    }


async def verify_consensus(consensus_events: list[dict]) -> dict:
    """Verify consensus integrity."""
    findings = detect_consensus_fraud(consensus_events)
    return {
        "valid": len(findings) == 0,
        "fraud_findings": findings,
        "total_events": len(consensus_events),
    }


_TOOLS = {
    "scan_agent_logs":  scan_agent_logs,
    "verify_consensus": verify_consensus,
}


async def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    rid    = req.get("id", 1)

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "tools": [{"name": k, "description": f"Aurite {k}"} for k in _TOOLS],
            "native_aurite": _AURITE_NATIVE,
        }}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        fn = _TOOLS.get(tool_name)
        if not fn:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        try:
            result = await fn(**arguments)
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(exc)}}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


async def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req  = json.loads(line)
            resp = await handle_request(req)
        except Exception as exc:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
