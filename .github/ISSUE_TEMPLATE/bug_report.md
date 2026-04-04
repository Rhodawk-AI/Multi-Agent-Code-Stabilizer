---
name: "🐛 System Bug / Core Failure"
about: Report a failure in the MACS agent swarm, routing, or verification layers.
title: "[BUG] "
labels: bug, triage
assignees: ''
---

## 🛫 Pre-flight Checks
- [ ] I have checked existing open and closed issues to ensure this is not a duplicate.
- [ ] I am using the latest `main` branch version of Rhodawk AI.
- [ ] This is NOT a security vulnerability (Security issues must be emailed to founder@rhodawkai.com per the `SECURITY.md`).

## 🛑 Failure Description
A clear and concise description of what broke within the MACS system. 
*(e.g., "The Architect agent got stuck in an infinite loop during formal verification," or "HelixDB failed to ingest a repository larger than 100k lines.")*

## 🔄 Reproduction Steps
Steps to reproduce the behavior:
1. Run command: `...`
2. Agent triggered: `...`
3. Resulting error or hallucination: `...`

## 📉 Expected vs. Actual Behavior
- **Expected:** What the system should have done (e.g., "The Validator agent should have rejected the patch and sent it back to the Engineer").
- **Actual:** What actually happened.

## 💻 Environment Setup
Please provide your environment details to help us debug the execution sandbox:
- **OS:** [e.g., Ubuntu 22.04, macOS Sonoma, Windows 11 / WSL2]
- **Python Version:** [e.g., 3.11.4]
- **Ollama Version (if using local models):** [e.g., 0.1.27]
- **Tier 1/2 Model Used:** [e.g., granite-code:8b]
- **Tier 3/4 Fallback Used:** [e.g., Claude 3.5 Sonnet, if applicable]

## 📜 Error Logs & Execution Traces
Paste the terminal output, Swarm communication logs, or stack trace here. Please wrap it in code blocks. 
*Note: Ensure you scrub any sensitive credentials or proprietary code before pasting.*

```text
[Paste logs here]
