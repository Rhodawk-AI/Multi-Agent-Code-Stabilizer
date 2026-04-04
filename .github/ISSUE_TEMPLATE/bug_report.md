---
name: "\U0001F41B System Bug / Core Failure"
about: Report a failure in the agent swarm, routing, or verification layers.
title: "[BUG] "
labels: bug, triage
assignees: ''
---

## 🛑 Failure Description
A clear and concise description of what broke within the MACS system. Did an agent hallucinate? Did the verification loop fail? 

## 🔄 Reproduction Steps
Steps to reproduce the behavior:
1. Run command '...'
2. Agent triggered '...'
3. See error

## 📉 Expected vs. Actual Behavior
**Expected:** What the system should have done (e.g., "The Validator agent should have rejected the patch").
**Actual:** What actually happened.

## 💻 Environment Setup
Please provide your environment details to help us debug the execution environment:
- **OS:** [e.g., Ubuntu 22.04, macOS Sonoma, Windows 11 / WSL2]
- **Python Version:** [e.g., 3.11.4]
- **Ollama Version:** [e.g., 0.1.27]
- **Primary Model Used:** [e.g., granite-code:8b]

## 📜 Error Logs & Traces
Paste the terminal output or log file here. Please wrap it in code blocks. 
*Note: Ensure you scrub any sensitive credentials or proprietary code before pasting.*

```text
[Paste logs here]
