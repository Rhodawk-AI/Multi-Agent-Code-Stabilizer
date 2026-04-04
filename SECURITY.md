# Security Policy

Rhodawk AI Code Stabilizer is designed to audit and stabilize safety-critical software. Because our architecture includes executing code, manipulating massive repositories, and orchestrating autonomous LLM agents, security is our absolute highest priority.

## Supported Versions

We currently provide security updates for the following versions:

| Version | Supported          | Notes                                      |
| ------- | ------------------ | ------------------------------------------ |
| `1.0.x` | :white_check_mark: | Current Mainline (Alpha / Dev Preview)     |
| `< 1.0` | :x:                | Pre-release architecture (unsupported)     |

## Reporting a Vulnerability

**DO NOT OPEN A PUBLIC GITHUB ISSUE FOR SECURITY VULNERABILITIES.**

Publicly disclosing a vulnerability in the orchestration layer, Aegis EDR, or local execution sandbox could put enterprise deployments at risk. 

To report a security issue, please email our security team directly:
📧 **founder@rhodawkai.com**

### What to include in your report:
* Type of issue (e.g., Prompt Injection, Sandbox Escape, JWT Auth bypass, Path Traversal).
* Step-by-step instructions to reproduce the vulnerability.
* Proof-of-Concept (PoC) code or execution logs.
* The specific model (e.g., `granite-code:8b`) and environment (OS/Python version) used during the exploit.

### Response SLA
1. We will acknowledge receipt of your vulnerability report within **24 hours**.
2. We will provide a triage assessment and expected timeline for a patch within **72 hours**.
3. Once the patch is merged, we will credit you in our Security Advisories (unless you prefer to remain anonymous).

## Scope of the Bug Bounty / Security Program

**In Scope:**
* **Aegis EDR Layer:** Any ability to bypass the pre-commit scan or execute malicious payloads on the host machine.
* **Orchestration Logic:** Ways to hijack the LangGraph/CrewAI flow to force the agents into an infinite destructive loop.
* **Authentication:** Bypassing the JWT middleware or HMAC-signed audit trails.
* **Data Exfiltration:** Forcing the system to leak proprietary codebase context from the HelixDB (Qdrant) memory layer to an external server.

**Out of Scope (Do not report):**
* General LLM hallucinations or logic errors that do not impact system security (these should be reported as standard bugs via Issue Templates).
* Vulnerabilities in third-party LLM providers (e.g., Anthropic, OpenRouter) or the Ollama runtime itself, unless Rhodawk AI implements them in an explicitly insecure manner.

## Safe Harbor
We consider activities conducted consistent with this policy to constitute "authorized" conduct. We will not initiate legal action or law enforcement investigation against anyone reporting vulnerabilities in good faith and in compliance with this policy.
