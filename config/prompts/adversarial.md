# Adversarial Robustness Audit — Section VIII
# For systems that process untrusted inputs or operate in adversarial environments

## A1 — Prompt Injection Prevention (CRITICAL)
For LLM-integrated systems:
- All user-supplied content fed to LLMs must be clearly delimited
- System prompts must not be overridable by user input
- Output from LLMs must be validated before acting on it

## A2 — Repo Content Sanitization (CRITICAL)
When reading code from untrusted repos:
- Strip or flag comment blocks containing instruction-like patterns
- Never execute or `eval()` content from the repo
- Validate all file paths returned by LLMs against repo root

## A3 — Credential Isolation (CRITICAL)
- GitHub write credentials must never be in the same process as LLM call credentials
- No credentials logged in plaintext
- All secrets must come from environment variables, never from config files in the repo

## A4 — Output Validation (MAJOR)
Before applying any LLM-generated fix:
- Verify the diff only modifies declared scope files
- Reject fixes that modify .env, secrets, CI config, or security infrastructure
- Run static analysis on generated code before writing to disk
