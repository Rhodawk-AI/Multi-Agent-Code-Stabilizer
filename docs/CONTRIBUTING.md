# Contributing to RHODAWK AI CODE STABILIZER

RHODAWK AI CODE STABILIZER is fully open source. We welcome contributions of all kinds.

## Ways to Contribute

### 1. Audit Plugins
The fastest way to add value. Write a plugin that enforces a standard we don't support yet:

- MISRA-C / MISRA-C++
- AUTOSAR
- DO-178C compliance checks
- CWE/CERT rules
- Language-specific idioms
- Domain-specific (medical devices, avionics, finance)

See `plugins/base.py` for the plugin interface. See `plugins/builtins/no_secrets.py` for a working example.

### 2. LLM Adapters
Add support for a new model provider via LiteLLM configuration in `agents/base.py`.

### 3. Language Support
Extend `utils/chunking.py` to add better AST-boundary detection for a new language.

### 4. Dashboard Features
The dashboard is in `ui/index.html` — it's a single HTML file with inline CSS/JS. No build step.

### 5. Tests
Add integration tests against real (small) open-source repos with known issues.

## Development Setup

```bash
git clone https://github.com/rhodawk-ai-code-stabilizer/rhodawk-ai-code-stabilizer
cd rhodawk-ai-code-stabilizer
pip install pdm
pdm install --dev
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# Run tests
make test

# Run linting
make lint
```

## Pull Request Rules

1. All tests must pass
2. New features need tests
3. New plugins need at least 3 test cases
4. No secrets in code (the `no_secrets` plugin will catch you)
5. Type annotations required on all public functions
6. Docstrings required on all public classes and functions

## Code of Conduct

Be excellent to each other. That's it.
