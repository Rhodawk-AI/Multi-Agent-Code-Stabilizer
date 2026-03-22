FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential patch \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for mcp_server (optional)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null || true

# Python deps
COPY pyproject.toml .
# BUG-7 FIX: python-jose[cryptography] is now explicit in the fallback pip list.
# Previously it was absent, so if the primary `pip install -e ".[dev]"` failed
# (missing extras, network timeout, incompatible Python version), jose would not
# be installed. auth/jwt_middleware.py would then silently accept every request
# as anonymous with wildcard scopes — full auth bypass with no visible error.
#
# The fallback list is the safety net. Every security-critical dependency must
# be in it.
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir \
    "python-jose[cryptography]>=3.3.0" \
    litellm \
    pydantic \
    fastapi \
    uvicorn \
    aiosqlite \
    asyncpg \
    typer \
    rich \
    tenacity \
    python-dotenv \
    networkx \
    chromadb \
    sentence-transformers \
    z3-solver \
    PyGithub \
    httpx \
    aiohttp \
    docker \
    qdrant-client

# App
COPY . .

# BUG-2 FIX: Reject the image at build time if RHODAWK_DEV_AUTH has somehow
# been baked into the build context. This is defence-in-depth — the primary
# fix is removing RHODAWK_DEV_AUTH=1 from .env entirely and enforcing
# SystemExit at runtime in api/app.py. This build-time check catches the case
# where someone has re-added it to a Dockerfile ENV or ARG line.
RUN if [ "${RHODAWK_DEV_AUTH}" = "1" ]; then \
        echo "FATAL: RHODAWK_DEV_AUTH=1 must not be set in the production image." >&2; \
        exit 1; \
    fi

# SEC-3 FIX: Run as a non-root user. Previously the container ran as root
# (Dockerfile had no USER directive), which meant any LLM-generated test code
# executed via TestRunnerAgent had full host filesystem and network access.
# Running as nobody limits the blast radius of any sandbox escape.
RUN addgroup --system rhodawk && adduser --system --ingroup rhodawk rhodawk
RUN chown -R rhodawk:rhodawk /app
USER rhodawk

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
