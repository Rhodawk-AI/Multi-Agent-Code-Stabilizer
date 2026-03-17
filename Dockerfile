FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install --no-cache-dir pdm

# Copy project files
COPY pyproject.toml .
COPY . .

# Install Python deps
RUN pdm install --no-editable

# Install MCP servers
RUN npm install -g @modelcontextprotocol/server-filesystem \
                   @modelcontextprotocol/server-github

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["pdm", "run", "rhodawk-ai-code-stabilizer"]
CMD ["--help"]
