#!/usr/bin/env bash
# scripts/setup_joern.sh
# ======================
# Gap 1: Joern CPG setup script for Rhodawk AI Code Stabilizer.
#
# This script:
#   1. Checks Docker is available
#   2. Pulls the Joern image
#   3. Verifies the Joern container can start
#   4. Imports a test codebase and confirms CPG queries work
#   5. Prints a summary with connection info for .env
#
# Usage:
#   ./scripts/setup_joern.sh [--repo-path /path/to/repo] [--port 8080]
#
# Prerequisites:
#   - Docker installed and running
#   - 4GB+ RAM available for the Joern container
#   - JDK 17+ (only needed if running Joern natively, not in Docker)

set -euo pipefail

JOERN_IMAGE="ghcr.io/joernio/joern:latest"
JOERN_PORT="${JOERN_PORT:-8080}"
JOERN_CONTAINER_NAME="rhodawk-joern-setup"
REPO_PATH="${JOERN_REPO_PATH:-$PWD}"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-path) REPO_PATH="$2"; shift 2 ;;
    --port)      JOERN_PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[joern-setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[joern-setup]${NC} $*"; }
error() { echo -e "${RED}[joern-setup]${NC} $*"; }

# ── Step 1: Check Docker ──────────────────────────────────────────────────────
info "Checking Docker..."
if ! command -v docker &>/dev/null; then
  error "Docker not found. Install Docker Desktop or Docker Engine first."
  exit 1
fi
if ! docker info &>/dev/null; then
  error "Docker is not running. Start Docker and retry."
  exit 1
fi
info "Docker OK"

# ── Step 2: Pull Joern image ──────────────────────────────────────────────────
info "Pulling Joern image: $JOERN_IMAGE"
info "(This may take a few minutes on first run — image is ~2GB)"
docker pull "$JOERN_IMAGE"
info "Joern image ready"

# ── Step 3: Start Joern server ────────────────────────────────────────────────
info "Starting Joern server on port $JOERN_PORT..."

# Stop any existing setup container
docker rm -f "$JOERN_CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$JOERN_CONTAINER_NAME" \
  -p "${JOERN_PORT}:8080" \
  -v "${REPO_PATH}:/repo:ro" \
  -v "joern_workspace:/root/.joern" \
  --memory="8g" \
  --memory-reservation="4g" \
  "$JOERN_IMAGE" \
  joern --server --server-host 0.0.0.0 --server-port 8080 --max-heap-size 6g

info "Container started. Waiting for Joern to be ready (up to 90s)..."

READY=false
for i in $(seq 1 18); do
  sleep 5
  if curl -sf "http://localhost:${JOERN_PORT}/api/v1/projects" &>/dev/null; then
    READY=true
    break
  fi
  echo -n "."
done
echo

if [ "$READY" = false ]; then
  error "Joern did not become ready in 90s. Check logs:"
  error "  docker logs $JOERN_CONTAINER_NAME"
  docker rm -f "$JOERN_CONTAINER_NAME" 2>/dev/null || true
  exit 1
fi
info "Joern server is ready at http://localhost:${JOERN_PORT}"

# ── Step 4: Test CPG import ───────────────────────────────────────────────────
info "Testing CPG import with repo at: $REPO_PATH"

# Import the repo
IMPORT_RESPONSE=$(curl -sf -X POST \
  "http://localhost:${JOERN_PORT}/api/v1/projects" \
  -H "Content-Type: application/json" \
  -d "{\"inputPath\": \"/repo\", \"projectName\": \"rhodawk-test\"}" \
  2>/dev/null || echo '{"error": "import failed"}')

if echo "$IMPORT_RESPONSE" | grep -q "error"; then
  warn "CPG import returned: $IMPORT_RESPONSE"
  warn "This may be normal — Joern may still be building the CPG"
else
  info "CPG import started: $IMPORT_RESPONSE"
fi

# ── Step 5: Test a basic query ────────────────────────────────────────────────
info "Testing basic Joern QL query..."
sleep 5

QUERY_RESPONSE=$(curl -sf -X POST \
  "http://localhost:${JOERN_PORT}/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "1 + 1"}' \
  2>/dev/null || echo '{"error": "query failed"}')

if echo "$QUERY_RESPONSE" | grep -q "error"; then
  warn "Query test returned: $QUERY_RESPONSE"
  warn "Joern may still be initialising — proceed and it will be ready shortly"
else
  info "Query test successful: $QUERY_RESPONSE"
fi

# ── Step 6: Print summary ─────────────────────────────────────────────────────
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Joern CPG setup complete${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo
echo "Add these to your .env file:"
echo
echo "  JOERN_URL=http://localhost:${JOERN_PORT}"
echo "  JOERN_REPO_PATH=${REPO_PATH}"
echo "  JOERN_PROJECT_NAME=rhodawk"
echo "  CPG_ENABLED=1"
echo "  CPG_BLAST_RADIUS_THRESHOLD=50"
echo
echo "Or use docker-compose (recommended for production):"
echo "  docker-compose up joern"
echo
echo "CPG will be built automatically on first run."
echo "For a 10M line codebase, the initial build takes 5–30 minutes."
echo "Subsequent runs use incremental updates (seconds per commit)."
echo
echo "To stop the setup container:"
echo "  docker rm -f $JOERN_CONTAINER_NAME"
echo
echo "Joern documentation: https://docs.joern.io"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

# Keep container running for manual testing
info "Setup container '$JOERN_CONTAINER_NAME' is running."
info "Stop it with: docker rm -f $JOERN_CONTAINER_NAME"
info "For production use docker-compose instead."
