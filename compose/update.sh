#!/usr/bin/env bash
# update.sh — Pull latest image and restart the ollama container on N04-RTX.
# Run this from /opt/deployment/ollama/fork/compose/ on N04-RTX.
set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/docker-compose.worker-rtx.yml"
IMAGE="ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest"

echo "[update] Pulling $IMAGE..."
docker pull "$IMAGE"

echo "[update] Restarting stack..."
docker compose -f "$COMPOSE_FILE" up -d

echo "[update] Waiting for API..."
until docker exec ollama curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 3
done
echo "[update] Done. Container status:"
docker compose -f "$COMPOSE_FILE" ps
echo ""
docker exec ollama ollama ps 2>/dev/null || true
