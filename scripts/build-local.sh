#!/usr/bin/env bash
# build-local.sh — Lokaler Build ohne GitHub Actions
# Verwendung: ./scripts/build-local.sh [cuda12|cuda11|all] [OLLAMA_VERSION]
#
# Beispiele:
#   ./scripts/build-local.sh cuda12 v0.9.0
#   ./scripts/build-local.sh all latest
#   ./scripts/build-local.sh cuda11

set -euo pipefail

TARGET="${1:-all}"
VERSION="${2:-latest}"
JOBS="${JOBS:-$(nproc)}"
OWNER="${GITHUB_OWNER:-$(git config user.name | tr '[:upper:]' '[:lower:]' | tr ' ' '-')}"
REGISTRY="ghcr.io/${OWNER}/ollama-legacy"

if [ "$VERSION" = "latest" ]; then
  VERSION=$(curl -sf https://api.github.com/repos/ollama/ollama/releases/latest | jq -r '.tag_name')
  echo "→ Aktuelle Ollama-Version: $VERSION"
fi

SHORT="${VERSION#v}"

build_cuda12() {
  echo "=== Baue cuda12-maxwell ($VERSION) ==="
  docker build \
    -f dockerfiles/Dockerfile.cuda12-maxwell \
    --build-arg OLLAMA_VERSION="$VERSION" \
    --build-arg JOBS="$JOBS" \
    -t "${REGISTRY}:cuda12-maxwell-${SHORT}" \
    -t "${REGISTRY}:cuda12-maxwell-latest" \
    .
  echo "✓ cuda12-maxwell-${SHORT} fertig"
}

build_cuda11() {
  echo "=== Baue cuda11-legacy ($VERSION) ==="
  docker build \
    -f dockerfiles/Dockerfile.cuda11-legacy \
    --build-arg OLLAMA_VERSION="$VERSION" \
    --build-arg JOBS="$JOBS" \
    -t "${REGISTRY}:cuda11-legacy-${SHORT}" \
    -t "${REGISTRY}:cuda11-legacy-latest" \
    .
  echo "✓ cuda11-legacy-${SHORT} fertig"
}

case "$TARGET" in
  cuda12) build_cuda12 ;;
  cuda11) build_cuda11 ;;
  all)    build_cuda12; build_cuda11 ;;
  *)      echo "Unbekanntes Target: $TARGET (cuda12|cuda11|all)"; exit 1 ;;
esac

echo ""
echo "Images:"
docker images "${REGISTRY}" --format "  {{.Repository}}:{{.Tag}}"
