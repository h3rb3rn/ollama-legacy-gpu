#!/usr/bin/env bash
# Update only the explicit target with its successful hardware-test evidence.
# Usage: ./update.sh --config /path/target.json --evidence /path/result.json \
#          --output /path/deployment-result.json
set -euo pipefail

exec python3 "$(dirname "$0")/../scripts/deploy-tested-release.py" "$@"
