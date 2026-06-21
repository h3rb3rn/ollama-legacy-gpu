#!/usr/bin/env bash
# ollama-entrypoint.sh — Container entrypoint that auto-detects GPU configuration
# before starting Ollama.
#
# Execution order:
#   1. Run gpu-detect.sh → /tmp/ollama-gpu-config.env
#   2. Source the generated env (overrides env_file values if set)
#   3. Log the effective configuration
#   4. Exec the original Ollama binary
#
# Environment variable precedence (highest to lowest):
#   1. Docker env / env_file (OLLAMA_HOST, OLLAMA_KEEP_ALIVE, etc.)   — stays
#   2. gpu-detect.sh output (CUDA_VISIBLE_DEVICES, TIER_THRESHOLD, FA) — overrides
#   3. Ollama built-in defaults                                         — fallback
#
# To disable auto-detection and use manual config:
#   Set OLLAMA_GPU_AUTODETECT=0 in your .env file.

set -euo pipefail

AUTODETECT="${OLLAMA_GPU_AUTODETECT:-1}"
DETECT_SCRIPT="/usr/local/bin/gpu-detect.sh"
CONFIG_FILE="/tmp/ollama-gpu-config.env"

if [[ "$AUTODETECT" == "1" ]] && [[ -x "$DETECT_SCRIPT" ]]; then
    echo "[gpu-detect] Running GPU auto-detection..."
    if "$DETECT_SCRIPT" "$CONFIG_FILE" 2>&1; then
        echo "[gpu-detect] Configuration written to $CONFIG_FILE"
        # Only source variables NOT already set in environment.
        # This allows manual overrides via docker-compose env_file to take precedence
        # for Ollama-specific settings, while we always apply GPU ordering vars.
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" =~ ^# ]] && continue
            case "$key" in
                CUDA_VISIBLE_DEVICES|OLLAMA_GPU_TIER_THRESHOLD|OLLAMA_GPU_OVERHEAD)
                    # Always apply GPU structural settings
                    export "$key"="$value"
                    ;;
                OLLAMA_FLASH_ATTENTION|OLLAMA_KV_CACHE_TYPE)
                    # Apply only if not manually set
                    if [[ -z "${!key:-}" ]]; then
                        export "$key"="$value"
                    fi
                    ;;
            esac
        done < "$CONFIG_FILE"
    else
        echo "[gpu-detect] Detection failed, using existing environment"
    fi
else
    echo "[gpu-detect] Auto-detection disabled (OLLAMA_GPU_AUTODETECT=0)"
fi

echo "[gpu-detect] Effective GPU config:"
echo "  CUDA_VISIBLE_DEVICES    = ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "  OLLAMA_GPU_TIER_THRESHOLD = ${OLLAMA_GPU_TIER_THRESHOLD:-0}"
echo "  OLLAMA_FLASH_ATTENTION  = ${OLLAMA_FLASH_ATTENTION:-<not set>}"
echo "  OLLAMA_KV_CACHE_TYPE    = ${OLLAMA_KV_CACHE_TYPE:-<not set>}"

exec /usr/bin/ollama "$@"
