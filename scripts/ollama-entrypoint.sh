#!/usr/bin/env bash
# ollama-entrypoint.sh — Container entrypoint with auto GPU detection and optimization.
#
# Flow:
#   1. gpu-detect.sh       → detects GPU capabilities, writes /tmp/ollama-gpu-config.env
#   2. Source config       → applies CUDA_VISIBLE_DEVICES, FAST_POOL settings
#   3. Start Ollama        → runs in background (so we can run optimizer after)
#   4. Wait for API ready  → polls /api/tags until Ollama is up
#   5. Auto-Optimizer      → if OLLAMA_PRIMARY_MODEL is set AND cache is stale:
#                             runs auto-optimize.py in background
#                             finds optimal OVERHEAD_SCALE + spec decoding settings
#                             caches result in /root/.ollama/auto-optimize/<sha>.json
#   6. Wait               → keeps container alive (wait $ollama_pid)
#
# Environment variables:
#   OLLAMA_GPU_AUTODETECT    1 = enable auto-detection (default), 0 = disable
#   OLLAMA_PRIMARY_MODEL     model name to pre-optimize at startup (e.g. "qwen3.6:35b")
#                            if not set, optimization runs on first model request trigger
#   OLLAMA_AUTO_OPTIMIZE     1 = enable auto-optimizer (default if PRIMARY_MODEL set)

set -euo pipefail

AUTODETECT="${OLLAMA_GPU_AUTODETECT:-1}"
AUTO_OPTIMIZE="${OLLAMA_AUTO_OPTIMIZE:-1}"
PRIMARY_MODEL="${OLLAMA_PRIMARY_MODEL:-}"
DETECT_SCRIPT="/usr/local/bin/gpu-detect.sh"
OPTIMIZE_SCRIPT="/usr/local/bin/auto-optimize.py"
CONFIG_FILE="/tmp/ollama-gpu-config.env"
SCALE_OVERRIDE="/tmp/ollama-scale-override"
OLLAMA_API="http://localhost:11434"

# ── Step 1: GPU auto-detection ────────────────────────────────────────────────
if [[ "$AUTODETECT" == "1" ]] && [[ -x "$DETECT_SCRIPT" ]]; then
    echo "[entrypoint] Running GPU auto-detection..."
    if "$DETECT_SCRIPT" "$CONFIG_FILE" 2>&1; then
        echo "[entrypoint] GPU config written to $CONFIG_FILE"
        # Apply GPU structural settings (always applied)
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" =~ ^# ]] && continue
            case "$key" in
                CUDA_VISIBLE_DEVICES|OLLAMA_GPU_TIER_THRESHOLD|OLLAMA_GPU_OVERHEAD|\
                OLLAMA_FAST_GPU_DEVICES|OLLAMA_FAST_POOL_VRAM_GB)
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
        echo "[entrypoint] GPU detection failed, using environment defaults"
    fi
else
    echo "[entrypoint] Auto-detection disabled"
fi

# Apply cached optimizer scale override if present
if [[ -f "$SCALE_OVERRIDE" ]]; then
    CACHED_SCALE=$(cat "$SCALE_OVERRIDE")
    echo "[entrypoint] Applying cached scale override: OLLAMA_LAYER_OVERHEAD_SCALE=$CACHED_SCALE"
    export OLLAMA_LAYER_OVERHEAD_SCALE="$CACHED_SCALE"
fi

echo "[entrypoint] Effective GPU config:"
echo "  CUDA_VISIBLE_DEVICES     = ${CUDA_VISIBLE_DEVICES:0:60}..."
echo "  OLLAMA_GPU_TIER_THRESHOLD = ${OLLAMA_GPU_TIER_THRESHOLD:-0}"
echo "  OLLAMA_FLASH_ATTENTION   = ${OLLAMA_FLASH_ATTENTION:-<not set>}"
echo "  OLLAMA_FAST_POOL_VRAM_GB = ${OLLAMA_FAST_POOL_VRAM_GB:-<not set>}"
echo "  OLLAMA_LAYER_OVERHEAD_SCALE = ${OLLAMA_LAYER_OVERHEAD_SCALE:-1.4}"

# ── Step 2: Start Ollama in background ───────────────────────────────────────
echo "[entrypoint] Starting Ollama server..."
/usr/bin/ollama "$@" &
OLLAMA_PID=$!

# ── Step 3: Wait for API ready ────────────────────────────────────────────────
echo "[entrypoint] Waiting for Ollama API..."
WAIT_SECS=0
MAX_WAIT=120
while [[ $WAIT_SECS -lt $MAX_WAIT ]]; do
    if curl -sf "$OLLAMA_API/api/tags" > /dev/null 2>&1; then
        echo "[entrypoint] Ollama API ready after ${WAIT_SECS}s"
        break
    fi
    sleep 2
    WAIT_SECS=$((WAIT_SECS + 2))
done

if [[ $WAIT_SECS -ge $MAX_WAIT ]]; then
    echo "[entrypoint] Ollama API not ready after ${MAX_WAIT}s, continuing anyway"
fi

# ── Step 4: Auto-Optimizer ────────────────────────────────────────────────────
if [[ "$AUTO_OPTIMIZE" == "1" ]] && [[ -x "$OPTIMIZE_SCRIPT" ]] && [[ -n "$PRIMARY_MODEL" ]]; then
    echo "[entrypoint] Launching auto-optimizer for model: $PRIMARY_MODEL"
    (
        # Give Ollama a moment to settle
        sleep 5

        # Check if model is available
        if ! curl -sf "$OLLAMA_API/api/tags" | python3 -c "
import json,sys
tags = json.load(sys.stdin)
names = [m['name'] for m in tags.get('models', [])]
primary = '$PRIMARY_MODEL'
# Check both with and without :latest suffix
if primary in names or primary + ':latest' in names or any(n.startswith(primary) for n in names):
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            echo "[auto-optimizer] Model '$PRIMARY_MODEL' not found locally, skipping optimization"
            exit 0
        fi

        echo "[auto-optimizer] Starting optimization for $PRIMARY_MODEL..."
        python3 "$OPTIMIZE_SCRIPT" "$PRIMARY_MODEL" "$OLLAMA_API"
        echo "[auto-optimizer] Optimization complete"
    ) &
    AUTO_OPT_PID=$!
    echo "[entrypoint] Auto-optimizer running in background (PID=$AUTO_OPT_PID)"
else
    if [[ -z "$PRIMARY_MODEL" ]]; then
        echo "[entrypoint] Auto-optimizer: set OLLAMA_PRIMARY_MODEL to enable optimization"
    fi
fi

# ── Step 5: Keep container alive ─────────────────────────────────────────────
echo "[entrypoint] Ollama running (PID=$OLLAMA_PID)"
wait $OLLAMA_PID
EXIT_CODE=$?
echo "[entrypoint] Ollama exited with code $EXIT_CODE"
exit $EXIT_CODE
