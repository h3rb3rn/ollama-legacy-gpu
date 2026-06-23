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
PROXY_SCRIPT="/usr/local/bin/ollama-proxy.py"
CONFIG_FILE="/tmp/ollama-gpu-config.env"
SCALE_OVERRIDE="/tmp/ollama-scale-override"
# Ollama serves internally on 11435; proxy on 11434 intercepts all requests
OLLAMA_INTERNAL_PORT="${OLLAMA_INTERNAL_PORT:-11435}"
OLLAMA_PROXY_PORT="${OLLAMA_PROXY_PORT:-11434}"
OLLAMA_API="http://localhost:${OLLAMA_INTERNAL_PORT}"
OLLAMA_BACKEND_URL="http://localhost:${OLLAMA_INTERNAL_PORT}"

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
                OLLAMA_FAST_GPU_DEVICES|OLLAMA_FAST_POOL_VRAM_GB|OLLAMA_CUDA_REVERSED|\
                OLLAMA_GPU_BANDWIDTHS|OLLAMA_GPU_MAX_BANDWIDTH|\
                OLLAMA_NONLEGACY_REVERSED|OLLAMA_NONLEGACY_VRAM_GB)
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

# ── Step 2: Start Ollama on internal port ────────────────────────────────────
echo "[entrypoint] Starting Ollama server on internal port ${OLLAMA_INTERNAL_PORT}..."
export OLLAMA_HOST="0.0.0.0:${OLLAMA_INTERNAL_PORT}"
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

# ── Step 4: Start Auto-Optimize Proxy ────────────────────────────────────────
if [[ "$AUTO_OPTIMIZE" == "1" ]] && [[ -x "$PROXY_SCRIPT" ]]; then
    echo "[entrypoint] Starting auto-optimize proxy on port ${OLLAMA_PROXY_PORT}..."
    export OLLAMA_BACKEND_URL OLLAMA_PROXY_PORT
    python3 "$PROXY_SCRIPT" \
        --backend "$OLLAMA_BACKEND_URL" \
        --port "$OLLAMA_PROXY_PORT" &
    PROXY_PID=$!
    echo "[entrypoint] Proxy running (PID=$PROXY_PID) on :${OLLAMA_PROXY_PORT} → ${OLLAMA_BACKEND_URL}"

    # Pre-optimize PRIMARY_MODEL if set (speeds up first real request)
    if [[ -n "$PRIMARY_MODEL" ]] && [[ -x "$OPTIMIZE_SCRIPT" ]]; then
        echo "[entrypoint] Pre-optimizing primary model: $PRIMARY_MODEL"
        (
            sleep 10  # let proxy settle
            # Check model exists
            if curl -sf "$OLLAMA_API/api/tags" 2>/dev/null | python3 -c "
import json,sys
t=json.load(sys.stdin); n=[m['name'] for m in t.get('models',[])]
sys.exit(0 if '$PRIMARY_MODEL' in n or '$PRIMARY_MODEL:latest' in n or any(x.startswith('$PRIMARY_MODEL') for x in n) else 1)
" 2>/dev/null; then
                echo "[pre-optimize] Starting: $PRIMARY_MODEL"
                python3 "$OPTIMIZE_SCRIPT" "$PRIMARY_MODEL" "$OLLAMA_API"
                echo "[pre-optimize] Complete"
            fi
        ) &
    fi
else
    echo "[entrypoint] Auto-optimize proxy disabled (OLLAMA_AUTO_OPTIMIZE=0)"
    echo "[entrypoint] WARNING: requests will go to internal port ${OLLAMA_INTERNAL_PORT} directly"
    echo "[entrypoint]          Set a port redirect or use OLLAMA_HOST in clients"
fi

# ── Step 5: Keep container alive ─────────────────────────────────────────────
echo "[entrypoint] Ollama running (PID=$OLLAMA_PID)"
wait $OLLAMA_PID
EXIT_CODE=$?
echo "[entrypoint] Ollama exited with code $EXIT_CODE"
exit $EXIT_CODE
