#!/usr/bin/env bash
# sweep-batch.sh <host> <batch>...
# Batch ladder for one fleet host: for each env batch value set OLLAMA_MAX_BATCH_SIZE/NUM_BATCH/
# LLAMA_ARG_BATCH/LLAMA_ARG_UBATCH, recreate the :11434 container, run scripts/bench-throughput.sh and
# print one table row with the effective -b, layers, rates, compute/host buffers and free VRAM per GPU.
# Stops at the first batch that no longer gives 42/42 layers at -c 262144.
# NOTE: host paths/compose layouts below are specific to this fleet (N04-RTX, N02-M60, N11-M10) and the
# script RESTARTS the container of that host — only run it on instances released for tests.
set -uo pipefail
HOST=$1; shift
REPO=/opt/deployment/ollama/fork/repo
SSH=(ssh -i "$HOME/.ssh/claude" "philipp@$HOST")

case $HOST in
  N04-RTX) CONT=ollama;          DIR=/opt/deployment/ollama/llm-studio/worker-rtx;   MODE=env;  ENVF=.env.stock-rtx; SVC=ollama ;;
  N02-M60) CONT=ollama-m60-pool; DIR=/opt/deployment/ollama/llm-studio/worker-m60;   MODE=env;  ENVF=.env.m60-pool;  SVC=ollama-m60-pool ;;
  N11-M10) CONT=ollama;          DIR=/opt/deployment/ollama/llm-studio/worker-tesla; MODE=json; SVC=ollama ;;
  *) echo "unknown host $HOST" >&2; exit 2 ;;
esac

set_batch() {  # $1 numeric
  [[ "$1" =~ ^[0-9]+$ ]] || { echo "invalid batch '$1'" >&2; return 1; }
  if [[ $MODE == env ]]; then
    "${SSH[@]}" "cd $DIR && sed -i 's/^OLLAMA_MAX_BATCH_SIZE=.*/OLLAMA_MAX_BATCH_SIZE=$1/; s/^OLLAMA_NUM_BATCH=.*/OLLAMA_NUM_BATCH=$1/; s/^LLAMA_ARG_BATCH=.*/LLAMA_ARG_BATCH=$1/; s/^LLAMA_ARG_UBATCH=.*/LLAMA_ARG_UBATCH=$1/' $ENVF && docker compose up -d --force-recreate --no-deps $SVC 2>&1 | tail -1"
  else
    "${SSH[@]}" "cd $DIR && python3 - <<EOF
import json
c=json.load(open('docker-compose.yml')); e=c['services']['$SVC']['environment']
for k in ('OLLAMA_MAX_BATCH_SIZE','OLLAMA_NUM_BATCH','LLAMA_ARG_BATCH','LLAMA_ARG_UBATCH'): e[k]='$1'
json.dump(c,open('docker-compose.yml','w'),indent=2)
EOF
docker compose up -d --force-recreate --no-deps $SVC 2>&1 | tail -1"
  fi
}

echo "| env batch | n_ctx | n_batch | Layer | Decode | Prefill | compute buf/GPU (MiB) | host buf (MiB) | free VRAM/GPU (MiB) | errors |"
echo "|---|---|---|---|---|---|---|---|---|---|"
for B in "$@"; do
  set_batch "$B" >/dev/null || break
  ROW=$("$REPO/scripts/bench-throughput.sh" "$HOST" "$CONT" qwen3.6:35b 2>&1 | tail -1)
  IFS='|' read -r _ _ _ _ NCTX NB NUB FA KV LAYER DEC PRE _ <<<"$ROW"
  LOGS=$("${SSH[@]}" "docker logs $CONT 2>&1")
  CB=$(grep -E "sched_reserve: +CUDA[0-3] compute buffer size" <<<"$LOGS" | tail -4 | grep -oE "[0-9.]+ MiB" | tr -d ' MiB' | paste -sd/ -)
  HB=$(grep -E "sched_reserve: +CUDA_Host compute buffer size" <<<"$LOGS" | tail -1 | grep -oE "[0-9.]+ MiB" | tr -d ' MiB')
  ERR=$(grep -ciE "cuda error|out of memory|failed to fit|illegal|abort|stoi" <<<"$LOGS" || true)
  FREE=$("${SSH[@]}" "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -4 | awk -F', ' '{printf \"%d/\", \$1-\$2}'")
  echo "| $B |$NCTX|$NB|$LAYER|$DEC|$PRE| ${CB:-?} | ${HB:-?} | ${FREE%/} | $ERR |"
  # stop at the first batch that no longer fits entirely on the GPUs at the full context
  if [[ $(tr -d ' ' <<<"$LAYER") != "42/42" || $(tr -d ' ' <<<"$NCTX") != "262144" ]]; then
    echo "# stopped: batch $B does not fit (layers=$LAYER ctx=$NCTX)"; break
  fi
done
