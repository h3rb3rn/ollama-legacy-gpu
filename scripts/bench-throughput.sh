#!/usr/bin/env bash
# bench-throughput.sh — decode + prefill throughput together with the settings that were
# EFFECTIVE in llama-server, read from the container log (not from the env file).
#
# Why: OLLAMA_MAX_BATCH_SIZE / OLLAMA_NUM_BATCH / LLAMA_ARG_BATCH only take effect in this
# fork (patch-ollama-batch.py). Stock Ollama ignores them and derives -b/-ub from VRAM and
# context, so "Batch 64" in an env file says nothing about a stock run. Every result line
# therefore carries n_batch / n_ubatch as logged by the runner that actually served it.
#
# usage: bench-throughput.sh <ssh-host> <container> <model> [port=11434] [prefill-words=2500]
# prints one markdown table row (and the header once with --header).
set -euo pipefail

if [[ "${1:-}" == "--header" ]]; then
  echo "| Host | Ollama | Image | n_ctx | n_batch | n_ubatch | FA | KV | Layer | Decode tok/s | Prefill tok/s |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|"
  exit 0
fi

HOST=$1; CONTAINER=$2; MODEL=$3; PORT=${4:-11434}; WORDS=${5:-2500}
SSH=(ssh -i "${SSH_KEY:-$HOME/.ssh/claude}" "philipp@$HOST")

gen() {  # $1 = JSON body
  "${SSH[@]}" "curl -s -m 900 localhost:$PORT/api/generate -d @-" <<<"$1"
}
rate() {  # stdin JSON -> "<count>" "<tok/s>" for the given prefix (eval|prompt_eval)
  python3 -c "
import json,sys
d=json.load(sys.stdin); p=sys.argv[1]
if 'error' in d: print('ERR', d['error'][:60]); sys.exit()
print(round(d[p+'_count']/d[p+'_duration']*1e9,1))" "$1"
}

# Warm-up (also loads the model so the log below describes this load), then 2 timed runs.
gen "{\"model\":\"$MODEL\",\"prompt\":\"warm\",\"stream\":false,\"options\":{\"num_predict\":8}}" >/dev/null
dec=()
for i in 1 2; do
  dec+=("$(gen "{\"model\":\"$MODEL\",\"prompt\":\"Write a long story about a robot.\",\"stream\":false,\"options\":{\"num_predict\":120,\"temperature\":0,\"seed\":1}}" | rate eval)")
done
pre=()
for i in 1 2; do
  body=$(python3 - "$WORDS" "$RANDOM$i" "$MODEL" <<'PY'
import json,sys,random
w=int(sys.argv[1]); random.seed(sys.argv[2])
words=["alpha","beta","gamma","delta","robot","server","kernel","memory","stream","token","cache","layer"]
txt=" ".join(random.choice(words) for _ in range(w))
print(json.dumps({"model":sys.argv[3],"prompt":sys.argv[2]+" "+txt+" Summarize in one word.","stream":False,
                  "options":{"num_predict":1,"temperature":0}}))
PY
)
  pre+=("$(gen "$body" | rate prompt_eval)")
done

# Effective settings of the most recent model load, from the runner's own log.
LOG=$("${SSH[@]}" "docker logs $CONTAINER 2>&1 | grep -E 'llama_context: (n_ctx|n_batch|n_ubatch|flash_attn)|llama_kv_cache: size|offloaded [0-9]+/[0-9]+ layers'")
pick() { { grep -oE "$1" <<<"$LOG" || true; } | tail -1 | grep -oE "[^= ]+$" || echo "?"; }
NCTX=$(pick 'llama_context: n_ctx +=  *[0-9]+')
NB=$(pick 'llama_context: n_batch +=  *[0-9]+')
NUB=$(pick 'llama_context: n_ubatch +=  *[0-9]+')
FA=$(pick 'llama_context: flash_attn += +[a-z]+')
# "llama_kv_cache: size = ... K (q4_0): ... V (q4_0): ..."
KV=$( { grep -E "llama_kv_cache: size" <<<"$LOG" || true; } | tail -1 | grep -oE "[KV] \([a-z0-9_]+\)" | tr -d '() ' | paste -sd/ - || true)
LAYERS=$( { grep -oE "offloaded [0-9]+/[0-9]+ layers" <<<"$LOG" || true; } | tail -1 | sed 's/offloaded //; s/ layers//' || true)
VER=$("${SSH[@]}" "curl -s localhost:$PORT/api/version" | python3 -c "import json,sys;print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "?")
IMG=$("${SSH[@]}" "docker inspect $CONTAINER --format '{{.Config.Image}}'" | cut -c1-40)

printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
  "$HOST" "$VER" "$IMG" "$NCTX" "$NB" "$NUB" "$FA" "${KV:-?}" "${LAYERS:-?}" \
  "${dec[0]} / ${dec[1]}" "${pre[0]} / ${pre[1]}"
