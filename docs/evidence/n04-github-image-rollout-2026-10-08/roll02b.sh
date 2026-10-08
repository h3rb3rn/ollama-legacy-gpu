#!/bin/bash
# rolling swap of the nine N02-M60 endpoints to the v0.40.1 image; singles are preloaded with the agent's num_ctx, the pool is NOT preloaded
cd /opt/deployment/ollama/llm-studio/worker-m60
TS=$(date +%Y%m%d-%H%M%S)
declare -A PORT=( [ollama-m60-pool]=11434 [ollama-m60-gpu4]=11435 [ollama-m60-gpu5]=11436 [ollama-m60-gpu6]=11437 [ollama-m60-gpu7]=11438 [ollama-m60-gpu8]=11439 [ollama-m60-gpu9]=11440 [ollama-m60-gpu10]=11441 [ollama-m60-gpu11]=11442 )
declare -A MODEL=( [ollama-m60-gpu4]="qwen3.5:4b|262144" [ollama-m60-gpu5]="granite4.2:3b|131072" [ollama-m60-gpu6]="granite4.2:3b|131072" [ollama-m60-gpu7]="gemma3:4b|131072" [ollama-m60-gpu8]="nemotron-3-nano:4b|262144" [ollama-m60-gpu9]="huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF:latest|262144" [ollama-m60-gpu10]="hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M|262144" [ollama-m60-gpu11]="hf.co/webAI-Official/TwIL-LM3-Pro:Q4_K_M|131072" )
for svc in ollama-m60-gpu11 ollama-m60-gpu7 ollama-m60-gpu6 ollama-m60-gpu10 ollama-m60-gpu5 ollama-m60-gpu4 ollama-m60-gpu8 ollama-m60-gpu9 ollama-m60-pool; do
  echo "[$(date +%T)] $svc: stop+rename"
  docker update --restart=no $svc >/dev/null; docker stop -t 30 $svc >/dev/null; docker rename $svc $svc-pre-gh0401-$TS
  docker compose -p n02-gh401 -f docker-compose.yml up -d $svc 2>&1 | grep -iE "error"
  for i in $(seq 1 60); do s=$(docker inspect $svc --format '{{.State.Health.Status}}' 2>/dev/null); [ "$s" = healthy ] && break; sleep 3; done
  if [ -n "${MODEL[$svc]:-}" ]; then
    m=${MODEL[$svc]%%|*}; c=${MODEL[$svc]##*|}; t0=$(date +%s)
    r=$(curl -s -m 600 localhost:${PORT[$svc]}/api/generate -d "{\"model\":\"$m\",\"keep_alive\":\"24h\",\"options\":{\"num_ctx\":$c}}" | cut -c1-60)
    echo "[$(date +%T)] $svc health=$s version=$(curl -s localhost:${PORT[$svc]}/api/version) preload ${m##*/} ctx=$c in $(( $(date +%s)-t0 ))s -> $r"
  else
    echo "[$(date +%T)] $svc health=$s version=$(curl -s localhost:${PORT[$svc]}/api/version) (no preload, the agent loads the model itself)"
  fi
done
echo ROLL-DONE
