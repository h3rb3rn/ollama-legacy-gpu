#!/bin/bash
# last three N02-M60 endpoints to the v0.40.1 image, NO preload (the agents send num_batch 64 and their own options)
cd /opt/deployment/ollama/llm-studio/worker-m60
TS=$(date +%Y%m%d-%H%M%S)
declare -A PORT=( [ollama-m60-pool]=11434 [ollama-m60-gpu8]=11439 [ollama-m60-gpu9]=11440 )
for svc in ollama-m60-gpu8 ollama-m60-gpu9 ollama-m60-pool; do
  echo "[$(date +%T)] $svc: stop+rename"
  docker update --restart=no $svc >/dev/null; docker stop -t 30 $svc >/dev/null; docker rename $svc $svc-pre-gh0401-$TS
  docker compose -p n02-gh401 -f docker-compose.yml up -d $svc 2>&1 | grep -iE "error"
  for i in $(seq 1 60); do s=$(docker inspect $svc --format '{{.State.Health.Status}}' 2>/dev/null); [ "$s" = healthy ] && break; sleep 3; done
  echo "[$(date +%T)] $svc health=$s version=$(curl -s localhost:${PORT[$svc]}/api/version) (no preload)"
done
echo ROLL-DONE
