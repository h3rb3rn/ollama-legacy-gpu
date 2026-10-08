#!/bin/bash
# usage: swap401.sh <container> <project> <composefile> <benchspec>   (rolls an N04 instance to the v0.40.1 image; old container kept stopped)
set -u
C=$1; P=$2; F=$3; B=$4; TS=$(date +%Y%m%d-%H%M%S)
cd /opt/deployment/ollama/llm-studio/worker-rtx
echo "[$(date +%T)] stop+rename $C -> $C-pre-gh0401-$TS"
docker update --restart=no $C >/dev/null; docker stop -t 30 $C >/dev/null; docker rename $C $C-pre-gh0401-$TS
docker compose -p $P -f $F up -d 2>&1 | grep -iE "error" 
for i in $(seq 1 60); do s=$(docker inspect $C --format '{{.State.Health.Status}}' 2>/dev/null); [ "$s" = healthy ] && break; sleep 5; done
echo "[$(date +%T)] health=$s version=$(curl -s localhost:$(docker port $C 11434/tcp | head -1 | sed 's/.*://')/api/version)"
OUTFILE=/home/philipp/n04-v0401.jsonl python3 /tmp/n04-bench.py "$B" | cut -c1-520
echo "old container kept as $C-pre-gh0401-$TS"
