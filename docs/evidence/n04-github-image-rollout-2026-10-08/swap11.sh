#!/bin/bash
set -u
C=ollama; P=n11-gh-m10; F=docker-compose.m10-pool.yml; TS=$(date +%Y%m%d-%H%M%S)
cd /opt/deployment/ollama/llm-studio/worker-tesla
echo "[$(date +%T)] stop+rename $C -> $C-pre-gh040-$TS"
docker update --restart=no $C >/dev/null; docker stop -t 30 $C >/dev/null; docker rename $C $C-pre-gh040-$TS
echo "[$(date +%T)] compose up"
docker compose -p $P -f $F up -d 2>&1 | tail -2
for i in $(seq 1 60); do s=$(docker inspect $C --format '{{.State.Health.Status}}' 2>/dev/null); [ "$s" = healthy ] && break; sleep 5; done
echo "[$(date +%T)] health=$s"
OUTFILE=/home/philipp/n11-rollout.jsonl python3 /tmp/n04-bench.py "11434|ollama|qwen3.6:35b|AFTER-n11-gh-0.40.0" | cut -c1-520
echo "old container kept as $C-pre-gh040-$TS"
