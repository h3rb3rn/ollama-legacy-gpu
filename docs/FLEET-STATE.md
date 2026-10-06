# Fleet-Stand (2026-10-06)

Die `:11434`-Instanzen von N04-RTX, N02-M60 und N11-M10 und der Zustand des Deployment-Repos
(`https://git.4noobs.de/h3rb3rn/ollama.git`). Gelesen an den Hosts per `docker inspect`, `git status` und `git log`.

Vorgaben für die Instanzen: 256k-KV-Cache bei q4_0, alles im VRAM und nichts im RAM, `qwen3.6:35b` primär und warm
(`MAX_LOADED_MODELS=1`, `NUM_PARALLEL=1`, `KEEP_ALIVE` 24 h).

## Live-Stand (Ollama 0.35.1; N04-RTX und N11-M10: `ollama-gaps:pipefix-20261005`, N02-M60: `ollama-gaps:kolibri-20261006`)

| Einstellung | N04-RTX | N02-M60 (12 GPUs) | N11-M10 |
|---|---|---|---|
| Karten im Pool | 3: 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 12× Tesla M60 (8 GiB), Greedy-Fill belegt 8 | 4× Tesla M10 (8 GiB) |
| Frei für anderes | 1× RTX 3060 (`GPU-63bfbd4b-…`, Bus 09:00.0), genutzt von ComfyUI | – | – |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | 262144 | 262144 |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | 64 | 64 |
| `OLLAMA_KV_CACHE_TYPE` | q4_0 | q4_0 | q4_0 |
| `LLAMA_ARG_FIT_TARGET` | 256 | 256 | 256 |
| `LLAMA_ARG_MMPROJ_OFFLOAD` | true | true | true |
| `OLLAMA_DRAFT_NUM_PREDICT` | 0 | 0 | 0 |
| `OLLAMA_GPU_AUTODETECT` | 0 | 1 | 0 |
| `OLLAMA_FORCE_GPU_LAYERS` / `OLLAMA_LAYER_OVERHEAD_SCALE` | – | 1 / 1.10 | – |
| `GGML_CUDA_GRAPHS_LEGACY` | – | – | 1 |
| `LLAMA_PIPELINE_PARALLEL` | nicht gesetzt (Pipeline an) | 0 | 0 |
| Layer / wirksamer Kontext | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode / Prefill (tok/s) | 42.3 / 1140–1172 | 15.0–16.0 / 91–93 | 9.3 / 24.2 |
| nicht im VRAM laut `/api/ps` | 0 MiB | 0 MiB | 0 MiB |
| Host-Puffer `CUDA_Host compute` | 1029 MiB | 32,8 MiB | 32,8 MiB |

Messwerte, Methode und Begründung der Einstellungen: [TUNING.md](TUNING.md).

Unterschiede zwischen den Hosts, die Vergleiche beeinflussen: N02-M60 läuft mit `OLLAMA_GPU_AUTODETECT=1` (aktiviert den
Fit des Forks), N11-M10 mit `GGML_CUDA_GRAPHS_LEGACY=1` (auf N11-M10 +35 % Decode, auf N02-M60 ohne Effekt).

Die Spalten Decode/Prefill, Layer-Zeilen und Host-Puffer für N02-M60 stammen aus dem früheren 4-GPU-Pool (`qwen3.6:35b`,
15,0–16,0 / 91–93 tok/s). Auf der 12-GPU-Instanz läuft `qwen3.6:35b` mit 10,4 tok/s (alle 12 GPUs); `Kolibri-1-Q4_K_M` (78,1B, 51/51 Layer,
`-c 262144`) mit 11,7–11,8 tok/s auf 8 GPUs, Messung in [FORK-VS-STOCK.md](FORK-VS-STOCK.md).

## Modell und MTP-Schutz

Alle drei Hosts führen `qwen3.6:35b` im aktuellen Registry-Stand: Digest `a7eb95c53bcf`, 35,5B Parameter (35 505 251 456),
21,07 GiB, 41 Blöcke, NextN-Kopf `blk.40.nextn.*`, Manifest `draft_num_predict 2`. Das Manifest wird von
`OLLAMA_DRAFT_NUM_PREDICT=0` auf 0 begrenzt (MTP aus, `patch-ollama-mtp-default.py`), ein Anfragewert behält Vorrang. Geprüft je
Host: Modell leer geladen, Runner-Kommandozeile ohne `--spec-*`-Argumente, danach Generierung ohne Fehler.

Zusätzlich liegt auf N02-M60 das Tag `qwen3.6:35b-nospec` (Digest `aae923a8c006`, 35,5B, vom 2026-09-05).

## Wo die Konfiguration liegt

Der ausgerollte Stand aller Hosts liegt im `main`-Branch des Deployment-Repos:

| Host | Dateien | Dienste |
|---|---|---|
| N04-RTX | `llm-studio/worker-rtx/docker-compose.yml` mit `.env.stock-rtx`, `.env.stock-rgtx`, `.env.tesla`; `docker-compose.m60-guard.yml`; `docker-compose.bonsai-m10-prod.yml`; `update.sh` | `ollama` (:11434), `ollama-rgtx` (:11435, Stock), `ollama-m60-guard` (:11442), `ollama-tesla-bonsai` (:11436), ComfyUI |
| N02-M60 | `llm-studio/worker-m60/docker-compose.single12.yml` mit `.env.m60-single12` | `ollama-m60-12` (:11434), eine Instanz über alle 12 GPUs |
| N11-M10 | `llm-studio/worker-tesla/docker-compose.yml` (JSON) | `ollama` (:11434) |
| Steuerhost | `llm-studio/vm-without-gpu/docker-compose.yml` | `open-webui`, `searxng` |

`ollama-m60-guard` und `ollama-tesla-bonsai` laufen ohne Compose-Label; ihre Compose-Dateien beschreiben den laufenden Stand.
Die Compose von N04-RTX bindet `ollama-m60-guard` per `extends` aus `docker-compose.m60-guard.yml` ein.
Die ignorierten `.env`-Dateien (u. a. `worker-tesla/.env`, `worker-legacy-gpu/.env`, `worker-host/.env`) liegen nur auf den Hosts.

## Git-Stand

Die Klone des Deployment-Repos auf N04-RTX, N02-M60, N11-M10 und dem Steuerhost stehen auf `origin/main`, ohne Abweichungen
bei getrackten oder unversionierten Dateien (Steuerhost: unversioniert nur `llm-studio/vm-without-gpu/.env.example` und
`.env.ollama`; `CLAUDE.md`, `.claude/` und `fork/` sind per `.gitignore` ausgenommen).
Die Historie enthält die Stände der drei Hosts als Merge (N02-M60 als Fast-Forward, N04-RTX als Merge, der Live-Stand von N04-RTX
gewinnt) und die Compose von N11-M10. Herkunft der älteren Commits: `8c97f04`, `34f264f` (2026-09-05, N02-Pools) sowie `82e2b66`,
`e3e21e5` (2026-10-03, MTP-Race-Fix, Merge mit origin) aus früheren Claude-Sessions, `badabbd` (2026-10-04) aus der Parallel-Session
`ai-village-f2`, die übrigen (2026-10-05) beschreiben den Live-Stand.

Der Quellcode des Forks liegt in `github.com/h3rb3rn/ollama-legacy-gpu` (Arbeitskopie: `/opt/deployment/ollama/fork/repo`).
Rohdaten der Bonsai-Berichte liegen unter `/opt/deployment/ollama/fork/.bonsai-work/` (`rtx-fa-20260929`, `rollout-20260930`,
`fleet-v035-20261001`) neben den Quell-Klonen `Bonsai-demo` und `mlx-vlm-0.7.2`.

## Offene Entscheidung
`OLLAMA_GPU_AUTODETECT` auf N02-M60: kann auf 0 gestellt werden (die Pipeline ist ausdrücklich aus); danach N02-M60 neu messen.
