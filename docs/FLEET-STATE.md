# Fleet-Stand (2026-10-05)

Die `:11434`-Instanzen von N04-RTX, N02-M60 und N11-M10 und der Zustand der Klone des Deployment-Repos
(`https://git.4noobs.de/h3rb3rn/ollama.git`). Gelesen an den Hosts per `docker inspect`, `git status` und `git log`.

Vorgaben für die Instanzen: 256k-KV-Cache bei q4_0, alles im VRAM und nichts im RAM, `qwen3.6:35b` primär und warm
(`MAX_LOADED_MODELS=1`, `NUM_PARALLEL=1`, `KEEP_ALIVE` 24 h).

## Live-Stand (Image `ollama-gaps:pipefix-20261005`, Ollama 0.35.1)

| Einstellung | N04-RTX | N02-M60 (Pool GPU0–3) | N11-M10 |
|---|---|---|---|
| Karten im Pool | 3: 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 4× Tesla M60 (8 GiB) | 4× Tesla M10 (8 GiB) |
| Frei für anderes | 1× RTX 3060 (`GPU-63bfbd4b-…`, Bus 09:00.0), genutzt von ComfyUI | – | – |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | 262144 | 262144 |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | 64 | 64 |
| `OLLAMA_KV_CACHE_TYPE` | q4_0 | q4_0 | q4_0 |
| `LLAMA_ARG_FIT_TARGET` | 256 | 256 | 256 |
| `LLAMA_ARG_MMPROJ_OFFLOAD` | true | true | true |
| `OLLAMA_DRAFT_NUM_PREDICT` | 0 | 0 | 0 |
| `OLLAMA_GPU_AUTODETECT` | 0 | 1 | 0 |
| `GGML_CUDA_GRAPHS_LEGACY` | – | – | 1 |
| `LLAMA_PIPELINE_PARALLEL` | nicht gesetzt (Pipeline an) | 0 | 0 |
| Layer / wirksamer Kontext | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode / Prefill (tok/s) | 42.3 / 1140–1172 | 15.0–16.0 / 91–93 | 9.3 / 24.2 |
| nicht im VRAM laut `/api/ps` | 0 MiB | 0 MiB | 0 MiB |
| Host-Puffer `CUDA_Host compute` | 1029 MiB | 32,8 MiB | 32,8 MiB |

Messwerte, Methode und Begründung der Einstellungen: [TUNING.md](TUNING.md).

Unterschiede zwischen den Hosts, die Vergleiche beeinflussen: N02-M60 läuft mit `OLLAMA_GPU_AUTODETECT=1` (aktiviert den
Fit des Forks), N11-M10 mit `GGML_CUDA_GRAPHS_LEGACY=1` (auf N11-M10 +35 % Decode, auf N02-M60 ohne Effekt).

## Modell und MTP-Schutz

Alle drei Hosts führen `qwen3.6:35b` im aktuellen Registry-Stand: Digest `a7eb95c53bcf`, 35,5B Parameter (35 505 251 456),
21,07 GiB, 41 Blöcke, NextN-Kopf `blk.40.nextn.*`, Manifest `draft_num_predict 2`. Das Manifest wird von
`OLLAMA_DRAFT_NUM_PREDICT=0` auf 0 begrenzt (MTP aus, `patch-ollama-mtp-default.py`), ein Anfragewert behält Vorrang. Geprüft je
Host: Modell leer geladen, Runner-Kommandozeile ohne `--spec-*`-Argumente, danach Generierung ohne Fehler.

Zusätzliche Tags auf den Hosts: N02-M60 `qwen3.6:35b-n02-nextn-20261003` (derselbe Stand mit `draft_num_predict 0`), N11-M10
`qwen3.6:35b-36b-20260610` und N04-RTX `qwen3.6:35b-36b-20260608` (Digest `07d35212591f`, 36,0B, 22,29 GiB, 40 Blöcke,
ohne NextN-Kopf; der ältere Registry-Stand, per `pull` nicht mehr erhältlich).

## Wo die Konfiguration liegt

| Host | Datei(en) | Versioniert? |
|---|---|---|
| N04-RTX | `llm-studio/worker-rtx/docker-compose.yml`, `.env.stock-rtx`, `docker-compose.m60-guard.yml`, `docker-compose.bonsai-m10-prod.yml` | ja (lokal committet, nicht gepusht) |
| N02-M60 | `llm-studio/worker-m60/docker-compose.yml`, `.env.m60-pool` | ja (lokal committet, nicht gepusht) |
| N11-M10 | `llm-studio/worker-tesla/docker-compose.yml` (JSON) | nein, per `llm-studio/worker-tesla/.gitignore` ausgeschlossen; der Live-Stand existiert nur auf dem Host |

Auf N04-RTX laufen außerdem `ollama-tesla-bonsai` (:11436, Compose `docker-compose.bonsai-m10-prod.yml`), `ollama-m60-guard` (:11442,
Compose `docker-compose.m60-guard.yml`, eingebunden per `extends`), `ollama-rgtx` (:11435, Stock) und ein ComfyUI-Container.

## Klone des Deployment-Repos

`origin/main` steht bei `9077a63` (2026-09-16). Die Klone sind divergent:

| Klon | vor / hinter origin | lokale, nicht gepushte Commits |
|---|---|---|
| Steuerhost `/opt/deployment/ollama` | 0 / 0 | keine; uncommittet: `.gitignore`, `worker-host/docker-compose.yml`, `worker-m60/docker-compose.yml` |
| N04-RTX | 7 / 2 | `8c97f04`, `badabbd`, `7cabec1`, `bdb9d51`, `e543b01`, `01903a9`, `f41b837`; es fehlen `82cc26b` und `9077a63` |
| N02-M60 | 8 / 0 | `8c97f04`, `34f264f`, `82e2b66`, `e3e21e5` (Merge von origin), `1f5d33f`, `6bcbe7d`, `8921ffb`, `8f5587a` |
| N11-M10 | 0 / 3 | keine; es fehlen `785e2a7`, `82cc26b`, `9077a63` |

Herkunft der lokalen Commits: `8c97f04` und `34f264f` (2026-09-05, N02-Pools; `8c97f04` liegt auf N04-RTX und N02-M60) sowie
`82e2b66` und `e3e21e5` (2026-10-03, MTP-Race-Fix, Merge mit origin) stammen aus früheren Claude-Sessions; `badabbd` (2026-10-04) aus
der Parallel-Session `ai-village-f2`; die übrigen (2026-10-05) beschreiben den Live-Stand dieser Tabelle.

Uncommittete Änderungen auf N11-M10 (Vorlagen für das N04-Design, nicht die Live-Config von N11-M10): `fork/compose/.env`
(`OLLAMA_CONTEXT_LENGTH=132768` – vermutlich 131072 gemeint –, `MAX_LOADED_MODELS=1`, `NUM_PARALLEL=1`; zuletzt geändert 2026-09-16)
und `llm-studio/worker-rtx/.env.tesla` (`OLLAMA_CONTEXT_LENGTH=262144`, zuletzt geändert 2026-08-29). Auf N11-M10 liegt außerdem
der Klon `tesla/ollama-legacy-gpu/` (alter Stand des GitHub-Forks, Commit vom 2026-06-25, lokale Änderungen an `compose/.env` und
`docker-compose.maxwell.yml`); die lokale `CLAUDE.md` nennt ihn als Deploy-Pfad von N11-M10. Auf N02-M60 liegt eine alte Kopie des
Gap-Prompts (`fork/RTX-TESLA-GAPS-PROMPT.md`).

## Offene Entscheidungen
1. Welcher Klon ist kanonisch, und wie werden die Historien zusammengeführt und gepusht? Vorschlag: `origin/main` (`9077a63`) als
   Basis, N02-M60 zuerst pushen (reiner Fast-Forward, enthält den Merge von origin), dann N04-RTX mit `git merge origin/main`
   (erwartete Konflikte nur in `worker-rtx/docker-compose.yml` und `worker-rtx/.env.m60`; dort gilt der Live-Stand), dann N11-M10 und
   Steuerhost per `git pull --ff-only`. Vor und nach jedem Merge `docker compose config` auf dem Host vergleichen.
2. `OLLAMA_GPU_AUTODETECT` auf N02-M60: kann auf 0 (die Pipeline ist ausdrücklich aus); danach N02-M60 neu messen.
3. N11-Vorlagen `fork/compose/.env` und `.env.tesla` (N04-Design): übernehmen, korrigieren oder verwerfen.
4. Soll `llm-studio/worker-tesla/docker-compose.yml` versioniert werden?
5. Backups per `.gitignore` ausnehmen? Den alten Klon `tesla/ollama-legacy-gpu/` auf N11-M10 archivieren (dann die `CLAUDE.md` anpassen)?
