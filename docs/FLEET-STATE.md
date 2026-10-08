# Fleet state (2026-10-08)

The Ollama instances of N04-RTX, N02-M60 and N11-M10 and the state of the deployment repository
(`https://git.4noobs.de/h3rb3rn/ollama.git`). Read on the hosts with `docker inspect`, `git status` and `git log`.

Requirements for the instances: 256k KV cache at q4_0, everything in VRAM and nothing in RAM, `qwen3.6:35b` primary and warm
(`MAX_LOADED_MODELS=1`, `NUM_PARALLEL=1`, `KEEP_ALIVE` 24 h).

## Live state (all hosts on Ollama v0.40.0, GitHub-built image `ghcr.io/h3rb3rn/ollama-legacy@sha256:f4afde44…`, run 37688616665)

| Setting | N04-RTX | N02-M60 (pool, `:11434`) | N11-M10 |
|---|---|---|---|
| Cards in the pool | 3: 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 4× Tesla M60 (8 GiB), GPU0-3 | 4× Tesla M10 (8 GiB) |
| Free for other uses | 1× RTX 3060 (`GPU-63bfbd4b-…`, bus 09:00.0), used by ComfyUI | – | – |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | 262144 | 262144 |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | 64 | 64 |
| `OLLAMA_KV_CACHE_TYPE` | q4_0 | q4_0 | q4_0 |
| `LLAMA_ARG_FIT_TARGET` | 256 | 256 | 256 |
| `LLAMA_ARG_MMPROJ_OFFLOAD` | true | true | true |
| `OLLAMA_DRAFT_NUM_PREDICT` | 0 | 0 | 0 |
| `OLLAMA_GPU_AUTODETECT` | 0 | 1 | 0 |
| `OLLAMA_FORCE_GPU_LAYERS` / `OLLAMA_LAYER_OVERHEAD_SCALE` | – | – | – |
| `GGML_CUDA_GRAPHS_LEGACY` | – | – | 1 |
| `LLAMA_PIPELINE_PARALLEL` | not set (pipeline on) | 0 | 0 |
| Layers / effective context | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode / prefill (tok/s) | 40–43 / 1028–1148 (v0.40.0; 42.3 / 1140–1172 on 0.35.1) | 15.0–16.0 / 91–93 | 9.2–9.3 / 24.8 (v0.40.0; 9.3 / 24.2 on 0.35.1) |
| Not in VRAM according to `/api/ps` | 0 MiB | 0 MiB | 0 MiB |
| Host buffer `CUDA_Host compute` | 1029 MiB | 32.8 MiB | 32.8 MiB |

Measurements, method and rationale of the settings: [TUNING.md](TUNING.md).

### N02-M60 AI-Village endpoints (Ollama v0.40.0, GitHub-built image)

One endpoint per agent; the port -> agent mapping is part of the agents' configuration (`OLLAMA_URL`) and must not change. The server-side
`OLLAMA_CONTEXT_LENGTH` of each instance equals the `num_ctx` its agent requests. Model store: `blobs` and `manifests` of `/opt/ollama/models`
read-only, private directory `/opt/ollama/models/m60-v040-<pool|gpuN>` per instance (the agents never pull or create models).
Checked by loading each agent's model with its context (`docs/evidence/n02-village-2026-10-07/`); the decode values are from the check on the GitHub image on 2026-10-08, measured one endpoint at a time while the agents were active (the check of 2026-10-07 loaded all nine models at once and is not comparable), the King's value is from its idle check:

| Port | Container | GPUs | Agent | Model | Context | Layers | On GPU | Decode (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 11434 | `ollama-m60-pool` | GPU0-3 | 01-king | `qwen3.6:35b` | 262144 | 42/42 | 100 % | 13.5 / 13.6 |
| 11435 | `ollama-m60-gpu4` | GPU4 | 02-explorer | `qwen3.5:4b` | 262144 | 34/34 | 100 % | 15.7 / 14.9 |
| 11436 | `ollama-m60-gpu5` | GPU5 | 03-librarian | `granite4.2:3b` | 131072 | 41/41 | 100 % | 18.1 / 20.6 |
| 11437 | `ollama-m60-gpu6` | GPU6 | 04-artisan | `granite4.2:3b` | 131072 | 41/41 | 100 % | 20.4 / 21.9 |
| 11438 | `ollama-m60-gpu7` | GPU7 | 05-interpreter | `gemma3:4b` | 131072 | 35/35 | 100 % | 15.0 / 15.8 |
| 11439 | `ollama-m60-gpu8` | GPU8 | 06-operator | `nemotron-3-nano:4b` | 262144 | 43/43 | 100 % | 22.8 / 12.5 |
| 11440 | `ollama-m60-gpu9` | GPU9 | 07-methodologist | `huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF:latest` | 262144 | 34/34 | 100 % | 16.4 / 16.2 |
| 11441 | `ollama-m60-gpu10` | GPU10 | 08-logician | `hf.co/XHToken/Spark-X2.5-4B-GGUF:Q4_K_M` | 262144 | 37/37 | 100 % | 15.0 / 14.2 |
| 11442 | `ollama-m60-gpu11` | GPU11 | 09-chronicler | `hf.co/webAI-Official/TwIL-LM3-Pro:Q4_K_M` | 131072 | 41/41 | 100 % | 22.4 / 21.6 |

All instances: batch 64 (host buffer 17–33 MiB), flash attention, q4_0 KV cache, MTP off, no CUDA errors. Think level, `num_predict` and
`keep_alive` come with each request and were not part of the check. The previous containers (local v0.40.0 build, 2026-10-08 14:31–14:40) are stopped as `ollama-m60-*-pre-gh040-20261008-143125`;
the 12-GPU container (Kolibri-1) is stopped as `ollama-m60-12-pre-village-20261007-233313`. The King's pool generated 39 long agent answers at a median of
7.8 tok/s (7.0–8.7) on the previous container and 8.3 tok/s on the first long answer of the new one (all nine endpoints busy).

### N11-M10 (Ollama v0.40.0, GitHub-built image)

`ollama` (:11434), 4× Tesla M10, the same digest as the N04-RTX instances (`sha256:f4afde44…`, run 37688616665). Settings unchanged from the Ollama 0.35.1 instance
(see the table above); the shared model store is mounted read-only (private directory `/opt/ollama/models/n11-v040`). The previous container is stopped and kept as
`ollama-pre-gh040-20261008-140513` (image `ollama-gaps:pipefix-20261005`), so it can be started again. Rollout check (`qwen3.6:35b`): 9.3 / 9.3 / 9.2 tok/s decode,
24.8 tok/s prefill, 42/42 layers, 100 % on the GPUs, no CUDA errors (before: 8.7 / 8.9 / 8.9 and 24.8).

**Do not send test requests to a live village endpoint.** While the agents were active, check requests to the King's pool (`:11434`) alternated with the
agents' requests and made Ollama reload the model repeatedly: 7 loads of `qwen3.6:35b` in 65 minutes against 2 loads in 15 hours on the previous container
(each load takes about 2.5 minutes, requests wait during that time; one agent request ended with HTTP 499 after 38 minutes). After the checks stopped, the model
stayed loaded and the agents generated normally. Observe a live endpoint with `/api/ps` and the container log only; run checks while the village is paused.

### N04-RTX instances (Ollama v0.40.0, GitHub-built image)

Image `ghcr.io/h3rb3rn/ollama-legacy@sha256:f4afde4402e55d28af4d09e36d4c7c6c12fa226c83746bbb7baac253dbedb719`: GitHub Actions run 37688616665,
revision `738c694`, Ollama v0.40.0 / llama.cpp b11351, tag `candidate-37688616665-1-cuda12-maxwell-native` (this build predates the
version tags; the newest CI build `cuda12-maxwell-0.40.1` is based on Ollama v0.40.1 and not deployed). All four instances use it, mount `blobs` and the legacy `manifests` of `/opt/ollama/models` read-only and write to a
private directory each (`n04-v040-rtx`, `n04-v040-rgtx`, `n04-v040`, `n04-v040-guard`). The previous containers are stopped and kept as
`ollama-pre-gh040-…`, `ollama-rgtx-pre-gh040-…`, `ollama-m10-pre-gh040-…` and `ollama-m60-guard-pre-gh040-…` (2026-10-08).

| Setting | `ollama` (:11434) | `ollama-rgtx` (:11435) |
|---|---|---|
| Cards | 2× RTX 2060 + 1× RTX 3060 (12 GiB) | RTX 2060 + GTX 1060 (6 GiB) |
| Context / batch | 262144 / 512 | 262144 / 512 |
| `LLAMA_ARG_FIT_TARGET` / `OLLAMA_DRAFT_NUM_PREDICT` / `LLAMA_ARG_MMPROJ_OFFLOAD` | 256 / 0 / true | 256 / 0 / true |
| `OLLAMA_SCHED_SPREAD` | true | false |
| Model checked | `qwen3.6:35b`, 42/42 layers | `moe-sovereign-planner-9b`, 33/33 layers |
| Replaces | the 0.35.1 fork build `pipefix-20261005` | stock Ollama 0.35.0 |

The same image serves the two Tesla instances:

| Setting | `ollama-m10` (:11436) | `ollama-m60-guard` (:11442) |
|---|---|---|
| Cards | 4× Tesla M10 (8 GiB) | 2× Tesla M60 (8 GiB) |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | 32768 |
| Batch | 64 (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | default (n_batch 1024) |
| `LLAMA_ARG_FIT_TARGET` / `OLLAMA_DRAFT_NUM_PREDICT` / `LLAMA_PIPELINE_PARALLEL` | 256 / 0 / 0 | 256 / 0 / 0 |
| `GGML_CUDA_GRAPHS_LEGACY` / `LLAMA_ARG_MMPROJ_OFFLOAD` | 1 / true | – / false |
| `OLLAMA_SCHED_SPREAD` / `OLLAMA_MAX_LOADED_MODELS` | true / 1 | false / 1 |
| Model store | `blobs` and `manifests` of `/opt/ollama/models` read-only, private directory `/opt/ollama/models/n04-v040` | same, private directory `…/n04-v040-guard` |
| Replaces | `ollama-tesla-bonsai` (Bonsai build, Ollama 0.34.1; stopped as `…-pre-v040-20261007-210927`), then the local v0.40.0 build | previous `ollama-m60-guard` (stopped as `…-pre-v040-20261007-210927`), then the local v0.40.0 build |

Bonsai models (`bonsai2:27b-pq2_0`) are no longer served on `:11436`. Measurements: [TUNING.md](TUNING.md#ollama-v0400-llamacpp-b11351).

Differences between the hosts that affect comparisons: N02-M60 runs with `OLLAMA_GPU_AUTODETECT=1` (activates the fork's
fit), N11-M10 with `GGML_CUDA_GRAPHS_LEGACY=1` (+35 % decode on N11-M10, no effect on N02-M60).

The decode/prefill, layer and host buffer values for N02-M60 are those of the earlier 4-GPU pool (`qwen3.6:35b`,
15.0–16.0 / 91–93 tok/s); on Ollama v0.40.0 the pool reaches 13.5 tok/s in the endpoint check below (A/B against the 0.35.1 build: [TUNING.md](TUNING.md)). The 12-GPU single instance (kept as `docker-compose.single12.yml`) ran `qwen3.6:35b` at 10.4 tok/s (all 12 GPUs) and `Kolibri-1-Q4_K_M` (78.1B, 51/51 layers,
`-c 262144`) runs at 11.1 / 10.9 tok/s decode and 67.4 tok/s prefill on 8 GPUs (greedy fill); measurements, context ladder and soak test in
[TUNING.md](TUNING.md).

## Model and MTP guard

All three hosts carry `qwen3.6:35b` in the current registry state: digest `a7eb95c53bcf`, 35.5B parameters (35,505,251,456),
21.07 GiB, 41 blocks, NextN head `blk.40.nextn.*`, manifest `draft_num_predict 2`. The manifest is capped to 0 by
`OLLAMA_DRAFT_NUM_PREDICT=0` (MTP off, `patch-ollama-mtp-default.py`); a request value takes precedence. Checked per
host: model loaded empty, runner command line without `--spec-*` arguments, then generation without errors.

In addition, N02-M60 holds the tag `qwen3.6:35b-nospec` (digest `aae923a8c006`, 35.5B, from 2026-09-05).

## Where the configuration lives

The deployed state of all hosts lives in the `main` branch of the deployment repository:

| Host | Files | Services |
|---|---|---|
| N04-RTX | `llm-studio/worker-rtx/docker-compose.yml` with `.env.stock-rtx`, `.env.stock-rgtx`, `.env.tesla`; `docker-compose.rtx-pool.yml` (`.env.rtx-pool`), `docker-compose.rgtx.yml` (`.env.rgtx`), `docker-compose.m10-pool.yml`, `docker-compose.m60-guard.yml` | `ollama` (:11434), `ollama-rgtx` (:11435, stock), `ollama-m60-guard` (:11442), `ollama-m10` (:11436), ComfyUI |
| N02-M60 | `llm-studio/worker-m60/docker-compose.yml` with `.env.m60-pool` and `.env.m60-single`; alternative layout `docker-compose.single12.yml` with `.env.m60-single12` | `ollama-m60-pool` (:11434) and `ollama-m60-gpu4` … `ollama-m60-gpu11` (:11435–11442), one endpoint per AI-Village agent |
| N11-M10 | `llm-studio/worker-tesla/docker-compose.m10-pool.yml` | `ollama` (:11434) |
| Control host | `llm-studio/vm-without-gpu/docker-compose.yml` | `open-webui`, `searxng` |

Each N04 instance is started from its own Compose file with its own project name (`docker compose -p n04-gh-rtx -f docker-compose.rtx-pool.yml up -d`, likewise `n04-gh-rgtx`, `n04-gh-m10`, `n04-gh-guard`); the old shared `docker-compose.yml` and `update.sh` were removed. N11-M10 uses `docker compose -p n11-gh-m10 -f docker-compose.m10-pool.yml up -d`.
The Compose file of N04-RTX includes `ollama-m60-guard` via `extends` from `docker-compose.m60-guard.yml`.
The ignored `.env` files (among others `worker-tesla/.env`, `worker-legacy-gpu/.env`, `worker-host/.env`) exist only on the hosts.

## Git state

The clones of the deployment repository on N04-RTX, N02-M60, N11-M10 and the control host are at `origin/main`, without deviations
in tracked or untracked files (control host: only `llm-studio/vm-without-gpu/.env.example` and
`.env.ollama` are untracked; `CLAUDE.md`, `.claude/` and `fork/` are excluded via `.gitignore`).
The history contains the states of the three hosts as a merge (N02-M60 as a fast-forward, N04-RTX as a merge, the live state of N04-RTX
wins) and the Compose file of N11-M10. Origin of the older commits: `8c97f04`, `34f264f` (2026-09-05, N02 pools) and `82e2b66`,
`e3e21e5` (2026-10-03, MTP race fix, merge with origin) from earlier Claude sessions, `badabbd` (2026-10-04) from the parallel session
`ai-village-f2`; the rest (2026-10-05) describe the live state.

The source code of the fork lives in `github.com/h3rb3rn/ollama-legacy-gpu` (working copy: `/opt/deployment/ollama/fork/repo`).
Raw data of the Bonsai reports lives under `/opt/deployment/ollama/fork/.bonsai-work/` (`rtx-fa-20260929`, `rollout-20260930`,
`fleet-v035-20261001`) next to the source clones `Bonsai-demo` and `mlx-vlm-0.7.2`.

## Open decision

`OLLAMA_GPU_AUTODETECT` on N02-M60: can be set to 0 (the pipeline is explicitly off); then measure N02-M60 again.
