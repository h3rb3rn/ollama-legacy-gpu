# Changelog

All notable changes to this fork are documented here.

---

## [Unreleased] — 2026-10-06

- **Kolibri-1 measurements:** 8–12 GPUs 10.4–11.1 tok/s decode and 65–69 tok/s prefill; greedy fill (`OLLAMA_FORCE_GPU_LAYERS=1`, scale 1.10) uses 8 of 12 GPUs, 7 fail with CUDA out of memory; needle tests pass up to 221k prompt tokens; 60-minute soak without errors. Details in docs/TUNING.md.
- **Kolibri-1:** `patch-llama-kolibri1.py` adds the `kolibri1` architecture (runtime part of the patch published with `Hob-forge/Kolibri-1-GGUF`). `Kolibri-1-Q4_K_M` (78.1B MoE, 47.5 GB) loads in one instance over 12× Tesla M60: 51/51 layers, `-c 262144`, batch 64, all in VRAM (53.2 GB). Decode 10.4–12.5 tok/s; German, reasoning, tool call and a 9421-token needle test pass.

- **VMM peer access:** `patch-llama-vmm-peer-access.py` limits the forced peer access of the CUDA VMM pool in NCCL builds to eight devices. 12 visible GPUs aborted with `peer mapping resources exhausted` (8 and 9 GPUs ran unpatched); 12× Tesla M60 now load `qwen3.6:35b` in one instance (42/42 layers, `-c 262144`, 10.4 tok/s decode).

## 2026-10-05

- **Based on:** Ollama v0.35.1 (llama.cpp b11232)
- **Image:** `ollama-gaps:kolibri-20261006` (`cuda12-maxwell`, CUDA 12.0.1, Archs 50;52;61;75;86, FA=ON, `BONSAI=OFF`)

Details: [docs/FORK-VS-STOCK.md](docs/FORK-VS-STOCK.md), [docs/TUNING.md](docs/TUNING.md), [docs/FLEET-STATE.md](docs/FLEET-STATE.md).

### Added

- **GPU-Abdeckung:** `cuda12-maxwell` baut 50…90 (Maxwell bis Hopper), damit gemischte Hosts ein Image nutzen. Die Images schreiben ihre
  Targets nach `/usr/lib/ollama/CUDA_ARCHS`; `gpu-detect.sh` maskiert nicht abgedeckte GPUs (`OLLAMA_UNSUPPORTED_GPU=mask|fail|ignore`),
  der Entrypoint bricht mit Exit 3 ab, wenn keine GPU passt. `patch-ollama-discovery.py` bestimmt die Compute Capability nach
  CUDA-Index und überspringt Geräte mit unbekannter CC (`OLLAMA_ALLOW_UNKNOWN_CC=1` erlaubt sie).
- **MTP-Schutz:** `patch-ollama-mtp-default.py` – `OLLAMA_DRAFT_NUM_PREDICT` ist serverweiter Standard und Obergrenze für
  `draft_num_predict` aus dem Modell-Manifest (0 = MTP aus); ein Anfragewert behält Vorrang; `gpu-detect.sh` setzt 0.
  Go-Test `TestDraftNumPredictServerDefault`.
- **Embedding im VRAM:** `patch-llama-input-gpu.py` legt die Eingabeschicht (Token-Embedding) auf das Gerät von Layer 0
  (`LLAMA_INPUT_LAYER_GPU=0` = Upstream-Platzierung). Empfohlen mit `LLAMA_ARG_FIT_TARGET=256`.
- **Pipeline-Parallelität:** `patch-llama-pipeline-parallel.py` – `LLAMA_PIPELINE_PARALLEL=0` schaltet sie ab (kleinere gepinnte
  Host-Puffer); ohne Variable unverändert.
- **CUDA Graphs auf Maxwell:** `patch-llama-cuda-graphs-legacy.py` – Opt-in `GGML_CUDA_GRAPHS_LEGACY=1`.
- **Messwerkzeuge:** `scripts/bench-throughput.sh` (wirksame Werte aus dem Runner-Log), `scripts/sweep-batch.sh`,
  `scripts/compare-batch-quality.py`; Daten in `docs/evidence/batch-quality-2026-10-05/`.
- **CI:** `BONSAI` ist standardmäßig `OFF` (Repository-Variable `OLLAMA_BONSAI` oder Dispatch-Input `bonsai`); die Spark-Compat-Schicht
  läuft nur bei `BONSAI=ON`. Bonsai bricht auf 0.35.1 beim Compat-Patch `002-clef.patch` des Prism-Baums.

### Changed

- `gpu-detect.sh` exportiert `OLLAMA_DRAFT_NUM_PREDICT=0` für alle Hosts (MTP ist auch auf der RTX-Gruppe langsamer).
- `patch-ollama-dynamic-pool.py` halbiert `-c` und `-np` bei gesetztem `OLLAMA_MAX_BATCH_SIZE` nur noch, wenn Flash Attention
  ausdrücklich aus ist; der Kontext bleibt wie konfiguriert (262144).

### Fixed

- **Fit (nextn-Slot):** `patch-llama-fit-nextn.py` zählt den nextn-Slot immer mit; bei MTP aus blieb sonst Layer 0 auf der CPU (`41/42`).
  Alle Instanzen laden `42/42`.

### Hardware-Stand

`qwen3.6:35b` (Registry-Stand `a7eb95c53bcf`, 35,5B) läuft auf N04-RTX (3 GPUs), N02-M60 und N11-M10 mit Kontext 262144, q4_0-KV und
42/42 Layern; Messwerte in [docs/TUNING.md](docs/TUNING.md). Offen: die Ursache der restlichen CPU-Bindung des Main-Threads auf Maxwell.

---

## [v0.30.0] — 2026-06-25

**Based on:** Ollama v0.30.10  
**Docker image:** `ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest`  
**Hardware:** N04-RTX — 4×Tesla M10 · 2×Tesla M60 · GTX 1060 · 3×RTX 2060 · 2×RTX 3060 (~114 GiB VRAM)

### Added

- **Flash Attention on all GPU architectures** (`scripts/patch-ollama-dynamic-pool.py`)  
  Discovery: `ggml_cuda_get_best_fattn_kernel()` returns `BEST_FATTN_KERNEL_TILE` for CC < 7.0  
  (not `NONE`). The TILE kernel runs on Maxwell/Pascal without tensor cores.  
  Result: FA=ON across all 12 GPUs — compute buffer 22–278 MiB instead of 11.4 GiB.

- **Partial-fill greedy fallback** (`scripts/patch-llama-tier-fitting.py`)  
  When greedy fill cannot place all layers, the previous code discarded all placements  
  and redistributed equally. New behaviour: keep greedy-placed layers, distribute only  
  overflow by remaining bandwidth-weighted budget. RTX 3060 retains its 9 greedy  
  layers; Tesla absorbs the overflow — correct pipeline order without OOM.

- **Dynamic GPU pool selection** (`scripts/patch-ollama-dynamic-pool.py`)  
  `selectGPUPool()` on every model load: small models → RTX fast pool (FA=ON, MMA);  
  large models → all 12 GPUs (FA=ON, TILE for Tesla). Prevents global FA=OFF from  
  a single legacy GPU in `CUDA_VISIBLE_DEVICES`.

- **NVML-based GPU auto-detection** (`scripts/gpu-detect.sh`)  
  Runs at container startup. Outputs `CUDA_VISIBLE_DEVICES` (worst→best bandwidth),  
  `OLLAMA_FAST_GPU_DEVICES`, tier threshold, bandwidth per GPU. No nvidia-smi required.

- **Auto-optimization proxy** (`scripts/ollama-proxy.py`, `scripts/auto-optimize.py`)  
  Transparent HTTP proxy (port 11434 → Ollama 11435). Benchmarks scale values and  
  MTP draft tokens per model; caches optimal config in `/root/.ollama/auto-optimize/`.  
  Result for qwen3.6:35b: scale=1.6, draft=2 → **24.3 tok/s** (vs ~16 default).

- **Layout cache** (`scripts/patch-ollama-dynamic-pool.py`, `scripts/auto-optimize.py`)  
  After a successful model load, measures per-GPU VRAM delta via NVML and writes  
  `--tensor-split` proportions to `/root/.ollama/layout-cache/`. Subsequent loads  
  inject the cached split, bypassing the `common_params_fit_impl` estimation loop.

- **Native CUBIN targets** (`dockerfiles/Dockerfile.cuda12-maxwell`)  
  Compiles with `-real` for CC 5.0–9.0: `50-real;52-real;60-real;61-real;70-real;  
  75-real;80-real;86-real;89-real;90-real`. Eliminates PTX JIT delay on cold start.

### Changed

- **CUDA 12.0.1 base image** — broadest driver compatibility for Maxwell (CC 5.0/5.2).
- **FetchContent fork integration** fixed: `FETCHCONTENT_FULLY_DISCONNECTED=ON` prevents  
  CMake from re-running `git checkout <upstream-sha>` after our `fit.cpp` replacement.  
  (→ now copies only `fit.cpp` from fork, keeping Ollama's bundled llama.cpp API.)
- **Proxy connection timeout** increased from 600s → 1800s: large models (llama4:scout)  
  take ~8 minutes to transfer 62 GiB across 12 GPUs. Short timeouts caused the scheduler  
  to propagate `context canceled`, killing llama-server mid-load.
- **`OLLAMA_NUM_PARALLEL=1`** in `.env`: halves KV-cache per GPU.  
  With partial fill, RTX 3060 at 80%+ model → only 2 GiB free → OOM on 2 GiB KV.  
  `np=1` leaves ~3 GiB headroom per GPU.

### Performance on N04-RTX

| Model | Pool | GPUs | FA | tok/s |
|-------|------|------|----|-------|
| qwen3.6:35b Q4\_K\_M | RTX fast | 3–4 of 12 | MMA (CC 7.5+) | **~24.3** (MTP draft=2) |
| llama4:scout Q4\_K\_M | Full 12-GPU | all 12 | MMA + TILE | **~1.7** |

*llama4:scout is bottlenecked by MoE inter-GPU communication over PCIe (no NVLink).  
Each decode step requires transferring active expert weights across 49 layers.*

### Known limitations

- BW-weighted overflow fix (`944fb9a`) in `patch-llama-tier-fitting.py` is not compiled  
  into the binary due to Docker PATCH_GUARD caching: the Python patch sees "Already patched"  
  on rebuild and skips re-application. A `--no-cache` CI build or PATCH_GUARD redesign  
  is needed to activate the fix. Current runtime uses raw-VRAM overflow distribution.
- `llama4:scout` + `qwen3.6:35b` cannot be loaded simultaneously: combined footprint  
  exceeds 114 GiB when KV-cache is included.

---

## [Unreleased]

- BW-weighted partial fill overflow (properly compiled via `--no-cache` build)
- `OLLAMA_NUM_PARALLEL` dynamic selection per pool (np=2 for fast pool, np=1 for full)
- Layout cache validation and cache-invalidation on model update
