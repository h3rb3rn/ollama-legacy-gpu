# Changelog

All notable changes to this fork are documented here.

---

## [Unreleased] — 2026-10-03 (RTX/Tesla gaps, siehe RTX-TESLA-GAPS-PROMPT.md)

Status: Python-Tests (37) und `go build/test ./discover ./server` auf v0.35.0 grün; Test-Image
`ollama-gaps:test-20261003` (CUDA 12.0.1, Archs 50;52;61;75;86, FA=ON, BONSAI=OFF) auf den `:11434`-Instanzen von
N11-M10, N02-M60 und N04-RTX getestet (2026-10-03/04).

Messwerte `qwen3.6:35b` (120 Token, temperature 0):
- N04-RTX (4 RTX, Pool :11434): Stock 0.35.0 33.05 tok/s → Fork-Image 40.0–40.6 tok/s; 42/42 Layer, kein „no kernel image“.
- N11-M10 (i5-3470T): 6.3–6.5 tok/s → mit `GGML_CUDA_GRAPHS_LEGACY=1` 8.6 tok/s (+35 %); 2048-Token-Lauf stabil (8.3 tok/s).
- N02-M60: 12.6–13.7 tok/s ohne, 12.4–13.2 mit Graph-Schalter (kein Effekt; Wirkung hängt von der Host-CPU ab).
- Gap 2: Modell ohne Modelfile-Override, Autodetect-Defaultpfad → `OLLAMA_DRAFT_NUM_PREDICT=0` gesetzt, kein Crash.

### Fixed / Added

- **Gap 1 — RTX/Turing/Ampere:** `cuda12-maxwell` baut jetzt `50…90` (inkl. 75/80/86/89/90), damit gemischte
  Hosts (N04-RTX: Maxwell + Pascal + Turing + Ampere in einem Container) ein Image bekommen; CUDA 13 kann
  Maxwell nicht. Vorher: Preset `50;52;60;61;70` → RTX wurde verworfen, bei unbestimmter CC aber trotzdem geladen.
  - `scripts/patch-ollama-discovery.py`: CC per CUDA-Index aus dem Gerätenamen statt Positionszähler; unbekannte
    CC bei bekannter Arch-Liste → Gerät wird übersprungen (Override `OLLAMA_ALLOW_UNKNOWN_CC=1`).
  - Images schreiben ihre Targets nach `/usr/lib/ollama/CUDA_ARCHS`; `gpu-detect.sh` maskiert nicht abgedeckte GPUs
    (`OLLAMA_UNSUPPORTED_GPU=mask|fail|ignore`), Entrypoint bricht mit Exit 3 ab, wenn keine GPU passt.
- **Gap 2 — MTP-Schutz ohne Proxy:** `scripts/patch-ollama-mtp-default.py` führt `OLLAMA_DRAFT_NUM_PREDICT` als
  serverweiten Default für `draft_num_predict` ein (explizite Modell-/Request-Werte gewinnen). `gpu-detect.sh`
  setzt 0 (zunächst nur bei Legacy-GPUs; seit der RTX-Messung für alle Hosts, da MTP auch dort langsamer ist). Gilt damit für jedes Modell, auch Pool-Container.
- **Gap 3 (Teilergebnis):** `scripts/patch-llama-cuda-graphs-legacy.py` — ggml-cuda deaktiviert CUDA-Graphs fest für CC < 7.0;
  opt-in `GGML_CUDA_GRAPHS_LEGACY=1`. Hilft auf CPU-schwachen Hosts (N11-M10), nicht auf N02-M60. Standardmäßig aus.
- **Beobachtung Gap 1:** Auf N04-RTX (12 GPUs) läuft ein Discovery-Durchlauf in den Watchdog-Timeout; die CC-Zeilen fehlen
  dann, die Geräte werden jetzt übersprungen statt blind geladen. Ein zweiter Durchlauf erkennt alle 4 RTX korrekt.
- **Gap 4 (gelöst):** nicht der nextn-Layer, sondern ein Off-by-one: der Fit zählte den nextn-Slot nur mit `load_mtp`, der Lader
  immer → Trunk-Layer 0 blieb bei MTP-aus auf der CPU (`41/42`). `scripts/patch-llama-fit-nextn.py`; auf allen drei Hosts 42/42.
- **Ollama 0.35.1:** Fork baut ohne Bonsai (`BONSAI=OFF`, neuer CI-Default; `OLLAMA_BONSAI`-Variable bzw. Dispatch-Input `bonsai`).
  Bonsai bricht auf 0.35.1 beim Compat-Patch `002-clef.patch` (Prism-Baum) — Nachzug folgt.
- **Tuning:** siehe `TUNING-2026-10-04.md` (Batch 512: Prefill ×1.75–2.1; MTP-Default 0 für alle Hosts).
- **Offen:** Gap 3 (restliche Ursache des CPU-gebundenen Main-Threads); Mixed-Pool Maxwell+RTX in
  *einem* Container und `OLLAMA_UNSUPPORTED_GPU`-Maskierung nicht auf Hardware getestet.

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
