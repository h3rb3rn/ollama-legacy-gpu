# Fork vs. Stock Ollama

Stand: 2026-10-05. Basis: Ollama v0.35.1, llama.cpp b11232 (per FetchContent
angezogen). Der Fork ist **kein Quellcode-Fork im Git-Sinn**: Das Docker-Build
holt den offiziellen Ollama-Tag und wendet Python-Patchskripte aus `scripts/`
auf die Go- und llama.cpp-Quellen an. Jedes Skript ist idempotent und bricht den
Build ab (fail closed), wenn der erwartete Anker im Upstream-Code nicht genau
einmal gefunden wird.

## Was der Fork anders macht (Kurzfassung)

| Thema | Stock Ollama | Fork |
| --- | --- | --- |
| GPU-Ziele | nur aktuelle Compute Capabilities | zusätzlich CC 5.0/5.2/6.x/7.0 (Maxwell, Pascal, Volta) per CUDA 12 / 11 |
| Flash Attention | global an/aus nach Erkennung | pro Tier: nur wenn alle beteiligten GPUs es können |
| GPU-Auswahl | Ollama-Scheduler | dynamischer Pool (schnelle GPUs zuerst, Legacy-GPUs nur bei Bedarf) |
| Batchgröße | intern berechnet (gemessen `-b 2048`), `OLLAMA_MAX_BATCH_SIZE` wirkungslos | `OLLAMA_MAX_BATCH_SIZE` wird ausgewertet |
| Kontext | wie konfiguriert | wie konfiguriert (die Kontext-Halbierung des Pool-Patches gilt nur bei ausdrücklich abgeschalteter Flash Attention) |
| Unbekannte GPU / CC | kein Abgleich mit den Build-Zielen des Images | Abgleich mit `CUDA_ARCHS`; Default `mask`, `fail`/`ignore` wählbar |
| MTP-Draft (qwen35moe) | per Modell-Manifest an (`draft_num_predict 2`), stürzt auf Maxwell ab (cuBLAS-Race) | `OLLAMA_DRAFT_NUM_PREDICT` ist Standard und Obergrenze auch für Manifest-Werte; Default 0 |
| Embedding-Tabelle (`token_embd`) | immer im Host-RAM | folgt Layer 0 in den VRAM |
| Fit (nextn-Slot) | zählt den Slot nur mit geladenem MTP; bei MTP aus bleibt Layer 0 auf der CPU (`41/42`) | zählt ihn immer mit (`42/42`) |
| Pipeline-Parallelität (Multi-GPU) | automatisch, vierfache Eingabepuffer im Host-RAM | per `LLAMA_PIPELINE_PARALLEL=0` abschaltbar |
| CUDA Graphs | auf Legacy-GPUs deaktiviert | per `GGML_CUDA_GRAPHS_LEGACY=1` zuschaltbar |
| Mehr als 8 GPUs in einem Prozess | NCCL-Build erzwingt VMM-Peer-Zugriff auf alle Geräte; ab 9 GPUs `peer mapping resources exhausted` | Erzwingung nur bis 8 Geräte (`patch-llama-vmm-peer-access.py`); 12× Tesla M60 laufen in einer Instanz |
| Architektur `kolibri1` (Aleph Alpha Kolibri-1, 78B MoE) | `unknown model architecture: 'kolibri1'` | Architektur über `patches/kolibri1/` (`patch-llama-kolibri1.py`) |
| Jinja `tojson` | Template-Fehler bei einigen Modellen | kompatibel |

## Patchskripte

Go (Ollama):

- `patch-ollama-fa.py` – tier-bewusste Flash-Attention-Entscheidung.
- `patch-ollama-dynamic-pool.py` – `selectGPUPool`: Modell läuft auf dem
  schnellen Pool, wenn es in dessen VRAM passt (Schwelle 75 %), sonst auf allen.
  Die Halbierung von `-c` und `-np` bei gesetztem `OLLAMA_MAX_BATCH_SIZE` gilt
  nur, wenn Flash Attention ausdrücklich aus ist.
- `patch-ollama-batch.py` – `OLLAMA_MAX_BATCH_SIZE`.
- `patch-ollama-discovery.py` – Compute Capability nach CUDA-Index, fail closed;
  `OLLAMA_ALLOW_UNKNOWN_CC` erlaubt unbekannte Karten explizit.
- `patch-ollama-mtp-default.py` – `OLLAMA_DRAFT_NUM_PREDICT`: Wert aus der
  Anfrage gewinnt; ein Wert aus dem Modell-Manifest wird auf die Variable begrenzt
  (0 = MTP aus); ohne beides gilt die Variable als Standard. Test:
  `TestDraftNumPredictServerDefault`.

llama.cpp:

- `patch-llama-tier-fitting.py` – Fitting über gemischte GPU-Tiers, Split-Buffer entfernt.
- `patch-llama-fit-nextn.py` – zählt den nextn-Slot im Fit immer mit.
- `patch-llama-input-gpu.py` – `llama_model::load_tensors` legt die
  Eingabeschicht (Token-Embedding) auf das Gerät von Layer 0.
  `LLAMA_INPUT_LAYER_GPU=0` stellt die Upstream-Platzierung wieder her.
  Die Platzierung läuft weiter über `select_weight_buft` (mit CPU-Fallback).
  `LLAMA_ARG_OVERRIDE_TENSOR=token_embd.weight=CUDA0` umgeht diese Prüfung und
  bricht den Scheduler ab (N11-M10, b11232).
- `patch-llama-pipeline-parallel.py` – `LLAMA_PIPELINE_PARALLEL=0` überspringt
  die Pipeline-Parallelität; ohne Variable unverändert.
- `patch-llama-vmm-peer-access.py` – `ggml_cuda_pool_vmm::alloc` erzwingt in NCCL-Builds die Peer-Freigabe nur bis 8 Geräte
  (CUDA erlaubt 8 Peers je Mapping); mit 9–12 GPUs erhält nur das besitzende Gerät Zugriff.
- `patch-llama-kolibri1.py` – spielt `patches/kolibri1/kolibri1-llama.cpp.patch` ein (Laufzeitteil des Patches aus
  `Hob-forge/Kolibri-1-GGUF`, MIT): Gating-Modus `SIGMOID_LOGIT_ADD`, Modellgraph, Tokenizer-Typ `kolibri1`. Ohne den Patch
  lädt kein Kolibri-1-GGUF. Das Skript berührt `src/CMakeLists.txt`, damit der `models/*.cpp`-Glob neu ausgewertet wird.
- `patch-llama-jinja-tojson.py` – `tojson`-Kompatibilität.
- `patch-llama-cuda-graphs-legacy.py` – Opt-in für CUDA Graphs auf CC < 7.0.

## Welche Patches welches Image bekommt

| Patch | cuda11-legacy | cuda12-maxwell | cuda13-rtx |
| --- | --- | --- | --- |
| tier-fitting, fit-nextn, input-gpu, pipeline-parallel, vmm-peer-access, kolibri1, jinja-tojson | ja | ja | ja |
| fa, dynamic-pool, batch | ja | ja | ja |
| discovery, mtp-default | nein | ja | ja |
| cuda-graphs-legacy | nein | ja | nein |

## Laufzeit-Skripte (im Image)

- `gpu-detect.sh` – erkennt GPUs, bildet Tiers und exportiert u. a.
  `OLLAMA_GPU_TIER_THRESHOLD`, `OLLAMA_FAST_GPU_DEVICES`,
  `OLLAMA_FAST_POOL_VRAM_GB`, `OLLAMA_DRAFT_NUM_PREDICT=0`. Prüft Karten gegen
  `CUDA_ARCHS` des Images (`OLLAMA_UNSUPPORTED_GPU=mask|fail|ignore`, Default `mask`).
- `ollama-entrypoint.sh`, `auto-optimize.py`, `ollama-proxy.py`,
  `inject-presets.py` – Start, Tuning, Proxy und Presets
  (`presets/gpu-targets.json`).

## Messwerkzeuge (im Repo)

- `scripts/bench-throughput.sh` – Decode und Prefill mit den **wirksamen** Werten aus dem
  Runner-Log (`n_ctx`, `n_batch`, `n_ubatch`, Flash Attention, KV-Typ, Layer).
- `scripts/sweep-batch.sh` – Batch-Leiter je Host (startet den Container neu).
- `scripts/compare-batch-quality.py` – vergleicht Antworten bei verschiedenen Batch-Größen.

## Umgebungsvariablen (fork-spezifisch)

| Variable | Wirkung | Default |
| --- | --- | --- |
| `OLLAMA_MAX_BATCH_SIZE` | Batchgröße (`-b`/`-ub`) | Stock-Wert |
| `OLLAMA_DRAFT_NUM_PREDICT` | MTP-Draft-Länge, Standard und Obergrenze (auch für Manifest-Werte), 0 = aus | 0 (durch gpu-detect) |
| `OLLAMA_UNSUPPORTED_GPU` | `mask`, `fail`, `ignore` | `mask` |
| `OLLAMA_ALLOW_UNKNOWN_CC` | unbekannte CC zulassen | aus |
| `GGML_CUDA_GRAPHS_LEGACY` | CUDA Graphs auf Legacy-GPUs | aus |
| `LLAMA_PIPELINE_PARALLEL` | `0` = Pipeline-Parallelität aus | an |
| `LLAMA_INPUT_LAYER_GPU` | `0` = Embedding wie Upstream auf der CPU | an |
| `LLAMA_ARG_FIT_TARGET` | Fit-Reserve je GPU in MiB (llama.cpp-Variable) | Upstream (~2 GiB) |

`LLAMA_ARG_FIT_TARGET` ist Upstream; der Fork empfiehlt `256`, weil mit dem Embedding im VRAM
sonst ein Experten-Tensor (210,82 MiB `ffn_down_exps`) in den Host-RAM ausweicht.

## Nicht geändert

API, Modellformat, Modelfile-Syntax, Registry und CLI entsprechen Stock.
KV-Cache-Konfiguration (`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`) ist wie
bei Stock. Der Bonsai-Pfad (`BONSAI=ON`) ist optional und im aktuellen Default
aus (`BONSAI=OFF`).
