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
| Batchgröße | intern berechnet (z. B. 2048), `OLLAMA_MAX_BATCH_SIZE` wirkungslos | `OLLAMA_MAX_BATCH_SIZE` wird ausgewertet |
| Unbekannte GPU / CC | kein Abgleich mit den Build-Zielen des Images | Abgleich mit `CUDA_ARCHS`; Default `mask`, `fail`/`ignore` wählbar |
| MTP-Draft (qwen35moe) | automatisch an, stürzt auf Maxwell ab (cuBLAS-Race) | per `OLLAMA_DRAFT_NUM_PREDICT` steuerbar, Maxwell-Default 0 |
| Embedding-Tabelle (`token_embd`) | immer im Host-RAM | folgt Layer 0 in den VRAM |
| Fit (nextn-Layer) | Off-by-one, ein Layer wird falsch zugeordnet | korrigiert |
| CUDA Graphs | auf Legacy-GPUs deaktiviert | per `GGML_CUDA_GRAPHS_LEGACY=1` zuschaltbar |
| Jinja `tojson` | Template-Fehler bei einigen Modellen | kompatibel |

## Patchskripte

Go (Ollama):

- `patch-ollama-fa.py` – tier-bewusste Flash-Attention-Entscheidung.
- `patch-ollama-dynamic-pool.py` – `selectGPUPool`: Modell läuft auf dem
  schnellen Pool, wenn es in dessen VRAM passt (Schwelle 75 %), sonst auf allen.
- `patch-ollama-batch.py` – `OLLAMA_MAX_BATCH_SIZE`.
- `patch-ollama-discovery.py` – Compute Capability nach CUDA-Index, fail closed;
  `OLLAMA_ALLOW_UNKNOWN_CC` erlaubt unbekannte Karten explizit.
- `patch-ollama-mtp-default.py` – `OLLAMA_DRAFT_NUM_PREDICT`; ein explizit
  gesetzter Request-Wert hat Vorrang.

llama.cpp:

- `patch-llama-tier-fitting.py` – Fitting über gemischte GPU-Tiers, Split-Buffer entfernt.
- `patch-llama-fit-nextn.py` – zählt den nextn-Slot im Fit immer mit (Fix in 676a3b2).
- `patch-llama-input-gpu.py` – **neu:** `llama_model::load_tensors` legt die
  Eingabeschicht (Token-Embedding) auf das Gerät von Layer 0.
  `LLAMA_INPUT_LAYER_GPU=0` stellt die Upstream-Platzierung wieder her.
  Die Platzierung läuft weiter über `select_weight_buft` (mit CPU-Fallback).
  `LLAMA_ARG_OVERRIDE_TENSOR=token_embd.weight=CUDA0` umgeht diese Prüfung und
  bricht den Scheduler ab (N11-M10, b11232).
- `patch-llama-jinja-tojson.py` – `tojson`-Kompatibilität.
- `patch-llama-cuda-graphs-legacy.py` – Opt-in für CUDA Graphs auf CC < 7.0.

## Welche Patches welches Image bekommt

| Patch | cuda11-legacy | cuda12-maxwell | cuda13-rtx |
| --- | --- | --- | --- |
| tier-fitting, fit-nextn, input-gpu, jinja-tojson | ja | ja | ja |
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

## Umgebungsvariablen (fork-spezifisch)

| Variable | Wirkung | Default |
| --- | --- | --- |
| `OLLAMA_MAX_BATCH_SIZE` | Batchgröße (`-b`/`-ub`) | Stock-Wert |
| `OLLAMA_DRAFT_NUM_PREDICT` | MTP-Draft-Länge, 0 = aus | 0 (durch gpu-detect) |
| `OLLAMA_UNSUPPORTED_GPU` | `mask`, `fail`, `ignore` | `mask` |
| `OLLAMA_ALLOW_UNKNOWN_CC` | unbekannte CC zulassen | aus |
| `GGML_CUDA_GRAPHS_LEGACY` | CUDA Graphs auf Legacy-GPUs | aus |
| `LLAMA_INPUT_LAYER_GPU` | `0` = Embedding wie Upstream auf der CPU | an |
| `LLAMA_ARG_FIT_TARGET` | Fit-Reserve je GPU in MiB (llama.cpp-Variable) | Upstream (~2 GiB) |

Hinweis: `LLAMA_ARG_FIT_TARGET` ist Upstream; der Fork empfiehlt `256`, weil mit
dem Embedding im VRAM sonst ein Experten-Tensor (210,82 MiB `ffn_down_exps`) in
den Host-RAM ausweicht.

## Nicht geändert

API, Modellformat, Modelfile-Syntax, Registry und CLI entsprechen Stock.
KV-Cache-Konfiguration (`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`) ist wie
bei Stock. Der Bonsai-Pfad (`BONSAI=ON`) ist optional und im aktuellen Default
aus (`BONSAI=OFF`).
