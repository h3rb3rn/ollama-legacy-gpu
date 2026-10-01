# Bonsai-Demo: Inferenz ohne Ollama / llama.cpp

Quellcodeprüfung am 2026-09-29. Lokal ausgecheckt, keine Setup-Skripte gestartet
und keine Modellgewichte zusätzlich heruntergeladen:

- `PrismML-Eng/Bonsai-demo`: `69c3a8beeab80283bfd45cb7b7a6b927075c29fd`,
  unter `../.bonsai-work/Bonsai-demo`.
- `Blaizzy/mlx-vlm`, Tag `v0.7.2`:
  `a74c7de90a344a2c2c7334acb4e48b57a40480e2`,
  unter `../.bonsai-work/mlx-vlm-0.7.2`.
- Der bereits vorhandene Prism-Backend-Quellstand des Ollama-Forks ist
  `adfffbe41b2cabcd51fff326ab045662265062bb`, Release
  `prism-b10743-adfffbe`. Genau dieses Release lädt die aktuelle Demo laut
  `scripts/download_binaries.sh:16` ebenfalls herunter.

## Tatsächliche Startwege

| Einstieg | Prozess / Bibliothek | Modellformat |
|---|---|---|
| `start_llama_server.sh` | Prism `llama-server` | GGUF, standardmäßig PQ2_0 |
| `run_llama.sh` | Prism llama.cpp-CLI | GGUF |
| `start_mlx_server.sh` mit Bonsai 2 | `.venv-vlm/bin/python -m mlx_vlm.server` | separater MLX-2-Bit-Pack |
| `run_mlx.sh` mit Bonsai 2 | `mlx_generate_bonsai2.py` → `mlx_vlm.load` / `generate` | separater MLX-2-Bit-Pack |
| `start_openwebui.sh` | Open WebUI verbindet sich mit einem der beiden Server | gemäß gewähltem Backend |
| `start_agent_server.sh` | Wrapper um `start_llama_server.sh` | GGUF |

`start_openwebui.sh:16–25` akzeptiert genau `llama` oder `mlx`. Für `mlx`
verlangt es explizit macOS/arm64. Die MLX-Einzelstarter prüfen ebenfalls die
Plattform. Ein eigener NVIDIA-PyTorch-/Triton-/TensorRT-Inferenzpfad für dieses
27B-Textmodell ist in der geprüften Demo nicht implementiert. Diese Aussage
betrifft die Demo; sie ist keine Aussage über sämtliche heutigen oder künftigen
MLX-Backends oder externe Bonsai-Projekte.

## Was MLX konkret anders ausführt

Für Bonsai 2 pinnt `scripts/requirements-mlx-vlm.txt` MLX 0.32.2,
MLX-VLM 0.7.2 und Transformers 5.14.1. Die separate `.venv-vlm` verwendet
den nativen Bonsai-2-Loader. Der im Setup zusätzlich vorkommende Prism-MLX-Fork
in `.venv` betrifft den älteren 1-Bit-Pfad.

`mlx_generate_bonsai2.py:40–61` verlangt `model_type=prism_hadamard_qwen35`,
setzt das Standardgerät auf `mx.gpu` und ruft `mlx_vlm.load()` auf.
Es lädt dabei nicht die vorhandene PQ2_0-GGUF-Datei, sondern den eigenen Pack
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`.

Die native Implementierung in
`mlx_vlm/models/prism_hadamard_qwen35/prism_hadamard_qwen35.py`:

1. Erbt das Qwen3.5-Modell und ersetzt die im Pack ausgewiesenen Linear- und
   Embedding-Module durch Hadamard-fähige quantisierte Module.
2. Wendet bei rotierten Linear-Modulen Vorzeichenvektoren und eine normierte,
   blockweise Walsh-Hadamard-Transformation auf die Eingabe an. Die Rotation
   rechnet in FP32 und konvertiert zurück zum ursprünglichen Aktivierungs-Datentyp.
3. Ruft `mx.quantized_matmul` mit 2 Bit und Gruppengröße 128 auf. Gewichte
   bleiben gepackt; dies ist keine vollständige FP16-Dekomprimierung des Modells.
4. Dekomprimiert bei Embeddings nur die ausgewählten Zeilen und transformiert
   sie gegebenenfalls invers zurück.

Damit existiert ein echter llama.cpp-unabhängiger Inferenzpfad. Er verwendet
MLX-Operationen und wird von der Demo für Apple Silicon angeboten; er ist kein
fertiger alternativer CUDA-Server für N04 und kein austauschbarer GGUF-Loader.

## Konkreter Unterschied zum eigenen CUDA-Build: Flash Attention

Die Demo startet `llama-server` in `start_llama_server.sh:195` mit `-fa on`.
Ihr CUDA-Bauskript lässt die Flash-Attention-Kompilierung auf dem Prism-Standard
`GGML_CUDA_FA=ON` (`ggml/CMakeLists.txt:210`).

Bei der ursprünglichen Prüfung setzten unsere Dockerfiles für CUDA 12 und 13
standardmäßig `ARG GGML_CUDA_FA=OFF`. Im CUDA-11-Dockerfile wurde ebenfalls
`-DGGML_CUDA_FA=OFF` gesetzt. Der RTX-Build wurde daraufhin auf
`GGML_CUDA_FA=ON` umgestellt; die Bonsai-CI-Matrix übergibt für CUDA 13
zusätzlich explizit `ON`. Die Legacy-Defaults bleiben unverändert.

Im tatsächlich gepinnten Prism-Quellstand ergibt sich folgende Kette:

- `ggml/src/ggml-cuda/CMakeLists.txt:162`: OFF definiert `GGML_CUDA_NO_FA`.
- `common.cuh:294`: dadurch wird `FLASH_ATTN_AVAILABLE` nicht definiert.
- `fattn.cu:359`: die CUDA-Auswahl liefert `BEST_FATTN_KERNEL_NONE`.
- `fattn.cu:587`: CUDA meldet die Flash-Attention-Operation als nicht unterstützt.
- `src/llama-context.cpp:3880`: quantisierter V-Cache erzwingt gleichzeitig
  Flash Attention, sofern sie nicht ausdrücklich deaktiviert wurde.

Ein Laufzeitflag oder `OLLAMA_FLASH_ATTENTION=true` kann fehlende kompilierte
CUDA-Kernel nicht ergänzen. Der GGML-Scheduler prüft die Unterstützung pro
Operation; deshalb ist CPU-Ausführung einzelner Operationen trotz vollständig
ausgelagerter Modell-Layer möglich.

## Laufzeitnachweis und Umsetzung

Im bisherigen N04-Image `ollama-bonsai:rtx-validation-20260928` meldet der
tatsächliche GGML-CUDA-Backend auf allen vier ursprünglichen RTX-GPUs sowohl
F16- als auch Q4_0-Flash-Attention als nicht unterstützt (acht negative Checks).
Das neue `scripts/check-cuda-flash-attn.py` prüft diese Fähigkeit direkt über
die GGML-C-API mit Tensor-Metadaten, ohne Modellgewichte zu laden.

Ein separater Diagnose-Runner mit demselben alten Image, PQ2_0-Modell, Q4-KV,
`-fa on` und `GGML_SCHED_DEBUG=2` ordnet die `FLASH_ATTN`-Operationen tatsächlich
der CPU zu. Der Trace enthält 336 solche Zuordnungsproben aus Graphaufbau,
Warmup und Ausführung; das sind keine 336 Modell-Layer. Dieser Befund belegt
den CPU-Fallback trotz `65/65` ausgelagerter Modell-Layer. Er misst noch nicht
den Zeitanteil dieser Operationen.

Die kontrollierte Ausgangsmessung mit identischem 128-Token-Decode liegt bei
12,84–13,12 Tokens/s (4096 Kontext) und 13,38–13,49 Tokens/s (262144 Kontext).
Die früheren etwa 1,8 Tokens/s wurden dabei nicht reproduziert und dürfen
nicht als Ausgangswert für den Beschleunigungsfaktor verwendet werden.
Die Messungen reservieren den maximalen Cache, enthalten aber kurze Prompts.

Build und anschließende RTX-Validierung werden in
[BONSAI-RTX-FA-VALIDATION-2026-09-30.md](BONSAI-RTX-FA-VALIDATION-2026-09-30.md)
dokumentiert. Die Produktionsinstanz `11436` gehört nicht zum Testumfang.

## Primärquellen

- [Demo-Startskript](https://github.com/PrismML-Eng/Bonsai-demo/blob/69c3a8beeab80283bfd45cb7b7a6b927075c29fd/scripts/start_llama_server.sh)
- [MLX-Serverstarter](https://github.com/PrismML-Eng/Bonsai-demo/blob/69c3a8beeab80283bfd45cb7b7a6b927075c29fd/scripts/start_mlx_server.sh)
- [MLX-Generierung](https://github.com/PrismML-Eng/Bonsai-demo/blob/69c3a8beeab80283bfd45cb7b7a6b927075c29fd/scripts/mlx_generate_bonsai2.py)
- [Native MLX-Modellimplementierung](https://github.com/Blaizzy/mlx-vlm/blob/a74c7de90a344a2c2c7334acb4e48b57a40480e2/mlx_vlm/models/prism_hadamard_qwen35/prism_hadamard_qwen35.py)
- [Prism CUDA-Flash-Attention-Auswahl](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/fattn.cu)
