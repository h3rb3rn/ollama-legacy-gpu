# Tuning und Messwerte (Stand 2026-10-07)

Modell `qwen3.6:35b` (Q4_K_M), Kontext 262144, KV-Cache q4_0, Flash Attention an, `LLAMA_ARG_FIT_TARGET=256`,
Ollama 0.35.1 mit den Patches dieses Repos. Messungen mit `scripts/bench-throughput.sh`: Decode mit dem Prompt
„Write a long story about a robot.“ (120 Token, `temperature 0`, `seed 1`), Prefill mit ~2500 Wörtern (je Lauf eindeutig),
je zwei Läufe. **Maßgeblich sind die wirksamen Werte aus dem Runner-Log** (`llama_context: n_ctx / n_batch / n_ubatch`),
nicht die Env-Datei und nicht `api/ps`: Stock-Ollama ignoriert die Batch-Variablen und rechnet selbst, `api/ps` meldet den
konfigurierten statt den tatsächlichen Kontext.

## Konfiguration je Instanz

| Einstellung | N04-RTX | N02-M60 | N11-M10 |
|---|---|---|---|
| Karten im Pool | 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 4× Tesla M60 (8 GiB) | 4× Tesla M10 (8 GiB) |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | 64 | 64 |
| Pipeline-Parallelität | an | aus (`LLAMA_PIPELINE_PARALLEL=0`) | aus (`LLAMA_PIPELINE_PARALLEL=0`) |
| Layer / wirksamer Kontext | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode tok/s | 42.3 / 41.6 | 15.0 / 16.0 | 9.3 / 9.3 |
| Prefill tok/s | 1140 / 1172 | 91.4 / 92.9 | 24.2 / 24.1 |
| von Ollama als „nicht im VRAM“ gerechnet | 0 MiB (0,0 %) | 0 MiB (0,0 %) | 0 MiB (0,0 %) |
| Host-Puffer `CUDA_Host compute` | 1029 MiB | 32,8 MiB | 32,8 MiB |

Die Gewichte, das KV und die Compute-Puffer liegen auf allen drei Instanzen im VRAM. Im Host-RAM bleiben nur gepinnte
Staging-Puffer: der Host-Puffer oben, ~1 MiB Output-Puffer und ~25 MiB CPU-Compute-Puffer des Vision-Teils.

## Batch-Größe

Die Batch-Größe bestimmt nur, wie viele Prompt-Token pro Durchlauf verarbeitet werden. Sie beschleunigt den Prefill, der
Decode (ein Token pro Schritt) ist unabhängig davon. Batch-Leiter bei Kontext 262144, je Host mit dem Modell und der GPU-Zahl
vom Messzeitpunkt (N04-RTX: vier GPUs; Modellstand je Host nicht einheitlich):

| Host | Env-Batch | wirksam `-b` | Decode tok/s | Prefill tok/s | Compute-Puffer je GPU | Host-Puffer |
|---|---|---|---|---|---|---|
| N04-RTX (4 GPUs) | 64 | 64 | 40.5 / 40.6 | 624 / 651 | ~650 MiB | – |
| | 512 | 512 | 40.4 / 40.4 (40.9 / 40.8) | 1021 / 1175 (1014 / 1171) | ~1,6 GiB | 1029 MiB |
| | 1024 | 1024 | 41.0 / 40.9 | 887 / 1009 | ~2,7 GiB | 2057 MiB |
| | 2048 | 2048 | 37.6 / 37.6 (nur 3 von 4 GPUs belegt) | 549 / 553 | ~1,75 GiB (3 GPUs) | 1040 MiB |
| N11-M10 | 64 | 64 | 9.0 / 9.0 | 23.9 / 23.8 | ~551 MiB | 32,8 MiB |
| | 128 | 128 | 9.1 / 9.1 | 32.2 / 32.2 | ~590 MiB | 65,3 MiB |
| | 256 | 256 | 9.3 / 9.3 | 40.2 / 40.2 | ~666 MiB | 130,3 MiB |
| | 512 | 512 | 9.2 / 9.1 | 49.1 / 49.2 | ~821 MiB | 260,3 MiB |
| | 1024 / 2048 | **512 (gekappt)** | 9.3 / 9.3 | 49 | ~821 MiB | 260,3 MiB |
| N02-M60 | 64 | 64 | 14.2 / 14.8 | 87 | ~551 MiB | 32,8 MiB |
| | 128 | 128 | 14.2 / 13.9 | 108.5 / 110.9 | ~589 MiB | 65,3 MiB |
| | 256 | 256 | 14.2 / 14.9 | 139.0 / 140.5 | ~666 MiB | 130,3 MiB |
| | 512 | 512 | 13.5 / 12.6 | 150 / 155 | ~821 MiB | 260,3 MiB |
| | 1024 / 2048 | **512 (gekappt)** | 13.8 / 13.4 | 154 | ~821 MiB | 260,3 MiB |

- **N04-RTX:** 512 ist das Optimum. 1024 ist langsamer, 2048 lässt eine GPU leer und fällt im Decode.
- **N11-M10, N02-M60:** höhere Env-Werte werden auf der Kommandozeile des Runners auf `-b 512` gekappt (Ursache nicht untersucht).
  512 passt mit 42/42, kostet aber 260 MiB Host-Puffer, den Ollama als ~1 % CPU anzeigt; bei 64 sind es 33 MiB und 0 %.
  Deshalb laufen die beiden 8-GiB-Instanzen mit 64. Der Prefill ist bei 512 etwa doppelt so schnell wie bei 64.
- **Qualität:** `scripts/compare-batch-quality.py` (feste Prompts, Greedy, frisch geladene Instanz je Lauf) auf N04-RTX: bei
  Batch 64 und 512 sind 6 von 6 Antworten identisch, darunter ein Needle-in-a-haystack-Test mit 18 044 Token Prompt; zwei
  Läufe mit 512 stimmen ebenfalls überein. Nur Turing/Ampere geprüft, nicht Maxwell (andere Flash-Attention-Kernel); 6 Prompts,
  bis zu 200 Token. Rohdaten: `docs/evidence/batch-quality-2026-10-05/`.
- **Wann 512 sich lohnt:** Prompt-Längen aus den Runner-Logs von N04-RTX (756 Requests): Median 210 Token, 75 % unter 1143,
  90 % unter 28 885, Maximum 181 904; 40 % unter 64 Token (kein Unterschied), 21 % mindestens 4096, 7 % mindestens 32 768.
  Die Zeilen tragen keinen Zeitstempel und zählen nur neu berechnete Token (Prompt-Cache-Treffer fehlen).

## Pipeline-Parallelität

Bei mehreren GPUs im Layer-Split schaltet llama.cpp die Pipeline-Parallelität ein, wenn alle Layer ausgelagert sind
(`n_gpu_layers > n_layer_all`) und keine Tensor-Overrides aktiv sind. Der Scheduler hält dann vier Kopien seiner Eingabepuffer
(`sched copies = 4`): der gepinnte Host-Puffer und die Compute-Puffer wachsen.

| Host | Pipeline | Host-Puffer | Compute-Puffer je GPU | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|---|
| N11-M10 (Batch 64) | an | 129,6 MiB (`RssShmem` 146 MiB) | 648 MiB | 9.2 / 9.2 | 24.1 |
| | aus | 32,8 MiB (`RssShmem` 47 MiB) | 551 MiB | 9.3 / 9.3 | 24.2 / 24.1 |
| N04-RTX (Batch 512, 3 GPUs) | an | 1029 MiB | ~1,6 GiB | 42.3 / 41.6 | 1140 / 1172 |
| | aus | 260,3 MiB | 822 MiB | 38.3–39.7 (Mittel 39.3) | 917–932 (Mittel 922) |

Auf Maxwell bringt die Pipeline keinen Vorteil und kostet gepinnten RAM: dort ist sie aus. Auf N04-RTX bringt sie etwa +7 %
Decode und +20 % Prefill gegen 770 MiB mehr Host-Puffer: dort bleibt sie an. Die Wahl trifft `LLAMA_PIPELINE_PARALLEL`.
Mit `OLLAMA_GPU_AUTODETECT=1` (N02-M60) erzeugt der Fit des Forks Tensor-Overrides (Fit-Dauer 22,7 s statt 0,9 s), die die
Pipeline ebenfalls abschalten.

## Host-Puffer

`CUDA_Host compute buffer` ≈ `n_ctx × n_ubatch × 2 Byte` (Attention-Maske), bei Pipeline mal vier:

| n_ctx | n_ubatch | Pipeline | berechnet | gemessen |
|---|---|---|---|---|
| 131 072 | 64 | aus | 16 MiB | 18,9 MiB |
| 131 072 | 256 | aus | 64 MiB | 68,3 MiB |
| 262 144 | 64 | aus | 32 MiB | 32,8 MiB |
| 262 144 | 512 | aus | 256 MiB | 260,3 MiB |
| 262 144 | 64 | an | 128 MiB | 129,6 MiB |
| 262 144 | 512 | an | 1024 MiB | 1029,1 MiB |

Der übrige Prozess-RSS (~0,8–1,0 GiB anonym, ~0,22–0,28 GiB Datei-Mappings der CUDA-Bibliotheken) ist Laufzeit-Overhead,
keine Modelldaten.

## Anzeige „CPU“ in `ollama ps`

`/api/ps` liefert `size` und `size_vram`; der Prozentwert ist `(size − size_vram) / size`. Auf N02-M60 waren das bei Batch 512
286 MiB (1,12 %), bei Batch 64 58 MiB (0,24 %) und mit abgeschalteter Pipeline 0 MiB. Auf N04-RTX meldet Ollama 0 MiB, obwohl
der Runner 1029 MiB gepinnten Host-Puffer anlegt: die Anzeige bildet den Host-Puffer nicht auf jedem Host ab.
Verlässlich ist `CUDA_Host compute buffer` im Runner-Log.

## N04-RTX: drei gegenüber vier GPUs

Gleiches Image, Kontext 262144, Batch 512, Pipeline an, gleiches Skript:

| | 4 GPUs (2 Läufe) | 3 GPUs: 2× RTX 2060 + 1× RTX 3060 (4 Läufe) |
|---|---|---|
| Decode tok/s | 40.4 / 40.4, 40.9 / 40.8 (Mittel 40.6) | 42.7 / 42.2, 42.2 / 42.7, 41.7 / 42.4, 42.1 / 42.3 (Mittel 42.3) |
| Prefill, 2. Messung | 1175 / 1171 | 1177 / 1174 / 1160 |
| VRAM gesamt | 31 198 MiB | 29 443 MiB |
| freier VRAM je GPU | 3,0 / 4,4 / 5,7 / 4,9 GiB | 1,3 / 3,4 / 2,7 GiB |

Mit drei GPUs ist der Decode etwa 4 % höher (eine Pipeline-Stufe weniger; nicht geprüft). Der Kontext ist vorab reserviert und
wächst nicht. Die vierte RTX-3060 ist für andere Aufgaben frei.

## Mehr als 9 GPUs in einer Instanz (N02-M60, 12× Tesla M60)

Die Container bekommen nur die jeweils getesteten Karten (`--gpus device=<UUIDs>` und passendes `CUDA_VISIBLE_DEVICES`); Ollama 0.35.1,
Kontext 262144, q4_0, Batch 64, `LLAMA_PIPELINE_PARALLEL=0`. Ein erster Versuch, der nur `CUDA_VISIBLE_DEVICES` verkleinerte, aber alle
12 Karten in den Container reichte, war ungültig (der Runner sah weiter alle 12) und ist nicht verwertet.

| Image | Modell | GPUs im Container | Ergebnis |
|---|---|---|---|
| `pipefix-20261005` (ohne Patch) | `qwen3.6:35b` | 8 | läuft, 11,2–11,5 tok/s Decode |
| `pipefix-20261005` (ohne Patch) | `qwen3.6:35b` | 9 | läuft, 10,7–10,8 tok/s Decode |
| `pipefix-20261005` (ohne Patch) | `qwen3.6:35b` | 12 | `CUDA error: peer mapping resources exhausted` beim ersten Laden (`cuMemSetAccess` in `ggml_cuda_pool_vmm::alloc`, `ggml-cuda.cu:635`) |
| `vmmpeer-20261006` (mit `patch-llama-vmm-peer-access.py`) | `qwen3.6:35b` | 12 | läuft, 10,4 tok/s |
| `kolibri-20261006` (mit Patch) | Kolibri-1 | 8, 9, 10, 11, 12 | läuft, Werte unten |

Nicht gemessen: 10 und 11 GPUs ohne Patch (Kolibri-1 lädt im Image ohne Kolibri-Patch nicht; mit Qwen nicht wiederholt).

**Ursache:** `libggml-cuda.so` ist gegen NCCL gelinkt (`libnccl.so.2` in `ldd`), dadurch ist `use_peer_access` in `ggml_cuda_pool_vmm::alloc`
immer wahr und `cuMemSetAccess` bekommt Zugriffsdeskriptoren für alle sichtbaren Geräte. CUDA erlaubt höchstens 8 Peers je Mapping, also
ein Gerät plus 8 Peers = 9 GPUs. Der Patch erzwingt den Peer-Zugriff nur bis 8 Geräte; das ist vorsichtig gewählt, denn 9 GPUs laufen auch
ohne Patch. Darüber erhält nur das besitzende Gerät Zugriff.

`qwen3.6:35b` ist auf 12 GPUs mit 10,4 tok/s langsamer als auf dem 4-GPU-Pool (13,9–14,8 tok/s in dieser Messreihe, 15,0–16,0 tok/s im
Dauerbetrieb des Pools): mehr Stufen, kein Gewinn bei einem Modell, das in vier Karten passt.

## Kolibri-1 auf N02-M60

`hf.co/Hob-forge/Kolibri-1-GGUF:Q4_K_M`: Architektur `kolibri1`, 78,1B Parameter (384 Experten, 6 aktiv), 47,5 GB, 51 Layer, natives Kontextfenster
262144 (GGUF `kolibri1.context_length`), vier von fünf Layern mit Sliding Window 513, jeder fünfte mit voller Attention ohne Positionskodierung.
Ohne `patch-llama-kolibri1.py` bricht das Laden mit `unknown model architecture: 'kolibri1'` ab. Image `kolibri-20261006`, Batch 64, q4_0,
`-c 262144`, alle 51 Layer im VRAM, nichts im RAM, Laden 253 s.

**GPU-Zahl** (Decode: 120 Token, `temperature 0`; Prefill: 2500 Wörter; Fit mit Spread, außer letzte Zeile):

| GPUs | VRAM gesamt | höchste GPU | Decode (tok/s) | Prefill (tok/s) |
|---|---|---|---|---|
| 8 | 51,8 GB | 7,5 GB | 11,1 / 10,9 | 68,9 |
| 9 | 52,4 GB | 6,7 GB | 10,8 / 10,7 | 67,5 |
| 10 | 53,1 GB | 6,7 GB | 10,7 / 10,5 | 66,6 |
| 11 | 53,2 GB | 5,8 GB | 10,4 / 10,4 | 65,0 |
| 12 | 53,3 GB | 5,8 GB | 10,5 / 10,4 | 65,0 |
| 12 sichtbar, Greedy-Fill 1.10 (belegt 8) | 51,3 GB | 7,2 GB | 11,1 / 10,9 | 67,4 |

Jede zusätzliche GPU kostet etwa 0,1–0,2 tok/s Decode und 1–2 tok/s Prefill; ein CUDA-Fehler trat in keiner Konfiguration auf.

**GPU-Reduktion** mit `OLLAMA_FORCE_GPU_LAYERS=1` (Greedy-Fill, höchster CUDA-Index zuerst, nur so viele GPUs wie nötig), 12 sichtbar:

| `OLLAMA_LAYER_OVERHEAD_SCALE` | GPUs mit Last | VRAM gesamt | höchste GPU | Ergebnis |
|---|---|---|---|---|
| 1.10 | 8 | 51,3 GB | 7,2 GB | läuft, 11,1 / 10,9 tok/s; im Dauerlauf stabil (siehe unten) |
| 1.03 | 7 | 50,9 GB | 8,1 GB | lädt, dann `CUDA error: out of memory` bei der ersten Anfrage, Neuladen (254 s je Versuch) |

Mit 1.10 liegt die Reserve je Karte bei etwa 0,9 GB; 7 GPUs lassen keinen Platz für die Compute-Puffer einer Anfrage. `OLLAMA_FORCE_GPU_LAYERS`
ersetzt den Fit für jedes Modell der Instanz (andere Modelle nicht mit Greedy-Fill gemessen).

**Kontext-Leiter** (Produktionskonfiguration, Greedy-Fill 8 GPUs, zwei Nadeln bei 25 % und 80 % Tiefe, `temperature 0`, `think true`):

| Prompt-Token | Prefill (tok/s) | Decode (tok/s) | Dauer | Nadeln gefunden | VRAM-Spitze je GPU |
|---|---|---|---|---|---|
| 9 421 | 58,5 | 11,8 | 3 min | 1/1 | – |
| 36 554 | 50,6 | 6,4 | 12,6 min | 2/2 | 7 314 MiB |
| 73 159 | 42,3 | 4,6 | 29 min | 2/2 | 7 314 MiB |
| 147 213 | 31,7 | 3,7 | 78 min | 2/2 | 7 314 MiB |
| 221 383 | 25,2 | 3,0 | 148 min | 2/2 | 7 314 MiB |

Der VRAM wächst mit der Prompt-Länge nicht (KV-Cache und Compute-Puffer sind bei `-c 262144` vorab belegt); die Temperatur blieb ≤ 59 °C.
Decode fällt von 11 auf 3 tok/s, Prefill von 69 auf 25 tok/s. Ein erster Lauf mit „245k“ war ungültig: der Prompt hatte etwa 282k Token, wurde
auf 131 074 Token gekürzt und die erste Nadel ging verloren.

**Dauerlauf** (60 min, Greedy-Fill 8 GPUs, wechselnd Reasoning-Rechnen, Tool-Call und 1500-Wörter-Prompt, nacheinander):
220 Anfragen, 0 Fehler, Decode min / Median / max 10,7 / 12,1 / 17,3 tok/s, längste Anfrage 50,7 s, VRAM je GPU konstant 7 314 MiB (Anfang = Ende),
Temperatur ≤ 55 °C, 0 CUDA-Fehler im Runner-Log. Xid-Meldungen des Treibers ließen sich nicht prüfen (`dmesg` und Kernel-Journal sind für
den Benutzer nicht lesbar).

Funktionsprüfung (Image `kolibri-20261006`): Deutsch fließend; Reasoning-Rechnen richtig (80); Tool-Call `get_weather` mit `{"city": "Heidelberg"}`.
Mit `think: false` landet der Gedankengang im `content` statt in `thinking`; mit `think: true` ist er getrennt (nicht weiter untersucht).

## Weitere Befunde

- **MTP (Speculative Decoding):** auf N04-RTX (4 GPUs) langsamer als ohne: 40.3 tok/s ohne MTP, 38.0 mit `draft_num_predict 2`,
  28.4 mit 4 (Fließtext-Prompt). Auf Maxwell stürzt es ab. `OLLAMA_DRAFT_NUM_PREDICT=0` ist Standard.
- **CUDA Graphs auf Maxwell** (`GGML_CUDA_GRAPHS_LEGACY=1`): +35 % Decode auf N11-M10 (6.4 → 8.6 tok/s, CPU-schwacher i5-3470T,
  Main-Thread bei 100 %), kein Effekt auf N02-M60 (12.4–13.2 gegen 12.6–13.7). Läuft auf N11-M10.
- **Thread-Zahl** (`LLAMA_ARG_THREADS` 4 / 2 / 1) hat bei vollständiger GPU-Auslagerung keinen Effekt (N11-M10, 8.95 tok/s).
- **Stresstest** (8025 Token Prompt, 1500 Token Decode, Batch 512): N11-M10 Prefill 45.7 / Decode 7.08 tok/s, N02-M60 148.6 / 9.32
  tok/s; beide ohne Fehler, VRAM danach unverändert.
- **Stock-Ollama 0.35.0 auf N04-RTX** (4 GPUs, Kontext 262144, Env-Batch 64): wirksam `-b 2048`, `41/42` Layer, die dritte
  GPU praktisch ungenutzt; 31.1–31.3 tok/s Decode, 277–281 tok/s Prefill. Der Fork lädt `42/42` auf alle GPUs.
- **Ohne `LLAMA_ARG_FIT_TARGET=256`** (Standard-Fit-Ziel) weicht `ffn_down_exps` (210,82 MiB) in den Host-Speicher aus.

## Grenzen

- Je Konfiguration wenige Läufe, keine Streuungsanalyse; Messungen auf Hosts mit anderer Hintergrundlast (ComfyUI, weitere
  Ollama-Container) können abweichen.
- Prefill und Decode bei sehr langem Kontext (> 32k), andere Modelle und parallele Requests sind nicht gemessen.
- Die Ursache der restlichen CPU-Bindung des Main-Threads auf Maxwell ist nicht vollständig geklärt (kein Profiling).
