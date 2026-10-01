# Warum PQ2_0 nicht proportional zur Dateigröße beschleunigt

Untersuchter Runtime-Stand: Prism `adfffbe41b2cabcd51fff326ab045662265062bb`,
Ollama v0.34.1, CUDA 13, N04 RTX-Testinstanz 11434. Produktionsinstanz 11436
wird nicht verändert. Die vorhandenen Modelltags bestätigen
`bonsai2:27b-pq2_0` als PQ2_0 und `qwen3.8:27b` als Q4_K_M.

## Qwen-Vergleich auf derselben Instanz

Das bereits vorhandene `qwen3.8:27b` wurde auf dem neuen Fork gemessen:
derselbe Zahlenlisten-Prompt, 262144 Kontextplätze, Q4_0-K/V-Cache, Batch 128,
Temperatur 0, Seed 42, 128 generierte Tokens und dieselben vier RTX-GPUs.
Die bestehenden modellabhängigen Chat-Templates blieben erhalten; Qwen hat
30 Prompt-Tokens, Bonsai 72. Verglichen wird ausschließlich die Decode-Phase.

| Modell | Lauf 1 Tokens/s | Lauf 2 Tokens/s |
|---|---:|---:|
| Qwen3.8:27b Q4_K_M, neuer Vergleich | 14,84 | 15,01 |
| Bonsai PQ2_0, vorherige RTX-Validierung | 21,56 | 21,49 |

Der Mittelwert beträgt 14,92 gegenüber 21,52 Tokens/s, rund 44 Prozent Gewinn.
Die Messungen verwenden dieselbe Konfiguration, sind aber zeitlich getrennte
Läufe und keine isolierte Quantisierungs-Ablation. Qwen hat 64 reguläre Layer,
`n_layer_all=65`; der Loader meldet 66/66 GPU-Layer. Qwen reserviert ebenfalls
1152 MiB Q4-KV pro GPU und hat fünf Graph-Splits (Bonsai sechs). Sein Tag lädt
zusätzlich einen Vision-Projektor; der Test enthält keine Bild-Eingaben.
Die längere kalte Ladezeit von Qwen ist nicht in der Decode-Rate enthalten.

Rohdaten: `../.bonsai-work/rtx-fa-20260929/qwen-comparison.jsonl`.
Nach dem Test wurde Qwen entladen. Es wurden keine neuen Gewichte heruntergeladen
oder bestehende Modellprofile verändert.

## Gemessene Platzierungskosten

Bei identischem Bonsai-Prompt, 4096 Kontextplätzen, Batch 128, Seed 42 und
128 generierten Tokens erreicht eine RTX 3060 im neuen Image im Mittel
28,66 Tokens/s. Die Verteilung auf zwei RTX 2060 und zwei RTX 3060 erreicht
21,45 Tokens/s. Das entspricht etwa 34,9 gegenüber 46,6 ms pro Token bzw.
25 Prozent weniger Durchsatz. Es ist der gemeinsame Effekt aus GPU-Auswahl,
Transfers und Synchronisation; die Messung isoliert keinen PCIe-Zeitanteil.

Auf denselben vier GPUs liegen 4096 und 262144 reservierte Kontextplätze
bei kurzen Prompts praktisch gleichauf: 21,45 und 21,52 Tokens/s. Die reine
Cache-Kapazität erklärt daher die aktuelle Decode-Rate nicht. Bei 256k
erzwang der zusätzliche Speicherbedarf jedoch den Wechsel von der vorherigen
Einzelkartenplatzierung zu einer Verteilung. Das war eine funktionierende,
noch nicht auf minimale GPU-Anzahl optimierte Konfiguration.

## Tatsächlicher PQ2-CUDA-Pfad

`ggml/src/ggml-cuda/vecdotq.cuh:978` implementiert
`vec_dot_pq2_0_q8_1`: Der Kernel liest gepackte 2-Bit-Codes, entpackt sie
über `__byte_perm` in Integer-Werte und berechnet die Skalarprodukte mit
`ggml_cuda_dp4a` gegen Q8_1-Aktivierungen. Gruppenskalen werden anschließend
angewendet. Das spart Gewichtstransfers, halbiert aber nicht automatisch die
Anzahl der Rechenschritte. Nullgewichte werden in diesem dichten Pfad nicht
als ausgelassene Matrixelemente behandelt. Die Gewichte bleiben im Speicher
gepackt; es wird kein vollständiges FP16-Modell materialisiert.

Auch Q4-Kernel müssen Gewichte entpacken. Aus diesem Codebefund allein folgt
deshalb weder, dass PQ2 insgesamt langsamer ist, noch wie groß sein Overhead
gegenüber Q4 ist. Der einzelne Decode verwendet hier den Matrix-Vektor-Pfad;
ein niedriges Speicherformat ist kein Nachweis nativer 2-Bit-Tensor-Core-
Arithmetik.

## Zusätzliche Aktivierungstransformationen

`src/llama-graph.cpp:1546` fügt in `build_lora_mm` für die im Modell markierten
Gewichte die passende Vorzeichen-/Hadamard-Transformation vor dem eigentlichen
Matrixprodukt ein. Transformationen derselben Aktivierung werden wiederverwendet.
`ggml/src/ggml-cuda/fwht.cu` führt die Transformation in Float-Arithmetik aus.
Der Graph-Fuser in `ggml-cuda.cu:3478` kann Vorzeichenmultiplikation und
Transformation zusammenfassen; die Transformation ist bereits CUDA-unterstützt.
Es wäre daher falsch, erneut eine generell fehlende Hadamard-Unterstützung
als Ursache zu nennen.

Attention, rekurrente Zustandsaktualisierung, Normalisierung, Aktivierungen
und Synchronisation werden durch halbierte Gewichtsbitbreite nicht ebenfalls
halbiert. Wie viel Laufzeit diese Bestandteile einschließlich Hadamard und
Entpacken konkret belegen, ist noch nicht mit einem CUDA-Zeitprofil gemessen.
Die bisherigen Scheduler-Traces beweisen Backend-Zuordnung, keine Zeitanteile.

## Quellen

- [Gepinnter PQ2-Dot-Product-Kernel](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/vecdotq.cuh#L978)
- [Gepinnter Graphaufbau](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/src/llama-graph.cpp#L1546)
- [CUDA-Hadamard-Kernel](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/fwht.cu)
- [Hersteller-Modellkarte: Repräsentation und Benchmarkbedingungen](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- [Eigene RTX-Messungen](BONSAI-RTX-FA-VALIDATION-2026-09-30.md)
