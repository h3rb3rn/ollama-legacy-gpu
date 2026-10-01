# Bonsai 2 27B auf Tesla M10/M60 im Ollama-Fork

Stand: 2026-09-28. Ergebnis: **technisch plausibel und portierbar, im untersuchten
Fork aber noch nicht unterstützt und nicht als vollständiges Modell auf Maxwell
validiert.** Ein Backend-Port samt Ollama-Importer-Anpassung ist erforderlich.

## Prüfgrundlage

- Lokales Packaging-Repository: `h3rb3rn/ollama-legacy-gpu`, HEAD `0bd466c`.
- Lokaler Ollama-Sparse-Checkout: `../ollama-src`, HEAD `e434a93`,
  `LLAMA_CPP_VERSION=b9672`. Fehlende Importer-Dateien wurden aus genau diesem
  Git-Stand gelesen. Dies ist keine Feststellung des produktiv laufenden Images.
- Bonsai-demo `main`: beim Abruf `9ef32054fe44797376792c869891163083d64bd0`.
- Untersuchte native Implementierung: Prism-Release `prism-b10709-9a9394a`,
  die in `BACKEND-SUPPORT.md` genannte Release-Basis. Kein beweglicher Branch als
  Implementierungsgrundlage. `prism-v7` zeigte beim Abruf auf
  `c1abda39458458ebfb4ec0722bd2224aab26e680`, wurde aber nicht anstelle der
  Release-Basis getestet.
- Direkt gelesen: erste 16 MiB des veröffentlichten PQ2_0-GGUF; der komplette
  Metadaten- und Tensorverzeichnisbereich endet bei Byte 11.120.982.
- Keine vollständigen Modellgewichte heruntergeladen, keine Inferenz auf einem
  Tesla-Host, keine Änderung laufender Dienste oder produktiver Images.

## Was übernommen werden muss

Das Demo stellt vor allem Installation und Startskripte bereit. Die entscheidende
Implementierung liegt im [PrismML-llama.cpp-Fork](https://github.com/PrismML-Eng/llama.cpp).
Die Modellgewichte sind ternär und zusätzlich in einer rotierten Basis gespeichert.
Die mathematisch passende Aktivierungstransformation gehört zur Inferenz.

Direkter Befund aus dem GGUF:

- Architektur `qwen35`, 64 Schichten, Embeddingbreite 5120, FFN-Breite 17408.
- Volle Attention alle vier Schichten; maximaler deklarierter Kontext 262144.
- 851 Tensoren: 402 vom Typ 142 (`PQ2_0`), 353 F32, 96 BF16.
- `general.file_type=141` — FileType und TensorType haben unterschiedliche IDs.
- `prism.hadamard.version=1`, Blockgröße 1024, normalisierte Sylvester-Walsh-
  Hadamard-Transformation, explizite Vorzeichen.
- 401 transformierte Gewichtsmatrizen; zusätzliche inverse Transformation nach
  dem Lookup von `token_embd.weight`.
- Vorzeichenvektoren für Breiten 5120, 6144 und 17408; insgesamt 28672 Einträge.
- `prism.hadamard.gdn_v_grouped=true`: die Feature-Reihenfolge im Gated-DeltaNet-
  Pfad muss ebenfalls berücksichtigt werden.

Ein einzelner FWHT-Kernel oder die Freischaltung einer Typnummer genügt daher
nicht. Benötigt werden Loader, Graphintegration, Vorzeichen, inverse Embedding-
Transformation und GDN-Anordnung. Die Implementierung liegt insbesondere in
`src/llama-model.{cpp,h}`, `src/llama-graph.{cpp,h}` und den GGML-Backends.
Die Integration muss auch CPU-Offload korrekt unterstützen.

Prism warnt ausdrücklich: Stock-llama.cpp lehnt PQ2_0/PTQ1_0 als unbekannte Typen
ab. Die separate Q2_0-Entwicklungsdatei kann dagegen laden und ohne die
Transformationen unbrauchbaren Text erzeugen. Erfolgreiches Laden ist kein
Korrektheitsnachweis. Siehe [Backend-Matrix](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md).

## Welches „Q2“ gemeint ist

Die Größen stammen aus der Hugging-Face-Datei-API; GB ist dezimal, GiB binär.

| Variante | Gewichtspackung | Dateigröße | Einordnung |
|---|---|---:|---|
| PQ2_0 | 128 Werte in 34 Byte, 2,125 Bit/Gewicht | 7.206.168.928 Byte / 6,71 GiB | Passendes 2-Bit-Ziel im verlinkten Hauptrepository |
| PTQ1_0 | 128 Werte in 28 Byte, 1,75 Bit/Gewicht | 5.946.648.928 Byte / 5,54 GiB | Dieselben ternären Gewichte dichter gepackt; mehr Speicherreserve |
| Q2_0 | 64 Werte in 18 Byte, 2,25 Bit/Gewicht | 7.626.008.928 Byte / 7,10 GiB | Separate Entwicklungsdatei; weiterhin Prism-Transformationen erforderlich |

PQ2_0 ist weder Q2_K noch IQ2. Ein Umbenennen der Typnummer oder ein normales
`ollama create --quantize Q2_K` stellt die Bonsai-Unterstützung nicht her.
Quelle: [Formate](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md),
[Modelldateien](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main),
[Q2_0-Entwicklungsdatei](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf-dev/tree/main).

## Maxwell: unterstützbare Operationen und Engpässe

Buildziele des vorhandenen Forks sind unter anderem `50-real;52-real`.
Der Dockerfile verwendet CUDA 12.0.1. Diese Basis ist für einen Port sinnvoll;
CUDA 13 entfernt die Offline-Kompilierung für Maxwell. Die pauschale Aussage
im bestehenden README, CUDA 12 schließe Maxwell aus, ist nicht zutreffend.
Siehe [NVIDIA CUDA 13 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

Der untersuchte Prism-Code enthält folgende Pfade:

1. `ggml/src/ggml-cuda/fwht.cu`: F32-Arithmetik, Warp-Shuffles und Shared Memory,
   einschließlich Vorzeichen und Breite 1024. Keine zwingenden Tensor Cores.
2. `mmvq.cu`: quantisierter Matrix-Vektor-Pfad auch für PQ2_0/PTQ1_0; regulärer
   Maxwell-Pfad für bis zu acht Ausgabespalten. `common.cuh::ggml_cuda_dp4a`
   enthält einen skalaren Ersatz für Geräte ohne Hardware-DP4A.
3. `mmq.cu::ggml_cuda_should_use_mmq`: der quantisierte Matrix-Matrix-Pfad wird
   unterhalb der DP4A-Architektur verworfen; PTQ1_0-MMQ benötigt auf NVIDIA
   den Turing-MMA-Pfad. Ein bloßes Setzen von FORCE_MMQ beseitigt diese Grenze
   nicht, weil die Architekturprüfung davor liegt.
4. Beim größeren Prompt-Batch folgt deshalb typischerweise
   `ggml-cuda.cu::ggml_cuda_mul_mat_cublas`. Auf Maxwell wird die betroffene
   quantisierte Matrix temporär nach F32 dequantisiert und mit cuBLAS berechnet.
   Die Gewichte bleiben grundsätzlich gepackt; es entsteht ein zusätzlicher
   Arbeitsbuffer pro betreffender Operation.

Damit besteht **kein aus der Kernarithmetik erkennbares grundsätzliches
Maxwell-Verbot**. Es gibt aber auch keinen Nachweis der beworbenen modernen
GPU-Geschwindigkeit. Prompt-Verarbeitung und Decode müssen getrennt gemessen
werden. Keine belastbare tok/s-Prognose ohne Hardwaretest.

Die Standardarchitekturen von `scripts/build_cuda_linux.sh` beginnen bei SM80;
für Maxwell sind explizite Buildziele nötig. Das Skript klont außerdem `prism`,
während die Formatdokumentation diesen Branch der älteren Formatgeneration
zuordnet. Für einen reproduzierbaren Port daher ausdrücklich den geprüften
Release-Commit beziehen, nicht ungeprüft den Standard-Clone verwenden.

## Speicher und sinnvolle Startkonfiguration

6,71 GiB PQ2_0-Dateigröße sind keine Garantie für Betrieb auf einer 8-GiB-GPU.
Zusätzlich entstehen KV-Cache, rekurrenter Zustand, CUDA-Kontext und Compute-
Buffer. Aus der GGUF-Geometrie folgen für den F16-KV-Cache ungefähr 64 KiB pro
Token: 4096 Token rund 256 MiB, 8192 Token rund 512 MiB; zusätzliche Zustände
und Arbeitsbuffer sind darin nicht enthalten.

Eine FFN-Gewichtsmatrix hat 5120 × 17408 Elemente. Ihre F32-Zwischendarstellung
im cuBLAS-Pfad allein beansprucht 340 MiB. Die Output-Matrix ist größer:
5120 × 248320, bei vollständiger F32-Dequantisierung rund 4,74 GiB. Das ist ein
bedingtes Risiko bei vielen gleichzeitig angeforderten Output-Zeilen, kein
pauschaler Zusatzbedarf bei jedem Decode-Schritt. Kleine Output-Batches können
den quantisierten MMVQ-Pfad nehmen. Der normale Embedding-Lookup dequantisiert
ebenfalls nicht die gesamte Embeddingtabelle.

Empfohlener Validierungsbeginn: Text-only, ein paralleler Request, 4096 Kontext,
PQ2_0 zunächst auf zwei freien GPU-Dies mit jeweils etwa 8 GiB. Anschließend
Einzel-GPU-Grenze, Batchgrößen und PTQ1_0 vergleichen. Der Speicher mehrerer Dies
einer M10/M60-Platine ist kein gemeinsamer Speicher; Layerverteilung ist nötig.
Zwei GPUs sind eine Arbeitshypothese mit mehr Reserve, keine hier gemessene
Freigabe. Vision würde weitere Modell- und Compute-Buffer benötigen.

Flash Attention getrennt validieren. Der vorhandene Dockerfile lässt
`GGML_CUDA_FA=OFF` als Default und dokumentiert Probleme großer Multi-GPU-Pools.
Die dortige Single-GPU-Freigabe belegt nicht automatisch Bonsai-Kompatibilität.
Auch der neue Prism-Pfad muss mit und ohne Flash Attention geprüft werden.

## Konkrete Änderungen im Ollama-Fork

| Bereich | Erforderliche Änderung |
|---|---|
| GGML/llama.cpp | PQ2_0-Typtraits, CPU- und CUDA-Operationen sowie vollständigen Hadamard-Modellvertrag übernehmen; PTQ1_0 optional mitführen |
| Ollama `fs/ggml/type.go` | Tensor-IDs 142/143 und FileType-Zuordnungen ergänzen; Q2_0 nur bei bewusst gewünschter zusätzlicher Variante |
| Ollama `fs/ggml/ggml.go` | Richtige Block- und Typgrößen: PQ2_0 128/34, PTQ1_0 128/28; unbekannte Typen nicht als Nullgröße behandeln |
| Import und API | Original-GGUF-Metadaten bewahren; Dateiende, Größe, Modellinformationen und Import kontrollieren |
| Backend-Build | Vollständigen kompatiblen nativen Payload aus demselben Quellstand bauen, einschließlich CPU und CUDA; keine fremde CUDA-.so in den alten GGML-Payload mischen |
| Ollama-Kompatibilität | Hook-Patch auf den gewählten Prism-Stand portieren oder Bonsai-Änderungen auf Ollamas gepinnte Basis zurückportieren |
| Bestehende Fork-Patches | GPU-Auswahl, Layerverteilung, Batch- und Flash-Attention-Anpassungen erneut anwenden und testen |
| Modellverhalten | Chat-Template, Stop-Tokens, Thinking und anschließend Tool-Calling prüfen |

Im geprüften Ollama-Stand sind PQ2_0/PTQ1_0 nicht definiert. Für unbekannte
Tensorarten liefert `TypeSize()` 0 und `BlockSize()` 256; der GGUF-Reader nutzt
`tensor.Size()` zur Ermittlung von Offsets und Dateiende. Der Importer ist somit
ein echter Integrationspunkt, nicht nur eine fehlende Bezeichnung in der UI.

`LLAMA_CPP_FORK` wird im vorhandenen Maxwell-Dockerfile ausdrücklich nicht
verdrahtet. Nur einen anderen Repository-URL als Build-Argument zu setzen reicht
nicht. Zusätzlich wurde der lokale Ollama-Hook-Patch mit `git apply --check`
gegen den Prism-Release geprüft: Er scheitert an `src/llama-model-loader.cpp`
und `tools/mtmd/clip.cpp`. Der komplette Backendtausch ist daher nicht direkt
anwendbar. Änderungen in Ollamas weiterem Architektur-Overlay sind ebenfalls
zu berücksichtigen; dieser Trockentest prüft nur den genannten Hook-Patch.

## Durchgeführte Prüfungen und offene Nachweise

- Repository, Commitstände, bestehende lokale Änderungen und Builddateien gelesen.
- GGUF-Metadaten und alle 851 Tensorverzeichniseinträge direkt geparst.
- Typgrößen, Graphtransformationen, CUDA-Dispatch und F32-Fallback im Quellcode
  nachvollzogen.
- Ollama-Hook-Patch-Trockentest: **fehlgeschlagen**, Konfliktstellen identifiziert.
- CUDA-Kompiliertests: siehe nachfolgenden Ergebnisabschnitt.
- Noch offen: vollständiger CUDA-12.0.1-Build, Linken des Ollama-Payloads,
  GGUF-Importtest, CPU/GPU-Korrektheitsvergleich, echte M10/M60-Inferenz,
  Prompt-/Decode-Benchmark, Speichermaximum und Multi-GPU-Stabilität.

Für die Umsetzung zuerst den nativen Prism-Runner isoliert auf Maxwell
validieren, anschließend dessen Integration in Ollama. Korrektheit anhand
gleicher Prompts und Logits/Perplexität innerhalb definierter numerischer
Toleranzen vergleichen; nur lesbarer Text reicht als Prüfkriterium nicht.
Erst danach eine Freigabe für das produktive Fork-Image erwägen.

## Ergebnis der CUDA-Kompiliertests

Alle sechs Translation-Unit-Kompilierungen waren erfolgreich (Exit 0,
leere Compiler-Logs):

| Quelldatei | sm_50 | sm_52 |
|---|---|---|
| `ggml/src/ggml-cuda/fwht.cu` | bestanden | bestanden |
| `ggml/src/ggml-cuda/convert.cu` | bestanden | bestanden |
| `ggml/src/ggml-cuda/mmvq.cu` | bestanden | bestanden |

Verwendet wurde das bereits lokal vorhandene Image
`nvidia/cuda:11.8.0-devel-ubuntu22.04`, ohne Netzwerk und ohne GPU-Zugriff.
Der Quellbaum war schreibgeschützt eingebunden. Compileraufruf je Kombination:

```sh
nvcc -std=c++17 -arch=sm_50 -Iggml/include -Iggml/src \
  -c ggml/src/ggml-cuda/fwht.cu -o /out/fwht-sm50.o
```

Analog für `sm_52`, `convert.cu` und `mmvq.cu`. Das belegt kompilierbaren
Gerätecode für die ausgewählten Kernoperationen auf beiden Zielarchitekturen.
Es ist ausdrücklich kein vollständiger Backend-Build, kein CUDA-12.0.1-Test
und kein Laufzeit- oder Genauigkeitstest. Die übrigen Translation Units und
das Zusammenspiel mit Qwen35/GDN bleiben zu prüfen.

Temporäre Prüfarbeitsdateien dieser Sitzung: `/tmp/bonsai-prism-audit`,
`/tmp/bonsai-cuda-smoke`, `/tmp/bonsai-gguf-audit.json` und
`/tmp/bonsai-inspect-gguf.py`. SHA-256 des geprüften Release-Quellarchivs:
`bd1fd5c4a5fd554ef8e9168472338f45fe598b94bba8810a233aca89c7d00cd1`.

## Umsetzungs- und Hardware-Nachtrag (2026-09-28)

Die zuvor als offen bezeichnete Umsetzung wurde inzwischen im Ollama-Fork
ergänzt: gepinnte Prism-Quelle, Ollama-Loader-Kompatibilität, PQ2_0/PTQ1_0
GGUF-Typen und Größen, Importhelfer sowie CUDA-12.0.1-Image für `sm_50` und
`sm_52`. Der vollständige Build auf N02-M60 ist erfolgreich. Die GGUF-Parser-
Tests bestanden.

Auf N04-RTX lief Ollama `0.34.1` mit vier Tesla M10 in einer einzelnen
Instanz (`ollama-tesla-bonsai`, Port 11436). Die Modellkopie wurde mit SHA-256
`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`
verifiziert und in den bereits von anderen Containern genutzten Pool
`/opt/ollama/models` importiert. `OLLAMA_SCHED_SPREAD=true` verteilte beim
4096-Kontext alle 65 Modell-Layer auf CUDA0–CUDA3. Eine Chat-Inferenz gab auf
die Frage nach 19×23 korrekt `437` zurück.

Mit `num_ctx=262144` lief ebenfalls eine Inferenz; der Speicherplaner lud dabei
53/65 Layer auf GPU und ließ 12 Layer im CPU-RAM. Nach Reduktion auf
`num_ctx=190000` wurde der Kontext intern auf 190208 Tokens gerundet. Ollama
reservierte dafür 2972 MiB KV je M10; der Modell-Loader meldete 65/65 Layer auf
GPU. Die Antwort war wiederum `437`. Die kurze Antwort mit vier generierten
Tokens maß 1,03 Prompt-Tokens/s und 0,45 Decode-Tokens/s. Ein längerer
Durchsatzlauf bei 190k erzeugte 64 Tokens: 0,20 Prompt-Tokens/s bei 42
Prompt-Tokens und 1,84 Decode-Tokens/s. Der Prompt-Wert schwankte zwischen
den kurzen Läufen deutlich; deshalb sind beide Phasen getrennt aufgeführt.

Das belegt Laden und Textinferenz im Fork sowie die Verteilung auf vier
Maxwell-GPUs, aber noch keinen numerischen CPU/GPU-Logitvergleich oder eine
Vollständigkeitsprüfung des Modellverhaltens (z. B. Tool-Calling). Die
Geschwindigkeit bleibt bei großem Kontext niedrig; dies ist kein
Produktionsleistungsnachweis.

## RTX- und CI-Nachtrag (2026-09-29)

Der RTX-Vollkontextlauf auf N04 bestand mit Q4-KV bei `num_ctx=262144`. Bei
einem kurzen 30-Token-Prompt wurden 13,16 Prompt-Tokens/s und 1,78 Decode-
Tokens/s gemessen; die Anfrage lief vollständig durch. Der KV-Cache belegte
4608 MiB, die Ollama-API meldete rund 11,97 GB VRAM für das Modell. Das prüft
die Reservierung und Stabilität bei maximaler Kontextkapazität, aber keinen
262k langen Prompt.

Die vier der Instanz zugewiesenen RTX-Karten wurden sichtbar, der Fitter lud
jedoch alle 65 Modell-Layer auf eine RTX 3060. Beim Decode lag die GPU-
Auslastung nur bei etwa 15 Prozent, während der N04-Host mit vier vCPUs und
hoher Grundlast lief. Ein separates Qwen3.5-4B-Kontrollmodell erreichte auf
demselben Host ebenfalls nur rund 1,19 Decode-Tokens/s bei etwa 2,5 CPU-Kernen
Runner-Auslastung. Diese Beobachtungen bestimmen die Engpassursache nicht.
CPU-Auslastung kann unter anderem durch CUDA-Aufrufe oder aktives Warten
entstehen; sie beweist weder CPU-Rechenarbeit am Modell noch eine Begrenzung
der GPU durch die CPU. Auch die Nutzung nur einer RTX ist allein kein
Leistungsfehler, wenn Modell und Cache darauf passen.

Korrektur der vorherigen Diagnose: Die nominellen Speicherbandbreiten der
vier M10 dürfen für einen einzelnen Decode-Strom nicht einfach addiert und
mit einer RTX verglichen werden. Bei Layer-Aufteilung werden aufeinander
folgende Modellabschnitte auf unterschiedlichen GPUs ausgeführt; zusätzlich
fallen Transfers und Synchronisation an. Ebenso ist ein PCIe-Engpass ohne
Transfer- und Zeitmessungen nicht nachgewiesen. Die bisherigen Läufe hatten
unterschiedliche Kontextgrößen und sind kein kontrollierter Vergleich.

Der geprüfte Prism-Quellstand hält den Input-Layer ausdrücklich auf der CPU
(`src/llama-model.cpp`); die Meldung `65/65` zählt die auslagerbaren Layer.
Host-Aufrufe für CUDA-Graph-Start und Synchronisation bleiben ebenfalls im
Runner. Das widerlegt eine vollständig CPU-unabhängige Ausführung, erklärt
aber noch nicht die niedrige gemessene Geschwindigkeit. Zur Ursachenbestimmung
fehlt eine zeitlich korrelierte CPU-/CUDA-Profilierung während des Decodes,
die Kernel-Laufzeit, Host-Wartezeit und Speichertransfers unterscheidet.

Die Actions bauen Bonsai nun in eigenen, an Ollama v0.34.1 gebundenen Images
für CUDA 11/K80, CUDA 12/M10-M60 und CUDA 13/RTX. Die bestehenden rollenden
Produktions-Tags bleiben davon getrennt. Das ist eine Workflow- und Dockerfile-
Änderung; die drei neuen Container-Images wurden hier noch nicht gebaut oder
veröffentlicht. `ollama:11436` ist inzwischen eine verifizierte
Produktionsinstanz und wurde während dieser Untersuchung nicht verändert.

## Verifizierte RTX-Korrektur (2026-09-30)

Der nachfolgende Scheduler-Trace weist im alten Image CPU-Ausführung von
`FLASH_ATTN` nach. Mit dem tatsächlich gebauten CUDA-13-Image und
`GGML_CUDA_FA=ON` werden diese Operationen auf CUDA ausgeführt. Bei identischer
4k-Einzelkarten-Arbeitslast steigt der Decode von rund 13 auf 28–29 Tokens/s.
Die früheren 1,78 Tokens/s wurden im kontrollierten Ausgangslauf nicht
reproduziert und sind kein geeigneter Nenner für den Beschleunigungsfaktor.

Bei 262144 Kontextplätzen benötigt GPU-Flash-Attention zusätzliche CUDA-Puffer.
Die bisherige pauschale Platzierungsheuristik unterschätzt diesen Bedarf und
verursacht OOM auf einer 12-GB-Karte. Mit nativer Speicherplanung auf den vier
ursprünglich zugewiesenen RTX-GPUs läuft der neue Fork auf N04:11434 erfolgreich
mit 21,49–21,56 Tokens/s gegenüber 13,38–13,49 im Ausgangslauf. Die Cache-
Kapazität ist maximal reserviert; dies ist kein Benchmark mit 256k belegten
Prompt-Tokens. Details: [RTX-Validierung](BONSAI-RTX-FA-VALIDATION-2026-09-30.md).
