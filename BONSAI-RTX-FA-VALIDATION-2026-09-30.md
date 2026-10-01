# Bonsai RTX: CUDA Flash Attention

## Implementierung und Umgebung

`Dockerfile.cuda13-rtx` kompiliert mit `GGML_CUDA_FA=ON`, einschließlich
quantisierter KV-Cache-Kernel. Die Bonsai-Matrix in GitHub Actions übergibt
diesen Wert explizit für CUDA 13. CUDA-11/12-Defaults bleiben unverändert.
`scripts/check-cuda-flash-attn.py` prüft die Fähigkeit des tatsächlich
geladenen GGML-Backends über dessen C-API.

Der CUDA-13-Build wurde auf N02-M60 erfolgreich erstellt; Inferenz und
Deployment erfolgten auf N04-RTX, Port 11434. Ollama v0.34.1 und Prism
`adfffbe41b2cabcd51fff326ab045662265062bb` sind gepinnt.
Image: `ollama-bonsai:rtx-fa-cuda13-20260929`, Config-Digest
`sha256:0f70cfaf1aba9580de10dae33f20b8580895ac28c17e94232177f9a1f182a9c4`.

Unveränderte Gerätezuweisung: zwei RTX 2060 und zwei RTX 3060 mit je 12 GB,
UUIDs `GPU-ed954d67-584b-f101-ae1e-5cd4df31cebc`,
`GPU-6cf8a010-9ab8-f640-b5d4-b88540b220b7`,
`GPU-6e62055a-46d4-e5e1-41d6-90efc6aa3dbb`,
`GPU-63bfbd4b-9dd7-75ee-e025-d7c8966fcc42`.
Gemeinsamer Mount: `/opt/ollama/models:/root/.ollama`.
Die Produktionsinstanz `ollama-tesla-bonsai:11436` wurde nicht verändert.

## Nachweis der Operationen

| Prüfung | Altes CUDA-12-Image ohne FA | Neues CUDA-13-Image mit FA |
|---|---|---|
| Backend-Fähigkeit F16 und Q4_0 auf vier RTX | 0/8 unterstützt | 8/8 unterstützt |
| `FLASH_ATTN` im Scheduler-Trace | CPU | CUDA0 |
| Anzahl protokollierter Zuordnungsproben | 336 | 336 |

Die Traces verwenden denselben PQ2_0-Blob, dieselbe RTX 3060, 4096 Kontext,
Q4-KV, Batch 128 und `-fa on`. Die 336 Proben stammen aus Graphaufbau,
Warmup und Ausführung, nicht aus 336 Modell-Layern. Die Meldung `65/65`
GPU-Layer allein hätte diesen Unterschied nicht aufgedeckt.

## Durchsatzvergleich

Identischer Prompt: `List the integers from 1 to 200, separated by spaces.
Do not explain.` Temperatur 0, Seed 42, 128 generierte Tokens, Batch 128,
eine Anfrage gleichzeitig. Decode-Rate = Ollama `eval_count / eval_duration`.
Ladezeit und Prompt-Verarbeitung sind nicht Bestandteil dieser Rate.

| Kontextkapazität | Alt, Tokens/s | Neu, Tokens/s | Platzierung |
|---|---:|---:|---|
| 4096, Lauf 1 | 13,12 | 28,45 | jeweils eine RTX 3060 |
| 4096, Lauf 2 | 12,84 | 28,88 | jeweils eine RTX 3060 |

Nach Umstellung auf native Speicherplanung (aktuell deployte Konfiguration):

| Kontextkapazität | Alt, Tokens/s (eine GPU) | Neu, Tokens/s (vier GPUs) |
|---|---:|---:|
| 4096, Lauf 1 | 13,12 | 21,15 |
| 4096, Lauf 2 | 12,84 | 21,76 |
| 262144, Lauf 1 | 13,49 | 21,56 |
| 262144, Lauf 2 | 13,38 | 21,49 |

Bei maximaler Cache-Kapazität beträgt der Gewinn damit rund 60 Prozent.
Die Verteilung auf vier GPUs reduziert den 4k-Decode gegenüber dem neuen
Einzelkartenlauf; der Vergleich stützt keine Addition der GPU-Bandbreiten.
Auch die vier Ausgaben der aktuell deployten Konfiguration sind bytegleich
zu den jeweiligen Ausgangsläufen.

Die beiden 4k-Ausgaben sind bytegleich zum jeweiligen Ausgangslauf. Der neue
4k-Lauf hat zwei Graph-Splits und 10,08 MiB CUDA-Host-Rechenpuffer. Die Änderung
umfasst CUDA-Version und FA-Build gemeinsam; der Geschwindigkeitsfaktor ist
ein Vergleich der Images, keine isolierte Zeitmessung einzelner FA-Kernel.
Die Ausgangsläufe wurden am Vortag aufgenommen. Frühere Werte um 1,8 Tokens/s
wurden bei der kontrollierten Ausgangsmessung nicht reproduziert.

## Befund bei 262144 Kontextplätzen

Der erste Versuch mit der bisherigen Platzierungsheuristik schlug mit CUDA
OOM fehl. Sie erzwingt `OLLAMA_LAYER_OVERHEAD_SCALE=1.1` und legt alle Gewichte
auf eine RTX 3060. Tatsächlich werden 6539,67 MiB Gewichte, 4608 MiB Q4-KV,
149,62 MiB rekurrenter Zustand und bis zu 1313,58 MiB Rechenpuffer benötigt.
Diese Kombination passt nicht auf die gewählte Karte. Der zusätzliche
CUDA-Puffer war im bisherigen CPU-Fallback nicht dort angefallen.

Die anschließende Validierung verwendet daher die native Speicherplanung:
GPU-Autodetektion und erzwungene Greedy-Platzierung sind für diese dedizierte
RTX-Instanz deaktiviert, die vier GPU-UUIDs bleiben explizit zugewiesen.
Die wiederholten 256k-Anfragen liefen erfolgreich durch. Pro Karte wurden
1152 MiB Q4-KV und 1313,58–1323,58 MiB Rechenpuffer reserviert; alle 65 Layer
liegen auf GPU. Der Graph hat sechs Splits. Die reproduzierbare Konfiguration
steht in [docker-compose.n04-bonsai-rtx.yml](compose/docker-compose.n04-bonsai-rtx.yml).

Diese Durchsatzläufe prüfen volle Cache-Kapazität mit kurzen Prompts, keine
262144 bereits belegten Kontextplätze. Ergänzende Stabilitätsläufe mit
derselben Kapazität sind ebenfalls abgeschlossen:

| Anfrage | Prompt-Tokens | Generierte Tokens | Decode Tokens/s | Ergebnis |
|---|---:|---:|---:|---|
| 19 × 23 | 66 | 46 | 23,42 | korrekt: 437 |
| Farb-Padding und Frage nach Frankreichs Hauptstadt | 8268 | 55 | 20,75 | korrekt: Paris |
| Längerer Sortieralgorithmen-Text | 77 | 2048 | 20,80 | Tokenlimit regulär erreicht |

Der 8268-Token-Prompt wurde mit 310,69 Tokens/s verarbeitet. Die lange
Ausgabe dauerte insgesamt 101,36 Sekunden; sie prüft den anhaltenden Decode,
nicht die fachliche Korrektheit eines vollständigen Tutorials. Keine dieser
Anfragen führte zu OOM, Runner-Abbruch oder Container-Neustart. Dies ist ein
begrenzter Stabilitätstest, kein Dauertest mit vollständig belegtem 256k-KV.

Abschließend ist `ollama:11434` gesund und hat null Neustarts. ID, Startzeit
(`2026-09-28T20:19:28.11938137Z`) und Neustartzähler (0) der geschützten
Produktionsinstanz `11436` sind unverändert. Das Modell wurde nach den Tests
entladen und kann bei der nächsten Anfrage erneut geladen werden. Der alte
RTX-Container bleibt gestoppt als `ollama-rtx-pre-fa-20260929` für Rollback.

Die Prüfung der Modellpfade unter `/tmp`, `/opt/ollama` und `/opt/deployment`
auf N04 fand nur die vorhandene 7206168928-Byte-Bonsai-Datei im gemeinsamen
Blob-Store. Für diese Validierung wurden keine Gewichte heruntergeladen
oder kopiert. Auf N02 lagen in den geprüften Pfaden keine Bonsai-GGUF-Dateien.

## Artefakte

Rohdaten und Diagnoseskripte liegen unter
`../.bonsai-work/rtx-fa-20260929/`: `before.jsonl`, `after.jsonl`,
`before-trace-summary.txt`, `after-trace-summary.txt`, `benchmark.py`,
`trace_backend.py`, `deploy.py`, `after-native-fit.jsonl`, `stability.jsonl`
und `native-fit-runtime-evidence.txt`. Die kompletten Scheduler-Traces liegen
auf N04 in `/tmp/bonsai-rtx-fa-20260929/`.
Die CI-Konfiguration ist lokal vorbereitet; sie wurde nicht gepusht oder
als GitHub-Actions-Lauf validiert. Der RTX-Build wurde lokal tatsächlich gebaut.

Die kalten Ladevorgänge der nativen Konfiguration dauern etwa 43 Sekunden.
Das Log enthält weiterhin Warnungen des GPU-Discovery-Watchdogs, gefolgt von
erfolgreicher Geräteerkennung, Modellladung und Inferenz. Diese verbleibende
Startverzögerung ist durch die FA-Änderung nicht behoben.
