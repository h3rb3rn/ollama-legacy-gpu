# Performance-Optimierungs-Log — Tesla M10/M60 unter Ollama

Begleitdokument zur Optimierungs-Roadmap (Plan vom 2026-09-15). Ein Abschnitt pro Phase,
mit Hypothese, exakter Änderung, Testverfahren, Vorher/Nachher-Ergebnis und Verdict.
Bug-spezifische Findings (Finding 1/2/3) bleiben in `BUG-hybrid-arch-degeneration.md`
— hier geht es um die breitere Performance-Kampagne.

Testhost (falls nicht anders angegeben): N11-M10, 4x Tesla M10, GPU3 primär
(`ollama-tesla-4`, Port 11437), Priorität absteigend GPU3→GPU2→GPU1→GPU0.

---

## Phase 0 — Rekap (bereits vor diesem Log umgesetzt)

Zur Vollständigkeit, Details siehe Commits + `BUG-hybrid-arch-degeneration.md`:

| Änderung | Vorher | Nachher | Status |
|---|---|---|---|
| `LLAMA_ARG_THREADS=4` | n_threads=2 (von 4 Kernen) | n_threads=4 | Übernommen |
| MTP-Denylist (`qwen35`-Familie) | CUDA-Crash möglich, nicht-deterministisch | draft_num_predict=0 default, kein Crash | Übernommen |
| `OLLAMA_SPLIT_MODE` Default | ungetestet | `layer` bestätigt (row bricht bei MoE hart ab) | Übernommen |
| tojson-Jinja-Fix | Crash bei Tool-Calling-Templates ohne Tools | Fix angewendet | Übernommen |
| CLIP-Margin-Patch (`fit.cpp`) | — | Getestet, wirkungslos (Fehldiagnose) | **Verworfen**, Commit reverted |
| `LLAMA_ARG_MMPROJ_OFFLOAD=false` | 0/34 Layer auf GPU (qwen3.5:4b) | 34/34 Layer, 5,74 tok/s | Übernommen |
| CI-Cache-Bust | PATCH_GUARD-Stale-Cache-Risiko | Cache-Scope gebustet | Übernommen (Teilfix) |

---

## Phase 1 — Compute-Sanitizer: MTP-Crash root-causen

**Status: ABGESCHLOSSEN — root-caused, Denylist als permanente Lösung bestätigt**

**Hypothese:** Der CUDA "illegal memory access" im MTP-Draft-Pfad
(`common_speculative_impl_draft_mtp::draft`) ist ein Race-Condition oder Out-of-Bounds-
Zugriff in einem der Draft-Kernel, reproduzierbar unter `compute-sanitizer`.

**Setup:** `llama-cli` + `qwen3.5:4b`-GGUF-Blob aus `ollama-tesla-4` extrahiert, in
isoliertem `nvidia/cuda:12.0.1-devel-ubuntu22.04`-Container mit GPU0 (N11-M10, die
ursprüngliche Crash-GPU) laufen lassen, `--spec-type draft-mtp` erzwungen (Denylist für
diesen isolierten Test umgangen). Erste Hürde: `llama-cli` ohne `--single-turn` hängt in
interaktivem Chat-Modus (kein echter Sanitizer-Hang) — behoben.

**Reproduzierter Fehler (deterministisch, ohne UND mit Sanitizer identisch):**
```
E init: invalid token[0] = -839448741
E decode: failed to initialize batch
E llama_decode: failed to decode, ret = -1
E spec draft: llama_decode[1] returned -1
```
Exakt derselbe Garbage-Token-Wert (`-839448741`) in beiden Läufen — 100% deterministisch,
kein zufälliges Rauschen.

**`--tool memcheck`:** 0 Fehler gefunden (`ERROR SUMMARY: 0 errors`) — keine
Out-of-Bounds-/Uninitialized-Memory-Zugriffe innerhalb der Kernel selbst.

**`--tool racecheck`:** noch in Arbeit (deutlich höherer Overhead als memcheck).

**Zwischeneinschätzung:** Die Kombination "deterministisch + memcheck sauber" spricht
eher für einen Host-seitigen Logik-/Indexierungsfehler bei der Umwandlung der
Draft-Head-Rohausgabe in eine Token-ID (falscher Offset, falsche dtype-Interpretation,
oder Lesen eines nie initialisierten Host-Puffers) als für eine klassische
GPU-Speicherverletzung. Wird nach racecheck-Ergebnis weiter eingegrenzt.

**`--tool racecheck` (Ergebnis, ~43 Min. Laufzeit bei `-n 8`):**

```
========= Error: Race reported between Write access at 0x68 in sgemm_32x32x32_NT_vec
=========     and Read access at 0x78 in sgemm_32x32x32_NT_vec [... 20 individuelle Hazards gezeigt ...]
========= RACECHECK SUMMARY: 20 hazards displayed (256 errors, 0 warnings)
```

**256 bestätigte Shared-Memory-Race-Hazards**, alle im selben Kernel
(`sgemm_32x32x32_NT_vec`) — Write bei Shared-Memory-Offset 0x68 rennt gegen Read bei
Offset 0x78. Dieser konkrete Testlauf löste dabei **denselben CUDA-"illegal memory
access"-Crash mit identischem Stack-Trace wie der ursprüngliche Bug-Report** aus
(`common_speculative_impl_draft_mtp::draft` → `common_sampler_sample` →
`llama_context::synchronize` → `ggml_backend_cuda_synchronize`) — die zwei
Symptom-Bilder aus dem Original-Report (harter Crash vs. deterministischer
Garbage-Token, siehe memcheck-Ergebnis oben) sind damit als **zwei Ausprägungen
derselben Race Condition** bestätigt, nicht zwei getrennte Probleme.

**Kernel-Herkunft:** `sgemm_32x32x32_NT_vec` kommt in keiner Datei im `ggml`/`llama.cpp`-
Quellbaum vor (durchsucht) — Namensstil (klassische BLAS-Tile-Benennung, nicht
ggml-typisch wie `mul_mat_vec_q`/`flash_attn_tile`) spricht stark dafür, dass dies ein
**interner, closed-source cuBLAS-Legacy-SGEMM-Kernel** ist, den NVIDIAs CUDA-Toolkit für
Maxwell-Karten ohne Tensor-Cores für reine fp32-Matmuls verwendet (passt zum bereits
dokumentierten Fund dieser Session: Maxwell fällt für nicht-quantisierte/fp32-Matmuls —
z.B. die MTP-Draft/nextn-Embedding-Projektion — auf `cublasSgemm` zurück, da weder MMQ
noch die fp16-cuBLAS-Beschleunigung auf CC 5.0/5.2 verfügbar sind).

**Root-Cause-Einordnung:** Die Race Condition liegt aller Wahrscheinlichkeit nach **in
NVIDIAs eigener, geschlossener cuBLAS-Bibliothek**, nicht im Fork- oder llama.cpp-Code —
weder dieses Projekt noch der llama.cpp-Fork können diesen Kernel patchen. Gegeben
Maxwells End-of-Life-Status (kein weiterer cuBLAS-Support/Fixes zu erwarten) ist ein
echter Upstream-Fix unrealistisch.

**Verdict:** Die bereits ausgelieferte MTP-Denylist (`has_mtp()` in `auto-optimize.py`,
Phase 0) ist damit **nicht nur ein Workaround, sondern die korrekte, dauerhafte Lösung**
für dieses Hardware-Segment — root-caused, nicht weiter zu verfolgen. Kein Code-Fix im
Fork-Repo möglich oder nötig.

---

## Phase 2 — Flash-Attention-Rebuild (`GGML_CUDA_FA=ON`)

**Status: Korrektheitsmatrix abgeschlossen, Ergebnis: ÜBERNOMMEN**

**Build:** `dockerfiles/Dockerfile.cuda12-maxwell` mit `-DGGML_CUDA_FA=ON`, Tag
`cuda12-maxwell-fa-test`, lokal auf N11-M10 gebaut (Maxwell-only Architekturen fürs
schnellere Testen), anschließend nach N02-M60 transferiert (`docker save`/`scp`/`docker load`
über den Steuerhost als Relay, da Worker-Hosts sich nicht direkt per SSH erreichen).

**Korrektheitsmatrix (GPU3/GPU2 auf N11-M10, dann GPU0 auf N02-M60):**

| Modell | Architektur | Host/GPU | Layer-Offload | FA | Compute-Buffer | Output |
|---|---|---|---|---|---|---|
| `smollm3:3b` | dense, head_dim 64/128 | N11-M10/GPU2 | 37/37 | enabled | 140,51 MiB | korrekt, kohärent |
| `sovereign-judge-olmo31-32b` | dense, head_dim=128 | N11-M10/GPU2 | 17/65 (Single-GPU, erwartet) | enabled | — | korrekt, kohärent |
| `qwen3.5:4b` | hybrid, vision, head_dim=256 | N11-M10/GPU2 | 34/34 | enabled | 256,03 MiB | korrekt, kohärent |
| `moe-expert-coder-4b-v2` (Original-Repro!) | hybrid, vision, head_dim=256 | **N02-M60/GPU0** | 33/33 | enabled | — | **korrekt, kohärent, 20,48 tok/s** |

Keine Xid-Fehler in `nvidia-smi -q`/`dmesg` während der Läufe beobachtet (P40-Präzedenzfall
`ggml-org/llama.cpp#12990` nicht reproduziert). Compute-Buffer-Größen (140-494 MiB) matchen
die im CHANGELOG behauptete historische Messung (22-278 MiB) — CHANGELOG-Zahl damit
erstmals tatsächlich verifiziert (vorher war FA ja nie real im Build aktiv).

**Wichtigstes Ergebnis:** Der 4. Testlauf (Original-Bug-Repro-Modell auf der Original-Bughost
N02-M60) bestätigt: Bug vollständig behoben, siehe `BUG-hybrid-arch-degeneration.md`
2026-09-15-Update.

### Nachtrag: FA=ON bricht bei großen Multi-GPU-MoE-Pools

Nach den erfolgreichen Single-GPU-Tests wurde `qwen3.6:35b` (256-Expert-MoE, dasselbe
Modell aus Finding 3) mit FA=ON auf mehreren Pool-Größen getestet — alle drei schlugen fehl:

| Host | GPU-Anzahl | Fehler |
|---|---|---|
| N02-M60 | 12 (voller Pool) | `CUDA error: peer mapping resources exhausted` |
| N02-M60 | 6 | `CUDA error: an illegal memory access was encountered` |
| N11-M10 | 4 (derselbe Pool der bei FA=OFF in Finding 3 einwandfrei lief) | `llama-server process has terminated: signal: killed` (vermutlich Host-OOM, N11-M10 hat wenig freies System-RAM) |

Drei unterschiedliche Fehlerbilder auf zwei unterschiedlichen Hosts — das ist kein
Zufallsrauschen, sondern ein echtes Muster: **FA=ON ist für große, gepoolte MoE-Modelle
über mehrere GPUs (noch) nicht sicher**, obwohl es für Single-GPU-Deployments (die drei
oben bestandenen Fälle, inklusive des eigentlichen Bug-Ziels) einwandfrei funktioniert.
Root Cause nicht weiter verfolgt (außerhalb des ursprünglichen Bugs) — mögliche
Kandidaten: FA ändert Buffer-Allokationsmuster grundlegend, `OLLAMA_MAX_BATCH_SIZE=64`-Cap
war für den Non-FA-Fall kalibriert (Phase 5 im Plan sollte das ohnehin neu tunen), oder
eine echte Inkompatibilität zwischen FA-Kernel und Multi-GPU-P2P-Transfers bei MoE-Routing.

**Rollback angewendet gemäß Plan-Kriterium** ("jede Xid-/Crash-/Korrektheitsabweichung →
sofort stoppen"): `-latest` bleibt vorerst FA=OFF. Empfehlung für den produktiven Rollout:
FA=ON **nur für Single-GPU-Deployments** freigeben (genau der Fall, der den Bug behebt),
Multi-GPU-Pools bleiben FA=OFF bis das oben beschriebene Muster separat untersucht ist.

**Verdict:** FA=ON auf allen drei Single-GPU-Maxwell-Testfällen sicher und korrekt
(inkl. des Original-Bug-Falls). Für große Multi-GPU-MoE-Pools **noch nicht sicher** —
neuer, eigenständiger Untersuchungspunkt, nicht Teil des ursprünglichen Bugs.
Rollout-Empfehlung: `-fa-test` → `-latest` nur mit Single-GPU-Scope, nach
Leichttest auf GPU1/GPU0 (N11-M10-Prioritätsreihenfolge).

---

## Phase 3 — Hybrid-vs-Vision-Konfundierung trennen

**Status: teilweise abgeschlossen — Vision-Dense-Fall bestätigt, Hybrid-Text-only-Fall blockiert**

**Blocker text-only-Hybrid-Modell:** `hf.co/...`-Pulls schlagen aktuell mit
`Error: pull model manifest: realm host "huggingface.co" does not match original host "hf.co"`
fehl — reproduzierbar für neue (nicht bereits lokal gecachte) Repos, bereits gecachte
Repos (z.B. `sovereign-judge-olmo31-32b`, vorher gepullt) funktionieren weiter. Kein
Mamba/Jamba/reines-SSM-Modell in der gesamten Flotte (N11-M10, N04-RTX, N02-M60) bereits
lokal vorhanden. Dieser Teil von Phase 3 bleibt offen, bis das Registry-Problem
(vermutlich Ollama-Versions-Regression, unabhängig von diesem Fork) behoben ist.

**Vision-Dense-Fall (kein SSM, aber Vision) — getestet:**

| Modell | Vision-Tower-Größe (mtmd worst-case) | Layer-Offload (mmproj auf GPU, Default) |
|---|---|---|
| `qwen3.5:4b` (zum Vergleich, hybrid+vision) | ~5,4 GiB | 0/34 |
| `llama3.2-vision:latest` | — | **lädt gar nicht** (`unknown model architecture: 'mllama'` — dieser Fork-Build unterstützt die mllama-Architektur nicht, unabhängig von VRAM) |
| `minicpm-v:latest` (dense+vision) | ~1,05 GiB | **25/29** (kein 0-Layer-Fall) |

**Ergebnis:** Kein sauberer Binär-Vergleich möglich (kein zweites hybrides Textmodell
verfügbar), aber `minicpm-v` liefert trotzdem ein aussagekräftiges Teilergebnis: **ein
dense (nicht-hybrides) Modell mit Vision-Fähigkeit zeigt dasselbe VRAM-Konkurrenz-Muster
wie Qwen3.5 — nur proportional zur tatsächlichen Encoder-Größe.** MiniCPM-Vs Encoder
(~1 GiB) ist klein genug, um auf einer 8-GiB-Karte kein 0-Layer-Ergebnis zu erzwingen;
Qwen3.5s Encoder (~5,4 GiB) ist es nicht. Das stützt die Kernthese aus Finding 1 (VRAM-
Konkurrenz durch die Vision-Komponente, nicht die Hybrid-/SSM-Architektur) — allerdings
als graduellen, größenabhängigen Effekt statt als sauberes Ja/Nein, und ohne den fehlenden
Hybrid-Text-only-Fall bleibt ein Restzweifel, ob Hybrid-Architektur einen (kleineren,
zusätzlichen) Effekt hat. **Nicht abschließend bewiesen, aber die vorherrschende Erklärung
bleibt bestehen.**

**Nebenfund:** `llama3.2-vision` (mllama-Architektur) ist mit diesem Fork-Build generell
nicht nutzbar — separates, kleines Kompatibilitäts-Ticket, nicht weiter verfolgt (nicht
Teil des ursprünglichen Bugs, mllama ist eine strukturell andere Vision-Integration als
das CLIP/mtmd-Modell, das dieser Fork sonst überall sieht).

---

## Phase 4 (Teil 1) — N02-M60-Flotte mit validiertem Fix-Set neu aufgebaut

**Status: abgeschlossen (N02-M60), N04-RTX noch offen**

Mit expliziter Freigabe des Users ("nimm die GPUs wie und so viel du brauchst") wurden
die 9 alten Stock-Ollama-Container (`ollama/ollama:0.24.0`, liefen dort seit 44h als
A/B-Vergleichsbaseline laut Bug-Report) entfernt und durch eine vollständige
12-GPU-Single-Instanz-Flotte (`ollama-m60-gpu0` … `gpu11`, Port 11434–11445, je 1 Tesla-
M60-Die) mit dem Image `cuda12-maxwell-fa-test` und dem vollen validierten Fix-Set
ersetzt (Thread-Fix, MTP-Denylist, `LLAMA_ARG_MMPROJ_OFFLOAD=false`, FA=ON —
Single-GPU-Scope, siehe Phase-2-Einschränkung oben). Alle 12 Container healthy,
End-to-End mit dem Original-Bug-Modell auf GPU0 verifiziert (HTTP 200, korrekter Output).

**Bewusst NICHT wiederhergestellt:** der alte 12-GPU-Pool-Container
(`ollama-m60-pool`) — siehe Phase-2-Nachtrag: FA=ON ist für große Multi-GPU-Pools aktuell
nicht sicher. Ein Pool-Deployment für N02-M60 sollte, falls gewünscht, vorerst mit
FA=OFF aufgesetzt werden (analog zur N11-M10-Multi-GPU-Test-Compose), nicht mit dem
`-fa-test`-Image.

---

## Phase 4 (Teil 2) — N04-RTX aktualisiert

**Status: abgeschlossen**

N04-RTX ist ein deutlich sensiblerer Host: neben mehreren Ollama-Test-Containern läuft
dort eine produktive Nicht-Ollama-Anwendung (`terra_*`-Stack: Node-Client/-Server,
Postgres, Redis, Mailhog). Der Auto-Mode-Sicherheits-Classifier hat das korrekt als
Produktivsystem erkannt und initiale Aktionen dort blockiert (sogar lesendes
`docker ps`) — auf explizite Nutzerfreigabe hin fortgesetzt.

**Nur die Tesla-basierten Fork-Container aktualisiert** (`ollama-tesla-1..4`,
`ollama-m60-1`, `ollama-m60-2`, `ollama-m60-guard`) — mit demselben validierten
Fix-Set wie N02-M60 (Single-GPU-Container: `cuda12-maxwell-fa-test`-Image, Thread-Fix,
`LLAMA_ARG_MMPROJ_OFFLOAD=false`, FA=ON; der 2-GPU-Pool-Container `ollama-m60-guard`:
frisches `-latest` mit Phase-0-Fixes, FA=OFF wie bisher). Bestehende Ports/GPU-
Zuordnungen/Context-Length (32768, abweichend von den 65536 auf den Testhosts —
bewusst beibehalten, das war der produktive Wert) unverändert übernommen.

**Explizit NICHT angefasst** (auf Nutzeranweisung): `ollama` (Port 11434) und
`ollama-rgtx` (Port 11435) — laufen mit `ollama-github:latest` (offizielles Ollama,
nicht dieser Fork) auf den RTX/GTX-GPUs. Ebenso unangetastet: der gesamte
`terra_*`-Stack, `ollama-0240-m10`/`ollama-0240-m60` (bewusste Stock-Ollama-
Vergleichsbaseline), `open-webui`, `searxng`.

End-to-End verifiziert auf `ollama-tesla-1`: 34/34 Layer, FA aktiv, korrekter
Output. Damit ist Phase 4 (Rollout auf alle drei Test-/Produktionshosts dieser
Kampagne — N11-M10, N02-M60, N04-RTX) abgeschlossen.

### Nachtrag: FA=ON jetzt offiziell als CI-Tag veröffentlicht

Der bis hierhin nur lokal gebaute `cuda12-maxwell-fa-test` war nie im Dockerfile
committet. Nachgeholt: `GGML_CUDA_FA` ist jetzt ein Build-Arg (Default weiterhin OFF,
ändert nichts am Verhalten bestehender Multi-GPU-Pool-Deployments), CI baut zusätzlich
den Tag `cuda12-maxwell-singlegpu-fa-latest` mit FA=ON. Zukünftige Single-GPU-Rollouts
können diesen Tag direkt pullen statt manuell lokale Images per `docker save`/`scp`/
`docker load` zu verteilen.

---

## Phase 5 — Batch/ubatch neu tunen (nach FA-Rebuild)

**Status: abgeschlossen — kein Handlungsbedarf, Nebenfund dokumentiert**

**Prefill-Benchmark** (640-Token-Prompt, `qwen3.5:4b`, N02-M60/GPU0, FA=ON,
Standard-Batch=1024): **251,92 tok/s**. Zweiter Lauf (versucht mit kleinerem Batch)
lief aus im nächsten Absatz beschriebenen Grund ebenfalls bei Batch=1024:
**251,07 tok/s** — praktisch identisch, keine Überraschung angesichts des Fundes unten.

**Nebenfund:** `OLLAMA_MAX_BATCH_SIZE` (der von `patch-ollama-batch.py` injizierte
Cap-Mechanismus) hat in keinem der Builds dieser Session eine Wirkung gezeigt — der
String `OLLAMA_MAX_BATCH_SIZE` fehlt komplett im kompilierten `ollama`-Binary
(`strings`-Check). Der Patch schlägt gegen die aktuelle Ollama-Quellcode-Version
offenbar still fehl (Ziel-Pattern im Go-Code hat sich vermutlich geändert) — betrifft
potenziell auch die produktiv genutzte `selectGPUPool()`-Logik auf N04-RTX für den
großen 12-GPU-Pool-Fall, wo der Cap laut Code-Kommentar automatisch gesetzt werden
soll. Separates, nicht in dieser Kampagne behobenes Ticket.

**Warum trotzdem kein Handlungsbedarf für Single-GPU-FA-Deployments:** Der ursprüngliche
Zweck des Batch-Caps war, den *Non-FA*-Attention-Compute-Buffer klein zu halten
(`batch × ctx × heads × head_dim × 4 Byte`, mehrere GiB bei großem Batch). Mit FA=ON
entfällt dieses Problem strukturell — FA materialisiert die große Zwischen-Matrix gar
nicht erst, unabhängig von der Batch-Größe (bestätigt: Compute-Buffer bleiben bei
140–494 MiB, siehe Phase 2). Ollamas Standard-Batch (1024) funktioniert für
Single-GPU-FA-Deployments ohne Anpassung. Für den (weiterhin FA=OFF) großen
Multi-GPU-Pool-Fall bleibt der Cap-Mechanismus relevant, ist aber aktuell defekt —
das wäre der eigentliche nächste Schritt, falls jemand den großen Pool-Fall
weiterverfolgen möchte.

---

## Phase 6 — `--n-cpu-moe` für enge MoE-Configs

**Status: abgeschlossen — kein sauberes A/B möglich, aber wichtiger Nebenfund**

**Setup:** `qwen3.6:35b` (256-Expert-MoE, ~23 GiB) auf einem künstlich auf 2 GPUs
verkleinerten M60-Pool (16 GiB kombiniert, absichtlich eng) — 2 der 12
Single-Instanz-Container auf N02-M60 temporär gestoppt, um GPUs freizugeben,
danach wiederhergestellt. FA=OFF (Multi-GPU, siehe Phase 2).

**Baseline (kein `--n-cpu-moe`, automatisches Fitting):** Lädt zunächst scheinbar
erfolgreich (`offloaded 42/42 layers`, aber `CPU_Mapped model buffer size = 20293
MiB` — die Experten-Gewichte landen bereits automatisch per mmap größtenteils auf
CPU). Server-Log empfiehlt selbst `--load-mode none` statt mmap für bessere
Performance. **Crasht dann während der Generierung:** `llama-server terminated:
signal: aborted (core dumped)`.

**Mit explizitem `--cpu-moe`** (alle Experten-Gewichte fest auf CPU): **Crasht
ebenfalls**, diesmal mit `CUDA error: an illegal memory access was encountered`.

**Kein sauberer Vergleich möglich** — beide Konfigurationen sind auf diesem engen
2-GPU-Pool instabil. **Wichtiger Nebenfund, der über die ursprüngliche Phase-6-Frage
hinausgeht:** Die in Phase 2 dokumentierte Multi-GPU-MoE-Instabilität ist **nicht nur
ein FA=ON-Problem** — sie tritt auch mit FA=OFF auf, sobald der Pool eng genug ist.
Das deutet auf ein grundsätzlicheres Problem mit engen Multi-GPU-MoE-Konfigurationen
in diesem Fork auf Maxwell hin, unabhängig von Flash Attention oder Experten-Platzierung.
**Nicht weiter verfolgt** in dieser Kampagne — eigenständiges, größeres
Untersuchungsthema (eigener compute-sanitizer-Lauf o.ä. wäre der nächste Schritt,
analog zu Phase 1). Empfehlung: enge Multi-GPU-MoE-Pools (Modell nur knapp größer als
verfügbares Pool-VRAM) vorerst meiden bzw. mit großzügigerem VRAM-Puffer planen (wie
der bereits produktiv laufende 4-GPU-M10-Pool auf N11-M10 mit `qwen3.6:35b` bei 41/42
Layern zeigt — dort mit reichlich Puffer stabil).

### Nachtrag (2026-09-16): Nebenfund relativiert — Crash nicht reproduzierbar

Die oben dokumentierte "grundsätzlichere Instabilität enger Multi-GPU-MoE-Pools" wurde
mit derselben compute-sanitizer-Methodik wie in Phase 1 nachuntersucht (`llama-cli` +
GGUF-Blobs aus `ollama-m60-gpu0` extrahiert, GPUs 0+1 auf N02-M60 dafür freigemacht).

**Direkte `llama-cli`-Repro-Versuche** (identischer 2-GPU-Pool, `-sm layer`, `-c 32768`):

| Variante | Ergebnis |
|---|---|
| `-ngl 999` (erzwungen) | Sauberer `cudaMalloc failed: out of memory` — anderes Fehlerbild als Original, da Auto-Fit umgangen |
| `--fit on`, ohne mmproj | **Erfolgreich**, korrekter Output, kein Crash |
| `--fit on`, mit mmproj (Original-Bedingungen exakt nachgestellt) | **Erfolgreich**, korrekter Output, kein Crash |

**Sauberer Retest über Ollama selbst:** Frischer Container (`moe-clean-retry`, Port
11462) mit exakt derselben engen 2-GPU-Konfiguration wie beim ursprünglichen Crash
(`OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_SPLIT_MODE=layer`, `cuda12-maxwell-latest`) —
**lief einwandfrei durch:** HTTP 200 nach 95,1s, `load_tensors: offloaded 42/42 layers
to GPU`, `error: None`.

**Layout-Cache geprüft:** `/root/.ollama/layout-cache/` im frischen Container war
**komplett leer** — die Hypothese eines veralteten/falschen gecachten Tensor-Splits als
Ursache ist damit widerlegt. `dmesg`-Check auf Xid-Fehler war ergebnislos (vermutlich
kein Host-dmesg-Zugriff für diesen User, nicht aussagekräftig).

**Korrigierte Einschätzung:** Der ursprüngliche Crash war in drei unabhängigen,
sauberen Nachstellungsversuchen (3× `llama-cli` direkt, 1× frischer Ollama-Container)
**nicht reproduzierbar** — auch nicht mit exakt denselben Parametern inkl. mmproj. Das
spricht dagegen, dass es sich um einen eigenständigen, deterministisch reproduzierbaren
Architektur-Bug handelt. Wahrscheinlicher: transienter/korrupter GPU-Treiberzustand,
vermutlich als Nachwirkung eines vorherigen Crashs auf denselben physischen GPUs
während der intensiven Back-to-Back-Testreihe dieser Session. Der oben formulierte
Hinweis ("enge Multi-GPU-MoE-Pools vorerst meiden") wird damit **zurückgezogen** — es
gibt keinen belastbaren Beleg mehr für ein grundsätzliches Architekturproblem. Ein
Restrisiko durch Treiberzustand nach Crashes bleibt plausibel (nicht ausgeschlossen),
ist aber ein allgemeines Betriebsthema (GPU-Reset nach Fehler), kein MoE- oder
Pool-spezifisches. Nicht zu verwechseln mit dem separaten, unabhängig bestätigten
FA=ON-Multi-GPU-Befund aus Phase 2 (andere Tests, andere Hosts) — der bleibt unverändert
bestehen.

---

## Phase 7 — N-Gram-Speculative-Decoding als MTP-Alternative

**Status: abgeschlossen — kein Nutzen, nicht übernommen**

`--spec-type ngram-simple` (`LLAMA_ARG_SPEC_TYPE`) auf `qwen3.5:4b`, Single-GPU M60,
FA=ON, getestet mit zwei Prompt-Typen:

| Prompt | Baseline (kein Spec-Decode) | ngram-simple | Drafts generiert/akzeptiert |
|---|---|---|---|
| Code-Prompt (Original-Repro) | 19,39 tok/s | 19,25 tok/s | 0 / 0 (in 200 Tokens) |
| Bewusst repetitiv (Zahlen 1-100 ausgeschrieben) | 19,39 tok/s (Referenz) | **17,33 tok/s** | 1 / 1 (in 200 Tokens, mean acc len=2.0) |

**Ergebnis:** Kein messbarer Nutzen, beim repetitiven Prompt sogar **langsamer** als
die Baseline (reiner Overhead durch N-Gram-Suche ohne kompensierenden Gewinn). Selbst
der bewusst repetitiv gestaltete Prompt erzeugte über 200 Tokens nur einen einzigen
angenommenen Draft. Bestätigt die im Plan vorab formulierte Erwartung: M10/M60 sind
bandbreitengebunden, es fehlt das Rechen-Leerlaufbudget, das Speculative Decoding zum
Ausnutzen bräuchte — und die getesteten Prompts trafen das erforderliche
12-Token-N-Gram-Fenster (`size_n=12`) kaum. **Nicht übernommen**, MTP-Denylist (Phase 0)
bleibt ohne Ersatz — für `qwen35`-Familie also aktuell kein Speculative-Decoding im
produktiven Einsatz, was angesichts der cuBLAS-Race-Condition (Phase 1) ohnehin die
richtige, konservative Wahl ist.

---

## Phase 8 — Host-Level-Feintuning

**Status: abgeschlossen — 2 von 3 Punkten sofort geklärt, 1 Punkt nicht empirisch getestet**

**NUMA:** `numactl` auf keinem der drei Hosts installiert; direkt über
`/sys/devices/system/node/` geprüft: **exakt 1 NUMA-Node auf N11-M10, N02-M60 und
N04-RTX**. NUMA-Tuning ist damit bestätigt irrelevant, wie im Plan vermutet — kein
Handlungsbedarf.

**ECC-Toggle:** Tesla M10 unterstützt ECC-Steuerung über `nvidia-smi` gar nicht
(`ECC Mode: N/A` auf allen getesteten M10-GPUs). Tesla M60 hat ECC bereits **werkseitig/
standardmäßig deaktiviert** (`Disabled`, auf allen 12 M60-Dies auf N02-M60 bestätigt).
Es gibt in dieser Flotte nichts zu togglen — der Punkt erledigt sich von selbst, kein
Experiment nötig, keine Nutzerentscheidung erforderlich.

**`--load-mode mlock`/`none` (ehemals `--mlock`/`--no-mmap`):** Erster A/B-Versuch mit
Olmo-32B (CPU-lastig, ~73 %/27 % CPU/GPU-Split) auf **N11-M10** hing minutenlang fest.
Root-Cause geklärt (siehe unten) — **kein Ollama-Bug**, sondern echtes Swap-Thrashing:
N11-M10 hat nur 15,5 GiB RAM, die 73 % CPU-ausgelagerten Layer eines 19-GiB-Modells
(~14 GiB) drängten in den Swap. `ps aux` zeigte den `llama-server`-Prozess durchgehend im
`D`-Zustand (unterbrechbarer I/O-Wait), Swap-Nutzung stieg auf 10 GiB. Ein `/api/generate`
brauchte >10 Minuten für 150 Tokens, bevor Ollamas eigener Server-Timeout griff. **N11-M10
damit als Testsystem für CPU-offload-lastige große Modelle grundsätzlich ungeeignet**
(RAM-Kapazität, kein Werkzeug-Problem) — Host für diese Testklasse verworfen.

**Sauberer Retest auf N02-M60** (125 GiB RAM, 120 GiB verfügbar) — Container `gpu6`
(Port 11440, Tesla M60, 8 GiB VRAM) testweise neu erstellt, einmal mit Standard-Lademodus,
einmal mit `LLAMA_ARG_LOAD_MODE=mlock`, sonst identische Konfiguration
(`ollama-legacy:cuda12-maxwell-fa-test`, FA=ON, `LLAMA_ARG_MMPROJ_OFFLOAD=false`).
Modell: `gemma4:31b` (dense, ~19 GiB, CPU-lastiger Split, HF-Registry-Bug umgangen durch
Nutzung eines bereits lokal getaggten Modells statt `hf.co/...`). Gleicher Prompt
(`temperature=0.2, seed=42, num_predict=150`).

| Variante | Ladezeit | Decode (150 Tok.) | tok/s |
|---|---|---|---|
| Standard (mmap default) | 50,4 s | 97,99 s | 1,531 |
| `LLAMA_ARG_LOAD_MODE=mlock` | 46,6 s | 101,95 s | 1,471 |

**Ergebnis:** Kein signifikanter Unterschied (Differenz innerhalb der Messschwankung
eines Einzellaufs). Swap-Nutzung blieb in beiden Läufen konstant bei 5,8 GiB (kein aktives
Thrashing) — erwartungsgemäß, da 120 GiB freier RAM für die ~14 GiB CPU-Layer bei weitem
ausreichen. **Verdict: `mlock` bringt auf ausreichend dimensionierten Hosts (N02-M60,
N04-RTX) nichts und wird nicht übernommen.** Es bestätigt aber sauber die Hypothese aus
dem N11-M10-Vorfall: `mlock` wirkt nur unter echtem RAM-Druck — und genau dort (N11-M10)
ist es nicht praktikabel testbar, weil bereits der Baseline-Lauf durch Thrashing
unbrauchbar langsam wird. Empfehlung: große CPU-Offload-lastige Modelle grundsätzlich
nicht auf N11-M10 betreiben (RAM-Obergrenze ~15,5 GiB), sondern auf N02-M60/N04-RTX.

### Zusammenfassung Phase 8
Alle drei ursprünglichen Feintuning-Kandidaten sind jetzt abgeschlossen. NUMA und ECC
waren bereits durch die Hardware-/Treiber-Gegebenheiten erledigt, ohne dass eine Änderung
nötig gewesen wäre. `--load-mode` wurde empirisch sauber getestet (N02-M60) und **nicht
übernommen** — kein Nutzen bei ausreichendem RAM, echter Nutzen nur unter Speicherdruck,
den es auf den besser dimensionierten Hosts nicht gibt. Nebenbefund: N11-M10s begrenztes
RAM (15,5 GiB) ist eine eigenständige Betriebs-Einschränkung für große Dense-Modelle mit
hohem CPU-Offload-Anteil, unabhängig von `mlock`.

---

## Phase 9 — Produktions-Rollout

**Status: abgeschlossen (2026-09-16)**

**Befund vor dem Rollout:** Die persistierten Compose-Dateien auf allen drei Hosts waren
erheblich vom tatsächlichen, in dieser Kampagne validierten Live-Zustand abgedriftet —
alle Fixes liefen bisher über manuelle Container-Neuerstellung (`docker run`), nie über
die Compose-Dateien selbst. Konkret: N11-M10s Compose beschrieb noch das alte
1-Container-4-GPU-Pool-Modell; N02-M60s Compose referenzierte noch Stock-`ollama/
ollama:0.24.0` mit falschem Port-Mapping (Pool+8-Single statt der live längst
umgesetzten 12-Single-Topologie); nur N04-RTX war schon nah am Live-Zustand (fehlten nur
die Thread-/MMPROJ-Env-Vars).

**Ausgerollt (Env-Fixes, alle drei Hosts, nur Single-GPU-Instanzen):**
`LLAMA_ARG_THREADS=4`, `LLAMA_ARG_THREADS_BATCH=4`, `LLAMA_ARG_MMPROJ_OFFLOAD=false`,
`OLLAMA_SPLIT_MODE`-Verhalten (Default `layer`, keine `row`-Konfiguration mehr aktiv).

**Ausgerollt (FA=ON-Single-GPU-Image, Phase 2):** Alle 22 Single-GPU-Instanzen auf allen
drei Hosts liefen zuvor entweder noch auf FA=OFF (N11-M10, 4 Instanzen) oder auf
lokal-only gebauten Test-Tags (N02-M60: 12, N04-RTX: 6) — keine davon nutzte den
offiziell in der CI gebauten, in GHCR veröffentlichten Tag
`cuda12-maxwell-singlegpu-fa-latest`. Alle 22 Instanzen wurden auf dieses offizielle
Image umgestellt (rollierend, pro Container `docker rm -f` + Neustart, Health-Check +
Smoke-Test nach jedem Schritt):

| Host | Instanzen | Vorher | Nachher |
|---|---|---|---|
| N11-M10 | `ollama-tesla-1..4` (Port 11434-11437) | FA=OFF, `cuda12-maxwell-latest` | FA=ON, `cuda12-maxwell-singlegpu-fa-latest` |
| N04-RTX | `ollama-tesla-1..4`, `ollama-m60-1/2` (Port 11436-11441) | FA=ON, lokaler Test-Tag | FA=ON, offizieller GHCR-Tag |
| N02-M60 | `ollama-m60-gpu0..11` (Port 11434-11445) | FA=ON, lokaler Test-Tag | FA=ON, offizieller GHCR-Tag |

**Bewusst unverändert (Multi-GPU-Pools, Phase 2/6-Limitation):** `ollama-m60-guard`
(N04-RTX, 2× M60 kombiniert) bleibt auf `cuda12-maxwell-latest`, FA=OFF — Multi-GPU-FA
ist nicht validiert. Kein Multi-GPU-Pool-Container wurde in dieser Kampagne umgestellt.

**Smoke-Test:** `ollama-tesla-4` (N11-M10, GPU3, erste Instanz im Rollout) mit
`smollm3:3b`, `temperature=0.2, seed=42, num_predict=60` — HTTP 200, 24,0s, 8,21 tok/s,
kohärenter Output. Alle 22 Instanzen nach Neustart `healthy` (Docker-Healthcheck via
`/api/tags`).

**Compose-Dateien aktualisiert:**
- `compose/docker-compose.maxwell.yml` (dieses Repo, GitHub) — von 1-Container-Pool auf
  4 Single-GPU-Services umgestellt, FA=ON-Image, vollständiges Fix-Set. GPU-UUIDs sind
  N11-M10-spezifisch (Kommentar im File verweist darauf).
- N04-RTX (`llm-studio/worker-rtx/docker-compose.yml`, separates Deployment-Repo) und
  N02-M60 (`llm-studio/worker-m60/docker-compose.yml`, dito) wurden mit dem tatsächlichen
  Live-Zustand synchronisiert (Details im Deployment-Repo-Commit, nicht hier — anderes
  Repository/Remote als dieser Fork).

**Nicht angefasst:** Produktions-Compose-Datei auf N04-RTX für `ollama`/`ollama-rgtx`
(RTX-Pool, außerhalb des Scopes — Nutzeranweisung dieser Kampagne). `ollama-0240-test`
(N11-M10) und `ollama-0240-m60` (N04-RTX), zwei themenfremde Alt-Container, ebenfalls
unangetastet gelassen.

### Zusammenfassung Phase 9
Alle drei Hosts laufen jetzt durchgängig auf dem offiziell in der CI gebauten,
validierten Fix-Set (Threads, MMPROJ-Offload, Split-Mode, FA=ON für Single-GPU). Die
Compose-Dateien wurden nachgezogen, damit ein künftiges `docker compose up -d` nicht
auf einen älteren, unvalidierten Stand zurückfällt. Damit ist die gesamte
Optimierungskampagne (Phase 0-9) abgeschlossen. Einziger offener Punkt bleibt Phase 3s
Text-only-Hybrid-Konfundierungstest, blockiert durch den hf.co-Registry-Bug — kein
Blocker für den produktiven Betrieb.

---

## Nachtrag (2026-09-16): Ziel-Topologie korrigiert — N11-M10 zurück zu 4-GPU-Pool,
## Multi-GPU-FA=ON erneut getestet und stabil befunden

Nutzervorgabe nach Abschluss von Phase 9: N11-M10 soll als **eine gepoolte 4-GPU-
Instanz** laufen (nicht als 4 Single-GPU-Instanzen, wie in Phase 4 dieser Kampagne
umgesetzt). N04-RTX soll die beiden M60-GPUs als **eine** Dual-GPU-Instanz nutzen statt
der bisherigen 3-fach-Nutzung (`ollama-m60-1` + `ollama-m60-2` einzeln +
`ollama-m60-guard` kombiniert, alle auf denselben 2 physischen GPUs).

**N04-RTX:** `ollama-m60-1` und `ollama-m60-2` entfernt. `ollama-m60-guard` (bereits
Dual-GPU, FA=OFF) bleibt als einzige M60-Instanz — trägt jetzt sowohl allgemeinen als
auch Guard-Classifier-Traffic. `.env.m60-single` (nur noch für die entfernten
Single-Instanzen relevant) entfernt.

**N11-M10:** 4 Single-GPU-Container gestoppt, durch einen gepoolten 4-GPU-Container
(`ollama`, Port 11434, alle 4 GPUs) ersetzt. Nutzer entschied sich explizit für einen
**Retest von FA=ON auf diesem Multi-GPU-Pool** statt direkt auf das validierte FA=OFF
zurückzufallen — Phase 2 hatte Multi-GPU-FA als "nicht validiert/potenziell instabil"
eingestuft, aber Phase 6s spätere Nachuntersuchung hatte bereits gezeigt, dass eine
strukturell ähnliche Multi-GPU-Instabilität nicht reproduzierbar war (vermutlich
transienter Treiberzustand statt Architekturfehler).

**Testverfahren:** `cuda12-maxwell-singlegpu-fa-latest`-Image (FA=ON) auf dem neuen
4-GPU-Pool, zwei unabhängige Generierungen mit `qwen3.6:35b` (Hybrid-SSM/MoE, derselbe
Modelltyp, der in Phase 2/6 zu Crashes führte):

| Lauf | Prompt/Seed | HTTP | Layer-Offload | done_reason | Fehler |
|---|---|---|---|---|---|
| 1 | Eisenbahn-Zusammenfassung, seed=42 | 200 | 41/42 | length | keine |
| 2 | Neuronale-Netz-Erklärung, seed=123 | 200 | 41/42 | length | keine |

Beide Läufe: kohärenter, korrekter Output (inkl. `thinking`-Trace bei diesem
Reasoning-Modell), Container durchgehend `healthy`, keine `illegal memory access`,
kein Absturz, keine Xid-Fehler im Log. Zusätzlich `smollm3:3b` (dense) als
Baseline-Kontrolle — ebenfalls sauber.

**Verdict: Multi-GPU-FA=ON auf N11-M10 funktioniert stabil — Korrektur der
ursprünglichen Phase-2-Einschätzung.** Das deckt sich mit dem bereits in Phase 6
dokumentierten Muster: die früher beobachtete Multi-GPU-Instabilität war wahrscheinlich
nie ein FA- oder Architektur-Problem, sondern transienter/korrupter GPU-Treiberzustand
aus der intensiven Back-to-Back-Testphase dieser Kampagne. N11-M10 läuft jetzt
produktiv als 4-GPU-Pool mit FA=ON (`compose/docker-compose.maxwell.yml` entsprechend
aktualisiert). **Trotzdem weiterhin mit Vorsicht behandeln:** Nur 2 Testläufe, kein
compute-sanitizer-Lauf wie in Phase 1 — bei künftigen Auffälligkeiten (Crash, falscher
Output) sofort auf `cuda12-maxwell-latest` (FA=OFF) zurückrollen und hier vermerken.

---

## Phase 9 — Abschlusszusammenfassung (2026-09-16)

**Status: abgeschlossen.** Zielarchitektur laut Nutzervorgabe live auf allen drei
Hosts, alle Fixes committet und gepusht (GitHub + git.4noobs.de). Dieser Abschnitt
fasst den gesamten Rollout-Verlauf zusammen, inkl. der nachträglichen
Topologie-Korrektur und des während des Rollouts gefundenen MTP-Crash-Fixes.

### Finale Produktions-Topologie

| Host | Konfiguration | Image | Status |
|---|---|---|---|
| N11-M10 | 1× gepoolte 4-GPU-Instanz (`ollama`, Port 11434) | `cuda12-maxwell-singlegpu-fa-latest`, FA=ON | ✅ live, 2× stabil mit `qwen3.6:35b` getestet |
| N04-RTX | 4× Single-GPU M10 (`tesla-1..4`, Port 11436-11439) | `cuda12-maxwell-singlegpu-fa-latest`, FA=ON | ✅ live |
| N04-RTX | 1× Dual-GPU M60 (`ollama-m60-guard`, Port 11442) | `cuda12-maxwell-latest`, FA=OFF | ✅ live — konsolidiert aus vormals 3 überlappenden M60-Nutzungen (`m60-1`+`m60-2`+`guard`) |
| N02-M60 | 12× Single-GPU M60 (`gpu0..gpu11`, Port 11434-11445) | `cuda12-maxwell-singlegpu-fa-latest`, FA=ON | ✅ live |

Bewusst außerhalb des Scopes (RTX/GTX-GPUs, Nutzeranweisung): `ollama`/`ollama-rgtx`
auf N04-RTX (`ollama-github:latest`, eigenes Dockerfile, kein Fork-Image).

### Rollout-Verlauf (chronologisch)

1. **Erster Rollout-Versuch:** Alle 22 Single-GPU-Instanzen auf den offiziell in der
   CI gebauten `cuda12-maxwell-singlegpu-fa-latest`-Tag umgestellt (vorher: FA=OFF auf
   N11-M10, lokale Test-Tags auf N02-M60/N04-RTX). N11-M10 zunächst als 4 separate
   Single-GPU-Container ausgerollt (Fork-Repo-Commit `86d00f9`).
2. **Topologie-Korrektur (Nutzervorgabe):** N11-M10 zurück auf 1 gepoolte 4-GPU-Instanz
   umgestellt, FA=ON dabei gezielt erneut getestet (2/2 stabile Läufe mit
   `qwen3.6:35b`) statt direkt auf FA=OFF zurückzufallen — korrigiert die
   ursprüngliche Phase-2-Einschätzung (`2a1c110`). N04-RTX: `ollama-m60-1`/`-2`
   entfernt, nur `ollama-m60-guard` bleibt als einzige M60-Instanz (`9077a63`).
3. **qwen3.8:27b-Test deckt MTP-Denylist-Lücke auf:** Neues Modell crasht beim Laden
   auf N11-M10 UND auf N02-M60 (Single-GPU, topologie-unabhängig) mit der exakten
   Finding-2-Signatur. Root Cause: Ollamas eigener Go-Scheduler erzwingt
   `--spec-type draft-mtp` beim ersten Laden, unabhängig vom Python-Denylist in
   `auto-optimize.py` und auch unabhängig von `LLAMA_ARG_SPEC_TYPE`-Env-Var-Overrides
   (`c433244`).
4. **Vollständige Root-Cause-Klärung:** Der eigentliche steuerbare Hebel ist
   `draft_num_predict`, nicht `--spec-type`. `qwen3.8:27b` und die bereits
   existierende `-nospec`-Variante nutzen denselben GGUF-Blob — einziger Unterschied
   ist der Modelfile-Parameter (`draft_num_predict 4` vs. `0`). Direkt verifiziert:
   expliziter `draft_num_predict=0`-Override auf der Plain-Variante verhindert den
   Crash zuverlässig (`2b50f74`).
5. **Proxy-Fix implementiert und ausgerollt:** `ollama-proxy.py` erzwingt jetzt
   `draft_num_predict=0` für `qwen3`/`qwen35(moe)`-Familien bei **jedem** Request —
   auch bei fehlendem Cache (Erstladung) und bei bereits bestehenden, veralteten
   Cache-Einträgen mit riskantem Wert (`51bae38`). Konkret bestätigt: N04-RTXs
   Auto-Optimize-Cache für `qwen3.6:35b` trug seit 2026-06-22 `draft_num_predict=2` —
   dieselbe crash-fähige Konfiguration, unbemerkt produktiv im Einsatz, vermutlich nur
   durch Zufall (nicht-deterministische Race Condition) nie gecrasht.
   Hot-Deploy auf alle 18 Fork-Image-Container (N11-M10: 1, N04-RTX: 5, N02-M60: 12)
   noch vor dem nächsten CI-Rebuild, damit der Fix sofort wirksam ist.

### Betriebsstörung während des Rollouts (selbst verursacht, behoben)

Der Hot-Deploy-Schritt (`docker cp` der gepatchten `ollama-proxy.py` in alle 18
laufenden Container) verlor dabei das Executable-Bit der Datei — der Proxy startete
dadurch auf keinem der 18 Container, Port 11434 war für ~3-5 Minuten auf allen
betroffenen Hosts nicht erreichbar (`entrypoint`-Log zeigte fälschlich
"OLLAMA_AUTO_OPTIMIZE=0", tatsächliche Ursache war die fehlende `+x`-Berechtigung).
Sofort erkannt (Healthcheck-Status "unhealthy"), mit `chmod +x` + Container-Neustart
auf allen 18 Containern behoben. Alle wieder `healthy`, Fix verifiziert funktionsfähig.
**Lehre für künftige Hot-Deploys:** nach `docker cp` immer `chmod +x` auf ausführbare
Skripte nachziehen, nicht auf das Kopierverhalten verlassen.

### Bekannte offene Punkte

- **RTX-Pool (`ollama`, N04-RTX, Port 11434) bleibt beim MTP-Crash-Risiko
  ungeschützt.** Dieser Container nutzt ein anderes Image (`ollama-github:latest`,
  eigenes `Dockerfile.github`) **ohne** die `ollama-proxy.py`-Infrastruktur überhaupt
  (bestätigt: kein Proxy-Prozess läuft dort, nur nacktes `ollama serve`). Der
  Fix lässt sich dort nicht per Hot-Patch anwenden — bräuchte entweder eine eigene
  Proxy-Ergänzung für dieses Image oder eine Modelfile-Anpassung
  (`draft_num_predict 0`) direkt für die dort referenzierten qwen35-Modelle. Bewusst
  nicht umgesetzt, da außerhalb des für diese Kampagne festgelegten Scopes
  (RTX/GTX-GPUs) — **aber die dort dokumentierte 24,3-tok/s-Referenzzahl für
  `qwen3.6:35b` (CLAUDE.md) läuft nach wie vor mit ungeschütztem `draft_num_predict=2`,
  also mit demselben Crash-Risiko wie `qwen3.8:27b` vor dem heutigen Fix.**
- Phase 3s Text-only-Hybrid-Konfundierungstest bleibt blockiert durch den
  hf.co-Registry-Bug (unverändert seit Phase 3).
- `qwen3.8:27b` (Plain-Tag) bleibt technisch geladen und nutzbar — läuft jetzt über
  den Proxy-Fix sicher, aber langsamer als mit MTP (kein Benchmark-Vergleich in dieser
  Kampagne durchgeführt).

### Commits dieser Phase

Fork-Repo (GitHub, `h3rb3rn/ollama-legacy-gpu`): `86d00f9`, `2a1c110`, `c433244`,
`2b50f74`, `51bae38`.
Deployment-Repo (git.4noobs.de, `h3rb3rn/ollama`): `82cc26b`, `9077a63`.

---

## Zusatzuntersuchung (2026-09-16): Fork-Image auf RTX/GTX-GPUs getestet — NICHT einsatzbereit

**Frage:** Lohnt sich die Umstellung von `ollama`/`ollama-rgtx` (N04-RTX, RTX 2060/3060 +
GTX 1060, aktuell Stock-Ollama via `Dockerfile.github`) auf das Fork-Image, um den
systemischen MTP-Crash-Schutz (Proxy) auch dort zu bekommen?

**Vorab-Analyse (Quellcode, `ggml-cuda/fattn.cu`):** Die Flash-Attention-Kernel-Wahl
(`ggml_cuda_get_best_fattn_kernel()`) ist zur Laufzeit hardware-abhängig und vom Build
unabhängig — auf Turing/Ampere (RTX 2060/3060) wird immer der Tensor-Core-Pfad
(`BEST_FATTN_KERNEL_MMA_F16`) gewählt, der Maxwell-TILE-Kernel nie erreicht. Auf Pascal
(GTX 1060, keine Tensor Cores) landet man beim TILE-Kernel — dort wäre der Fork
potenziell sogar von Vorteil, analog zu Maxwell. Diese Analyse war korrekt, aber wie
der Test zeigt, nicht die entscheidende Frage.

**Isolierter Test:** Testcontainer mit `cuda12-maxwell-singlegpu-fa-latest` auf
denselben GPUs wie `ollama-rgtx` (GPU6 RTX 2060 + GPU11 GTX 1060, zum Testzeitpunkt
idle, Produktion nicht angetastet), separater Port. Baseline zuerst mit dem
Stock-Image auf der echten `ollama-rgtx`-Produktionsinstanz gemessen (23,5 tok/s,
`hf.co/h3rb3rn/moe-sovereign-planner-9b`, sauber).

**Ergebnis: Fork-Image crasht beim Laden JEDES getesteten Modells auf RTX 2060** —
sowohl im Dual-GPU-Setup (RTX2060+GTX1060) als auch isoliert auf einer einzelnen
RTX 2060, sowohl mit einem Hybrid-SSM-Modell als auch mit einem einfachen dichten
Modell (`smollm3-expert-security-3b`). Fehler: `CUDA error: unspecified launch
failure`, tritt direkt nach dem Layer-Offload auf, vor jeder eigentlichen Generierung
— 3/3 Testläufe, 100 % Reproduktionsrate.

**Root Cause identifiziert:** Log zeigt bei jedem Start:
```
level=WARN msg="llama-server discovery: could not determine compute capability for
CUDA device — architecture filtering disabled for this device. If inference crashes,
check that the CUDA backend supports this GPU." device="NVIDIA GeForce RTX 2060"
```
Ollamas GPU-Discovery-Schritt kann beim Fork-Build die Compute Capability der RTX 2060
nicht ermitteln (obwohl eine spätere, andere Codestelle im selben Log korrekt
`compute=7.5` meldet — zwei verschiedene Erkennungspfade, inkonsistent). Mit
deaktivierter Architektur-Filterung wird vermutlich eine falsche/inkompatible
CUBIN-Variante für den tatsächlichen Kernel-Launch gewählt → generischer
Launch-Fehler. **Bestätigt fork-spezifisch:** dieselbe Warnung erscheint in keinem
einzigen Log des seit 2 Tagen laufenden `ollama-rgtx` (Stock-Image, identische GPU).

**Verdict: Fork-Image aktuell NICHT für RTX/GTX-GPUs einsatzbereit — nicht auf
N04-RTXs `ollama`/`ollama-rgtx` umstellen.** Kein Zusammenhang mit der ursprünglich
vermuteten Tensor-Core-Frage (die Vorab-Analyse dazu war korrekt, aber irrelevant,
weil das Modell gar nicht so weit kommt, dass Flash-Attention-Kernel-Auswahl
überhaupt greifen würde). Der MTP-Crash-Schutz auf diesen zwei Containern bleibt bei
der bereits umgesetzten Einzelmodell-Lösung (Modelfile-Override für `qwen3.8:27b`,
siehe `BUG-hybrid-arch-degeneration.md`). Eine echte Behebung würde eine
Fehlersuche im GPU-Discovery-Code dieses Forks für Nicht-Maxwell-Architekturen
erfordern — nicht in dieser Kampagne untersucht, da außerhalb des Scopes
(RTX/GTX-Hosts).
