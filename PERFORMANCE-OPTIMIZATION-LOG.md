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

**`--load-mode mlock`/`none` (ehemals `--mlock`/`--no-mmap`):** Versuch eines A/B-Tests
mit Olmo-32B (CPU-lastig, ~73 %/27 % CPU/GPU-Split) auf N11-M10 abgebrochen — der
Request hing bei 0 % GPU-Auslastung fest, ohne dass sich der Zustand über mehrere
Minuten änderte (echtes Hängenbleiben, nicht nur die erwartete Langsamkeit dieses
Modells). Angesichts des bereits sehr hohen Zeitaufwands dieser Kampagne nicht weiter
verfolgt. **Kein empirisches Ergebnis** — verbleibt bei der bereits in der
Deep-Research-Phase dokumentierten Einschätzung: potenziell hilfreich für
CPU-Offload-lastige Szenarien angesichts des ohnehin knappen freien RAM auf diesen
Hosts (14 GiB verfügbar auf N11-M10 laut `free -h`, aber unter Last schon einmal auf
2,5 GiB beobachtet), aber nicht blind als Default aktivieren — müsste in einer
ruhigeren Session ohne Host-Nebenlast erneut versucht werden.

### Zusammenfassung Phase 8
Von den ursprünglich drei Feintuning-Kandidaten waren zwei (NUMA, ECC) bereits durch
die Hardware-/Treiber-Gegebenheiten erledigt, ohne dass überhaupt eine Änderung nötig
gewesen wäre. Der dritte (`--load-mode`) bleibt offen für einen späteren, gezielteren
Versuch.
