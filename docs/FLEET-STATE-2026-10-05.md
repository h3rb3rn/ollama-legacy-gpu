# Fleet-Stand und Git-Lage der Deployment-Klone, 2026-10-05

Stand der `:11434`-Instanzen von N04-RTX, N02-M60 und N11-M10 sowie der Zustand der vier Klone des Deployment-Repos
(`https://git.4noobs.de/h3rb3rn/ollama.git`). Alles wurde am 2026-10-05 an den Hosts gelesen (Container-Env per
`docker inspect`, Git-Stand per `git status`/`git log`). Die Absicht hinter Änderungen aus früheren Sessions ist nur dort
angegeben, wo sie belegt ist.

Vorgaben des Users für die Instanzen: 256k-KV-Cache bei q4_0, **alles im VRAM und nichts im RAM**, `qwen3.6:35b` primär und warm
(`MAX_LOADED_MODELS=1`, `NUM_PARALLEL=1`, `KEEP_ALIVE` 24 h).

## Live-Stand (Image `ollama-gaps:gap2fix-20261005`, Ollama 0.35.1)

| Einstellung | N04-RTX | N02-M60 (Pool GPU0–3) | N11-M10 |
|---|---|---|---|
| Karten im Pool | 3: 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 4× Tesla M60 (8 GiB) | 4× Tesla M10 (8 GiB) |
| Frei für anderes | 1× RTX 3060 (`GPU-63bfbd4b-…`, Bus 09:00.0) für ComfyUI | – | – |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | 262144 | 262144 |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | **64** | **64** |
| `OLLAMA_KV_CACHE_TYPE` | q4_0 | q4_0 | q4_0 |
| `LLAMA_ARG_FIT_TARGET` | 256 | 256 | 256 |
| `LLAMA_ARG_MMPROJ_OFFLOAD` | true | true | true |
| `OLLAMA_DRAFT_NUM_PREDICT` | 0 | 0 | 0 |
| `OLLAMA_GPU_AUTODETECT` | 0 | **1** | 0 |
| `GGML_CUDA_GRAPHS_LEGACY` | – | – | **1** |
| Layer / wirksamer Kontext | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode / Prefill (tok/s), Modell `a7eb95c53bcf` | 42.2–42.4 / 1043–1156 | 14.2–15.3 / 86.4–87.0 | 9.2 / 24.1 |
| nicht im VRAM laut `/api/ps` | 0 MiB (0,0 %) | 58 MiB (0,24 %) | 0 MiB (0,0 %) |
| Host-Puffer `CUDA_Host compute` | 1029 MiB | 32,8 MiB | **129,6 MiB** (Pipeline-Parallelität) |

Messwerte und Methode: [TUNING-2026-10-04.md](../TUNING-2026-10-04.md). Warum der Batch je Host verschieden ist: auf den 8-GiB-Hosts
hinterließ Batch 512 einen gepinnten Host-Puffer von 260 MiB (angezeigt als „1 % CPU“); N04-RTX bleibt bei 512, dort zeigt
Ollama 100 % GPU (der Runner legt dort trotzdem 1029 MiB Host-Puffer an, die Anzeige bildet ihn nicht ab).

Zwei Abweichungen zwischen den Hosts, die Vergleiche verfälschen können:
- **N02-M60 `OLLAMA_GPU_AUTODETECT=1`:** Rest eines Tests der Standardkette (`gpu-detect.sh` exportiert dann u. a. den
  MTP-Default 0). Es wurde nie entschieden, ob das bleiben soll; alle N02-Messungen vom 2026-10-05 liefen damit.
- **N11-M10 `GGML_CUDA_GRAPHS_LEGACY=1`:** Opt-in-CUDA-Graphs auf Maxwell (dort +35 % Decode gemessen, auf N02-M60 ohne Effekt).
  Die N11-Zahlen enthalten diesen Effekt, die N02-Zahlen nicht.

## Modellversionen `qwen3.6:35b` und MTP-Schutz (Stand 2026-10-05, nachmittags)

Alle drei Hosts laufen jetzt mit dem **aktuellen Registry-Stand** `qwen3.6:35b` = Digest `a7eb95c53bcf` (35,5B Parameter,
35 505 251 456, 21,07 GiB, 41 Blöcke, NextN-Kopf `blk.40.nextn.*`, Manifest `draft_num_predict 2`) und dem Image
`ollama-gaps:gap2fix-20261005`. Vorher: N11-M10 und N04-RTX hatten den älteren Registry-Stand (`07d35212591f`, 36,0B, 22,29 GiB,
40 Blöcke, ohne NextN-Kopf, lokal geladen 2026-06-08/10), N02-M60 die gleiche Modelldatei wie heute, aber mit lokal gesetztem
`draft_num_predict 0` (`637d2bc25380`). Der Unterschied „35.5B“ gegenüber „36.0B“ im Dashboard war also kein lokaler Umbau, sondern
ein älterer gegenüber dem aktuellen Registry-Stand.

| Host | Sicherung der vorherigen Version | Befund nach dem Pull |
|---|---|---|
| N02-M60 | Tag `qwen3.6:35b-n02-nextn-20261003` (`637d2bc25380`, `draft_num_predict 0`) | läuft, 42/42, 14,2 / 15,3 tok/s, Prefill ~87 |
| N11-M10 | Tag `qwen3.6:35b-36b-20260610` (`07d35212591f`, 36,0B; per `pull` nicht mehr zu bekommen) | läuft, 42/42, 9,2 / 9,2 tok/s, Prefill 24,1 |
| N04-RTX | Tag `qwen3.6:35b-36b-20260608` (`07d35212591f`, 36,0B) | läuft, 42/42, 42,2 / 42,4 tok/s, Prefill 1043–1156 |

**Gap 2 (MTP-Absturz auf Maxwell) ist damit auch für frisch gezogene Registry-Modelle geschlossen.** Das Registry-Manifest setzt
`draft_num_predict 2` ausdrücklich; die frühere Version des Patches behandelte einen Manifest-Wert wie einen Anfragewert und ließ ihn
durch (ein Test-Pull auf N02-M60 hatte den Schutz kurz ausgehebelt; Tag aus dem Backup wiederhergestellt, keine MTP-Aktivität, kein
CUDA-Fehler). Jetzt begrenzt `OLLAMA_DRAFT_NUM_PREDICT` Manifest-Werte (0 = MTP aus), Anfragen mit eigenem Wert behalten Vorrang.
Abnahme je Host: Modell leer geladen (ohne zu generieren), Runner-Kommandozeile **ohne** `--spec-*`-Argumente, danach Generierung ohne
Fehler. Go-Test: `TestDraftNumPredictServerDefault`.

**Pipeline-Parallelität und Host-Puffer.** llama.cpp schaltet bei mehreren GPUs im Layer-Split die Pipeline-Parallelität ein, wenn
`n_gpu_layers > n_layer_all` gilt und keine Tensor-Overrides aktiv sind; sie vervierfacht die gepinnten Eingabepuffer
(`sched copies = 4`). Gemessen mit dem neuen Modell bei Batch 64 auf N11-M10: Host-Puffer 129,6 statt 32,8 MiB, `RssShmem` 146 statt
50 MiB, Compute-Puffer 648 statt 551 MiB je GPU, **ohne Prefill-Gewinn** (24,1 gegenüber 23,9 tok/s). Auf N02-M60 ist sie mit
demselben Modell aus: dort aktiviert `OLLAMA_GPU_AUTODETECT=1` über `TIER_THRESHOLD=4` den Fit des Forks (22,7 s statt 0,9 s), der
eine teilweise verteilte Schicht (`n_part=1`) und damit Tensor-Overrides erzeugt, und die schalten die Pipeline ab. Deshalb gilt:
`OLLAMA_GPU_AUTODETECT=1` auf N02-M60 **nicht** auf 0 stellen, solange es keinen eigenen Schalter gegen die Pipeline gibt.
N04-RTX läuft weiter mit Pipeline (Host-Puffer 1029 MiB bei Batch 512, von Ollama nicht als „CPU“ angezeigt).

## Wo die Konfiguration liegt

| Host | Datei(en) | Versioniert? |
|---|---|---|
| N04-RTX | `llm-studio/worker-rtx/docker-compose.yml`, `.env.stock-rtx`, `docker-compose.m60-guard.yml`, `docker-compose.bonsai-m10-prod.yml` | ja (lokal committet, nicht gepusht) |
| N02-M60 | `llm-studio/worker-m60/docker-compose.yml`, `.env.m60-pool` | ja (lokal committet, nicht gepusht) |
| N11-M10 | `llm-studio/worker-tesla/docker-compose.yml` (JSON) | **nein**, per `llm-studio/worker-tesla/.gitignore` ausgeschlossen; der Live-Stand existiert nur auf dem Host |

Backups neben den Dateien: `*.pre-3gpu-20261005`, `*.pre-vramonly-20261005`, `*.pre-ctxfix-20261005`, `*.pre-gaps-test-20261003`,
`*.bak-pre-*-20261004` (Parallel-Session), ältere `*.bak*`/`*.pre-*`. Sie sind untracked und bewusst nicht committet.

## Die vier Klone des Deployment-Repos

`origin/main` steht bei `9077a63` (2026-09-16). Die Klone sind divergent:

| Klon | vor / hinter origin | lokale, nicht gepushte Commits |
|---|---|---|
| Steuerhost `/opt/deployment/ollama` | 0 / 0 | keine; uncommittet: `.gitignore`, `worker-host/docker-compose.yml`, `worker-m60/docker-compose.yml` (nicht aus dieser Session) |
| N04-RTX | 5 / 2 | `8c97f04`, `badabbd`, `7cabec1`, `bdb9d51`, `e543b01`; es fehlen `82cc26b` und `9077a63` |
| N02-M60 | 6 / 0 | `8c97f04`, `34f264f`, `82e2b66`, `e3e21e5` (Merge von origin), `1f5d33f`, `6bcbe7d` |
| N11-M10 | 0 / 3 | keine; es fehlen `785e2a7`, `82cc26b`, `9077a63` |

Herkunft der lokalen Commits (aus Autor, Datum und Trailern):
- `8c97f04` (2026-09-05, Single-Instance-Pool über 12 M60) und `34f264f` (2026-09-05, Aufteilung 4-GPU-Pool + 8 Einzel-GPU-Instanzen):
  mit `Co-Authored-By: Claude Sonnet 5` aus einer früheren Session (`session_01QPg8K989h2UFgboxMReJFS`). `8c97f04` liegt
  auf N04-RTX **und** N02-M60 lokal und wurde nie gepusht.
- `82e2b66` (2026-10-03, MTP-Race-Ursache, 4-GPU-Pool mit FA=ON/q4_0) und `e3e21e5` (Merge mit origin): frühere Session
  desselben Tages, aus der auch der Gap-Prompt stammt.
- `badabbd` (2026-10-04, N04 auf Fork-Image, Batch 64): Parallel-Session `ai-village-f2`.
- 2026-10-05, diese Session, **lokal, nicht gepusht**: `7cabec1` (N04, Live-Stand ctxfix/512), `bdb9d51` (N04, Compose-Definitionen von
  `ollama-tesla-bonsai` und `ollama-m60-guard`), `e543b01` (N04, drei GPUs, `ollama-m60-guard` per `extends` auf die neue Datei),
  `1f5d33f` (N02, Live-Stand ctxfix/512), `6bcbe7d` (N02, Batch 512 → 64).

## Dateien aus einer Codex-Session (laut User) und was damit geschah

| Datei | Befund | Entscheidung |
|---|---|---|
| N04 `docker-compose.bonsai-m10-prod.yml` | stimmt mit dem laufenden `ollama-tesla-bonsai` überein (Image-SHA, Port 11436, Kontext 190000, q4_0) | **übernommen** (`bdb9d51`) |
| `docker-compose.fleet-v035.json` (N04) | gerenderter Snapshot des Rollouts vom 2026-10-01; enthielt als einzige Quelle die Definition von `ollama-m60-guard` (:11442), die die Live-Compose per `extends` einbindet | Dienst als `docker-compose.m60-guard.yml` **übernommen** (`bdb9d51`, rendert identisch bis auf das Image-Standard-`PATH`); Snapshot archiviert |
| `docker-compose.fleet-v035.json` (N02, N11) | Pool-/Einzel-Snapshots mit überholten Images (Kontext 131072/65536); von keiner Compose referenziert | archiviert |
| N02 `.env.m60-single12` (2026-09-06), `docker-compose.yml.reconstructed` (2026-09-11) | Benchmark-Env für eine 12-GPU-Einzelinstanz bzw. ältere Rekonstruktion; überholt | archiviert |
| N04 `ollama/.env.backup` (2026-06-16) | altes Env-Backup | archiviert |
| N11 `tesla/ollama-legacy-gpu/` | alter Klon des GitHub-Forks (Stand 2026-06-25, lokale Änderungen an `compose/.env` und `docker-compose.maxwell.yml`, nichts ungepusht); die lokale `CLAUDE.md` nennt ihn noch als Deploy-Pfad für N11-M10 | **bewusst liegengelassen** |

Archiv: `/opt/deployment/ollama-archive/codex-2026-10-01/` auf jedem Host (außerhalb des Repos, mit README, per `mv` zurückholbar).

**Vorfall beim Aufräumen:** Die Live-Compose von N04-RTX band `ollama-m60-guard` per `extends:` aus `fleet-v035.json` ein (im
Commit `HEAD` getrackt referenziert). Ich hatte die Datei archiviert, ohne die Live-Compose zu prüfen; dadurch scheiterte jeder
`docker compose`-Befehl in diesem Projekt („no such file“), aufgefallen beim Neustart für die GPU-Reduzierung. Der laufende
Container war nie betroffen. Behoben: `extends` zeigt auf `docker-compose.m60-guard.yml`. Auf N02-M60 und N11-M10 verwies nichts auf
archivierte Dateien (Compose parst, geprüft).

## Weitere nicht committete Funde

| Fund | Befund | Entscheidung |
|---|---|---|
| N11 `fork/compose/.env`: `OLLAMA_CONTEXT_LENGTH` 262144 → **132768**, `MAX_LOADED_MODELS` 2 → 1, `NUM_PARALLEL` 2 → 1 | Vorlage für den **12-GPU-Pool von N04-RTX** (Kopfzeile, Dynamic-Pool-Kommentare), nicht die Live-Config von N11-M10; zuletzt geändert 2026-09-16; Urheber unbekannt; 132768 ist vermutlich ein Tippfehler für 131072. Die Live-Compose von N11-M10 hat bereits 1/1 und 262144 | **nicht committet**, braucht Entscheidung |
| N11 `llm-studio/worker-rtx/.env.tesla`: `OLLAMA_CONTEXT_LENGTH` 32768 → 262144 | Datei im `worker-rtx`-Verzeichnis (N04-Design); zuletzt geändert 2026-08-29; Urheber unbekannt | **nicht committet**, braucht Entscheidung |
| N02 `fork/RTX-TESLA-GAPS-PROMPT.md` | veraltete Kopie des Gap-Prompts (kanonisch auf dem Steuerhost); sein Gap 4 ist überholt (Ursache war der fit-Off-by-one, Fix `676a3b2`) | nicht committet, löschen oder ignorieren |
| `*.bak*`, `*.pre-*` | Backups | untracked lassen oder per `.gitignore` ausnehmen |

Ein Scan der Diffs und der Kandidaten-Dateien auf Muster wie `key`, `token`, `secret`, `password` fand keine Zugangsdaten
(einfacher Mustertest, kein vollständiges Audit).

## Offene Entscheidungen für den User
1. Welcher Klon ist kanonisch, und wie werden die Historien zusammengeführt und gepusht? Vorschlag der Parallel-Session:
   `origin/main` (`9077a63`) als Basis, N02 als Quelle seiner eigenen Commits. Bisher wurde nichts gemergt oder gepusht.
2. `OLLAMA_GPU_AUTODETECT` auf N02-M60: bei 1 lassen (schaltet indirekt die Pipeline-Parallelität ab, siehe oben); bei einem eigenen Schalter gegen die Pipeline kann es auf 0.
3. N11-Vorlagen `fork/compose/.env` und `.env.tesla` (N04-Design): übernehmen, korrigieren (132768?) oder verwerfen.
4. Soll `llm-studio/worker-tesla/docker-compose.yml` versioniert werden (heute ausgeschlossen)?
5. Backups per `.gitignore` ausnehmen? Den alten Klon `tesla/ollama-legacy-gpu/` auf N11-M10 archivieren (dann die `CLAUDE.md`
   anpassen)?
