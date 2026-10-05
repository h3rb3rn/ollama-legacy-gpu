# Fleet-Stand und Git-Lage der Deployment-Klone, 2026-10-05

Stand der `:11434`-Instanzen von N04-RTX, N02-M60 und N11-M10 sowie der Zustand der vier Klone des Deployment-Repos
(`https://git.4noobs.de/h3rb3rn/ollama.git`). Alles wurde am 2026-10-05 an den Hosts gelesen (Container-Env per
`docker inspect`, Git-Stand per `git status`/`git log`). Die Absicht hinter Änderungen aus früheren Sessions ist nur dort
angegeben, wo sie belegt ist.

## Live-Stand (Image `ollama-gaps:ctxfix-20261005`, Ollama 0.35.1)

| Einstellung | N04-RTX | N02-M60 (Pool GPU0–3) | N11-M10 |
|---|---|---|---|
| Karten | 2× RTX 2060 + 2× RTX 3060 (12 GiB) | 4× Tesla M60 (8 GiB) | 4× Tesla M10 (8 GiB) |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | 262144 | 262144 |
| Batch (`OLLAMA_MAX_BATCH_SIZE`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`) | 512 | 512 | 512 |
| `OLLAMA_KV_CACHE_TYPE` | q4_0 | q4_0 | q4_0 |
| `LLAMA_ARG_FIT_TARGET` | 256 | 256 | 256 |
| `LLAMA_ARG_MMPROJ_OFFLOAD` | true | true | true |
| `OLLAMA_DRAFT_NUM_PREDICT` | 0 | 0 | 0 |
| `OLLAMA_GPU_AUTODETECT` | 0 | **1** | 0 |
| `GGML_CUDA_GRAPHS_LEGACY` | – | – | **1** |
| Layer / wirksamer Kontext | 42/42, `-c 262144` | 42/42, `-c 262144` | 42/42, `-c 262144` |
| Decode / Prefill (tok/s) | 40.9 / 1014–1171 | 13.8–14.5 / 155 | 9.1 / 49 |

Die Messwerte stehen mit Methode in [TUNING-2026-10-04.md](../TUNING-2026-10-04.md).

Zwei Abweichungen zwischen den Hosts, die Vergleiche verfälschen können:
- **N02-M60 `OLLAMA_GPU_AUTODETECT=1`:** Rest eines Tests der Standardkette (`gpu-detect.sh` exportiert dann u. a. den
  MTP-Default 0). Es wurde nie entschieden, ob das bleiben soll; alle N02-Messungen vom 2026-10-05 liefen damit. Die beiden
  anderen Hosts laufen mit 0.
- **N11-M10 `GGML_CUDA_GRAPHS_LEGACY=1`:** Opt-in-CUDA-Graphs auf Maxwell (dort +35 % Decode gemessen, auf N02-M60 ohne Effekt).
  Die N11-Zahlen enthalten diesen Effekt, die N02-Zahlen nicht.

## Wo die Konfiguration liegt

| Host | Datei(en) | Versioniert? |
|---|---|---|
| N04-RTX | `llm-studio/worker-rtx/docker-compose.yml`, `.env.stock-rtx` | ja (lokal committet, nicht gepusht) |
| N02-M60 | `llm-studio/worker-m60/docker-compose.yml`, `.env.m60-pool` | ja (lokal committet, nicht gepusht) |
| N11-M10 | `llm-studio/worker-tesla/docker-compose.yml` (JSON) | **nein**, per `llm-studio/worker-tesla/.gitignore` ausgeschlossen; der Live-Stand existiert nur auf dem Host |

Backups neben den Dateien: `*.pre-ctxfix-20261005` (diese Session), `*.pre-gaps-test-20261003`, `*.bak-pre-*-20261004`
(Parallel-Session), ältere `*.bak*`/`*.pre-*`. Sie sind untracked und bewusst nicht committet.

## Die vier Klone des Deployment-Repos

`origin/main` steht bei `9077a63` (2026-09-16). Die Klone sind divergent:

| Klon | vor / hinter origin | lokale, nicht gepushte Commits |
|---|---|---|
| Steuerhost `/opt/deployment/ollama` | 0 / 0 | keine; uncommittet: `.gitignore`, `worker-host/docker-compose.yml`, `worker-m60/docker-compose.yml` (nicht aus dieser Session) |
| N04-RTX | 3 / 2 | `8c97f04`, `badabbd`, `7cabec1`; es fehlen `82cc26b` und `9077a63` |
| N02-M60 | 5 / 0 | `8c97f04`, `34f264f`, `82e2b66`, `e3e21e5` (Merge von origin), `1f5d33f` |
| N11-M10 | 0 / 3 | keine; es fehlen `785e2a7`, `82cc26b`, `9077a63` |

Herkunft der lokalen Commits (aus Autor, Datum und Trailern):
- `8c97f04` (2026-09-05, Single-Instance-Pool über 12 M60) und `34f264f` (2026-09-05, Aufteilung 4-GPU-Pool + 8 Einzel-GPU-Instanzen):
  mit `Co-Authored-By: Claude Sonnet 5` aus einer früheren Session (`session_01QPg8K989h2UFgboxMReJFS`). `8c97f04` liegt
  auf N04-RTX **und** N02-M60 lokal und wurde nie gepusht.
- `82e2b66` (2026-10-03, MTP-Race-Ursache, 4-GPU-Pool mit FA=ON/q4_0) und `e3e21e5` (Merge mit origin): frühere Session
  desselben Tages, aus der auch der Gap-Prompt stammt.
- `badabbd` (2026-10-04, N04 auf Fork-Image, Batch 64): Parallel-Session `ai-village-f2`.
- `7cabec1` (N04) und `1f5d33f` (N02), 2026-10-05: Live-Stand von oben, diese Session. **Lokal, nicht gepusht.**

## Nicht committet und warum

| Fund | Befund | Entscheidung |
|---|---|---|
| N11 `fork/compose/.env`: `OLLAMA_CONTEXT_LENGTH` 262144 → **132768**, `MAX_LOADED_MODELS` 2 → 1, `NUM_PARALLEL` 2 → 1 | zuletzt geändert 2026-09-16 22:36; Urheber unbekannt (nicht diese Session, nicht die Parallel-Session); 132768 ist vermutlich ein Tippfehler für 131072 | **nicht committet**, braucht Entscheidung des Users |
| N11 `llm-studio/worker-rtx/.env.tesla`: `OLLAMA_CONTEXT_LENGTH` 32768 → 262144 | zuletzt geändert 2026-08-29; Urheber unbekannt | **nicht committet**, braucht Entscheidung |
| `fork/compose/docker-compose.fleet-v035.json` (auf allen drei Hosts) | wahrscheinlich Flottenrollout 2026-10-01 (mtime 2026-10-01 23:46); Inhalt nicht auf Gleichheit geprüft | nicht committet |
| N11 `tesla/ollama-legacy-gpu/` | Verzeichnis von 2026-06-25, Urheber unbekannt | nicht committet |
| N02 `.env.m60-single12`, `docker-compose.yml.reconstructed` | Urheber unbekannt | nicht committet |
| N04 `llm-studio/worker-rtx/docker-compose.bonsai-m10-prod.yml`, `ollama/.env.backup` | Urheber unbekannt | nicht committet |
| N02 `fork/RTX-TESLA-GAPS-PROMPT.md` | veraltete Kopie des Gap-Prompts (kanonisch auf dem Steuerhost); sein Gap 4 ist überholt (Ursache war der fit-Off-by-one, Fix `676a3b2`) | nicht committet, löschen oder ignorieren |
| `*.bak*`, `*.pre-*` | Backups | untracked lassen oder per `.gitignore` ausnehmen |

Ein Scan der Diffs und der Kandidaten-Dateien auf Muster wie `key`, `token`, `secret`, `password` fand keine Zugangsdaten
(einfacher Mustertest, kein vollständiges Audit).

## Offene Entscheidungen für den User
1. Welcher Klon ist kanonisch, und wie werden die Historien zusammengeführt und gepusht? Vorschlag der Parallel-Session:
   `origin/main` (`9077a63`) als Basis, N02 als Quelle seiner eigenen Commits. Bisher wurde nichts gemergt oder gepusht.
2. `OLLAMA_GPU_AUTODETECT` auf N02-M60: bei 1 bleiben oder auf 0 zurück (dann N02 neu messen).
3. N11-Änderungen in `fork/compose/.env` und `.env.tesla`: übernehmen, korrigieren (132768?) oder verwerfen.
4. Soll `llm-studio/worker-tesla/docker-compose.yml` versioniert werden (heute ausgeschlossen)?
5. Backups per `.gitignore` ausnehmen?
