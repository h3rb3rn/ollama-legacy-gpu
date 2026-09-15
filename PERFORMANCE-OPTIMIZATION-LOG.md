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

**Status: in Arbeit**

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
