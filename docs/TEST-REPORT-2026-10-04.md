# Testbericht 2026-10-04: Input-Layer in VRAM, Fork vs. Stock

Alle Werte sind auf den Hosts gemessen (`HOST_VERIFIED`), die Unit-Tests lokal
(`LOCAL_VERIFIED`). Rohdaten stehen nicht im Repo; die Zahlen stammen aus den
Messläufen vom 2026-10-04.

## Testobjekt und gemeinsame Einstellungen

| Einstellung | Wert |
| --- | --- |
| Image | `ollama-gaps:input-gpu-test-20261004` (Ollama v0.35.1, CUDA 12.0.1, Archs 50;52;61;75;86) |
| Modell | `qwen3.6:35b` (qwen35moe, 42 Layer inkl. nextn-Slot) |
| Kontext | 131072 |
| KV-Cache | q4_0 |
| Flash Attention | an |
| Batch (`OLLAMA_MAX_BATCH_SIZE`) | 64 |
| `LLAMA_ARG_FIT_TARGET` | 256 (MiB) |
| `OLLAMA_DRAFT_NUM_PREDICT` | 0 (N02-M60), sonst Image-Default |
| mmproj | an |
| Langer Prompt | 11 682 Token |

## Ergebnisse (Fork-Image, alles in VRAM)

| Host | GPUs | Layer | Gewichte im Host | Decode (kurz) | Langer Prompt |
| --- | --- | --- | --- | --- | --- |
| N11-M10 | 4x Tesla M10 | 42/42 | nein | 9,0 → 9,5 tok/s, Ladezeit 118 → 66 s | 633 s, Prefill 19,0, Decode 7,0 tok/s |
| N04-RTX | RTX 2060 + RTX 3060 | 42/42 | nein | 40,0 → 40,3 tok/s | 27 s, Prefill 498, Decode 37,5 tok/s |
| N02-M60 | 4x Tesla M60 | 42/42 | nein (CUDA0-3 zus. 20,2 GiB) | 14,1 → 15,3 tok/s, Ladezeit ~121 s | 178 s, Prefill 69,3, Decode 13,1 tok/s |

„→“ = vorher (Embedding auf CPU) → nachher. Bei allen Läufen: HTTP 200, kein
Xid, kein OOM, Container blieb gesund. VRAM auf GPU0: N11-M10 7840/8192 MiB,
N02-M60 7501/8192 MiB (knapp).

## Fork vs. Stock (N04-RTX, Batch 64)

A/B-Test von Fork und Stock mit Batch 64 auf N04-RTX. Stock ignoriert
`OLLAMA_MAX_BATCH_SIZE` und rechnet selbst (2048). Ein früherer Stock-Stall
ließ sich nicht reproduzieren; seine Ursache ist ungeklärt.

## Unit-Tests

`python3 -m unittest discover -s tests -v` im Worktree. Neu:
`test_input_gpu_patch_follows_layer_zero_keeps_opt_out_and_is_idempotent`
(Anker, Opt-out-Zweig, Idempotenz, fail closed). Die Patch-Guard-Tests prüfen
außerdem, dass fehlende Upstream-Dateien den Build abbrechen lassen.

## Grenzen

- Nur das `cuda12-maxwell`-Image wurde gebaut und getestet; `cuda11-legacy` und
  `cuda13-rtx` enthalten den Patch, sind aber ungetestet.
- Nicht per Konfiguration vermeidbar bleiben Staging-Puffer im Host:
  ~1 MiB Output, 17–66 MiB CUDA_Host (Compute), ~25 MiB mmproj (CPU-Compute).
- Kontext bleibt bei 131072 (256k nicht getestet).
- Decode auf Maxwell ist host-gebunden (GPU nahezu im Leerlauf).
- Je Host ein Lauf pro Konfiguration, keine Streuungsangaben.
- Batch 64 vs. 512: Entscheidung steht aus.
