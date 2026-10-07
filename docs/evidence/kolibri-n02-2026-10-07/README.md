# Kolibri-1 on N02-M60, 2026-10-07

Raw results of `kol-suite.py` (Image `ollama-gaps:kolibri-20261006`, Kolibri-1 Q4_K_M, `-c 262144`, q4_0, batch 64).

- `results-matrix-ladder-soak.jsonl`: GPU-count matrix (8–12 GPUs, greedy fill), context ladder, 60-minute soak.
  The ladder row with `target_tokens 245000` is **invalid**: its prompt exceeded the context and was truncated to 131 074 tokens
  (first needle lost). The `xid` field of the soak row is **not valid**: `dmesg` and the kernel journal are not readable for the test user.
- `results-ladder-250k-rerun.jsonl`: valid repeat of the longest rung (250 112 prompt tokens, both needles found).
- `results-qwen-unpatched-boundary.jsonl`: `qwen3.6:35b` on `pipefix-20261005` (without the VMM patch) with 8 and 9 GPUs (Kolibri-1 does not
  load in that image); a third row (4 GPUs, patched image) belongs to an aborted run.
- `kol-suite.py`: driver as run on N02-M60 (the ladder rerun used `LADDER=250000 TOK_PER_PARA=83.6`).
- The German words in the prompts of `kol-suite.py` (filler text, needle sentences, questions) are deliberate test data, not documentation.
