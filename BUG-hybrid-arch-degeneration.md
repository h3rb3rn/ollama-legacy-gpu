# Bug Report: Hybrid/Recurrent-Architecture Models Degenerate on Maxwell (cuda12-maxwell fork)

**Status:** **RESOLVED** as of 2026-09-15 (see bottom of file) — the exact original repro
model (`moe-expert-coder-4b-v2`) was re-run on real Tesla M60 hardware (N02-M60) with the
accumulated fixes (FA rebuild + `LLAMA_ARG_MMPROJ_OFFLOAD=false` + MTP fix) and produced
coherent, non-repeating output at 20.48 tok/s with full 33/33-layer GPU residency. Root
cause was never the SSM/Mamba kernel — see the full timeline below for how the diagnosis
evolved (CLIP compute-buffer starvation → wrong fix tried and reverted → real fix found
→ confirmed fixed on the original hardware+model combination).
**Severity:** High — makes the entire Qwen3.5/3.8 hybrid-attention model family unusable
for inference quality on the M10/M60 Tesla fleet (~16 physical GPUs across N02-M60,
N04-RTX, N11-M10), while dense-transformer models on the same hardware are unaffected.
**Scope:** `ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest`
(built from `dockerfiles/Dockerfile.cuda12-maxwell` in this repo, using
`LLAMA_CPP_FORK=https://github.com/h3rb3rn/llama.cpp-legacy-gpu.git`).

---

## Summary

`moe-expert-coder-4b-v2` (a LoRA fine-tune of Qwen3.5-4B, a **hybrid linear-attention /
full-attention decoder** — 8 full-attention layers + 24 Mamba/SSM-style linear-attention
layers) produces **byte-identical degenerate repetition-loop output** on every tested
Maxwell-class GPU (Tesla M60 and Tesla M10, 8 physical cards tested across 2 hosts) when
served through this fork, while the exact same GGUF file produces correct, coherent
C++20 code on RTX (Ampere/Turing) hardware via the unrelated `ollama-github:latest`
build.

This is fully deterministic and reproducible — not sampling noise. It has been narrowed
to the model's **hybrid/recurrent architecture** interacting badly with **this specific
fork on Maxwell**, not to Maxwell hardware in general, not to GPU pooling, and not to
the fork's own custom patches (which do not touch the SSM code path at all — see
"What has been ruled out" below).

---

## Exact Reproduction

```
Model:    moe-expert-coder-4b-v2 (Qwen3.5-4B base, LoRA fine-tuned)
Prompt:   "Implement a lock-free single-producer single-consumer ring buffer in
           C++20 using std::atomic with explicit acquire/release memory ordering."
Options:  temperature=0.2, seed=42, f16 KV-cache, Flash Attention disabled
```

**On Maxwell (Tesla M60 and Tesla M10, via this fork, tested on single-GPU AND on a
4-GPU pool — identical result either way):**

Output is exactly this 4664-character block, repeated verbatim until the token cap:

```
[IMPLEMENTING CORRECTNESS VERDICT]
The lock-free SPSC ring buffer implementation is correct.
- The write side is a loop that increments the tail pointer and wraps around.
- The read side is a loop that increments the head pointer and wraps around.
```

(this four-line block repeats identically until `num_predict` is exhausted)

**On RTX (Ampere/Turing, via `ollama-github:latest`, same GGUF, same prompt, same
seed):** genuine, correct, non-repeating C++20 code.

The degenerate text appears from **token 1** — this is not a gradual numerical drift
over a long generation, it is wrong from the very first tokens of the response.

---

## What has been ruled out

1. **GPU pooling / VRAM is not the cause.** Tested the identical model+prompt+seed on
   a single Tesla M60, and separately on a 4x-Tesla-M60 pool (`ollama-m60-pool`,
   32 GiB combined). Byte-identical degenerate output both times.

2. **Not a training/fine-tuning defect.** The same GGUF file produces correct output
   on RTX hardware. The model itself is fine; the failure is specific to this
   fork+Maxwell combination.

3. **Not Maxwell hardware in general.** `SmolLM3-3B` (a dense, non-hybrid Transformer
   — no Mamba/SSM layers) produces correct, coherent output on the same Maxwell GPUs,
   through **both** this fork and stock `ollama/ollama:0.24.0`. This is the key
   isolating result: dense architectures are fine on Maxwell; the hybrid architecture
   specifically is not.

4. **Not this fork's own patches.** Per `README-LEGACY-GPU.md` in
   `llama.cpp-legacy-gpu`, the fork modifies **only** `common/fit.cpp` (the GPU
   layer-distribution/fitting algorithm — how many layers go on which GPU). It does
   not touch `ggml/src/ggml-cuda/ssm-scan.cu`, `ssm-conv.cu`, or any other compute
   kernel. Whatever is wrong is either upstream llama.cpp/ggml-cuda behavior on
   Maxwell, or a regression introduced by however far this fork's rebase has drifted
   from upstream.

5. **Not an obvious kernel-level architecture guard.** A source read of
   `ggml/src/ggml-cuda/ssm-scan.cu` (both the Mamba-1 `ssm_scan_f32` and Mamba-2
   `ssm_scan_f32_group` kernels) found no `__CUDA_ARCH__`/compute-capability guard that
   would exclude or special-case Maxwell (CC 5.0/5.2). All arithmetic in the kernel is
   plain fp32 (`expf`, `log1pf`, standard shared-memory + warp patterns), nothing
   obviously requiring newer hardware. `ggml_cuda_pdl_sync()` (the one exotic call in
   the file) is correctly guarded to Hopper-only (`__CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER`)
   and is a no-op on Maxwell — not the cause.

## Suggestive-but-unconfirmed evidence

- Testing `qwen3.8:27b` (same Qwen3.5-hybrid family, used as the Spur-1 Judge base) on
  this fork's M60 pool showed a **related but distinct** pathology: extreme and
  worsening slowdown (~1.74 tok/s at the start of generation, degrading to ~0.15 tok/s
  by token 230), and the llama.cpp server log printed, unprompted:
  ```
  forcing full prompt re-processing due to lack of cache data (likely due to SWA or
  hybrid/recurrent memory, see
  https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
  ```
  This points at the hybrid/recurrent state-caching path specifically, independent of
  the coder-4B model. (Note: `qwen3.8:27b`, the plain, non-fine-tuned tag, separately
  has an **unrelated, already-understood** crash via MTP speculative decoding —
  Xid31 — fixed by using the `-nospec` tag. Do not conflate the two; the slowdown/cache
  behavior above was observed on `qwen3.8:27b-nospec`, with spec-decode already ruled
  out.)
- A compute-sanitizer (`memcheck`/`racecheck`) run against the SSM kernel while
  reproducing the coder-4B failure was set up (CUDA 12.0.1 devel container,
  `compute-sanitizer` confirmed to correctly enumerate the target Tesla M60) but was
  **not completed** — deprioritized mid-session in favor of other work. This is the
  most promising concrete next step; see below.

## Suggested next steps for debugging

1. **Finish the compute-sanitizer run.** Reproduce the coder-4B failure via
   `llama-cli` (not the full Ollama server) directly under
   `compute-sanitizer --tool memcheck` and then `--tool racecheck`, on a single Tesla
   M60 or M10, with a short `-n` (e.g. 16-32 tokens — sanitizer overhead is large).
   Binaries and the model GGUF blob can be extracted directly from a running
   `ollama-legacy` container (`/usr/lib/ollama/llama-cli` +
   `/root/.ollama/models/blobs/sha256-<hash>` for `moe-expert-coder-4b-v2`, hash
   `sha256:d050b119ed9db4571acb993feea5f4457339822c332a2dd98b88c535a394b40c`).
   A CUDA 12.0.1 `nvidia/cuda:12.0.1-devel-ubuntu22.04` container with the GPU attached
   was already confirmed to work for this on N02-M60.
2. **Bisect the `llama.cpp-legacy-gpu` fork's rebase history** against upstream
   `ggml-org/llama.cpp` around the SSM/Mamba kernel files, looking for a point where
   Maxwell (CC 5.0/5.2) SSM behavior changed or broke upstream, or where this fork's
   pinned commit diverged from a later upstream fix.
3. **Test a non-hybrid-but-still-"exotic" architecture** (e.g. a small Mamba-only or a
   different hybrid family, if one is available) on the same fork+Maxwell combination,
   to determine whether the bug is "any model touching the SSM kernels" or specific to
   Qwen3.5's particular hybrid layer interleaving (8 full-attention / 24 linear, with
   the linear layers using `A_log`, `conv1d`, `dt_bias` state).
4. **Check CUB (`cub::BlockLoad`/`BlockStore` with `BLOCK_LOAD_WARP_TRANSPOSE`)
   compatibility with Maxwell's SM design** — the Mamba-1 scan kernel path
   (`USE_CUB`, active since `CUDART_VERSION >= 11070`) uses this; it is a candidate for
   subtle warp-scheduling assumptions that could differ on Maxwell vs. newer
   architectures, though no direct evidence of this was gathered.
5. Confirm whether **`OLLAMA_FLASH_ATTENTION=0` + `f16` KV-cache** (this fork's
   Maxwell-forced config, since Maxwell lacks tensor cores for FA) interacts with the
   hybrid model's own internal attention layers (the 8 full-attention layers) in a way
   that differs from how RTX serves the same model — RTX runs the same
   `moe-expert-coder-4b-v2` model but through an entirely different Ollama build
   (`ollama-github:latest`, official, FA available), so the FA-off path in the fork's
   full-attention layers has not been isolated as a variable on its own (i.e., it has
   not yet been tested: FA-off on RTX for this same model, to separate "FA off" from
   "this fork" as the cause).

## Reference facts for whoever picks this up

- Fork repo: `h3rb3rn/ollama-legacy-gpu` (this repo), Dockerfile
  `dockerfiles/Dockerfile.cuda12-maxwell`.
- llama.cpp fork used by the build: `h3rb3rn/llama.cpp-legacy-gpu`
  (`legacy-gpu-support` branch), local checkout at `/opt/deployment/llama.cpp` on the
  dev machine. Only `common/fit.cpp` is modified from upstream (see its own
  `README-LEGACY-GPU.md`).
- Deployed container image tag: `ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest`.
- Affected hardware: Tesla M60 (compute capability 5.2) and Tesla M10 (compute
  capability 5.0), across hosts N02-M60, N04-RTX (the Tesla/M60 containers there, not
  the RTX ones), and N11-M10.
- Model used for the primary repro: `h3rb3rn/moe-expert-coder-4b-v2` (or the released
  `h3rb3rn/moe-expert-coder-4b` on HuggingFace, same base architecture) — GGUF blob
  hash `sha256:d050b119ed9db4571acb993feea5f4457339822c332a2dd98b88c535a394b40c`.
- Comparison model that works fine on the same fork+hardware: any
  `h3rb3rn/smollm3-expert-*-3b` (dense, no SSM layers).
- As of 2026-09-13, N02-M60's entire fleet (9 containers) was migrated to stock
  `ollama/ollama:0.24.0` (a much older official release, predating this fork's custom
  patches) specifically to get a clean baseline for comparison — N04-RTX's and
  N11-M10's Tesla containers were deliberately left on the
  `ollama-legacy:cuda12-maxwell-latest` fork for A/B comparison. **Re-running the exact
  coder-4B repro on N02-M60's now-stock-0.24.0 fleet, and comparing against N04-RTX's
  or N11-M10's still-on-fork containers, is a fast, already-set-up way to confirm
  whether the bug is fork-specific or also present in stock recent-ish Ollama/llama.cpp
  on Maxwell** — this comparison was suggested but not yet run as of this report.

---

## 2026-09-14 update: two concrete, reproduced findings — re-scope needed

Debugging session on N11-M10 (4x Tesla M10, 8 GiB each), testing `qwen3.5:4b` (base
model of the same architecture family as `moe-expert-coder-4b-v2`, hybrid 8
full-attention + 24 SSM layers, **also vision-capable / carries a CLIP tower** —
`qwen35.vision.*` metadata) against `smollm3:3b` (dense, text-only control) on fresh
single-GPU containers, image `ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-latest`,
env matching the original repro (`OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=f16`).

### Finding 1 — the CLIP/vision compute buffer, not context length, starves GPU offload

At any tested context length (8192 through 65536), `qwen3.5:4b` gets 0/34 or ~1/34
layers offloaded to GPU on a single Tesla M10 - i.e. the model runs almost entirely on
CPU. Root cause visible directly in the llama-server logs:

```
operator(): id=0, n_layer=34, n_part=0, overflow_type=4, mem=3333 MiB   <- fit.cpp's own
                                                                            probe: 34
                                                                            layers would
                                                                            fit in 8120 MiB
...
clip_ctx: CLIP using CUDA0 backend
reserve_compute_meta:      CUDA0 compute buffer size =  4628.88 MiB     <- real reservation
                                                                            once CLIP is
                                                                            accounted for
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/34 layers to GPU
```

The fork's greedy layer-fitting probe (`common/fit.cpp`, the "Greedy Layer-Fill" logic
described in this repo's `CLAUDE.md`) estimates ~3.3 GiB to offload the full model -
comfortably under the 8 GiB budget - but that probe does not account for the vision/CLIP
tower's compute-buffer requirement. `qwen35`/`qwen35moe` checkpoints in this fleet are
Qwen-VL-style omni/vision-capable models (see `qwen35.vision.*` / `qwen35moe.vision.*`
GGUF metadata), and the CLIP compute buffer reservation is ~4.6 GiB, unconditional, and
context-length-independent - it is reserved even for a pure-text prompt with no image
input. On an 8 GiB Maxwell card this alone consumes most of the VRAM budget before a
single decoder layer or KV-cache byte is placed, so the fit-probe's optimistic estimate
collides with reality at the actual reservation step and the algorithm falls back to
(near-)zero GPU layers - independent of context length, independent of GPU pooling (see
Finding 3).

`smollm3:3b` (no CLIP tower) offloads 33/37 layers at the same context length (65536)
without issue and produces correct, coherent output for the exact repro prompt.

This directly confounds the original bug report's "hybrid architecture" conclusion:
every model used in the original repro (`moe-expert-coder-4b-v2`, the `qwen3.8:27b`
judge) is both hybrid/SSM and vision-capable, while the control (`SmolLM3-3B`) is
neither. The two variables (hybrid/SSM vs. dense, vision-capable vs. text-only) were
never separated. It is currently unknown whether the degenerate output is caused by the
SSM layers, the CPU-fallback forced by the CLIP buffer, or both together.

### Finding 2 — MTP speculative decoding crashes with a CUDA illegal memory access

With `qwen3.5:4b` partially offloaded (CPU-heavy, per Finding 1) and MTP speculative
decoding active by default (this fork's `has_mtp()` heuristic in
`scripts/auto-optimize.py` treats any `qwen3*`/`qwen35*` family model as MTP-capable),
the exact bug-report repro prompt (same text, `temperature=0.2, seed=42`) produced, on
one Tesla M10 (GPU index 0):

```
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_backend_cuda_synchronize
  cudaStreamSynchronize(cuda_ctx->stream())
  ...
  common_speculative_impl_draft_mtp::draft
  common_sampler_sample
  llama_context::synchronize
```

i.e. the crash is inside the MTP draft path, not the SSM scan kernel. This is the same
code path (`nextn`/MTP draft head, present as `blk.32.nextn.*` tensors in the GGUF)
implicated in the already-documented, "unrelated" `qwen3.8:27b` Xid31 crash mentioned
above (fixed there via the `-nospec` tag) - except here it surfaces on `qwen3.5:4b` (no
`-nospec` variant currently in use) and manifests as a hard crash rather than a slow
degradation.

Critically, this is not fully deterministic across identical hardware: the identical
model/prompt/seed on a second, otherwise-identical Tesla M10 (GPU index 3, same host)
did not crash. Instead this fork's own auto-optimizer (`scripts/auto-optimize.py`,
triggered automatically via the built-in optimization proxy) ran its MTP draft-length
benchmark (`draft=0/2/4`), concluded `draft_num_predict=0` (MTP disabled) was fastest for
this model, cached that config, and subsequent requests served correct, coherent output -
no degeneration, no crash. This strongly suggests:

- The degenerate "repeats one block forever" symptom from the original report and the
  hard CUDA crash observed here are plausibly the same underlying MTP/draft-state bug,
  just manifesting differently depending on exactly when/where the corruption is read -
  a hard synchronize()-time fault in one case, a stable-but-wrong repeated draft output
  accepted into the stream in the other.
- Whether a request succeeds may depend on whether the auto-optimizer's benchmark sweep
  (which itself exercises the buggy MTP draft path at `draft=2`/`draft=4`) survives long
  enough to reach its own conclusion and disable MTP, or crashes mid-sweep.

### Finding 3 - pooling across all 4 GPUs DOES fix the CLIP-starvation failure mode

`qwen3.6:35b` (`qwen35moe`, 256-expert MoE, same hybrid 10-full-attention/30-SSM layout,
also vision-capable - `qwen35moe.vision.block_count=27`) was deployed on the combined
4x Tesla M10 pool (32 GiB, `OLLAMA_SCHED_SPREAD=true`, ctx=65536). Result: **41/42 layers
offloaded to GPU** (near-full residency - CUDA0 hosts the 4657.81 MiB CLIP compute buffer
plus its share of decoder layers, CUDA1-3 hold the rest), vs. 0/34 for `qwen3.5:4b` on a
single 8 GiB M10 (Finding 1). With enough total pool VRAM, the same fixed ~4.6 GiB CLIP
buffer is a much smaller fraction of the budget and no longer starves the decoder layers.

Running the exact bug-report repro prompt (`temperature=0.2, seed=42`) against this
config produced **long, coherent, correct, non-repeating chain-of-thought reasoning**
about the SPSC ring buffer problem (padding to avoid false sharing, correct
acquire/release pairing discussion, etc.) - no degenerate repetition, no crash. (Note:
this model emits an extensive `thinking` block before any final `response` text; both of
our test runs, at `num_predict=150` and `600`, were consumed entirely by `thinking` and
hit `done_reason="length"` before reaching the final answer - that is a test-harness
token-budget artifact, not a sign of a problem; the thinking content itself was
consistently on-topic and non-degenerate throughout.)

**This is a meaningful data point against "GPU pooling is not the cause" as originally
stated**, at least for the CLIP-starvation failure mode: pooling *does* help here, just
not for the reason anyone originally tested it (VRAM headroom for the vision buffer, not
avoiding some SSM cross-GPU state-splitting bug). The original report's pooling test used
a 4x Tesla M60 pool (~15.4 GiB combined per the `worker-tesla` docs) with an unspecified
model size - if that pool's VRAM was similarly tight relative to model + CLIP buffer
size, it plausibly hit the *same* per-GPU starvation Finding 1 describes, on the
CUDA0-equivalent coordinator GPU, even while pooled. Pool size relative to (weights +
~5-6 GiB CLIP buffer) looks like the actual variable, not "pooled vs. not pooled" per se.

### Revised suggested next steps

1. Separate the two confounded variables. Test a vision-capable but non-hybrid dense
   model, and a hybrid/SSM model without a vision tower (a text-only Qwen3.5 variant, if
   one exists, or a GGUF with the `mmproj`/CLIP tensors stripped), to determine
   independently whether CLIP-buffer starvation alone, hybrid architecture alone, or both
   together are required to reproduce the degenerate output.
2. Check whether the fork's `fit.cpp` probe can be fixed to account for the CLIP compute
   buffer before committing to a layer count, instead of discovering the shortfall only
   at the real reservation step and collapsing to ~0 GPU layers.
3. Reproduce Finding 2 (MTP crash) with `-nospec`-equivalent handling for `qwen3.5:4b`
   (force `draft_num_predict=0` / disable MTP from the very first request, bypassing the
   auto-optimizer's own benchmark sweep) and confirm whether the degenerate-repetition
   symptom from the original report disappears entirely when MTP is disabled from the
   start - this is now the single highest-value experiment to run.
4. The compute-sanitizer plan (original next step 1) is still valid but should be
   re-targeted at the MTP draft kernels / `common_speculative_impl_draft_mtp`, not
   `ssm-scan.cu`, given Finding 2.
5. Re-run the original M60-pool repro (`qwen3.6:35b`/`qwen3.8:27b` judge on the 4x M60
   pool, ~15.4 GiB combined) and log the actual layer-offload count, the same way this
   session did for N11-M10. Given Finding 3, check whether that pool was VRAM-tight
   enough (weights + ~5-6 GiB CLIP buffer vs. ~15 GiB total) to reproduce Finding 1's
   starvation pattern despite being "pooled" - this would resolve whether the original
   "pooling is not the cause" conclusion holds once CLIP-buffer VRAM pressure is
   controlled for.

### Test artifacts from this session (N11-M10)

- `compose/docker-compose.n11-single-test.yml` - 4x single-GPU test containers
  (`ollama-tesla-1..4`, ports 11434-11437, one Tesla M10 each via `device_ids`).
- `compose/docker-compose.n11-multigpu-test.yml` - 4-GPU pool test container
  (`ollama-multigpu-test`, port 11434, `count: all`).
- Models used: `qwen3.5:4b` + `hf.co/bartowski/HuggingFaceTB_SmolLM3-3B-GGUF:q4_K_M`
  (single-GPU), `qwen3.6:35b` + `hf.co/h3rb3rn/sovereign-judge-olmo31-32b:Q4_K_M`
  (4-GPU pool - note: the plain `hf.co/bartowski/allenai_Olmo-3.1-32B-Instruct-GGUF`
  tag fails to load entirely on this fork, unrelated bug - see below).
- `OLLAMA_CONTEXT_LENGTH=65536` used throughout (empirically the largest round value
  that still gave reasonable GPU residency for the dense control models; irrelevant for
  `qwen3.5:4b`'s CLIP-starvation failure, which is context-length-independent per
  Finding 1).

### Unrelated bug noticed in passing: broken Jinja template crashes model load

`hf.co/bartowski/allenai_Olmo-3.1-32B-Instruct-GGUF:Q4_K_M` fails to load at all on this
fork (`llama-server` exits immediately, every time, regardless of context length or GPU
layout):

```
error: Unable to generate parser for this template. Automatic parser generation failed:
Error: Unknown (built-in) filter 'tojson' for type Undefined (hint: 'tools')
```

The GGUF's embedded tool-calling Jinja chat template uses the `tojson` filter, which this
fork's minijinja-based template engine doesn't implement. An `ollama create` with a
custom `TEMPLATE` in the Modelfile does **not** work around this - the fork still parses
and fails on the GGUF's embedded template at `llama-server` startup regardless of the
Modelfile. `"raw": true` in the API request doesn't help either (the parser runs at
server-start, before any request). Workaround used in this session: switch to a
differently-packaged GGUF of the same model (`hf.co/h3rb3rn/sovereign-judge-olmo31-32b`,
already in use on N04-RTX) that has a working template. Worth a separate, small bug
report if `tojson` support is easy to add to the minijinja integration - this will hit
any tool-calling-template GGUF, not just Olmo.

---

## 2026-09-15 update: Phase 0 fixes implemented and verified on N11-M10

A broader optimization + debugging pass (deep research into Maxwell/CUDA12/llama.cpp
community knowledge, a full source read of both this repo and the vendored
`/opt/deployment/llama.cpp` checkout) produced a phased plan. Phase 0 ("safe, do-first,
no rebuild required") has been implemented and empirically verified on N11-M10:

- **MTP force-disabled for the qwen3/qwen35 family** (`scripts/auto-optimize.py`,
  `has_mtp()` now denylists this family via `MTP_CRASH_DENYLIST_ARCH_SUBSTRINGS`,
  skipping the Phase-2 draft-sweep entirely and caching `draft_num_predict=0` directly).
  **Verified**: re-ran the exact repro prompt (`temperature=0.2, seed=42`) against
  `qwen3.5:4b` on GPU index 0 - the same GPU that previously crashed with a CUDA
  illegal-memory-access inside the MTP draft path - with `draft_num_predict=0` forced
  from the first request. Result: no crash, 200/200 tokens generated, coherent
  non-degenerate output. This is strong evidence the degenerate-repetition symptom and
  the hard crash are both downstream of the same MTP instability, and that disabling MTP
  alone (independent of the still-open Finding 1 CLIP-buffer fix) already prevents both
  failure modes.
- **CPU thread count fixed**: `LLAMA_ARG_THREADS=4` / `LLAMA_ARG_THREADS_BATCH=4` added
  to the test compose files (N11-M10 has 4 physical cores; Ollama's own default was
  silently using only 2). Verified via container logs (`n_threads = 4`, was `n_threads = 2`).
- **`--split-mode row` confirmed broken for MoE models on this hardware, `layer` confirmed
  working**: A/B tested `qwen3.6:35b` (256-expert MoE) on the 4x-M10 pool.
  `OLLAMA_SPLIT_MODE=row` fails immediately (~4.6s) with
  `error loading model: device CUDA0 does not support split buffers` - a hard load
  failure, not just a slowdown. `OLLAMA_SPLIT_MODE=layer` (the fork's own code default
  when unset) loads successfully (~89s cold load with the thread fix applied) and
  generates coherent output at ~7 tok/s. This empirically confirms and extends
  `patch-ollama-dynamic-pool.py`'s existing comment ("row causes CUDA errors with MoE
  models (llama4:scout)") to `qwen3.6:35b` as well - `layer` should remain the only
  split-mode used for MoE models on Maxwell pools; the multi-GPU test compose file's
  `OLLAMA_SPLIT_MODE` default was corrected to `layer` after this was found (an earlier,
  now-fixed version of the test compose accidentally defaulted to `row`).
- **`tojson` filter fix applied** to `/opt/deployment/llama.cpp/common/jinja/value.cpp`:
  `value_undefined_t::get_builtins()` was missing a `{"tojson", tojson}` entry that every
  other value type already has (confirmed by reading `value_to_json_internal()`, which
  already explicitly handles `val->is_undefined()` -&gt; `"null"` - the serializer was
  always undefined-safe, only the filter *registration* was missing). One-line fix, not
  yet rebuilt/deployed (requires a Docker image rebuild - queued for the same rebuild
  pass as Finding 1's CLIP-buffer fix, see below). This is vendored/unmodified-from-upstream
  code (confirmed via `git log` on the file - no fork-specific commits), worth upstreaming
  in addition to carrying locally.
- **CI Docker-layer-cache bug**: root-caused precisely - `PATCH_GUARD` in
  `scripts/patch-llama-tier-fitting.py` is a fixed string used both as the idempotency
  marker *and* as a literal runtime env var name embedded in the generated C++
  (`std::getenv(PATCH_GUARD)`), so it cannot be casually renamed without also checking
  for collisions with other scripts' env var names and without re-verifying the
  tier-threshold GPU-distribution logic end-to-end (this logic is N04-RTX-specific, not
  exercised by N11-M10's homogeneous pool) - left as a flagged follow-up rather than
  changed hastily. The safe half was applied instead: `.github/workflows/build.yml`'s GHA
  cache `scope=` keys were bumped (`cuda12-maxwell-v2` -&gt; `-v3`, `cuda11-legacy` -&gt;
  `-v2`) to force a fresh, unpoisoned cache on the next CI build.

### Key new finding that reframes the FA question (Finding 1 follow-up)

A full source read of `/opt/deployment/llama.cpp/ggml/src/ggml-cuda/fattn.cu` and
`fattn-tile.cu` found that **Flash Attention is not usable on this hardware at all
today** - `dockerfiles/Dockerfile.cuda12-maxwell` passes `-DGGML_CUDA_FA=OFF` to CMake
(present since the fork's very first commit), which undefines `FLASH_ATTN_AVAILABLE` and
makes `ggml_cuda_get_best_fattn_kernel()` unconditionally return `BEST_FATTN_KERNEL_NONE`
- the TILE/VEC/WMMA/MMA kernels are never even compiled into the binary. This directly
contradicts this repo's own `CHANGELOG.md` (`[v0.30.0]`, claims FA-on-all-architectures
via the TILE kernel, with a measured 11.4 GiB -&gt; 22-278 MiB compute-buffer shrink) and
`scripts/patch-ollama-dynamic-pool.py`'s runtime FA-enable logic (currently dead code).
The Dockerfile's own comment justifying `GGML_CUDA_FA=OFF` ("Flash Attention requires CC
&gt;= 8.0") is itself factually wrong for the TILE kernel path.

Critically, `fattn.cu`'s kernel dispatch (`case 256:` in the head-size switch,
`fattn-tile.cu`'s explicit `ggml_cuda_flash_attn_ext_tile_case&lt;256, 256&gt;` instantiation,
and a compiled `fattn-tile-instance-dkq256-dv256.cu` template instance) confirms
**head_dim=256 (Qwen3.5/3.6's exact configuration) is explicitly supported by the TILE
kernel** - this had been an open question (external research suggested only 64/128 were
supported) that a direct source read resolved in the more favorable direction. Re-enabling
FA plausibly helps free VRAM headroom for the exact model family Finding 1 is about, not
just other models - but this needs an actual gated rebuild-and-test pass (real residual
risk: a documented, unconfirmed-root-cause upstream precedent exists for old-Tesla
multi-GPU+FA instability, `ggml-org/llama.cpp#12990`) before it can be trusted or shipped.
**Not yet attempted - queued as the next, higher-risk phase, pending sign-off on scope
and risk tolerance.**

### Still open (queued, not yet started)

- Phase 1: compute-sanitizer run targeted at the MTP draft kernels (root-cause Finding 2
  properly, rather than only mitigating it via the denylist above).
- Phase 2: gated FA rebuild (`GGML_CUDA_FA=ON`), full correctness + compute-buffer +
  performance verification matrix, before ever promoting past a test tag.
- Phase 3: `common/fit.cpp` CLIP-buffer-awareness fix (Finding 1's actual root-cause fix -
  the probe in `common_get_device_memory_data_impl()` never constructs or accounts for a
  CLIP/mtmd context at all, confirmed via a direct read of `common/fit.cpp` and
  `common/common.cpp:1196-1210`) + the hybrid-vs-vision-capable confound test matrix.
- Phase 3.3: re-run the M60-pool repro with explicit layer-offload logging (needs
  N02-M60/N04-RTX access, out of this session's N11-M10 scope).

---

## 2026-09-15 update: Finding 1 corrected — it was never a fit.cpp bug, and it's fixed

Follow-up to the Phase 3 work above. Wrote and built a `common/fit.cpp` CLIP-margin
patch (inflating `fit_params_target[0]` by a fixed estimate before fitting, as
Finding 1 originally proposed) and tested it on N11-M10. **It had zero effect on the
outcome** — still 0/34 layers offloaded. Digging into why:

`tools/server/server-context.cpp` (upstream llama.cpp, unmodified by this fork)
**already** does exactly what the patch was trying to add: it calls
`mtmd_get_memory_usage()` to get mmproj's own worst-case memory estimate, logs it
(`"[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB"`), and adds it
directly to `params_base.fit_params_target[i]` per-device — **before** `common_init_result`
/ `common_fit_params` (the code Finding 1 targeted) ever runs. Confirmed via a
diagnostic build that logged `fit_params_target[0]` at exactly the point Finding 1's
patch would have modified it: **already 7028 MiB**, entirely from this pre-existing
upstream mechanism (mtmd's ~5.4 GiB worst-case estimate for `qwen3.5:4b`'s vision tower,
plus base overhead) — before the patch's own addition ever ran. Adding another ~4.75 GiB
on top just made an already-negative budget more negative; the CLIP-margin patch has
been **reverted** (both the llama.cpp-fork commit and the Ollama-fork's Python patch
script), since it was redundant with, and briefly stacked on top of, upstream's own
already-correct accounting.

**So Finding 1's actual mechanism was right (vision tower reservation starves GPU
layers on a tight single 8 GiB card) but the diagnosis of *where* the gap was wrong**
— there's no missing CLIP-awareness in the fitting probe. The real situation: mtmd's
worst-case estimate (~5.4 GiB) plus base overhead genuinely doesn't leave enough of an
8 GiB M10's VRAM for even one `qwen3.5:4b` decoder layer (needs ~2 GiB minimum) once
the vision tower is reserved on the same device.

### The actual, confirmed fix: `LLAMA_ARG_MMPROJ_OFFLOAD=false`

llama-server already exposes `--mmproj-offload` / `--no-mmproj-offload`
(env: `LLAMA_ARG_MMPROJ_OFFLOAD`, default enabled) to control whether the multimodal
projector runs on GPU at all. Setting it to `false` keeps CLIP on CPU, which removes
its VRAM reservation entirely instead of trying to shrink or reallocate it. Tested on
N11-M10 (GPU3, `qwen3.5:4b`, single Tesla M10):

| Config | GPU layers | Output | Gen. speed |
|---|---|---|---|
| `mmproj` on GPU (default) | 0/34 | N/A (CPU-only) | 1.2-4.2 tok/s (CPU-bound) |
| `mmproj` on CPU (`LLAMA_ARG_MMPROJ_OFFLOAD=false`) | **34/34** | correct, coherent, non-degenerate (exact bug-report repro prompt) | **5.7 tok/s** |

This is now the fastest single-GPU M10 result recorded this session — faster than the
CPU-fallback case by ~35-80%, with full GPU residency and no quality loss. It's a pure
deployment-config fix, no code changes needed. Added to
`compose/docker-compose.n11-single-test.yml` as the default for single-GPU Maxwell
test deployments. Irrelevant for text-only models (no mmproj to offload) and not
needed on the 4-GPU pool (Finding 3 already showed `qwen3.6:35b` reaching 41/42 layers
with mmproj on GPU there — combined 32 GiB is enough headroom even for the worst-case
estimate).

**Remaining open question**: does keeping CLIP on CPU meaningfully slow down actual
image/vision inference requests (as opposed to the text-only benchmark used here)? Not
yet tested — this session only exercised text prompts. Worth a follow-up test with a
real multimodal request before treating this as a universal default for any
vision-capable deployment, as opposed to specifically for text-heavy single-GPU
Maxwell nodes where vision is a secondary/rare use case.

---

## 2026-09-15 (cont.) — RESOLVED: original repro re-run on real M60 hardware

Context: a follow-up optimization campaign (`PERFORMANCE-OPTIMIZATION-LOG.md`) rebuilt
the fork with `GGML_CUDA_FA=ON` (Phase 2) after confirming via source read that the
TILE kernel explicitly supports head_dim=256 (Qwen3.5/3.6's configuration). Correctness
was verified on N11-M10 (Tesla M10) across three model classes — all passed, no crashes,
no Xid errors, coherent output, compute buffer shrank to 140-494 MiB matching the
CHANGELOG's historical claim.

With N02-M60 (the **original** host this bug was first reported on) made fully available
for testing, the **exact original repro model** (`moe-expert-coder-4b-v2`, same GGUF blob
hash as the original report) was re-run on a single Tesla M60 die (GPU index 0) with the
full accumulated fix set:

- `GGML_CUDA_FA=ON` (Phase 2 rebuild, `cuda12-maxwell-fa-test` tag)
- `LLAMA_ARG_MMPROJ_OFFLOAD=false` (keeps CLIP off GPU — the actual Finding-1 fix)
- MTP fix active (the shipped image's `has_mtp()` denylist forces `draft_num_predict=0`
  for the `qwen35` family this model belongs to — no explicit override needed, this is
  the real default behavior)
- Same prompt, same `temperature=0.2, seed=42` as the original report

**Result:** `load_tensors: offloaded 33/33 layers to GPU`, `flash_attn = enabled`,
**20.48 tok/s**, and — critically — **coherent, non-repeating output**:

```
[IMPLEMENTING CORRECTNESS VERDICT]
The following lock-free ring buffer implementation is correct under the conditions that:
1. The buffer size is a power of 2 (enables bitwise modulo)
2. The producer and consumer are the only threads accessing the buffer
3. std::atomic_flag is used correctly with release/acquire semantics
4. The head and tail pointers are properly initialized and updated
[... continues as one coherent, non-repeating analysis, cut off by token limit ...]
```

Note the response still opens with the same `[IMPLEMENTING CORRECTNESS VERDICT]` header
seen in the original report's degenerate output — that phrasing is apparently a stable
stylistic trait of this specific fine-tune (consistent with its likely role as a
"judge"/verifier expert in a larger MoE-expert ensemble, per this deployment's naming
conventions, rather than a raw code generator) — **not** a sign of the bug persisting.
The defining symptom of the original report — the same ~4-line block repeating verbatim
until the token cap — is completely gone.

**This closes the investigation.** None of the three ruled-out/re-scoped mechanisms
(SSM/Mamba kernel, GPU pooling, this fork's layer-fitting patches) were ever the cause.
The real, now-confirmed cause was two independent, compounding issues, both fixed by
configuration/build changes rather than any change to the SSM code path:

1. **CLIP/vision-tower GPU memory competing with the text decoder** for VRAM on
   small-VRAM Maxwell cards (fixed: `LLAMA_ARG_MMPROJ_OFFLOAD=false`).
2. **MTP speculative decoding instability** for the `qwen3`/`qwen35` family on Maxwell
   (fixed: denylisted by default in `scripts/auto-optimize.py`).

Flash Attention (`GGML_CUDA_FA=ON`) was not strictly required to fix the degenerate
output (the mmproj-offload fix alone already resolved it on N11-M10 in the earlier test),
but is validated as safe and beneficial for single-GPU deployments (smaller compute
buffers, full layer offload alongside CLIP staying off-GPU) — see
`PERFORMANCE-OPTIMIZATION-LOG.md` for a separately-discovered limitation (FA=ON is not
yet safe for large multi-GPU MoE pools, unrelated to this bug, tracked there).

### Finding 2, closed: root-caused via compute-sanitizer

Follow-up `compute-sanitizer` work (full details in `PERFORMANCE-OPTIMIZATION-LOG.md`,
Phase 1) found the exact cause of the MTP crash: **`--tool racecheck` found 256 confirmed
shared-memory race hazards** in a kernel named `sgemm_32x32x32_NT_vec` (Write at shared-mem
offset 0x68 racing a Read at offset 0x78), and the exact same test run reproduced the
**identical CUDA "illegal memory access" crash with the identical stack trace** as the
original report (`common_speculative_impl_draft_mtp::draft` → `common_sampler_sample` →
`llama_context::synchronize` → `ggml_backend_cuda_synchronize`). A separate,
sanitizer-free repro also produced a **100% deterministic garbage draft-token value**
(`-839448741`, identical across repeated runs) with `--tool memcheck` reporting zero
errors — consistent with a race (not a memory-safety violation) that sometimes crashes
and sometimes just yields stale/wrong data, i.e. **both symptom types from the original
report are the same underlying race, not two separate bugs.**

The kernel name (`sgemm_32x32x32_NT_vec`) does not appear anywhere in the `ggml`/
llama.cpp source tree and follows classic closed-source BLAS-library tile-kernel naming
(not ggml's own naming conventions) — almost certainly an internal legacy cuBLAS SGEMM
kernel that NVIDIA's CUDA toolkit uses for plain fp32 matmuls on tensor-core-less Maxwell
hardware, exactly the code path the MTP draft's nextn-embedding projection hits (see this
session's earlier finding that Maxwell falls back to `cublasSgemm` for non-quantized
matmuls, having neither MMQ/dp4a nor fp16-cuBLAS acceleration available).

**Conclusion: this is very likely a bug inside NVIDIA's own closed-source cuBLAS,
un-patchable from this project, on end-of-life Maxwell hardware with no further vendor
support expected.** The MTP denylist already shipped in Phase 0
(`scripts/auto-optimize.py`'s `has_mtp()`) is therefore not a stopgap but the correct,
permanent fix for this hardware segment. No further action planned on this finding.
