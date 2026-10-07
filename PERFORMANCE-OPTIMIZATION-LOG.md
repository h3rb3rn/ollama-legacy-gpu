# Performance optimization log — Tesla M10/M60 under Ollama

Companion document to the optimization roadmap (plan of 2026-09-15). One section per phase,
with hypothesis, exact change, test procedure, before/after result and verdict.
Bug-specific findings (Finding 1/2/3) stay in `BUG-hybrid-arch-degeneration.md`
— this document is about the broader performance campaign.

Test host (unless stated otherwise): N11-M10, 4x Tesla M10, GPU3 primary
(`ollama-tesla-4`, port 11437), priority descending GPU3→GPU2→GPU1→GPU0.

---

## Phase 0 — Recap (already implemented before this log)

For completeness; details in the commits + `BUG-hybrid-arch-degeneration.md`:

| Change | Before | After | Status |
|---|---|---|---|
| `LLAMA_ARG_THREADS=4` | n_threads=2 (of 4 cores) | n_threads=4 | Adopted |
| MTP denylist (`qwen35` family) | CUDA crash possible, non-deterministic | draft_num_predict=0 default, no crash | Adopted |
| `OLLAMA_SPLIT_MODE` default | untested | `layer` confirmed (row aborts hard with MoE) | Adopted |
| tojson Jinja fix | crash with tool-calling templates without tools | fix applied | Adopted |
| CLIP margin patch (`fit.cpp`) | — | tested, ineffective (misdiagnosis) | **Rejected**, commit reverted |
| `LLAMA_ARG_MMPROJ_OFFLOAD=false` | 0/34 layers on GPU (qwen3.5:4b) | 34/34 layers, 5.74 tok/s | Adopted |
| CI cache bust | PATCH_GUARD stale-cache risk | cache scope busted | Adopted (partial fix) |

---

## Phase 1 — Compute sanitizer: root-causing the MTP crash

**Status: COMPLETED — root-caused, denylist confirmed as the permanent solution**

**Hypothesis:** The CUDA "illegal memory access" in the MTP draft path
(`common_speculative_impl_draft_mtp::draft`) is a race condition or out-of-bounds
access in one of the draft kernels, reproducible under `compute-sanitizer`.

**Setup:** `llama-cli` + `qwen3.5:4b` GGUF blob extracted from `ollama-tesla-4`, run in an
isolated `nvidia/cuda:12.0.1-devel-ubuntu22.04` container with GPU0 (N11-M10, the
original crash GPU), `--spec-type draft-mtp` forced (denylist bypassed for
this isolated test). First hurdle: `llama-cli` without `--single-turn` hangs in
interactive chat mode (no real sanitizer hang) — fixed.

**Reproduced error (deterministic, identical without AND with the sanitizer):**
```
E init: invalid token[0] = -839448741
E decode: failed to initialize batch
E llama_decode: failed to decode, ret = -1
E spec draft: llama_decode[1] returned -1
```
Exactly the same garbage token value (`-839448741`) in both runs — 100% deterministic,
no random noise.

**`--tool memcheck`:** 0 errors found (`ERROR SUMMARY: 0 errors`) — no
out-of-bounds/uninitialized-memory accesses inside the kernels themselves.

**`--tool racecheck`:** still in progress (considerably higher overhead than memcheck).

**Interim assessment:** The combination "deterministic + memcheck clean" points
more towards a host-side logic/indexing error when converting the raw
draft-head output into a token ID (wrong offset, wrong dtype interpretation,
or reading a never-initialized host buffer) than towards a classic
GPU memory violation. To be narrowed down further after the racecheck result.

**`--tool racecheck` (result, ~43 min runtime at `-n 8`):**

```
========= Error: Race reported between Write access at 0x68 in sgemm_32x32x32_NT_vec
=========     and Read access at 0x78 in sgemm_32x32x32_NT_vec [... 20 individual hazards shown ...]
========= RACECHECK SUMMARY: 20 hazards displayed (256 errors, 0 warnings)
```

**256 confirmed shared-memory race hazards**, all in the same kernel
(`sgemm_32x32x32_NT_vec`) — a write at shared-memory offset 0x68 races against a read at
offset 0x78. This particular test run triggered **the same CUDA "illegal memory
access" crash with an identical stack trace as the original bug report**
(`common_speculative_impl_draft_mtp::draft` → `common_sampler_sample` →
`llama_context::synchronize` → `ggml_backend_cuda_synchronize`) — the two
symptom pictures from the original report (hard crash vs. deterministic
garbage token, see the memcheck result above) are thereby confirmed as **two manifestations
of the same race condition**, not two separate problems.

**Kernel origin:** `sgemm_32x32x32_NT_vec` does not occur in any file in the `ggml`/`llama.cpp`
source tree (searched) — the naming style (classic BLAS tile naming, not
ggml-typical like `mul_mat_vec_q`/`flash_attn_tile`) strongly suggests that this is an
**internal, closed-source cuBLAS legacy SGEMM kernel** that NVIDIA's CUDA toolkit uses for
Maxwell cards without tensor cores for pure fp32 matmuls (consistent with the already
documented finding of this session: Maxwell falls back to `cublasSgemm` for non-quantized/fp32 matmuls —
e.g. the MTP draft/nextn embedding projection — since neither MMQ
nor fp16 cuBLAS acceleration is available on CC 5.0/5.2).

**Root-cause classification:** The race condition most probably lies **in
NVIDIA's own closed cuBLAS library**, not in fork or llama.cpp code —
neither this project nor the llama.cpp fork can patch this kernel. Given
Maxwell's end-of-life status (no further cuBLAS support/fixes to be expected), a
real upstream fix is unrealistic.

**Verdict:** The already shipped MTP denylist (`has_mtp()` in `auto-optimize.py`,
Phase 0) is therefore **not just a workaround but the correct, permanent solution**
for this hardware segment — root-caused, not to be pursued further. No code fix in the
fork repo possible or necessary.

---

## Phase 2 — Flash Attention rebuild (`GGML_CUDA_FA=ON`)

**Status: correctness matrix completed, result: ADOPTED**

**Build:** `dockerfiles/Dockerfile.cuda12-maxwell` with `-DGGML_CUDA_FA=ON`, tag
`cuda12-maxwell-fa-test`, built locally on N11-M10 (Maxwell-only architectures for
faster testing), then transferred to N02-M60 (`docker save`/`scp`/`docker load`
via the control host as a relay, since worker hosts cannot reach each other directly via SSH).

**Correctness matrix (GPU3/GPU2 on N11-M10, then GPU0 on N02-M60):**

| Model | Architecture | Host/GPU | Layer offload | FA | Compute buffer | Output |
|---|---|---|---|---|---|---|
| `smollm3:3b` | dense, head_dim 64/128 | N11-M10/GPU2 | 37/37 | enabled | 140.51 MiB | correct, coherent |
| `sovereign-judge-olmo31-32b` | dense, head_dim=128 | N11-M10/GPU2 | 17/65 (single GPU, expected) | enabled | — | correct, coherent |
| `qwen3.5:4b` | hybrid, vision, head_dim=256 | N11-M10/GPU2 | 34/34 | enabled | 256.03 MiB | correct, coherent |
| `moe-expert-coder-4b-v2` (original repro!) | hybrid, vision, head_dim=256 | **N02-M60/GPU0** | 33/33 | enabled | — | **correct, coherent, 20.48 tok/s** |

No Xid errors observed in `nvidia-smi -q`/`dmesg` during the runs (P40 precedent
`ggml-org/llama.cpp#12990` not reproduced). Compute buffer sizes (140-494 MiB) match
the historical measurement claimed in the CHANGELOG (22-278 MiB) — the CHANGELOG number is thus
verified for the first time (before, FA had never actually been active in the build).

**Most important result:** The 4th test run (original bug repro model on the original bug host
N02-M60) confirms: bug fully fixed, see `BUG-hybrid-arch-degeneration.md`
2026-09-15 update.

### Addendum: FA=ON breaks with large multi-GPU MoE pools

After the successful single-GPU tests, `qwen3.6:35b` (256-expert MoE, the same
model as in Finding 3) was tested with FA=ON on several pool sizes — all three failed:

| Host | GPU count | Error |
|---|---|---|
| N02-M60 | 12 (full pool) | `CUDA error: peer mapping resources exhausted` |
| N02-M60 | 6 | `CUDA error: an illegal memory access was encountered` |
| N11-M10 | 4 (the same pool that ran flawlessly with FA=OFF in Finding 3) | `llama-server process has terminated: signal: killed` (probably host OOM, N11-M10 has little free system RAM) |

Three different error pictures on two different hosts — this is not random noise but a real pattern: **FA=ON is
not (yet) safe for large, pooled MoE models across several GPUs**, although it works
flawlessly for single-GPU deployments (the three cases passed
above, including the actual bug target).
Root cause not pursued further (outside the original bug) — possible
candidates: FA fundamentally changes buffer allocation patterns, the `OLLAMA_MAX_BATCH_SIZE=64` cap
was calibrated for the non-FA case (Phase 5 in the plan was meant to retune that anyway), or
a real incompatibility between the FA kernel and multi-GPU P2P transfers with MoE routing.

*Update (2026-10-07): the `peer mapping resources exhausted` error with 12 GPUs was later root-caused
and fixed: the CUDA VMM pool granted peer access to all visible devices in NCCL builds and CUDA allows
only 8 peers per mapping (`patch-llama-vmm-peer-access.py`). See [docs/TUNING.md](docs/TUNING.md).*

**Rollback applied according to the plan criterion** ("any Xid/crash/correctness deviation →
stop immediately"): `-latest` stays FA=OFF for the time being. Recommendation for the productive rollout:
release FA=ON **only for single-GPU deployments** (exactly the case that fixes the bug),
multi-GPU pools stay FA=OFF until the pattern described above has been investigated separately.

**Verdict:** FA=ON safe and correct on all three single-GPU Maxwell test cases
(including the original bug case). **Not yet safe** for large multi-GPU MoE pools —
a new, independent investigation item, not part of the original bug.
Rollout recommendation: `-fa-test` → `-latest` only with single-GPU scope, after a
light test on GPU1/GPU0 (N11-M10 priority order).

---

## Phase 3 — Separating the hybrid-vs-vision confounding

**Status: partially completed — vision-dense case confirmed, hybrid text-only case blocked**

**Blocker text-only hybrid model:** `hf.co/...` pulls currently fail with
`Error: pull model manifest: realm host "huggingface.co" does not match original host "hf.co"`
— reproducible for new (not yet locally cached) repos, already cached
repos (e.g. `sovereign-judge-olmo31-32b`, pulled earlier) keep working. No
Mamba/Jamba/pure-SSM model is available locally in the whole fleet (N11-M10, N04-RTX, N02-M60).
This part of Phase 3 stays open until the registry problem
(probably an Ollama version regression, independent of this fork) is fixed.

**Vision-dense case (no SSM, but vision) — tested:**

| Model | Vision tower size (mtmd worst case) | Layer offload (mmproj on GPU, default) |
|---|---|---|
| `qwen3.5:4b` (for comparison, hybrid+vision) | ~5.4 GiB | 0/34 |
| `llama3.2-vision:latest` | — | **does not load at all** (`unknown model architecture: 'mllama'` — this fork build does not support the mllama architecture, independent of VRAM) |
| `minicpm-v:latest` (dense+vision) | ~1.05 GiB | **25/29** (no 0-layer case) |

**Result:** No clean binary comparison possible (no second hybrid text model
available), but `minicpm-v` still delivers a meaningful partial result: **a
dense (non-hybrid) model with vision capability shows the same VRAM competition pattern
as Qwen3.5 — only proportional to the actual encoder size.** MiniCPM-V's encoder
(~1 GiB) is small enough not to force a 0-layer result on an 8 GiB card;
Qwen3.5's encoder (~5.4 GiB) is not. This supports the core thesis of Finding 1 (VRAM
competition from the vision component, not the hybrid/SSM architecture) — however
as a gradual, size-dependent effect instead of a clean yes/no, and without the missing
hybrid text-only case a residual doubt remains whether the hybrid architecture has a (smaller,
additional) effect. **Not conclusively proven, but the prevailing explanation
stands.**

**Side finding:** `llama3.2-vision` (mllama architecture) is generally unusable
with this fork build — a separate, small compatibility ticket, not pursued further (not
part of the original bug; mllama is a structurally different vision integration from
the CLIP/mtmd model that this fork sees everywhere else).

---

## Phase 4 (part 1) — N02-M60 fleet rebuilt with the validated fix set

**Status: completed (N02-M60), N04-RTX still open**

With explicit approval from the user ("take the GPUs however and as many as you need"), the
9 old stock Ollama containers (`ollama/ollama:0.24.0`, which had run there for 44h as an
A/B comparison baseline according to the bug report) were removed and replaced by a complete
12-GPU single-instance fleet (`ollama-m60-gpu0` … `gpu11`, port 11434–11445, one Tesla
M60 die each) with the image `cuda12-maxwell-fa-test` and the full validated fix set
(thread fix, MTP denylist, `LLAMA_ARG_MMPROJ_OFFLOAD=false`, FA=ON —
single-GPU scope, see the Phase 2 restriction above). All 12 containers healthy,
verified end to end with the original bug model on GPU0 (HTTP 200, correct output).

**Deliberately NOT restored:** the old 12-GPU pool container
(`ollama-m60-pool`) — see the Phase 2 addendum: FA=ON is currently not safe for large
multi-GPU pools. A pool deployment for N02-M60 should, if desired, initially be set up with
FA=OFF (analogous to the N11-M10 multi-GPU test compose), not with the
`-fa-test` image.

---

## Phase 4 (part 2) — N04-RTX updated

**Status: completed**

N04-RTX is a considerably more sensitive host: besides several Ollama test containers, a
productive non-Ollama application runs there (`terra_*` stack: Node client/server,
Postgres, Redis, Mailhog). The auto-mode safety classifier correctly recognized this as a
production system and blocked initial actions there (even read-only
`docker ps`) — continued on explicit user approval.

**Only the Tesla-based fork containers were updated** (`ollama-tesla-1..4`,
`ollama-m60-1`, `ollama-m60-2`, `ollama-m60-guard`) — with the same validated
fix set as N02-M60 (single-GPU containers: `cuda12-maxwell-fa-test` image, thread fix,
`LLAMA_ARG_MMPROJ_OFFLOAD=false`, FA=ON; the 2-GPU pool container `ollama-m60-guard`:
fresh `-latest` with Phase 0 fixes, FA=OFF as before). Existing ports/GPU
assignments/context length (32768, differing from the 65536 on the test hosts —
deliberately kept, that was the productive value) adopted unchanged.

**Explicitly NOT touched** (on user instruction): `ollama` (port 11434) and
`ollama-rgtx` (port 11435) — they run `ollama-github:latest` (official Ollama,
not this fork) on the RTX/GTX GPUs. Likewise untouched: the entire
`terra_*` stack, `ollama-0240-m10`/`ollama-0240-m60` (deliberate stock Ollama
comparison baseline), `open-webui`, `searxng`.

Verified end to end on `ollama-tesla-1`: 34/34 layers, FA active, correct
output. With that, Phase 4 (rollout to all three test/production hosts of this
campaign — N11-M10, N02-M60, N04-RTX) is completed.

### Addendum: FA=ON now officially published as a CI tag

The `cuda12-maxwell-fa-test` built only locally up to this point had never been committed in the Dockerfile.
Made up for: `GGML_CUDA_FA` is now a build arg (default still OFF,
changes nothing about the behavior of existing multi-GPU pool deployments), CI additionally builds
the tag `cuda12-maxwell-singlegpu-fa-latest` with FA=ON. Future single-GPU rollouts
can pull this tag directly instead of manually distributing local images via `docker save`/`scp`/
`docker load`.

---

## Phase 5 — Retuning batch/ubatch (after the FA rebuild)

**Status: completed — no action needed, side finding documented**

**Prefill benchmark** (640-token prompt, `qwen3.5:4b`, N02-M60/GPU0, FA=ON,
default batch=1024): **251.92 tok/s**. A second run (attempted with a smaller batch)
also ran at batch=1024 for the reason described in the next paragraph:
**251.07 tok/s** — practically identical, no surprise given the finding below.

**Side finding:** `OLLAMA_MAX_BATCH_SIZE` (the cap mechanism injected by
`patch-ollama-batch.py`) showed no effect in any of the builds of this session — the
string `OLLAMA_MAX_BATCH_SIZE` is completely missing from the compiled `ollama` binary
(`strings` check). The patch apparently fails silently against the current Ollama source
version (the target pattern in the Go code has probably changed) — potentially also affects
the productively used `selectGPUPool()` logic on N04-RTX for the
large 12-GPU pool case, where the cap is supposed to be set automatically according to the code comment.
A separate ticket not fixed in this campaign.

**Why there is still no action needed for single-GPU FA deployments:** The original
purpose of the batch cap was to keep the *non-FA* attention compute buffer small
(`batch × ctx × heads × head_dim × 4 bytes`, several GiB at large batch). With FA=ON
this problem disappears structurally — FA does not materialize the large intermediate matrix at all, regardless of the batch size (confirmed: compute buffers stay
at 140–494 MiB, see Phase 2). Ollama's default batch (1024) works for
single-GPU FA deployments without adjustment. For the (still FA=OFF) large
multi-GPU pool case the cap mechanism remains relevant but is currently broken —
that would be the actual next step if someone wants to
pursue the large pool case.

---

## Phase 6 — `--n-cpu-moe` for tight MoE configs

**Status: completed — no clean A/B possible, but an important side finding**

**Setup:** `qwen3.6:35b` (256-expert MoE, ~23 GiB) on an M60 pool artificially reduced to 2 GPUs
(16 GiB combined, deliberately tight) — 2 of the 12
single-instance containers on N02-M60 temporarily stopped to free GPUs,
restored afterwards. FA=OFF (multi-GPU, see Phase 2).

**Baseline (no `--n-cpu-moe`, automatic fitting):** Initially seems to load
successfully (`offloaded 42/42 layers`, but `CPU_Mapped model buffer size = 20293
MiB` — the expert weights already end up mostly on the
CPU via mmap automatically). The server log itself recommends `--load-mode none` instead of mmap for better
performance. **Then crashes during generation:** `llama-server terminated:
signal: aborted (core dumped)`.

**With explicit `--cpu-moe`** (all expert weights fixed on the CPU): **Also crashes**,
this time with `CUDA error: an illegal memory access was encountered`.

**No clean comparison possible** — both configurations are unstable on this tight
2-GPU pool. **Important side finding that goes beyond the original Phase 6 question:**
The multi-GPU MoE instability documented in Phase 2 is **not only
an FA=ON problem** — it also occurs with FA=OFF as soon as the pool is tight enough.
This points to a more fundamental problem with tight multi-GPU MoE configurations
in this fork on Maxwell, independent of Flash Attention or expert placement.
**Not pursued further** in this campaign — a separate, larger
investigation topic (a separate compute-sanitizer run or similar would be the next step,
analogous to Phase 1). Recommendation: avoid tight multi-GPU MoE pools (model only just larger than
the available pool VRAM) for the time being or plan with a more generous VRAM buffer (as
the already productive 4-GPU M10 pool on N11-M10 with `qwen3.6:35b` at 41/42
layers shows — stable there with plenty of buffer).

### Addendum (2026-09-16): Side finding put into perspective — crash not reproducible

The "more fundamental instability of tight multi-GPU MoE pools" documented above was
re-investigated with the same compute-sanitizer methodology as in Phase 1 (`llama-cli` +
GGUF blobs extracted from `ollama-m60-gpu0`, GPUs 0+1 on N02-M60 freed for it).

**Direct `llama-cli` repro attempts** (identical 2-GPU pool, `-sm layer`, `-c 32768`):

| Variant | Result |
|---|---|
| `-ngl 999` (forced) | Clean `cudaMalloc failed: out of memory` — different error picture than the original, since auto-fit is bypassed |
| `--fit on`, without mmproj | **Successful**, correct output, no crash |
| `--fit on`, with mmproj (original conditions reproduced exactly) | **Successful**, correct output, no crash |

**Clean retest via Ollama itself:** Fresh container (`moe-clean-retry`, port
11462) with exactly the same tight 2-GPU configuration as in the original crash
(`OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_SPLIT_MODE=layer`, `cuda12-maxwell-latest`) —
**ran through flawlessly:** HTTP 200 after 95.1s, `load_tensors: offloaded 42/42 layers
to GPU`, `error: None`.

**Layout cache checked:** `/root/.ollama/layout-cache/` in the fresh container was
**completely empty** — the hypothesis of an outdated/wrong cached tensor split as the
cause is thereby refuted. The `dmesg` check for Xid errors was inconclusive (probably no
host dmesg access for this user, not meaningful).

**Corrected assessment:** The original crash was **not reproducible** in three independent,
clean reproduction attempts (3× `llama-cli` directly, 1× fresh Ollama container) —
not even with exactly the same parameters including mmproj. This
argues against it being an independent, deterministically reproducible
architecture bug. More likely: transient/corrupt GPU driver state,
probably as an aftereffect of a previous crash on the same physical GPUs
during the intensive back-to-back test series of this session. The hint formulated above
("avoid tight multi-GPU MoE pools for the time being") is thereby **withdrawn** — there
is no reliable evidence of a fundamental architecture problem anymore. A
residual risk from driver state after crashes remains plausible (not ruled out),
but is a general operations topic (GPU reset after an error), not MoE- or
pool-specific. Not to be confused with the separate, independently confirmed
FA=ON multi-GPU finding from Phase 2 (other tests, other hosts) — that one stays
unchanged.

---

## Phase 7 — N-gram speculative decoding as an MTP alternative

**Status: completed — no benefit, not adopted**

`--spec-type ngram-simple` (`LLAMA_ARG_SPEC_TYPE`) on `qwen3.5:4b`, single-GPU M60,
FA=ON, tested with two prompt types:

| Prompt | Baseline (no spec decode) | ngram-simple | Drafts generated/accepted |
|---|---|---|---|
| Code prompt (original repro) | 19.39 tok/s | 19.25 tok/s | 0 / 0 (in 200 tokens) |
| Deliberately repetitive (numbers 1-100 spelled out) | 19.39 tok/s (reference) | **17.33 tok/s** | 1 / 1 (in 200 tokens, mean acc len=2.0) |

**Result:** No measurable benefit; with the repetitive prompt even **slower** than
the baseline (pure overhead from the n-gram search without compensating gain). Even
the deliberately repetitive prompt produced only a single accepted
draft over 200 tokens. Confirms the expectation formulated in advance in the plan: M10/M60 are
bandwidth-bound, they lack the idle compute budget that speculative decoding would need
to exploit — and the tested prompts hardly hit the required
12-token n-gram window (`size_n=12`). **Not adopted**, the MTP denylist (Phase 0)
stays without replacement — so for the `qwen35` family there is currently no speculative decoding
in productive use, which, given the cuBLAS race condition (Phase 1), is the
right, conservative choice anyway.

---

## Phase 8 — Host-level fine tuning

**Status: completed — 2 of 3 items settled immediately, 1 item not tested empirically**

**NUMA:** `numactl` is not installed on any of the three hosts; checked directly via
`/sys/devices/system/node/`: **exactly 1 NUMA node on N11-M10, N02-M60 and
N04-RTX**. NUMA tuning is thus confirmed irrelevant, as assumed in the plan — no
action needed.

**ECC toggle:** Tesla M10 does not support ECC control via `nvidia-smi` at all
(`ECC Mode: N/A` on all M10 GPUs tested). Tesla M60 has ECC already **disabled at the factory/by
default** (`Disabled`, confirmed on all 12 M60 dies on N02-M60).
There is nothing to toggle in this fleet — the item resolves itself, no
experiment needed, no user decision required.

**`--load-mode mlock`/`none` (formerly `--mlock`/`--no-mmap`):** First A/B attempt with
Olmo-32B (CPU-heavy, ~73 %/27 % CPU/GPU split) on **N11-M10** hung for minutes.
Root cause clarified (see below) — **not an Ollama bug** but real swap thrashing:
N11-M10 has only 15.5 GiB of RAM, the 73 % CPU-offloaded layers of a 19 GiB model
(~14 GiB) pushed into swap. `ps aux` showed the `llama-server` process permanently in
`D` state (interruptible I/O wait), swap usage rose to 10 GiB. A `/api/generate`
took >10 minutes for 150 tokens before Ollama's own server timeout kicked in. **N11-M10
is thus fundamentally unsuitable as a test system for CPU-offload-heavy large models**
(RAM capacity, not a tooling problem) — host discarded for this test class.

**Clean retest on N02-M60** (125 GiB RAM, 120 GiB available) — container `gpu6`
(port 11440, Tesla M60, 8 GiB VRAM) recreated for the test, once with the default load mode,
once with `LLAMA_ARG_LOAD_MODE=mlock`, otherwise identical configuration
(`ollama-legacy:cuda12-maxwell-fa-test`, FA=ON, `LLAMA_ARG_MMPROJ_OFFLOAD=false`).
Model: `gemma4:31b` (dense, ~19 GiB, CPU-heavy split, HF registry bug worked around by
using an already locally tagged model instead of `hf.co/...`). Same prompt
(`temperature=0.2, seed=42, num_predict=150`).

| Variant | Load time | Decode (150 tok.) | tok/s |
|---|---|---|---|
| Default (mmap default) | 50.4 s | 97.99 s | 1.531 |
| `LLAMA_ARG_LOAD_MODE=mlock` | 46.6 s | 101.95 s | 1.471 |

**Result:** No significant difference (difference within the measurement variation
of a single run). Swap usage stayed constant at 5.8 GiB in both runs (no active
thrashing) — as expected, since 120 GiB of free RAM is far more than enough for the ~14 GiB of CPU layers. **Verdict: `mlock` brings nothing on adequately sized hosts (N02-M60,
N04-RTX) and is not adopted.** It does cleanly confirm the hypothesis from
the N11-M10 incident: `mlock` only works under real RAM pressure — and exactly there (N11-M10)
it cannot practically be tested, because even the baseline run becomes
unusably slow through thrashing. Recommendation: fundamentally do not run large CPU-offload-heavy models
on N11-M10 (RAM ceiling ~15.5 GiB), but on N02-M60/N04-RTX.

### Summary of Phase 8
All three original fine-tuning candidates are now completed. NUMA and ECC
were already settled by the hardware/driver circumstances without any change
being needed. `--load-mode` was tested cleanly and empirically (N02-M60) and **not
adopted** — no benefit with sufficient RAM, real benefit only under memory pressure,
which does not exist on the better-sized hosts. Side finding: N11-M10's limited
RAM (15.5 GiB) is an independent operational restriction for large dense models with a
high CPU-offload share, independent of `mlock`.

---

## Phase 9 — Production rollout

**Status: completed (2026-09-16)**

**Finding before the rollout:** The persisted Compose files on all three hosts had drifted
considerably from the actual live state validated in this campaign —
all fixes had so far gone through manual container recreation (`docker run`), never through
the Compose files themselves. Specifically: N11-M10's Compose still described the old
1-container 4-GPU pool model; N02-M60's Compose still referenced stock `ollama/
ollama:0.24.0` with a wrong port mapping (pool+8 singles instead of the
12-single topology long implemented live); only N04-RTX was already close to the live state (only
the thread/MMPROJ env vars were missing).

**Rolled out (env fixes, all three hosts, single-GPU instances only):**
`LLAMA_ARG_THREADS=4`, `LLAMA_ARG_THREADS_BATCH=4`, `LLAMA_ARG_MMPROJ_OFFLOAD=false`,
`OLLAMA_SPLIT_MODE` behavior (default `layer`, no `row` configuration active anymore).

**Rolled out (FA=ON single-GPU image, Phase 2):** All 22 single-GPU instances on all
three hosts previously ran either still on FA=OFF (N11-M10, 4 instances) or on
locally built test tags (N02-M60: 12, N04-RTX: 6) — none of them used the
tag `cuda12-maxwell-singlegpu-fa-latest` built officially in CI and published to GHCR.
All 22 instances were switched to this official image (rolling, per container `docker rm -f` + restart, health check +
smoke test after each step):

| Host | Instances | Before | After |
|---|---|---|---|
| N11-M10 | `ollama-tesla-1..4` (port 11434-11437) | FA=OFF, `cuda12-maxwell-latest` | FA=ON, `cuda12-maxwell-singlegpu-fa-latest` |
| N04-RTX | `ollama-tesla-1..4`, `ollama-m60-1/2` (port 11436-11441) | FA=ON, local test tag | FA=ON, official GHCR tag |
| N02-M60 | `ollama-m60-gpu0..11` (port 11434-11445) | FA=ON, local test tag | FA=ON, official GHCR tag |

**Deliberately unchanged (multi-GPU pools, Phase 2/6 limitation):** `ollama-m60-guard`
(N04-RTX, 2× M60 combined) stays on `cuda12-maxwell-latest`, FA=OFF — multi-GPU FA
is not validated. No multi-GPU pool container was switched in this campaign.

**Smoke test:** `ollama-tesla-4` (N11-M10, GPU3, first instance in the rollout) with
`smollm3:3b`, `temperature=0.2, seed=42, num_predict=60` — HTTP 200, 24.0s, 8.21 tok/s,
coherent output. All 22 instances `healthy` after restart (Docker healthcheck via
`/api/tags`).

**Compose files updated:**
- `compose/docker-compose.maxwell.yml` (this repo, GitHub) — switched from a 1-container pool to
  4 single-GPU services, FA=ON image, complete fix set. GPU UUIDs are
  N11-M10-specific (comment in the file points to this).
- N04-RTX (`llm-studio/worker-rtx/docker-compose.yml`, separate deployment repo) and
  N02-M60 (`llm-studio/worker-m60/docker-compose.yml`, ditto) were synchronized with the actual
  live state (details in the deployment repo commit, not here — different
  repository/remote than this fork).

**Not touched:** Production Compose file on N04-RTX for `ollama`/`ollama-rgtx`
(RTX pool, outside the scope — user instruction of this campaign). `ollama-0240-test`
(N11-M10) and `ollama-0240-m60` (N04-RTX), two unrelated legacy containers, also left
untouched.

### Summary of Phase 9
All three hosts now run consistently on the validated fix set built officially in CI (threads, MMPROJ offload, split mode, FA=ON for
single-GPU). The Compose files were brought up to date so that a future `docker compose up -d` does not
fall back to an older, unvalidated state. This completes the entire
optimization campaign (Phase 0-9). The only open item remains Phase 3's
text-only hybrid confounding test, blocked by the hf.co registry bug — no
blocker for productive operation.

---

## Addendum (2026-09-16): Target topology corrected — N11-M10 back to a 4-GPU pool,
## multi-GPU FA=ON retested and found stable

User requirement after completion of Phase 9: N11-M10 is to run as **one pooled 4-GPU
instance** (not as 4 single-GPU instances, as implemented in Phase 4 of this campaign).
N04-RTX is to use the two M60 GPUs as **one** dual-GPU instance instead of the
previous triple use (`ollama-m60-1` + `ollama-m60-2` individually +
`ollama-m60-guard` combined, all on the same 2 physical GPUs).

**N04-RTX:** `ollama-m60-1` and `ollama-m60-2` removed. `ollama-m60-guard` (already
dual-GPU, FA=OFF) stays as the only M60 instance — now carries both general and
guard classifier traffic. `.env.m60-single` (only relevant for the removed
single instances) removed.

**N11-M10:** 4 single-GPU containers stopped, replaced by one pooled 4-GPU container
(`ollama`, port 11434, all 4 GPUs). The user explicitly decided on a
**retest of FA=ON on this multi-GPU pool** instead of falling back directly to the validated FA=OFF — Phase 2 had classified multi-GPU FA as "not validated/potentially unstable", but Phase 6's later re-investigation had already shown that a
structurally similar multi-GPU instability was not reproducible (probably
transient driver state instead of an architecture bug).

**Test procedure:** `cuda12-maxwell-singlegpu-fa-latest` image (FA=ON) on the new
4-GPU pool, two independent generations with `qwen3.6:35b` (hybrid SSM/MoE, the same
model type that led to crashes in Phase 2/6):

| Run | Prompt/seed | HTTP | Layer offload | done_reason | Error |
|---|---|---|---|---|---|
| 1 | railway summary, seed=42 | 200 | 41/42 | length | none |
| 2 | neural network explanation, seed=123 | 200 | 41/42 | length | none |

Both runs: coherent, correct output (including the `thinking` trace with this
reasoning model), container `healthy` throughout, no `illegal memory access`,
no crash, no Xid errors in the log. Additionally `smollm3:3b` (dense) as a
baseline control — also clean.

**Verdict: multi-GPU FA=ON on N11-M10 works stably — correction of the
original Phase 2 assessment.** This matches the pattern already documented in Phase 6:
the multi-GPU instability observed earlier was probably
never an FA or architecture problem but transient/corrupt GPU driver state
from the intensive back-to-back test phase of this campaign. N11-M10 now runs
productively as a 4-GPU pool with FA=ON (`compose/docker-compose.maxwell.yml` updated
accordingly). **Still to be treated with caution:** only 2 test runs, no
compute-sanitizer run as in Phase 1 — on future anomalies (crash, wrong
output) roll back immediately to `cuda12-maxwell-latest` (FA=OFF) and note it here.

---

## Phase 9 — Final summary (2026-09-16)

**Status: completed.** Target architecture per user requirement live on all three
hosts, all fixes committed and pushed (GitHub + git.4noobs.de). This section
summarizes the entire rollout history, including the subsequent
topology correction and the MTP crash fix found during the rollout.

### Final production topology

| Host | Configuration | Image | Status |
|---|---|---|---|
| N11-M10 | 1× pooled 4-GPU instance (`ollama`, port 11434) | `cuda12-maxwell-singlegpu-fa-latest`, FA=ON | ✅ live, tested 2× stable with `qwen3.6:35b` |
| N04-RTX | 4× single-GPU M10 (`tesla-1..4`, port 11436-11439) | `cuda12-maxwell-singlegpu-fa-latest`, FA=ON | ✅ live |
| N04-RTX | 1× dual-GPU M60 (`ollama-m60-guard`, port 11442) | `cuda12-maxwell-latest`, FA=OFF | ✅ live — consolidated from formerly 3 overlapping M60 uses (`m60-1`+`m60-2`+`guard`) |
| N02-M60 | 12× single-GPU M60 (`gpu0..gpu11`, port 11434-11445) | `cuda12-maxwell-singlegpu-fa-latest`, FA=ON | ✅ live |

Deliberately outside the scope (RTX/GTX GPUs, user instruction): `ollama`/`ollama-rgtx`
on N04-RTX (`ollama-github:latest`, own Dockerfile, no fork image).

### Rollout history (chronological)

1. **First rollout attempt:** All 22 single-GPU instances switched to the
   `cuda12-maxwell-singlegpu-fa-latest` tag built officially in CI (before: FA=OFF on
   N11-M10, local test tags on N02-M60/N04-RTX). N11-M10 initially rolled out as 4 separate
   single-GPU containers (fork repo commit `86d00f9`).
2. **Topology correction (user requirement):** N11-M10 switched back to 1 pooled 4-GPU instance,
   FA=ON deliberately retested (2/2 stable runs with
   `qwen3.6:35b`) instead of falling back directly to FA=OFF — corrects the
   original Phase 2 assessment (`2a1c110`). N04-RTX: `ollama-m60-1`/`-2`
   removed, only `ollama-m60-guard` remains as the only M60 instance (`9077a63`).
3. **qwen3.8:27b test reveals an MTP denylist gap:** New model crashes on load
   on N11-M10 AND on N02-M60 (single GPU, topology-independent) with the exact
   Finding 2 signature. Root cause: Ollama's own Go scheduler forces
   `--spec-type draft-mtp` on first load, independent of the Python denylist in
   `auto-optimize.py` and also independent of `LLAMA_ARG_SPEC_TYPE` env var overrides
   (`c433244`).
4. **Complete root-cause clarification:** The actual controllable lever is
   `draft_num_predict`, not `--spec-type`. `qwen3.8:27b` and the already
   existing `-nospec` variant use the same GGUF blob — the only difference
   is the Modelfile parameter (`draft_num_predict 4` vs. `0`). Directly verified:
   an explicit `draft_num_predict=0` override on the plain variant reliably prevents
   the crash (`2b50f74`).
5. **Proxy fix implemented and rolled out:** `ollama-proxy.py` now enforces
   `draft_num_predict=0` for `qwen3`/`qwen35(moe)` families on **every** request —
   even with a missing cache (first load) and with already existing, stale
   cache entries with a risky value (`51bae38`). Specifically confirmed: N04-RTX's
   auto-optimize cache for `qwen3.6:35b` had carried `draft_num_predict=2` since 2026-06-22 —
   the same crash-capable configuration, unnoticed in productive use, probably only
   by chance (non-deterministic race condition) never crashed.
   Hot deploy to all 18 fork-image containers (N11-M10: 1, N04-RTX: 5, N02-M60: 12)
   before the next CI rebuild so that the fix takes effect immediately.

### Operational disruption during the rollout (self-inflicted, fixed)

The hot-deploy step (`docker cp` of the patched `ollama-proxy.py` into all 18
running containers) lost the file's executable bit — the proxy therefore
did not start on any of the 18 containers, port 11434 was unreachable for ~3-5 minutes on all
affected hosts (`entrypoint` log wrongly showed
"OLLAMA_AUTO_OPTIMIZE=0", the actual cause was the missing `+x` permission).
Detected immediately (healthcheck status "unhealthy"), fixed with `chmod +x` + container restart
on all 18 containers. All `healthy` again, fix verified functional.
**Lesson for future hot deploys:** after `docker cp` always follow up with `chmod +x` on executable
scripts, do not rely on the copy behavior.

### Known open items

- **RTX pool (`ollama`, N04-RTX, port 11434) stays unprotected against the MTP crash risk.**
  This container uses a different image (`ollama-github:latest`, own
  `Dockerfile.github`) **without** the `ollama-proxy.py` infrastructure at all
  (confirmed: no proxy process runs there, just bare `ollama serve`). The
  fix cannot be applied there by hot patch — it would need either its own
  proxy addition for this image or a Modelfile adjustment
  (`draft_num_predict 0`) directly for the qwen35 models referenced there. Deliberately
  not implemented, since outside the scope defined for this campaign
  (RTX/GTX GPUs) — **but the 24.3 tok/s reference number documented there for
  `qwen3.6:35b` (CLAUDE.md) still runs with an unprotected `draft_num_predict=2`,
  i.e. with the same crash risk as `qwen3.8:27b` before today's fix.**
- Phase 3's text-only hybrid confounding test remains blocked by the
  hf.co registry bug (unchanged since Phase 3).
- `qwen3.8:27b` (plain tag) stays technically loaded and usable — now runs safely via
  the proxy fix, but slower than with MTP (no benchmark comparison carried out in this
  campaign).

### Commits of this phase

Fork repo (GitHub, `h3rb3rn/ollama-legacy-gpu`): `86d00f9`, `2a1c110`, `c433244`,
`2b50f74`, `51bae38`.
Deployment repo (git.4noobs.de, `h3rb3rn/ollama`): `82cc26b`, `9077a63`.

---

## Additional investigation (2026-09-16): Fork image tested on RTX/GTX GPUs — NOT ready for use

**Question:** Is it worth switching `ollama`/`ollama-rgtx` (N04-RTX, RTX 2060/3060 +
GTX 1060, currently stock Ollama via `Dockerfile.github`) to the fork image in order to get
the systemic MTP crash protection (proxy) there as well?

**Preliminary analysis (source code, `ggml-cuda/fattn.cu`):** The Flash Attention kernel choice
(`ggml_cuda_get_best_fattn_kernel()`) is hardware-dependent at runtime and independent of the build
— on Turing/Ampere (RTX 2060/3060) the tensor-core path
(`BEST_FATTN_KERNEL_MMA_F16`) is always chosen, the Maxwell TILE kernel is never reached. On Pascal
(GTX 1060, no tensor cores) you end up with the TILE kernel — there the fork
would potentially even be advantageous, analogous to Maxwell. This analysis was correct, but as
the test shows, not the decisive question.

**Isolated test:** Test container with `cuda12-maxwell-singlegpu-fa-latest` on the
same GPUs as `ollama-rgtx` (GPU6 RTX 2060 + GPU11 GTX 1060, idle at test time,
production untouched), separate port. Baseline first measured with the
stock image on the real `ollama-rgtx` production instance (23.5 tok/s,
`hf.co/h3rb3rn/moe-sovereign-planner-9b`, clean).

**Result: the fork image crashes on loading EVERY model tested on an RTX 2060** —
both in the dual-GPU setup (RTX2060+GTX1060) and isolated on a single
RTX 2060, both with a hybrid SSM model and with a simple dense
model (`smollm3-expert-security-3b`). Error: `CUDA error: unspecified launch
failure`, occurs directly after the layer offload, before any actual generation
— 3/3 test runs, 100 % reproduction rate.

**Root cause identified:** The log shows at every start:
```
level=WARN msg="llama-server discovery: could not determine compute capability for
CUDA device — architecture filtering disabled for this device. If inference crashes,
check that the CUDA backend supports this GPU." device="NVIDIA GeForce RTX 2060"
```
Ollama's GPU discovery step cannot determine the compute capability of the RTX 2060 in the fork build
(although a later, different code location in the same log correctly
reports `compute=7.5` — two different detection paths, inconsistent). With
architecture filtering disabled, a wrong/incompatible
CUBIN variant is presumably chosen for the actual kernel launch → generic
launch error. **Confirmed fork-specific:** the same warning appears in no
single log of `ollama-rgtx`, which has been running for 2 days (stock image, identical GPU).

**Verdict: fork image currently NOT ready for use on RTX/GTX GPUs — do not switch
N04-RTX's `ollama`/`ollama-rgtx`.** No connection with the originally
suspected tensor-core question (the preliminary analysis on that was correct but irrelevant,
because the model does not get far enough for Flash Attention kernel selection
to take effect at all). The MTP crash protection on these two containers stays with the
single-model solution already implemented (Modelfile override for `qwen3.8:27b`,
see `BUG-hybrid-arch-degeneration.md`). A real fix would require a
troubleshooting of the GPU discovery code of this fork for non-Maxwell architectures
— not investigated in this campaign, since outside the scope
(RTX/GTX hosts).

*Update (2026-10-05): resolved in the Ollama 0.35.1 fork by `patch-ollama-discovery.py`, which determines the compute
capability by CUDA index and fails closed; the current image runs on RTX 2060/3060. See
[docs/FORK-VS-STOCK.md](docs/FORK-VS-STOCK.md).*
