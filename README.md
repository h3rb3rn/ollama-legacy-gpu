# Ollama legacy GPU fork with Bonsai support

Ollama runtime and Docker builds for NVIDIA legacy and current GPUs, including
native integration of the [Prism Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo)
and [Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

**Status, 2026-10-05:** the build is based on **Ollama v0.35.1** (llama.cpp
b11232) with `BONSAI=OFF` by default. The newest patch keeps the token embedding
table in VRAM (`patch-llama-input-gpu.py`); with `LLAMA_ARG_FIT_TARGET=256`
`qwen3.6:35b` (ctx 262144, KV q4_0, batch 64 or 512) runs **42/42 layers with no weight
buffer on the host** on N11-M10, N04-RTX (:11434) and N02-M60. (Earlier reports
state ctx 131072: that was a bug in the dynamic-pool patch, which halved `-c` whenever
`OLLAMA_MAX_BATCH_SIZE` was set; fixed in `49e786d`.) N04-RTX :11435
runs stock for comparison. Only the `cuda12-maxwell` image was built and run for
this test; `cuda11-legacy` and `cuda13-rtx` carry the patch untested.

- [What the fork does differently from stock](docs/FORK-VS-STOCK.md) – patches,
  scripts, per-image patch matrix and environment variables.
- [Test report 2026-10-04](docs/TEST-REPORT-2026-10-04.md) – results with all
  settings used, and the limitations.
- [CHANGELOG](CHANGELOG.md).

Older results below (Bonsai, 0.35.0 fleet rollout, 0.34.1 RTX qualification) are
kept as history. See [the 0.35.0 fleet report](FLEET-ROLLOUT-0.35.0-2026-10-01.md).

### Fork-specific environment variables

| Variable | Effect |
| --- | --- |
| `OLLAMA_MAX_BATCH_SIZE` | batch size (stock computes its own and ignores it) |
| `OLLAMA_DRAFT_NUM_PREDICT` | MTP draft length, `0` disables (default 0 via gpu-detect; avoids the Maxwell cuBLAS race) |
| `OLLAMA_UNSUPPORTED_GPU` | `mask` (default), `fail`, `ignore` for GPUs not in the image's `CUDA_ARCHS` |
| `OLLAMA_ALLOW_UNKNOWN_CC` | explicitly allow unknown compute capabilities |
| `GGML_CUDA_GRAPHS_LEGACY` | opt-in CUDA graphs on CC < 7.0 |
| `LLAMA_INPUT_LAYER_GPU` | `0` restores upstream CPU placement of the embedding |
| `LLAMA_ARG_FIT_TARGET` | fit reserve per GPU in MiB (upstream variable; `256` needed so no expert tensor (`ffn_down_exps`, ~211 MiB) spills to the host) |

### Batch size: what it changes (measured 2026-10-05, ctx 262144, `ctxfix-20261005`)

`OLLAMA_MAX_BATCH_SIZE` / `LLAMA_ARG_BATCH` / `LLAMA_ARG_UBATCH` only take effect in this fork;
stock Ollama ignores them and picks its own (measured `-b 2048`). The effective value is the one in the
runner log (`llama_context: n_batch`), not the env file. `scripts/bench-throughput.sh` prints it next to
each measurement.

| Host | Batch | Decode tok/s | Prefill tok/s | Host buffer (`CUDA_Host compute`) |
| --- | --- | --- | --- | --- |
| N04-RTX (4× RTX) | 64 / 512 | 40.5 / 40.4 | 624–651 / 1021–1175 | 1029 MiB at 512 |
| N02-M60 (4× M60) | 64 / 512 | 14.2–14.8 / 12.6–13.5 | 87 / 150–155 | 260 MiB at 512 |
| N11-M10 (4× M10) | 64 / 512 | 9.1 / 9.1–9.2 | 24 / 49 | 260 MiB at 512 |

- **Performance:** larger batch speeds up the *prompt* phase (×1.7–2.1); token generation is unchanged.
  It matters for long prompts: of 756 requests seen in N04-RTX logs, 21 % had ≥ 4096 and 7 % ≥ 32768 tokens,
  while 40 % had < 64 tokens (no difference there).
- **Quality:** on N04-RTX, 6/6 greedy answers (incl. a 18k-token needle test) were identical at batch 64 and 512,
  and identical between two runs at 512. Not verified on Maxwell.
- **Chosen per instance** (goal: 256k KV cache at q4_0 fully in VRAM): **512 on all three**. N04-RTX (12 GiB cards): 512 is
  the optimum, 1024 (prefill 887–1009 tok/s) and 2048 (≈550 tok/s, one GPU left empty) are slower. N11-M10 / N02-M60 (8 GiB
  cards): 512 is the ceiling, higher env values are capped to `-b 512`; it fits with 42/42 layers.
  `scripts/sweep-batch.sh` reproduces the ladder (it restarts the host's container).
- **Cost:** batch 512 needs ~0.27 GB pinned host RAM at ctx 262144 (`n_ctx × n_ubatch × 2 B`; ~0.03 GB at 64)
  and 0.1–0.5 GiB more VRAM per GPU; on 8 GiB cards only ~0.3 GiB VRAM stays free.
  Weights themselves are 100 % in VRAM. Details and limits: [TUNING-2026-10-04.md](TUNING-2026-10-04.md).

## GPU compatibility

This table describes **this repository's compiled targets**, not every GPU
supported by a CUDA Toolkit. A matching compute capability is build coverage;
only the listed hardware tests establish runtime validation.
The authoritative matrix is [presets/gpu-targets.json](presets/gpu-targets.json).

| Variant | Compiled CC | Tesla / data-center examples | GeForce / gamer examples | Runtime status |
| --- | --- | --- | --- | --- |
| CUDA11.8 `cuda11-legacy` | 3.7 | Tesla K80 | No GeForce model claimed for CC3.7 | **Untested**; compilation passed, F16 KV /FAOFF |
| CUDA11.8 `cuda11-legacy` | 5.0,5.2 | Tesla M10, M60, M40 | GTX750/750Ti; GTX950/960/970/980/980Ti; Maxwell TITAN X | **Untested with CUDA11**; compiled targets only |
| CUDA12.0 `cuda12-maxwell` | 5.0 | Tesla M10 | GTX750/750Ti | **M10 tested:** 0.35 Q8 stability / Q4 rollout, four GPUs,190k; gamer cards untested |
| CUDA12.0 `cuda12-maxwell` | 5.2 | Tesla M60, M40 | GTX950/960/970/980/980Ti; Maxwell TITAN X | **M60 tested:** 0.35 Q4/190k and Q8/131k, two GPUs; gamer cards/M40 untested |
| CUDA12.0 `cuda12-maxwell` | 6.0,6.1 | Tesla P100, P4, P40 | GTX1050/1050Ti/1060/1070/1070Ti/1080/1080Ti; Pascal TITAN X/Xp | Compiled targets, **fork inference untested** |
| CUDA12.0 `cuda12-maxwell` | 7.0 | Tesla V100 | TITAN V | Compiled target, **untested** |
| CUDA13 `cuda13-rtx` | 7.5 | T4 | GTX16 series; RTX20 series; TITAN RTX | **RTX2060 tested** with Bonsai Q4 at256k; other cards untested |
| CUDA13 `cuda13-rtx` | 8.0,8.6 | A100, A10 | RTX30 series | **RTX3060 tested** with Bonsai Q4 at256k; other cards untested |
| CUDA13 `cuda13-rtx` | 8.9,9.0,10.0,12.0 | L4/L40, H100, B200 | RTX40/RTX50 series | Build targets; **untested** |

CUDA13 cannot compile Maxwell, Pascal or Volta. GTX10 cards therefore route
to CUDA12; GTX16 cards are Turing and route to CUDA13. CUDA11's current preset
does **not** contain KeplerCC3.0/3.5: GTX650/660/670/680/760/770/780/780Ti and
Kepler TITAN are not covered. Jetson/mobile variants and other unlisted compute
capabilities are not automatically supported. Driver compatibility and sufficient
VRAM are required independently of the target architecture.

Sources: [NVIDIA legacy compute capabilities](https://developer.nvidia.com/cuda/gpus/legacy),
[current compute capabilities](https://developer.nvidia.com/cuda/gpus),
[CUDA13 architecture removals](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

## Bonsai demo integration and the Q2 model

The demo's NVIDIA path uses Prism llama.cpp; its separate MLX path targets
Apple Silicon. This fork integrates the NVIDIA backend into ordinary Ollama
APIs, without launching the demo's standalone server.

- `BONSAI=ON` replaces the complete native backend with checksum-verified Prism
  commit `adfffbe41b2cabcd51fff326ab045662265062bb` from
  [source.json](patches/bonsai/source.json). Native libraries and private GGML
  enums stay together; they must not be mixed with upstream libraries.
- [prepare-bonsai.py](scripts/prepare-bonsai.py) applies the Ollama GGUF parser
  extensions and native compatibility hooks, including Prism rotation metadata.
- [import-bonsai.py](scripts/import-bonsai.py) imports the original published
  GGUF unchanged, retaining its embedded template and model parameters.
- Published two-bit **PQ2_0** weights are tested. **PTQ1_0** parser/import support
  is present but not hardware-qualified. PQ2_0 is not ordinary `Q2_K` or the
  different `Q2_0` development format.
- **Weight quantization and KV quantization are independent:** all Bonsai tests
  below use PQ2_0 weights; Q8_0/Q4_0 describes the attention cache.
- CUDA12/13 release builds enable Flash Attention. Quantized KV requires both
  compiled kernels and runtime FA; setting an environment variable cannot add
  kernels omitted from an older image. CUDA11 keeps FAOFF and F16 KV.
- Initial integration covers text inference. It does not import a separate
  vision projector automatically.

See [BONSAI.md](BONSAI.md) for build/import instructions and the
[demo runtime/code audit](BONSAI-DEMO-RUNTIME-AUDIT-2026-09-29.md) for the verified
MLX/Prism paths and Flash Attention differences.

## KV cache configuration

Configuration is the same as Stock Ollama and independent of model weight
quantization. The server-wide default is F16; installing the fork does not
automatically enable quantization. The deployed Tesla fleet defaults to Q4; set the container environment:

```yaml
environment:
  OLLAMA_FLASH_ATTENTION: "1"
  OLLAMA_KV_CACHE_TYPE: q4_0
```

Use `q8_0` for Q8 or `f16` for the upstream default. Recreate the container after
changing its Compose environment. The setting applies to all models served by
that instance, not a per-model Modelfile parameter. Context size remains a
separate setting (`OLLAMA_CONTEXT_LENGTH` / request `num_ctx`).
All twelve Tesla production services now use Q4 with FA enabled, including
N04:11436 at190k. Q4 is configured in deployment profiles; upstream's unset
environment default remains F16.
N04 Stock ports11434/11435 use Q4.
See [Ollama's FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).

## Actual Bonsai PQ2_0 hardware results

Model: `bonsai2:27b-pq2_0`; one request at a time; batch128, seed42,
temperature0. Maxwell requests190000 context, allocated as190208.
The cache capacity was allocated; tests did **not** fill all190k/256k slots.

**Ollama 0.35.0, CUDA12, native hardware qualification:**

| GPUs on N04 | KV | Allocated context | KV total | Repeated 128-token decode | 2048-token stability |
| --- | --- | --- | --- | --- | --- |
| Four Tesla M10 | Q8_0 | 190208 | 6.17 GiB | 1.213 / 1.806 tok/s | 1.648 tok/s |
| Two Tesla M60 | Q4_0 | 190208 | 3.27 GiB | 7.319 / 7.357 tok/s | 7.079 tok/s |
| Two Tesla M60 | Q8_0 | 131072 | 4.25 GiB | 7.333 / 7.313 tok/s | 7.102 tok/s |

All three passed arithmetic437/763, actual CUDA cache capability checks and
65/65 model-layer offload. The final serialized M10 production replacement
also passed a fresh 256-token test at1.793 tok/s. Qualification runs partly
overlapped; these measurements do not isolate hardware bottlenecks.
M60 Q8 at190k did not fit fully on these two GPUs (53/65 layers), so it was
stopped and is not qualified. M60 ordinary Qwen regression passed Q4 and Q8
with2048-token stability at18.191/18.375 tok/s.
M10 Q4 on0.35 passed the new256-token rollout check at1.798tok/s, with
65/65 GPU layers and3.27GiB KV. Its long2048-token Q4 result below remains
the historical0.34.1 qualification.

**Historical Ollama 0.34.1 results:**

| GPUs on N04 | KV type | Context | KV total | Repeated128-token decode | 2048-token stability |
| --- | --- | --- | --- | --- | --- |
| Four Tesla M10, Q8 production | Q8_0 | 190k | 6.17GiB | 1.837 /1.838tok/s | 1.802tok/s |
| Four Tesla M10, Q4 candidate | Q4_0 | 190k | 3.27GiB | 1.831 /1.833tok/s | 1.790tok/s |
| Two Tesla M60, temporary production test | Q4_0 | 190k | 3.27GiB | 7.517 /7.508tok/s | 7.236tok/s |
| Two RTX2060 +two RTX3060, fork qualification | Q4_0 | 256k | 4.50GiB | 21.557 /21.491tok/s | 20.802tok/s |

Two arithmetic answers do not constitute a comprehensive quantization quality
evaluation; layer offload alone does not prove every graph operation runs on GPU.

M10 F16 KV previously used11.61GiB: Q8 saves46.875%, Q4 saves71.875% of **KV
memory**, not of total model/runtime memory. No M60 F16 speed baseline was
measured. The M60/M10 speed difference does not isolate a PCIe, bandwidth or
kernel bottleneck.

Reports and reproducible evidence:

- [M10 Q8 production rollout](BONSAI-M10-PRODUCTION-Q8-2026-10-01.md).
- [M10 Q8/Q4 candidate validation](BONSAI-M10-190K-2026-09-30.md).
- [M60 Q4 production test and restoration](BONSAI-M60-PRODUCTION-Q4-2026-10-01.md).
- [RTX Flash Attention validation](BONSAI-RTX-FA-VALIDATION-2026-09-30.md).
- [PQ2 performance analysis](BONSAI-PQ2-PERFORMANCE-2026-09-30.md).
- [Historical machine-readable test evidence](tests/evidence/README.md).
- [0.35.0 fleet and native evidence](docs/evidence/fleet-v035-20261001/README.md).
- [Isolated Spark2.5 compatibility backend](compat/README.md).

Current deployment differs from test topology: N04 ports11434/11435 now run
**Stock Ollama0.35.0** at the user's request. Port11436 remains the qualified
Bonsai fork, now with Q4. M60 port11442 also runs the0.35 fork with Q4.
All share `/opt/ollama/models:/root/.ollama`; no duplicate model pool is needed.
See [Stock deployment verification](STOCK-N04-DEPLOYMENT-2026-10-01.md).

## Build and import

Build the hardware-qualified Maxwell version:

```bash
docker build -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.35.0 --build-arg BONSAI=ON \
  --build-arg 'CUDA_ARCHITECTURES=50-real;52-real' \
  --build-arg GGML_CUDA_FA=ON --build-arg JOBS=4 \
  -t ollama-bonsai:local-maxwell .
```

For a tested N04 runtime configuration see
[the M10 Compose manifest](compose/docker-compose.n04-bonsai-m10.yml).
Its pinned local image ID exists on N04; build/import a verified image first
before using that manifest elsewhere. It does not pull a published registry tag.

Import the GGUF already downloaded by Bonsai-demo:

```bash
export OLLAMA_HOST=http://127.0.0.1:11436
python3 scripts/import-bonsai.py \
  /path/to/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --context 190000 --batch 64
ollama run bonsai2:27b-pq2_0
```

Keep the existing shared pool when updating an image; never download another
copy merely to rebuild or deploy the runtime.

## GitHub Actions

The upstream check detects published Ollama releases; the build workflow applies
Bonsai patches and compiles CUDA11/12/13 candidates. This is a versioned source
build, not a merge of Ollama's entire repository into this packaging repository.
Patch-anchor failures, parser checks and regression failures stop the build.

Without `OLLAMA_GPU_TEST_TARGETS`, Actions builds **candidate images only**.
It does not promote `latest`, deploy production or record the version as fully
validated. With a provisioned inventory, immutable candidates must pass real
K80/M10/M60/RTX hardware tests before configured rollout, promotion and version
recording. Stock11434/11435 are excluded from automatic fork deployment.
CUDA11 remains hardware-untested until a real K80 gate passes.
See [GPU-RELEASE-AUTOMATION.md](GPU-RELEASE-AUTOMATION.md) for runner/config setup.

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s compat -p 'test_*.py' -v
bash -n compose/update.sh
git diff --check
```

The build additionally runs Go GGUF parser and legacy batch-cap tests against
the selected Ollama release. Recorded native builds: CUDA11/Ollama0.34.1,
CUDA12/Ollama0.34.1, CUDA13/Ollama0.34.1 and CUDA12/Ollama0.35.0.
The CUDA12 Actions job adds the isolated Maxwell Spark compatibility overlay
from its exact native candidate digest; per-process library checks reject mixing.
GPU evidence is historical and image-specific; it does not qualify every future
release or every gamer card in the compatibility table.

## Related projects and license

[Ollama](https://github.com/ollama/ollama),
[Prism llama.cpp](https://github.com/PrismML-Eng/llama.cpp),
[legacy llama.cpp fork](https://github.com/h3rb3rn/llama.cpp-legacy-gpu).
MIT license; see [LICENSE](LICENSE). Model weights retain their own license and
are not distributed in this repository.
