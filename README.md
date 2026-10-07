# Ollama legacy GPU fork

Ollama runtime and Docker builds for NVIDIA legacy and current GPUs (Maxwell to Blackwell),
with optional native integration of the [Prism Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo)
and [Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

## Status (2026-10-05)

The build is based on **Ollama v0.35.1** (llama.cpp b11232) with `BONSAI=OFF` by default. The image
`ollama-gaps:pipefix-20261005` (N02-M60: `ollama-gaps:kolibri-20261006`) serves `qwen3.6:35b` (35.5B, Q4_K_M) on the `:11434` instances of three hosts with a
**262144-token context, q4_0 KV cache and all 42 layers on the GPUs**. Weights, KV cache and compute buffers live in VRAM; only
small pinned staging buffers stay in host memory.

| Host | GPUs in the pool | Batch | Pipeline parallelism | Decode | Prefill | Host buffer |
| --- | --- | --- | --- | --- | --- | --- |
| N04-RTX | 2× RTX 2060 + 1× RTX 3060 (12 GiB) | 512 | on | 42.3 tok/s | ~1150 tok/s | 1029 MiB |
| N02-M60 | 12× Tesla M60 (8 GiB), one instance | 64 | off | 10.4 tok/s (`qwen3.6:35b`, 12 GPUs); Kolibri-1 11.1 tok/s (8 GPUs) | 65–69 tok/s | – |
| N11-M10 | 4× Tesla M10 (8 GiB) | 64 | off | 9.3 tok/s | ~24 tok/s | 32.8 MiB |

- [What the fork does differently from stock](docs/FORK-VS-STOCK.md) – patches, scripts, per-image patch matrix, environment variables.
- [Tuning and measurements](docs/TUNING.md) – batch size, pipeline parallelism, host buffers, quality check, prompt lengths.
- [Fleet state](docs/FLEET-STATE.md) – live configuration per host, model versions, state of the deployment clones.
- [CHANGELOG](CHANGELOG.md) and [release automation](GPU-RELEASE-AUTOMATION.md).

### Fork-specific environment variables

| Variable | Effect |
| --- | --- |
| `OLLAMA_MAX_BATCH_SIZE` | batch size (stock computes its own and ignores it; measured `-b 2048`) |
| `OLLAMA_DRAFT_NUM_PREDICT` | server-wide default **and upper bound** for `draft_num_predict`, also for the model manifest (the registry's `draft_num_predict 2`); a request value wins; `0` disables MTP (default 0 via gpu-detect; avoids the Maxwell cuBLAS race) |
| `OLLAMA_UNSUPPORTED_GPU` | `mask` (default), `fail`, `ignore` for GPUs not in the image's `CUDA_ARCHS` |
| `OLLAMA_ALLOW_UNKNOWN_CC` | explicitly allow unknown compute capabilities |
| `GGML_CUDA_GRAPHS_LEGACY` | opt-in CUDA graphs on CC < 7.0 |
| `LLAMA_PIPELINE_PARALLEL` | `0` disables llama.cpp pipeline parallelism (multi-GPU layer split): saves pinned host RAM (4× input buffers) and ~100 MiB VRAM per GPU; no gain on Maxwell, +7 % decode / ~+20 % prefill on the RTX pool |
| `LLAMA_INPUT_LAYER_GPU` | `0` restores upstream CPU placement of the embedding |
| `OLLAMA_FORCE_GPU_LAYERS` | `1` replaces the fit by a greedy fill (highest CUDA index first, as few GPUs as needed); Kolibri-1 Q4_K_M uses 8 of 12 Tesla M60 |
| `OLLAMA_LAYER_OVERHEAD_SCALE` | factor on the per-layer weights in the greedy fill (reserve for compute buffers and KV); `1.10` works on 8 GiB M60, `1.03` runs out of memory |
| `LLAMA_ARG_FIT_TARGET` | fit reserve per GPU in MiB (upstream variable; `256` keeps expert tensors such as `ffn_down_exps`, ~211 MiB, out of host memory) |

### Tuning summary

`OLLAMA_MAX_BATCH_SIZE` / `LLAMA_ARG_BATCH` / `LLAMA_ARG_UBATCH` only take effect in this fork. The effective value is the one in the
runner log (`llama_context: n_batch`), not the env file; `scripts/bench-throughput.sh` prints it next to every measurement.

- **Batch size** speeds up the prompt phase only (×1.7–2.1 for 512 over 64); decode is unaffected. On the 8 GiB hosts 512 is the
  ceiling (higher values are capped to `-b 512`) and leaves a 260 MiB pinned host buffer, so they run with 64; on N04-RTX 512 is the optimum.
- **Quality:** on N04-RTX, 6/6 greedy answers (including an 18k-token needle test) are identical at batch 64 and 512 (not verified on Maxwell).
- **Host buffer** ≈ `n_ctx × n_ubatch × 2 B`, ×4 with pipeline parallelism.
- **Prompt lengths** seen in N04-RTX logs: median 210 tokens, 21 % ≥ 4096, 7 % ≥ 32768; batch 512 matters for the long ones.
- **MTP** is off (`OLLAMA_DRAFT_NUM_PREDICT=0`): it is slower on the RTX pool and crashes on Maxwell.

Details, tables and limits: [docs/TUNING.md](docs/TUNING.md).

### More than nine GPUs and Kolibri-1

- **Twelve GPUs in one instance:** NCCL builds forced peer access on every VMM pool allocation; CUDA allows one device plus 8 peers per mapping,
  so 12 visible GPUs aborted with `peer mapping resources exhausted` (8 and 9 GPUs run without the patch). `patch-llama-vmm-peer-access.py`
  limits the forcing to eight devices; 12× Tesla M60 now load `qwen3.6:35b` in one instance (10.4 tok/s, slower than the 4-GPU pool).
- **Kolibri-1** (`kolibri1`, 78.1B MoE, native context 262144): `patch-llama-kolibri1.py` adds the architecture. `Kolibri-1-Q4_K_M` runs with
  `-c 262144`, all 51 layers in VRAM on 8 to 12 GPUs (10.4–11.1 tok/s decode, 65–69 tok/s prefill); with `OLLAMA_FORCE_GPU_LAYERS=1` and scale
  `1.10` it uses 8 of 12 GPUs (7 run out of memory). Needle tests pass up to 221k prompt tokens (decode falls to 3 tok/s there); a 60-minute
  soak (220 requests) ran without errors.

Tables, context ladder and limits: [docs/TUNING.md](docs/TUNING.md).

## GPU compatibility

This table describes **this repository's compiled targets**, not every GPU
supported by a CUDA Toolkit. A matching compute capability is build coverage;
only the listed hardware tests establish runtime validation.
The authoritative matrix is [presets/gpu-targets.json](presets/gpu-targets.json).

| Variant | Compiled CC | Tesla / data-center examples | GeForce / gamer examples | Runtime status |
| --- | --- | --- | --- | --- |
| CUDA11.8 `cuda11-legacy` | 3.7 | Tesla K80 | No GeForce model claimed for CC3.7 | **Untested**; compilation passed, F16 KV /FAOFF |
| CUDA11.8 `cuda11-legacy` | 5.0,5.2 | Tesla M10, M60, M40 | GTX750/750Ti; GTX950/960/970/980/980Ti; Maxwell TITAN X | **Untested with CUDA11**; compiled targets only |
| CUDA12.0 `cuda12-maxwell` | 5.0 | Tesla M10 | GTX750/750Ti | **M10 tested** (four GPUs, `qwen3.6:35b` at 262144 and Bonsai at 190k); gamer cards untested |
| CUDA12.0 `cuda12-maxwell` | 5.2 | Tesla M60, M40 | GTX950/960/970/980/980Ti; Maxwell TITAN X | **M60 tested** (four GPUs, `qwen3.6:35b` at 262144; two GPUs, Bonsai); gamer cards/M40 untested |
| CUDA12.0 `cuda12-maxwell` | 6.0,6.1 | Tesla P100, P4, P40 | GTX1050/1050Ti/1060/1070/1070Ti/1080/1080Ti; Pascal TITAN X/Xp | Compiled targets, **fork inference untested** |
| CUDA12.0 `cuda12-maxwell` | 7.0 | Tesla V100 | TITAN V | Compiled target, **untested** |
| CUDA12.0 `cuda12-maxwell` | 7.5,8.0,8.6,8.9,9.0 | T4, A100, A10, L4/L40, H100 | RTX20/30/40 series | **RTX2060/RTX3060 tested** (`qwen3.6:35b` at 262144); other cards untested |
| CUDA13 `cuda13-rtx` | 7.5 | T4 | GTX16 series; RTX20 series; TITAN RTX | **RTX2060 tested** with Bonsai Q4 at256k; other cards untested |
| CUDA13 `cuda13-rtx` | 8.0,8.6 | A100, A10 | RTX30 series | **RTX3060 tested** with Bonsai Q4 at256k; other cards untested |
| CUDA13 `cuda13-rtx` | 8.9,9.0,10.0,12.0 | L4/L40, H100, B200 | RTX40/RTX50 series | Build targets; **untested** |

CUDA13 cannot compile Maxwell, Pascal or Volta. The `cuda12-maxwell` image therefore also contains the Turing to Hopper targets,
so hosts that mix Tesla and RTX cards run one image; GTX16 and newer cards can alternatively use CUDA13. CUDA11's current
preset does **not** contain KeplerCC3.0/3.5: GTX650/660/670/680/760/770/780/780Ti and
Kepler TITAN are not covered. Jetson/mobile variants and other unlisted compute
capabilities are not automatically supported. Driver compatibility and sufficient
VRAM are required independently of the target architecture. A GPU that none of the image's `CUDA_ARCHS` covers is masked at
start (`OLLAMA_UNSUPPORTED_GPU`).

Sources: [NVIDIA legacy compute capabilities](https://developer.nvidia.com/cuda/gpus/legacy),
[current compute capabilities](https://developer.nvidia.com/cuda/gpus),
[CUDA13 architecture removals](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

## Bonsai demo integration and the Q2 model

The demo's NVIDIA path uses Prism llama.cpp; its separate MLX path targets
Apple Silicon. This fork integrates the NVIDIA backend into ordinary Ollama
APIs, without launching the demo's standalone server. `BONSAI=ON` builds it; the default is `BONSAI=OFF`. The pinned Prism tree
does not accept the compat patch `002-clef.patch` of Ollama 0.35.1, so `BONSAI=ON` currently applies to the pinned release only.

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
- **Weight quantization and KV quantization are independent:** Q8_0/Q4_0 describes the attention cache.
- CUDA12/13 release builds enable Flash Attention. Quantized KV requires both
  compiled kernels and runtime FA; setting an environment variable cannot add
  kernels omitted from an image. CUDA11 keeps FAOFF and F16 KV.
- Initial integration covers text inference. It does not import a separate
  vision projector automatically.

See [BONSAI.md](BONSAI.md) for build/import instructions and the
[demo runtime/code audit](BONSAI-DEMO-RUNTIME-AUDIT-2026-09-29.md) for the verified
MLX/Prism paths and Flash Attention differences. Measurement reports:

- [M10 Q8 production rollout](BONSAI-M10-PRODUCTION-Q8-2026-10-01.md).
- [M10 Q8/Q4 candidate validation](BONSAI-M10-190K-2026-09-30.md).
- [M60 Q4 production test](BONSAI-M60-PRODUCTION-Q4-2026-10-01.md).
- [RTX Flash Attention validation](BONSAI-RTX-FA-VALIDATION-2026-09-30.md).
- [PQ2 performance analysis](BONSAI-PQ2-PERFORMANCE-2026-09-30.md).
- [Machine-readable test evidence](tests/evidence/README.md).
- [Isolated Spark2.5 compatibility backend](compat/README.md).

## KV cache configuration

Configuration is the same as Stock Ollama and independent of model weight
quantization. The server-wide default is F16; installing the fork does not
automatically enable quantization. The deployed instances use Q4; set the container environment:

```yaml
environment:
  OLLAMA_FLASH_ATTENTION: "1"
  OLLAMA_KV_CACHE_TYPE: q4_0
```

Use `q8_0` for Q8 or `f16` for the upstream default. Recreate the container after
changing its Compose environment. The setting applies to all models served by
that instance, not a per-model Modelfile parameter. Context size remains a
separate setting (`OLLAMA_CONTEXT_LENGTH` / request `num_ctx`). The flash-attention kernels
compiled into the images support the pairs `q4_0-q4_0`, `q8_0-q8_0`, `f16-f16` and `bf16-bf16`; K-quants
such as Q4_K_M are weight formats and not available for the KV cache.
See [Ollama's FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).

## Build and import

Build the image for mixed Maxwell and RTX hosts (as deployed):

```bash
docker build -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.35.1 --build-arg BONSAI=OFF \
  --build-arg 'CUDA_ARCHITECTURES=50-real;52-real;61-real;75-real;86-real' \
  --build-arg GGML_CUDA_FA=ON --build-arg JOBS=5 \
  -t ollama-gaps:local .
```

Bonsai variant (pinned release only):

```bash
docker build -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.35.0 --build-arg BONSAI=ON \
  --build-arg 'CUDA_ARCHITECTURES=50-real;52-real' \
  --build-arg GGML_CUDA_FA=ON --build-arg JOBS=4 \
  -t ollama-bonsai:local-maxwell .
```

For a tested N04 Bonsai runtime configuration see
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
the patches and compiles CUDA11/12/13 candidates. `BONSAI` defaults to `OFF`; the repository variable
`OLLAMA_BONSAI` or the dispatch input `bonsai` switches it on. This is a versioned source
build, not a merge of Ollama's entire repository into this packaging repository.
Patch-anchor failures, parser checks and regression failures stop the build.

Without `OLLAMA_GPU_TEST_TARGETS`, Actions builds **candidate images only**.
It does not promote `latest`, deploy production or record the version as fully
validated. With a provisioned inventory, immutable candidates must pass real
K80/M10/M60/RTX hardware tests before configured rollout, promotion and version
recording. CUDA11 remains hardware-untested until a real K80 gate passes.
See [GPU-RELEASE-AUTOMATION.md](GPU-RELEASE-AUTOMATION.md) for runner/config setup.

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s compat -p 'test_*.py' -v
bash -n compose/update.sh
git diff --check
```

The build additionally runs the Go GGUF parser and legacy batch-cap tests against the selected Ollama release when `BONSAI=ON`.
The Go patches are compile- and test-checked against the Ollama release tree (`go build ./discover ./server`,
`go test ./discover ./server`). The CUDA12 Actions job adds the isolated Maxwell Spark compatibility overlay
only for Bonsai builds. GPU evidence is image-specific; it does not qualify every future
release or every gamer card in the compatibility table.

## Related projects and license

[Ollama](https://github.com/ollama/ollama),
[Prism llama.cpp](https://github.com/PrismML-Eng/llama.cpp),
[legacy llama.cpp fork](https://github.com/h3rb3rn/llama.cpp-legacy-gpu).
MIT license; see [LICENSE](LICENSE). Model weights retain their own license and
are not distributed in this repository.
