# Bonsai 2 through the Ollama Maxwell fork

`BONSAI=ON` builds Ollama with the Prism runtime needed by the Bonsai demo's
`Ternary-Bonsai-2-27B-PQ2_0.gguf`. It also imports `PTQ1_0`. This is native
Ollama integration: the regular `/api/create`, `/api/chat`, `/api/generate` and
OpenAI-compatible API use the patched runner. No separate inference proxy is
required for the Prism transforms.

The published PQ2_0 model has passed Q8/Q4 tests on four Tesla M10 GPUs and
Q4 tests on two Tesla M60 GPUs. M10 production now uses the qualified Q8
configuration. See the production reports below for measured limits.

## Build

From this repository:

```sh
docker build -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.34.1 \
  --build-arg BONSAI=ON \
  --build-arg GGML_CUDA_FA=ON \
  --build-arg 'CUDA_ARCHITECTURES=50-real;52-real' \
  --build-arg JOBS=2 \
  -t ollama-legacy:bonsai-maxwell .
```

The build uses CUDA 12.0.1 and a checksum-verified Prism source archive pinned
in [source.json](patches/bonsai/source.json). All native libraries come from
that source. The build retains Ollama's compatibility layer and the fork's
GPU placement patches. Updating Ollama or Prism requires revalidating this
pair. New Ollama releases must pass patch application, compilation, parser
tests and hardware gates before promotion; they are not rejected solely
because their version differs from the original v0.34.1 baseline.

`BONSAI=OFF` retains the existing upstream-backed build.

## Start a separate instance

Choose free GPU indices appropriate to the host. The tested N04 configuration
uses four Tesla M10 dies, each with 8 GiB of independent VRAM. `OLLAMA_SCHED_SPREAD`
is enabled so the model is distributed across all four cards. Bind the same
`/opt/ollama/models` pool used by the other N04 Ollama containers when shared
model storage is desired.

```sh
docker run -d --name ollama-tesla-bonsai \
  --gpus '"device=GPU-UUID-0,GPU-UUID-1,GPU-UUID-2,GPU-UUID-3"' \
  -p 11436:11434 \
  -e OLLAMA_NUM_PARALLEL=1 \
  -e OLLAMA_SCHED_SPREAD=true \
  -e OLLAMA_FLASH_ATTENTION=true \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_INTERNAL_PORT=11434 \
  -e OLLAMA_GPU_AUTODETECT=0 \
  -e OLLAMA_AUTO_OPTIMIZE=0 \
  -e CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1,GPU-UUID-2,GPU-UUID-3 \
  -v /opt/ollama/models:/root/.ollama \
  ollama-legacy:bonsai-maxwell
```

PQ2_0 weights occupy about 6.71 GiB before KV cache and working buffers. The
tested N04 model profile defaults to 190000 context (rounded by llama.cpp to
190208) and one request at a time. Lower `--context` for faster prompt
processing or smaller cards. PTQ1_0 weights occupy about
5.54 GiB. Release CUDA12/13 builds enable Flash Attention; CUDA11 remains
FAOFF/F16. The direct Docker build above explicitly enables the kernels.

## Import the file from Bonsai-demo

After the demo downloads its GGUF, pass that exact file to the fork's importer:

```sh
export OLLAMA_HOST=http://127.0.0.1:11436
python3 scripts/import-bonsai.py \
  /path/to/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
ollama run bonsai2:27b-pq2_0
```

The helper uses the installed Ollama CLI to upload the original GGUF unchanged,
sets conservative context/batch defaults and the model's sampling parameters,
and lets Ollama use the embedded chat template. `--ollama` can select the fork's
CLI executable; `--model`, `--context`, and `--batch` override the defaults.
`--dry-run` prints the generated Modelfile. A normal Modelfile with `FROM` pointing
to the same GGUF is supported as well.

An existing Open WebUI can connect to the new Ollama endpoint and select
`bonsai2:27b-pq2_0`. The demo's `start_llama_server.sh` still starts its standalone
Prism server; it is not the launcher for this Ollama instance.

```sh
curl http://127.0.0.1:11436/api/chat -d '{
  "model":"bonsai2:27b-pq2_0",
  "messages":[{"role":"user","content":"What is the capital of France?"}],
  "stream":false
}'
```

This initial path is text inference. It does not automatically import the
separate vision projector. Ordinary Q2_K and the Q2_0 development checkpoint
are different formats; relabeling either as PQ2_0 is invalid.

## Validation

- Pinned source archive and idempotent preparation: checked locally.
- Ollama GGUF parser suite, including on-disk PQ2_0/PTQ1_0 IDs, payload lengths,
  rotation metadata and malformed quantization rows: passed locally.
- Full CUDA 12.0.1 image build: passed on N02-M60, CUDA targets `sm_50` and
  `sm_52`, Ollama `v0.34.1`, Prism `adfffbe41b2cabcd51fff326ab045662265062bb`.
- Model SHA-256 verified as
  `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`;
  imported as `bonsai2:27b-pq2_0` into the shared N04 Ollama pool.
- Four-GPU test on N04 Tesla M10: all 65/65 model layers were offloaded at
  4096 context. `OLLAMA_SCHED_SPREAD=true` split the weights across CUDA0–CUDA3.
  A short arithmetic chat returned the correct answer `437`.
- At 262144 context the request completed, but the memory planner placed 12
  layers on CPU. The user then requested a lower 190k cache.
- At `num_ctx=190000`, llama.cpp rounded to 190208 tokens and reserved 2972 MiB
  KV on each M10. All 65/65 layers remained on GPUs; a short arithmetic chat
  returned `437`. A warm-run 64-token fixed-output sample measured 0.20 prompt
  tokens/s (42 prompt tokens) and 1.84 generated tokens/s. A separate four-token
  chat sample measured 1.03 prompt tokens/s and 0.45 generated tokens/s; short
  samples vary considerably. The first cold model start took about 41–57
  seconds in these tests.

The original Maxwell container serves port `11436` and uses the shared
`/opt/ollama/models` bind mount, with the original GGUF retained in the shared
Ollama content-addressed blob store. Contexts above 4096 are significantly
slower on Maxwell, even when the full model fits on the four GPUs.

## RTX CUDA 13 and Flash Attention

The RTX image now compiles `GGML_CUDA_FA=ON`. Runtime flags alone cannot
enable omitted CUDA kernels. A scheduler trace of the old image showed
`FLASH_ATTN` on CPU despite `65/65` GPU layers; the new image assigns these
operations to CUDA. The backend capability probe passes for F16 and Q4_0 on
both RTX 2060 and RTX 3060:

```sh
docker exec ollama python3 /usr/local/bin/check-cuda-flash-attn.py
```

Build the validated combination with:

```sh
docker build -f dockerfiles/Dockerfile.cuda13-rtx \
  --build-arg OLLAMA_VERSION=v0.34.1 --build-arg BONSAI=ON \
  --build-arg 'CUDA_ARCHITECTURES=75-real;86-real' \
  --build-arg GGML_CUDA_FA=ON --build-arg JOBS=4 \
  -t ollama-bonsai:rtx-fa-cuda13-20260929 .
```

N04 port `11434` uses the original four RTX GPUs and the existing shared
model pool. Its configuration is captured in
[docker-compose.n04-bonsai-rtx.yml](compose/docker-compose.n04-bonsai-rtx.yml).
This dedicated instance uses native memory fitting: the dynamic fast-pool
heuristic's forced 1.1 overhead underestimates Bonsai's 256k KV and CUDA
compute buffers, causing OOM when it packs everything onto one 12-GB card.

With native fitting, 262144 context slots, Q4_0 KV and all 65 GPU layers,
the same 128-token decode benchmark improved from 13.38–13.49 to
21.49–21.56 tokens/s. A separate 4k single-RTX run reached 28.45–28.88
tokens/s. These measurements allocate the full cache but do not fill it
with a 256k-token prompt. See the [validation report](BONSAI-RTX-FA-VALIDATION-2026-09-30.md)
for the placement changes, trace evidence and limits of the comparison.

The revised Bonsai Actions matrix selects the new Ollama release, applies
the compatibility patches, and runs the GGUF tests with the pinned complete
Prism backend. It builds CUDA 11/K80, CUDA 12/Maxwell-Pascal-Volta and CUDA
13/Turing-or-newer candidates. The workflow is prepared locally; no CI run,
registry publication or automatic deployment is claimed by this validation.
See [release automation](GPU-RELEASE-AUTOMATION.md) for the required hardware
gates and current activation status.
The verified production instance on port `11436` was left unchanged.

Request the tested maximum context explicitly; the shared model's stored
defaults were not changed:

```sh
curl http://127.0.0.1:11434/api/generate -d '{
  "model":"bonsai2:27b-pq2_0",
  "prompt":"Compute 19 * 23.",
  "stream":false,
  "options":{"num_ctx":262144,"num_batch":128,"num_predict":256}
}'
```

The implementation uses the [Prism llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
as required by the [Bonsai backend contract](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md).
The user's [video reference](https://www.youtube.com/watch?v=9sPRhdnVSd4)
describes the same runtime requirement and links to the same demo and model.
