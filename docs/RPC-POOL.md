# Cross-host GPU pool (llama.cpp RPC)

One Ollama instance uses the GPUs of several hosts, so that a model larger than the VRAM of one host can be loaded.
Built for servers with 12 Tesla M10 each (96 GiB per host) connected by 1 GbE. Status: proof of concept. The image
is built and the path works end to end between two containers on CPU (see "Verified"); it has **not yet run on GPU
hardware**.

## How it works

Ollama starts `llama-server` (llama.cpp) for every model. llama.cpp has an RPC backend: `rpc-server` on a remote host
exports that host's GPUs, and `llama-server --rpc host:port` treats them as additional devices. Layers are split
across local and remote devices as in any multi-GPU load; for a remote device the graph and the activations travel
over TCP.

```
head host                                   worker host(s)
┌──────────────────────────────┐            ┌───────────────────────────┐
│ ollama serve                 │            │ rpc-server -H <ip> -c     │
│  └ llama-server --rpc w:50052│── TCP ────▶│  CUDA0 … CUDA11           │
│     CUDA0 … CUDA11 (local)   │            │  tensor cache /cache/rpc  │
└──────────────────────────────┘            └───────────────────────────┘
```

The device list inside `llama-server` is `[RPC devices..., CUDA0 … CUDAn-1]`: remote devices come first.

## What the fork adds

| Part | File | Purpose |
| --- | --- | --- |
| Build flag | `dockerfiles/Dockerfile.cuda12-maxwell` (`-DGGML_RPC=ON`) | builds `libggml-rpc.so` and `rpc-server`; the build fails if either is missing from the image |
| Go patch | `scripts/patch-ollama-rpc.py` | `OLLAMA_RPC_SERVERS` → `--rpc` on the model launch; skips the pool logic that counts local CUDA devices |
| Tool | `scripts/rpc-seed-cache.cpp` → `/usr/lib/ollama/rpc-seed-cache` | fills the worker's tensor cache from a local GGUF, so weights are not sent over the network |

Only the `cuda12-maxwell` image has these parts. llama.cpp b11351 names the binary `ggml-rpc-server`; the image
links it as `/usr/lib/ollama/rpc-server`.

### `OLLAMA_RPC_SERVERS`

Comma-separated `host:port` list on the head, at most 16 entries (`GGML_RPC_MAX_SERVERS`). Unset: no change.

- The list is passed as `--rpc` to the `llama-server` of a model load only. It is not exported as `LLAMA_ARG_RPC`:
  that variable would also reach `llama-server --list-devices` in the GPU discovery, where the remote devices have
  no compute capability and `patch-ollama-discovery.py` drops unknown devices.
- `selectGPUPool()` returns before the fast-pool decision: no `CUDA_VISIBLE_DEVICES` restriction, no `--main-gpu`,
  no cached `--tensor-split`. All three count local CUDA devices and would be off by the number of RPC devices.
- Entries without a valid port are ignored with a warning.

### Layer placement

With `OLLAMA_FORCE_GPU_LAYERS=1` the greedy fill (`patch-llama-tier-fitting.py`) starts at the highest device index.
Because RPC devices come first in the list, the local GPUs are filled first and the remote ones take the rest. The
model is cut once between the hosts, so the hidden state crosses the network once per decoded token (and the
result once back). A model that fits into the head never uses a worker.

Leave `OLLAMA_GPU_BANDWIDTHS` unset (`OLLAMA_GPU_AUTODETECT=0`) on the head: the factors are indexed in local CUDA
order and would be shifted by the number of RPC devices.

### Weights stay local: `rpc-seed-cache`

For every weight tensor above 10 MiB the head sends the FNV-1a hash of the data first. A worker started with `-c`
looks for `$LLAMA_CACHE/rpc/<hash as 16 hex digits>` and loads the tensor from there; only on a miss the head sends
the data (about 110 MB/s on 1 GbE, half an hour per 200 GiB) and the worker stores it.

`rpc-seed-cache MODEL.gguf` writes those cache files from the local copy of the model:

```
rpc-seed-cache [--cache-dir DIR] [--threads N] [--dry-run] MODEL.gguf [MODEL2.gguf ...]
```

- Default directory is the one `rpc-server -c` uses (`$LLAMA_CACHE/rpc`).
- Files are written under a temporary name and renamed; existing files of the right size are skipped.
- Tensors up to 10 MiB (norms, biases) are always sent inline; that is a few MiB per model.
- The cache is a second copy of the model on the worker's disk. Reading tensors straight from the blob would need a
  patch to `rpc-server` and is left for later.
- The head must load the same bytes the worker hashed: same GGUF blob on every host.

Checked against `rpc-server` (2026-11-12, qwen2.5:0.5b): the file the server stored on a cache miss and the file the
tool wrote from the blob have the same name and the same bytes; with the seeded cache the next load stored nothing.

## Deployment

Compose templates and the bring-up sequence are in the deployment repository under `llm-studio/worker-m10x12/`.

Worker (`rpc-server` is part of the image, the Ollama entrypoint is bypassed):

```
/usr/lib/ollama/rpc-server -H <cluster ip> -p 50052 -c
LLAMA_CACHE=/cache
```

`GGML_BACKEND_PATH` stays unset on the worker: `libggml-cuda.so` sits next to the binary and is loaded from there.

Head: `OLLAMA_RPC_SERVERS=<worker>:50052[,…]`, `OLLAMA_FORCE_GPU_LAYERS=1`, `OLLAMA_LAYER_OVERHEAD_SCALE=1.10`,
`LLAMA_PIPELINE_PARALLEL=0`, `OLLAMA_GPU_AUTODETECT=0`, `OLLAMA_AUTO_OPTIMIZE=0`, `OLLAMA_LOAD_TIMEOUT=60m`.

`rpc-server` has no authentication and no encryption (upstream: "never expose the RPC server to an open network").
Bind it to the cluster address and restrict the port to the head.

## Building the image

The Actions workflow only runs on the default branch. Until `feat/rpc-pool` is merged, build locally:

```bash
docker build -f dockerfiles/Dockerfile.cuda12-maxwell \
  --build-arg OLLAMA_VERSION=v0.40.1 --build-arg GGML_CUDA_FA=ON \
  --build-arg "CUDA_ARCHITECTURES=50-real;52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real" \
  -t ghcr.io/h3rb3rn/ollama-legacy:cuda12-maxwell-rpc-0.40.1 .
```

Head and workers must run the same image: the RPC protocol version (7.0.0 in llama.cpp b11351) is compared on
connect and changes with the ggml operator set.

## Verified

Two containers of the image on one host without GPU, 2026-11-12 (the worker exports its CPU device):

- Without `OLLAMA_RPC_SERVERS`: no `--rpc` on the launch, model loads and answers as before.
- With `OLLAMA_RPC_SERVERS=<worker>:50052`: device list `RPC0 : <worker>:50052`, greedy fill puts 25/25 layers on it,
  model, KV and compute buffers are allocated on the worker, the request is answered.
- Tensor cache: see `rpc-seed-cache` above.

## Open points

- **Scheduler view.** Ollama's scheduler knows the head's 12 GPUs only. Placement is delegated to `llama-server`, and
  the VRAM preflight applies only when other models have to be evicted, so a model above 96 GiB should start; this
  has to be confirmed on hardware. Related: with a model larger than the local VRAM Ollama may switch mmap off for the
  load, which is harmless for RPC devices.
- **12 CUDA devices in one `rpc-server`** are untested. The VMM peer-access limit (`patch-llama-vmm-peer-access.py`)
  is in the image.
- **The proxy and the layout cache** (`auto-optimize.py`) measure local VRAM through NVML and are off in this mode.
- **Throughput.** Every graph evaluation on a remote device is a round trip; expect decode rates below the 12-GPU
  single-host numbers in `TUNING.md`. Numbers follow once the hardware is available.
