# ollama-legacy-gpu

Automated Ollama builds with backward compatibility for older NVIDIA GPUs (CUDA 11 + CUDA 12).

Built for running [MoE Sovereign](https://github.com/h3rb3rn/moe-sovereign) on legacy hardware.

## Supported GPUs

| Image variant | GPU models | Compute Capability | CUDA |
|---|---|---|---|
| `cuda12-maxwell-*` | Tesla M10, Tesla M60 | 5.0/5.2 (Maxwell) | 12.0.1 |
| `cuda11-legacy-*` | Tesla K80, M10, M60 | 3.7 (Kepler) + 5.0/5.2 | 11.8.0 |

> **Tesla K80** requires the `cuda11-legacy` variant. CUDA 12 dropped support for CC 3.7.

## Why this repo?

- Official `ollama/ollama:latest` images ship with CUDA 12.5/12.8 — driver requirements too new for M10/M60
- Ollama dropped CUDA 11 presets in an early release; K80 (CC 3.7) needs them re-injected
- The `ollama37` project is maintained irregularly
- This repository automatically builds every new Ollama release for legacy hardware

## Technical background

**CUDA 12 / Maxwell (CC 5.0/5.2):**
The official `llama_cuda_v12_linux` preset already includes `50-virtual;52-virtual` — Maxwell is supported out of the box. The only problem was the CUDA runtime version: 12.5/12.8 requires driver ≥ R550, which is no longer officially certified for M10/M60. This repo builds with CUDA 12.0.1 (minimum CUDA 12, broadest driver compatibility).

**CUDA 11 / Kepler (CC 3.7):**
Ollama removed CUDA 11 presets in an early release. `scripts/inject-presets.py` adds `llama_cuda_v11_linux` to `llama/server/CMakePresets.json` without patching upstream source. The injection is non-destructive and idempotent.

**PTX-JIT:** Both images use `-virtual` architecture targets (PTX instead of CUBIN). On first model load, kernels are JIT-compiled (~3–30 seconds depending on GPU generation) and cached in `~/.ollama/cuda-cache`. Subsequent loads are instant.

## Quick start

```bash
# Tesla M10 / M60 (CUDA 12):
docker pull ghcr.io/<OWNER>/ollama-legacy:cuda12-maxwell-latest
docker run --rm --gpus all -p 11434:11434 \
  -v /opt/ollama:/root/.ollama \
  ghcr.io/<OWNER>/ollama-legacy:cuda12-maxwell-latest

# Tesla K80 (CUDA 11):
docker pull ghcr.io/<OWNER>/ollama-legacy:cuda11-legacy-latest
docker run --rm --gpus all -p 11434:11434 \
  -v /opt/ollama:/root/.ollama \
  ghcr.io/<OWNER>/ollama-legacy:cuda11-legacy-latest
```

With Docker Compose:
```bash
# M10/M60:
docker compose -f compose/docker-compose.maxwell.yml up -d

# K80:
docker compose -f compose/docker-compose.k80.yml up -d
```

## Pulling HuggingFace models

```bash
# These images support hf.co/ URLs:
docker exec ollama ollama pull hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
```

> **Note:** Ollama's model name validator rejects repository names longer than ~100 characters.
> For models with long names, download the GGUF file manually and import via Modelfile:
> ```bash
> wget -O /tmp/model.gguf "https://huggingface.co/<user>/<repo>/resolve/main/<file>.gguf"
> echo "FROM /tmp/model.gguf" | docker exec -i ollama ollama create mymodel -f -
> ```

## Local build

```bash
git clone https://github.com/<OWNER>/ollama-legacy-gpu
cd ollama-legacy-gpu

# Build both images:
./scripts/build-local.sh all v0.9.0

# Maxwell only (faster):
./scripts/build-local.sh cuda12 latest
```

## Automatic updates

GitHub Actions checks every Monday for a new Ollama release and triggers a build automatically. Results appear in [Packages](../../packages).

## GPU inventory reference

| Node | GPUs | Image | Ollama node name |
|---|---|---|---|
| N04-RTX | RTX + Tesla M10/M60 | `cuda12-maxwell-latest` | N04-RTX |
| N11-M10 | 20× Tesla M10 | `cuda12-maxwell-latest` | N11-M10 |
| N12-M60 | 1× Tesla M60 | `cuda12-maxwell-latest` | N12-M60 |
| N13-K80 | 7× Tesla K80 | `cuda11-legacy-latest` | N13-K80 |

## License

All build scripts and configuration files in this repository: [Apache 2.0](LICENSE)  
Ollama itself: [MIT](https://github.com/ollama/ollama/blob/main/LICENSE)
