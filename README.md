# ollama-legacy-gpu

Automatische Ollama-Builds mit Abwärtskompatibilität für ältere NVIDIA-GPUs (CUDA 11 + CUDA 12).

Gebaut für den Betrieb von [MoE Sovereign](https://github.com/h3rb-rn/moe-sovereign) auf Legacy-Hardware.

## Unterstützte GPUs

| Image-Variante | GPU-Modelle | Compute Capability | CUDA |
|---|---|---|---|
| `cuda12-maxwell-*` | Tesla M10, Tesla M60 | 5.2 (Maxwell) | 12.0.1 |
| `cuda11-legacy-*` | Tesla K80, M10, M60 | 3.7 (Kepler) + 5.2 | 11.8.0 |

> **Tesla K80** erfordert zwingend die `cuda11-legacy`-Variante. CUDA 12 unterstützt CC 3.7 nicht mehr.

## Warum dieser Fork?

- Offizielle `ollama/ollama:latest`-Images basieren auf CUDA 12.5/12.8 — zu neue Treiber für M10/M60
- `hf.co/`-URL-Support erfordert Ollama ≥ 0.30.0
- Das `ollama37`-Projekt wird unregelmäßig gepflegt
- Dieses Repository baut automatisch jede neue Ollama-Version für Legacy-Hardware

## Technischer Hintergrund

**CUDA 12 / Maxwell (CC 5.2):**
Das offizielle `llama_cuda_v12_linux`-Preset enthält bereits `50-virtual;52-virtual` — Maxwell ist unterstützt. Das Problem war nur die CUDA-Runtime-Version: 12.5/12.8 erfordert Treiber ≥ R550, der für M10/M60 nicht mehr offiziell zertifiziert wird. Dieses Repo baut mit CUDA 12.0.1 (minimale CUDA 12, breiteste Treiberkompatibilität).

**CUDA 11 / Kepler (CC 3.7):**
Ollama entfernte CUDA-11-Presets ab v0.30.x. `scripts/inject-presets.py` fügt `llama_cuda_v11_linux` zur `llama/server/CMakePresets.json` hinzu, ohne den Upstream-Source zu patchen. Die Injektion ist nicht-destruktiv und idempotent.

**PTX-JIT:** Beide Images nutzen `-virtual` Architekturziele (PTX statt CUBIN). Beim ersten Modell-Load werden Kernel ~3–10 Sekunden JIT-kompiliert und dann in `~/.ollama/cuda-cache` gecacht.

## Schnellstart

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

Mit Docker Compose:
```bash
# M10/M60:
docker compose -f compose/docker-compose.maxwell.yml up -d

# K80:
docker compose -f compose/docker-compose.k80.yml up -d
```

## HuggingFace-Modelle pullen (ab Ollama 0.30.0)

```bash
# Diese Images unterstützen hf.co/-URLs:
docker exec ollama ollama pull hf.co/DavidAU/Qwen3.6-40B-...
```

## Lokaler Build

```bash
git clone https://github.com/<OWNER>/ollama-legacy-gpu
cd ollama-legacy-gpu

# Beide Images bauen:
./scripts/build-local.sh all v0.9.0

# Nur Maxwell (schneller):
./scripts/build-local.sh cuda12 latest
```

## Automatische Updates

GitHub Actions prüft jeden Montag ob eine neue Ollama-Version verfügbar ist und löst automatisch einen Build aus. Ergebnisse erscheinen in [Packages](../../packages).

## GPU-Inventar-Empfehlung

| Knoten | GPUs | Image | Ollama-Node-Name |
|---|---|---|---|
| N04-RTX | RTX-GPUs | `ollama/ollama:latest` | N04-RTX |
| N11-M10 | 20× Tesla M10 | `cuda12-maxwell-latest` | N11-M10 |
| N12-M60 | 1× Tesla M60 | `cuda12-maxwell-latest` | N12-M60 |
| N13-K80 | 7× Tesla K80 | `cuda11-legacy-latest` | N13-K80 |

## Lizenz

Alle Build-Skripte und Konfigurationsdateien in diesem Repository: [Apache 2.0](LICENSE)
Ollama selbst: [MIT](https://github.com/ollama/ollama/blob/main/LICENSE)
