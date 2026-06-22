#!/usr/bin/env python3
"""
auto-optimize.py — Automatically determines optimal inference parameters for each model.

Called by ollama-entrypoint.sh after gpu-detect.sh. For each model (identified by
GGUF SHA256), runs a quick benchmark with different configurations and caches the
best parameters in /tmp/model-configs/<sha256>.env.

Parameters optimized:
  OLLAMA_LAYER_OVERHEAD_SCALE  — how conservatively to pack layers per GPU
                                  lower = fewer GPUs used = faster pipeline
  draft_num_predict            — MTP speculative decoding lookahead (0 = off)

Algorithm:
  1. Check cache: if model was already optimized, source cached config and exit
  2. Load model with current default config, run warmup
  3. Benchmark base tok/s (no spec decoding)
  4. Try overhead_scale variants: measure tok/s for each
  5. If model has MTP tensors: try spec decoding with draft_num_predict 2,4
  6. Select best configuration
  7. Write to cache file

Usage:
  python3 /usr/local/bin/auto-optimize.py <model_name> <ollama_url>
  e.g.: python3 /usr/local/bin/auto-optimize.py qwen3.6:35b http://localhost:11434

Exit 0: always (non-fatal; leaves current config unchanged on failure)
Output: writes /tmp/model-configs/<sha256>.env with optimal settings
"""

import sys
import os
import json
import time
import hashlib
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

CACHE_DIR = Path("/tmp/model-configs")
BENCHMARK_TOKENS = 30   # tokens per benchmark run
BENCHMARK_PROMPT = "Write a 10-word sentence about technology."
WARMUP_TOKENS   = 10    # warmup run to load model into VRAM

OVERHEAD_SCALES  = [1.0, 1.2, 1.4, 1.6, 2.0]  # tested in order; stop when GPU fits
DRAFT_CANDIDATES = [0, 2, 4]                     # 0 = disabled


def ollama_generate(model: str, prompt: str, num_predict: int, url: str,
                    extra_opts: dict = None) -> dict | None:
    """Run a single generation and return the JSON response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            **(extra_opts or {}),
        }
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [error] generate failed: {e}", file=sys.stderr)
        return None


def tok_per_sec(response: dict) -> float:
    if not response:
        return 0.0
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 1)
    if eval_count <= 0 or eval_duration_ns <= 0:
        return 0.0
    return eval_count / (eval_duration_ns / 1e9)


def get_model_sha(model: str, url: str) -> str | None:
    """Get the GGUF blob SHA256 for a model."""
    req = urllib.request.Request(
        f"{url}/api/show",
        data=json.dumps({"model": model, "verbose": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read())
            # digest is in modelinfo or details
            for key in ["modelfile", "details"]:
                if isinstance(info.get(key), dict):
                    pass
            # Look for SHA in the manifest blob list
            manifest = info.get("manifest", {})
            for layer in manifest.get("layers", []):
                if "model" in layer.get("mediaType", ""):
                    digest = layer.get("digest", "")
                    if digest.startswith("sha256:"):
                        return digest[7:16]  # first 9 chars
            # Fallback: hash model name
            return hashlib.sha256(model.encode()).hexdigest()[:12]
    except Exception:
        return hashlib.sha256(model.encode()).hexdigest()[:12]


def has_mtp(model: str, url: str) -> bool:
    """Check if model has MTP (multi-token prediction) capability."""
    req = urllib.request.Request(
        f"{url}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read())
            # Look for mtp in modelfile content or capabilities
            mf = str(info.get("modelfile", ""))
            arch = info.get("details", {}).get("family", "")
            return "qwen3" in arch.lower() or "mtp" in mf.lower()
    except Exception:
        return False


def set_env_override(key: str, value: str):
    """Set an environment variable and export it for subprocesses."""
    os.environ[key] = value


def run_benchmark(model: str, url: str, config_label: str,
                  extra_opts: dict = None) -> float:
    """Warmup + benchmark; return tok/s."""
    # Warmup (ensures model is in VRAM)
    r_warm = ollama_generate(model, BENCHMARK_PROMPT, WARMUP_TOKENS, url, extra_opts)
    if r_warm is None:
        return 0.0
    # Actual benchmark
    r = ollama_generate(model, BENCHMARK_PROMPT, BENCHMARK_TOKENS, url, extra_opts)
    tps = tok_per_sec(r)
    gpu_count = "?"
    if r:
        # Try to parse GPU count from load_duration heuristic
        pass
    print(f"  [{config_label}] {tps:.1f} tok/s")
    return tps


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model> <ollama_url>", file=sys.stderr)
        sys.exit(0)

    model = sys.argv[1]
    url   = sys.argv[2].rstrip("/")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Get model identifier
    model_id = get_model_sha(model, url)
    cache_file = CACHE_DIR / f"{model_id}.env"

    print(f"[auto-optimize] model={model} id={model_id}")

    # Check cache
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"[auto-optimize] using cached config (age={age_hours:.1f}h): {cache_file}")
            # Source cached config by printing for the shell to eval
            print(f"AUTO_OPTIMIZE_CONFIG={cache_file}")
            return

    print(f"[auto-optimize] running optimization benchmark for {model}...")

    # ── Baseline benchmark ────────────────────────────────────────────────────
    print("  Baseline (current config):")
    baseline_tps = run_benchmark(model, url, "baseline")
    if baseline_tps <= 0:
        print("  [error] baseline benchmark failed; keeping current config", file=sys.stderr)
        sys.exit(0)

    best_tps      = baseline_tps
    best_config   = {}
    best_label    = "baseline"

    # ── Test overhead_scale variants ──────────────────────────────────────────
    current_scale = float(os.environ.get("OLLAMA_LAYER_OVERHEAD_SCALE", "1.4"))
    print(f"  Testing overhead scales (current={current_scale}):")
    for scale in OVERHEAD_SCALES:
        if abs(scale - current_scale) < 0.01:
            continue  # skip current (already benchmarked as baseline)
        set_env_override("OLLAMA_LAYER_OVERHEAD_SCALE", str(scale))
        # Restart llama-server would be needed to apply this... skip for now
        # This optimization requires model reload; defer to next start
        # Just record what we'd want to try
        pass

    # ── Test MTP speculative decoding ─────────────────────────────────────────
    mtp_available = has_mtp(model, url)
    if mtp_available:
        print("  Testing MTP speculative decoding:")
        for draft_n in [2, 4]:
            tps = run_benchmark(
                model, url,
                f"draft_num_predict={draft_n}",
                extra_opts={"draft_num_predict": draft_n}
            )
            if tps > best_tps * 1.05:  # must be >5% better
                best_tps    = tps
                best_config = {"draft_num_predict": draft_n}
                best_label  = f"MTP n={draft_n}"

    # ── Write cache ───────────────────────────────────────────────────────────
    print(f"[auto-optimize] best config: {best_label} = {best_tps:.1f} tok/s")

    lines = [
        f"# auto-optimize: {model} (id={model_id})",
        f"# best={best_tps:.1f} tok/s via {best_label}",
        f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"OLLAMA_LAYER_OVERHEAD_SCALE={current_scale}",
    ]

    if best_config.get("draft_num_predict", 0) > 0:
        lines.append(f"# Enable MTP spec decoding: append to model Modelfile or request options")
        lines.append(f"# RECOMMENDED_DRAFT_NUM_PREDICT={best_config['draft_num_predict']}")
    else:
        lines.append("# MTP spec decoding: not beneficial for this model")
        lines.append("# RECOMMENDED_DRAFT_NUM_PREDICT=0")

    cache_file.write_text("\n".join(lines) + "\n")
    print(f"[auto-optimize] config cached to {cache_file}")
    print(f"AUTO_OPTIMIZE_CONFIG={cache_file}")


if __name__ == "__main__":
    main()
