#!/usr/bin/env python3
"""
auto-optimize.py — Closed-loop automatic parameter optimizer for Ollama models.

Finds optimal OVERHEAD_SCALE (GPU packing density) and spec decoding settings
through iterative benchmarking with model reloads between each test.

Architecture:
  - selectGPUPool() in Ollama reads from OVERRIDE_FILE (/tmp/ollama-scale-override)
    on every model load, so changing this file takes effect on the next reload.
  - auto-optimize.py writes different scales to OVERRIDE_FILE, triggers model
    unload+reload via Ollama API, and benchmarks tok/s.
  - Optimal config is cached in CACHE_DIR/<sha256>.json (persistent via volume).
  - ollama-entrypoint.sh calls this in background after first model load.

Flow:
  Container starts → gpu-detect.sh → Ollama serves with default scale (1.4)
  First model load → auto-optimizer spawned in background
    → tries scales [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    → tries spec decoding [0, 2, 4] at best scale
    → caches result → future loads use optimal config immediately

Usage:
  python3 /usr/local/bin/auto-optimize.py <model_name> [ollama_url]
  Environment: OLLAMA_FAST_POOL_VRAM_GB, OLLAMA_FAST_GPU_DEVICES (from gpu-detect.sh)
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

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_URL       = "http://localhost:11434"
OVERRIDE_FILE    = Path("/tmp/ollama-scale-override")
CACHE_DIR        = Path("/tmp/model-configs")           # cache in tmpfs
PERSIST_CACHE    = Path("/root/.ollama/auto-optimize")  # persisted via volume
BENCHMARK_TOKENS = 25
WARMUP_TOKENS    = 5
BENCHMARK_PROMPT = "List 3 programming languages and their main use cases."
CACHE_TTL_HOURS  = 168  # 7 days

# Scales to try (lowest first = fewest GPUs = fastest if it fits)
SCALE_CANDIDATES = [1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
DRAFT_CANDIDATES = [0, 2, 4]

# ── Ollama API helpers ────────────────────────────────────────────────────────

def api(method: str, path: str, body: dict = None, timeout: int = 300) -> dict | None:
    url = f"{OLLAMA_URL}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [api] {method} {path} failed: {e}", file=sys.stderr)
        return None


def generate(model: str, extra_opts: dict = None, n_tokens: int = BENCHMARK_TOKENS) -> dict | None:
    return api("POST", "/api/generate", {
        "model": model,
        "prompt": BENCHMARK_PROMPT,
        "stream": False,
        "options": {"num_predict": n_tokens, **(extra_opts or {})},
    }, timeout=300)


def unload(model: str):
    """Unload model from VRAM (keep_alive=0)."""
    api("POST", "/api/generate", {
        "model": model, "prompt": "", "stream": False,
        "keep_alive": "0s", "options": {"num_predict": 0},
    }, timeout=30)
    time.sleep(3)


def model_sha(model: str) -> str:
    """Get first 12 chars of GGUF blob SHA for cache key."""
    info = api("POST", "/api/show", {"model": model, "verbose": False}, timeout=30)
    if info:
        for layer in info.get("manifest", {}).get("layers", []):
            d = layer.get("digest", "")
            if d.startswith("sha256:") and "model" in layer.get("mediaType", ""):
                return d[7:19]
    return hashlib.sha256(model.encode()).hexdigest()[:12]


def tok_per_sec(resp: dict | None) -> float:
    if not resp:
        return 0.0
    n = resp.get("eval_count", 0)
    d = resp.get("eval_duration", 1)
    return n / (d / 1e9) if n > 0 and d > 0 else 0.0


def gpu_count_used() -> int:
    """Count GPUs with >500 MiB used via NVML ctypes (no nvidia-smi required)."""
    try:
        import ctypes
        for lib in ["libnvidia-ml.so.1", "/usr/local/nvidia/lib64/libnvidia-ml.so.1"]:
            try:
                nvml = ctypes.CDLL(lib)
                break
            except OSError:
                continue
        else:
            return 0
        nvml.nvmlInit_v2()
        count = ctypes.c_uint()
        nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
        class _Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]
        result = 0
        for i in range(count.value):
            h = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
            m = _Mem()
            nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m))
            if m.used > 500 * 1024 * 1024:
                result += 1
        nvml.nvmlShutdown()
        return result
    except Exception:
        return 0


def has_mtp(model: str) -> bool:
    """Heuristic: qwen3.x models typically have built-in MTP."""
    info = api("POST", "/api/show", {"model": model}, timeout=30)
    if not info:
        return False
    arch = info.get("details", {}).get("family", "").lower()
    return "qwen3" in arch or "qwen35" in arch


# ── Cache management ──────────────────────────────────────────────────────────

def load_cache(sha: str) -> dict | None:
    for cache_dir in [PERSIST_CACHE, CACHE_DIR]:
        f = cache_dir / f"{sha}.json"
        if f.exists():
            age = (time.time() - f.stat().st_mtime) / 3600
            if age < CACHE_TTL_HOURS:
                try:
                    data = json.loads(f.read_text())
                    print(f"[auto-optimize] cache hit (age={age:.0f}h): {data}")
                    return data
                except Exception:
                    pass
    return None


def save_cache(sha: str, config: dict):
    for cache_dir in [PERSIST_CACHE, CACHE_DIR]:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{sha}.json").write_text(json.dumps(config, indent=2))
        except Exception as e:
            print(f"  [cache] write to {cache_dir} failed: {e}", file=sys.stderr)


# ── Override file (read by selectGPUPool on every model load) ─────────────────

def write_override(scale: float):
    """Write scale override; selectGPUPool() reads this on each model load."""
    OVERRIDE_FILE.write_text(f"{scale}\n")


def clear_override():
    try:
        OVERRIDE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(model: str, scale: float | None, draft_n: int = 0,
                  label: str = "") -> tuple[float, int]:
    """
    Load model with given scale, benchmark tok/s, return (tok/s, gpu_count).
    Triggers unload + reload via API.
    """
    if scale is not None:
        write_override(scale)

    # Unload to force reload with new scale
    unload(model)

    # Warmup: triggers reload with new OVERHEAD_SCALE
    extra = {"draft_num_predict": draft_n} if draft_n > 0 else {}
    warmup = generate(model, extra, WARMUP_TOKENS)
    if not warmup:
        return 0.0, 0

    # Brief pause to let GPU settle
    time.sleep(1)
    gpus = gpu_count_used()

    # Benchmark
    resp = generate(model, extra, BENCHMARK_TOKENS)
    tps = tok_per_sec(resp)

    tag = label or (f"scale={scale}" if scale else "current")
    if draft_n > 0:
        tag += f" draft={draft_n}"
    print(f"  [{tag}] {tps:.1f} tok/s on {gpus} GPU(s)")
    return tps, gpus


# ── Main optimizer ────────────────────────────────────────────────────────────

def optimize(model: str) -> dict:
    """
    Run full optimization loop. Returns optimal config dict:
    {"scale": float, "draft_num_predict": int, "tok_per_sec": float, "gpus": int}
    """
    sha = model_sha(model)
    print(f"[auto-optimize] optimizing model={model} sha={sha}")

    # ── Phase 1: Find optimal overhead scale (affects GPU count) ─────────────
    # Binary search / linear scan from lowest scale (fewest GPUs) upward.
    # Stop at first scale where model loads successfully (no OOM).
    print("Phase 1: overhead_scale optimization (fewer GPUs = faster pipeline)")

    best_scale = float(os.environ.get("OLLAMA_LAYER_OVERHEAD_SCALE", "1.4"))
    best_tps   = 0.0
    best_gpus  = 99
    scale_results = {}

    for scale in SCALE_CANDIDATES:
        tps, gpus = run_benchmark(model, scale, label=f"scale={scale}")
        scale_results[scale] = (tps, gpus)
        if tps <= 0:
            print(f"    scale={scale} → OOM or failed, skipping higher values")
            continue  # try next scale (more headroom)
        if tps > best_tps or gpus < best_gpus:
            # Prefer: higher tps AND fewer GPUs
            if gpus <= best_gpus:  # don't accept more GPUs for marginal tps gain
                best_tps   = tps
                best_scale = scale
                best_gpus  = gpus

    print(f"  Best scale: {best_scale} → {best_tps:.1f} tok/s on {best_gpus} GPU(s)")

    # ── Phase 2: Spec decoding test at ALL scales ────────────────────────────────
    # MTP benefit depends on per-GPU forward-pass time, which varies with GPU count.
    # Optimal (scale, draft) combination may differ from optimal scale without MTP.
    # Example: scale=1.6 (4 GPUs, shorter passes) + MTP can beat scale=1.1 (3 GPUs)
    # without MTP, even though scale=1.1 wins in the no-MTP Phase 1 comparison.
    best_draft = 0
    mtp_available = has_mtp(model)

    if mtp_available:
        print(f"Phase 2: MTP spec decoding test at all scales")
        for scale in SCALE_CANDIDATES:
            if scale_results.get(scale, (0, 0))[0] <= 0:
                continue  # skip scales that failed in Phase 1
            for draft_n in DRAFT_CANDIDATES:
                if draft_n == 0:
                    continue  # baseline already measured in Phase 1
                tps, gpus = run_benchmark(model, scale, draft_n=draft_n,
                                          label=f"scale={scale} draft={draft_n}")
                if tps > best_tps * 1.05:  # must be >5% improvement over current best
                    best_tps   = tps
                    best_scale = scale
                    best_draft = draft_n
                    best_gpus  = gpus
                    print(f"    scale={scale} draft={draft_n} is new best!")

    # ── Restore optimal settings ──────────────────────────────────────────────
    write_override(best_scale)
    unload(model)

    # Final load with optimal config (no benchmarking, just ensure it's warm)
    extra = {"draft_num_predict": best_draft} if best_draft > 0 else {}
    generate(model, extra, WARMUP_TOKENS)

    if best_tps <= 0:
        print(f"[auto-optimize] all benchmarks failed (tps=0) — not caching, using default scale={best_scale}",
              file=sys.stderr)
        write_override(best_scale)
        return {"model": model, "sha": sha, "scale": best_scale,
                "draft_num_predict": 0, "tok_per_sec": 0.0, "gpus": 0,
                "optimized_at": time.strftime("%Y-%m-%d %H:%M:%S"), "failed": True}

    config = {
        "model": model,
        "sha": sha,
        "scale": best_scale,
        "draft_num_predict": best_draft,
        "tok_per_sec": best_tps,
        "gpus": best_gpus,
        "optimized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scale_results": {str(k): list(v) for k, v in scale_results.items()},
    }
    save_cache(sha, config)
    print(f"[auto-optimize] done: {config}")
    return config


def apply_cached(config: dict):
    """Apply cached settings to override file."""
    scale = config.get("scale", 1.4)
    write_override(scale)
    print(f"[auto-optimize] applied cached settings: scale={scale}, "
          f"draft={config.get('draft_num_predict', 0)}, "
          f"tps={config.get('tok_per_sec', '?'):.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else None
    if not model:
        print("Usage: auto-optimize.py <model_name> [ollama_url]", file=sys.stderr)
        sys.exit(0)

    if len(sys.argv) > 2:
        global OLLAMA_URL
        OLLAMA_URL = sys.argv[2].rstrip("/")

    sha = model_sha(model)
    cached = load_cache(sha)
    if cached:
        apply_cached(cached)
        # Write a Modelfile hint for draft_num_predict
        if cached.get("draft_num_predict", 0) > 0:
            hint = f"# HINT: set draft_num_predict={cached['draft_num_predict']} in request for {model}"
            Path("/tmp/model-configs/hints.txt").write_text(hint + "\n")
        return

    # Run optimization
    optimize(model)


if __name__ == "__main__":
    main()
