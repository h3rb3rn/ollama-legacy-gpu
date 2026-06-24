#!/usr/bin/env python3
"""
ollama-proxy.py — Transparent HTTP proxy that automatically optimizes any model
on first use and applies cached parameters on subsequent requests.

Architecture:
  [Client:11434] → [Proxy:11434] → [Ollama:11435]
                         ↓
                   Model detected?
                   Cache exists? → apply scale override → forward
                   No cache?     → forward immediately (async optimize in BG)
                                   first response = default settings
                                   next response = optimal settings

Optimization trigger:
  Any request to /api/generate, /api/chat, /v1/chat/completions
  → extract model name from JSON body
  → if no cache: spawn auto-optimize.py in background thread
  → if cache exists: write scale to /tmp/ollama-scale-override

This ensures:
  - Zero latency on first request (optimization runs in BG)
  - Optimal settings automatically applied on next model load
  - Works for ALL models (qwen3.6:35b, llama4:scout, etc.)
  - Cache persisted in /root/.ollama/auto-optimize/ (volume-mounted)

Usage:
  python3 /usr/local/bin/ollama-proxy.py [--backend http://localhost:11435] [--port 11434]
"""

import http.server
import http.client
import urllib.parse
import urllib.request
import json
import os
import sys
import time
import threading
import subprocess
import hashlib
import logging
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BACKEND_URL     = os.environ.get("OLLAMA_BACKEND_URL", "http://localhost:11435")
LISTEN_PORT     = int(os.environ.get("OLLAMA_PROXY_PORT", "11434"))
OPTIMIZE_SCRIPT = "/usr/local/bin/auto-optimize.py"
CACHE_DIR       = Path("/root/.ollama/auto-optimize")
SCALE_OVERRIDE  = Path("/tmp/ollama-scale-override")
OPTIMIZE_LOCK        = {}  # model → threading.Event (prevent duplicate runs per model)
OPTIMIZE_LOCK_MUTEX  = threading.Lock()
GLOBAL_OPT_SEMAPHORE = threading.Semaphore(1)  # only one optimizer runs at a time

INFERENCE_PATHS = {"/api/generate", "/api/chat", "/v1/chat/completions",
                   "/v1/responses", "/api/embed", "/api/embeddings"}

logging.basicConfig(level=logging.INFO,
                    format="[proxy] %(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("proxy")


def get_model_sha(model: str) -> str:
    """Get model SHA from Ollama API (used as cache key)."""
    try:
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/show",
            data=json.dumps({"model": model, "verbose": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            info = json.loads(resp.read())
            for layer in info.get("manifest", {}).get("layers", []):
                d = layer.get("digest", "")
                if d.startswith("sha256:") and "model" in layer.get("mediaType", ""):
                    return d[7:19]
    except Exception:
        pass
    return hashlib.sha256(model.encode()).hexdigest()[:12]


def apply_cached_config(sha: str) -> bool:
    """If cache exists for sha, write scale override and return True."""
    cache_file = CACHE_DIR / f"{sha}.json"
    if not cache_file.exists():
        return False
    try:
        config = json.loads(cache_file.read_text())
        scale = config.get("scale", 1.4)
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h > 168:  # 7 days
            return False
        SCALE_OVERRIDE.write_text(f"{scale}\n")
        log.info(f"applied cached config: sha={sha} scale={scale} "
                 f"tps={config.get('tok_per_sec','?')} gpus={config.get('gpus','?')}"
                 f" (age={age_h:.0f}h)")
        return True
    except Exception as e:
        log.warning(f"failed to apply cache for {sha}: {e}")
        return False


def run_optimization(model: str, sha: str):
    """Run auto-optimize.py in background thread.

    Only one optimizer runs at a time (GLOBAL_OPT_SEMAPHORE) to avoid
    concurrent model reloads interfering with each other's benchmarks.
    Per-model lock (OPTIMIZE_LOCK) prevents duplicate runs for the same model.
    """
    with OPTIMIZE_LOCK_MUTEX:
        if sha in OPTIMIZE_LOCK:
            return  # already optimizing or queued
        OPTIMIZE_LOCK[sha] = threading.Event()

    try:
        log.info(f"starting background optimization: model={model} sha={sha}")
        # Wait until no other model is being optimized
        with GLOBAL_OPT_SEMAPHORE:
            result = subprocess.run(
                ["python3", OPTIMIZE_SCRIPT, model, BACKEND_URL],
                capture_output=False,   # let stdout/stderr flow to container logs
                timeout=3600
            )
            if result.returncode == 0:
                log.info(f"optimization complete: sha={sha}")
                apply_cached_config(sha)
            else:
                log.warning(f"optimization failed (exit {result.returncode}): model={model}")
    except subprocess.TimeoutExpired:
        log.warning(f"optimization timed out after 1h: sha={sha}")
    except Exception as e:
        log.warning(f"optimization error: {e}")
    finally:
        with OPTIMIZE_LOCK_MUTEX:
            ev = OPTIMIZE_LOCK.pop(sha, None)
            if ev:
                ev.set()


def extract_model(path: str, body: bytes) -> str | None:
    """Extract model name from request path or body."""
    if not body:
        return None
    try:
        data = json.loads(body)
        return data.get("model") or data.get("name")
    except Exception:
        return None


def get_cached_draft(sha: str) -> int:
    """Return optimal draft_num_predict from cache (0 = disabled)."""
    cache_file = CACHE_DIR / f"{sha}.json"
    if cache_file.exists():
        try:
            config = json.loads(cache_file.read_text())
            return config.get("draft_num_predict", 0)
        except Exception:
            pass
    return 0


def inject_optimal_options(body: bytes, sha: str) -> bytes:
    """
    Inject cached optimal options (draft_num_predict) into request body.
    Only adds options not already set by the caller.
    """
    try:
        data = json.loads(body)
    except Exception:
        return body

    draft_n = get_cached_draft(sha)
    if draft_n <= 0:
        return body

    # Don't override if caller explicitly set draft_num_predict
    opts = data.setdefault("options", {})
    if "draft_num_predict" not in opts:
        opts["draft_num_predict"] = draft_n
        log.debug(f"injected draft_num_predict={draft_n} for sha={sha[:8]}")
        return json.dumps(data).encode()

    return body


# Track model→sha mapping (reduces API calls)
_sha_cache: dict[str, str] = {}
_sha_lock = threading.Lock()


def get_sha_cached(model: str) -> str:
    with _sha_lock:
        if model in _sha_cache:
            return _sha_cache[model]
    sha = get_model_sha(model)
    if sha:
        with _sha_lock:
            _sha_cache[model] = sha
    return sha or ""


def maybe_optimize_model(model: str | None) -> str:
    """Check model cache; trigger optimization if stale/missing. Returns sha."""
    if not model:
        return ""
    try:
        sha = get_sha_cached(model)
        if not sha:
            return ""
        if apply_cached_config(sha):
            return sha  # cached, applied
        # No cache: trigger background optimization (non-blocking)
        with OPTIMIZE_LOCK_MUTEX:
            if sha not in OPTIMIZE_LOCK:
                t = threading.Thread(
                    target=run_optimization,
                    args=(model, sha),
                    daemon=True,
                    name=f"optimizer-{sha[:8]}"
                )
                t.start()
                log.info(f"triggered optimization for new model: {model} ({sha})")
            else:
                log.debug(f"optimization already running for: {model}")
        return sha
    except Exception as e:
        log.debug(f"maybe_optimize error: {e}")
        return ""


# ── HTTP Proxy Handler ────────────────────────────────────────────────────────

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Transparent HTTP proxy with model-aware optimization trigger."""

    def log_message(self, fmt, *args):
        # Suppress per-request logs (Ollama already logs them)
        pass

    def _forward(self, method: str, body: bytes):
        """Forward request to Ollama backend and pipe response back."""
        parsed = urllib.parse.urlparse(BACKEND_URL)
        host = parsed.hostname
        port = parsed.port or 80

        # Check if this is an inference request worth optimizing
        if self.path in INFERENCE_PATHS and method == "POST":
            model = extract_model(self.path, body)
            if model:
                # Run optimization check (fast path: returns sha from cache)
                sha = maybe_optimize_model(model)
                # Inject optimal draft settings if cached
                if sha and body:
                    body = inject_optimal_options(body, sha)

        # Forward to backend
        # Use longer timeout for inference/generate paths: large models (llama4:scout 62 GB)
        # can take >8 minutes to load before the first token arrives. The Ollama scheduler
        # propagates a canceled HTTP connection as "context canceled", killing llama-server.
        # Match OLLAMA_LOAD_TIMEOUT (20 min) so the scheduler's own timeout fires first.
        _backend_timeout = 1800 if self.path in INFERENCE_PATHS else 60
        try:
            conn = http.client.HTTPConnection(host, port, timeout=_backend_timeout)
            # Forward headers, skip hop-by-hop
            skip_headers = {"host", "connection", "transfer-encoding",
                            "keep-alive", "proxy-connection", "te", "trailers",
                            "upgrade", "proxy-authorization"}
            fwd_headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in skip_headers}
            fwd_headers["Host"] = f"{host}:{port}"
            if body:
                fwd_headers["Content-Length"] = str(len(body))

            conn.request(method, self.path, body=body, headers=fwd_headers)
            backend_resp = conn.getresponse()

            self.send_response(backend_resp.status)
            for key, val in backend_resp.getheaders():
                if key.lower() not in {"connection", "transfer-encoding"}:
                    self.send_header(key, val)
            self.end_headers()

            # Stream response body
            while True:
                chunk = backend_resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as e:
            log.warning(f"proxy error ({method} {self.path}): {e}")
            try:
                self.send_error(502, f"Backend error: {e}")
            except Exception:
                pass

    def do_GET(self):
        self._forward("GET", b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._forward("POST", body)

    def do_DELETE(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._forward("DELETE", body)

    def do_HEAD(self):
        self._forward("HEAD", b"")

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._forward("PUT", body)


# ── Server startup ────────────────────────────────────────────────────────────

class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def wait_for_backend(backend_url: str, timeout: int = 120):
    """Wait until Ollama backend is ready."""
    log.info(f"waiting for backend: {backend_url}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{backend_url}/api/tags", timeout=3)
            log.info("backend ready")
            return True
        except Exception:
            time.sleep(2)
    log.warning("backend not ready after timeout, starting proxy anyway")
    return False


def main():
    global BACKEND_URL, LISTEN_PORT

    # Parse arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--backend" and i < len(sys.argv):
            BACKEND_URL = sys.argv[i + 1]
        elif arg == "--port" and i < len(sys.argv):
            LISTEN_PORT = int(sys.argv[i + 1])
        elif arg.startswith("--backend="):
            BACKEND_URL = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            LISTEN_PORT = int(arg.split("=", 1)[1])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wait_for_backend(BACKEND_URL)

    server = ThreadedHTTPServer(("0.0.0.0", LISTEN_PORT), ProxyHandler)
    log.info(f"proxy listening on :{LISTEN_PORT} → {BACKEND_URL}")
    log.info("auto-optimizer: any model request triggers background optimization")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("proxy stopped")


if __name__ == "__main__":
    main()
