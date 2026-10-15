"""
prefix_router.py — choose which Ollama backend should answer a request (prefix affinity).

A conversation's next turn shares its whole history with the previous one. The backend that
served the previous turn still holds that history in its prompt cache, so sending the turn back
there saves re-reading the prompt. This module keeps, per backend, an LRU of hashed prompt
prefixes and picks the backend with the lowest cost:

  cost = (prompt characters the backend would have to read again) / 4
         + LOAD_TOKENS * requests already running there

Pure logic, no I/O: ollama-proxy.py feeds it requests and backend health. Used only when
OLLAMA_BACKEND_URLS is set; otherwise the proxy does not import this module.

Idea and constants from Strata's serve/replicas.py (LOAD_TOKENS, 64-token block hashes). The
block size here is in characters, because Ollama's tokenizer is not available to the proxy.
"""
import collections
import hashlib
import json
import threading
import time

BLOCK_CHARS = 256        # characters per hashed block of a plain prompt
LOAD_TOKENS = 3000       # cost of one request already running on a backend, in prompt tokens
INDEX_MAX = 50000        # prefixes remembered per backend (LRU)
DOWN_S = 30.0            # a backend that failed before answering is skipped this long
MODEL_SUFFIX = ":latest"


def _norm_model(name):
    name = (name or "").strip()
    return name if ":" in name else name + MODEL_SUFFIX


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def prefix_keys(body):
    """Return ([(key, chars_upto), ...], total_chars) for a request body.

    Chat-style bodies (`messages`) get one key per message; `tools` are part of the first key,
    since a changed tool list changes the prompt. Plain `prompt` strings get one key per full
    block of BLOCK_CHARS. Keys form a hash chain, so a key implies the whole prefix before it.
    """
    if not isinstance(body, dict):
        return [], 0
    units = []
    if isinstance(body.get("messages"), list):
        if body.get("tools"):
            units.append(("tools", _text(body["tools"])))
        for m in body["messages"]:
            if isinstance(m, dict):
                units.append((str(m.get("role", "")), _text(m.get("content", ""))))
        return _chain_units(units)
    prompt = body.get("prompt")
    if isinstance(prompt, str) and prompt:
        blocks = [prompt[i:i + BLOCK_CHARS] for i in range(0, len(prompt) - BLOCK_CHARS + 1, BLOCK_CHARS)]
        return _chain_units([("prompt", b) for b in blocks], total=len(prompt))
    return [], 0


def _chain_units(units, total=None):
    keys, prev, chars = [], "", 0
    for role, content in units:
        chars += len(content)
        prev = hashlib.sha1(f"{prev}\0{role}\0{content}".encode("utf-8")).hexdigest()
        keys.append((prev, chars))
    return keys, (chars if total is None else total)


class PrefixRouter:
    def __init__(self, backends, clock=time.monotonic):
        if not backends:
            raise ValueError("PrefixRouter needs at least one backend")
        self.backends = list(backends)
        self._clock = clock
        self._lock = threading.Lock()
        self._lru = {b: collections.OrderedDict() for b in self.backends}
        self._inflight = {b: 0 for b in self.backends}
        self._down_until = {b: 0.0 for b in self.backends}
        self._models = {b: None for b in self.backends}   # None = not yet known

    # ── health and models ────────────────────────────────────────────────────

    def mark_down(self, backend):
        with self._lock:
            self._down_until[backend] = self._clock() + DOWN_S

    def set_models(self, backend, names):
        with self._lock:
            self._models[backend] = {_norm_model(n) for n in names}

    def has_model(self, backend, model):
        with self._lock:
            models = self._models[backend]
        return models is None or _norm_model(model) in models

    def _healthy(self, backend, now):
        return self._down_until[backend] <= now

    # ── routing ──────────────────────────────────────────────────────────────

    def choose(self, model, keys, total_chars, exclude=()):
        """Return the backend for this request, or None if no backend is left."""
        now = self._clock()
        with self._lock:
            best, best_cost = None, None
            for b in self.backends:
                if b in exclude or not self._healthy(b, now):
                    continue
                if self._models[b] is not None and _norm_model(model) not in self._models[b]:
                    continue
                matched = self._matched_chars(b, keys)
                cost = max(0, total_chars - matched) / 4 + LOAD_TOKENS * self._inflight[b]
                if best_cost is None or cost < best_cost:
                    best, best_cost = b, cost
            return best

    def _matched_chars(self, backend, keys):
        lru = self._lru[backend]
        for key, upto in reversed(keys):
            if key in lru:
                return upto
        return 0

    def begin(self, backend):
        with self._lock:
            self._inflight[backend] += 1

    def end(self, backend, keys, ok):
        with self._lock:
            self._inflight[backend] = max(0, self._inflight[backend] - 1)
            if ok:
                lru = self._lru[backend]
                for key, _ in keys:
                    lru[key] = True
                    lru.move_to_end(key)
                while len(lru) > INDEX_MAX:
                    lru.popitem(last=False)

    def snapshot(self):
        """Current state for logging and tests."""
        with self._lock:
            return {b: {"inflight": self._inflight[b], "prefixes": len(self._lru[b])}
                    for b in self.backends}
