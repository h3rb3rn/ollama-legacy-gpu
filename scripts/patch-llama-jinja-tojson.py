#!/usr/bin/env python3
"""
patch-llama-jinja-tojson.py — Adds a missing `tojson` filter registration for
undefined values in llama.cpp's in-house Jinja engine (common/jinja/value.cpp).

Every other value type (object, array, string, int, float, bool, none) already
registers {"tojson", tojson} in its get_builtins() table. value_undefined_t is
the only one that doesn't — a plain omission, not a serialization gap:
value_to_json_internal() already explicitly handles val->is_undefined() and
outputs "null", so the generic tojson() function works correctly for
undefined values once it's actually reachable as a filter name.

Symptom without this patch: llama-server crashes at startup for any chat
template that calls {{ something|tojson }} where `something` ends up
Undefined (e.g. {{ tools|tojson }} when no tools were supplied in the
request) — "Unknown (built-in) filter 'tojson' for type Undefined". Since the
template is parsed/exercised at model load, this is a hard load failure, not
a per-request error. Hit with an Olmo-3.1-32B GGUF (bartowski quant); likely
affects any tool-calling-template GGUF used without tools present.

See BUG-hybrid-arch-degeneration.md (2026-09-15 update) for the investigation
that found this. common/jinja/ is vendored, unmodified-from-upstream code
(no h3rb3rn-authored history) — this is a local carry of a fix worth
upstreaming too.

Usage (inside Docker builder, after cmake configure, before cmake build):
    python3 scripts/patch-llama-jinja-tojson.py <ollama-src-root>
"""

import sys
import subprocess
from pathlib import Path

PATCH_GUARD = "// [OLLAMA_JINJA_TOJSON_UNDEFINED_v1]"
SOURCE_FILE = "common/jinja/value.cpp"

TARGET = (
    "const func_builtins & value_undefined_t::get_builtins() const {\n"
    "    static const func_builtins builtins = {\n"
    '        {"default", default_value},\n'
)
REPLACEMENT = (
    "const func_builtins & value_undefined_t::get_builtins() const {\n"
    "    static const func_builtins builtins = {\n"
    f"        {PATCH_GUARD}\n"
    '        {"default", default_value},\n'
    '        {"tojson", tojson},\n'
)


def find_value_cpp(ollama_root: Path) -> Path | None:
    candidates = [
        ollama_root / "build" / "llama-server-cuda_v12" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "build" / "llama-server-cuda_v11" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "llama" / "llama.cpp" / SOURCE_FILE,
    ]
    for c in candidates:
        if c.is_file():
            return c
    result = subprocess.run(
        ["grep", "-r", "-l", "value_undefined_t::get_builtins", str(ollama_root), "--include=*.cpp"],
        capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.strip().splitlines():
        p = Path(line)
        if p.is_file():
            return p
    return None


def patch(path: Path) -> bool:
    content = path.read_text()

    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True

    if TARGET not in content:
        print(f"  Marker not found in {path} — layout may have changed upstream", file=sys.stderr)
        return False

    content = content.replace(TARGET, REPLACEMENT, 1)
    path.write_text(content)
    print(f"  tojson-for-undefined patch applied to {path.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)

    ollama_root = Path(sys.argv[1])
    print(f"Looking for llama.cpp {SOURCE_FILE} under {ollama_root}...")
    target = find_value_cpp(ollama_root)
    if not target:
        print(f"  {SOURCE_FILE} not found — skipping (cmake configure may not have run yet)")
        sys.exit(0)  # non-fatal

    print(f"  Target: {target}")
    if not patch(target):
        print("Patch failed — build continues with original behavior.", file=sys.stderr)

    # Always exit 0 (non-fatal)
    sys.exit(0)


if __name__ == "__main__":
    main()
