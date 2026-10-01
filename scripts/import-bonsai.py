#!/usr/bin/env python3
"""Import the Bonsai demo's GGUF into the BONSAI=ON Ollama fork.

OLLAMA_HOST selects the server as with the normal Ollama CLI. The GGUF is
uploaded unchanged; its embedded template and Prism rotation metadata remain
available to the native backend.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path, help="PQ2_0 or PTQ1_0 file downloaded by Bonsai-demo")
    parser.add_argument("--model", help="Ollama model name (default derived from the packing)")
    parser.add_argument("--context", type=int, default=190000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--ollama", default="ollama", help="Ollama CLI executable")
    parser.add_argument("--dry-run", action="store_true", help="Print the Modelfile without importing")
    args = parser.parse_args()
    path = args.gguf.resolve(strict=True)
    packing = next((q for q in ("PQ2_0", "PTQ1_0") if path.name.endswith(f"-{q}.gguf")), None)
    if not packing:
        parser.error("use the published Bonsai PQ2_0 or PTQ1_0 GGUF, retaining its original filename")
    if any(c in str(path) for c in "\r\n"):
        parser.error("newlines in model paths are unsupported")
    with path.open("rb") as file:
        if file.read(4) != b"GGUF":
            parser.error("file does not have a GGUF header")
    if not 1 <= args.context <= 262144 or args.batch < 1:
        parser.error("context must be 1..262144 and batch must be positive")
    model = args.model or f"bonsai2:27b-{packing.lower()}"
    modelfile = f"""FROM {json.dumps(str(path), ensure_ascii=False)}
PARAMETER num_ctx {args.context}
PARAMETER num_batch {args.batch}
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER min_p 0.05
PARAMETER repeat_penalty 1.0
"""
    if args.dry_run:
        print(modelfile, end="")
        return
    with tempfile.TemporaryDirectory(prefix="ollama-bonsai-import-") as tmp:
        config = Path(tmp) / "Modelfile"
        config.write_text(modelfile)
        subprocess.run([args.ollama, "create", model, "-f", str(config)], check=True)
    print(f"Imported {model}. Run: {args.ollama} run {model}")


if __name__ == "__main__":
    main()
