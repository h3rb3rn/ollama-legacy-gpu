#!/usr/bin/env python3
"""
patch-ollama-mtp-default.py — adds OLLAMA_DRAFT_NUM_PREDICT as a server-wide default
for draft_num_predict (server/routes.go).

Ollama enables native MTP speculative decoding (`--spec-type draft-mtp`) for every
model whose GGUF carries an MTP head (e.g. qwen35moe, nextn_predict_layers=1) because
api.DefaultOptions() sets DraftNumPredict=4. On Maxwell this crashes in
common_speculative_impl_draft_mtp (illegal memory access, Xid 31;
BUG-hybrid-arch-degeneration.md, Finding 2). The proxy only protects containers that
run it; pool-style containers (OLLAMA_AUTO_OPTIMIZE=0) were unprotected and had to be
patched per model.

With this patch the default is applied in the server itself, for every model and every
container: when neither the model's options nor the request set draft_num_predict, the
value of OLLAMA_DRAFT_NUM_PREDICT is used. gpu-detect.sh exports 0 whenever a legacy
(CC < 70) GPU is visible and the variable is not set manually. Explicit model/request
values still win, so deliberate MTP use stays possible.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [OLLAMA_DRAFT_NUM_PREDICT patch]"
TARGET_FILE = "server/routes.go"

ANCHOR = '''	if model != nil && model.DraftPath == "" && !draftNumPredictSet {
		opts.DraftNumPredict = 0
	}
'''
INSERT = '''
	''' + PATCH_GUARD + ''' server-wide default when neither model nor request set draft_num_predict.
	if !draftNumPredictSet {
		if raw := strings.TrimSpace(os.Getenv("OLLAMA_DRAFT_NUM_PREDICT")); raw != "" {
			var n int
			if _, err := fmt.Sscanf(raw, "%d", &n); err == nil && n >= 0 {
				opts.DraftNumPredict = n
			} else {
				slog.Warn("ignoring invalid OLLAMA_DRAFT_NUM_PREDICT", "value", raw)
			}
		}
	}
'''


def find_target(root: Path):
    for base in (root, root / "src"):
        if (base / TARGET_FILE).is_file():
            return base / TARGET_FILE
    return None


def patch(path: Path) -> bool:
    content = path.read_text()
    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True
    if content.count(ANCHOR) != 1:
        print(f"  ERROR: anchor not found exactly once in {path.name}", file=sys.stderr)
        return False
    path.write_text(content.replace(ANCHOR, ANCHOR + INSERT, 1))
    print(f"  MTP default patch applied to {path.name}")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        return 2
    target = find_target(Path(sys.argv[1]))
    if target is None:
        print(f"ERROR: {TARGET_FILE} not found under {sys.argv[1]}", file=sys.stderr)
        return 1
    return 0 if patch(target) else 1


if __name__ == "__main__":
    sys.exit(main())
