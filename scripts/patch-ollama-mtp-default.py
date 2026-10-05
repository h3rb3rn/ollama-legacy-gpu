#!/usr/bin/env python3
"""
patch-ollama-mtp-default.py — OLLAMA_DRAFT_NUM_PREDICT as a server-wide default AND upper bound for
draft_num_predict (server/routes.go).

Ollama enables native MTP speculative decoding (`--spec-type draft-mtp`) for models whose GGUF carries an
MTP head (e.g. qwen35moe, nextn_predict_layers=1) through the model's own manifest: the registry's
`qwen3.6:35b` ships `PARAMETER draft_num_predict 2`. Without a server override that value is used. On
Maxwell this crashes in common_speculative_impl_draft_mtp (illegal memory access, Xid 31;
BUG-hybrid-arch-degeneration.md, Finding 2). The proxy only protects containers that run it, and a
per-model Modelfile override has to be redone for every pulled model.

With this patch, when OLLAMA_DRAFT_NUM_PREDICT is set:
  * a request that sets draft_num_predict keeps its value (deliberate use stays possible);
  * a value from the model manifest is capped at OLLAMA_DRAFT_NUM_PREDICT (0 disables MTP);
  * with neither, OLLAMA_DRAFT_NUM_PREDICT is the default.
gpu-detect.sh exports 0, so a freshly pulled MTP model no longer needs a manual Modelfile override.
(An earlier version treated a manifest value like a request value, so the registry's own
`draft_num_predict 2` bypassed the protection.)
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
	''' + PATCH_GUARD + ''' OLLAMA_DRAFT_NUM_PREDICT: default for unset values, upper bound for model-manifest values.
	// A request that sets draft_num_predict always wins.
	if raw := strings.TrimSpace(os.Getenv("OLLAMA_DRAFT_NUM_PREDICT")); raw != "" && !hasOption(requestOpts, "draft_num_predict") {
		var n int
		if _, err := fmt.Sscanf(raw, "%d", &n); err == nil && n >= 0 {
			if model != nil && hasOption(model.Options, "draft_num_predict") {
				if opts.DraftNumPredict > n {
					opts.DraftNumPredict = n
				}
			} else {
				opts.DraftNumPredict = n
			}
		} else {
			slog.Warn("ignoring invalid OLLAMA_DRAFT_NUM_PREDICT", "value", raw)
		}
	}
'''

GO_TEST_FILE = "draft_default_test.go"
GO_TEST = '''package server

import "testing"

// [OLLAMA_DRAFT_NUM_PREDICT patch] tests
func TestDraftNumPredictServerDefault(t *testing.T) {
	s := &Server{}
	manifest := &Model{Options: map[string]any{"draft_num_predict": float64(2)}}
	plain := &Model{}

	t.Setenv("OLLAMA_DRAFT_NUM_PREDICT", "0")
	if opts, err := s.modelOptions(manifest, nil); err != nil || opts.DraftNumPredict != 0 {
		t.Fatalf("manifest value 2 must be capped to 0, got %d (err %v)", opts.DraftNumPredict, err)
	}
	if opts, err := s.modelOptions(manifest, map[string]any{"draft_num_predict": float64(3)}); err != nil || opts.DraftNumPredict != 3 {
		t.Fatalf("request value must win, got %d (err %v)", opts.DraftNumPredict, err)
	}
	if opts, err := s.modelOptions(plain, nil); err != nil || opts.DraftNumPredict != 0 {
		t.Fatalf("unset must follow the server default, got %d (err %v)", opts.DraftNumPredict, err)
	}

	t.Setenv("OLLAMA_DRAFT_NUM_PREDICT", "4")
	if opts, err := s.modelOptions(manifest, nil); err != nil || opts.DraftNumPredict != 2 {
		t.Fatalf("a manifest value below the cap must stay, got %d (err %v)", opts.DraftNumPredict, err)
	}

	t.Setenv("OLLAMA_DRAFT_NUM_PREDICT", "")
	if opts, err := s.modelOptions(manifest, nil); err != nil || opts.DraftNumPredict != 2 {
		t.Fatalf("without the variable the manifest value applies, got %d (err %v)", opts.DraftNumPredict, err)
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
    (path.parent / GO_TEST_FILE).write_text(GO_TEST)
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
