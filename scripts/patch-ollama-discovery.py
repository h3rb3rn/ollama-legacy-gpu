#!/usr/bin/env python3
"""
patch-ollama-discovery.py — hardens CUDA device validation in discover/llama_server.go.

Upstream looks up a device's compute capability with a positional counter over the
`--list-devices` output (`ccByIndex[deviceIndex]`), but the CC lines come from
llama-server's own "Device N:" numbering. On mixed pools (Tesla + GTX + RTX) the two
can disagree, and when no CC is found upstream only logs a warning and keeps the
device ("architecture filtering disabled for this device"). A GPU whose
architecture was not compiled into the image then passes discovery and crashes on the
first kernel launch ("no kernel image is available for execution on the device").

This patch:
  1. Resolves the CC by the CUDA index in the device name ("CUDA7") when present.
  2. Fails closed: if the compiled architectures are known but the device CC is not,
     the device is skipped with a clear message instead of being loaded blind.
     OLLAMA_ALLOW_UNKNOWN_CC=1 restores the old permissive behaviour.
"""

import sys
from pathlib import Path

PATCH_GUARD = "// [OLLAMA_DISCOVERY_CC patch]"
TARGET_FILE = "discover/llama_server.go"

LOOKUP_OLD = '''		if library == "CUDA" {
			cc, ok := ccByIndex[deviceIndex]
'''
LOOKUP_NEW = '''		if library == "CUDA" {
			''' + PATCH_GUARD + ''' resolve the CC by the CUDA index in the device name.
			cc, ok := ccByIndex[cudaDeviceIndex(name, deviceIndex)]
'''

UNKNOWN_OLD = '''			} else if !ok {
				slog.Warn("llama-server discovery: could not determine compute capability for CUDA device — "+
'''
UNKNOWN_NEW = '''			} else if !ok && len(cudaArchSet) > 0 && os.Getenv("OLLAMA_ALLOW_UNKNOWN_CC") != "1" {
				slog.Warn("llama-server discovery: skipping CUDA device — compute capability unknown, "+
					"cannot verify it against the compiled architectures. "+
					"Set OLLAMA_ALLOW_UNKNOWN_CC=1 to load it anyway.",
					"device", description, "archs", cudaArchs, "libDirs", libDirs)
				deviceIndex++
				continue
			} else if !ok {
				slog.Warn("llama-server discovery: could not determine compute capability for CUDA device — "+
'''

COMPUTE_OLD = "computeMajor, computeMinor := computeVersion(library, deviceIndex, gfxByIndex, ccByIndex)"
COMPUTE_NEW = '''computeIndex := deviceIndex
		if library == "CUDA" {
			computeIndex = cudaDeviceIndex(name, deviceIndex)
		}
		computeMajor, computeMinor := computeVersion(library, computeIndex, gfxByIndex, ccByIndex)'''

HELPER = '''
''' + PATCH_GUARD + ''' helper
var cudaNameIndexRegex = regexp.MustCompile(`^CUDA(\\d+)$`)

// cudaDeviceIndex returns the CUDA index encoded in a llama-server device name such
// as "CUDA3", or the positional fallback when the name carries none.
func cudaDeviceIndex(name string, fallback int) int {
	if m := cudaNameIndexRegex.FindStringSubmatch(strings.TrimSpace(name)); m != nil {
		if idx, err := strconv.Atoi(m[1]); err == nil {
			return idx
		}
	}
	return fallback
}
'''


TEST_FILE = "discover/llama_server_test.go"
TEST_OLD = '''				name: "CUDA without compute capability fails open",
				output: `system_info: n_threads = 4 | CUDA : ARCHS = 750,800 |
Available devices:
  CUDA0: Some Future GPU (8192 MiB, 8000 MiB free)
`,
				want: []wantDevice{{
					name:     "CUDA0",
					library:  "CUDA",
					totalMiB: 8192,
				}},
'''
TEST_NEW = '''				name: "CUDA without compute capability fails closed when archs are known",
				output: `system_info: n_threads = 4 | CUDA : ARCHS = 750,800 |
Available devices:
  CUDA0: Some Future GPU (8192 MiB, 8000 MiB free)
`,
				want: nil,
'''
NEW_TEST_FILE = "discover/llama_server_cc_index_test.go"
NEW_TEST = '''package discover

import "testing"

// [OLLAMA_DISCOVERY_CC patch] tests
func TestCudaDeviceIndex(t *testing.T) {
	for _, tc := range []struct {
		name     string
		fallback int
		want     int
	}{
		{"CUDA0", 5, 0},
		{"CUDA11", 0, 11},
		{" CUDA3 ", 9, 3},
		{"Vulkan0", 4, 4},
		{"CUDA_Host", 2, 2},
	} {
		if got := cudaDeviceIndex(tc.name, tc.fallback); got != tc.want {
			t.Errorf("cudaDeviceIndex(%q, %d) = %d, want %d", tc.name, tc.fallback, got, tc.want)
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
    for anchor in (LOOKUP_OLD, UNKNOWN_OLD, COMPUTE_OLD):
        if content.count(anchor) != 1:
            print(f"  ERROR: anchor not found exactly once in {path.name}:\n{anchor}", file=sys.stderr)
            return False
    content = content.replace(LOOKUP_OLD, LOOKUP_NEW, 1)
    content = content.replace(UNKNOWN_OLD, UNKNOWN_NEW, 1)
    content = content.replace(COMPUTE_OLD, COMPUTE_NEW, 1)
    path.write_text(content.rstrip("\n") + "\n" + HELPER)
    print(f"  Discovery CC patch applied to {path.name}")
    patch_tests(path.parent.parent)
    return True


def patch_tests(root: Path) -> None:
    """Align the upstream fail-open test with the new behaviour and add index tests."""
    test = root / TEST_FILE
    if test.is_file():
        text = test.read_text()
        if TEST_OLD in text:
            test.write_text(text.replace(TEST_OLD, TEST_NEW, 1))
            print(f"  Updated fail-open expectation in {test.name}")
        elif TEST_NEW not in text:
            print(f"  WARNING: fail-open test case not found in {test.name}", file=sys.stderr)
        (root / NEW_TEST_FILE).write_text(NEW_TEST)


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
