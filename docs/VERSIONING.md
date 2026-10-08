# Versioning and tags

The fork is built on top of an official Ollama release (the patch scripts are applied to the Ollama source at build time, see
[FORK-VS-STOCK.md](FORK-VS-STOCK.md)). **The tag is always the Ollama version the build is based on.**

## Git tags

`vX.Y.Z` marks the last commit of this repository that is based on Ollama `vX.Y.Z`. A new Ollama base gets a new tag; commits on the same
base (fixes, documentation) move on after the tag, so the tag stays at the state that was built and deployed.

| Tag | Commit | Base | Note |
| --- | --- | --- | --- |
| `v0.30.0` | `778fe91` | Ollama v0.30.10 | first release; the number is the fork's own release number |
| `v0.30.10` | `778fe91` | Ollama v0.30.10 | same commit as `v0.30.0`, named by the base |
| `v0.30.11` | `5e25fbc` | Ollama v0.30.11 | |
| `v0.32.15` | `aa9cd4a` | Ollama v0.32.15 | commit that records the version in `.last-built-version` |
| `v0.33.2` | `300485d` | Ollama v0.33.2 | same |
| `v0.33.3` | `4fa2d7c` | Ollama v0.33.3 | same |
| `v0.34.0` | `a6ce3cb` | Ollama v0.34.0 | same |
| `v0.34.1` | `672f7d3` | Ollama v0.34.1 | same; the Bonsai-capable build is pinned to this base |
| `v0.34.2` | `6c35cd1` | Ollama v0.34.2 | same |
| `v0.34.4` | `f8eb7f3` | Ollama v0.34.4 | same |
| `v0.35.1` | `26ef996` | Ollama v0.35.1, llama.cpp b11232 | last commit based on 0.35.1 (`BONSAI=OFF`); the images `pipefix-20261005` and `kolibri-20261006` were built from this line |
| `v0.40.0` | `738c694` | Ollama v0.40.0, llama.cpp b11351 | the GitHub Actions candidates of run 37688616665 carry this revision |

Not tagged: the bases v0.31.x and v0.32.0–v0.32.14, for which images exist in the registry but the history has no commit that identifies
them. The tags of the older bases point to the state at which the release workflow recorded the version, not to a verified hardware release.

## Image tags (`ghcr.io/h3rb3rn/ollama-legacy`)

| Tag | Meaning |
| --- | --- |
| `<variant>-<ollama version>`, e.g. `cuda12-maxwell-0.40.1` | newest build on that Ollama version (moves when the same base is rebuilt); set by every non-Bonsai build since run 37744806101 |
| `candidate-<run id>-<attempt>-<variant>-native` | immutable build of one workflow run; the digest is the reference |
| `<variant>-latest` | only set by the promotion job after hardware validation and rollout (disabled without a configured GPU runner inventory) |

`<variant>` is `cuda12-maxwell`, `cuda11-legacy` or `cuda13-rtx`. The workflow resolves the newest published Ollama release, so the version tag follows
that release, not the base that was validated on the hosts: run 37744806101 (revision `b22a75e`) built Ollama v0.40.1 (tag `cuda12-maxwell-0.40.1`), while
the digest deployed on N04-RTX is the v0.40.0 build of run 37688616665, which predates the version tags and is reachable only through its candidate tag
`candidate-37688616665-1-cuda12-maxwell-native`. There is no git tag `v0.40.1` yet; it follows when that build has been validated on hardware. All hosts now run the GitHub-built digest of run 37688616665. The local images used earlier
(`ollama-gaps:pipefix-20261005`, `ollama-gaps:kolibri-20261006`, `ollama-gaps:v040-20261007`) were built from the commits named above and remain on the hosts only for rollback.
