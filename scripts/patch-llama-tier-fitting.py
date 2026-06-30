#!/usr/bin/env python3
"""
patch-llama-tier-fitting.py — Patches common_params_fit_impl in llama.cpp/common/fit.cpp
to support GPU-tier-aware layer distribution.

The patch makes two modifications to fit.cpp:

  Patch 1 — "no changes needed" bypass:
    common_params_fit_impl returns early when all GPU memory targets are met.
    With OLLAMA_GPU_TIER_THRESHOLD > 0, we force the filling loop to run
    so legacy GPUs can be explicitly excluded (targets set to 0).

  Patch 2 — Two-pass filling loop:
    The existing back-to-front filling loop is wrapped in a two-pass structure:
      Pass 1: fill fast GPUs (CUDA index >= threshold)
      Pass 2: extend to legacy GPUs only if layers remain
    Additionally, legacy GPU targets are forced to 0 before the loop,
    so they only receive layers when fast pool is exhausted.

Result: models that fit in the fast pool (RTX/GTX) never touch Tesla GPUs.
         Large models (> fast pool) automatically extend to Tesla.

OLLAMA_GPU_TIER_THRESHOLD env var (in .env.multigpu):
    0 (default) = disabled, original behavior
    N           = CUDA devices 0..N-1 are legacy, N..nd-1 are fast

Usage (inside Docker builder, after cmake configure, before cmake build):
    python3 scripts/patch-llama-tier-fitting.py <ollama-src-root>
"""

import sys
import re
import subprocess
from pathlib import Path

PATCH_GUARD    = "OLLAMA_GPU_TIER_THRESHOLD_v2"
LOOP_MARKER    = "for (int id = nd - 1; id >= 0; id--) {"
DENSE_LOG        = 'filling dense layers back-to-front:'
NO_CHANGES_STR   = 'targets for free memory can be met on all devices, no changes needed'
TARGETS_STR      = 'targets.push_back(dmds_full[id].free - margins[id]);'
FORCE_LAYERS_VAR = 'OLLAMA_FORCE_GPU_LAYERS'
SOURCE_FILE      = "common/fit.cpp"

# Patch 0: OLLAMA_FORCE_GPU_LAYERS — bypasses the entire fitting algorithm
# and forces n_gpu_layers = hp_ngl+1 (all) with equal tensor_split.
# Used by Dynamic Pool when model file size fits in the fast GPU pool.
# This is safe because selectGPUPool() already verified model_size <= 75% of pool.
FORCE_LAYERS_CODE = '''
    // [OLLAMA_FORCE_GPU_LAYERS patch] Greedy sequential GPU filling.
    // Bypasses the conservative fitting algorithm. Fills best GPUs (highest CUDA
    // index = highest bandwidth) first until all model layers are placed.
    // Uses minimum number of GPUs needed — fewer GPUs = shorter pipeline = faster.
    // Caller (selectGPUPool) verified model_size <= 75% of fast pool VRAM.
    {{
        const char * _force = std::getenv("{var}");
        if (_force && _force[0] && std::atoi(_force) > 0) {{
            // Compute model bytes per layer from actual dmds_full measurement
            int64_t _sum_model_bytes = 0;
            for (size_t _id = 0; _id < nd; _id++) {{
                _sum_model_bytes += dmds_full[_id].mb.model;
            }}
            // Overhead scale: accounts for compute buffers + KV-cache + alignment overhead.
            //
            // Adaptive mode (default): overhead is ~fixed per GPU regardless of quantization.
            //   With FA, compute buffers ≈ 1.5 GB/GPU for 131k context.
            //   KV-cache (q4_0, 131k, 10 attn layers) ≈ 0.7 GB total.
            //   Fixed overhead per layer ≈ (1.5 × nd + 0.7) GB / hp_ngl
            //
            // Example: Q4_K_M (542 MB/layer): scale ≈ 1 + 357/542 = 1.66... but capped at 1.4
            //          Q8_0  (1047 MB/layer): scale ≈ 1 + 357/1047 = 1.34 → fewer GPUs needed
            //          FP16  (2095 MB/layer): scale ≈ 1 + 357/2095 = 1.17 → even fewer GPUs
            //
            // OLLAMA_LAYER_OVERHEAD_SCALE overrides the adaptive calculation.
            // OLLAMA_OVERHEAD_PER_GPU_MB sets fixed overhead per GPU in MB (default 1536).
            float _overhead_mb_per_gpu = 1536.0f; // 1.5 GB compute + KV fraction
            if (const char * _o = std::getenv("OLLAMA_OVERHEAD_PER_GPU_MB")) {{
                float _v = std::stof(_o);
                if (_v > 0 && _v < 10240) _overhead_mb_per_gpu = _v;
            }}
            // Adaptive scale: total overhead / total layer weight
            float _total_overhead_mb = _overhead_mb_per_gpu * nd;
            float _total_model_mb    = (float)_sum_model_bytes / (1024.0f * 1024.0f);
            float _adaptive_scale    = 1.0f + (_total_overhead_mb / _total_model_mb);
            // Cap adaptive scale: not too aggressive (min 1.05) or too conservative (max 2.0)
            _adaptive_scale = std::max(1.05f, std::min(2.0f, _adaptive_scale));

            float _ovhd = _adaptive_scale; // default: adaptive
            if (const char * _s = std::getenv("OLLAMA_LAYER_OVERHEAD_SCALE")) {{
                float _v = std::stof(_s);
                if (_v > 1.0f && _v < 5.0f) _ovhd = _v; // explicit override wins
            }}
            float _bytes_per_layer = (_sum_model_bytes > 0 && hp_ngl > 0) ?
                ((float)_sum_model_bytes / hp_ngl) * _ovhd :
                600.0f * 1024 * 1024 * _ovhd; // 600 MB fallback

            // Initialize all tensor_split to 0 so GPUs not reached by greedy loop
            // get 0 layers (not stale values from earlier in the function).
            if (tensor_split) {{
                for (size_t _id = 0; _id < nd; _id++) tensor_split[_id] = 0.0f;
            }}
            // Bandwidth-weighted budgets: OLLAMA_GPU_BANDWIDTHS=83,83,...,360,360
            // Parsed from gpu-detect.sh (worst→best order matches CUDA_VISIBLE_DEVICES).
            // effective_budget[i] = raw_budget[i] * (max_bw / bw[i])
            // Slow GPUs (M10: 83 GB/s) get 4.3× lower effective budget than their VRAM
            // allows — they contribute fewer layers, avoiding pipeline bottlenecks.
            float _bw_factors[64]; // max 64 GPUs
            for (size_t _id = 0; _id < nd && _id < 64; _id++) _bw_factors[_id] = 1.0f;
            if (const char * _bws_env = std::getenv("OLLAMA_GPU_BANDWIDTHS")) {{
                const char * _max_env = std::getenv("OLLAMA_GPU_MAX_BANDWIDTH");
                float _max_bw = _max_env ? std::stof(_max_env) : 360.0f;
                std::string _bws_str(_bws_env);
                size_t _bi = 0, _pos = 0;
                while (_bi < nd && _bi < 64) {{
                    size_t _comma = _bws_str.find(',', _pos);
                    float _bw = std::stof(_bws_str.substr(_pos, _comma == std::string::npos ? std::string::npos : _comma - _pos));
                    _bw_factors[_bi++] = (_bw > 0) ? (_max_bw / _bw) : 1.0f;
                    if (_comma == std::string::npos) break;
                    _pos = _comma + 1;
                }}
            }}
            // Greedy sequential fill: best GPU (highest CUDA index) first.
            // Budget scaled by bandwidth factor: slow GPUs (M10) get proportionally
            // fewer layers to prevent pipeline bottlenecks in mixed-GPU inference.
            uint32_t _layers_left  = hp_ngl + 1;
            uint32_t _total_layers = 0;
            for (int _id = (int)nd - 1; _id >= 0 && _layers_left > 0; _id--) {{
                int64_t _raw_budget = dmds_full[_id].free - (int64_t)margins_s[_id];
                // Apply bandwidth scaling: divide by factor (factor>1 for slow GPUs)
                int64_t _budget = (_raw_budget > 0) ?
                    (int64_t)((float)_raw_budget / _bw_factors[_id]) : 0;
                uint32_t _n = (_budget > 0) ?
                    std::min(_layers_left, (uint32_t)((float)_budget / _bytes_per_layer)) : 0;
                if (tensor_split) tensor_split[_id] = (float)_n;
                _layers_left  -= _n;
                _total_layers += _n;
            }}
            // Partial-fill fallback: keep the greedy-placed layers and distribute only
            // the overflow by raw VRAM (no bandwidth weighting). This preserves the
            // bandwidth-priority ordering so RTX cards keep their greedy share and only
            // Tesla/GTX absorb the small overflow.
            //
            // Previous full-reset VRAM-weighted fallback discarded all greedy placement:
            //   RTX 3060 had 9 layers from greedy → dropped to ~6.5 GiB (leaving 5.5 GiB idle)
            // Now: RTX 3060 keeps 9 layers + tiny raw-VRAM overflow share → ~10 GB used
            if (_total_layers < (uint32_t)(hp_ngl + 1)) {{
                uint32_t _overflow = (uint32_t)(hp_ngl + 1) - _total_layers;
                // Distribute overflow proportional to REMAINING bandwidth-weighted budget.
                // After greedy, fast GPUs (RTX) have exhausted their budget; slow GPUs
                // (Tesla/GTX) still have budget left. This sends overflow to slow GPUs
                // that have headroom, preventing RTX from being packed beyond safe KV limits.
                float _total_remaining = 0.0f;
                for (size_t _id = 0; _id < nd; _id++) {{
                    float _raw = (float)(dmds_full[_id].free) - (float)margins_s[_id];
                    float _bw_budget = (_raw > 0) ? (_raw / _bw_factors[_id]) : 0;
                    float _used = tensor_split ? (tensor_split[_id] * _bytes_per_layer) : 0;
                    float _remaining = _bw_budget - _used;
                    if (_remaining > 0) _total_remaining += _remaining;
                }}
                LOG_WRN("%s: greedy fill placed only %d/%d layers (overhead=%.2fx too tight); "
                        "distributing %d overflow layer(s) by remaining BW budget across %zu GPU(s)\\n",
                        __func__, _total_layers, hp_ngl + 1, _ovhd, _overflow, nd);
                if (nd > 1 && tensor_split && _total_remaining > 0) {{
                    for (size_t _id = 0; _id < nd; _id++) {{
                        float _raw = (float)(dmds_full[_id].free) - (float)margins_s[_id];
                        float _bw_budget = (_raw > 0) ? (_raw / _bw_factors[_id]) : 0;
                        float _used = tensor_split ? (tensor_split[_id] * _bytes_per_layer) : 0;
                        float _remaining = _bw_budget - _used;
                        if (_remaining > 0) tensor_split[_id] += (_remaining / _total_remaining) * (float)_overflow;
                    }}
                    mparams->tensor_split = tensor_split;
                }}
                mparams->n_gpu_layers = hp_ngl + 1;
                tensor_buft_overrides[0] = {{nullptr, nullptr}};
                mparams->tensor_buft_overrides = tensor_buft_overrides;
                return;
            }}
            // Count GPUs actually used (greedy: only highest-bandwidth GPUs)
            size_t _gpus_used = 0;
            if (tensor_split) {{
                for (size_t _id = 0; _id < nd; _id++) if (tensor_split[_id] > 0) _gpus_used++;
            }}
            LOG_INF("%s: {var}=%s → greedy fill: %d/%d layers on %zu/%zu GPU(s), "
                    "%.0f MB/layer (overhead=%.1fx)\\n",
                    __func__, _force, _total_layers, hp_ngl + 1, _gpus_used, nd,
                    _bytes_per_layer / 1024 / 1024, _ovhd);
            mparams->n_gpu_layers = _total_layers;
            if (nd > 1 && tensor_split) mparams->tensor_split = tensor_split;
            if (tensor_buft_overrides) {{
                tensor_buft_overrides[0].pattern = nullptr;
                tensor_buft_overrides[0].buft    = nullptr;
                mparams->tensor_buft_overrides    = tensor_buft_overrides;
            }}
            return;
        }}
    }}
'''.format(var=FORCE_LAYERS_VAR)


def find_fit_cpp(ollama_root: Path) -> Path | None:
    candidates = [
        ollama_root / "build" / "llama-server-cuda_v12" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "build" / "llama-server-cuda_v11" / "_deps" / "llama_cpp-src" / SOURCE_FILE,
        ollama_root / "llama" / "llama.cpp" / SOURCE_FILE,
    ]
    for c in candidates:
        if c.is_file():
            return c
    result = subprocess.run(
        ["grep", "-r", "-l", DENSE_LOG, str(ollama_root), "--include=*.cpp"],
        capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.strip().splitlines():
        p = Path(line)
        if p.is_file():
            return p
    return None


def apply_patch1_bypass_no_changes(content: str, indent: str) -> tuple[str, bool]:
    """
    Patch 1: prevent 'no changes needed' early return when tier_threshold > 0.
    Forces the filling loop to run so targets can be zeroed for legacy GPUs.
    Uses regex to handle any indentation level.
    """
    import re as _re
    # Match the LOG_TRC line followed by any-whitespace return;
    pattern = (
        r'(LOG_TRC\("%s: ' + _re.escape(NO_CHANGES_STR) + r'\\n", __func__\);)'
        r'(\s*\n\s*)(return;)'
    )
    m = _re.search(pattern, content)
    if not m:
        return content, False
    old = m.group(0)
    ws_before_return = m.group(2)  # newline + indentation of 'return;'
    ret_indent = ws_before_return.lstrip('\n')  # just the spaces

    new = (
        f'// GPU-tier patch 1: if tier_threshold is set and legacy GPUs have usage,\n'
        f'{ret_indent}// skip the early return and let the filling loop redistribute.\n'
        f'{ret_indent}{{\n'
        f'{ret_indent}    const int _t1 = []() -> int {{ const char * t = std::getenv("{PATCH_GUARD}"); return (t && t[0]) ? std::atoi(t) : 0; }}();\n'
        f'{ret_indent}    bool _legacy_used = false;\n'
        f'{ret_indent}    if (_t1 > 0) {{\n'
        f'{ret_indent}        for (size_t _id = 0; _id < (size_t)_t1 && _id < nd; _id++) {{\n'
        f'{ret_indent}            if (dmds_full[_id].mb.total() > 0) {{ _legacy_used = true; break; }}\n'
        f'{ret_indent}        }}\n'
        f'{ret_indent}    }}\n'
        f'{ret_indent}    if (!_legacy_used) {{\n'
        f'{ret_indent}        LOG_TRC("%s: {NO_CHANGES_STR}\\n", __func__);\n'
        f'{ret_indent}        return;\n'
        f'{ret_indent}    }}\n'
        f'{ret_indent}    LOG_TRC("%s: GPU-tier: bypassing early return (threshold=%d)\\n", __func__, _t1);\n'
        f'{ret_indent}}}\n'
    )
    return content.replace(old, new, 1), True


def apply_patch2_targets_and_loop(content: str, indent: str, lines: list) -> tuple[str, bool]:
    """
    Patch 2: zero legacy GPU targets + two-pass filling loop.
    """
    # A: Zero targets for legacy GPUs after targets[] is populated
    old_targets = TARGETS_STR
    if old_targets not in content:
        return content, False

    # Find the targets loop: 'for (size_t id = 0; id < nd; id++) { targets.push_back(...)'
    new_targets_append = (
        f'{old_targets}\n'
        f'{indent}    }}\n'
        f'{indent}    // GPU-tier patch 2a: zero targets for legacy GPUs so they get 0 layers\n'
        f'{indent}    // when fast pool is sufficient. Restored to natural values via overflow.\n'
        f'{indent}    {{\n'
        f'{indent}        const int _t2 = []() -> int {{ const char * t = std::getenv("{PATCH_GUARD}"); return (t && t[0]) ? std::atoi(t) : 0; }}();\n'
        f'{indent}        if (_t2 > 0) {{\n'
        f'{indent}            for (size_t _id = 0; _id < (size_t)_t2 && _id < nd; _id++) {{\n'
        f'{indent}                targets[_id] = 0;  // legacy GPU excluded from primary fill\n'
        f'{indent}            }}\n'
        f'{indent}            LOG_TRC("%s: GPU-tier: zeroed targets for %d legacy GPUs (CUDA0..%d)\\n",\n'
        f'{indent}                    __func__, _t2, _t2 - 1);\n'
        f'{indent}        }}\n'
        f'{indent}    }}\n'
        f'{indent}    for (size_t id = 0; id < nd; id++) {{'
    )
    # We need to replace just the targets.push_back line inside its for loop.
    # Find the for loop that contains targets.push_back and restructure.
    # Pattern: "for (size_t id = 0; id < nd; id++) {" followed by "targets.push_back"
    TARGETS_LOOP = 'for (size_t id = 0; id < nd; id++) {'

    # Find position of targets loop before targets.push_back
    pos_push = content.find(old_targets)
    pos_loop_start = content.rfind(TARGETS_LOOP, 0, pos_push)
    if pos_loop_start == -1:
        return content, False

    pos_loop_end = content.find('}', pos_push)
    if pos_loop_end == -1:
        return content, False

    # Extract the full loop
    loop_block = content[pos_loop_start : pos_loop_end + 1]

    # Build replacement: keep the loop content, add the tier patch after it
    tier_zero_block = (
        f'\n{indent}    // GPU-tier patch 2a: zero targets for legacy GPUs\n'
        f'{indent}    {{\n'
        f'{indent}        const int _t2 = []() -> int {{ const char * t = std::getenv("{PATCH_GUARD}"); return (t && t[0]) ? std::atoi(t) : 0; }}();\n'
        f'{indent}        if (_t2 > 0) {{\n'
        f'{indent}            for (size_t _id = 0; _id < (size_t)_t2 && _id < nd; _id++) {{\n'
        f'{indent}                targets[_id] = 0;  // legacy GPU: excluded from primary filling\n'
        f'{indent}            }}\n'
        f'{indent}            LOG_TRC("%s: GPU-tier: zeroed targets for %d legacy CUDA devices\\n", __func__, _t2);\n'
        f'{indent}        }}\n'
        f'{indent}    }}'
    )
    new_loop_block = loop_block + tier_zero_block
    content = content.replace(loop_block, new_loop_block, 1)

    # B: Two-pass filling loop
    # Find the back-to-front loop
    loop_marker = LOOP_MARKER
    if loop_marker not in content:
        return content, False

    lines2 = content.splitlines(keepends=True)
    loop_start = None
    dense_line = next((i for i, l in enumerate(lines2) if DENSE_LOG in l), None)
    if dense_line is None:
        return content, False
    for i in range(dense_line, min(dense_line + 30, len(lines2))):
        if loop_marker in lines2[i]:
            loop_start = i
            break
    if loop_start is None:
        return content, False

    depth, loop_end = 0, None
    for i in range(loop_start, len(lines2)):
        depth += lines2[i].count('{') - lines2[i].count('}')
        if depth == 0 and i > loop_start:
            loop_end = i
            break
    if loop_end is None:
        return content, False

    print(f"  Filling loop: lines {loop_start+1}–{loop_end+1}")

    body_lines = lines2[loop_start + 1 : loop_end]
    body_text = "".join(body_lines)

    patch_code = (
        f"{indent}// GPU-tier patch 2b: two-pass filling (fast GPUs first, legacy only if needed)\n"
        f"{indent}// CUDA_VISIBLE_DEVICES may restrict visible GPUs to fast-only (Dynamic Pool).\n"
        f"{indent}// Clamp threshold to nd so Pass 1 always runs even when threshold >= nd.\n"
        f"{indent}{{\n"
        f"{indent}    const int _tf_raw = []() -> int {{ const char * t = std::getenv(\"{PATCH_GUARD}\"); return (t && t[0]) ? std::atoi(t) : 0; }}();\n"
        f"{indent}    // Clamp: if all visible GPUs are fast (threshold >= nd), treat as threshold=0\n"
        f"{indent}    const int _tf = (_tf_raw > 0 && _tf_raw < (int)nd) ? _tf_raw : 0;\n"
        f"{indent}    auto _fill = [&](int _from, int _to) {{\n"
        f"{indent}        for (int id = _from; id >= _to; id--) {{\n"
        f"{body_text}"
        f"{indent}        }}\n"
        f"{indent}    }};\n"
        f"{indent}    _fill((int)nd - 1, _tf);  // Pass 1: fast GPUs (CUDA >= threshold, or all if no legacy)\n"
        f"{indent}    uint32_t _done_tf = 0;\n"
        f"{indent}    if (_tf > 0) {{\n"
        f"{indent}        for (const auto& _n : ngl_per_device) _done_tf += _n.n_layer;\n"
        f"{indent}        if (_done_tf < hp_ngl + 1) {{\n"
        f"{indent}            LOG_TRC(\"%s: GPU-tier: fast pool full, extending to legacy GPUs (remaining=%u)\\n\",\n"
        f"{indent}                    __func__, hp_ngl + 1 - _done_tf);\n"
        f"{indent}            // Restore legacy GPU targets to allow overflow\n"
        f"{indent}            for (size_t _id = 0; _id < (size_t)_tf && _id < nd; _id++) {{\n"
        f"{indent}                targets[_id] = dmds_full[_id].free - margins[_id];\n"
        f"{indent}            }}\n"
        f"{indent}            _fill(_tf - 1, 0);  // Pass 2: legacy GPUs\n"
        f"{indent}        }} else {{\n"
        f"{indent}            LOG_TRC(\"%s: GPU-tier: all layers fit in fast pool — legacy GPUs idle\\n\", __func__);\n"
        f"{indent}        }}\n"
        f"{indent}    }}\n"
        f"{indent}    // Patch 3: if all layers placed in fast pool AND MoE model, skip Step 4.\n"
        f"{indent}    // Step 4 (MoE dense-to-full front-to-back) causes UINT32 underflow when\n"
        f"{indent}    // Step 3 concentrated all layers on fast GPUs and CUDA11 drops to 0.\n"
        f"{indent}    // Skipping Step 4 keeps all layers on fast GPUs without corruption.\n"
        f"{indent}    if (_tf > 0 && hp_nex > 0) {{\n"
        f"{indent}        uint32_t _all = 0;\n"
        f"{indent}        for (const auto& _n : ngl_per_device) _all += _n.n_layer;\n"
        f"{indent}        if (_all >= hp_ngl + 1) {{\n"
        f"{indent}            LOG_TRC(\"%s: GPU-tier: skipping Step 4 MoE conversion \"\n"
        f"{indent}                    \"(all %u layers in fast pool, threshold=%d)\\n\",\n"
        f"{indent}                    __func__, _all, _tf);\n"
        f"{indent}            set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);\n"
        f"{indent}            return;\n"
        f"{indent}        }}\n"
        f"{indent}    }}\n"
        f"{indent}}} // end GPU-tier two-pass filling\n"
    )

    original_block = "".join(lines2[loop_start : loop_end + 1])
    content = content.replace(original_block, patch_code, 1)
    return content, True


def apply_patch0_force_layers(content: str) -> tuple[str, bool]:
    """
    Patch 0: OLLAMA_FORCE_GPU_LAYERS early return.
    Inserted near the beginning of common_params_fit_impl, after nd is computed.
    """
    # Find where nd (number of devices) is computed and used first
    # Anchor: the line that first uses nd in a meaningful way
    anchor = 'const size_t nd = devs.size();'
    if anchor not in content:
        # Try alternative
        anchor2 = 'const size_t nd'
        if anchor2 not in content:
            return content, False
        idx = content.find(anchor2)
    else:
        idx = content.find(anchor)

    # Find end of that line
    end_of_line = content.find('\n', idx)
    if end_of_line == -1:
        return content, False

    # Insert Patch 0 after that line
    new_content = content[:end_of_line + 1] + FORCE_LAYERS_CODE + content[end_of_line + 1:]
    if new_content == content:
        return content, False

    return new_content, True


def patch(path: Path) -> bool:
    content = path.read_text()

    if PATCH_GUARD in content:
        print(f"  Already patched: {path}")
        return True

    lines = content.splitlines(keepends=True)
    loop_start = next((i for i, l in enumerate(lines) if LOOP_MARKER in l), None)
    if loop_start is None or DENSE_LOG not in content:
        print(f"  Markers not found in {path}", file=sys.stderr)
        return False

    # Determine indent from the filling loop line
    indent = re.match(r'^(\s*)', lines[loop_start]).group(1) if loop_start else '    '

    # Apply Patch 0: OLLAMA_FORCE_GPU_LAYERS early return
    content, ok0 = apply_patch0_force_layers(content)
    if not ok0:
        print(f"  Patch 0 (FORCE_GPU_LAYERS) failed", file=sys.stderr)
        return False
    print(f"  Patch 0 applied: OLLAMA_FORCE_GPU_LAYERS bypass")

    # Re-read lines after Patch 0 changed content
    lines = content.splitlines(keepends=True)
    loop_start = next((i for i, l in enumerate(lines) if LOOP_MARKER in l), None)
    if loop_start is None:
        print(f"  Loop marker lost after Patch 0", file=sys.stderr)
        return False

    # Apply Patch 1: bypass 'no changes needed' when tier_threshold > 0
    content, ok1 = apply_patch1_bypass_no_changes(content, indent)
    if not ok1:
        print(f"  Patch 1 (no-changes bypass) failed — string not found", file=sys.stderr)
        return False
    print(f"  Patch 1 applied: 'no changes needed' bypass")

    # Apply Patch 2: zero legacy targets + two-pass filling
    lines2 = content.splitlines(keepends=True)
    content, ok2 = apply_patch2_targets_and_loop(content, indent, lines2)
    if not ok2:
        print(f"  Patch 2 (targets + filling loop) failed", file=sys.stderr)
        return False
    print(f"  Patch 2 applied: legacy GPU targets zeroed + two-pass filling")

    path.write_text(content)
    print(f"  Full GPU-tier-fitting patch applied to {path.name}")
    return True


def patch_ggml_cuda(path: Path) -> bool:
    if not path.is_file():
        return False
    print(f"  Found ggml-cuda.cu: {path}")
    content = path.read_text()
    
    if "// [OLLAMA_P2P_FIX]" in content:
        print("  ggml-cuda.cu already patched (P2P fix present)")
        return True
        
    set_target = '        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));'
    set_replacement = '        // [OLLAMA_P2P_FIX]\n        ggml_cuda_set_device(id);\n        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], buf_host, original_size, cudaMemcpyHostToDevice, cudaStreamPerThread));'
    
    get_target = '        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));'
    get_replacement = '        // [OLLAMA_P2P_FIX]\n        ggml_cuda_set_device(id);\n        CUDA_CHECK(cudaMemcpyAsync(buf_host, extra->data_device[id], original_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));'
    
    sync_target = '    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {\n        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));'
    sync_replacement = '    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {\n        ggml_cuda_set_device(id);\n        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));'

    if set_target not in content or get_target not in content:
        print("  Error: split buffer target strings not found in ggml-cuda.cu", file=sys.stderr)
        return False
        
    content = content.replace(set_target, set_replacement, 1)
    content = content.replace(get_target, get_replacement, 1)
    content = content.replace(sync_target, sync_replacement, 2)
    
    path.write_text(content)
    print("  ggml-cuda.cu P2P set_device patch successfully applied!")
    return True


def find_fit_cpp_and_patch(ollama_root: Path) -> bool:
    print(f"Looking for llama.cpp {SOURCE_FILE} under {ollama_root}...")
    target = find_fit_cpp(ollama_root)
    if not target:
        print(f"  {SOURCE_FILE} not found — skipping (cmake configure may not have run yet)")
        return True  # non-fatal
    print(f"  Target: {target}")
    ok = patch(target)
    if not ok:
        return False
        
    # Also patch ggml-cuda.cu if it exists in the build dir
    ggml_cuda_path = ollama_root / "build" / "llama-server-cuda_v12" / "_deps" / "llama_cpp-src" / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    if ggml_cuda_path.is_file():
        patch_ggml_cuda(ggml_cuda_path)
    else:
        # Also try other build directory (v11) if it exists
        ggml_cuda_path_v11 = ollama_root / "build" / "llama-server-cuda_v11" / "_deps" / "llama_cpp-src" / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
        if ggml_cuda_path_v11.is_file():
            patch_ggml_cuda(ggml_cuda_path_v11)
            
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ollama-source-root>", file=sys.stderr)
        sys.exit(1)
    if not find_fit_cpp_and_patch(Path(sys.argv[1])):
        print("Patch failed — build continues with original behavior.", file=sys.stderr)
    # Always exit 0 (non-fatal)
    sys.exit(0)


if __name__ == "__main__":
    main()

