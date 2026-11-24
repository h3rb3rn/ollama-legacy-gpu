import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompatibilityGuards(unittest.TestCase):
    def test_missing_upstream_files_fail_the_build(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('patch-ollama-fa', 'patch-ollama-dynamic-pool', 'patch-ollama-batch', 'patch-ollama-discovery', 'patch-ollama-mtp-default', 'patch-llama-cuda-graphs-legacy', 'patch-llama-fit-nextn', 'patch-llama-pipeline-parallel',
                         'patch-llama-input-gpu', 'patch-llama-tier-fitting', 'patch-llama-jinja-tojson', 'patch-llama-nccl-config', 'patch-llama-jev-decision'):
                with self.subTest(patch=name):
                    result = subprocess.run([sys.executable, str(SCRIPTS / (name + '.py')), root],
                                            capture_output=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)

    def test_reapplying_pool_patch_does_not_duplicate_go_functions(self):
        module = load('patch-ollama-dynamic-pool')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'source.go'
            path.write_text('params = appendFlashAttentionArgs(params, launch.gpus)\n'
                            '// LlamaServerFlashAttention\nfunc LlamaServerFlashAttention() {}\n')
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)

    def test_removed_split_buffer_requires_both_device_selections(self):
        module = load('patch-llama-tier-fitting')
        content = ''.join('static void ggml_backend_cuda_buffer_' + op + '_tensor(void) {\n'
                          '    ggml_cuda_set_device(ctx->device);\n'
                          '    CUDA_CHECK(cudaMemcpyAsync(data));\n}\n'
                          for op in ('set', 'get'))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ggml-cuda.cu'
            path.write_text(content)
            self.assertTrue(module.patch_ggml_cuda(path))
            path.write_text(content.replace('ggml_cuda_set_device(ctx->device);', '', 1))
            self.assertFalse(module.patch_ggml_cuda(path))

    def test_discovery_patch_is_idempotent_and_fails_closed(self):
        module = load('patch-ollama-discovery')
        src = (module.LOOKUP_OLD + '\n' + module.UNKNOWN_OLD + '\n' + module.COMPUTE_OLD + '\n')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'llama_server.go'
            path.write_text(src)
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertIn('OLLAMA_ALLOW_UNKNOWN_CC', first)
            self.assertIn('cudaDeviceIndex(name, deviceIndex)', first)
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            path.write_text('unrelated go source\n')
            self.assertFalse(module.patch(path))

    def test_mtp_default_patch_is_idempotent_and_respects_explicit_options(self):
        module = load('patch-ollama-mtp-default')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'routes.go'
            path.write_text(module.ANCHOR)
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertIn('!hasOption(requestOpts, "draft_num_predict")', first)
            self.assertIn('opts.DraftNumPredict > n', first)
            self.assertTrue((Path(root) / 'draft_default_test.go').exists())
            self.assertIn('OLLAMA_DRAFT_NUM_PREDICT', first)
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            path.write_text('unrelated\n')
            self.assertFalse(module.patch(path))

    def test_cuda_graph_patch_is_opt_in_and_idempotent(self):
        module = load('patch-llama-cuda-graphs-legacy')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ggml-cuda.cu'
            path.write_text(module.FUNC + '    if (graph->graph == nullptr) {\n' + module.OLD + '    }\n}\n')
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertIn('getenv("GGML_CUDA_GRAPHS_LEGACY")', first)
            self.assertIn('&& !ggml_cuda_legacy_graphs_enabled()', first)
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            path.write_text('unrelated\n')
            self.assertFalse(module.patch(path))

    def test_fit_nextn_patch_counts_slot_unconditionally_and_is_idempotent(self):
        module = load('patch-llama-fit-nextn')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'fit.cpp'
            path.write_text(module.OLD)
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertNotIn('load_mtp', first)
            self.assertIn('hp_ngl        += llama_model_n_layer_nextn(model);', first)
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            path.write_text('unrelated\n')
            self.assertFalse(module.patch(path))

    def test_input_gpu_patch_follows_layer_zero_keeps_opt_out_and_is_idempotent(self):
        module = load('patch-llama-input-gpu')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'llama-model.cpp'
            path.write_text(module.INCLUDE_OLD + 'int x;\n' + module.OLD)
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertIn('#include <cstdlib>', first)
            self.assertIn('pimpl->dev_input = get_layer_buft_list(0);', first)
            self.assertIn('LLAMA_INPUT_LAYER_GPU', first)
            # the upstream CPU placement stays reachable as the opt-out branch
            self.assertIn('pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };', first)
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            # fails closed when upstream moves the block or the include
            path.write_text(module.INCLUDE_OLD + 'unrelated\n')
            self.assertFalse(module.patch(path))
            path.write_text('unrelated\n' + module.OLD)
            self.assertFalse(module.patch(path))

    def test_ctx_halving_only_without_flash_attention(self):
        module = load('patch-ollama-dynamic-pool')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'source.go'
            path.write_text('params = appendFlashAttentionArgs(params, launch.gpus)\n'
                            '// LlamaServerFlashAttention\nfunc LlamaServerFlashAttention() {}\n')
            self.assertTrue(module.patch(path))
            code = path.read_text()
        cond = code.index('LlamaServerFlashAttention(launch.gpus) == ml.FlashAttentionDisabled')
        halving = code.index('halving -c to match num_parallel=1')
        cap = code.index('OLLAMA_MAX_CTX_FULL_POOL')
        # -c/-np halving sits inside the "FA explicitly disabled" condition; the explicit
        # OLLAMA_MAX_CTX_FULL_POOL cap stays outside of it.
        self.assertLess(code.index('OLLAMA_CTX_HALVING_FA_GUARD'), cond)
        self.assertLess(cond, halving)
        self.assertLess(halving, cap)
        self.assertEqual(code.count('{'), code.count('}'))

    def test_pipeline_parallel_opt_out_is_idempotent_and_default_unchanged(self):
        module = load('patch-llama-pipeline-parallel')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'llama-context.cpp'
            path.write_text(module.OLD + '            model.n_gpu_layers() > model.hparams.n_layer_all;\n')
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertIn('getenv("LLAMA_PIPELINE_PARALLEL")', first)
            # without the variable the original condition still applies: the new term only adds an AND
            self.assertIn('llama_pp_allowed &&\n            model.n_devices() > 1 &&', first)
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            path.write_text('unrelated\n')
            self.assertFalse(module.patch(path))

    def test_vmm_peer_access_limit_is_idempotent_and_keeps_small_setups(self):
        module = load('patch-llama-vmm-peer-access')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ggml-cuda.cu'
            path.write_text('before\n' + module.OLD + 'after\n')
            self.assertTrue(module.patch(path))
            first = path.read_text()
            self.assertIn('ggml_cuda_info().device_count <= 8', first)
            self.assertIn('use_peer_access = true;', first)
            self.assertEqual(first.count('{'), first.count('}'))
            self.assertTrue(module.patch(path))
            self.assertEqual(path.read_text(), first)
            path.write_text('unrelated\n')
            self.assertFalse(module.patch(path))

    def test_kolibri1_patch_applies_once_and_fails_closed(self):
        module = load('patch-llama-kolibri1')
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / 'src').mkdir()
            (base / 'src' / 'llama-arch.h').write_text('unrelated\n')
            self.assertFalse(module.patch(base))
            (base / 'src' / 'llama-arch.h').write_text('LLM_ARCH_KOLIBRI1 already here\n')
            self.assertTrue(module.patch(base))
        patch_text = module.PATCH_FILE.read_text()
        self.assertEqual(patch_text.count('\ndiff --git '), 8)
        self.assertIn('LLM_ARCH_KOLIBRI1', patch_text)
        self.assertNotIn('\n+++ b/conversion/', patch_text)


    def test_nccl_config_patch_guards_maxctas_and_is_idempotent(self):
        module = load('patch-llama-nccl-config')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ggml-cuda.cu'
            path.write_text('unrelated\n')
            self.assertFalse(module.patch(path))
            # llama.cpp renamed the constant (DGX_SPARK -> GB10, b12651): both spellings must be patched
            for const in ('GGML_CUDA_CC_DGX_SPARK', 'GGML_CUDA_CC_GB10'):
                upstream = ('    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;\n'
                            f'    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == {const}) {{\n'
                            '        config.maxCTAs = 8;\n'
                            '    }\n')
                path.write_text(upstream)
                self.assertTrue(module.patch(path))
                patched = path.read_text()
                self.assertIn('#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 17, 1)', patched)
                self.assertIn(f'cc == {const}', patched)
                self.assertIn('GGML_UNUSED(cc);', patched)
                # the member is only reached inside the version guard
                self.assertLess(patched.index('#if NCCL_VERSION_CODE'), patched.index('config.maxCTAs'))
                self.assertLess(patched.index('config.maxCTAs'), patched.index('#else'))
                self.assertTrue(module.patch(path))
                self.assertEqual(path.read_text(), patched)

    def test_jev_decision_patch_is_atomic_and_idempotent(self):
        module = load('patch-llama-jev-decision')
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / 'common').mkdir()
            (base / 'common' / 'common.h').write_text('unrelated\n')
            # anchors are missing: the patch fails and writes nothing
            self.assertFalse(module.patch(base))
            self.assertEqual((base / 'common' / 'common.h').read_text(), 'unrelated\n')
            # a tree in which every anchor occurs exactly once
            sources = {}
            for rel, old, _ in module.EDITS:
                sources[rel] = sources.get(rel, '') + old
            for rel, text in sources.items():
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                (base / rel).write_text(text)
            self.assertTrue(module.patch(base))
            patched = {rel: (base / rel).read_text() for rel in sources}
            self.assertIn('COMMON_DECISION_TYPE_JEV', patched[Path('common/common.h')])
            self.assertTrue(module.patch(base))
            self.assertEqual(patched, {rel: (base / rel).read_text() for rel in sources})
            # a duplicated anchor is refused and the tree stays as it was
            (base / 'tools/server/server-decision.h').write_text(sources[Path('tools/server/server-decision.h')] * 2)
            (base / 'common' / 'common.h').write_text(sources[Path('common/common.h')])
            before = (base / 'common' / 'common.cpp').read_text()
            (base / 'common' / 'common.cpp').write_text(sources[Path('common/common.cpp')])
            self.assertFalse(module.patch(base))
            self.assertEqual((base / 'common' / 'common.cpp').read_text(), sources[Path('common/common.cpp')])

if __name__ == '__main__':
    unittest.main()
