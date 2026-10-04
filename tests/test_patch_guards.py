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
            for name in ('patch-ollama-fa', 'patch-ollama-dynamic-pool', 'patch-ollama-batch', 'patch-ollama-discovery', 'patch-ollama-mtp-default', 'patch-llama-cuda-graphs-legacy', 'patch-llama-fit-nextn',
                         'patch-llama-tier-fitting', 'patch-llama-jinja-tojson'):
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
            self.assertIn('if !draftNumPredictSet', first)
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


if __name__ == '__main__':
    unittest.main()
