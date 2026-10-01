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
            for name in ('patch-ollama-fa', 'patch-ollama-dynamic-pool', 'patch-ollama-batch',
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


if __name__ == '__main__':
    unittest.main()
