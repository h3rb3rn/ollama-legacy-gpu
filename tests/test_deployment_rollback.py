"""Exercise replacement failure paths without touching real containers."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deploy', ROOT / 'scripts/deploy-tested-release.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class Rollback(unittest.TestCase):
    def setUp(self):
        self.old = {'Id': 'old-id', 'HostConfig': {
            'RestartPolicy': {'Name': 'on-failure', 'MaximumRetryCount': 3}}}
        self.state = {
            'old-id': {'name': 'production', 'running': True, 'restart': 'on-failure:3'},
            'new-id': {'name': 'candidate', 'running': False, 'restart': 'on-failure:3'},
            'unrelated': {'name': 'other-production', 'running': True, 'restart': 'always'},
        }
        self.report = {'passed': False}

    def docker(self, *args):
        _, operation, *parts = args
        target = parts[-2] if operation == 'rename' else parts[-1]
        row = self.state[target]
        if operation == 'stop':
            row['running'] = False
        elif operation == 'start':
            row['running'] = True
        elif operation == 'rename':
            self.assertNotIn(parts[-1], [r['name'] for k, r in self.state.items() if k != target])
            row['name'] = parts[-1]
        elif operation == 'update':
            row['restart'] = parts[0].removeprefix('--restart=')
        else:
            self.fail(f'Unexpected Docker operation: {operation}')
        return target

    def invoke(self, verify):
        with patch.object(deploy.candidate, 'run', side_effect=self.docker), \
             patch.object(deploy.candidate, 'inspect', return_value=self.old), \
             patch.object(deploy.candidate, 'wait_api', return_value=('url', {'version': 'test'})):
            deploy.replace(self.old, 'production', 'new-id', 'previous', 'candidate', verify, self.report)

    def test_success_retains_stopped_previous_container(self):
        self.invoke(lambda: None)
        self.assertTrue(self.report['passed'])
        self.assertEqual(self.state['new-id']['name'], 'production')
        self.assertTrue(self.state['new-id']['running'])
        self.assertEqual(self.state['old-id'], {'name': 'previous', 'running': False, 'restart': 'no'})

    def test_bad_inference_restores_identity_name_and_restart_policy(self):
        def bad_inference():
            raise RuntimeError('incorrect answer')
        with self.assertRaisesRegex(RuntimeError, 'incorrect answer'):
            self.invoke(bad_inference)
        self.assertFalse(self.report['passed'])
        self.assertTrue(self.report['rolled_back'])
        self.assertEqual(self.state['old-id'], {
            'name': 'production', 'running': True, 'restart': 'on-failure:3'})
        self.assertFalse(self.state['new-id']['running'])
        self.assertEqual(self.state['new-id']['name'], 'candidate-failed')
        self.assertEqual(self.state['unrelated'], {
            'name': 'other-production', 'running': True, 'restart': 'always'})

    def test_interruption_also_rolls_back(self):
        def interrupted():
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(interrupted)
        self.assertTrue(self.report['rolled_back'])
        self.assertTrue(self.state['old-id']['running'])

    def test_container_start_failure_restores_previous_instance(self):
        normal = self.docker

        def failed_start(*args):
            if args == ('docker', 'start', 'new-id'):
                raise RuntimeError('CUDA runtime initialization failed')
            return normal(*args)

        with patch.object(self, 'docker', side_effect=failed_start):
            with self.assertRaisesRegex(RuntimeError, 'CUDA runtime'):
                self.invoke(lambda: self.fail('Inference must not run after failed start'))
        self.assertTrue(self.report['rolled_back'])
        self.assertTrue(self.state['old-id']['running'])


if __name__ == '__main__':
    unittest.main()
