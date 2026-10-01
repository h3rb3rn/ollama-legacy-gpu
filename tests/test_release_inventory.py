import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inventory', ROOT / 'scripts/release-inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class HardwareGate(unittest.TestCase):
    def setUp(self):
        self.rows = [{'id': family, 'family': family, 'variant': variant,
                      'runner': ['self-hosted', 'test-fixture-host'],
                      'config': '/test-fixture/' + family + '.json', 'deploy': family == 'm10'}
                     for family, variant in inventory.FAMILIES.items()]

    def test_complete_inventory(self):
        self.assertEqual(inventory.validate(self.rows)['include'], self.rows)

    def test_k80_cannot_be_omitted(self):
        with self.assertRaises(ValueError):
            inventory.validate(self.rows[1:])

    def test_cuda13_cannot_be_assigned_to_maxwell(self):
        self.rows[1]['variant'] = 'cuda13-rtx'
        with self.assertRaises(ValueError):
            inventory.validate(self.rows)

    def test_hosted_runner_cannot_claim_hardware_validation(self):
        self.rows[1]['runner'] = ['ubuntu-latest']
        with self.assertRaises(ValueError):
            inventory.validate(self.rows)

    def test_duplicate_evidence_names_are_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaises(ValueError):
            inventory.validate(self.rows)

    def test_implicit_deploy_permission_is_rejected(self):
        self.rows[1]['deploy'] = 'false'
        with self.assertRaises(ValueError):
            inventory.validate(self.rows)


if __name__ == '__main__':
    unittest.main()
