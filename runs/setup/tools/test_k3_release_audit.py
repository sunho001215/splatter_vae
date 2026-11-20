"""Focused contracts for the actual immutable gate and both source fingerprint readers."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import k3_release_audit as audit


class ReleaseBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bootstrap = audit.git('show', f'{audit.COMMIT}:scripts/_bootstrap.py').decode()
        cls.scheduler = audit.git('show', 'HEAD:ops/jobs/scheduler.py').decode()

    def setUp(self):
        self.fingerprints = {'s4d/science.py': 'immutable'}
        self.report = {'exit_code': 0, 'counts': {'passed': 13, 'failed': 0, 'error': 0, 'skipped': 0},
                       'source_sha256': self.fingerprints, 'ids': list(audit.IDS),
                       'commit': audit.COMMIT, 'acceptance_kind': 'targeted'}

    def test_actual_immutable_gate_rejects_targeted_success_for_exact_nine(self):
        self.assertIn('all tests must pass', audit.probe_gate(self.bootstrap, self.report, self.fingerprints))

    def test_actual_immutable_gate_accepts_synthetic_complete_suite_control(self):
        self.report['full_suite_passed'] = True
        self.assertEqual('accepted', audit.probe_gate(self.bootstrap, self.report, self.fingerprints))

    def test_actual_immutable_gate_rejects_stale_suite(self):
        self.report['full_suite_passed'] = True
        self.assertIn('stale', audit.probe_gate(self.bootstrap, self.report, {'s4d/science.py': 'new'}))

    def test_actual_immutable_gate_rejects_skips(self):
        self.report['full_suite_passed'] = True
        self.report['counts']['skipped'] = 1
        self.assertNotEqual('accepted', audit.probe_gate(self.bootstrap, self.report, self.fingerprints))

    def test_targeted_record_never_impersonates_full_suite(self):
        native = audit.native_summary()
        self.assertEqual(120, native['optimizer_updates'])
        self.assertTrue(native['finite_losses_and_gradients'])
        self.assertNotIn('full_suite_passed', native)

    def readers(self, root):
        namespace = {'REPO': root, 'hashlib': hashlib, 'third_party_identity': lambda _: {},
                     'third_party_identity_unused': None, 'dict': dict}
        bootstrap = audit.function(self.bootstrap, 'source_fingerprints', dict(namespace))
        namespace['subprocess'] = SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout='base'))
        scheduler = audit.function(self.scheduler, 'source_identity', namespace)
        return bootstrap, lambda: scheduler()['source_sha256']

    def test_ops_driver_merge_stales_both_existing_source_readers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'mujoco_mig_setup.py').write_text('# fixed\n')
            driver = root / 'ops/campaign/remote_campaign_driver.py'
            driver.parent.mkdir(parents=True)
            driver.write_text('# original\n')
            for reader in self.readers(root):
                before = reader()
                driver.write_text('# exact-ID branch, unchanged unrelated semantics\n')
                self.assertNotEqual(before, reader())
                driver.write_text('# original\n')

    def test_runs_wrapper_merge_leaves_both_existing_source_readers_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'mujoco_mig_setup.py').write_text('# fixed\n')
            wrapper = root / 'runs/setup/tools/remote_campaign_driver.py'
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text('# original\n')
            for reader in self.readers(root):
                before = reader()
                wrapper.write_text('# new exact-ID reader\n')
                self.assertEqual(before, reader())
                wrapper.write_text('# original\n')

    def test_allowlist_is_exact_nine_not_prefix(self):
        self.assertEqual(9, len(set(audit.IDS)))
        self.assertNotIn(audit.IDS[0] + '-extra', audit.IDS)
        self.assertNotIn('rwk0-20261120-fql-ours-covering-S-orig-k0-400k', audit.IDS)


if __name__ == '__main__':
    unittest.main()
