"""Focused short-socket and exact-history activation regression tests."""
import copy
import ast
import json
import multiprocessing.connection
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

import k0_isolated_control as control
import k0_isolated_release as adapter


class TempTests(unittest.TestCase):
    def test_worker_repairs_tmp_after_prepare_and_before_guard_or_science(self):
        tree = ast.parse(Path(adapter.__file__).with_name('k0_isolated_worker.py').read_text())
        calls = [n.value.func for n in tree.body if isinstance(n, (ast.Expr, ast.Assign))
                 and isinstance(n.value, ast.Call)]
        names = [ast.unparse(n) for n in calls]
        self.assertLess(names.index('prepare_worker'), names.index('configure_worker_temp'))
        self.assertLess(names.index('configure_worker_temp'), names.index('guard_gpus'))
        self.assertLess(names.index('guard_gpus'), names.index('runpy.run_path'))

    def test_all_nine_longest_listener_paths_fit_linux_sun_path(self):
        # These are CPython's actual prefixes and longest generated random suffix.
        suffix = 'x' * len(next(tempfile._RandomNameSequence()))
        for identity in adapter.IDS:
            directory = Path('/tmp') / ('k0i-' + adapter.digest(identity.encode())[:10])
            path = directory / ('pymp-' + suffix) / ('listener-' + suffix)
            self.assertLessEqual(len(os.fsencode(path)) + 1, 108)  # terminating NUL
        source = __import__('inspect').getsource(multiprocessing.connection.arbitrary_address)
        self.assertIn("prefix='listener-'", source)
        self.assertIn("prefix='pymp-'", __import__('inspect').getsource(
            multiprocessing.util.get_temp_dir))

    def test_private_mode_full_id_marker_reuse_and_collision_refusal(self):
        identity = 'k5-unit-' + uuid.uuid4().hex
        directory = Path('/tmp') / ('k0i-' + adapter.digest(identity.encode())[:10])
        try:
            with patch.object(adapter, 'IDS', {identity}), patch.dict(os.environ), patch.object(tempfile, 'tempdir'):
                self.assertEqual(directory, adapter.configure_worker_temp(identity))
                self.assertEqual(0o700, stat.S_IMODE(directory.stat().st_mode))
                self.assertEqual(os.geteuid(), directory.stat().st_uid)
                self.assertEqual(directory, adapter.configure_worker_temp(identity))
                (directory / 'row-id').write_text('another full identity\n')
                with self.assertRaises(ValueError):
                    adapter.configure_worker_temp(identity)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_existing_symlink_wrong_mode_missing_marker_or_wrong_owner_refused(self):
        identity = 'k5-unit-' + uuid.uuid4().hex
        directory = Path('/tmp') / ('k0i-' + adapter.digest(identity.encode())[:10])
        with patch.object(adapter, 'IDS', {identity}):
            try:
                directory.symlink_to('/tmp', target_is_directory=True)
                with self.assertRaises(OSError):
                    adapter.configure_worker_temp(identity)
                directory.unlink()
                directory.mkdir(mode=0o755)
                with self.assertRaises(ValueError):
                    adapter.configure_worker_temp(identity)
                directory.chmod(0o700)
                with self.assertRaises(FileNotFoundError):
                    adapter.configure_worker_temp(identity)
                with patch.object(adapter.os, 'geteuid', return_value=os.geteuid() + 1):
                    with self.assertRaises(ValueError):
                        adapter.configure_worker_temp(identity)
            finally:
                if directory.is_symlink():
                    directory.unlink()
                else:
                    shutil.rmtree(directory, ignore_errors=True)

    def test_real_torch_dataloader_tensor_ipc_after_adapter_bootstrap(self):
        probe = Path(__file__).with_name('k5_dataloader_probe.py')
        result = subprocess.run([sys.executable, '-B', '-I', str(probe)], text=True,
                                capture_output=True, timeout=75)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        proof = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(6, proof['workers'])
        self.assertEqual(192, proof['samples'])
        print('DATALOADER_PROOF ' + json.dumps(proof, sort_keys=True))


class HistoryTests(unittest.TestCase):
    def fixture(self):
        identity = next(i for i in adapter.IDS if '-pre-ours-' in i)
        history = [{'event': 'launched', 'id': identity, 'host': 'local', 'attempt': 1,
                    'commit': adapter.COMMIT, 'acceptance_kind': 'k0-isolated-targeted-v1',
                    'isolated_source_root': '/sealed', 'time': 1},
                   {'event': 'exited', 'id': identity, 'host': 'local', 'attempt': 1, 'code': 143, 'time': 2}]
        grant = {'kind': 'k0-isolated-targeted-v1', 'rows': {identity: {'row': {
            'max_restarts': 1, 'args': ['--resume', 'auto']}}},
                 'retry_history': {identity: adapter.digest(adapter.canonical(history))}}
        return identity, history, grant, {'code_root': '/sealed'}

    def test_reviewed_failure_allowed_without_history_mutation(self):
        identity, history, grant, snapshot = self.fixture()
        before = copy.deepcopy(history)
        control.check_history({identity: history}, grant, snapshot)
        self.assertEqual(before, history)

    def test_first_activation_still_requires_no_target_history(self):
        identity, history, grant, snapshot = self.fixture()
        grant.pop('retry_history')
        control.check_history({}, grant, snapshot)
        with self.assertRaises(ValueError):
            control.check_history({identity: history}, grant, snapshot)

    def test_unreviewed_missing_new_or_changed_target_events_refused(self):
        identity, history, grant, snapshot = self.fixture()
        for events in ({}, {identity: history[:-1]}, {identity: history + [history[-1]]},
                       {identity: [{**history[0], 'time': 3}, history[1]]}):
            with self.subTest(events=len(events)), self.assertRaises(ValueError):
                control.check_history(events, grant, snapshot)

    def test_even_reviewed_success_active_remote_exhausted_or_repin_refused(self):
        for field, value in (('code', 0), ('host', 'remote'), ('attempt', 2), ('commit', 'other'),
                             ('isolated_source_root', '/changed')):
            identity, history, grant, snapshot = self.fixture()
            index = 1 if field == 'code' else 0
            history[index][field] = value
            grant['retry_history'][identity] = adapter.digest(adapter.canonical(history))
            with self.subTest(field=field), self.assertRaises(ValueError):
                control.check_history({identity: history}, grant, snapshot)
        for change in ({'max_restarts': 0}, {'args': []}, {'args': ['--resume', 'never']}):
            identity, history, grant, snapshot = self.fixture()
            grant['rows'][identity]['row'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                control.check_history({identity: history}, grant, snapshot)

    def test_control_plan_copy_is_readonly_accepts_unrelated_history_and_refuses_target_drift(self):
        import yaml
        identity, history, grant, snapshot = self.fixture()
        grant['rows'] = {i: {'row': {'id': i, 'max_restarts': 1, 'args': ['--resume', 'auto']}}
                         for i in adapter.IDS}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'experiments').mkdir()
            (root / 'runs/setup').mkdir(parents=True)
            (root / 'runs' / identity).mkdir()  # existing run directory is supported
            queue = root / 'experiments/queue.yaml'
            queue.write_text(yaml.safe_dump({'jobs': [v['row'] for v in grant['rows'].values()]}))
            pauses = root / 'runs/setup/admission-pauses.json'
            pause_doc = {'pauses': [{'name': 'other', 'active': True, 'ids': ['unrelated']},
                {'name': 'k0-20261120-held', 'active': True, 'ids': sorted(adapter.IDS)}]}
            pauses.write_text(json.dumps(pause_doc))
            registry = root / 'experiments/registry.jsonl'
            registry.write_text(''.join(json.dumps(e) + '\n' for e in history))
            candidate = root / 'candidate.json'
            candidate.write_text(json.dumps(grant))
            before = {p: p.read_bytes() for p in (queue, pauses, registry, candidate)}
            with patch.object(adapter, 'load_authorization', return_value=(grant, snapshot, 'sha')), patch.object(
                    control.fcntl, 'flock', side_effect=AssertionError('preview took a lock')):
                plan = control.plan('activate', candidate, root=root)
                self.assertEqual(before, {p: p.read_bytes() for p in before})
                self.assertFalse(plan['target'].exists())
                after = json.loads(plan['after'])
                self.assertEqual(pause_doc['pauses'][0], after['pauses'][0])
                self.assertFalse(after['pauses'][1]['active'])
                with registry.open('a') as stream:
                    stream.write(json.dumps({'id': 'another-row', 'event': 'launched'}) + '\n')
                control.plan('activate', candidate, root=root)  # other rows remain irrelevant
                with registry.open('a') as stream:
                    stream.write(json.dumps(history[-1]) + '\n')
                with self.assertRaises(ValueError):
                    control.plan('activate', candidate, root=root)
                registry.write_bytes(before[registry])
                pause_doc['pauses'][1]['active'] = False
                pauses.write_text(json.dumps(pause_doc))
                with self.assertRaises(ValueError):
                    control.plan('activate', candidate, root=root)


if __name__ == '__main__':
    unittest.main()
