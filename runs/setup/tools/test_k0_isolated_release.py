"""Focused adapter contracts; all mutation targets are temporary fake modules/files."""
import json
import ast
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import k0_isolated_release as adapter
import k0_isolated_control as control
import k3_release_audit as audit


class AdapterTests(unittest.TestCase):
    def enterContext(self, context):  # Python 3.10 compatibility
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def fixture(self):
        self.identity = sorted(adapter.IDS)[0]
        self.row = {'id': self.identity, 'script': 'scripts/dsreset_train.py', 'gpu': 'any',
                    'host': 'local', 'priority': 5, 'args': ['--name', self.identity]}
        self.grant = {'rows': {self.identity: {'row': self.row}}, 'kind': 'k0-isolated-targeted-v1',
                      'adapter_sha256': 'adapter'}
        self.snapshot = {'code_root': '/isolated/ffedec5', 'source_files': {
            'scripts/dsreset_train.py': {'sha256': 'exact-source'}}}
        self.trace = []
        def delegated(label, result):
            def method(*args, **kwargs):
                self.trace.append((label, args, kwargs))
                return result
            return method
        self.scheduler = ModuleType('ops.jobs.scheduler')
        self.scheduler.launch = delegated('launch', 123)
        self.scheduler.record = delegated('record', 'recorded')
        self.local = ModuleType('ops.jobs.realworld_local')
        self.local.localize = delegated('localize', ('original-args', 'original-inputs'))
        self.local._inputs = delegated('inputs', (['localized'], {'inputs': {'verified': 'digest'}}))
        self.local._arguments = delegated('arguments', SimpleNamespace(config=['parsed']))
        self.local.applies = lambda *a: True
        self.local.LocalInputError = type('LocalInputError', (ValueError,), {})
        self.local.MIRROR = Path('/main/runs/realworld-inputs')
        self.local.MANIFEST = self.local.MIRROR / 'MANIFEST.json'
        self.local.REMOTE_PREFIX = '/remote'
        self.local.FINISHED_DIR = Path('/main/runs/setup/finished')
        package = ModuleType('ops.jobs')
        package.scheduler, package.realworld_local = self.scheduler, self.local
        modules = {'ops': ModuleType('ops'), 'ops.jobs': package,
                   'ops.jobs.scheduler': self.scheduler, 'ops.jobs.realworld_local': self.local}
        self.stack = self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(adapter, 'load_authorization', return_value=(self.grant, self.snapshot, 'grant')))
        self.enterContext(patch.object(adapter, 'fence'))
        self.enterContext(patch.object(Path, 'exists', return_value=True))
        self.assertTrue(adapter.install(Path('/fake/authorization.json')))

    def test_missing_grant_is_noop_without_production_imports(self):
        with patch.object(Path, 'exists', return_value=False), patch.object(adapter, 'load_authorization') as load:
            self.assertFalse(adapter.install(Path('/missing')))
            load.assert_not_called()

    def test_invalid_optional_grant_does_not_break_driver(self):
        with patch.object(Path, 'exists', return_value=True), patch.object(adapter, 'load_authorization', side_effect=ValueError):
            self.assertFalse(adapter.install(Path('/invalid')))

    def test_other_row_localize_delegates_identical_objects_and_keywords(self):
        self.fixture()
        row, state, roles = {'id': 'existing'}, {}, {'unchanged': True}
        self.assertEqual(('original-args', 'original-inputs'), self.local.localize(row, state, roles=roles))
        label, args, kwargs = self.trace[-1]
        self.assertEqual('localize', label)
        self.assertIs(args[0], row)
        self.assertIs(args[1], state)
        self.assertIs(kwargs['roles'], roles)

    def test_other_row_launch_is_byte_for_byte_delegated(self):
        self.fixture()
        row, args = {'id': 'existing'}, ['original', 'argv']
        self.assertEqual(123, self.scheduler.launch(row, 'MIG-test', 7, Path('/runs'), args=args, cpu_threads=8))
        label, positional, kwargs = self.trace[-1]
        self.assertEqual(('launch', (row, 'MIG-test', 7, Path('/runs'))), (label, positional))
        self.assertIs(positional[0], row)
        self.assertIs(kwargs['args'], args)
        self.assertEqual(8, kwargs['cpu_threads'])

    def test_other_row_record_is_unchanged(self):
        self.fixture()
        event = {'id': 'existing', 'event': 'launched', 'host': 'local', 'commit': 'old'}
        self.scheduler.record(event, Path('/registry'))
        self.assertIs(self.trace[-1][1][0], event)
        self.assertEqual('old', event['commit'])

    def test_exact_row_uses_existing_input_validator_with_isolated_config_root(self):
        self.fixture()
        result, proof = self.local.localize(self.row, {}, roles={})
        self.assertEqual(['localized'], result)
        self.assertEqual('inputs', self.trace[-1][0])
        self.assertEqual(Path('/isolated/ffedec5'), self.trace[-1][1][-2])
        self.assertEqual(adapter.COMMIT, proof['execution_scope']['source_commit'])
        self.assertEqual({'verified': 'digest'}, proof['inputs'])

    def test_other_row_parser_is_unchanged(self):
        self.fixture()
        row = {'id': 'existing', 'args': ['unchanged']}
        self.local._arguments(row)
        self.assertEqual('arguments', self.trace[-1][0])
        self.assertIs(row, self.trace[-1][1][0])

    def test_exact_evaluator_retains_k0_config_and_all_existing_parser_checks(self):
        self.fixture()
        self.row.update(realworld_target='covering', args=[
            '--config', 'configs/realworld/k0/covering-S-orig-k0-400k.yaml'])
        result = self.local._arguments(self.row)
        self.assertEqual(['configs/realworld/k0/covering-S-orig-k0-400k.yaml'], result.config)
        self.assertEqual('arguments', self.trace[-1][0])
        self.assertIn('configs/realworld/ours_matrix/covering-S-orig-scratch400k.yaml', self.trace[-1][1][0]['args'])

    def test_exact_row_keeps_guarded_original_launcher_and_thread_quota(self):
        self.fixture()
        self.scheduler.launch(self.row, 'MIG-test', 1, Path('/runs'), args=['localized'], cpu_threads=8)
        label, args, kwargs = self.trace[-1]
        self.assertEqual('launch', label)
        self.assertEqual('runs/setup/tools/k0_isolated_worker.py', args[0]['script'])
        self.assertEqual(8, kwargs['cpu_threads'])
        self.assertEqual(['localized'], kwargs['args'][2:])
        self.assertEqual('scripts/dsreset_train.py', self.row['script'])

    def test_modified_scientific_args_priority_and_host_are_refused(self):
        self.fixture()
        for key, value in (('args', ['different']), ('priority', 4), ('host', 'remote')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.check_row({**self.row, key: value}, self.grant)
        adapter.check_row({**self.row, 'gpu': 'MIG-chosen'}, self.grant)

    def test_only_exact_row_launched_provenance_gets_new_identity(self):
        self.fixture()
        event = {'id': self.identity, 'event': 'launched', 'host': 'local', 'commit': 'main'}
        self.scheduler.record(event)
        recorded = self.trace[-1][1][0]
        self.assertEqual(adapter.COMMIT, recorded['commit'])
        self.assertEqual('main', event['commit'])
        self.assertEqual('grant', recorded['isolated_authorization_sha256'])

    def test_revocation_fence_refuses_changed_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'grant.json'
            path.write_text('original')
            expected = adapter.digest(path.read_bytes())
            adapter.fence(path, expected)
            path.write_text('revoked')
            with self.assertRaises(ValueError):
                adapter.fence(path, expected)

    def test_existing_pause_schema_only_changes_named_active_bit(self):
        doc = {'pauses': [{'name': 'other', 'active': True, 'ids': ['existing']},
                          {'name': 'k0-20261120-held', 'active': True, 'ids': sorted(adapter.IDS)}]}
        after = json.loads(control.pause_document(json.dumps(doc).encode(), False))
        self.assertEqual(doc['pauses'][0], after['pauses'][0])
        self.assertEqual(doc['pauses'][1]['ids'], after['pauses'][1]['ids'])
        self.assertFalse(after['pauses'][1]['active'])
        self.assertEqual(doc, json.loads(control.pause_document(json.dumps(after).encode(), True)))

    def test_existing_pause_prefix_or_wrong_id_set_is_refused(self):
        for entry in ({'ids': ['wrong']}, {'ids': sorted(adapter.IDS), 'prefixes': ['rw']}):
            with self.assertRaises(ValueError):
                control.pause_document(json.dumps({'pauses': [{'name': 'k0-20261120-held', **entry}]}).encode(), False)

    def test_child_has_real_guard_call_recognized_by_deployed_scheduler(self):
        text = Path(adapter.__file__).with_name('k0_isolated_worker.py').read_text()
        self.assertRegex(text, re.compile(r'^(guard_gpus|enforce_allowed_gpus)\(\)\s*$', re.MULTILINE))
        self.assertLess(text.index('guard_gpus()'), text.index('runpy.run_path'))

    def test_runtime_path_extension_is_limited_to_own_output_and_private_cache(self):
        root = Path('/isolated/ffedec5')
        def die(message):
            raise ValueError(message)
        original = audit.function(audit.git('show', f'{adapter.COMMIT}:ops/resources/gpu_guard.py').decode(),
                                  'validate_runtime_path', {'Path': Path, 'REPO': root, 'RUNTIME_ROOT': None, '_die': die})
        tree = ast.parse(Path(adapter.__file__).read_text())
        prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_worker')
        node = next(n for n in prepare.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_runtime_path')
        namespace = {'Path': Path, 'MAIN': Path('/main'), 'own_output': Path('/main/runs/exact-id'),
                     'own_cache': Path('/main/.cache/k0-isolated/exact-id'), 'original_validate': original}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<runtime-path-adapter>', 'exec'), namespace)
        validate = namespace['validate_runtime_path']
        self.assertEqual(Path('/main/runs/exact-id/completion.json'), validate('/main/runs/exact-id/completion.json'))
        self.assertEqual(Path('/main/.cache/k0-isolated/exact-id/torch'), validate('/main/.cache/k0-isolated/exact-id/torch', cache=True))
        for denied in ('/main/runs/other/completion.json', '/main/.cache/other/torch', '/main/scripts/train.py'):
            with self.subTest(path=denied), self.assertRaises(ValueError):
                validate(denied)

    def sealed_fixture(self, directory):
        # Entirely synthetic acceptance fixture: never copied to a shared path.
        root = directory / 'code'
        root.mkdir()
        source = root / 'science.py'
        source.write_text('# immutable fixture\n')
        source.chmod(0o444)
        root.chmod(0o555)
        snapshot = directory / 'snapshot.json'
        snapshot.write_bytes(adapter.canonical({'code_root': str(root), 'commit': adapter.COMMIT,
            'source_files': {'science.py': {'sha256': adapter.digest(source.read_bytes())}}}))
        rows = {}
        for identity in adapter.IDS:
            row = {'id': identity, 'host': 'local', 'priority': 5, 'planned_commit': adapter.COMMIT,
                   'release_source': 'local', 'local_placement_pause': 'k0-20261120-held'}
            rows[identity] = {'row': row, 'row_sha256': adapter.digest(adapter.canonical(row))}
        artifact = directory / 'synthetic-evidence.json'
        artifact.write_text('{}')
        evidence = directory / 'inventory.json'
        evidence.write_bytes(adapter.canonical({'ids': sorted(adapter.IDS), 'commit': adapter.COMMIT,
            'rows': rows, 'targeted_scientific_evidence_verified': True,
            'native': {'optimizer_updates': 120, 'finite_losses_and_gradients': True,
                       'artifacts_sha256': {str(artifact): adapter.digest(artifact.read_bytes())}},
            'inherited_targeted_logs': {},
            'k2_cpu_artifacts_sha256': {str(artifact): adapter.digest(artifact.read_bytes())}}))
        grant = directory / 'authorization.json'
        grant.write_bytes(adapter.canonical({'kind': 'k0-isolated-targeted-v1', 'active': True,
            'commit': adapter.COMMIT, 'rows': rows,
            'adapter_sha256': adapter.digest(Path(adapter.__file__).read_bytes()),
            'worker_sha256': adapter.digest(Path(adapter.__file__).with_name('k0_isolated_worker.py').read_bytes()),
            'snapshot': str(snapshot), 'snapshot_sha256': adapter.digest(snapshot.read_bytes()),
            'targeted_evidence': str(evidence), 'targeted_evidence_sha256': adapter.digest(evidence.read_bytes())}))
        return grant, root, artifact

    def test_complete_synthetic_grant_loads_and_changed_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            grant, root, artifact = self.sealed_fixture(Path(temp))
            loaded, snapshot, sha = adapter.load_authorization(grant)
            self.assertEqual(adapter.IDS, set(loaded['rows']))
            self.assertEqual(str(root), snapshot['code_root'])
            self.assertEqual(adapter.digest(grant.read_bytes()), sha)
            artifact.write_text('changed')
            with self.assertRaises(ValueError):
                adapter.load_authorization(grant)

    def test_writable_source_is_refused_even_if_content_is_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            grant, root, artifact = self.sealed_fixture(Path(temp))
            (root / 'science.py').chmod(0o644)
            with self.assertRaises(ValueError):
                adapter.load_authorization(grant)


if __name__ == '__main__':
    unittest.main()
