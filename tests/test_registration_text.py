"""Legacy registrar replay: no production command imports, filesystem, locks, or processes."""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from ops.jobs._registration_text import render_appended_jobs

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name('fixtures') / 'registration'
NAMES = ('register_snerl_eight_task_20261020.py', 'register_afk2_20261020.py')
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side',
         'pick-place', 'stick-push')
QUEUE = '/fixture/experiments/queue.yaml'
BLOCK = (b'# fixed queue\nhosts: {local: {}, remote: {}}\njobs:\n- id: old\n  host: local\n'
         b'# suffix comment\nnotes: "caf\xc3\xa9"  \n')
RENDER_CASES = {
    'block_suffix_unicode': BLOCK,
    'no_final_newline': b'jobs:\n- id: old',
    'whitespace': b'jobs:  \n- id: old  \n  # retained boundary comment\n\nnotes: yes\n',
    'empty_block': b'jobs:\nnotes: tail\n',
    'flow_empty': b'jobs: []\nnotes: tail\n',
    'flow_rows': b'jobs: [{id: old}]\nnotes: tail\n',
    'missing_jobs': b'hosts: {}\n',
    'empty_document': b'',
    'scalar_document': b'hello\n',
    'sequence_document': b'- jobs\n',
    'scalar_jobs': b'jobs: hello\n',
    'malformed_yaml': b'jobs: [\n',
    'invalid_utf8': b'jobs: \xff\n',
    'duplicate_jobs': b'jobs:\n- id: first\njobs:\n- id: second\n',
    'complex_key': b'? [a, b]\n: value\njobs:\n- id: old\n',
}
ROW_CASES = {
    'ordered': [{'id': 'new', 'host': 'local', 'nested': {'b': 2, 'a': 1}}],
    'json_values': [{'id': 'new', 'unicode': '\uc720\ub2c8\ucf54\ub4dc', 'tuple': (1, None, True),
                     'mapping': {3: 'integer key'}, 'float': 1.25, 'multiline': 'a\nb\n'}],
    'empty': [],
    'unsupported_json': [{'id': 'new', 'bad': {1, 2}}],
}


def legacy_render(source, before, rows, *, yaml_module=yaml, json_module=json):
    """Extract the actual original expression; never import an operational command."""
    tree = ast.parse(source)
    body = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'text', 'node'} for t in node.targets):
            body.append(node)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'staged.write_text'):
            body.append(ast.copy_location(ast.Assign(targets=[ast.Name(id='result', ctx=ast.Store())],
                                                     value=node.value.args[0]), node))
    namespace = {'before': before, 'new_jobs': rows, 'yaml': yaml_module, 'json': json_module}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<legacy-render>', 'exec'), namespace)
    return namespace['result']


def outcome(call):
    try:
        return {'result': call()}
    except Exception as exc:
        return {'error': [type(exc).__name__, str(exc)]}


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('case', RENDER_CASES)
@pytest.mark.parametrize('rows', ROW_CASES)
def test_render_bytes_parsed_structure_and_legacy_errors(name, case, rows):
    before, values = RENDER_CASES[case], copy.deepcopy(ROW_CASES[rows])
    untouched = copy.deepcopy(values)
    source = (FIXTURES / (name + '.txt')).read_text()
    expected = outcome(lambda: legacy_render(source, before, values))
    actual = outcome(lambda: render_appended_jobs(before, values, yaml=yaml, json=json))
    assert actual == expected
    assert values == untouched
    if 'result' in actual:
        assert type(actual['result']) is str
        assert actual['result'].encode() == expected['result'].encode()
        assert outcome(lambda: yaml.safe_load(actual['result'])) == outcome(lambda: yaml.safe_load(expected['result']))


class Sandbox:
    """A closed, in-memory filesystem and allowlisted imports for a complete command pass."""

    def __init__(self, failure=None):
        self.failure = failure
        self.events = []
        self.imports = []
        self.files = {}
        self.clock_calls = 0
        self.queue_loads = 0
        self.queue_reads = 0
        self.rows = None
        self.namespace = None
        sandbox = self

        class VPath:
            def __init__(self, value):
                self.value = str(PurePosixPath(str(value)))
                assert self.value == '/fixture' or self.value.startswith('/fixture/'), self.value
                assert '..' not in PurePosixPath(self.value).parts

            def __str__(self):
                return self.value

            def __truediv__(self, value):
                return VPath(str(PurePosixPath(self.value) / value))

            @property
            def stem(self):
                return PurePosixPath(self.value).stem

            def with_suffix(self, suffix):
                return VPath(str(PurePosixPath(self.value).with_suffix(suffix)))

            def glob(self, pattern):
                sandbox.events.append(['glob', self.value, pattern])
                return [VPath(p) for p in sorted(sandbox.files) if PurePosixPath(p).parent == PurePosixPath(self.value)
                        and PurePosixPath(p).match(pattern)]

            def exists(self):
                sandbox.events.append(['exists', self.value])
                return self.value in sandbox.files or (sandbox.failure == 'proof_exists' and self.value.endswith('.json'))

            def is_file(self):
                sandbox.events.append(['is_file', self.value])
                return self.value in sandbox.files

            def open(self, mode):
                assert mode == 'a' and self.value.endswith('.lock')
                sandbox.events.append(['open', self.value, mode])
                sandbox.fail('lock_open')
                sandbox.files.setdefault(self.value, b'')
                return Lock(self.value)

            def read_bytes(self):
                return sandbox.read(self.value, 'rb')

            def read_text(self):
                return sandbox.read(self.value, 'r').decode()

            def write_bytes(self, data):
                return sandbox.write(self.value, data, 'wb')

            def write_text(self, text):
                sandbox.write(self.value, text.encode(), 'w')
                return len(text)

        class Lock:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                sandbox.events.append(['enter_lock', self.path])
                return self

            def __exit__(self, *args):
                sandbox.events.append(['close_lock', self.path])

        class Clock:
            @classmethod
            def now(cls):
                sandbox.clock_calls += 1
                sandbox.events.append(['now', sandbox.clock_calls])
                sandbox.fail('clock_prepared' if sandbox.clock_calls == 1 else 'clock_registered')
                return cls()

            def astimezone(self):
                sandbox.events.append(['astimezone'])
                return self

            def isoformat(self):
                sandbox.events.append(['isoformat'])
                return '2026-10-20T12:34:56+00:00'

        self.Path = VPath
        self.Clock = Clock

    def fail(self, stage):
        if self.failure == stage:
            raise OSError('synthetic failure: ' + stage)

    def read(self, path, mode):
        self.events.append(['open', path, mode])
        if path == QUEUE and mode == 'rb':
            self.queue_reads += 1
            self.fail('queue_before_read' if self.queue_reads == 1 else 'queue_after_read')
        if 'queue-candidate-' in path and mode == 'rb':
            self.fail('candidate_read')
        value = self.files[path]
        self.events.extend([['read', path], ['close', path]])
        return value

    def write(self, path, data, mode):
        self.events.append(['open', path, mode])
        stage = ('candidate_write' if 'queue-candidate-' in path else
                 'backup_write' if 'queue-before-' in path else
                 'proof_write' if path.endswith('.json') else 'temp_write')
        self.fail(stage)
        self.files[path] = data
        self.events.extend([['write', path, len(data), hashlib.sha256(data).hexdigest()], ['close', path]])
        return len(data)

    def flock(self, lock, operation):
        self.events.append(['flock', lock.path, operation])
        self.fail('flock')

    def replace(self, source, target):
        self.events.append(['replace', str(source), str(target)])
        self.fail('replace')
        self.files[str(target)] = self.files.pop(str(source))

    def prepare(self):
        """Synthetic prerequisites only, plus committed config bytes copied into memory."""
        for path in (REPO / 'configs').rglob('*.yaml'):
            if path.parent == REPO / 'configs/baselines/snerl' and path.stem not in TASKS:
                continue  # Reproduce the historical eight-task prerequisite without changing any current configs.
            self.files['/fixture/' + str(path.relative_to(REPO))] = path.read_bytes()
        rows = [
            {'id': 's4-snerl-hammer-rem-b', 'host': 'remote', 'script': 'scripts/train_snerl.py',
             'mem_gb': 7, 'ram_gb': 13, 'disk_gb': 21, 'required_results': ['encoder.pt']},
            {'id': 's3-m3d-seed0-door-open-200k', 'host': 'local', 'args': [
                'train.steps=200000', 'train.seed=0', 'aug.synth_views=2', 'aug.synth_render=true',
                'loss.depth_valid_far=true', 'loss.motion_space=gaussian']},
        ]
        self.files[QUEUE] = ('# synthetic queue; caf\u00e9\nhosts: {local: {}, remote: {}}\njobs:\n'
                             + yaml.safe_dump(rows, sort_keys=False)
                             + '\n# suffix survives\nnotes: "\uc720\ub2c8\ucf54\ub4dc"  \n').encode()
        self.files['/fixture/runs/s4-snerl-hammer-rem-b/encoder.pt'] = b'synthetic encoder, never torch-loaded'

    def load_queue(self, path=None):
        from ops.jobs.scheduler import load_queue

        self.queue_loads += 1
        target = self.Path(QUEUE) if path is None else path
        self.events.append(['load_queue', str(target)])
        result = load_queue(target)
        if self.failure == 'validation' and 'queue-candidate-' in str(target):
            result['jobs'] = []
        return result

    def registry(self):
        self.events.append(['read_registry'])
        value = self.files['/fixture/runs/s4-snerl-hammer-rem-b/encoder.pt']
        return {'s4-snerl-hammer-rem-b': {'status': 'done', 'sync': {
            'files': {'encoder.pt': hashlib.sha256(value).hexdigest()}}}}

    def execute(self, source, name, argv):
        self.prepare()
        # Rebind ROOT before executing any top-level code; imports cannot escape the allowlist.
        tree = ast.parse(source, name)
        roots = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets)]
        assert len(roots) == 1 and ast.literal_eval(roots[0].value.args[0]) == '/home/ws/ws/splatter4d'
        roots[0].value.args[0] = ast.Constant('/fixture')
        fake_sys = SimpleNamespace(argv=[name, *argv], path=SimpleNamespace(insert=lambda i, p:
                                   self.events.append(['sys.path.insert', i, p])))
        allowed = {
            'argparse': argparse, 'fcntl': SimpleNamespace(flock=self.flock, LOCK_EX=2),
            'hashlib': hashlib, 'json': json, 'os': SimpleNamespace(replace=self.replace), 'sys': fake_sys,
            'datetime': SimpleNamespace(datetime=self.Clock), 'pathlib': SimpleNamespace(Path=self.Path),
            'yaml': yaml, 'scripts': SimpleNamespace(jobs=SimpleNamespace(
                QUEUE=self.Path(QUEUE), REGISTRY=self.Path('/fixture/experiments/registry.jsonl'),
                DONE='done', load_queue=self.load_queue, read_registry=self.registry)),
            'ops.jobs._registration_text': SimpleNamespace(render_appended_jobs=render_appended_jobs),
        }

        def safe_import(module, globals=None, locals=None, fromlist=(), level=0):
            assert level == 0 and module in allowed, 'unintended import: ' + module
            self.imports.append(module)
            return allowed[module]

        namespace = {'__name__': '__main__', '__file__': name,
                     '__builtins__': {**vars(builtins), '__import__': safe_import,
                                      'open': lambda *a, **kw: pytest.fail('unintended real open')}}
        stdout, stderr = io.StringIO(), io.StringIO()
        error, code = None, 0
        with (patch.object(sys, 'argv', fake_sys.argv), contextlib.redirect_stdout(stdout),
              contextlib.redirect_stderr(stderr)):
            try:
                exec(compile(ast.fix_missing_locations(tree), name, 'exec'), namespace)
            except SystemExit as exc:
                code = exc.code
            except Exception as exc:
                error = [type(exc).__name__, str(exc)]
        self.rows = namespace.get('new_jobs')
        self.namespace = namespace
        return {'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(), 'exit': code, 'error': error,
                'events': self.events, 'files': {p: b.decode() for p, b in sorted(self.files.items())
                                               if not p.startswith('/fixture/configs/')},
                'rows': self.rows}


FAILURES = (None, 'proof_exists', 'lock_open', 'flock', 'queue_before_read', 'candidate_write', 'validation',
            'backup_write', 'candidate_read', 'temp_write', 'replace', 'queue_after_read', 'proof_write',
            'clock_prepared', 'clock_registered')


def replay_pair(name, failure=None, argv=None):
    if argv is None:
        argv = ['--publish'] if 'snerl' in name else ['--tasks', 'hammer', 'pick-place', '--publish']
    old = Sandbox(failure)
    new = Sandbox(failure)
    expected = old.execute((FIXTURES / (name + '.txt')).read_text(), name, argv)
    actual = new.execute((REPO / 'runs/setup/tools' / name).read_text(), name, argv)
    assert actual == expected
    assert [n for n in new.imports if n != 'ops.jobs._registration_text'] == old.imports
    return expected, actual


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('failure', FAILURES)
def test_complete_publication_and_failure_transcripts(name, failure):
    expected, _ = replay_pair(name, failure)
    if failure is None or (failure == 'backup_write' and 'afk2' in name):
        assert expected['error'] is None and expected['stdout'].startswith('REGISTERED ')
        assert len(expected['rows']) == (55 if 'snerl' in name else 16)
    else:
        assert expected['error'] is not None


CLI_CASES = ([], ['--help'], ['--unknown'], ['--tasks'], ['--tasks', 'hammer'],
             ['--tasks', 'hammer', '--tasks', 'pick-place'], ['--tasks', 'missing', '--publish'],
             ['--tasks', 'hammer', 'hammer', '--publish'], ['--tasks', 'hammer', '--tag', 'fixture', '--publish'])


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('argv', CLI_CASES)
def test_manual_or_argparse_cli_transcripts(name, argv):
    replay_pair(name, argv=argv)


@pytest.mark.parametrize('name', NAMES)
def test_render_dependencies_are_supplied_by_the_caller_at_call_time(name):
    source = (REPO / 'runs/setup/tools' / name).read_text()
    call = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == '_render_appended_jobs')
    assert {k.arg: ast.unparse(k.value) for k in call.keywords} == {'yaml': 'yaml', 'json': 'json'}
    trace = []
    yaml_proxy = SimpleNamespace(compose=lambda text: (trace.append('compose'), yaml.compose(text))[1],
                                 safe_dump=lambda *a, **kw: (trace.append('safe_dump'), yaml.safe_dump(*a, **kw))[1])
    json_proxy = SimpleNamespace(dumps=lambda rows: (trace.append('dumps'), json.dumps(rows))[1],
                                 loads=lambda text: (trace.append('loads'), json.loads(text))[1])
    namespace = {'before': BLOCK, 'new_jobs': ROW_CASES['ordered'], '_render_appended_jobs': render_appended_jobs,
                 'yaml': yaml_proxy, 'json': json_proxy}
    actual = eval(compile(ast.Expression(call), name, 'eval'), namespace)
    assert trace == ['compose', 'dumps', 'loads', 'safe_dump']
    assert actual == legacy_render((FIXTURES / (name + '.txt')).read_text(), BLOCK, ROW_CASES['ordered'])


def test_private_helper_import_is_silent_and_does_not_import_research_dependencies():
    import subprocess

    code = ("import sys\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "before_path = list(sys.path)\n"
            "before_modules = set(sys.modules)\n"
            "from ops.jobs import _registration_text\n"
            "assert sys.path == before_path\n"
            "assert set(sys.modules) - before_modules == {'ops', 'ops.jobs', 'ops.jobs._registration_text'}\n")
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True,
                            cwd=REPO, timeout=30)
    assert (result.returncode, result.stdout, result.stderr) == (0, '', '')
