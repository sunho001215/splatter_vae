"""Replay historical merge commands with no real operational I/O.

The .py.txt oracles are unchanged sources from 447be17. Both command versions
run with only ROOT rebound in the AST; every operational dependency is fake.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib.util
import inspect
import io
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / 'tests/fixtures/release_adoption'
SCRIPTS = (
    'merge_adopt_4fff862_20261106.py',
    'merge_adopt_e323513_20261106.py',
    'merge_adopt_c9fc448_20261106.py',
    'merge_adopt_782f140_20261107.py',
    'merge_adopt_080f330_20261107.py',
)
HEADS = ('6a3afb3', '7d1a926', '2a35cfb', '826fefa', 'e9338cb')
HOLDS = (None, '', 'existing role hold\n')
CASES = (
    'success', 'missing_argument', 'extra_argument', 'help_is_positional',
    'hold_exists', 'hold_read_initial', 'hold_write', 'origin_run', 'bad_origin',
    'open_local', 'enter_local', 'open_remote', 'enter_remote', 'clock_start',
    'lock_local', 'lock_remote', 'clock_elapsed', 'print_locks', 'head_run', 'bad_head',
    'merge_run', 'short_head_run', 'short_head_nonzero', 'print_merged',
    'adopt_run', 'adopt_nonzero', 'print_output', 'strftime', 'print_adopted',
    'exit_remote', 'exit_local', 'hold_unlink', 'hold_read_final',
    'hold_changed', 'hold_removed', 'print_removed', 'print_kept',
    'merge_and_cleanup_error', 'adopt_and_changed_hold',
)
ROOT = '/fixture/release'
HOLD = ROOT + '/runs/setup/remote-hold-admissions'
_ACTIVE_REPLAYS = []


class _SandboxEscape(BaseException):
    """Never convert an escaped operation into an ordinary command outcome."""


def _audit_replay(event, args):
    if not _ACTIVE_REPLAYS:
        return
    if event == 'open':
        path, mode, flags = args
        helper = REPO / 'ops/jobs/_release_adoption.py'
        allowed = (str(helper), importlib.util.cache_from_source(str(helper)))
        if str(path) not in allowed or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise _SandboxEscape('unintended filesystem access: ' + str(path))
    elif event.startswith(('subprocess.', 'socket.', 'os.')):
        raise _SandboxEscape('unintended operational event: ' + event)


sys.addaudithook(_audit_replay)


@contextlib.contextmanager
def _no_operational_io():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    _ACTIVE_REPLAYS.append(True)
    try:
        yield
    finally:
        _ACTIVE_REPLAYS.pop()
        sys.dont_write_bytecode = previous


def _exception_record(exc):
    record = {'type': type(exc).__name__, 'message': str(exc),
              'code': exc.code if isinstance(exc, SystemExit) else None,
              'suppress_context': exc.__suppress_context__,
              'context': _exception_record(exc.__context__) if exc.__context__ is not None else None,
              'cause': _exception_record(exc.__cause__) if exc.__cause__ is not None else None}
    if isinstance(exc, subprocess.CalledProcessError):
        record.update(returncode=exc.returncode, cmd=exc.cmd, stdout=exc.stdout, stderr=exc.stderr)
    return record


class Replay:
    def __init__(self, script, case, hold):
        self.script, self.case = script, case
        self.commit = script.split('_')[2] + '0' * 33
        self.head = HEADS[SCRIPTS.index(script)] + '1' * 33
        self.events = []
        self.files = {ROOT + '/docs/tests.json': 'pre-existing evidence\n',
                      ROOT + '/experiments/registry.jsonl': 'pre-existing state\n'}
        if hold is not None:
            self.files[HOLD] = hold
        self.reads = self.clocks = 0
        self.current_head = self.head

    def event(self, stage, *values):
        self.events.append([stage, *values])
        failures = {
            'merge_and_cleanup_error': ('merge_run', 'hold_unlink'),
            'adopt_and_changed_hold': (),
        }.get(self.case, (self.case,))
        if stage in failures:
            if stage in ('origin_run', 'head_run', 'merge_run'):
                raise subprocess.CalledProcessError(19, values[0], output='git stdout', stderr='git stderr')
            raise OSError('injected ' + stage)

    def path(self, value):
        replay = self

        class FakePath:
            def __init__(self, name):
                self.name = str(name)
                assert self.name == ROOT or self.name.startswith(ROOT + '/')
                assert '..' not in PurePosixPath(self.name).parts

            def __truediv__(self, other):
                return FakePath(str(PurePosixPath(self.name) / other))

            def __str__(self):
                return self.name

            def exists(self):
                replay.event('hold_exists', self.name)
                return self.name in replay.files

            def read_text(self):
                replay.reads += 1
                replay.event('hold_read_initial' if replay.reads == 1 else 'hold_read_final', self.name)
                if self.name not in replay.files:
                    raise FileNotFoundError(self.name)
                return replay.files[self.name]

            def write_text(self, text):
                replay.event('hold_write', self.name, text)
                replay.files[self.name] = text
                return len(text)

            def unlink(self, **kwargs):
                replay.event('hold_unlink', self.name, kwargs)
                replay.files.pop(self.name, None)

        return FakePath(value)

    def open(self, path, mode):
        name = str(path)
        kind = 'local' if name.endswith('stage2-driver.lock') else 'remote'
        assert name == ROOT + '/runs/setup/' + ('stage2-driver.lock' if kind == 'local' else 'remote-driver.lock')
        assert mode == 'w'
        self.event('open_' + kind, name, mode)
        self.files[name] = ''
        replay = self

        class Lock:
            def __enter__(self):
                replay.event('enter_' + kind)
                return self

            def __exit__(self, exc_type, exc, tb):
                replay.event('exit_' + kind, exc_type.__name__ if exc_type else None)

            lock_kind = kind

        return Lock()

    def flock(self, lock, mode):
        assert mode == 2
        self.event('lock_' + lock.lock_kind, mode)

    def clock(self):
        self.clocks += 1
        self.event('clock_start' if self.clocks == 1 else 'clock_elapsed')
        return (100.0, 102.6)[self.clocks - 1]

    def strftime(self, fmt):
        self.event('strftime', fmt)
        return '12:34:56'

    def run(self, argv, **kwargs):
        if argv[0] == 'git':
            assert argv[:3] == ['git', '-C', ROOT]
            operation = argv[3:]
            stages = {('rev-parse', 'refs/remotes/origin/splatter4d'): 'origin_run',
                      ('rev-parse', 'HEAD'): 'head_run', ('rev-parse', '--short', 'HEAD'): 'short_head_run'}
            stage = 'merge_run' if operation[:1] == ['merge'] else stages[tuple(operation)]
        else:
            assert argv[:3] == [ROOT + '/.venv/bin/python', '-I',
                                ROOT + '/runs/setup/tools/adopt_remote_suite_evidence.py']
            stage = 'adopt_run'
        self.event(stage, argv, kwargs)
        stdout, stderr, code = '', '', 0
        if stage == 'origin_run':
            stdout = ('wrong-origin' if self.case == 'bad_origin' else self.commit) + '\n'
        elif stage == 'head_run':
            stdout = ('wrong-head' if self.case == 'bad_head' else self.current_head) + '\n'
        elif stage == 'merge_run':
            assert argv[3:] == ['merge', '--ff-only', '-q', self.commit]
            self.current_head = self.commit
        elif stage == 'short_head_run':
            stdout = self.commit[:7] + '\n'
            code = 17 if self.case == 'short_head_nonzero' else 0
        elif stage == 'adopt_run':
            stdout, stderr = 'discarded stdout:' + 'o' * 2000, 'discarded stderr:' + 'e' * 2000
            if self.case in ('hold_changed', 'adopt_and_changed_hold'):
                self.files[HOLD] = 'changed during operation\n'
            if self.case == 'hold_removed':
                self.files.pop(HOLD, None)
            code = 23 if self.case in ('adopt_nonzero', 'adopt_and_changed_hold') else 0
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)

    def print(self, *args, **kwargs):
        stage = {'merged': 'print_merged', 'adopted at': 'print_adopted',
                 'temporary hold removed': 'print_removed',
                 'pre-existing role hold kept unchanged': 'print_kept'}.get(args[0], 'print_output')
        if str(args[0]).startswith('locks acquired'):
            stage = 'print_locks'
        self.event(stage, list(args), kwargs)
        builtins.print(*args, **kwargs)

    def execute(self, source, run_name='__main__'):
        # Rebind only the operational root before any top-level command can run.
        tree = ast.parse(source)
        roots = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in n.targets)]
        assert len(roots) == 1
        assert ast.unparse(roots[0].value) == "Path('/home/ws/ws/splatter4d')"
        roots[0].value = ast.Name(id='_fixture_root', ctx=ast.Load())
        ast.fix_missing_locations(tree)
        argv = [self.script, 'verify-fixture']
        if self.case == 'missing_argument':
            argv.pop()
        elif self.case == 'extra_argument':
            argv += ['ignored', 'also ignored']
        elif self.case == 'help_is_positional':
            argv[1] = '--help'
        fake_sys = SimpleNamespace(argv=argv, path=list(sys.path))
        modules = {'fcntl': SimpleNamespace(flock=self.flock, LOCK_EX=2),
                   'subprocess': SimpleNamespace(run=self.run), 'sys': fake_sys,
                   'time': SimpleNamespace(time=self.clock, strftime=self.strftime),
                   'pathlib': SimpleNamespace(Path=Path)}

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            assert level == 0
            if name == 'importlib' and fromlist == ('util',):
                return SimpleNamespace(util=importlib.util)
            assert name in modules, 'unintended command import: ' + name
            return modules[name]

        safe_builtins = {**vars(builtins), '__import__': safe_import, 'open': self.open, 'print': self.print}
        namespace = {'__builtins__': safe_builtins, '__name__': run_name,
                     '__file__': str(REPO / 'runs/setup/tools' / self.script), '_fixture_root': self.path(ROOT)}
        out, err = io.StringIO(), io.StringIO()
        original_path = list(sys.path)
        error = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), _no_operational_io():
            try:
                exec(compile(tree, self.script, 'exec'), namespace)
            except (Exception, SystemExit) as exc:
                error = _exception_record(exc)
        assert sys.path == original_path == fake_sys.path
        if '_impl' in namespace:
            assert namespace['_impl'].__file__ == str(REPO / 'ops/jobs/_release_adoption.py')
        return {'events': self.events, 'files': self.files, 'head': self.current_head,
                'stdout': out.getvalue(), 'stderr': err.getvalue(), 'exception': error,
                'exit_code': 0 if error is None else 1}


def _command_source(script):
    relative = 'runs/setup/tools/' + script
    path = REPO / relative
    if path.is_file():
        return path.read_text()
    # Release containers hide tracked runs/ records behind their runtime mount.
    # Read the immutable candidate blob; never import or execute that command.
    assert Path('/.dockerenv').exists(), f'{path} is missing'
    source = subprocess.run(['git', '-C', str(REPO), 'show', 'HEAD:' + relative],
                            capture_output=True, text=True, check=True).stdout
    assert 'ops/jobs/_release_adoption.py' in source, 'committed candidate wrapper is required'
    return source


def replay(script, case, hold, *, baseline=False, run_name='__main__'):
    source = (FIXTURES / (script + '.txt')).read_text() if baseline else _command_source(script)
    return Replay(script, case, hold).execute(source, run_name)


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('hold', HOLDS, ids=('no-hold', 'empty-hold', 'role-hold'))
@pytest.mark.parametrize('case', CASES)
def test_historical_command_side_effects_match(script, hold, case):
    expected = replay(script, case, hold, baseline=True)
    assert replay(script, case, hold) == expected
    assert expected['files'][ROOT + '/docs/tests.json'] == 'pre-existing evidence\n'
    assert expected['files'][ROOT + '/experiments/registry.jsonl'] == 'pre-existing state\n'
    if case in ('success', 'extra_argument', 'help_is_positional', 'short_head_nonzero'):
        assert expected['exception'] is None
        assert expected['files'].get(HOLD) == hold
        assert 'discarded stdout' not in expected['stdout'] and 'discarded stderr' not in expected['stdout']
    if case == 'adopt_nonzero':
        assert expected['exception']['type'] == 'SystemExit'


def surface(source):
    """Adapt the previous capture.py AST/signature logic; never import a command."""
    tree = ast.parse(source)
    imports, definitions, constants = {}, {}, {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                public = alias.asname or alias.name
                if not public.startswith('_'):
                    imports[public] = (node.module if isinstance(node, ast.ImportFrom) else None, alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('ROOT', 'COMMIT_PREFIX', 'verify'):
                constants[name] = ast.unparse(node.value)
    return {'docstring': ast.get_docstring(tree, clean=False), 'imports': imports,
            'definitions': definitions, 'constants': constants}


@pytest.mark.parametrize('script', SCRIPTS)
def test_command_declared_surface_and_docstring_are_preserved(script, monkeypatch):
    source = _command_source(script)
    assert surface(source) == surface((FIXTURES / (script + '.txt')).read_text())
    path = REPO / 'runs/setup/tools' / script
    is_file, exists = Path.is_file, Path.exists

    def git_blob(argv, **kwargs):
        assert argv == ['git', '-C', str(REPO), 'show', 'HEAD:runs/setup/tools/' + script]
        assert kwargs == {'capture_output': True, 'text': True, 'check': True}
        return SimpleNamespace(stdout=source)

    monkeypatch.setattr(Path, 'is_file', lambda p: False if p == path else is_file(p))
    monkeypatch.setattr(Path, 'exists', lambda p: True if p == Path('/.dockerenv') else exists(p))
    monkeypatch.setattr(subprocess, 'run', git_blob)
    assert _command_source(script) == source


def test_helper_import_is_inert_and_standard_library_only():
    path = REPO / 'ops/jobs/_release_adoption.py'
    tree = ast.parse(path.read_text())
    assert all(isinstance(n, (ast.Expr, ast.FunctionDef)) for n in tree.body)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    before_path, before_modules = list(sys.path), set(sys.modules)
    spec = importlib.util.spec_from_file_location('_release_adoption_import_test', path)
    module = importlib.util.module_from_spec(spec)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        spec.loader.exec_module(module)
    assert (out.getvalue(), err.getvalue()) == ('', '')
    assert sys.path == before_path and set(sys.modules) == before_modules
    assert str(inspect.signature(module.merge_adopt)) == (
        '(root, expected_head_prefix, commit_prefix, verify, hold_text, *, subprocess, fcntl, time, open, print)')


def test_replay_rejects_real_filesystem_access_before_it_happens():
    source = (FIXTURES / (SCRIPTS[0] + '.txt')).read_text()
    with pytest.raises(_SandboxEscape, match='unintended filesystem access'):
        Replay(SCRIPTS[0], 'success', None).execute(source + "\nPath('/tmp/p2-unintended-write').write_text('escape')\n")


@pytest.mark.parametrize('script', SCRIPTS)
@pytest.mark.parametrize('mode', ('isolated', 'module', 'legacy'))
def test_checkout_helper_lookup_without_sys_path_changes(script, mode):
    # Execute the sandbox under a real -I interpreter with deliberately poisoned
    # import search state. The command must still load its checkout's helper.
    code = '''
import importlib.util
import sys
from types import SimpleNamespace
test_file, script, mode = sys.argv[1:]
spec = importlib.util.spec_from_file_location('replay_test', test_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.insert(0, '/fixture/main-checkout')
poison = SimpleNamespace(merge_adopt=lambda *args, **kw: (_ for _ in ()).throw(AssertionError('wrong helper')))
sys.modules['ops.jobs._release_adoption'] = poison
sys.modules['_release_adoption'] = poison
name = {'isolated': '__main__', 'module': 'runs.setup.tools.' + script[:-3], 'legacy': script[:-3]}[mode]
before = list(sys.path)
expected = module.replay(script, 'success', None, baseline=True, run_name=name)
actual = module.replay(script, 'success', None, run_name=name)
assert actual == expected and actual['exception'] is None
assert sys.path == before
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(Path(__file__).resolve()), script, mode],
                            capture_output=True, text=True, timeout=30)
    assert (result.returncode, result.stdout, result.stderr) == (0, '', '')
