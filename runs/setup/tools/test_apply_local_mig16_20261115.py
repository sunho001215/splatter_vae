"""Queue tool tests stay beside the operational tool; all mutations use tmp_path."""
import importlib.util
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location('mig16_tool', Path(__file__).with_name('apply_local_mig16_20261115.py'))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.fixture(params=["nested", "legacy"])
def root(tmp_path, request):
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'runs/setup').mkdir(parents=True)
    (tmp_path / 'experiments/queue.yaml').write_text(
        '# keep\nhosts:\n  local:\n    limits:\n      old: {max_jobs: 3, max_mem_gb: 22}\n'
        '  remote:\n    enabled: false\njobs:\n  - {id: untouched, args: [literal]}\n')
    if request.param == 'legacy':
        queue = tmp_path / 'experiments/queue.yaml'
        queue.write_text('# keep\nlimits:\n  old: {max_jobs: 3, max_mem_gb: 22}\n'
                         'host_ram_reserve_gb: 60\nhosts:\n  remote:\n    enabled: false\n'
                         'jobs:\n  - {id: untouched, args: [literal]}\n')
    return tmp_path


def local_limits(data):
    return data['limits'] if 'limits' in data else data['hosts']['local']['limits']


def test_dry_apply_idempotence_and_rollback(root):
    queue = root / 'experiments/queue.yaml'
    before = queue.read_bytes()
    assert tool.update(root)['changed']
    assert queue.read_bytes() == before
    result = tool.update(root, apply=True)
    assert Path(result['backup']).read_bytes() == before
    after = queue.read_bytes()
    assert after[after.index(b'  remote:'):] == before[before.index(b'  remote:'):]
    assert len(local_limits(yaml.safe_load(after))) == 9
    assert not tool.update(root, apply=True)['changed']
    tool.update(root, rollback=True, live=lambda _: [])
    assert yaml.safe_load(queue.read_bytes()) == yaml.safe_load(before)
    assert not tool.update(root, rollback=True, live=lambda _: [])['changed']
    assert len(list(queue.parent.glob('*.before-local-mig16-*'))) == 2


def test_live_rollback_refused_before_write(root):
    tool.update(root, apply=True)
    queue = root / 'experiments/queue.yaml'
    before = queue.read_bytes()
    with pytest.raises(ValueError, match='live.*|GPU 6/7'):
        tool.update(root, rollback=True, live=lambda _: [{'id': 'busy', 'gpu': tool.NEW_MIGS[0]}])
    assert queue.read_bytes() == before


def test_conflicting_limit_is_never_overwritten(root):
    tool.update(root, apply=True)
    queue = root / 'experiments/queue.yaml'
    data = yaml.safe_load(queue.read_text())
    local_limits(data)[tool.NEW_MIGS[0]]['max_jobs'] = 1
    queue.write_text(yaml.safe_dump(data))
    before = queue.read_bytes()
    with pytest.raises(ValueError, match='conflicting'):
        tool.update(root, apply=True)
    assert queue.read_bytes() == before


def test_ram400_and_one_step_rollback(root, monkeypatch):
    from ops.jobs import realworld_local, rwprep_mig
    from ops.resources import memory_budget as mb
    from ops.resources import memory_watch as watch
    queue = root / 'experiments/queue.yaml'
    # Start from the real original eight-MIG configuration, with all rows and remote fields preserved.
    queue.write_bytes(queue.read_bytes().replace(b'old: {max_jobs: 3, max_mem_gb: 22}',
        ('\n' + '\n'.join('      ' + gpu + ': {max_jobs: 3, max_mem_gb: 22}'
                           for gpu in tool.APPROVED_HOST_GPUS['local'][:8])).encode()))
    before = yaml.safe_load(queue.read_bytes())
    control = root / 'runs/setup/local-memory-budget.json'
    monkeypatch.setattr(mb, 'CONTROL', control)
    monkeypatch.setattr(mb, 'snapshot', lambda: {'budget_usage': 0})
    monkeypatch.setattr(mb, 'session_nonfile_pss', lambda: {})
    assert mb.CEILING == 150_000_000_000
    tool.update(root)
    assert not control.exists()
    for mode, ceiling, slots in [('apply_ram400', 400_000_000_000, 16),
                                 ('rollback', 150_000_000_000, 8)]:
        result = tool.update(root, **{mode: True}, live=lambda _: [])
        assert result['ceiling_bytes'] == mb.CEILING == ceiling
        assert mb.TRAINING_CEILING == ceiling - 12_000_000_000
        assert realworld_local._capacity({'jobs': []}, {})['ram_free_bytes'] == mb.TRAINING_CEILING
        after = yaml.safe_load(queue.read_bytes())
        assert len(local_limits(after)) == slots
        normalized = {'hosts': {'local': {'limits': local_limits(after)}}}
        assert len(rwprep_mig.eligible_devices(dict.fromkeys(tool.APPROVED_HOST_GPUS['local']),
                                              normalized, 'local')) == slots
        assert after['jobs'] == before['jobs'] and after['hosts']['remote'] == before['hosts']['remote']
        assert not tool.update(root, **{mode: True}, live=lambda _: [])['changed']
    assert yaml.safe_load(queue.read_bytes()) == before
    assert watch.mb is mb


def test_ram400_live_rollback_refusal_preserves_both_files(root):
    tool.update(root, apply_ram400=True)
    paths = [root / 'experiments/queue.yaml', root / 'runs/setup/local-memory-budget.json']
    before = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError, match='rollback refused'):
        tool.update(root, rollback=True, live=lambda _: [{'id': 'busy'}])
    assert [p.read_bytes() for p in paths] == before


def test_ram400_cli_and_shell_rollback(root):
    import json
    import subprocess
    import sys
    script = Path(tool.__file__)
    # Isolate process discovery: the suite itself runs on GPU 6/7 and must not permit a real rollback.
    code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('tool', sys.argv.pop(1)); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "m.update.__kwdefaults__['live'] = lambda _: []; m.main()")
    for flag, ceiling in [('--apply-ram400', 400_000_000_000), ('--rollback', 150_000_000_000)]:
        result = subprocess.run([sys.executable, '-c', code, str(script), '--root', str(root), flag],
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)['ceiling_bytes'] == ceiling
    subprocess.run(['bash', '-n', str(script.with_name('rollback_local_mig16_20261115.sh'))], check=True)
