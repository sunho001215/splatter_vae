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
