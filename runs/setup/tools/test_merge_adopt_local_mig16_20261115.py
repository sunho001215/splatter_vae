"""Evidence validation uses synthetic reports; never merge or adopt in tests."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('adopt16', Path(__file__).with_name('merge_adopt_local_mig16_20261115.py'))
adopt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adopt)


@pytest.fixture
def report(tmp_path, monkeypatch):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_one.py').touch()
    monkeypatch.setattr(adopt, 'git', lambda root, *args: 'a' * 40 if args[0] == 'rev-parse' else '')
    data = dict(full_suite_passed=True, exit_code=0, host='local', source_changed_during_suite=False,
                counts=dict(passed=1, failed=0, error=0, skipped=0), pytest_args=[['tests/test_one.py']],
                gpu_mapping=[dict(uuid=adopt.APPROVED_HOST_GPUS['local'][-1])])
    def write():
        (tmp_path / 'docs/tests.json').write_text(json.dumps(data))
        return adopt.verify(tmp_path, 'a' * 40)
    return data, write


def test_candidate_gpu_and_complete_shards(report):
    data, write = report
    assert write() == data
    data['pytest_args'] = [['tests/test_missing.py']]
    with pytest.raises(ValueError, match='exactly'):
        write()


@pytest.mark.parametrize('key,value', [('full_suite_passed', False), ('exit_code', 1),
                                      ('host', 'remote'), ('source_changed_during_suite', True)])
def test_incomplete_or_changed_evidence_refused(report, key, value):
    data, write = report
    data[key] = value
    with pytest.raises(ValueError):
        write()


def test_unauthorized_device_refused(report):
    data, write = report
    data['gpu_mapping'][0]['uuid'] = 'GPU-unauthorized'
    with pytest.raises(ValueError, match='authorized'):
        write()
