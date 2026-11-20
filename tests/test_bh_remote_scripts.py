"""BH shipped entrypoints and unchanged two-wave science; CPU subprocesses only."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENE = dict(lookat=[.4, 0, .1], pose_scale=2.5, bounds=[-.5, -.6, -.05, .9, .6, .7])


def child(source):
    result = subprocess.run(
        [sys.executable, '-B', '-I', '-c', 'import sys;sys.path.insert(0,' + repr(str(ROOT)) + ');' + source],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


@pytest.fixture(scope='module')
def graph():
    return child('''
from s4d.rwlearn.ball_hiding import bootstrap
bootstrap()
from s4d.rwlearn.pnpv2_spec import SPEC,load_spec
from ops.jobs.pnpv2_plan import build_rows
from s4d.rwlearn.bh_training import rows
import json
spec=load_spec(SPEC)
release=dict(commit='a'*40,image='sha256:'+'b'*64,source='local')
scene=''' + repr(SCENE) + '''
print(json.dumps(dict(original=build_rows(spec,release,waves=('rgb','geometry'),scene=scene),
    adapted=rows(spec,release,waves=('rgb','geometry'),scene=scene))))
''')


def test_remote_entrypoint_relocation_preserves_science_and_local_rows(graph):
    original, adapted = graph['original'], graph['adapted']
    assert len(original) == len(adapted) == 90
    remote = [r for r in adapted if r['host'] == 'remote']
    assert sum(r['pnpv2_wave'] == 'rgb' for r in remote) == 16
    assert sum(r['pnpv2_wave'] == 'geometry' for r in remote) == 26
    for before, after in zip(original, adapted):
        expected = {**before, 'registration_tool': 'runs/setup/tools/bh_register.py',
                    'release_source': 'local', 'evidence_manifest': after['evidence_manifest'],
                    'priority': after['priority'], 'host': after['host'], 'pin_host': after['pin_host'],
                    'script': before['script'].replace('scripts/pnpv2_',
                        'scripts/bh_' if after['host'] == 'remote' else 'runs/setup/tools/bh_')}
        assert expected == after
    for row in remote:
        assert row['script'].startswith('scripts/bh_')
        path = ROOT / row['script']
        assert path.is_file()
        source = path.read_text()
        assert source.index('guard_gpus()') < source.index('bootstrap()') < source.index('from s4d.rwlearn.pnpv2_entry')
        assert 'runs.setup' not in source
        assert 'Path(__file__).resolve().parent)' in source
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run']
        if calls:
            target = calls[0].args[0].value
            assert (ROOT / (target.replace('.', '/') + '.py')).is_file()


def test_every_remote_row_passes_real_launcher_schema(graph):
    from ops.containers.launch import validate_spec
    from ops.jobs.remote_backend import container_labels
    from ops.remote.access import REMOTE_ROOT
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS
    for row in graph['adapted']:
        if row['host'] != 'remote':
            continue
        spec = dict(version=1, id=row['id'], host='remote', attempt=1,
                    container='s4d-' + row['id'] + '-a1', gpu=APPROVED_HOST_GPUS['remote'][0],
                    gpu_backend='cdi', commit=row['planned_commit'], image_digest=row['planned_image'],
                    root=REMOTE_ROOT, host_config='configs/hosts/remote.yaml', script=row['script'], args=row['args'],
                    role=row['role'], ram_gb=row['ram_gb'], disk_gb=row['disk_gb'], mem_gb=row['mem_gb'],
                    max_jobs=4, max_mem_gb=21.5, disk_floor_gb=0, disk_floor_fraction=.15,
                    host_ram_reserve_gb=120, host_ram_ramp_minutes=10, evidence={},
                    required_results=row['required_results'],
                    labels=container_labels(row['id'], 1, row['planned_commit'], row['planned_image']))
        assert validate_spec(spec)['script'] == row['script']
        with pytest.raises(ValueError, match='Entry script must be a repository Python script'):
            validate_spec({**spec, 'script': row['script'].replace('scripts/bh_', 'runs/setup/tools/bh_')})
