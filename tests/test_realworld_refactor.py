"""Compatibility and serialization invariants for the real-world module boundaries."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from ops.inference_server import command_support, run_episode
from s4d.realworld import depth_stages, workers


@pytest.mark.parametrize('response,allow_mock,expected', [
    ({'protocol': 'wrong', 'error': 'bad'}, False, "mock: protocol 'wrong' != 's4d.realworld.worker/1'"),
    ({'protocol': workers.PROTOCOL, 'ok': False, 'error': 'bad'}, False, 'mock: bad'),
    ({'protocol': workers.PROTOCOL, 'ok': True, 'model': {'is_mock': True}}, False,
     'mock: mock output refused (allow_mock=False)'),
    ({'protocol': workers.PROTOCOL, 'ok': True, 'model': {'is_mock': True}}, True,
     'mock: upstream commit None != pinned pinned'),
])
def test_shared_response_validation_preserves_rejection_order(response, allow_mock, expected):
    spec = workers.WorkerSpec('mock', sys.executable, expected_commit='pinned')
    before = json.dumps(response)
    with pytest.raises(workers.WorkerError) as caught:
        workers.validate_response(spec, response, allow_mock)
    assert str(caught.value) == expected
    assert json.dumps(response) == before


def test_command_helpers_preserve_remote_worker_fields_and_pool_projection():
    cfg = {'workers': {'mock': {'python': 'python', 'script': 'ignored', 'env': {'ignored': 'value'},
                               'timeout_s': '7', 'checkpoint': 'weights', 'expected_commit': 'old'}}}
    spec = run_episode.spec(cfg, 'mock', python='override', expected_commit='new')
    assert workers.asdict(spec) == {'name': 'mock', 'python': 'override', 'script': None, 'repo_dir': None,
                                   'expected_commit': 'new', 'checkpoint': 'weights', 'env': {}, 'timeout_s': 7.0}
    keys = ('ssh', 'container', 'workdir', 'source_root', 'retries', 'backoff_s', 'call_timeout_s', 'hello_timeout_s',
            'workers', 'env')
    config = {k: [] for k in (*keys, 'extra')}
    projected = command_support.pool_config(config, 'head')
    assert list(projected) == [*keys, 'expected_source_commit']
    assert all(projected[k] is config[k] for k in keys)


def test_local_head_keeps_command_globals(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout='  pinned\n' if argv[-1] == 'HEAD' else '')

    monkeypatch.setattr(run_episode, 'REPO', tmp_path)
    monkeypatch.setattr(run_episode, 'subprocess', SimpleNamespace(run=run))
    assert run_episode.local_head() == 'pinned'
    assert [argv for argv, _ in calls] == [
        ['git', '-C', str(tmp_path), 'status', '--porcelain', '--untracked-files=no'],
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'],
    ]
    assert all(kwargs == {'capture_output': True, 'text': True, 'check': True} for _, kwargs in calls)
    monkeypatch.setattr(run_episode.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='dirty'))
    with pytest.raises(SystemExit) as caught:
        run_episode.local_head()
    assert str(caught.value) == f'local checkout {tmp_path} has uncommitted changes; commit before a remote run'


def test_prediction_helpers_keep_schema_and_indexing(tmp_path):
    with h5py.File(tmp_path / 'prediction.h5', 'w', track_order=True) as f:
        depth_stages._create_prediction_planes(f, 3, (4, 5))
        depth_stages._create_prediction_pose(f, 3)
        assert list(f) == ['depth', 'valid', 'conf', 'non_ambiguous', 'cam2world_pred', 'metric_scale']
        for name, dtype in (('depth', np.float32), ('valid', bool), ('conf', np.float16), ('non_ambiguous', bool)):
            ds = f[name]
            assert (ds.shape, ds.dtype, ds.chunks, ds.compression) == ((3, 4, 5), dtype, (1, 4, 5), 'lzf')
        for name, shape in (('cam2world_pred', (3, 4, 4)), ('metric_scale', (3,))):
            ds = f[name]
            assert (ds.shape, ds.dtype, ds.chunks, ds.compression) == (shape, np.float32, None, None)
        out = {name: np.arange(2 * 2 * np.prod(ds.shape[1:] or (1,))).reshape(2, 2, *ds.shape[1:])
               for name, ds in f.items()}
        depth_stages._write_prediction(f, 2, out, 1, 0, tuple(out))
        for name in f:
            np.testing.assert_array_equal(f[name][2], out[name][1, 0].astype(f[name].dtype))


@pytest.mark.parametrize('mask_state', ['none', 'all', 'partial'])
def test_shared_depth_evidence_preserves_partial_mask_fallback(mask_state):
    from s4d.realworld import calibration as cal

    frames = [SimpleNamespace(depth=np.full((3, 4), float(i + 1)), robot_mask=None) for i in range(3)]
    for i, frame in enumerate(frames):
        if mask_state == 'all' or (mask_state == 'partial' and i == 0):
            frame.robot_mask = np.arange(12).reshape(3, 4) % 2 == 0
    no_depth = SimpleNamespace(depth=None, robot_mask=None)
    usable, masked = cal._depth_evidence_frames([no_depth, *frames])
    assert usable == frames and masked == (mask_state == 'all')
    cfg = cal.CalibrationConfig()
    expected = (np.stack([f.robot_mask & (f.depth > 0) for f in frames]) if masked else
                cal.moving_masks(np.stack([f.depth for f in frames]), cfg.motion_threshold_m))
    np.testing.assert_array_equal(cal._depth_evidence_masks(usable, masked, cfg), expected)


def test_calibration_fixture_legacy_exports_share_objects():
    import realworld_calibration_fixtures as fixtures
    import test_realworld_calibration as legacy

    for name in ('make_episode', 'HW', 'K', 'T_TRUE', 'JOINTS', 'LANDMARKS', 'CalibrationFrame', 'RobotSurface',
                 'landmarks', 'look_at_T_cam_base', 'project', 'splat_depth'):
        assert getattr(legacy, name) is getattr(fixtures, name)
    assert legacy.make_episode.__module__ == legacy.__name__
    assert pickle.loads(pickle.dumps(legacy.make_episode)) is legacy.make_episode


@pytest.mark.parametrize('command', ['run_episode', 'contract_checks', 'depth_report'])
@pytest.mark.parametrize('help_only', [False, True])
def test_commands_keep_initialization_without_scipy(command, help_only):
    code = '''
import importlib
import importlib.abc
import runpy
import sys
root, command, help_only = sys.argv[1:]
sys.path.insert(0, root)

class NoScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoScipy())
before = sys.path.count(root)
if help_only == 'True':
    sys.argv = [root + '/ops/inference_server/' + command + '.py', '--help']
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except SystemExit as exc:
        assert exc.code == 0
else:
    module = importlib.import_module('ops.inference_server.' + command)
    assert module.local_head.__module__ == 'ops.inference_server.run_episode'
assert sys.path.count(root) - before == (1 if command == 'run_episode' else 2)
assert 'ops.inference_server.run_episode' in sys.modules or command == 'run_episode'
assert 's4d.realworld.cli' in sys.modules
assert 's4d.realworld.diagnostics' not in sys.modules
if command == 'depth_report' and help_only == 'False':
    assert module._med([None, 'nan', 1, 3]) == 2
    grids = {key: module.np.ones((2, 1, 3), dtype=bool if key == 'static' else float)
             for key in ('raw', 'ma', 'static')}
    result = module.aggregate([], grids, ['cam'], [])
    assert result['static_stability']['cam']['pixels'] == 3
assert 'scipy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(run_episode.REPO), command, str(help_only)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith('usage: ') if help_only else result.stdout == ''


@pytest.mark.parametrize('semantics', ['robotiq_position_register', 'normalized_closure'])
def test_composed_manifest_preserves_description_and_semantics(tmp_path, semantics):
    from dataclasses import asdict

    from s4d.realworld.adapters._robot_manifest import _composed_robot

    urdf = tmp_path / 'robot.urdf'
    mount = np.eye(4)
    mount[:3, 3] = [0.01, 0.02, 0.03]
    manifest = {'mount': {'gripper_root': 'gripper', 'T_parent_gripper': mount.tolist(), 'source': 'measured'},
                'urdf_sha256': 'digest', 'provenance': {'arm': 'fixture'}}
    urdf.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    rc = {'urdf_path': str(urdf), 'base_frame': 'base', 'flange_frame': 'flange',
          'joint_names': ['j1', 'j2'], 'gripper_joint': 'finger'}
    description, decoded = _composed_robot(rc, gripper_state_semantics=semantics)
    assert decoded == manifest
    doc = asdict(description)
    assert doc == {'arm_model': 'franka_panda', 'gripper_model': 'robotiq_2f_85', 'urdf_path': str(urdf),
                   'mesh_root': None, 'base_frame': 'base', 'flange_frame': 'flange', 'gripper_base_frame': 'gripper',
                   'T_flange_gripper': tuple(map(tuple, mount)), 'mount_source': 'measured',
                   'joint_names': ('j1', 'j2'), 'gripper_state_semantics': semantics, 'gripper_joint': 'finger',
                   'asset_sha256': {'urdf': 'digest'}}
    urdf.with_suffix('.manifest.json').unlink()
    with pytest.raises(FileNotFoundError) as caught:
        _composed_robot(rc, gripper_state_semantics=semantics)
    assert caught.value.filename == str(urdf.with_suffix('.manifest.json'))
