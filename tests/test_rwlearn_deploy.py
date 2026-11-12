"""CPU-only FIXTURES, never trained performance evidence (``s4d.rwlearn.deploy.fixtures``). Baseline fixtures use
their real encoder adapters; no checkpoint-existence skips: every condition creates synthetic weights. Real-checkpoint
equality and recorded-episode preprocessing evidence live in scripts/check_realworld_deploy.py.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from s4d.data.droid.largescale.crop import center_box, crop_intrinsics, crop_rgb
from s4d.realworld.frame_preprocessing import nearest_resample, training_depth
from s4d.rwlearn import actions as A
from s4d.rwlearn.deploy import (
    BindingRefused,
    ContractViolation,
    DeployRefusal,
    Frame,
    HistoryWindow,
    IncompleteHistory,
    InputSpec,
    IrregularHistory,
    ObservationWindow,
    SessionMismatch,
    Unsynchronized,
    preprocess_frame,
)
from s4d.rwlearn.deploy import contracts as C
from s4d.rwlearn.deploy.fixtures import (
    FIXTURE_CONDITIONS,
    make_fixture_policy,
    session_choices,
    synthetic_frames,
    synthetic_window,
)
from s4d.rwlearn.deploy.realsense import RealSenseRGBD, align_recorded_depth, native_correction


def binding(window, ref="measured-pose-42"):
    t = window.observation_time_ns
    return window.binding(ref, pose_time_ns=t - 3_000_000, gripper_time_ns=t - 40_000_000)


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("deploy-fixtures")
    yield {c: make_fixture_policy(root / c, c) for c in FIXTURE_CONDITIONS}
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize("condition", FIXTURE_CONDITIONS)
def test_fixture_infer_equals_offline_and_binds_action(policies, condition):
    policy = policies[condition]
    spec = policy.reset_session(**session_choices(policy, "equality"))
    assert C.InputSpec.from_json(json.loads(json.dumps(spec.to_json(), allow_nan=False))) == spec
    assert C.InputSpec.from_json(policy.input_spec.to_json()) == policy.input_spec
    window = synthetic_window(spec)
    pose = np.array([0.2, -0.1, 0.4, 0.1, 0.2, 0.3, 0.9])
    noise = torch.linspace(-1, 1, 7)[None]
    ids = binding(window)
    obs = window.observation()
    if condition == "e2e_cnn":
        expected = policy.offline.act(obs, pose, 107.5, noise=noise)
    else:  # training precision: the FQL dataset stores float16 features (fql_data.encode_timeline)
        feat = torch.from_numpy(policy.offline.features(obs).numpy().astype(np.float16).astype(np.float32))
        expected = policy.offline.act_from_features(feat, A.measured_proprio(pose, 107.5), pose, noise=noise)
        unrounded = policy.offline.act_from_features(policy.offline.features(obs), A.measured_proprio(pose, 107.5),
                                                     pose, noise=noise)
        assert np.abs(unrounded["action_norm"] - expected["action_norm"]).max() < 1e-2
        offline = policy.offline.act(obs, pose, 107.5, noise=noise)  # rounds like training (DIAG 2026-11-11)
        assert all(np.array_equal(np.asarray(offline[k]), np.asarray(expected[k])) for k in expected)
    actual = policy.infer(window, pose, 107.5, ids, noise=noise.numpy().astype(np.float64))
    for k in expected:
        np.testing.assert_array_equal(actual[k], expected[k])
    assert actual["metadata"]["feature_precision"] == (None if condition == "e2e_cnn" else "float16")
    np.testing.assert_array_equal(actual["measured_pose7"], pose)
    assert actual["obs_ids"] == ids and actual["metadata"]["fixture"] and not actual["metadata"]["padded"]
    assert {"flange_pose", "delta_phys", "gripper_closure", "gripper_register", "action_norm",
            "execution_flange_pose"} <= set(actual)
    assert actual["timings"]["encoder_ms"] > 0 and actual["timings"]["fql_ms"] > 0
    ids["reference_id"] = "changed"
    assert actual["obs_ids"]["reference_id"] == "measured-pose-42"
    assert "encoder_ms" in policy.warm_up(window, iterations=1) and "fql_ms" in policy.warmup()
    policy.reset_session(**session_choices(policy, "after-reset"))
    with pytest.raises(ValueError, match="session"):
        policy.infer(window, pose, 100, binding(window), noise=noise)


def test_spec_constants_are_the_training_rules():
    assert inspect.signature(A.step_table).parameters["max_state_age_ns"].default == C.STATE_MAX_AGE_NS
    assert A.MAX_GRIPPER_AGE_NS == C.GRIPPER_MAX_AGE_NS
    from s4d.realworld.stages.depth import sensor_depth
    from s4d.rwlearn.store import runs_from

    p = inspect.signature(sensor_depth).parameters
    assert (p["min_depth_m"].default, p["max_depth_m"].default) == C.SENSOR_DEPTH_RANGE_M
    assert inspect.signature(runs_from).parameters["max_gap_factor"].default == C.RUN_GAP_FACTOR
    assert InputSpec.timestamp_tolerance_fraction == A.PERIOD_TOLERANCE
    from s4d.rwlearn import fql_data
    from s4d.rwlearn.deploy.policy import TRAINING_FEATURE_DTYPE

    assert ".to(torch.float16)" in inspect.getsource(fql_data.encode_timeline)
    assert TRAINING_FEATURE_DTYPE == torch.float16 and C.FEATURE_PRECISION == "float16"


def test_contract_missing_inputs_and_bad_bindings(policies):
    policy = policies["mvp3d_single"]
    choices = session_choices(policy, "contracts")
    for field in ("camera_id", "calibration", "native_depth_correction", "native_hw"):
        bad = dict(choices)
        bad.pop(field)
        with pytest.raises((ValueError, TypeError)):
            policy.reset_session(**bad)
    bad = copy.deepcopy(choices)
    bad["native_depth_correction"]["cam_a"]["stage"] = "foundation_stereo_depth"
    with pytest.raises(ValueError, match="sensor_depth"):
        policy.reset_session(**bad)
    bad = copy.deepcopy(choices)
    bad["native_depth_correction"]["cam_a"].update(factor=1.02, validated=False)
    with pytest.raises(ValueError, match="evidence"):
        policy.reset_session(**bad)
    rgb_only = policies["dynamo"]
    for extra in ({"calibration": choices["calibration"]},
                  {"native_depth_correction": choices["native_depth_correction"]}):
        with pytest.raises(ValueError, match="does not consume"):
            rgb_only.reset_session(**session_choices(rgb_only, "extra"), **extra)
    spec = policy.reset_session(**choices)
    window = synthetic_window(spec)
    ids = binding(window, 1)
    ids["frame_ids"]["cam_a"][-1] += 1
    with pytest.raises(ValueError, match="IDs"):
        policy.infer(window, [0, 0, 0, 0, 0, 0, 1], 3, ids)
    t = window.observation_time_ns
    for pose_t, grip_t in ((t - 16_000_000, t), (t + 1, t), (t, t - 101_000_000)):
        with pytest.raises(ValueError, match="sample"):
            policy.infer(window, [0, 0, 0, 0, 0, 0, 1], 3,
                         window.binding(1, pose_time_ns=pose_t, gripper_time_ns=grip_t))
    with pytest.raises(ValueError, match="pose_time_ns"):
        policy.infer(window, [0, 0, 0, 0, 0, 0, 1], 3, {**binding(window, 1), "pose_time_ns": None})
    fs = list(window.frames["cam_a"])
    fs[0] = Frame(fs[0].seq, fs[0].timestamp_ns, fs[0].rgb)
    with pytest.raises(ValueError, match="depth"):
        ObservationWindow(spec, {"cam_a": tuple(fs)}).observation()
    with pytest.raises(ValueError, match="trained camera"):
        policy.input_spec.bind(**{**choices, "camera_id": "unknown"})
    rgb_spec = rgb_only.reset_session(**session_choices(rgb_only, "rgb-depth"))
    f = synthetic_frames(spec, 1)["cam_a"][0]
    with pytest.raises(ValueError, match="RGB-only"):
        HistoryWindow(rgb_spec).push("cam_a", Frame(0, 0, np.zeros((32, 32, 3), np.uint8), f.depth, f.valid))


def test_history_refuses_missing_frames_irregular_time_and_m5(policies):
    policy = policies["snerl"]
    spec = policy.reset_session(**session_choices(policy, "history"))
    window = synthetic_window(spec)
    h = HistoryWindow(spec)
    for c, frames in window.frames.items():
        for f in frames[1:]:
            h.push(c, f)
    with pytest.raises(ValueError, match="incomplete"):
        h.assemble(dict.fromkeys(spec.camera_ids, 2))
    h.reset()
    for c, frames in window.frames.items():
        for f in frames:
            h.push(c, f)
        with pytest.raises(ValueError, match="duplicate"):
            h.push(c, frames[-1])
    # M5 evidence is optional and informational since 2026-11-12 (user directive): missing / moving / stateless sets
    # are measured by sync_report (warnings), only malformed evidence is refused
    end = dict.fromkeys(spec.camera_ids, 2)
    for motion in (None, [0.001, 0.006, 0.001], {0: 0.001, 1: 0.001}, [float("inf")] * 3, {}):
        assert h.assemble(end, surface_disp_m=motion).frame_ids == {c: [0, 1, 2] for c in spec.camera_ids}
    rep = h.assemble(end, surface_disp_m=[0.001, 0.006, 0.001]).sync_report()
    assert rep["motion_over_bound"] and not rep["mismatch"] and any("M5" in w for w in rep["warnings"])
    rep = h.assemble(end, surface_disp_m={0: 0.001, 1: 0.001}).sync_report()
    assert rep["surface_disp_m"] == [0.001, 0.001, None] and not rep["motion_over_bound"] and not rep["warnings"]
    rep = h.assemble(end, surface_disp_m=[float("inf")] * 3).sync_report()
    assert rep["surface_disp_m"] == ["no_state"] * 3 and rep["motion_over_bound"]
    assert h.assemble(end, surface_disp_m={}).sync_report()["surface_disp_m"] is None
    for bad in ([float("nan")] * 3, [0.001] * 2, [-0.001] * 3, [True] * 3):
        with pytest.raises(ValueError, match="M5 evidence"):
            h.assemble(end, surface_disp_m=bad)
    assert h.assemble(end, surface_disp_m={0: 0.0, 1: 0.002, 2: 0.005}).frame_ids == \
        {c: [0, 1, 2] for c in spec.camera_ids}
    frames = copy.deepcopy(window.frames)
    f = frames["cam_a"][1]
    frames["cam_a"] = (frames["cam_a"][0], Frame(1, f.timestamp_ns + 9_000_000, f.rgb), frames["cam_a"][2])
    with pytest.raises(ValueError, match="irregular"):
        ObservationWindow(spec, frames, (0, 0, 0)).validate()
    h.reset()
    with pytest.raises(ValueError, match="missing camera frame"):
        h.assemble(dict.fromkeys(spec.camera_ids, 2), surface_disp_m=[0] * 3)


def test_run_start_padding_is_the_training_rule(policies):
    """``store.history``: a history inside a run is padded with the run's first frame; a dropped frame starts a new
    run (never compressing the spacing); only with explicit pad_run_start."""
    policy = policies["reviwo"]
    spec = policy.reset_session(**session_choices(policy, "padding"))
    frames = synthetic_frames(spec, 8)["cam_a"]
    h = HistoryWindow(spec)
    h.push("cam_a", frames[0])
    with pytest.raises(ValueError, match="incomplete"):
        h.assemble({"cam_a": 0})
    w = h.assemble({"cam_a": 0}, pad_run_start=True)
    assert w.padded and w.frame_ids == {"cam_a": [0, 0, 0]}
    h.push("cam_a", frames[1])
    assert h.assemble({"cam_a": 1}, pad_run_start=True).frame_ids == {"cam_a": [0, 0, 1]}
    h.push("cam_a", frames[2])
    w = h.assemble({"cam_a": 2}, pad_run_start=True)
    assert not w.padded and w.frame_ids == {"cam_a": [0, 1, 2]}
    for f in frames[4:6]:  # frame 3 dropped: a new run starts at 4
        h.push("cam_a", f)
    with pytest.raises(ValueError, match="incomplete"):
        h.assemble({"cam_a": 5})
    assert h.assemble({"cam_a": 5}, pad_run_start=True).frame_ids == {"cam_a": [4, 4, 5]}
    out = policy.infer(w, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(w), noise=torch.zeros(1, 7))
    assert out["metadata"]["padded"] is False
    with pytest.raises(ValueError, match="duplicate"):
        ObservationWindow(spec, {"cam_a": (frames[0], frames[0], frames[1])}).validate()
    mv = policies["mvp3d_multiview"]
    spec = mv.reset_session(**session_choices(mv, "padding-mv"))
    sf = synthetic_frames(spec, 3)
    h = HistoryWindow(spec)
    for c in spec.camera_ids:
        for f in sf[c][(1 if c == "cam_b" else 0):]:  # cam_b's run starts one step later
            h.push(c, f)
    w = h.assemble(dict.fromkeys(spec.camera_ids, 2), surface_disp_m=[0.001] * 3, pad_run_start=True)
    assert w.padded and w.frame_ids == {c: [1, 1, 2] for c in spec.camera_ids}
    out = mv.infer(w, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(w), noise=torch.zeros(1, 7))
    assert out["metadata"]["padded"] is True


def test_cpu_preprocessing_is_exact_and_torch_free():
    code = ("import sys; from s4d.rwlearn.deploy import InputSpec, Frame, HistoryWindow, preprocess_frame; "
            "import s4d.rwlearn.deploy.realsense; assert not {'torch', 'scipy', 'h5py'} & set(sys.modules), "
            "sorted({'torch', 'scipy', 'h5py'} & set(sys.modules))")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "."})
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    d = rng.uniform(0, 3, (480, 640)).astype(np.float32)
    valid = rng.random(d.shape) > 0.2
    d[~valid] = np.nan
    K = np.array([[600, 0, 320.5], [0, 600, 240.5], [0, 0, 1]])
    got = preprocess_frame(rgb, depth=d, valid=valid, K_native=K)
    box = center_box(480, 640)
    np.testing.assert_array_equal(got["rgb"], crop_rgb(rgb, box, 128))
    np.testing.assert_array_equal(got["depth"], nearest_resample(training_depth(d, valid), box, 128))
    from s4d.realworld.train_store import box_affine

    np.testing.assert_array_equal(got["K"], box_affine(box, 128).apply_K(K))  # the store's K rule
    np.testing.assert_array_equal(got["K"], crop_intrinsics(K, box, 128))
    np.testing.assert_array_equal(got["valid"], got["depth"] > 0)
    corr = {"stage": "sensor_depth", "factor": 1.0172, "validated": True, "evidence": {"document": "fixture"}}
    scaled = preprocess_frame(rgb, depth=d, valid=valid, native_correction=corr)["depth"]
    np.testing.assert_array_equal(scaled, nearest_resample(training_depth(d * np.float32(1.0172), valid), box, 128))
    with pytest.raises(ValueError, match="together"):
        preprocess_frame(rgb, depth=d)
    with pytest.raises(ValueError, match="sensor_depth"):
        preprocess_frame(rgb, depth=d, valid=valid, native_correction={**corr, "stage": "foundation_stereo_depth"})


def test_recorded_alignment_equals_sensor_depth_kernel():
    from s4d.realworld.kernels import register_depth

    raw = np.array([[0, 300, 9000, 600], [500, 600, 10, 700]], np.uint16)
    K, T = np.eye(3), np.eye(4)
    got = align_recorded_depth(raw, K, K, T, raw.shape, 0.001)
    expected, _ = register_depth({"color_hw": raw.shape, "depth_scale_m": 0.001, "min_depth_m": 0.1,
                                  "max_depth_m": 6.0, "splat": 2},
                                 {"depth_raw": raw[None], "K_depth": K, "K_color": K, "T_color_depth": T})
    for k in got:
        np.testing.assert_array_equal(got[k], expected[k][0])


def test_realsense_optional_sdk_fixture():
    """FIXTURE frames and SDK, no pyrealsense2 import or device required."""
    intr = SimpleNamespace(width=4, height=2, fx=3, fy=3, ppx=1.5, ppy=0.5, coeffs=[0] * 5)
    profile = SimpleNamespace(get_intrinsics=lambda: intr, format=lambda: "bgr8")
    profile.as_video_stream_profile = lambda: profile

    class SDKFrame:
        def __init__(self, data):
            self.profile, self.data = profile, data

        def get_data(self):
            return self.data

        def get_timestamp(self):
            return 13.5

        def get_frame_timestamp_domain(self):
            return "hardware_clock"

        def get_frame_number(self):
            return 10

    rgb = np.full((2, 4, 3), [1, 2, 3], np.uint8)
    raw = np.array([[0, 500, 1000, 40], [1000, 1000, 500, 4000]], np.uint16)  # 40 -> 0.08 m, 4000 -> 8 m: invalid
    frames = SimpleNamespace(get_color_frame=lambda: SDKFrame(rgb), get_depth_frame=lambda: SDKFrame(raw))
    calls = []
    sdk = SimpleNamespace(stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8", rgb8="rgb8"),
                          align=lambda stream: (calls.append(stream) or SimpleNamespace(process=lambda f: f)))
    # device extrinsics depth -> colour: identity rotation (column-major), colour 1 mm behind the depth camera
    ext = SimpleNamespace(rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[0.015, 0.0, 0.001])
    streams = {"depth": SimpleNamespace(get_extrinsics_to=lambda other: ext), "color": object()}
    sdk.stream.depth = "depth"
    pipeline = SimpleNamespace(get_device=lambda: SimpleNamespace(
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.002)), get_stream=streams.__getitem__)
    helper = RealSenseRGBD(pipeline, correction=native_correction(), out_size=2, rs_module=sdk)
    got = helper.process(frames)
    m = raw.astype(np.float64) * 0.002
    valid = (raw > 0) & (m >= 0.1) & (m <= 6.0)  # gate on the device z, as the sensor_depth stage
    # R = I: colour-camera z = depth-camera z + t_z exactly (rs.align writes depth-camera z, DEPTH-COMPAT.md)
    expected = preprocess_frame(rgb[..., ::-1].copy(), depth=np.where(valid, m + 0.001, 0).astype(np.float32),
                                valid=valid, K_native=np.array([[3, 0, 2], [0, 3, 1], [0, 0, 1]]), out_size=2,
                                native_correction=native_correction())
    for k in expected:
        np.testing.assert_array_equal(got[k], expected[k])
    assert calls == ["color"] and got["metadata"]["color_timestamp_ms"] == 13.5
    assert got["metadata"]["depth_source"] == "native_realsense"
    with pytest.raises(ValueError, match="sensor_depth"):
        RealSenseRGBD(pipeline, correction={"stage": "foundation_stereo_depth", "factor": 1.0}, rs_module=sdk)


def test_typed_refusals_and_concurrent_sessions(policies):
    """Refusal classes (ValueError subclasses) for the serving layer; several live sessions share one loaded model."""
    assert all(issubclass(e, DeployRefusal) and issubclass(e, ValueError) for e in
               (BindingRefused, ContractViolation, IncompleteHistory, IrregularHistory, SessionMismatch, Unsynchronized))
    policy = policies["mvp3d_multiview"]
    a = policy.reset_session(**session_choices(policy, "typed-a"))
    b = policy.reset_session(**session_choices(policy, "typed-b"), concurrent=True)
    assert policy.live_sessions == ("typed-a", "typed-b")
    wa, wb = synthetic_window(a), synthetic_window(b, seed=3)
    noise = torch.zeros(1, 7)
    ra = policy.infer(wa, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(wa), noise=noise)
    rb = policy.infer(wb, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(wb), noise=noise)
    assert ra["metadata"]["session_id"] == "typed-a" and rb["metadata"]["session_id"] == "typed-b"
    with pytest.raises(SessionMismatch):
        policy.reset_session(**session_choices(policy, "typed-a"))
    with pytest.raises(BindingRefused):
        policy.infer(wa, [0.3, 0, 0.4, 0, 0, 0, 1], 3, {**binding(wa), "reference_id": None})
    h = HistoryWindow(a)
    frames = synthetic_frames(a, 3)
    for c in a.camera_ids:
        h.push(c, frames[c][0])
        h.push(c, frames[c][2])
    with pytest.raises(IncompleteHistory):
        h.assemble(dict.fromkeys(a.camera_ids, 2), surface_disp_m=[0.0] * 3)
    moving = ObservationWindow(a, wa.frames, (0.0, 0.01, 0.0))
    moving.validate()  # M5 motion above 5 mm: a warning since 2026-11-12, never a refusal
    assert moving.sync_report()["motion_over_bound"] and not moving.sync_report()["mismatch"]
    shifted = dict(wa.frames)
    f = shifted["cam_b"]
    shifted["cam_b"] = tuple(Frame(x.seq, x.timestamp_ns + 40_000_000, x.rgb, x.depth, x.valid, x.metadata) for x in f)
    late = ObservationWindow(a, shifted, (0.0,) * 3)
    late.validate()  # 40 ms >= half a 15 Hz period: not mutual-nearest, served with a warning
    rep = late.sync_report()
    cams = list(a.camera_ids)
    t = late.timestamps_ns
    assert rep["mismatch"] and rep["max_offset_ms"] == np.abs(t - t[:, :1]).max() / 1e6 >= 40.0 and \
        rep["bound_ms"] == pytest.approx(a.frame_period_s * 500) and rep["warnings"]
    assert rep["offsets_ms"][f"{cams[0]}|{cams[1]}"] == (t[-1, 0] - t[-1, 1]) / 1e6 <= -39.0
    assert rep["capture_ns"] == {c: int(t[-1, i]) for i, c in enumerate(cams)}
    assert rep["age_rel_ms"] == {c: (t[-1].max() - t[-1, i]) / 1e6 for i, c in enumerate(cams)}
    assert late.observation_time_ns == max(x.timestamp_ns for c in cams for x in shifted[c][-1:])
    served = policy.infer(late, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(late), noise=noise)
    assert np.isfinite(np.asarray(served["action_norm"])).all()
    assert ObservationWindow(a, wa.frames, (0.0,) * 3).sync_report()["warnings"] == []
    with pytest.raises(Unsynchronized):  # structural: run-start padding must stay step-aligned across cameras
        ObservationWindow(a, {**wa.frames, cams[0]: (wa.frames[cams[0]][0],) * 2 + wa.frames[cams[0]][1:2]},
                          (0.0,) * 3, padded=True).validate()
    with pytest.raises(IrregularHistory):
        ObservationWindow(a, {**wa.frames, "cam_a": (f[0], f[2], f[1])}, (0.0,) * 3).validate()
    with pytest.raises(ContractViolation):
        policy.reset_session(**{**session_choices(policy, "typed-c"), "camera_id": "cam_a"})
    policy.end_session("typed-a")
    with pytest.raises(SessionMismatch):
        policy.infer(wa, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(wa), noise=noise)
    policy.infer(wb, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(wb), noise=noise)
    policy.reset_session(**session_choices(policy, "typed-d"))  # default: replaces every live session
    assert policy.live_sessions == ("typed-d",)
    with pytest.raises(SessionMismatch):
        policy.infer(wb, [0.3, 0, 0.4, 0, 0, 0, 1], 3, binding(wb), noise=noise)


def test_colour_z_conversion_and_rs_align_emulation():
    """rs.align writes depth-camera z; the exact colour-z conversion (DEPTH-COMPAT.md) on a rotated, translated rig,
    and the z-buffer / border rules of the validated rs.align emulation."""
    from scipy.spatial.transform import Rotation

    from s4d.rwlearn.deploy.realsense import aligned_native_depth, depth_z_to_color_z, emulate_rs_align, extrinsics_matrix

    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.02, -0.03, 0.01]).as_matrix()
    T[:3, 3] = [0.015, -0.002, 0.003]
    zc_true = np.random.default_rng(0).uniform(0.4, 1.5, (240, 320))
    v, u = np.mgrid[0:240, 0:320] + 0.5
    p_c = np.stack([(u - 160) / 300 * zc_true, (v - 120) / 300 * zc_true, zc_true], -1)
    p_d = (p_c - T[:3, 3]) @ T[:3, :3]  # inv(T) applied to colour points
    got = depth_z_to_color_z(p_d[..., 2], K, T)
    np.testing.assert_allclose(got, zc_true.astype(np.float32), rtol=0, atol=2e-7)
    assert depth_z_to_color_z(np.zeros((2, 3)), K, T).max() == 0
    ext = SimpleNamespace(rotation=T[:3, :3].T.ravel().tolist(), translation=T[:3, 3].tolist())
    np.testing.assert_array_equal(extrinsics_matrix(ext), T)
    z16 = np.full((6, 8), 1000, np.uint16)
    z16[2, 3] = 500  # nearer surface wins its rectangle (z-buffer)
    Kd = np.array([[8.0, 0, 4.0], [0, 8.0, 3.0], [0, 0, 1]])
    out = emulate_rs_align(z16, Kd, Kd, np.eye(4), (6, 8), 0.001)
    assert out.dtype == np.uint16 and (out > 0).all() and out[2, 3] == 500 and out[3, 4] == 500 and out[0, 0] == 1000
    depth, valid = aligned_native_depth(out, 0.001, Kd, np.eye(4))
    np.testing.assert_array_equal(depth, out.astype(np.float64) * 0.001)
    assert valid.all()


def test_feature_standardisation_is_applied_by_the_deploy_path(tmp_path):
    """A checkpoint with frozen-feature standardisation (encoder buffers + meta record) served through DeployPolicy:
    ``infer`` == ``act`` == the adapter on (float16(f) - mu) / sigma, bitwise. Deploy-contract sidecars must carry
    exactly the checkpoint's record (a stale sidecar without it, or a record without the checkpoint's, is refused)."""
    from s4d.rwlearn.deploy import DeployPolicy
    from s4d.rwlearn.fql import FQLAgent
    from s4d.rwlearn.fql_data import ObsEncoder, feature_stats_sha256

    make_fixture_policy(tmp_path / "raw", "reviwo")
    meta = torch.load(tmp_path / "raw" / "policy.pt", map_location="cpu", weights_only=False)["meta"]
    d = int(meta["feat_dim"])
    rng = np.random.default_rng(0)
    mean, std = rng.normal(size=d).astype(np.float32), rng.uniform(0.5, 2.0, d).astype(np.float32)
    record = {"kind": "train_standardize", "rows": 100, "dim": d, "std_floor_rel": 1e-3, "floored_dims": 0,
              "sha256": feature_stats_sha256(mean, std)}
    torch.manual_seed(4)

    def enc():
        return ObsEncoder(d, False, proj=8, feature_stats={**record, "mean": mean, "std": std})

    agent = FQLAgent.create(1, enc().out_dim, 7, {"actor_hidden_dims": (8,), "value_hidden_dims": (8,)}, encoder=enc)
    fixture_meta = {**meta, "feature_norm": record,
                    "observation_contract": {**meta["observation_contract"], "feature_norm": record}}
    torch.save({"meta": {**meta, "feature_norm": record}, "agent": agent.state_dict()}, tmp_path / "stale.pt")
    with pytest.raises(ValueError, match="feature_norm"):  # embedded fixture contract without the record
        DeployPolicy(tmp_path / "stale.pt", export_path=tmp_path / "raw" / "encoder.pt")
    torch.save({"meta": fixture_meta, "agent": agent.state_dict()}, tmp_path / "norm.pt")
    policy = DeployPolicy(tmp_path / "norm.pt", export_path=tmp_path / "raw" / "encoder.pt")
    window = synthetic_window(policy.reset_session(**session_choices(policy, "norm")))
    pose = [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    out = policy.infer(window, pose, 100, binding(window), noise=torch.zeros(1, A.ACTION_DIM))
    assert np.array_equal(out["action_norm"], policy.act(window.observation(), pose, 100,
                                                         noise=torch.zeros(1, A.ACTION_DIM))["action_norm"])
    actor = policy.offline.agent.actor_onestep_flow
    twin = ObsEncoder(d, False, proj=8)
    twin.head.load_state_dict(actor.encoder.head.state_dict())
    feat = policy.offline.features(window.observation()).to(torch.float16).float()
    z = (feat - torch.from_numpy(mean)) / torch.from_numpy(std)
    proprio = torch.as_tensor(A.measured_proprio(np.array(pose), 100)).float()[None]
    expected = actor.mlp(torch.cat([twin({"feat": z, "proprio": proprio}), torch.zeros(1, A.ACTION_DIM)], -1))
    assert np.array_equal(out["action_norm"], expected.clamp(-1, 1).detach().numpy()[0])

    plain = {k: v for k, v in meta.items() if k not in ("fixture", "observation_contract")}
    sidecar = dict(meta["observation_contract"])
    with_norm = {**plain, "feature_norm": record}
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps({**sidecar, "feature_norm": record}))
    assert C.read_training_contract(with_norm, path)["feature_norm"] == record
    assert "feature_norm" not in C.read_training_contract(plain, sidecar)  # no standardisation anywhere: as before
    for ckpt_meta, source in ((with_norm, sidecar), (plain, {**sidecar, "feature_norm": record}),
                              (with_norm, {**sidecar, "feature_norm": {**record, "sha256": "0" * 64}})):
        with pytest.raises(ValueError, match="feature_norm"):
            C.read_training_contract(ckpt_meta, source)


def test_export_root_resolves_repo_relative_export(tmp_path):
    """Staged checkpoint roots mirror ``runs/``: export = export_root / meta["export_path"] (sha256-checked)."""
    from s4d.rwlearn.deploy import DeployPolicy

    make_fixture_policy(tmp_path / "runs" / "pre", "reviwo")
    ckpt = torch.load(tmp_path / "runs" / "pre" / "policy.pt", map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="relative"):
        DeployPolicy(tmp_path / "runs" / "pre" / "policy.pt", export_root=tmp_path)
    ckpt["meta"]["export_path"] = "runs/pre/encoder.pt"
    torch.save(ckpt, tmp_path / "staged.pt")
    policy = DeployPolicy(tmp_path / "staged.pt", export_root=tmp_path)
    assert policy.input_spec.condition == "reviwo" and policy.input_spec.feature_precision == "float16"
    with pytest.raises(FileNotFoundError):
        DeployPolicy(tmp_path / "staged.pt", export_root=tmp_path / "elsewhere")


def test_check_realworld_deploy_skips_and_lists_windows_with_an_incomplete_history(monkeypatch):
    """scripts/check_realworld_deploy.py equality: a recorded window whose trained history is incomplete (a dropped
    camera frame) is skipped and listed, the others are assembled (it used to abort the whole run)."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    import s4d.rwlearn.deploy as D
    from s4d.rwlearn.deploy.errors import IncompleteHistory

    spec_ = importlib.util.spec_from_file_location("crd", Path(__file__).resolve().parents[1] /
                                                   "scripts/check_realworld_deploy.py")
    crd = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(crd)
    numbers = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10])  # frame 4 dropped

    class Window:
        def __init__(self, spec):
            self.pushed = []

        def push(self, cam, frame):
            self.pushed.append(frame)

        def assemble(self, ends):
            if self.pushed[-1] - self.pushed[0] != 2:
                raise IncompleteHistory("incomplete history or missing camera frame at trained spacing")
            return ("window", ends)

    monkeypatch.setattr(D, "HistoryWindow", Window)
    rec = SimpleNamespace(timestamps_ns=lambda c: np.arange(len(numbers)), frame_numbers=lambda c: numbers,
                          frame=lambda c, i, **kw: int(numbers[i]))
    spec = SimpleNamespace(modalities=["rgb"], native_depth_correction={}, camera_ids=["cam"], views=1)
    skipped = []
    out = crd._windows(None, spec, rec, SimpleNamespace(windows=8), None, skipped)
    assert skipped == [{"cam": 4}, {"cam": 5}] and len(out) == 6
    assert all(info["stream_index"]["cam"] not in (4, 5) for _, info in out)


def _rs_align_per_pixel(z16, K_depth, K_color, T_color_depth, color_hw, depth_units_m):
    """The previous per-pixel ``emulate_rs_align`` (until 2026-11-12), kept as the reference of the vectorised one."""
    f32 = np.float32
    Hc, Wc = (int(x) for x in color_hw)
    Kd = np.asarray(K_depth, np.float64) - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]])
    Kc = np.asarray(K_color, np.float64) - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]])
    ys, xs = np.nonzero(z16)
    zr = z16[ys, xs]
    depth = zr.astype(f32) * f32(depth_units_m)
    fxd, fyd, cxd, cyd = f32(Kd[0, 0]), f32(Kd[1, 1]), f32(Kd[0, 2]), f32(Kd[1, 2])
    fxc, fyc, cxc, cyc = f32(Kc[0, 0]), f32(Kc[1, 1]), f32(Kc[0, 2]), f32(Kc[1, 2])
    R = np.asarray(T_color_depth, np.float64)[:3, :3].astype(f32)
    t = np.asarray(T_color_depth, np.float64)[:3, 3].astype(f32)

    def corner(off):
        px, py = xs.astype(f32) + f32(off), ys.astype(f32) + f32(off)
        X, Y, Z = ((px - cxd) / fxd) * depth, ((py - cyd) / fyd) * depth, depth
        Xo = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0]
        Yo = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1]
        Zo = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z + t[2]
        u, v = (Xo / Zo) * fxc + cxc, (Yo / Zo) * fyc + cyc
        return np.trunc(u + f32(0.5)).astype(np.int64), np.trunc(v + f32(0.5)).astype(np.int64)

    x0, y0 = corner(-0.5)
    x1, y1 = corner(0.5)
    ok = (x0 >= 0) & (y0 >= 0) & (x1 < Wc) & (y1 < Hc)
    x0, y0, x1, y1, zr = x0[ok], y0[ok], x1[ok], y1[ok], zr[ok]
    out = np.full(Hc * Wc, np.iinfo(np.uint16).max, np.int64)
    w, h = np.maximum(x1 - x0 + 1, 0), np.maximum(y1 - y0 + 1, 0)
    for dy in range(int(h.max(initial=0))):
        for dx in range(int(w.max(initial=0))):
            m = (dy < h) & (dx < w)
            np.minimum.at(out, (y0[m] + dy) * Wc + (x0[m] + dx), zr[m].astype(np.int64))
    out[out == np.iinfo(np.uint16).max] = 0
    return out.reshape(Hc, Wc).astype(np.uint16)


@pytest.mark.parametrize("case", ["d435_like", "wide_depth_grid", "close_and_far", "zero_translation"])
def test_vectorised_rs_align_emulation_is_bitwise_the_per_pixel_rule(case):
    """The vectorised replay emulation == the previous per-pixel implementation, bit for bit: holes, depth
    discontinuities (z-buffer), border rectangles, a depth grid differing from the colour grid, 65535 values, a rig
    without translation (0/0 on holes only)."""
    from scipy.spatial.transform import Rotation

    from s4d.rwlearn.deploy.realsense import emulate_rs_align

    rng = np.random.default_rng(len(case))
    dhw, chw = ((48, 64), (48, 64)) if case != "wide_depth_grid" else ((60, 80), (48, 64))
    Kd = np.array([[50.0, 0, dhw[1] / 2 + 0.3], [0, 51.0, dhw[0] / 2 - 0.2], [0, 0, 1]])
    Kc = np.array([[60.0, 0, chw[1] / 2 - 0.4], [0, 60.5, chw[0] / 2 + 0.1], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [0.01, -0.02, 0.015]).as_matrix()
    T[:3, 3] = [0.0, 0.0, 0.0] if case == "zero_translation" else [0.015, -0.001, 0.002]
    lo, hi = (150, 400) if case == "close_and_far" else (400, 3000)
    z16 = rng.integers(lo, hi, dhw).astype(np.uint16)
    z16[rng.random(dhw) < 0.15] = 0  # holes
    z16[10:20, 30:40] = lo  # a near patch over a far background
    z16[0, :] = 65535
    out = emulate_rs_align(z16, Kd, Kc, T, chw, 0.001)
    ref = _rs_align_per_pixel(z16, Kd, Kc, T, chw, 0.001)
    assert out.dtype == ref.dtype == np.uint16 and out.shape == ref.shape == chw
    assert (out > 0).mean() > 0.5
    np.testing.assert_array_equal(out, ref)
