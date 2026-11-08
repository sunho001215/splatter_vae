"""S2M2 candidate depth stage (study 2026-11-08): the same worker task, conversion, validity rules, registration and
persisted layout as the FoundationStereo stage (B1), so consumers switch by stage name; optional frame subsets (the
calibration pool frames); the B1 stage itself is unchanged (config, chunk ids, worker parameters). CPU numpy only; the
model output is a fixed array, never S2M2 output."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
H, W, N = 12, 16, 7
K = np.array([[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]])


def _episode(tmp_path, monkeypatch, n=N):
    from s4d.realworld import depth_stages, ir_stereo_depth, pipeline
    from s4d.realworld import io as rio

    cam = "cam/color"
    g = SimpleNamespace(rig_id="cam", left_stream="cam/ir_left", right_stream="cam/ir_right", baseline_m=0.05,
                        T_color=lambda: np.eye(4))
    stream = SimpleNamespace(intrinsics=SimpleNamespace(height=H, width=W))
    meta = SimpleNamespace(rig_rgbd=lambda _: SimpleNamespace(color_stream=cam), stream=lambda _: stream,
                           stream_timestamps_ns={"cam/ir_left": list(range(n)), cam: list(range(n))})
    reads = []

    def read_frames(root, ep, stream_id, idx):
        reads.append((stream_id, list(idx)))
        return np.stack([np.full((H, W), i, np.uint8) for i in idx])

    monkeypatch.setattr(pipeline, "load_meta", lambda *_: meta)
    monkeypatch.setattr(pipeline, "read_frames", read_frames)
    monkeypatch.setattr(ir_stereo_depth, "ir_rigs", lambda _: [g])
    monkeypatch.setattr(depth_stages, "view_K", lambda *_: K)
    rio.run_stage(rio.episode_dir(tmp_path, "ep") / "normalized", stage="ingest", version="test",
                  config={}, inputs={}, build=lambda _: {})
    return cam, reads


class _Runner:
    """Worker = disparity 2 + frame index (left and right reference), extra S2M2 outputs included; kernels run in
    process exactly as the LocalRunner does."""

    def __init__(self, extra=False):
        self.calls, self.extra = [], extra

    def worker(self, spec, task, params, arrays, *, chunk_id):
        self.calls.append((task, dict(params), chunk_id))
        frame = arrays["left"][:, 0, 0].astype(np.float32)
        d = np.broadcast_to(2.0 + frame[:, None, None], arrays["left"].shape).astype(np.float32).copy()
        out = {"disparity_left": d, "disparity_right": d.copy()}
        if self.extra:
            out.update(occlusion_left=np.ones_like(d, np.float16), confidence_left=np.ones_like(d, np.float16))
        return {"model": {"name": spec.name}, "meta": {"m": 1}}, out, None

    def kernel(self, task, params, arrays, *, chunk_id):
        from s4d.realworld.kernels import run_kernel

        self.calls.append((task, None, chunk_id))
        out, meta = run_kernel(task, params, arrays)
        return out, meta, None

    def ack(self, handle):
        return None

    def describe(self):
        return {"kind": "test"}


def test_fs_stage_config_worker_params_and_chunk_ids_are_unchanged(tmp_path, monkeypatch):
    from s4d.realworld import io as rio
    from s4d.realworld.baseline_stages import worker_identity
    from s4d.realworld.ir_stereo_depth import STAGE_VERSION, ir_stereo_depth
    from s4d.realworld.workers import WorkerSpec

    cam, _ = _episode(tmp_path, monkeypatch, n=10)
    spec = WorkerSpec("foundation_stereo", "python")
    runner = _Runner()
    rec = ir_stereo_depth(tmp_path, "ep", spec, runner)
    assert rec["config"] == {"chunk": 8, "valid_iters": 32, "hiera": 0, "min_disparity": 0.5, "min_depth_m": 0.1,
                             "max_depth_m": 6.0, "splat": 2, "worker_identity": worker_identity(spec),
                             "kernel": "ir_disparity_to_color"}
    assert rec["version"] == STAGE_VERSION == "2"
    workers = [c for c in runner.calls if c[0] == "ir_pair_disparity"]
    assert [c[1] for c in workers] == [{"valid_iters": 32, "hiera": 0}] * 2
    assert [c[2] for c in runner.calls] == ["fs/cam/color/00000-00007", "fs2c/cam/color/00000-00007",
                                            "fs/cam/color/00008-00009", "fs2c/cam/color/00008-00009"]
    with h5py.File(rio.stage_dir(tmp_path, "ep", "foundation_stereo_depth") / "cam__color.h5") as f:
        assert set(f) == {"depth", "valid"} and "FoundationStereo" in f.attrs["source"]


def test_s2m2_stage_has_the_fs_output_contract_and_conversion(tmp_path, monkeypatch):
    from s4d.realworld import io as rio
    from s4d.realworld.depth_stages import load_view_depth
    from s4d.realworld.ir_stereo_depth import S2M2_STAGE_VERSION, ir_stereo_depth, s2m2_ir_depth
    from s4d.realworld.workers import WorkerSpec

    cam, _ = _episode(tmp_path, monkeypatch)
    ir_stereo_depth(tmp_path, "ep", WorkerSpec("foundation_stereo", "python"), _Runner())
    runner = _Runner(extra=True)
    s2 = s2m2_ir_depth(tmp_path, "ep", WorkerSpec("s2m2", "python"), runner)
    assert s2["version"] == S2M2_STAGE_VERSION and s2["stage"] == "s2m2_depth"
    assert {k: s2["config"][k] for k in ("model_type", "num_refine", "precision", "allow_negative")} == \
        {"model_type": "XL", "num_refine": 3, "precision": "fp16", "allow_negative": False}
    assert "frames" not in s2["config"]
    assert runner.calls[0][1] == {"model_type": "XL", "num_refine": 3, "precision": "fp16", "allow_negative": False}
    assert runner.calls[0][2].startswith("s2m2/") and runner.calls[1][2].startswith("s2m2" + "2c/")
    a = rio.stage_dir(tmp_path, "ep", "foundation_stereo_depth") / "cam__color.h5"
    b = rio.stage_dir(tmp_path, "ep", "s2m2_depth") / "cam__color.h5"
    with h5py.File(a) as fa, h5py.File(b) as fb:
        assert set(fa) == set(fb) == {"depth", "valid"}
        assert set(fa.attrs) == set(fb.attrs)
        for key in fa.attrs:
            if key != "source":
                np.testing.assert_array_equal(np.asarray(fa.attrs[key]), np.asarray(fb.attrs[key]), err_msg=key)
        assert "S2M2-XL" in fb.attrs["source"]
        assert fa["depth"][:].tobytes() == fb["depth"][:].tobytes()  # same disparity -> bitwise same depth
    depth, valid, Kc = load_view_depth(tmp_path, "ep", cam, [0, 3], "s2m2_depth")
    z = K[0, 0] * 0.05 / np.array([2.0, 5.0])
    for k in range(2):
        assert valid[k].any()
        np.testing.assert_allclose(depth[k][valid[k]], z[k], rtol=1e-5)
    np.testing.assert_array_equal(Kc, K)
    summary = s2["summary"]
    assert summary["persisted_datasets"] == ["depth", "valid"] and summary["model"] == {"name": "s2m2"}


def test_s2m2_stage_frame_subset(tmp_path, monkeypatch):
    from s4d.realworld import io as rio
    from s4d.realworld.depth_stages import load_view_depth
    from s4d.realworld.ir_stereo_depth import s2m2_ir_depth
    from s4d.realworld.workers import WorkerSpec

    cam, reads = _episode(tmp_path, monkeypatch)
    runner = _Runner()
    rec = s2m2_ir_depth(tmp_path, "ep", WorkerSpec("s2m2", "python"), runner, chunk=2, frames={cam: [5, 0, 2, 2]})
    assert rec["config"]["frames"] == {cam: [0, 2, 5]}
    assert [r[1] for r in reads if r[0] == "cam/ir_left"] == [[0, 2], [5]]
    assert [c[2] for c in runner.calls if c[0] == "ir_pair_disparity"] == ["s2m2/cam/color/s00000-00002-n2",
                                                                          "s2m2/cam/color/s00005-00005-n1"]
    with h5py.File(rio.stage_dir(tmp_path, "ep", "s2m2_depth") / "cam__color.h5") as f:
        assert set(f) == {"depth", "valid", "computed"}
        np.testing.assert_array_equal(f["computed"][:], np.isin(np.arange(N), [0, 2, 5]))
        valid = f["valid"][:]
        assert valid[[0, 2, 5]].reshape(3, -1).any(1).all() and not valid[[1, 3, 4, 6]].any()
        assert (f["depth"][:][~valid] == 0).all()
    depth, valid, _ = load_view_depth(tmp_path, "ep", cam, [5], "s2m2_depth")
    np.testing.assert_allclose(depth[0][valid[0]], K[0, 0] * 0.05 / 7.0, rtol=1e-5)
    assert rec["summary"][cam]["frames_computed"] == 3
    other = s2m2_ir_depth(tmp_path, "ep", WorkerSpec("s2m2", "python"), _Runner(), chunk=2, frames={cam: [0, 2]})
    assert other["fingerprint"] != rec["fingerprint"]
    with pytest.raises(ValueError, match="outside"):
        s2m2_ir_depth(tmp_path, "ep", WorkerSpec("s2m2", "python"), _Runner(), frames={cam: [N]}, force=True)
    with pytest.raises(ValueError):
        s2m2_ir_depth(tmp_path, "ep", WorkerSpec("s2m2", "python"), _Runner(), model_type="XXL")


def _worker_module():
    path = REPO / "s4d/realworld/workers/s2m2_worker.py"
    spec = importlib.util.spec_from_file_location("s2m2_worker_under_test", path)
    module = importlib.util.module_from_spec(spec)
    old = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old
    return module


def test_s2m2_worker_input_and_right_reference_helpers():
    w = _worker_module()
    assert [w.checkpoint_name(t) for t in ("S", "M", "L", "XL")] == ["CH128NTR1.pth", "CH192NTR2.pth",
                                                                     "CH256NTR3.pth", "CH384NTR3.pth"]
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    three = w.gray_to_model_input(img)
    assert three.shape == (3, 4, 3) and three.dtype == np.uint8 and (three == img[..., None]).all()
    with pytest.raises(ValueError):
        w.gray_to_model_input(img.astype(np.float32))
    d_right = np.tile(np.linspace(1, 4, 20, dtype=np.float32), (6, 1))
    left = np.random.default_rng(0).integers(0, 255, (6, 20, 3), dtype=np.uint8)
    right = np.random.default_rng(1).integers(0, 255, (6, 20, 3), dtype=np.uint8)

    def oracle(a, b):  # inputs must be (flip(right), flip(left)); its left-reference disparity is flip(d_right)
        assert (a == right[:, ::-1]).all() and (b == left[:, ::-1]).all()
        return d_right[:, ::-1], np.ones_like(d_right)

    out = w.flip_and_swap(oracle, left, right)
    np.testing.assert_array_equal(out[0], d_right)


def test_cli_registers_the_s2m2_stage_without_making_it_standard():
    from s4d.realworld import cli
    from s4d.realworld.pipeline import RGBD_LIFT_DEPTH

    assert cli.STAGE_WORKERS["s2m2_depth"] == "s2m2"
    assert "s2m2_depth" in cli.OPTIONAL_STAGES and "s2m2_depth" in cli.RECORD_STAGES
    assert "s2m2_depth" not in cli.default_stages({"stages": {"foundation_stereo_depth": {}}})
    assert RGBD_LIFT_DEPTH == "foundation_stereo_depth"
