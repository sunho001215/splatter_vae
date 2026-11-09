"""FoundationStereo stage regression: config, worker parameters, chunk IDs and persisted layout (CPU only)."""

from types import SimpleNamespace

import h5py
import numpy as np

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
    """Worker = disparity 2 + frame index (left and right reference), optional confidence outputs included; kernels run in
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
