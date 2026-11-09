"""CPU/mock-only TE integration: no upstream imports, checkpoints, or geometry models executed."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from s4d.realworld import io as rio
from s4d.realworld.adapters.synthetic_adapter import SyntheticAdapter
from s4d.realworld.contract import RgbdGeometry
from s4d.realworld.export import export
from s4d.realworld.geometry import make_T, transform_points
from s4d.realworld.pipeline import chunk_queries, ingest, load_meta, load_tracks2d, read_frames, to_model_rgb
from s4d.realworld.runners import LocalRunner
from s4d.realworld.stages.motion import MotionFilterConfig, track_motion
from s4d.realworld.stages.trackeverything import query_chunks, resize_inputs, track_everything
from s4d.realworld.tracking import EDGE, NO_DEPTH, NOT_TRACKED, OCCLUDED, OUT, VALID, depth_support_status, query_grid
from s4d.realworld.viewer_data import load_camera_tracks
from s4d.realworld.workers import WorkerError, WorkerSpec, call_worker
from s4d.realworld.workers import trackeverything_worker as te


class ColourEpisode(SyntheticAdapter):
    """The existing synthetic ray-cast scene exposed as registered RGB-D cameras."""

    def __init__(self, path, frames=20, pause=10, image_hw=(48, 64), focal=65):
        super().__init__({"asset_dir": str(path), "frames": frames, "image_hw": image_hw, "focal": focal})
        self.pause = pause

    def read_meta(self, episode_id):
        m = super().read_meta(episode_id)
        streams = tuple(replace(s, modality="rgb", channels=3, channel_order="RGB", sensor_name="synthetic colour")
                        if s.eye == "left" else replace(s, modality="depth", channels=1, channel_order="DEPTH",
                                                       dtype="uint16", depth_scale_m=0.001, depth_kind="z")
                        for s in m.streams)
        rgbd = tuple(RgbdGeometry(g.rig_id, g.left_stream, g.right_stream,
                                  "metric" if g.status == "metric" else "unverified",
                                  tuple(map(tuple, np.eye(4))), pairing="synthetic registered depth", max_pair_offset_ns=0)
                     for g in m.stereo)
        stamps = {k: v + (np.arange(len(v)) >= self.pause) * 200_000_000 for k, v in m.stream_timestamps_ns.items()}
        traj = m.trajectory
        traj = replace(traj, timestamps_ns=traj.timestamps_ns +
                       (traj.timestamps_ns >= self.pause * 66_666_667) * 200_000_000)
        return replace(m, streams=streams, stereo=(), rgbd=rgbd, stream_timestamps_ns=stamps, trajectory=traj)

    def iter_frames(self, episode_id, stream_id, start=0, stop=None):
        rig, eye = stream_id.split("/")
        stamps = self.read_meta(episode_id).stream_timestamps_ns[stream_id]
        for i in range(start, min(stop or self.frames, self.frames)):
            image, depth, _, _ = self.render(rig, "left", i)
            if eye == "left":
                grey = np.clip(image, 0, 255).astype(np.uint8)
                # Different channels exercise RGB/BGR preservation.
                frame = np.stack([grey, grey // 2, 255 - grey], -1)
            else:
                frame = np.rint(depth * 1000).astype(np.uint16)
            yield i, int(stamps[i]), frame


def episode(tmp_path, frames=20, pause=10):
    a = ColourEpisode(tmp_path / "assets", frames, pause)
    root, ep = tmp_path / "out", "synthetic_ep000"
    ingest(a, ep, root)
    def build_calibration(tmp):
        summary = {}
        for i in (0, 1):
            cam = f"rig{i}/left"
            doc = {"status": "accepted", "T_cam_base": a.T_cam_base(f"rig{i}", "left").tolist(),
                   "source": {"kind": "synthetic calibration oracle", "is_mock": True}}
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}.json", doc)
            summary[cam] = doc
        return summary

    rio.run_stage(rio.stage_dir(root, ep, "calibrate"), stage="calibrate", version="oracle",
                  config={}, inputs={}, build=build_calibration)

    def build_depth(tmp):
        summary = {}
        for rig in ("rig0", "rig1"):
            cam = f"{rig}/left"
            depth = np.stack([a.render(rig, "left", f)[1] for f in range(frames)])
            with h5py.File(tmp / f"{rio.safe_name(cam)}.h5", "w") as h:
                h.create_dataset("depth", data=depth)
                h.create_dataset("valid", data=depth > 0)
                h.attrs["K"] = a.K()
            summary[cam] = {"model": {"name": "synthetic metric depth", "is_mock": True}}
        return summary

    rio.run_stage(rio.stage_dir(root, ep, "sensor_depth"), stage="sensor_depth", version="oracle",
                  config={}, inputs={}, build=build_depth)
    return a, root, ep


MOCK_SPEC = WorkerSpec("mock", sys.executable, env={"S4D_TE_MOCK": "1"})


class SyntheticRunner(LocalRunner):
    """Inject normalized cam0 predictions into the file-protocol mock worker, never a TE model. ``check_pose=False``
    accepts an estimated (not the oracle) calibration: the predictions stay the oracle base-frame points."""

    def __init__(self, path, adapter, check_pose=True):
        super().__init__(path)
        self.adapter, self.calls, self.acks, self.check_pose = adapter, [], [], check_pose

    def worker(self, spec, task, params, arrays, *, chunk_id, allow_mock):
        assert task == "track_queries" and params == {"mode": "sparse", "num_iters": 3}
        _, rig, eye, interval = chunk_id.split("/")
        start, end = map(int, interval.split("-"))
        cam = f"{rig}/{eye}"
        a = self.adapter
        assert len(arrays["video"]) == end - start <= 16
        if self.check_pose:
            np.testing.assert_allclose(arrays["w2c"], np.broadcast_to(a.T_cam_base(rig, eye), arrays["w2c"].shape))
        scale = np.array(arrays["video"].shape[1:3][::-1]) / np.array(a.hw[::-1])
        np.testing.assert_allclose(arrays["K"][0], np.diag([*scale, 1]) @ a.K())
        q = arrays["queries"]
        coords, visibility = [], []
        for birth in np.unique(q[:, 0]).astype(int):
            xy = q[q[:, 0] == birth, 1:] / scale
            rc = np.rint(xy[:, ::-1] - 0.5).astype(int)
            np.testing.assert_allclose(xy, rc[:, ::-1] + 0.5)
            xyz, vis = a.gt_points(cam, start + birth, rc, list(range(start, end)))
            coords.append(xyz)
            visibility.append(vis)
        base, vis = np.concatenate(coords, 1), np.concatenate(visibility, 1)
        norm = 2.75  # deliberate nonunit normalization + nontrivial rotation/translation
        coords = transform_points(arrays["w2c"][0], base) / norm
        pre = np.arange(end - start)[:, None] < q[None, :, 0]
        coords[pre] = 0  # upstream's dangerous prebirth placeholders, with visibility logit 0
        logits = np.where(vis, 8.0, -8.0).astype(np.float32)
        logits[pre] = 0
        extra = {"query_coords": coords[None].astype(np.float32), "query_visibs": logits[None],
                 "norm_factor": np.array(norm), "query_dynamic_logits": np.ones((1, *vis.shape), np.float32)}
        self.calls.append({"chunk_id": chunk_id, "queries": q.copy(), "arrays": arrays})
        response, out, _ = super().worker(spec, task, params, {**arrays, **extra},
                                         chunk_id=chunk_id, allow_mock=allow_mock)
        return response, out, chunk_id

    def ack(self, handle):
        # Check the manifest AND data digest at acknowledgement time.
        root = self.workdir.parent / "out"
        stores = list((rio.episode_dir(root, "synthetic_ep000") / "derived").glob(".trackeverything_clips.chunks.*"))
        assert len(stores) == 1
        s = rio.ChunkStore(stores[0].with_name("trackeverything_clips"), "unused")
        s.dir = stores[0]
        assert s.get(handle.removeprefix("trackeverything/")) is not None
        self.acks.append(handle)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("te")
    a, root, ep = episode(tmp)
    runner = SyntheticRunner(tmp / "worker", a)
    records = track_everything(root, ep, MOCK_SPEC, runner, depth_stage="sensor_depth", window=16,
                               step=8, grid_stride=4, model_hw=(32, 48))
    return a, root, ep, runner, records


def test_stage_layout_query_grid_and_base_coordinates(run):
    a, root, ep, runner, records = run
    assert len(runner.calls) == len(runner.acks) == 6  # two cameras, three query windows each
    rc = query_grid(a.hw, 4, 2)
    for cam in ("rig0/left", "rig1/left"):
        names = [f"{rio.safe_name(cam)}_q{q:05d}.npz" for q, _ in chunk_queries(a.frames, 16, 8)]
        assert records["track2d"]["summary"][cam]["chunks"] == records["lift3d"]["summary"][cam]["chunks"] == names
        for name in names:
            tr = load_tracks2d(rio.stage_dir(root, ep, "track2d") / name, cam)
            np.testing.assert_array_equal(tr.point_ids, np.arange(len(rc)))
            P, gt_vis = a.gt_points(cam, tr.query_frame, rc, list(tr.frame_indices))
            with np.load(rio.stage_dir(root, ep, "lift3d") / name) as z:
                ok = z["status"] == VALID
                assert ok.any()
                np.testing.assert_allclose(z["xyz_base"][ok], P[ok], atol=5e-7)
                np.testing.assert_allclose(z["xyz_cam"][ok], transform_points(a.T_cam_base(cam.split("/")[0], "left"),
                                                                            P)[ok], atol=5e-7)
                assert np.isnan(z["xyz_base"][~ok]).all() and np.isnan(z["xyz_cam"][~ok]).all()
                assert str(z["camera_id"]) == cam
            assert tr.source["name"] == "TrackEverything synthetic oracle" and tr.source["is_mock"]
            born = np.isfinite(tr.uv[0]).all(-1) & gt_vis[0]
            np.testing.assert_allclose(tr.uv[0, born], (rc[:, ::-1] + 0.5)[born], atol=1e-4)
            assert (tr.confidence == 1).all()
    assert "skipped" in records["lift3d"]["summary"]["rig2/left"]


def test_stage_records_te_provenance_per_camera(run):
    """Each camera entry pins the s4d source, upstream commit, checkpoint, seed and the worker's runtime identities;
    clips whose worker reported no runtime (here: the mock) are counted, never back-filled."""
    _, _, _, runner, records = run
    for stage in ("track2d", "lift3d"):
        for cam in ("rig0/left", "rig1/left"):
            p = records[stage]["summary"][cam]["provenance"]
            assert p["source"] == runner.describe() and p["expected_commit"] == MOCK_SPEC.expected_commit
            assert p["clips_called"] == 3 and p["clips_without_runtime"] == 3 and p["runtime"] == []
            assert "no deterministic mode" in p["inference"]
        assert "provenance" not in records[stage]["summary"]["rig2/left"]  # skipped camera


def test_worker_runtime_provenance_records_hardware_flags_hub_weights_and_seed(tmp_path, monkeypatch):
    import hashlib

    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / "dinov2_vits14_pretrain.pth").write_bytes(b"weights")
    props = SimpleNamespace(name="GPU X", uuid="GPU-abc", major=9, minor=0, total_memory=80)
    cudnn = SimpleNamespace(version=lambda: 91002, enabled=True, benchmark=False, deterministic=False, allow_tf32=True)
    backends = SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))
    fake = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: props),
                           hub=SimpleNamespace(get_dir=lambda: str(hub)),
                           _C=SimpleNamespace(_cuda_getDriverVersion=lambda: 12090),
                           version=SimpleNamespace(cuda="12.9"), __version__="2.8.0", backends=backends,
                           get_float32_matmul_precision=lambda: "highest",
                           are_deterministic_algorithms_enabled=lambda: False)
    monkeypatch.setattr(te, "_RUNTIME", {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    rt = te.runtime_provenance(fake)
    assert rt["gpu"] == {"name": "GPU X", "uuid": "GPU-abc", "capability": [9, 0], "total_memory": 80}
    assert rt["cuda_visible_devices"] == "GPU-abc" and rt["cuda_driver_api"] == 12090 and rt["cudnn"] == 91002
    assert rt["torch"] == "2.8.0" and rt["cuda_runtime"] == "12.9"
    assert rt["torch_hub_checkpoints_sha256"] == {"dinov2_vits14_pretrain.pth": hashlib.sha256(b"weights").hexdigest()}
    assert rt["seed"] == te.SEED == 0 and rt["deterministic_mode"] is False
    assert rt["flags"]["cudnn_benchmark"] is False and rt["flags"]["cudnn_allow_tf32"] is True
    assert rt["flags"]["float32_matmul_precision"] == "highest" and rt["flags"]["deterministic_algorithms"] is False
    cudnn.benchmark = True  # flags are re-read on every response, the static identity is cached
    assert te.runtime_provenance(fake)["flags"]["cudnn_benchmark"] is True


def test_te_inference_frames_are_the_query_windows_including_pause(run):
    """Every clip is exactly one ``pipeline.chunk_queries`` window of one camera's recorded frames (a recording pause
    included), resized as the model input; nothing else is sent."""
    a, root, ep, runner, _ = run
    meta = load_meta(root, ep)
    expected = [(cam, q, idx) for cam in ("rig0/left", "rig1/left")
                for q, idx in chunk_queries(len(meta.stream_timestamps_ns[cam]), 16, 8)]
    assert [("/".join(c["chunk_id"].split("/")[1:3]), int(c["chunk_id"].split("/")[-1].split("-")[0]))
            for c in runner.calls] == [(cam, q) for cam, q, _ in expected]
    for call, (cam, q, idx) in zip(runner.calls, expected):
        video = to_model_rgb(read_frames(root, ep, cam, idx), meta.stream(cam))
        np.testing.assert_array_equal(resize_inputs(video, np.ones(video.shape[:3]), a.K(), (32, 48))[0],
                                      call["arrays"]["video"])
        name = f"{rio.safe_name(cam)}_q{q:05d}.npz"
        with np.load(rio.stage_dir(root, ep, "lift3d") / name) as te_out:
            np.testing.assert_array_equal(te_out["frame_indices"], np.asarray(idx))
    for q in (0, 8):
        name = f"rig0__left_q{q:05d}.npz"
        with np.load(rio.stage_dir(root, ep, "lift3d") / name) as z:
            after = z["frame_indices"] >= a.pause
            assert after.any() and (z["status"][after] == VALID).any()


def test_motion_export_and_viewer_consume_unmodified_contract(run, tmp_path):
    _, root, ep, _, _ = run
    rec = track_motion(root, ep, cfg=MotionFilterConfig(min_bin_samples=2, min_segment_samples=5,
                                                      robot_points_per_link=8))
    assert rec["summary"]["counts"]["candidate"] > 0
    tracks = load_camera_tracks(root, ep, "rig0/left")
    assert tracks is not None
    export(root, ep, tmp_path / "cache", depth_stage="sensor_depth", track_source="track_motion",
           allow_mock=True, resolutions={"tiny": (48, 64)}, max_gap_factor=1.5)
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["MOCK_MODEL_OUTPUTS"] and manifest["cameras"] == ["rig0/left", "rig1/left"]
    assert manifest["track_source"] == "track_motion"


def test_idempotency(run):
    _, root, ep, runner, _ = run
    calls = len(runner.calls)
    recs = track_everything(root, ep, MOCK_SPEC, runner, depth_stage="sensor_depth", grid_stride=4, model_hw=(32, 48))
    assert all(v["skipped"] for v in recs.values()) and len(runner.calls) == calls


def test_normalization_exact_birth_logits_and_status():
    E = make_T(np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]), [1., -2., 0.3])
    w2c = np.broadcast_to(E, (4, 4, 4)).copy()
    queries = np.array([[0, 2.5, 3.5], [2, 4.5, 5.5]], np.float32)
    cam = np.broadcast_to(np.array([[0., 0, 1], [0.2, 0.1, 2]]), (4, 2, 3)).copy()
    norm = 3.25
    coords = cam[None] / norm
    vis = np.full((1, 4, 2), 2., np.float32)
    vis[0, 1, 0] = -2
    cam[2, 0, 0] = 40  # out of image
    coords[0, 2, 0] = cam[2, 0] / norm
    out = te.predictions_to_base(coords, vis, norm, w2c, queries, np.full((1, 4, 2), 7.))
    assert np.isnan(out["xyz_base"][:2, 1]).all()
    assert np.isnan(out["vis_logit"][:2, 1]).all() and np.isnan(out["dyn_logit"][:2, 1]).all()
    np.testing.assert_allclose(out["xyz_base"][2:, 1], transform_points(np.linalg.inv(E), cam[2:, 1]), atol=2e-7)
    assert out["vis_logit"][0, 0] == 2  # output remains a logit
    K = np.array([[10., 0, 8], [0, 10, 8], [0, 0, 1]])
    t2, t3 = query_chunks(out, q=0, indices=list(range(5)), clip_start=0, query_slice=slice(0, 2),
                          timestamps_ns=np.arange(5), K=K, w2c=E, hw=(16, 16), source={"camera_id": "cam"})
    np.testing.assert_array_equal(t3["status"], [[VALID, NOT_TRACKED], [OCCLUDED, NOT_TRACKED],
                                               [OUT, VALID], [VALID, VALID], [NOT_TRACKED, NOT_TRACKED]])
    assert t2["visibility"][0, 0] == pytest.approx(1 / (1 + np.exp(-2)))


def test_continuous_resize_and_original_query_rays():
    video = np.zeros((1, 480, 640, 3), np.uint8)
    video[..., 0] = 123
    depth = np.arange(480 * 640, dtype=np.float32).reshape(1, 480, 640)
    K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    rgb, resized, km, scale = resize_inputs(video, depth, K, (384, 512))
    np.testing.assert_array_equal(km[:2], K[:2] * 0.8)
    np.testing.assert_array_equal(scale, [0.8, 0.8])
    assert rgb.shape == (1, 384, 512, 3) and (rgb[..., 0] == 123).all() and (rgb[..., 1:] == 0).all()
    rows = np.floor((np.arange(384) + .5) / .8).astype(int)
    cols = np.floor((np.arange(512) + .5) / .8).astype(int)
    np.testing.assert_array_equal(resized, depth[:, rows[:, None], cols[None, :]])
    xy = query_grid((480, 640), 16, 8)[:, ::-1] + .5
    rays = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(K).T
    rays_model = np.column_stack([xy * scale, np.ones(len(xy))]) @ np.linalg.inv(km).T
    np.testing.assert_allclose(rays, rays_model, atol=1e-15)


@pytest.mark.parametrize("window", [17, 64])
def test_te_refuses_multiple_inference_windows(window):
    with pytest.raises(ValueError, match="window in 1..16"):
        track_everything(None, "ep", None, None, depth_stage="sensor_depth", window=window)


def test_query_depth_uses_floor_after_continuous_resize_and_never_invents():
    # Original centre 8.5 -> resized coordinate 6.8. floor selects 6, while upstream round selected 7.
    depth = torch.ones((1, 16, 16))
    depth[0, 6, 6], depth[0, 7, 7] = 2.0, 9.0
    query = np.array([[0, 6.8, 6.8]], np.float32)
    K, E = torch.eye(3)[None], torch.eye(4)[None]
    out = te.continuous_query_unprojection(query, depth, K, E)
    np.testing.assert_allclose(out[0, 0], [0, 13.6, 13.6, 2.], atol=1e-6)
    depth[0, 6, 6] = 0
    with pytest.raises(ValueError, match="depth invention is disabled"):
        te.continuous_query_unprojection(query, depth, K, E)


def test_unsupported_births_are_counted_and_never_sent_to_te(tmp_path):
    a, root, ep = episode(tmp_path)
    # Constant depth isolates support: one centre hole, one discontinuity and one missing neighbour.
    for cam in ("rig0/left", "rig1/left"):
        with h5py.File(rio.stage_dir(root, ep, "sensor_depth") / f"{rio.safe_name(cam)}.h5", "r+") as h:
            d = np.ones_like(h["depth"][:])
            d[:, 2, 2] = 0
            d[:, 2, 5] = 1.1
            d[:, 2, 9] = 0
            h["depth"][:] = d
            h["valid"][:] = d > 0
    runner = SyntheticRunner(tmp_path / "worker", a)
    records = track_everything(root, ep, MOCK_SPEC, runner, depth_stage="sensor_depth", grid_stride=4, model_hw=a.hw)
    expected = [NO_DEPTH, EDGE, EDGE]
    rc = query_grid(a.hw, 4, 2)
    for call in runner.calls:
        uv = call["queries"][:, 1:]
        assert (depth_support_status(call["arrays"]["depth"][:1], uv[None]) == VALID).all()
        assert not any(np.array_equal(x, y) for x in uv for y in (rc[:3, ::-1] + .5))
    for cam in ("rig0/left", "rig1/left"):
        info = records["lift3d"]["summary"][cam]
        assert info["queries"] == 3 * len(rc)
        assert info["unsupported_birth_depth"] == 3 and info["unsupported_birth_edge"] == 6
        for name in info["chunks"]:
            with np.load(rio.stage_dir(root, ep, "lift3d") / name) as z:
                np.testing.assert_array_equal(z["status"][0, :3], expected)
                assert (z["status"][1:, :3] == NOT_TRACKED).all()
                assert np.isnan(z["xyz_base"][:, :3]).all()


def test_no_supported_births_skip_inference(tmp_path):
    a, root, ep = episode(tmp_path)
    for cam in ("rig0/left", "rig1/left"):
        with h5py.File(rio.stage_dir(root, ep, "sensor_depth") / f"{rio.safe_name(cam)}.h5", "r+") as h:
            h["depth"][:] = 0
    runner = SyntheticRunner(tmp_path / "worker", a)
    records = track_everything(root, ep, MOCK_SPEC, runner, depth_stage="sensor_depth", grid_stride=4, model_hw=a.hw)
    assert not runner.calls
    for cam in ("rig0/left", "rig1/left"):
        info = records["lift3d"]["summary"][cam]
        assert info["unsupported_birth_depth"] == info["queries"]
        assert info["unsupported_birth_edge"] == 0


def test_inference_hook_captures_dynamic_and_always_cleans_up():
    class FakeModel(torch.nn.Module):
        def forward(self):
            return SimpleNamespace(query_dynamic_logits=torch.full((1, 2, 1), 4.)), []

    class FakeTracker:
        def __init__(self):
            self.model = FakeModel()
            self.fail = False

        def infer(self, video, **kwargs):
            assert kwargs["mode"] == "sparse" and kwargs["use_input_resolution"] is True
            assert {"depths", "intrinsics", "extrinsics", "query_points_2d"} <= kwargs.keys()
            self.model()
            if self.fail:
                raise RuntimeError("inference interrupted")
            return SimpleNamespace(query_coords=torch.ones((1, 2, 1, 3)),
                                   query_visibs=torch.zeros((1, 2, 1)), norm_factor=2.)

    tracker = FakeTracker()
    args = (tracker, np.zeros((2, 16, 16, 3), np.uint8), np.ones((2, 16, 16), np.float32),
            np.broadcast_to(np.eye(3), (2, 3, 3)), np.broadcast_to(np.eye(4), (2, 4, 4)), np.array([[1, .5, .5]]), 3)
    out, norm = te.infer_queries(*args)
    assert norm == 2 and out["dyn_logit"][1, 0] == 4 and np.isnan(out["dyn_logit"][0, 0])
    assert not tracker.model._forward_hooks
    tracker.fail = True
    with pytest.raises(RuntimeError, match="interrupted"):
        te.infer_queries(*args)
    assert not tracker.model._forward_hooks


def test_local_loader_uses_pinned_package_and_hashes_once(tmp_path, monkeypatch):
    # Fake import modules, never the supplied read-only upstream model code.
    repo = tmp_path / "trackeverything"
    repo.mkdir()
    (repo / "demo_hf.py").touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test-only local placeholder")
    fake = ModuleType("trackeverything")
    demo = ModuleType("trackeverything.demo_hf")
    demo.__file__ = str(repo / "demo_hf.py")
    fake.demo_hf = demo
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(model=SimpleNamespace(seq_len=16))

    demo.TrackEverything = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "trackeverything", fake)
    monkeypatch.setitem(sys.modules, "trackeverything.demo_hf", demo)
    monkeypatch.setattr(te, "_CACHE", {})
    monkeypatch.setattr(te, "git_commit", lambda _: "d" * 40)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    hashes, sha = [], te.sha256

    def hashed(path):
        hashes.append(path)
        return sha(path)

    monkeypatch.setattr(te, "sha256", hashed)
    first = te._model(repo, checkpoint, "cuda")
    second = te._model(repo, checkpoint, "cuda")
    assert first is second and len(calls) == len(hashes) == 1
    assert calls[0] == (str(checkpoint), {"device": "cuda", "geometry_backend": "pi3"})
    assert sys.path[0] == str(repo.parent)
    assert demo._unproject_2d_to_3d is te.continuous_query_unprojection
    assert first[1]["commit"] == "d" * 40 and first[1]["checkpoint_sha256"] == sha(str(checkpoint))
    assert first[1]["is_mock"] is False


def test_nonfinite_predictions_and_points_behind_camera():
    xyz = np.array([[[0, 0, -1], [np.nan, 0, 1], [0, 0, 1], [0, 0, 1]]], np.float32)
    out = {"xyz_base": xyz, "vis_logit": np.array([[8, 8, np.nan, 8]], np.float32),
           "birth_t": np.array([0, 0, 0, 1])}
    _, t3 = query_chunks(out, q=0, indices=[0], clip_start=0, query_slice=slice(0, 4),
                         timestamps_ns=np.array([0]), K=np.eye(3), w2c=np.eye(4), hw=(16, 16), source={"camera_id": "c"})
    np.testing.assert_array_equal(t3["status"], [[OUT, NOT_TRACKED, NOT_TRACKED, NOT_TRACKED]])
    assert np.isnan(t3["xyz_base"]).all()


def test_resume_reuses_durably_saved_clips_after_ack_failure(tmp_path):
    a, root, ep = episode(tmp_path)
    runner = SyntheticRunner(tmp_path / "worker", a)
    ack = runner.ack

    def interrupted(handle):
        ack(handle)
        raise RuntimeError("connection lost after durable save")

    runner.ack = interrupted
    opts = {"depth_stage": "sensor_depth", "grid_stride": 4, "model_hw": (48, 64)}
    with pytest.raises(RuntimeError, match="connection lost"):
        track_everything(root, ep, MOCK_SPEC, runner, **opts)
    assert len(runner.calls) == 1
    assert rio.read_record(rio.stage_dir(root, ep, "track2d")) is None
    runner.ack = ack
    recs = track_everything(root, ep, MOCK_SPEC, runner, **opts)
    assert len(runner.calls) == 6 and all(not v["skipped"] for v in recs.values())
    assert not list(rio.stage_dir(root, ep, "track2d").parent.glob(".trackeverything_clips.chunks.*"))
    calls = len(runner.calls)
    track_everything(root, ep, MOCK_SPEC, runner, force=True, **opts)
    assert len(runner.calls) == calls + 6


def test_resume_after_track2d_swap_needs_no_new_inference(tmp_path, monkeypatch):
    a, root, ep = episode(tmp_path)
    runner = SyntheticRunner(tmp_path / "worker", a)
    run_stage = rio.run_stage

    def fail_second_swap(*args, **kwargs):
        if kwargs["stage"] == "lift3d":
            raise RuntimeError("interrupted before lift3d swap")
        return run_stage(*args, **kwargs)

    opts = {"depth_stage": "sensor_depth", "grid_stride": 4, "model_hw": (32, 48)}
    monkeypatch.setattr(rio, "run_stage", fail_second_swap)
    with pytest.raises(RuntimeError, match="before lift3d swap"):
        track_everything(root, ep, MOCK_SPEC, runner, **opts)
    assert rio.read_record(rio.stage_dir(root, ep, "track2d")) is not None
    assert len(runner.calls) == 6
    monkeypatch.setattr(rio, "run_stage", run_stage)
    recs = track_everything(root, ep, MOCK_SPEC, runner, **opts)
    assert recs["track2d"]["skipped"] and not recs["lift3d"]["skipped"] and len(runner.calls) == 6


def test_worker_refuses_download_or_multiview_and_mock_without_opt_in(tmp_path):
    with pytest.raises(ValueError, match="existing local file"):
        te._model(tmp_path / "trackeverything", "katefgroup/TrackEverything", "cuda")
    with pytest.raises(ValueError, match="only single-view"):
        te.handle({"task": "multiview", "params": {}}, {})
    with pytest.raises(ValueError, match="uint8 RGB"):
        te.validate_inputs({"video": np.zeros((1, 2, 16, 16, 3)), "depth": [], "K": [], "w2c": [], "queries": []})
    # Worker protocol rejects a mock identity unless the caller explicitly opts in.
    with pytest.raises(WorkerError, match="mock output refused"):
        call_worker(WorkerSpec("mock", sys.executable), "echo_tracks", {}, {"flows": np.zeros(1), "visconf": np.zeros(1)},
                    tmp_path / "protocol")
