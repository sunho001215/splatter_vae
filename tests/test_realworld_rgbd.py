"""CPU-only RGB-D contracts, recorder decoding, geometry, durable stages and depth-only calibration."""

import importlib.util
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from s4d.realworld import io as rio
from s4d.realworld import pipeline
from s4d.realworld.adapters.panda_teleop import PandaTeleopH5Adapter
from s4d.realworld.calibration import CalibrationConfig, calibrate_camera, explained_fraction, moving_masks
from s4d.realworld.contract import (
    CameraStream,
    CameraView,
    EpisodeContractError,
    EpisodeMeta,
    Intrinsics,
    Provenance,
    RgbdGeometry,
    RobotDescription,
    RobotTrajectory,
    StereoGeometry,
    meta_from_json,
    meta_to_json,
    validate_episode,
)
from s4d.realworld.depth_stages import frame_sets, load_view_depth, sensor_depth
from s4d.realworld.export import export
from s4d.realworld.geometry import (
    invert_T,
    make_T,
    pixel_centers,
    resize_affine,
    rotation_angle_deg,
    warp_affine,
)
from s4d.realworld.kernels import register_depth
from s4d.realworld.runners import LocalRunner
from s4d.realworld.synthetic import synthetic_robot_urdf


@pytest.fixture
def rgbd_meta():
    intr = Intrinsics(32, 24, 30, 31, 15.5, 11.5, "integer_centers")
    color = CameraStream("rig/color", "rig", "mono", "rgb", 3, "RGB", "uint8", intr, True)
    depth = CameraStream("rig/depth", "rig", "mono", "depth", 1, "DEPTH", "uint16", intr, True,
                         depth_scale_m=0.001, depth_kind="z")
    stamps = np.arange(6, dtype=np.int64) * 33_333_333 + 1_000_000_000
    return EpisodeMeta(
        "rgbd", (color, depth), (), RobotDescription("franka_panda", "unknown", None, None),
        RobotTrajectory(stamps.copy(), np.zeros((6, 7)), np.zeros(6)),
        {color.stream_id: stamps, depth.stream_id: stamps + 40_000}, Provenance("test", "1", "synthetic://"),
        clock="synthetic", rgbd=(RgbdGeometry("rig", color.stream_id, depth.stream_id, "metric",
                                            tuple(map(tuple, np.eye(4))), max_pair_offset_ns=100_000),))


def test_rgbd_contract_and_disk_round_trip(rgbd_meta, tmp_path):
    assert validate_episode(rgbd_meta).errors == []
    doc, arrays = meta_to_json(rgbd_meta)
    rio.atomic_write_json(tmp_path / "episode.json", doc)
    rio.atomic_write_npz(tmp_path / "arrays.npz", **arrays)
    with np.load(tmp_path / "arrays.npz", allow_pickle=False) as z:
        loaded = meta_from_json(json.loads((tmp_path / "episode.json").read_text()), dict(z))
    assert loaded.rgbd == rgbd_meta.rgbd
    assert loaded.streams == rgbd_meta.streams
    np.testing.assert_array_equal(loaded.trajectory.joint_positions, rgbd_meta.trajectory.joint_positions)
    for sid, stamps in rgbd_meta.stream_timestamps_ns.items():
        np.testing.assert_array_equal(loaded.stream_timestamps_ns[sid], stamps)
    assert validate_episode(loaded).errors == []


@pytest.mark.parametrize("change, message", [
    ({"channels": 3}, "1-channel DEPTH uint16"),
    ({"dtype": "uint8"}, "1-channel DEPTH uint16"),
    ({"channel_order": "GRAY"}, "1-channel DEPTH uint16"),
    ({"depth_scale_m": None}, "positive depth_scale_m"),
    ({"depth_kind": "ray_length"}, "depth_kind"),
])
def test_depth_stream_contract_refusals(rgbd_meta, change, message):
    meta = replace(rgbd_meta, streams=(rgbd_meta.streams[0], replace(rgbd_meta.streams[1], **change)))
    with pytest.raises(EpisodeContractError, match=message):
        validate_episode(meta).raise_if_errors()


@pytest.mark.parametrize("case, message", [
    ("rgb_scale", "only apply to depth"),
    ("count", "frame counts differ"),
    ("offset", "frames up to.*apart"),
    ("tolerance", "pairing tolerance"),
    ("nonrigid", "rigid T_color_depth"),
    ("duplicate", "rig is declared twice"),
])
def test_rgbd_rig_contract_refusals(rgbd_meta, case, message):
    meta, g = rgbd_meta, rgbd_meta.rgbd[0]
    if case == "rgb_scale":
        meta = replace(meta, streams=(replace(meta.streams[0], depth_scale_m=0.001), meta.streams[1]))
    elif case in ("count", "offset"):
        stamps = meta.stream_timestamps_ns[g.depth_stream]
        stamps = stamps[:-1] if case == "count" else stamps + g.max_pair_offset_ns
        meta = replace(meta, stream_timestamps_ns={**meta.stream_timestamps_ns, g.depth_stream: stamps})
    elif case == "tolerance":
        meta = replace(meta, rgbd=(replace(g, max_pair_offset_ns=None),))
    elif case == "nonrigid":
        T = np.eye(4)
        T[0, 0] = 1.1
        meta = replace(meta, rgbd=(replace(g, T_color_depth=tuple(map(tuple, T))),))
    elif case == "duplicate":
        meta = replace(meta, rgbd=(g, g))
    with pytest.raises(EpisodeContractError, match=message):
        validate_episode(meta).raise_if_errors()


def _with_stereo(meta):
    """Put streams in reverse order to ensure views follow rig declarations, not stream order or sorting."""
    streams, stereo, stamps = list(meta.streams), [], dict(meta.stream_timestamps_ns)
    for rig in ("z_stereo", "a_stereo"):
        left, right = f"{rig}/left", f"{rig}/right"
        for sid, eye in ((right, "right"), (left, "left")):
            streams.insert(0, replace(meta.streams[0], stream_id=sid, rig_id=rig, eye=eye))
            stamps[sid] = meta.trajectory.timestamps_ns.copy()
        stereo.append(StereoGeometry(rig, left, right, "metric",
                                     tuple(map(tuple, make_T(np.eye(3), [-0.05, 0, 0]))), 0.05))
    return replace(meta, streams=tuple(streams), stereo=tuple(stereo), stream_timestamps_ns=stamps)


def test_views_are_stereo_then_rgbd_in_declaration_order(rgbd_meta):
    meta = _with_stereo(rgbd_meta)
    g = meta.rgbd[0]
    other = replace(g, rig_id="z_rgbd", color_stream="z_rgbd/color", depth_stream="z_rgbd/depth")
    other_streams = tuple(replace(s, rig_id="z_rgbd", stream_id=s.stream_id.replace("rig/", "z_rgbd/"))
                          for s in rgbd_meta.streams)
    meta = replace(meta, rgbd=(other, g), streams=meta.streams + other_streams,
                   stream_timestamps_ns={**meta.stream_timestamps_ns,
                                         **{s.stream_id: meta.stream_timestamps_ns[original.stream_id]
                                            for s, original in zip(other_streams, rgbd_meta.streams)}})
    assert validate_episode(meta).errors == []
    assert meta.views() == [CameraView("z_stereo", "z_stereo/left", "stereo"),
                            CameraView("a_stereo", "a_stereo/left", "stereo"),
                            CameraView("z_rgbd", "z_rgbd/color", "rgbd"), CameraView("rig", "rig/color", "rgbd")]
    # Duplicating a rig across geometry types is rejected too.
    bad = replace(rgbd_meta, stereo=(StereoGeometry("rig", "rig/color", "rig/depth", "missing"),))
    assert any("rig is declared twice" in e for e in validate_episode(bad).errors)


def test_schema_1_0_document_without_rgbd_still_loads(rgbd_meta):
    meta = _with_stereo(rgbd_meta)
    meta = replace(meta, rgbd=(), streams=tuple(s for s in meta.streams if s.rig_id != "rig"),
                   stream_timestamps_ns={sid: ts for sid, ts in meta.stream_timestamps_ns.items()
                                         if not sid.startswith("rig/")},
                   schema_version="s4d.realworld.episode/1.0.0")
    doc, arrays = meta_to_json(meta)
    del doc["rgbd"]
    del doc["conventions"]["sensor_depth"]
    for stream in doc["streams"]:
        del stream["depth_scale_m"], stream["depth_kind"]
    loaded = meta_from_json(json.loads(json.dumps(doc)), arrays)
    assert loaded.rgbd == () and loaded.schema_version.endswith("/1.0.0")
    assert validate_episode(loaded).errors == []


def _encoded(array, fmt):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt)
    return np.frombuffer(buf.getvalue(), np.uint8)


@pytest.fixture
def panda_file(tmp_path):
    """Small recorder v2 schema: independently encoded RGB JPEG and native z16 PNG for two devices."""
    path = tmp_path / "episode.h5"
    expected = {}
    t0 = 123.1234567894
    robot_t = t0 + np.arange(8) / 30 - 0.02
    gripper_t, positions = t0 + np.array([0.01, 0.08, 0.15]), np.array([3.0, 120.0, 230.0])
    with h5py.File(path, "w") as f:
        f.attrs.update(format="panda_teleop_episode", format_version=2,
                       conventions_json=json.dumps({"time": "host CLOCK_MONOTONIC", "length": "m"}),
                       T_flange_controller_ee=np.eye(4), T_flange_tcp=np.eye(4), camera_calibration="",
                       hostname="synthetic-recorder", software_version="test", task_name="test", status="complete")
        f["segments/t_start"], f["segments/t_end"] = [t0], [t0 + 0.2]
        for name, a in {"t": robot_t, "q": np.arange(56).reshape(8, 7) * 0.001, "dq": np.zeros((8, 7)),
                        "tau_ext": np.zeros((8, 7)), "joint_frames": np.tile([0, 0, 0, 0, 0, 0, 1], (8, 7, 1)),
                        "flange_pose": np.tile([0.1, 0.2, 0.3, 0, 0, 0, 1], (8, 1))}.items():
            f[f"robot/{name}"] = a
        f["gripper/t"], f["gripper/position"] = gripper_t, positions
        for c, serial in enumerate(("101", "202")):
            g = f.create_group(f"cameras/cam_{serial}")
            color_t = t0 + np.arange(6) / 30 + c * 0.0001
            depth_t = color_t + 0.000040123
            T = make_T(Rotation.from_euler("xyz", [0.002, -0.004, 0.003]).as_matrix(), [0.015, -0.001, 0.002])
            cal = {"streams": {}, "depth_scale": 0.001 / (c + 1), "calibration_id": f"cal-{serial}",
                   "device": {"serial_number": serial, "name": "Synthetic D400", "sdk_version": "test",
                              "firmware_version": "test"},
                   "emitter": {"emitter_enabled": True, "laser_power": 150},
                   "extrinsics_convention": "p_target = R_target_from_source @ p_source + t_target_from_source_m",
                   "extrinsics": {"color_from_depth": {"source": "depth", "target": "color",
                                  "R_target_from_source": T[:3, :3].tolist(),
                                  "t_target_from_source_m": T[:3, 3].tolist()}}}
            for kind, (h, w), fx in (("color", (24, 32), 30.2), ("depth", (28, 36), 32.7)):
                cal["streams"][kind] = {"width": w, "height": h, "fps": 30,
                                        "format": "bgr8" if kind == "color" else "z16",
                                        "intrinsics": {"width": w, "height": h, "fx": fx + c, "fy": fx + 0.7,
                                                       "ppx": w / 2 - 0.7, "ppy": h / 2 - 0.3,
                                                       "distortion_model": "none", "distortion_coeffs": [0] * 5}}
            g.attrs["calibration_json"] = json.dumps(cal)
            g["t"], g["stereo/t"] = color_t, depth_t
            for name in ("t_valid", "stereo/t_valid", "stereo/depth_ok"):
                g[name] = np.ones(6, np.uint8)
            g["segment"] = np.zeros(6, np.int32)
            g["stereo/color_index"] = np.arange(6)
            g["stereo/color_offset_ms"] = (color_t - depth_t) * 1000
            g["hw_timestamp_ms"], g["t_recv"] = color_t * 1000, color_t + 0.065
            g["stereo/hw_timestamp_ms"], g["stereo/t_recv"] = depth_t * 1000, depth_t + 0.065
            colors, depths = [], []
            for k in range(6):
                color = np.empty((24, 32, 3), np.uint8)
                color[:] = [180 + k, 30 + c * 20, 10 + k * 2]  # visibly different R/B values
                raw = (np.arange(28 * 36).reshape(28, 36) + 1000 + k * 100 + c * 1000).astype(np.uint16)
                raw[0, :4] = [0, 50, 8000, 50000]  # invalid, near, far, high unsigned value
                depths.append(raw)
                jpeg = _encoded(color, "JPEG")
                colors.append(np.asarray(Image.open(io.BytesIO(jpeg.tobytes())).convert("RGB")))
                for name, payload in (("color", jpeg), ("stereo/depth", _encoded(raw, "PNG"))):
                    if name not in g:
                        g.create_dataset(name, (6,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                    g[name][k] = payload
            expected[serial] = {"cal": cal, "T": T, "color_t": color_t, "depth_t": depth_t,
                                "colors": colors, "depths": depths}
    adapter = PandaTeleopH5Adapter({"episodes": {"panda": str(path)}})
    return SimpleNamespace(path=path, adapter=adapter, expected=expected, robot_t=robot_t,
                           grip=np.interp(robot_t, gripper_t, positions))


def test_panda_metadata_uses_recorded_calibration_and_timestamps(panda_file):
    p = panda_file
    meta = p.adapter.read_meta("panda")
    assert validate_episode(meta).errors == []
    assert p.adapter.list_episodes() == ["panda"]
    assert len(meta.rgbd) == 2 and len(meta.streams) == 4 and meta.stereo == ()
    for serial, exp in p.expected.items():
        for kind in ("color", "depth"):
            stream = meta.stream(f"{serial}/{kind}")
            intr = stream.intrinsics
            rec = exp["cal"]["streams"][kind]["intrinsics"]
            assert intr.pixel_convention == "integer_centers"
            assert (intr.width, intr.height) == (rec["width"], rec["height"])
            K_integer = np.array([[rec["fx"], 0, rec["ppx"]], [0, rec["fy"], rec["ppy"]], [0, 0, 1]])
            np.testing.assert_array_equal(intr.K("integer_centers"), K_integer)
            K_cont = K_integer.copy()
            K_cont[:2, 2] += 0.5
            np.testing.assert_array_equal(intr.K(), K_cont)
            expected_ns = [round(float(t) * 1e9) for t in exp[f"{kind}_t"]]
            np.testing.assert_array_equal(meta.stream_timestamps_ns[stream.stream_id], expected_ns)
        ds = meta.stream(f"{serial}/depth")
        assert ds.depth_scale_m == exp["cal"]["depth_scale"] and ds.depth_kind == "z"
        np.testing.assert_array_equal(meta.rig_rgbd(serial).T(), exp["T"])
    np.testing.assert_array_equal(meta.trajectory.timestamps_ns, [round(float(t) * 1e9) for t in p.robot_t])
    np.testing.assert_array_equal(meta.trajectory.gripper_state, p.grip)
    assert meta.robot.gripper_state_semantics == "robotiq_position_register"
    assert meta.trajectory.ee_frame == "panda_link8"
    np.testing.assert_array_equal(meta.trajectory.ee_pose_base[:, :3, 3], np.tile([0.1, 0.2, 0.3], (8, 1)))


def test_panda_decodes_rgb_and_lossless_unsigned_depth(panda_file):
    for serial, exp in panda_file.expected.items():
        for kind, dtype, shape, images in (("color", np.uint8, (24, 32, 3), exp["colors"]),
                                          ("depth", np.uint16, (28, 36), exp["depths"])):
            frames = list(panda_file.adapter.iter_frames("panda", f"{serial}/{kind}"))
            assert len(frames) == 6
            for k, (index, stamp, image) in enumerate(frames):
                assert index == k and stamp == round(float(exp[f"{kind}_t"][k]) * 1e9)
                assert image.dtype == dtype and image.shape == shape
                np.testing.assert_array_equal(image, images[k])
            subset = list(panda_file.adapter.iter_frames("panda", f"{serial}/{kind}", 2, 4))
            assert [i for i, _, _ in subset] == [2, 3]
            np.testing.assert_array_equal(subset[0][2], images[2])


@pytest.mark.parametrize("case, message", [("pairing", "not paired 1:1"), ("format", "not a panda_teleop_episode v2"),
                                          ("color_invalid", "flagged by the recorder"),
                                          ("depth_invalid", "flagged by the recorder")])
def test_panda_refuses_invalid_recordings(panda_file, case, message):
    with h5py.File(panda_file.path, "r+") as f:
        if case == "format":
            f.attrs["format"] = "another_recorder"
        elif case == "pairing":
            f["cameras/cam_101/stereo/color_index"][:] = [1, 0, 2, 3, 4, 5]
        else:
            name = "t_valid" if case == "color_invalid" else "stereo/t_valid"
            f[f"cameras/cam_101/{name}"][2] = 0
    with pytest.raises(ValueError, match=message):
        panda_file.adapter.read_meta("panda")


@pytest.mark.parametrize("normal", [(0, 0, 1), (0.06, -0.04, 1)], ids=["frontoparallel", "tilted"])
def test_register_depth_matches_analytic_color_plane(normal):
    hw_d, hw_c = (180, 240), (160, 220)
    K_d = np.array([[260.0, 0, 119.3], [0, 261.0, 90.2], [0, 0, 1]])
    K_c = np.array([[255.0, 0, 110.4], [0, 257.0, 79.6], [0, 0, 1]])
    R = Rotation.from_euler("xyz", [0.15, -0.2, 0.1], degrees=True).as_matrix()
    T = make_T(R, [0.015, 0, 0])
    n, distance, scale = np.asarray(normal), 1.2, 0.0001
    # Independent ray/plane intersection: n_depth . X_depth = distance.
    uv_d = pixel_centers(*hw_d)
    rays_d = np.concatenate([uv_d, np.ones((*hw_d, 1))], axis=-1) @ np.linalg.inv(K_d).T
    raw = np.rint(distance / (rays_d @ n) / scale).astype(np.uint16)
    raw[70:73, 100:104], raw[100:102, 150:155], raw[80:82, 140:142] = 50, 65000, 0
    out, stats = register_depth({"color_hw": hw_c, "depth_scale_m": scale, "min_depth_m": 0.1,
                                 "max_depth_m": 3.0, "splat": 2},
                                {"depth_raw": raw[None], "K_depth": K_d, "K_color": K_c, "T_color_depth": T})
    uv_c = pixel_centers(*hw_c)
    rays_c = np.concatenate([uv_c, np.ones((*hw_c, 1))], axis=-1) @ np.linalg.inv(K_c).T
    n_c = R @ n
    analytic = (distance + n_c @ T[:3, 3]) / (rays_c @ n_c)
    valid, depth = out["valid"][0], out["depth"][0]
    assert depth.dtype == np.float32 and valid.dtype == bool
    np.testing.assert_array_equal(valid, depth > 0)
    assert valid[8:-8, 8:-8].mean() > 0.9
    assert np.abs(depth[valid] - analytic[valid]).max() < 0.001
    assert depth.max() <= 3.0 and (depth[~valid] == 0).all()
    assert stats["frames"][0]["rejected_below_min"] == 12
    assert stats["frames"][0]["rejected_above_max"] == 10
    assert stats["frames"][0]["sensor_valid_fraction"] == pytest.approx((raw > 0).mean())


@pytest.fixture
def mapanything_helpers(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "s4d/realworld/workers/mapanything_worker.py"
    # The standalone worker inserts its protocol directory; keep that import-path change local to this test.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("rgbd_mapanything_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mapanything_nearest_resize_matches_geometry_both_directions(mapanything_helpers):
    helpers = mapanything_helpers
    rng = np.random.default_rng(2026)
    a = rng.integers(0, 65536, (2, 480, 640), dtype=np.uint16)
    for hw in ((392, 518), (480, 640)):
        aff = resize_affine(a.shape[-2:], hw)
        expected = np.stack([warp_affine(image, aff, hw, order=0) for image in a])
        resized = helpers.resize_nearest(a, hw)
        np.testing.assert_array_equal(resized, expected)
        for n_dst, n_src in zip(hw, a.shape[-2:]):
            idx = helpers.nearest_index(n_dst, n_src)
            np.testing.assert_array_equal(idx, np.floor((np.arange(n_dst) + 0.5) * n_src / n_dst).astype(int))
        a = resized


def test_mapanything_intrinsics_and_integer_pixel_rays(mapanything_helpers):
    helpers = mapanything_helpers
    K = np.array([[603.2, 1.3, 318.8], [0, 597.4, 241.1], [0, 0, 1]])
    for src, dst in (((480, 640), (392, 518)), ((392, 518), (480, 640))):
        scaled = helpers.scaled_K(K, src, dst)
        np.testing.assert_allclose(scaled, resize_affine(src, dst).apply_K(K), rtol=0, atol=1e-12)
        before = scaled.copy()
        K_ma = helpers.to_mapanything_K(scaled)
        np.testing.assert_array_equal(scaled, before)
        ij = np.array([[0, 0], [dst[0] - 1, dst[1] - 1], [123, 245]])
        xy1 = np.column_stack([ij[:, ::-1], np.ones(len(ij))])
        ours = (xy1 + [0.5, 0.5, 0]) @ np.linalg.inv(scaled).T
        upstream = xy1 @ np.linalg.inv(K_ma).T
        np.testing.assert_allclose(upstream, ours, rtol=0, atol=1e-14)


def test_nearest_affine_preserves_pixel_cells_and_zeroes_outside_image():
    from s4d.realworld.geometry import PixelAffine

    image = np.array([[1, 2], [3, 4]], np.uint16)
    # Each pixel occupies a 2x2 block after upscaling, with a one-pixel zero border from translation.
    out = warp_affine(image, PixelAffine(2, 2, 1, 1), (6, 6), order=0)
    expected = np.pad(np.repeat(np.repeat(image, 2, axis=0), 2, axis=1), 1)
    np.testing.assert_array_equal(out, expected)


def test_frame_sets_match_indices_timestamps_and_spread():
    ref = np.array([1_000, 1_100, 1_200, 1_300], np.int64)
    stamps = {"a": ref, "b": ref + [2, -3, 4, -1], "c": ref + [-4, 1, 0, 3]}
    sets = frame_sets(SimpleNamespace(stream_timestamps_ns=stamps), list(stamps), max_sync_ns=5)
    np.testing.assert_array_equal(sets["index"], np.tile(np.arange(4)[:, None], (1, 3)))
    np.testing.assert_array_equal(sets["stamps_ns"], np.stack(list(stamps.values()), axis=1))
    np.testing.assert_array_equal(sets["spread_ns"], [6, 4, 4, 4])
    assert sets["streams"] == list(stamps) and sets["max_sync_ns"] == 5
    assert frame_sets(SimpleNamespace(stream_timestamps_ns=stamps), list(stamps))["max_sync_ns"] == 33


@pytest.mark.parametrize("other, tolerance", [([0, 100, 200, 300, 400], 50),
                                             ([0, 100, 300, 400], 150), ([40, 140, 240, 340], 30)],
                         ids=["extra-frame", "reused-frame", "too-far"])
def test_frame_sets_refuse_nonbijective_or_unsynchronized_frames(other, tolerance):
    meta = SimpleNamespace(stream_timestamps_ns={"ref": np.array([0, 100, 200, 300]), "other": np.array(other)})
    with pytest.raises(ValueError, match="do not pair 1:1"):
        frame_sets(meta, ["ref", "other"], max_sync_ns=tolerance)


def test_chunk_store_durability_corruption_fingerprints_and_clear(tmp_path):
    stage = tmp_path / "sensor_depth"
    fp = rio.stage_fingerprint("sensor_depth", "1", {"chunk": 2}, {"ingest": "original"})
    assert fp == rio.stage_fingerprint("sensor_depth", "1", {"chunk": 2}, {"ingest": "original"})
    store = rio.ChunkStore(stage, fp)
    arrays = {"depth": np.arange(24, dtype=np.float32).reshape(2, 3, 4), "valid": np.ones((2, 3, 4), bool)}
    assert store.get("missing") is None
    rec = store.put("rig/color/0-1", arrays, {"frames": [{"rejected_above_max": 3}]})
    got, metadata = rio.ChunkStore(stage, fp).get("rig/color/0-1")
    for name, array in arrays.items():
        np.testing.assert_array_equal(got[name], array)
    assert metadata == {"frames": [{"rejected_above_max": 3}]}
    with open(store.dir / store.MANIFEST, "ab") as f:
        f.write(b'{"chunk_id": "torn')
    recovered = rio.ChunkStore(stage, fp)
    assert len(recovered.records()) == 1 and recovered.get("rig/color/0-1") is not None
    recovered.put("rig/color/2-3", arrays)
    assert rio.ChunkStore(stage, fp).get("rig/color/2-3") is not None  # append after a crash stays durable
    (store.dir / rec["file"]).write_bytes(b"corrupted npz")
    assert rio.ChunkStore(stage, fp).get("rig/color/0-1") is None
    other_fp = rio.stage_fingerprint("sensor_depth", "1", {"chunk": 3}, {"ingest": "original"})
    other = rio.ChunkStore(stage, other_fp)
    assert store.dir != other.dir and other.get("rig/color/2-3") is None
    other.put("separate", arrays)
    recovered.clear()
    assert not recovered.dir.exists() and recovered.get("rig/color/2-3") is None
    assert other.get("separate") is not None
    other.clear()
    assert not other.dir.exists()


def test_sensor_depth_local_runner_resumes_only_remaining_chunks(panda_file, tmp_path, monkeypatch):
    ep = "panda"
    roots = [tmp_path / "uninterrupted", tmp_path / "resumed"]
    for root in roots:
        pipeline.ingest(panda_file.adapter, ep, root, chunk=2)
    full = LocalRunner(tmp_path / "worker")
    all_calls = []
    kernel = full.kernel

    def counted(task, params, arrays, *, chunk_id):
        all_calls.append(chunk_id)
        return kernel(task, params, arrays, chunk_id=chunk_id)

    monkeypatch.setattr(full, "kernel", counted)
    complete = sensor_depth(roots[0], ep, full, chunk=2)
    assert len(all_calls) == 6

    class Interrupted(LocalRunner):
        def __init__(self):
            super().__init__(tmp_path / "worker")
            self.calls, self.acked = [], []

        def kernel(self, task, params, arrays, *, chunk_id):
            if len(self.calls) == 4:
                raise RuntimeError("simulated interruption")
            self.calls.append(chunk_id)
            outputs, meta, _ = super().kernel(task, params, arrays, chunk_id=chunk_id)
            return outputs, meta, chunk_id

        def ack(self, handle):
            self.acked.append(handle)

    interrupted = Interrupted()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sensor_depth(roots[1], ep, interrupted, chunk=2)
    directory = rio.stage_dir(roots[1], ep, "sensor_depth")
    assert rio.read_record(directory) is None
    assert interrupted.acked == interrupted.calls == all_calls[:4]
    remaining = []
    plain = LocalRunner(tmp_path / "worker")
    plain_kernel = plain.kernel

    def count_remaining(task, params, arrays, *, chunk_id):
        remaining.append(chunk_id)
        return plain_kernel(task, params, arrays, chunk_id=chunk_id)

    monkeypatch.setattr(plain, "kernel", count_remaining)
    resumed = sensor_depth(roots[1], ep, plain, chunk=2)
    assert remaining == all_calls[4:]
    assert complete["summary"] == resumed["summary"]
    assert not list(directory.parent.glob(".sensor_depth.chunks.*"))
    meta = pipeline.load_meta(roots[1], ep)
    assert pipeline.default_depth_stage(meta, "calibrate") == "sensor_depth"
    assert pipeline.default_depth_stage(meta, "lift") == "mapanything_depth"
    for view in meta.views():
        sid = view.stream_id
        a = load_view_depth(roots[0], ep, sid, np.arange(6), "sensor_depth")
        b = load_view_depth(roots[1], ep, sid, np.arange(6), "sensor_depth")
        for actual, expected in zip(b, a):
            np.testing.assert_array_equal(actual, expected)
        assert a[0].dtype == np.float32 and a[1].dtype == bool and a[1].mean() > 0.8
        np.testing.assert_array_equal(a[1], a[0] > 0)
        np.testing.assert_array_equal(a[2], meta.stream(sid).intrinsics.K())
        assert complete["summary"][sid]["rejected_above_max_total"] > 0
        np.testing.assert_array_equal(pipeline.view_depth(roots[1], ep, meta, sid, [5, 0], "sensor_depth")[0],
                                      a[0][[5, 0]])
        np.testing.assert_array_equal(pipeline.view_frames(roots[1], ep, meta, sid, [3, 1]),
                                      np.stack([panda_file.expected[view.rig_id]["colors"][i] for i in (3, 1)]))
    assert sensor_depth(roots[1], ep, plain, chunk=2)["skipped"]
    assert remaining == all_calls[4:]


@pytest.fixture(scope="module")
def depth_only_robot():
    from test_realworld_calibration import HW, T_TRUE, K, make_episode

    frames = make_episode(T_TRUE, n=20, outlier_rate=0.0, depth_noise=0.002)
    for f in frames:
        f.landmarks_base, f.landmarks_uv, f.landmarks_conf = np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
    return frames, K, HW, T_TRUE


def test_rgbd_calibration_recovers_pose_without_prior_or_landmarks(depth_only_robot):
    frames, K, hw, truth = depth_only_robot
    cfg = CalibrationConfig(evidence="rgbd", image_hw=hw, init_axes=100, init_rolls=12, init_top=8,
                            init_frames=6, init_obs_points=250)
    result = calibrate_camera("rig/color", frames, K, cfg)  # deliberately no T_init
    assert result.status == "accepted", (result.reasons, result.diagnostics)
    assert result.init_method == "rgbd_robot_registration"
    assert result.diagnostics["landmark_observations"] == 0
    assert rotation_angle_deg(result.T_cam_base[:3, :3], truth[:3, :3]) < 1.0
    assert np.linalg.norm(invert_T(result.T_cam_base)[:3, 3] - invert_T(truth)[:3, 3]) < 0.01
    assert result.diagnostics["depth_abs_median_m"] < cfg.max_depth_median_m
    assert result.diagnostics["heldout_depth_abs_median_m"] < cfg.max_heldout_depth_median_m
    assert result.diagnostics["explained_fraction_heldout"] >= cfg.min_explained_fraction
    assert explained_fraction(truth, frames, K, cfg) > cfg.min_explained_fraction
    wrong = truth.copy()
    wrong[:3, 3] += [0.3, 0.2, 0]
    assert explained_fraction(wrong, frames, K, cfg) < cfg.min_explained_fraction


def test_rgbd_calibration_rejects_static_robot(depth_only_robot):
    frames, K, hw, _ = depth_only_robot
    rng = np.random.default_rng(7)
    static = [replace(frames[0], t_index=i, timestamp_ns=f.timestamp_ns,
                      depth=(frames[0].depth + rng.normal(0, 0.002, hw)).astype(np.float32))
              for i, f in enumerate(frames)]
    assert not moving_masks(np.stack([f.depth for f in static]), 0.03).any()
    result = calibrate_camera("rig/color", static, K, CalibrationConfig(evidence="rgbd", image_hw=hw))
    assert result.status == "rejected" and result.T_cam_base is None
    assert any("too little robot motion" in reason for reason in result.reasons)


def test_export_drops_windows_crossing_recording_gap(rgbd_meta, tmp_path):
    """Tiny real export with prepared upstream records; no model or calibration fitting needed."""
    stamps = np.arange(8, dtype=np.int64) * 33_333_333 + 1_000_000_000
    stamps[4:] += 5_000_000_000
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(synthetic_robot_urdf())
    robot = replace(rgbd_meta.robot, gripper_model="synthetic_two_finger", urdf_path=str(urdf),
                    gripper_base_frame="gripper_base", gripper_joint="finger_joint")
    meta = replace(rgbd_meta, robot=robot,
                   trajectory=RobotTrajectory(stamps, np.zeros((8, 7)), np.zeros(8)),
                   stream_timestamps_ns={s.stream_id: stamps.copy() for s in rgbd_meta.streams})

    class Adapter:
        name, version, config = "rgbd-export-fixture", "1", {}

        def read_meta(self, episode_id):
            return meta

        def source_files(self, episode_id):
            return []

        def iter_frames(self, episode_id, stream_id, start=0, stop=None):
            for i in range(start, 8 if stop is None else stop):
                image = np.full((24, 32, 3), 60, np.uint8) if stream_id.endswith("color") else \
                    np.full((24, 32), 1000, np.uint16)
                yield i, int(stamps[i]), image

    root, ep, cam = tmp_path / "out", meta.episode_id, "rig/color"
    pipeline.ingest(Adapter(), ep, root)
    sensor_depth(root, ep, LocalRunner(tmp_path / "worker"), chunk=2, splat=1)
    for stage in ("calibrate", "track2d", "lift3d"):
        directory = rio.stage_dir(root, ep, stage)
        rio.atomic_write_json(directory / rio.RECORD,
                              {"build_id": stage, "summary": {cam: {"chunks": ["tracks.npz"]}},
                               "config": {"is_mock": True}})
    rio.atomic_write_json(rio.stage_dir(root, ep, "calibrate") / "rig__color.json",
                          {"status": "accepted", "T_cam_base": np.eye(4).tolist()})
    rio.atomic_write_npz(rio.stage_dir(root, ep, "lift3d") / "tracks.npz", frame_indices=np.arange(8), query_frame=0,
                         point_ids=np.array([0]), xyz_base=np.tile([[[0, 0, 1.0]]], (8, 1, 1)),
                         status=np.zeros((8, 1), np.uint8))
    manifests = []
    for gap in (None, 1.5):
        out = tmp_path / f"cache-{gap}"
        export(root, ep, out, resolutions={"tiny": (12, 16)}, depth_stage="sensor_depth", allow_mock=True,
               max_gap_factor=gap)
        manifests.append(json.loads((out / "manifest.json").read_text()))
    unfiltered, filtered = manifests
    assert len(unfiltered["windows"]) == 10
    assert unfiltered["windows_dropped_for_time_gaps"] == 0
    assert filtered["windows_dropped_for_time_gaps"] == 6
    assert [w["canonical_indices"] for w in filtered["windows"]] == [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]]
    assert filtered["depth_stage"] == "sensor_depth"


def test_rgbd_mapanything_preserves_sensor_input_and_primary_outputs(panda_file, tmp_path, monkeypatch):
    from s4d.realworld import depth_stages as D
    from s4d.realworld.workers import WorkerSpec

    root, ep = tmp_path / "out", "panda"
    pipeline.ingest(panda_file.adapter, ep, root)
    meta = pipeline.load_meta(root, ep)
    runner = LocalRunner(tmp_path / "worker")
    D.sensor_depth(root, ep, runner)
    poses = {v.stream_id: make_T(np.eye(3), [j * 0.1, 0, 0]) for j, v in enumerate(meta.views())}

    def calibration(tmp):
        for stream, T in poses.items():
            rio.atomic_write_json(tmp / f"{rio.safe_name(stream)}.json", {"status": "accepted", "T_cam_base": T.tolist()})
        return {}

    rio.run_stage(rio.stage_dir(root, ep, "calibrate"), stage="calibrate", version="test", config={}, inputs={},
                  build=calibration)
    # The mock's depth-input echo tests this stage's contracts without a robot optimisation or upstream model.
    monkeypatch.setattr(pipeline, "RobotGeometry", lambda _: None)
    monkeypatch.setattr(D, "set_robot_motion", lambda m, stamps, geom: np.zeros(len(stamps)))
    calls = []
    worker = runner.worker

    def capture(spec, task, params, arrays, **kwargs):
        calls.append(arrays)
        return worker(spec, task, params, arrays, **kwargs)

    monkeypatch.setattr(runner, "worker", capture)
    backend = D.MapAnythingBackend(WorkerSpec("mock", sys.executable), runner, {}, allow_mock=True)
    rec = D.mapanything_depth(root, ep, backend)
    streams = [v.stream_id for v in meta.views()]
    assert rec["config"]["input_depth_stage"] == "sensor_depth"
    assert "include_right_eyes" not in rec["config"] and "stereo_rectify" not in rec["inputs"]
    assert rec["summary"]["views"] == streams
    for v, stream in enumerate(streams):
        np.testing.assert_array_equal(calls[0]["images"][:, v], pipeline.read_frames(root, ep, stream, list(range(4))))
        np.testing.assert_array_equal(calls[0]["K"][v], meta.stream(stream).intrinsics.K())
        np.testing.assert_array_equal(calls[0]["cam2world"][v], np.linalg.inv(poses[stream]))
        expected, valid, K = D.load_view_depth(root, ep, stream, np.arange(6), "sensor_depth")
        actual, actual_valid, actual_K = D.load_view_depth(root, ep, stream, np.arange(6))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_valid, valid)
        np.testing.assert_array_equal(actual_K, K)
        with h5py.File(D.depth_file(root, ep, "mapanything_depth", stream), "r") as f:
            assert f.attrs["grid"] == "view pinhole grid (as recorded)"
            np.testing.assert_array_equal(f["input_depth_valid"][:], valid)
            np.testing.assert_array_equal(f.attrs["T_cam_base"], poses[stream])
    assert D.mapanything_depth(root, ep, backend)["skipped"]
