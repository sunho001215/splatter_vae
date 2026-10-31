"""Large-scale DROID x PointWorld x DROID-3D data path on schema-faithful miniature fixtures (CPU only)."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

import droid_largescale_fixtures as F  # noqa: E402

from s4d.data.contract import PAIRS, collate, validate_batch  # noqa: E402
from s4d.data.droid.largescale import calibration as C  # noqa: E402
from s4d.data.droid.largescale import geometry as G  # noqa: E402
from s4d.data.droid.largescale import join  # noqa: E402
from s4d.data.droid.largescale.dataset import DroidLargeScaleDataset, worker_open_files  # noqa: E402
from s4d.data.droid.largescale.handles import ByteLRU, HandlePool  # noqa: E402
from s4d.data.droid.largescale.rlds_index import RldsStepReader, build_index, load_episode_rows  # noqa: E402
from s4d.data.droid.largescale.tfrecord import (  # noqa: E402
    TFRecordError,
    crc32c,
    encode_example,
    feature_values,
    iter_record_spans,
    walk_example,
    write_records,
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return F.build_all(tmp_path_factory.mktemp("droid_ls"))


@pytest.fixture(scope="module")
def env_pw(tmp_path_factory):
    return F.build_all(tmp_path_factory.mktemp("droid_ls_pw"), calibration="pointworld")


def seed_all(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_ds(env, **kw):
    return DroidLargeScaleDataset(env["manifest"], **F.dataset_kwargs(env, **kw))


# ------------------------------------------------------------------------------------------------ TFRecord / Example
def _example_class():
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    fd = descriptor_pb2.FileDescriptorProto(name="s4d_example_test.proto", package="s4dtest", syntax="proto3")
    for name, ftype in (("BytesList", 12), ("FloatList", 2), ("Int64List", 3)):
        m = fd.message_type.add(name=name)
        f = m.field.add(name="value", number=1, label=3, type=ftype)
        if ftype != 12:
            f.options.packed = True
    feat = fd.message_type.add(name="Feature")
    feat.oneof_decl.add(name="kind")
    for i, (n, t) in enumerate((("bytes_list", "BytesList"), ("float_list", "FloatList"), ("int64_list", "Int64List"))):
        feat.field.add(name=n, number=i + 1, label=1, type=11, type_name=f".s4dtest.{t}", oneof_index=0)
    feats = fd.message_type.add(name="Features")
    entry = feats.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, label=1, type=9)
    entry.field.add(name="value", number=2, label=1, type=11, type_name=".s4dtest.Feature")
    feats.field.add(name="feature", number=1, label=3, type=11, type_name=".s4dtest.Features.FeatureEntry")
    ex = fd.message_type.add(name="Example")
    ex.field.add(name="features", number=1, label=1, type=11, type_name=".s4dtest.Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.MessageFactory(pool).GetPrototype(pool.FindMessageTypeByName("s4dtest.Example"))


def test_example_encoding_matches_protobuf_and_walker_finds_spans():
    Example = _example_class()
    feats = {
        "episode_metadata/file_path": (
            "bytes",
            [b"gs://x/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023/trajectory.h5"],
        ),
        "steps/observation/cartesian_position": ("float", np.arange(12, dtype=np.float32) / 7),
        "steps/is_first": ("int64", [1, 0, -3]),
        "steps/observation/wrist_image_left": ("bytes", [b"\xff\xd8a", b"\xff\xd8bc"]),
    }
    payload = encode_example(feats)
    msg = Example.FromString(payload)
    fm = msg.features.feature
    assert list(fm["steps/observation/wrist_image_left"].bytes_list.value) == [b"\xff\xd8a", b"\xff\xd8bc"]
    np.testing.assert_array_equal(
        np.array(fm["steps/observation/cartesian_position"].float_list.value, np.float32),
        feats["steps/observation/cartesian_position"][1],
    )
    assert list(fm["steps/is_first"].int64_list.value) == [1, 0, -3]
    assert msg.SerializeToString(deterministic=True) == payload  # byte-identical to protobuf's own serializer
    spans = walk_example(payload)
    assert feature_values(payload, spans["steps/observation/wrist_image_left"]) == [b"\xff\xd8a", b"\xff\xd8bc"]
    np.testing.assert_array_equal(feature_values(payload, spans["steps/is_first"]), [1, 0, -3])
    np.testing.assert_allclose(
        feature_values(payload, spans["steps/observation/cartesian_position"]), np.arange(12) / 7, rtol=1e-6
    )


def test_tfrecord_framing_crc_and_corruption(tmp_path):
    assert crc32c(b"123456789") == 0xE3069283  # Castagnoli check value
    path = tmp_path / "x.tfrecord"
    spans = write_records(path, [b"abc", b"", b"hello world"])
    fd = os.open(path, os.O_RDONLY)
    try:
        assert list(iter_record_spans(fd, verify_payload=True)) == spans
    finally:
        os.close(fd)
    raw = bytearray(path.read_bytes())
    raw[spans[2][0]] ^= 1
    path.write_bytes(bytes(raw))
    fd = os.open(path, os.O_RDONLY)
    try:
        assert len(list(iter_record_spans(fd))) == 3  # headers intact
        with pytest.raises(TFRecordError, match="payload CRC"):
            list(iter_record_spans(fd, verify_payload=True))
    finally:
        os.close(fd)
    raw[4] ^= 1  # length bytes
    path.write_bytes(bytes(raw))
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(TFRecordError, match="length CRC"):
            list(iter_record_spans(fd))
    finally:
        os.close(fd)


def test_rlds_index_reads_exact_frames_and_state_in_place(env):
    gt = env["gt"]
    rows = load_episode_rows(env["tmp"] / "rlds_index")
    assert [(r["shard"], r["row"], r["ordinal"], r["num_steps"]) for r in rows] == [(0, 0, 0, 12), (1, 0, 1, 48)]
    reader = RldsStepReader(env["tmp"] / "rlds_index", env["tmp"] / "rlds", max_open_files=1)
    steps = [0, 6, 13, 47]
    out = reader.read(
        1,
        0,
        steps,
        cameras=("ext1", "ext2", "wrist"),
        numeric=("cartesian_position", "gripper_position", "joint_position", "action"),
    )
    for cam, role in enumerate(("ext1", "ext2", "wrist")):
        assert out[role].shape == (4, 180, 320, 3) and out[role].dtype == np.uint8
        for k, s in enumerate(steps):  # the frame colour encodes (raw step, camera): the right JPEG was read
            np.testing.assert_allclose(out[role][k].reshape(-1, 3).mean(0), F.frame_color(s, cam), atol=3)
    np.testing.assert_allclose(out["cartesian_position"], gt["cart"][steps], atol=1e-6)  # float_list = float32
    np.testing.assert_allclose(out["joint_position"], gt["q"][steps], atol=1e-6)
    reader.read(0, 0, [1], cameras=("wrist",))
    assert len(reader.files) == 1 and reader.files.opens == 2  # bounded: the other shard was closed
    with pytest.raises(IndexError):
        reader.read(0, 0, [12])
    reader.close()


def test_rlds_index_refuses_a_changed_shard(env, tmp_path):
    shutil.copytree(env["tmp"] / "rlds", tmp_path / "rlds")
    with open(tmp_path / "rlds" / "droid_101-train.tfrecord-00001-of-00002", "ab") as stream:
        stream.write(b"\0")
    reader = RldsStepReader(env["tmp"] / "rlds_index", tmp_path / "rlds")
    with pytest.raises(ValueError, match="changed size"):
        reader.read(1, 0, [0])


# ------------------------------------------------------------------------------------------------ conventions
def test_euler_quaternion_pose_and_inverse_follow_scipy():
    rng = np.random.default_rng(0)
    angles = rng.uniform(-3, 3, (20, 3))
    np.testing.assert_allclose(G.euler_xyz_to_matrix(angles), Rotation.from_euler("xyz", angles).as_matrix(), atol=1e-12)
    quat = Rotation.from_euler("xyz", angles).as_quat()  # x, y, z, w
    np.testing.assert_allclose(G.quat_xyzw_to_matrix(quat), Rotation.from_quat(quat).as_matrix(), atol=1e-12)
    T = G.pose6_to_matrix(np.concatenate([rng.normal(size=(20, 3)), angles], -1))
    np.testing.assert_allclose(G.invert(T) @ T, np.broadcast_to(np.eye(4), T.shape), atol=1e-12)


def test_franka_fk_ready_pose_matches_the_published_flange_position():
    T = C.franka_fk([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    np.testing.assert_allclose(T[:3, 3], [0.3069, 0.0, 0.5903], atol=2e-3)
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-6)  # flange z points down


def test_rotate180_camera_equals_rotating_the_image():
    rng = np.random.default_rng(1)
    K = np.array([[95.0, 0, 157.3], [0, 96.0, 91.2], [0, 0, 1]])
    c2w = F.lookat_c2w((0.3, 0.1, 0.6), (0.4, 0.0, 0.0))
    pts = np.stack([rng.uniform(0.2, 0.6, 50), rng.uniform(-0.2, 0.2, 50), np.zeros(50)], -1)
    uv, z = G.project(pts, K, G.invert(c2w))
    K2, c2w2 = G.rotate180_camera(K, c2w, 320, 180)
    uv2, z2 = G.project(pts, K2, G.invert(c2w2))
    np.testing.assert_allclose(uv2, np.stack([320 - uv[:, 0], 180 - uv[:, 1]], -1), atol=1e-9)
    np.testing.assert_allclose(z2, z, atol=1e-12)


def test_droid3d_flip_and_image_rotation_cancel_and_match_the_stored_wrist_depth(env):
    gt = env["gt"]
    w2c = C.droid3d_wrist_w2c(gt["q"], gt["dq"], gt["T_gw_static"], gt["dT_gw"], image_rotation_deg=180)
    direct = G.invert(C.franka_fk(gt["q"] + gt["dq"]) @ G.invert(gt["dT_gw"]) @ G.invert(gt["T_gw_static"]))
    np.testing.assert_allclose(w2c, direct, atol=1e-12)  # FLIP == Rz(180): no net rotation for stored images
    frustum = C.droid3d_wrist_w2c(gt["q"], gt["dq"], gt["T_gw_static"], gt["dT_gw"], image_rotation_deg=0)
    pts = np.stack([np.full(5, 0.4), np.linspace(-0.05, 0.05, 5), np.zeros(5)], -1)
    with h5py.File(env["tmp"] / "pw/depth_320x180" / f"{F.EPISODE}_depth.h5") as f:
        depth = f[f"{F.SERIALS['wrist']}+wrist"]["depth"][10] / 1000.0
    ok, _, res = __import__("s4d.data.droid.pointworld", fromlist=["depth_test"]).depth_test(
        pts, gt["K_wrist"], w2c[10], depth, continuous=True
    )
    assert ok.all() and np.abs(res).max() < 2e-3
    uv_img, _ = G.project(pts, gt["K_wrist"], w2c[10])
    uv, z = G.project(pts, gt["K_wrist"], frustum[10])
    # the released frustum is the same optical axis rotated by 180 degrees: pixels mirror through the centre
    np.testing.assert_allclose(uv, np.stack([320 - uv_img[:, 0], 180 - uv_img[:, 1]], -1), atol=1e-6)
    assert np.abs(uv - uv_img).max() > 5


def test_original_droid_hand_eye_from_metadata_reproduces_per_frame_wrist_poses(env):
    gt = env["gt"]
    T_base_cam = gt["c2w_wrist"]
    cart = gt["cart"]
    # DROID metadata wrist_cam_extrinsics = frame-0 composed pose as [x,y,z,euler xyz]
    meta = np.concatenate([T_base_cam[0][:3, 3], Rotation.from_matrix(T_base_cam[0][:3, :3]).as_euler("xyz")])
    T_ee_cam = C.droid_wrist_hand_eye(meta, cart[0])
    rebuilt = G.wrist_camera_c2w(cart, T_ee_cam)
    np.testing.assert_allclose(rebuilt[0], T_base_cam[0], atol=1e-9)
    # with DROID-3D joint offsets the camera rides on FK(q+dq), not on the controller pose: later frames differ
    assert np.abs(rebuilt[-1][:3, 3] - T_base_cam[-1][:3, 3]).max() < 0.01


def test_world_to_camera_direction_is_decided_by_depth(env):
    gt = env["gt"]
    pts = F.track_world_true("ext1", 0, 1, 40, 0)[0]
    depth = F.plane_depth(gt["K_ext"], gt["c2w_true"]["ext1"])
    res = G.w2c_direction_residual(pts, depth, gt["K_ext"], gt["w2c_true"]["ext1"])
    assert res["as_w2c"] < 5e-3 < 0.05 < res["as_c2w"]


def test_static_camera_rebase_is_exact_and_naive_matrix_swap_is_not(env):
    gt = env["gt"]
    X_true = F.track_world_true("ext1", 0, 3, 30, 0)
    X_pw = G.transform_points(G.invert(gt["w2c_pw"]["ext1"]) @ gt["w2c_true"]["ext1"], X_true)
    X_new = G.rebase_static_camera_points(X_pw, gt["w2c_pw"]["ext1"], gt["w2c_true"]["ext1"])
    np.testing.assert_allclose(X_new, X_true, atol=1e-12)
    np.testing.assert_allclose(
        G.transform_points(gt["w2c_true"]["ext1"], X_new), G.transform_points(gt["w2c_pw"]["ext1"], X_pw), atol=1e-12
    )  # same camera-frame points
    naive = G.transform_points(gt["w2c_true"]["ext1"], X_pw)  # swapped matrix, old world points
    assert np.abs(naive - G.transform_points(gt["w2c_true"]["ext1"], X_true)).max() > 0.005


# ------------------------------------------------------------------------------------------------ join
def test_identity_helpers():
    suffix = "AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023"
    for path in (F.SCENE, F.SCENE + "trajectory.h5", "/x/droid_raw/1.0.1/" + suffix + "/recordings/MP4"):
        assert join.scene_suffix(path) == suffix
    assert join.uuid_matches_suffix(F.EPISODE, suffix)
    assert not join.uuid_matches_suffix("AUTOLab+0d4edc83+2023-10-21-19h-07m-05s", suffix)
    with pytest.raises(ValueError):
        join.scene_suffix("success/2023-10-21/Sat_Oct_21_19:07:04_2023/trajectory.h5")  # no lab: not an identity


def test_manifest_records_identity_calibration_and_provenance(env):
    assert env["report"]["accepted"] == 1 and not env["report"]["rejected"]
    row = join.load_manifest(env["manifest"])[0]
    gt = env["gt"]
    assert row["rlds"]["shard"] == 1 and row["uuid_matches_folder"] and row["calibration"] == "droid3d"
    assert row["cameras"]["ext1"]["serial"] == F.SERIALS["ext1"] and row["cameras"]["wrist"]["serial"] == F.SERIALS["wrist"]
    np.testing.assert_allclose(row["cameras"]["ext2"]["w2c"], gt["w2c_true"]["ext2"], atol=1e-12)
    np.testing.assert_allclose(row["cameras"]["ext2"]["w2c_pointworld"], gt["w2c_pw"]["ext2"], atol=1e-6)
    np.testing.assert_allclose(row["cameras"]["ext1"]["K"], gt["K_ext"], atol=1e-5)
    np.testing.assert_allclose(row["cameras"]["wrist"]["K"], gt["K_wrist"], atol=1e-9)
    assert row["cameras"]["ext1"]["K_droid3d_minus_pw_max_px"] < 1e-4
    assert row["frames"]["valid"]["wrist"] == [True] * 16 + [False] * 8  # late wrist frames are not raw step 2i
    assert all(row["frames"]["valid"]["ext1"]) and all(row["frames"]["valid"]["ext2"])
    assert "rebased exactly" in row["cameras"]["ext1"]["tracks"] and row["provenance"]["droid3d"]["key"] == "episode_uuid"


def _rebuild(tmp, mutate, calibration="droid3d", **kw):
    gt = F.truth()
    shards = F.write_rlds_shards(tmp / "rlds", gt, decoy_duplicate=kw.pop("decoy_duplicate", False))
    build_index(shards, tmp / "rlds_index")
    F.write_pointworld(tmp / "pw", gt, **kw.pop("pw", {}))
    ext, intr, he = F.write_droid3d(tmp / "d3", gt, **kw.pop("d3", {}))
    mutate(tmp, ext, intr, he)
    reader = RldsStepReader(tmp / "rlds_index", tmp / "rlds")
    report = join.build_manifest(
        tmp / "pw",
        [F.EPISODE],
        load_episode_rows(tmp / "rlds_index"),
        tmp / "m.jsonl",
        calibration=calibration,
        droid3d_extrinsics=C.JsonlIndex(ext),
        droid3d_intrinsics=C.JsonlIndex(intr),
        hand_eye=C.load_hand_eye(he),
        rlds_reader=reader,
    )
    reader.close()
    return report


def _noop(*_):
    pass


def _set_line(path, fn):
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(fn(r) if r["episode_uuid"] == F.EPISODE else r) + "\n" for r in lines))


@pytest.mark.parametrize(
    "case, mutate, kw, reason",
    [
        (
            "droid3d uuid missing",
            lambda t, e, i, h: _set_line(e, lambda r: dict(r, episode_uuid=r["episode_uuid"] + "x")),
            {},
            "not in DROID-3D",
        ),
        ("duplicate RLDS suffix", _noop, {"decoy_duplicate": True}, "no unique RLDS episode"),
        ("wrong wrist serial", _noop, {"d3": {"wrist_serial": "13062452"}}, "wrist serial"),
        ("flow/depth wrist attr", _noop, {"pw": {"wrist_attr": "13062452"}}, "wrist serial"),
        (
            "droid3d scene path",
            lambda t, e, i, h: _set_line(e, lambda r: dict(r, scene_path=r["scene_path"].replace("19:07:04", "19:07:05"))),
            {},
            "scene_path suffix",
        ),
        ("camera json", lambda t, e, i, h: _patch_cameras(t), {}, "optimized_extrinsics"),
        ("robot state", lambda t, e, i, h: _patch_pose(t), {}, "frame mapping rejected"),
    ],
)
def test_join_rejects_mismatched_identities(tmp_path, case, mutate, kw, reason):
    report = _rebuild(tmp_path, mutate, **kw)
    assert report["accepted"] == 0, case
    assert reason in report["rejected"][F.EPISODE], report


def test_join_ignores_traj_id_and_accepts_case_variant_robot_keys(tmp_path):
    report = _rebuild(tmp_path, lambda t, e, i, h: _set_line(e, lambda r: dict(r, traj_id=0)))
    assert report["accepted"] == 1


def _patch_cameras(tmp):
    path = tmp / "pw/cameras" / f"{F.EPISODE}_cameras.json"
    data = json.loads(path.read_text())
    data[F.SERIALS["ext1"]]["optimized_extrinsics"][0][3] += 0.01
    path.write_text(json.dumps(data))


def _patch_pose(tmp):
    with h5py.File(tmp / "pw/flows-fs-optimized" / f"{F.EPISODE}_flows.h5", "r+") as f:
        f["5:16"]["gripper_pose"][3, 0] += 0.01  # one frame off by 1 cm: the canonical -> raw mapping is not proven


# ------------------------------------------------------------------------------------------------ dataset
def test_windows_stay_in_clips_and_use_validated_frames(env):
    ds = make_ds(env)
    assert [c for _, c, _ in ds.windows[:1]] == ["0:11"]
    clips = {c for _, c, _ in ds.windows}
    assert clips == set(F.CLIPS)
    for _, clip, i in ds.windows:
        s, e = map(int, clip.split(":"))
        assert s <= i and i + 6 < e
    assert len(ds) == sum(int(c.split(":")[1]) - int(c.split(":")[0]) - 6 for c in F.CLIPS)
    item = ds[ds.windows.index((0, "5:16", 7))]
    assert item["meta"]["t_indices"] == (7, 10, 13) and item["meta"]["raw_indices"] == (14, 20, 26)
    for t, step in enumerate((14, 20, 26)):
        for v in range(2):
            mean = item["images"][t, v].float().mean((1, 2)).numpy()
            np.testing.assert_allclose(mean, F.frame_color(step, v), atol=3)
        np.testing.assert_allclose(item["wrist_images"][t].float().mean((1, 2)).numpy(), F.frame_color(step, 2), atol=3)
    late = ds[ds.windows.index((0, "10:21", 12))]  # canonical 12, 15, 18: the wrist depth of 18 is not raw step 36
    assert late["wrist_valid"].tolist() == [True, True, False]
    assert float(late["wrist_depth"][2].abs().sum()) == 0


def test_stride_and_invalid_external_frames_shrink_the_window_list(tmp_path):
    env2 = F.build_all(tmp_path)
    full = make_ds(env2, stride=2)
    assert all(i + 4 < int(c.split(":")[1]) for _, c, i in full.windows)
    rows = [json.loads(x) for x in env2["manifest"].read_text().splitlines()]
    rows[0]["frames"]["valid"]["ext2"][7] = False
    env2["manifest"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    fewer = make_ds(env2, stride=2)
    assert all(7 not in (i, i + 2, i + 4) for _, _, i in fewer.windows) and len(fewer) < len(full)


def test_tracks_are_positions_rebased_to_droid3d_and_released_under_pointworld(env, env_pw):
    gt = env["gt"]
    for e, rebased in ((env, True), (env_pw, False)):
        ds = make_ds(e)
        item = ds[ds.windows.index((0, "5:16", 5))]
        rel = np.array([0, 3, 6])
        for v, role in enumerate(("ext1", "ext2")):
            X_true = F.track_world_true(role, 5, 16, (40, 52)[v], seed=100 * 1 + v)[rel]
            expected = X_true if rebased else G.transform_points(G.invert(gt["w2c_pw"][role]) @ gt["w2c_true"][role], X_true)
            n = int(item["track_count"][v])
            w = item["track_weight"][:, v, :n].numpy()
            for p, (a, b, _) in enumerate(PAIRS):
                ok = w[p] > 0
                np.testing.assert_allclose(item["track_src"][p, v, :n].numpy()[ok], expected[a][ok], atol=2e-3)
                np.testing.assert_allclose(
                    item["track_disp"][p, v, :n].numpy()[ok], (expected[b] - expected[a])[ok], atol=2e-3
                )
            assert w[:, -2:].sum() == 0 and w[:, :-2].min() == 1  # released visibility respected, depth test passes
            expected_w2c = gt["w2c_true"][role] if rebased else gt["w2c_pw"][role]
            np.testing.assert_allclose(item["w2c"][v].numpy(), expected_w2c, atol=1e-5)


def test_naively_swapped_calibration_fails_the_depth_test(env):
    """Keeping PointWorld world points under DROID-3D matrices would mis-supervise: the depth test rejects them."""
    from s4d.data.droid.pointworld import depth_test

    gt = env["gt"]
    X_true = F.track_world_true("ext1", 0, 1, 40, 0)[0]
    X_pw = G.transform_points(G.invert(gt["w2c_pw"]["ext1"]) @ gt["w2c_true"]["ext1"], X_true)
    depth = F.plane_depth(gt["K_ext"], gt["c2w_true"]["ext1"])
    good = depth_test(
        G.rebase_static_camera_points(X_pw, gt["w2c_pw"]["ext1"], gt["w2c_true"]["ext1"]),
        gt["K_ext"],
        gt["w2c_true"]["ext1"],
        depth,
        tolerance=0.005,
        continuous=True,
    )[0]
    bad = depth_test(X_pw, gt["K_ext"], gt["w2c_true"]["ext1"], depth, tolerance=0.005, continuous=True)[0]
    assert good.mean() > 0.8 and bad.sum() * 2 < good.sum()


def test_wrist_pose_depth_and_track_visibility_are_aligned_per_frame(env, env_pw):
    gt = env["gt"]
    ds = make_ds(env)
    item = ds[ds.windows.index((0, "0:11", 2))]
    raw = np.array(item["meta"]["raw_indices"])
    expected = C.droid3d_wrist_w2c(gt["q"][raw], gt["dq"], gt["T_gw_static"], gt["dT_gw"])
    np.testing.assert_allclose(item["wrist_w2c"].numpy(), expected, atol=1e-5)
    assert not torch.allclose(item["wrist_w2c"][0], item["wrist_w2c"][2])  # the wrist camera moves within a window
    torch.testing.assert_close(item["wrist_c2w"] @ item["wrist_w2c"], torch.eye(4).expand(3, 4, 4), atol=1e-5, rtol=0)
    assert item["wrist_track_weight"].sum() > 0
    # PointWorld recipe: mean_mat composes with the *controller* EE pose (cartesian_position), no joint offsets
    state = {"cartesian_position": gt["cart"][raw], "joint_position": gt["q"][raw]}
    pw_model = {"T_gw_static": gt["T_gw_static"].tolist(), "pose_model": "pointworld"}
    expected_pw = G.pose6_to_matrix(gt["cart"][raw]) @ G.invert(gt["T_gw_static"])
    np.testing.assert_allclose(G.invert(join.wrist_pose_w2c(pw_model, state)), expected_pw, atol=1e-9)
    # the fixture's hand-eye is flange-relative (DepthWorld) while cartesian_position includes a 10.34 cm tool offset:
    # used with the PointWorld recipe it misplaces the camera, and the per-episode wrist QA catches it from the data
    rows = join.load_manifest(env_pw["manifest"])
    assert rows[0]["cameras"]["wrist"] is None and "inconsistent" in rows[0]["wrist_qa"]["decision"]
    other = make_ds(env_pw)[0]
    assert not other["wrist_valid"].any()
    assert abs(rows[0]["frames"]["flange_to_cartesian_m"] - 0.1034) < 1e-4


def test_items_are_deterministic_and_survive_pickling(env):
    ds = make_ds(env, max_tracks=16)
    a = ds[3]
    b = pickle.loads(pickle.dumps(ds))[3]  # a DataLoader worker copy reopens its own files
    for key in a:
        if key != "meta":
            assert torch.equal(a[key], b[key]), key
    assert int(a["track_count"].max()) == 16  # deterministic subsample of 40/52 tracks


def test_bounded_open_files_and_caches(env):
    import resource

    ds = make_ds(env, max_open_files=1)
    for i in range(len(ds)):  # warm-up: every code path and allocator arena once
        ds[i]
    before, peak = worker_open_files(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(3):
        for i in range(len(ds)):
            ds[i]
    assert worker_open_files() <= before + 1
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak < 20 * 1024  # KiB: no growth with items read
    assert len(ds.pw.flow_files) <= 1 and len(ds.pw.depth_files) <= 1 and len(ds.rlds.files) <= 1
    assert ds.pw.flow_files.peak == ds.pw.depth_files.peak == ds.rlds.files.peak == 1
    pool = HandlePool(lambda k: k, lambda h: None, max_open=2)
    for k in "abcab":
        pool.get(k)
    assert len(pool) == 2 and pool.opens == 5  # a, b, c (evicts a), a (evicts b), b (evicts c)
    lru = ByteLRU(10)
    for k in range(5):
        lru.get(k, lambda k=k: (k, 4))
    assert lru.bytes <= 10 and lru.get(4, lambda: (None, 4)) == 4


def test_rank_sharding_and_resume_with_the_trainer_sampler(env):
    from s4d.train.loop import ResumableDistributedSampler

    ds = make_ds(env)
    shards = []
    for rank in range(2):
        s = ResumableDistributedSampler(ds, num_replicas=2, rank=rank, shuffle=True, seed=5, drop_last=True)
        s.set_epoch(3)
        shards.append(list(s))
    assert not set(shards[0]) & set(shards[1]) and len(shards[0]) == len(shards[1]) == len(ds) // 2
    again = ResumableDistributedSampler(ds, num_replicas=2, rank=1, shuffle=True, seed=5, drop_last=True)
    again.set_epoch(3)
    again.skip = 4
    assert list(again) == shards[1][4:]


def test_collate_contract_and_two_worker_loader(env):
    ds = make_ds(env)
    batch = collate([ds[0], ds[5]])
    sizes = validate_batch(batch)
    assert sizes == {"B": 2, "T": 3, "V": 2, "H": 36, "W": 64}
    assert batch["wrist_w2c"].shape == (2, 3, 4, 4) and batch["meta"]["episode"] == [F.EPISODE] * 2
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, num_workers=2, collate_fn=collate, multiprocessing_context="fork", sampler=list(range(6))
    )
    batches = list(loader)
    assert len(batches) == 3 and torch.equal(batches[0]["images"][0], ds[0]["images"])
    assert torch.equal(batches[2]["track_src"][1], ds[5]["track_src"])


def test_trainer_forward_and_backward_on_a_fixture_batch_ignore_privileged_extras(env, monkeypatch):
    from s4d.config import load_config
    from s4d.losses import rgb
    from s4d.model import render
    from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
    from s4d.model.encoder import Encoder, EncoderConfig
    from s4d.train.loop import Model, forward_losses

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_review_options import _fake_rasterizer

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    ds = make_ds(env, image_height=16, image_width=32, max_tracks=32)
    batch = collate([ds[0], ds[7]])
    cfg = load_config([Path(__file__).resolve().parents[1] / "configs/droid/pretrain.yaml"], ["model.anchor_stats=null"])
    cfg["train"]["bf16"] = False
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    torch.manual_seed(0)
    enc = Encoder(
        EncoderConfig(
            image_height=16, image_width=32, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8, drop_path=0
        )
    )
    model = Model(
        enc, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group, dynamic=group, num_slots=2))
    ).train()
    seed_all(1)
    out = forward_losses(model, batch, cfg, 10, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
    stripped = {k: v for k, v in batch.items() if not k.startswith(("wrist_", "robot_"))}
    model.zero_grad()
    seed_all(1)
    again = forward_losses(model, stripped, cfg, 10, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert torch.equal(again["total"].detach(), out["total"].detach())  # privileged extras never reach the model


# ------------------------------------------------------------------------------------------------ trainer wiring / CLIs
def test_trainer_loader_resume_replays_the_same_batches(env):
    from s4d.train.ddp import DistContext
    from s4d.train.loop import infinite, make_train_loader

    ds = make_ds(env, max_tracks=16)
    cfg = {"train": {"num_workers": 0, "batch_size": 2}}
    loader = make_train_loader(ds, cfg, DistContext(0, 0, 1, torch.device("cpu")), seed=3)
    stream = infinite(loader, start_step=0)
    first = [next(stream)["meta"]["t_indices"] for _ in range(len(loader) + 2)]  # crosses an epoch boundary
    resumed = infinite(make_train_loader(ds, cfg, DistContext(0, 0, 1, torch.device("cpu")), seed=3), start_step=5)
    assert [next(resumed)["meta"]["t_indices"] for _ in range(len(first) - 5)] == first[5:]


def test_config_regime_builds_episode_disjoint_splits(env, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import build_data

    from s4d.config import load_config

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(
        [repo / "configs/droid/pretrain.yaml", repo / "configs/droid/largescale.yaml"],
        [
            f"data.manifest={env['manifest']}",
            f"data.pointworld_root={env['tmp'] / 'pw'}",
            f"data.rlds_index_dir={env['tmp'] / 'rlds_index'}",
            f"data.rlds_data_root={env['tmp'] / 'rlds'}",
            "data.validation_percent=0",
            "train.num_workers=0",
            "train.batch_size=2",
            "model.encoder.image_height=16",
            "model.encoder.image_width=32",
        ],
    )
    with pytest.raises(ValueError, match="no valid windows in split 'validation'"):  # the one episode is train
        build_data(cfg)
    from s4d.data.droid.largescale import dataset as D

    assert D.episode_split(F.EPISODE, 0) == "train" and D.episode_split(F.EPISODE, 100) == "validation"
    calls = iter(["train", "validation"])  # one fixture episode: let each split take it once to test the wiring
    monkeypatch.setattr(D, "episode_split", lambda episode, percent: next(calls))
    train_ds, val_loaders, probe_loaders, n_cams = build_data(cfg)
    assert isinstance(train_ds, DroidLargeScaleDataset) and n_cams == 2 and probe_loaders == {}
    assert train_ds[0]["images"].shape == (3, 2, 3, 16, 32)
    batch = next(iter(val_loaders[3]))
    assert validate_batch(batch)["B"] == 2 and batch["wrist_images"].shape == (2, 3, 3, 16, 32)
    bad = load_config([repo / "configs/droid/pretrain.yaml", repo / "configs/droid/largescale.yaml"])
    with pytest.raises(ValueError, match="data.manifest"):
        build_data(bad)


def test_index_and_manifest_clis_are_resumable_and_refuse_writing_into_the_release(env, tmp_path):
    import importlib.util

    repo = Path(__file__).resolve().parents[1]

    def script(name):
        spec = importlib.util.spec_from_file_location(f"cli_{name}", repo / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    index = script("droid_ls_build_rlds_index")
    with pytest.raises(SystemExit, match="inside the read-only"):
        index.main(["--rlds-root", str(env["tmp"] / "rlds"), "--out", str(env["tmp"] / "rlds" / "idx")])
    index.main(["--rlds-root", str(env["tmp"] / "rlds"), "--out", str(tmp_path / "idx"), "--range", "1:2", "--no-finalize"])
    assert not (tmp_path / "idx" / "index.json").exists()
    with pytest.raises(FileNotFoundError):  # shard 0 missing: cannot finalize a partial index
        index.main(["--rlds-root", str(env["tmp"] / "rlds"), "--out", str(tmp_path / "idx"), "--range", "1:2"])
    stamp = (tmp_path / "idx" / "droid_101-train.tfrecord-00001-of-00002.npz").stat().st_mtime_ns
    index.main(["--rlds-root", str(env["tmp"] / "rlds"), "--out", str(tmp_path / "idx")])
    assert (tmp_path / "idx" / "droid_101-train.tfrecord-00001-of-00002.npz").stat().st_mtime_ns == stamp  # skipped
    assert load_episode_rows(tmp_path / "idx") == load_episode_rows(env["tmp"] / "rlds_index")
    manifest = script("droid_ls_build_manifest")
    d3 = env["tmp"] / "d3"
    manifest.main(
        [
            "--pointworld-root",
            str(env["tmp"] / "pw"),
            "--rlds-index",
            str(tmp_path / "idx"),
            "--rlds-root",
            str(env["tmp"] / "rlds"),
            "--droid3d-extrinsics",
            str(d3 / "extrinsics.jsonl"),
            "--droid3d-intrinsics",
            str(d3 / "camera_intrinsics.jsonl"),
            "--hand-eye",
            str(d3 / "gripper2wrist_transforms.json"),
            "--out",
            str(tmp_path / "m.jsonl"),
        ]
    )
    assert (tmp_path / "m.jsonl").read_text() == env["manifest"].read_text()


# ------------------------------------------------------------------------------------------------ wrist QA
def test_wrist_qa_accepts_the_true_model_and_records_orientation(env):
    qa = join.load_manifest(env["manifest"])[0]["wrist_qa"]
    assert qa["decision"] == "ok" and qa["depth_rotation_deg"] == 0 and qa["count"] >= 50
    assert qa["median_residual_m"]["as_stored"] < 3e-3 < qa["median_residual_m"]["rotated180"]


def test_wrist_depth_stored_rotated_is_detected_and_undone(env, tmp_path):
    rotated = F.build_all(tmp_path, wrist_depth_rotated=True)
    qa = join.load_manifest(rotated["manifest"])[0]["wrist_qa"]
    assert qa["decision"] == "ok" and qa["depth_rotation_deg"] == 180
    a, b = make_ds(env), make_ds(rotated)
    i = a.windows.index((0, "0:11", 1))
    torch.testing.assert_close(b[i]["wrist_depth"], a[i]["wrist_depth"], atol=0, rtol=0)


def test_wrong_hand_eye_disables_the_wrist_but_keeps_external_windows(tmp_path):
    bad = F.build_all(tmp_path, hand_eye_error_m=0.08)
    row = join.load_manifest(bad["manifest"])[0]
    assert row["cameras"]["wrist"] is None and any("wrist QA" in r for r in row["wrist_unavailable"])
    ds = make_ds(bad)
    item = ds[0]
    assert not item["wrist_valid"].any() and item["wrist_depth"].abs().sum() == 0 and item["wrist_track_weight"].sum() == 0
    assert item["track_weight"].sum() > 0
