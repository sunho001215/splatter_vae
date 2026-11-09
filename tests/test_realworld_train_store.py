"""Compact real-world training store (``s4d.realworld.train_store``) and its loaders: geometry under the square
crop/resize, depth/track/time alignment, the batch contract against the former window export on the same synthetic
episode, online SNeRL DINO targets against the cached-teacher path, our model's forward/losses on CPU, and the
per-camera frame interface of the baselines/FQL (synthetic episodes with flagged MOCK model outputs; CPU only)."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from s4d.data.contract import collate, validate_batch
from s4d.data.droid.largescale import crop as CR
from s4d.data.droid.pointworld import PAIR_TIMES, project
from s4d.realworld import io as rio
from s4d.realworld import train_store as TS
from s4d.realworld.train_windows import RealworldWindowDataset, inside_image

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ pure geometry
def test_square_crop_affine_maps_projections_and_pixels_exactly():
    rng = np.random.default_rng(0)
    box = TS.square_box(480, 640)
    assert (box.x0, box.y0, box.side) == (80, 0, 480)
    aff = TS.box_affine(box, 128)
    K = np.array([[609.6, 0, 324.6], [0, 608.4, 246.8], [0, 0, 1.0]])
    np.testing.assert_allclose(aff.apply_K(K), CR.crop_intrinsics(K, box, 128), rtol=0, atol=1e-12)
    w2c = np.eye(4)
    w2c[:3, 3] = [0.1, -0.2, 0.3]
    pts = rng.uniform([-0.5, -0.4, 0.5], [0.5, 0.4, 2.0], (500, 3))
    uv_nat, z = project(pts, K, w2c)
    uv_128, z128 = project(pts, aff.apply_K(K).astype(np.float32), w2c)
    np.testing.assert_allclose(aff.apply_uv(uv_nat), uv_128, atol=2e-3)  # float32 projection
    np.testing.assert_array_equal(z, z128)
    # nearest resampling == the DROID crop_depth rule, dtype kept; every output pixel samples the native pixel that
    # contains its centre mapped back through the affine
    img = rng.integers(0, 255, (480, 640), np.uint8)
    out = TS.nearest_resample(img, box, 128)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, CR.crop_depth(img.astype(np.float32), box, 128).astype(np.uint8))
    centres = np.stack(np.meshgrid(np.arange(128) + 0.5, np.arange(128) + 0.5, indexing="xy"), -1)
    src = np.floor(aff.inverse().apply_uv(centres)).astype(int)
    np.testing.assert_array_equal(out, img[src[..., 1], src[..., 0]])
    # crop-dependent visibility: points projecting into the side strips cut by the crop are outside
    side = np.array([[(40 - 324.6) / 609.6, 0.0, 1.0], [(600 - 324.6) / 609.6, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert inside_image(side, aff.apply_K(K), np.eye(4), 128).tolist() == [False, False, True]
    assert not inside_image(np.array([[0.0, 0.0, -1.0]]), aff.apply_K(K), np.eye(4), 128)[0]  # behind the camera


def test_training_depth_is_lossless_and_display_depth_bounded():
    from s4d.realworld.vis_store import DEPTH_UNIT_M, quantize_depth

    d = np.array([0.0, 0.1, 0.10004, 1.23456789, 12.5, np.nan], np.float32)
    valid = np.array([False, True, True, True, True, False])
    out = TS.training_depth(d, valid)
    assert out.dtype == np.float32 and out[0] == 0 and out[-1] == 0
    np.testing.assert_array_equal(out[1:5], d[1:5])  # bitwise: no quantization, no range limit
    with pytest.raises(ValueError, match="finite"):
        TS.training_depth(d, np.ones(6, bool))
    q = quantize_depth(d[:4])  # viewer store only: 0.1 mm steps
    assert q.dtype == np.uint16 and q[0] == 0 and (q[1:] > 0).all()
    assert np.abs(q * DEPTH_UNIT_M - d[:4]).max() <= DEPTH_UNIT_M / 2 + 1e-7
    with pytest.raises(ValueError, match="uint16"):
        quantize_depth(np.array([7.0]))


def test_window_enumeration_rules():
    """Reference-timeline windows: robot state, sampled-gap tolerance, recording segments, per-camera eligibility
    (synchronised, one segment, one query window: the latest) and the minimum number of eligible views."""
    from s4d.realworld.train_windows import enumerate_windows

    n = 12
    stamps = np.arange(n, dtype=np.int64) * 100
    stamps[9:] += 1000  # recording pause before reference frame 9
    ref_seg = np.r_[np.zeros(9, int), np.ones(3, int)]
    robot_ok = np.ones(n, bool)
    robot_ok[0] = False
    frames = np.stack([np.arange(n), np.arange(n)], 1).astype(np.int32)
    frames[5, 1] = -1  # camera 1 has no synchronised frame at reference frame 5
    matched = frames >= 0
    segs = [ref_seg, ref_seg]
    cf = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]])  # overlapping query windows 0 and 4
    tables = [(cf, np.array([0, 4])), (cf, np.array([0, 4]))]
    rows, counts = enumerate_windows(stamps, ref_seg, robot_ok, frames, matched, segs, tables, strides=(1,),
                                     max_gap_factor=1.5, min_views=2)
    got = {r[0]: r[2:] for r in rows}
    # t0=0: robot state invalid; 3, 4, 5: camera 1 lacks frame 5; 7, 8: cross the pause
    assert got == {1: [0, 0], 2: [0, 0], 6: [1, 1], 9: [1, 1]}
    assert counts == {"windows_dropped_for_time_gaps": 2, "dropped_recording_segment": 0, "dropped_too_few_views": 3}
    rows1, _ = enumerate_windows(stamps, ref_seg, robot_ok, frames, matched, segs, tables, strides=(1,),
                                 max_gap_factor=1.5, min_views=1)
    single = {r[0]: r[2:] for r in rows1}
    # camera 1 not eligible, camera 0 still usable; frames 4-7 lie in both query windows -> the latest (query 4)
    assert single[3] == [0, -1] and single[4] == [1, -1] and single[5] == [1, -1]
    # stride 2 over a dropped reference frame (one 2-period step): the former export's sampled check (3 <= 1.5 * 2
    # periods) kept it; every step inside the span must be regular
    drop = np.delete(np.arange(n + 1, dtype=np.int64) * 100, 3)  # reference frame 3 dropped
    rows3, counts3 = enumerate_windows(drop, np.zeros(n, int), np.ones(n, bool), frames, np.ones_like(matched),
                                       [np.zeros(n, int)] * 2, tables, strides=(2,), max_gap_factor=1.5, min_views=1)
    assert {r[0] for r in rows3}.isdisjoint({0, 1, 2}) and {3, 4} <= {r[0] for r in rows3}
    assert counts3["windows_dropped_for_time_gaps"] == 3
    # a pause shorter than the gap tolerance is still a recording-segment break
    stamps2 = np.arange(n, dtype=np.int64) * 100
    rows2, counts2 = enumerate_windows(stamps2, ref_seg, np.ones(n, bool), frames, matched, segs, tables, strides=(1,),
                                       max_gap_factor=1.5, min_views=1)
    assert {r[0] for r in rows2}.isdisjoint({7, 8}) and counts2["dropped_recording_segment"] == 2


def test_camera_targets_crop_membership_per_pair():
    """A track or gripper point inside the stored image at times 0 and 2 but outside at time 1 is supervised only for
    the pair (0, 2); motion-segment breaks are never supervised."""
    from s4d.realworld.train_windows import camera_targets

    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]], np.float32)
    w2c = np.eye(4, dtype=np.float32)
    inside, outside = [0.0, 0.0, 1.0], [2.0, 0.0, 1.0]  # u = 264 > 128
    pts = np.array([[inside, inside], [outside, inside], [inside, inside]], np.float32)  # (3,J=2,3)
    pts[:, 1, 0] += np.array([0.0, 0.01, 0.02], np.float32)  # track 1 moves inside the image
    valid = np.ones((3, 2), bool)
    seg = np.array([[0, 0], [0, 0], [0, 1]])  # track 1 changes motion segment at time 2
    grip = pts[:, :1].copy()
    gvis = np.ones((3, 1), bool)
    depth = np.ones((3, 128, 128), np.float32)
    tg = camera_targets(pts, valid, seg, grip, gvis, K, w2c, depth, 128)
    assert tg["track_weight"][:, 0].tolist() == [0, 0, 1]  # pairs (0,1), (1,2), (0,2)
    assert tg["track_weight"][:, 1].tolist() == [1, 0, 0]  # segment break: (1,2) and (0,2) unsupervised
    assert tg["gripper_weight"][:, 0].tolist() == [0, 0, 1]
    assert tg["motion_score"].shape == (3, 1, 128, 128)


def test_mutual_sync_handles_offsets_drops_and_refuses_ambiguous_tolerance():
    period = 66_700_000
    ref = np.arange(10, dtype=np.int64) * period
    late = ref + 22_900_000  # constant offset
    dropped = np.delete(ref + 5_000_000, 4)  # one frame dropped
    frames, err = TS.mutual_sync([ref, late, dropped], 30_000_000)
    assert frames[:, 0].tolist() == list(range(10)) and frames[:, 1].tolist() == list(range(10))
    assert frames[4, 2] == -1 and frames[5, 2] == 4 and frames[3, 2] == 3
    assert err[0, 1] == 22_900_000
    frames, _ = TS.mutual_sync([ref, late], 20_000_000)
    assert (frames[:, 1] == -1).all()
    with pytest.raises(ValueError, match="half the frame period"):
        TS.mutual_sync([ref, late], period // 2)


# ------------------------------------------------------------------------------------------------ synthetic store
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from s4d.realworld.export import export
    from s4d.realworld.stages.motion import MotionFilterConfig, track_motion
    from s4d.realworld.stages.semantics import sam3_masks
    from s4d.realworld.workers import WorkerSpec
    from tests.test_realworld_pipeline import synthetic_rgbd_episode

    tmp = tmp_path_factory.mktemp("store")
    run = synthetic_rgbd_episode(tmp)
    root, ep = run["root"], run["ep"]
    track_motion(root, ep, cfg=MotionFilterConfig(min_bin_samples=10, min_segment_samples=4,
                                                  reference_robot_distance_m=0.05))
    sam3_masks(root, ep, WorkerSpec("mock", sys.executable, timeout_s=120), cameras=["rig0/left", "rig1/left"],
               prompts={"robot arm": "robot", "red mug": "object"}, allow_mock=True)
    old = export(root, ep, tmp / "old", resolutions={"square": (128, 128)}, depth_stage="sensor_depth",
                 allow_mock=True, aspect="center_crop", track_source="track_motion", max_gap_factor=1.5)
    with pytest.raises(RuntimeError, match="MOCK"):
        TS.write_episode(root, ep, tmp / "refused", task="synthetic", depth_stage="sensor_depth")
    rec = TS.write_episode(root, ep, tmp / "store", task="synthetic", depth_stage="sensor_depth", allow_mock=True)
    return {"root": root, "ep": ep, "tmp": tmp, "rec": rec, "old": old, "adapter": run["adapter"]}


def _dataset(built, **kw):
    with pytest.warns(UserWarning, match="MOCK"):
        return RealworldWindowDataset([built["tmp"] / "store"], split="all", **kw)


@pytest.fixture
def pipeline_named_store(built, tmp_path):
    """The REAL writer, with dispatcher episode ids and recorded raw filenames (no HDF5 attribute patching)."""
    import hashlib
    import shutil

    from s4d.rwlearn import gate
    from tests.test_rwlearn_gate import entry

    root, paths, episodes = tmp_path / "processed", [], {}
    for i in range(2):
        eid, filename = f"drawer-open-{i:06d}", f"episode_{i:06d}.h5"
        shutil.copytree(rio.episode_dir(built["root"], built["ep"]), rio.episode_dir(root, eid))
        meta_path = rio.episode_dir(root, eid) / "normalized" / "episode.json"
        meta = json.loads(meta_path.read_text())
        sha = hashlib.sha256(f"synthetic raw {i}".encode()).hexdigest()
        meta["episode_id"] = eid
        meta["provenance"]["source_files"] = [{"path": filename, "sha256": sha, "bytes": 15}]
        meta_path.write_text(json.dumps(meta))
        out = tmp_path / "compact" / eid
        TS.write_episode(root, eid, out, task="drawer-open", depth_stage="sensor_depth", allow_mock=True)
        paths.append(out / TS.FILE)
        rec = entry("drawer-open", i, sha=sha, validated=True)
        rec["filename"] = filename
        stages = rec["preprocessing"]["stages"]
        build = rio.build_id(rio.read_record(rio.stage_dir(root, eid, "calibrate")))
        stages["calibrate"]["metrics"]["build_id"] = build
        stages[gate.CALIBRATION_STAGE]["metrics"].update(calibrate_build_id=build, cameras=["rig0", "rig1"])
        episodes[f"drawer-open/{filename}"] = rec
    manifest = {"schema": gate.SCHEMA, "schema_version": gate.SCHEMA_VERSION, "episodes": episodes,
                "counts": {"drawer-open": {"target": 2, "complete": True}}}
    with pytest.warns(UserWarning, match="MOCK"):
        store = TS.CompactStore(paths)
    yield store, gate.evaluate(manifest, ["drawer-open"], target=2), paths
    store.close()


def test_pipeline_names_pass_inputs_outcomes_split_and_both_gates(pipeline_named_store, tmp_path):
    import subprocess

    import yaml

    from s4d.rwlearn import gate, outcomes
    from s4d.rwlearn.baseline_data import check_gate
    from s4d.rwlearn.identity import store_identities
    from s4d.rwlearn.store import split_episodes

    store, report, paths = pipeline_named_store
    eps = store.episodes()
    assert eps == [f"drawer-open/drawer-open-{i:06d}" for i in range(2)]
    assert [store.raw_recording(ep) for ep in eps] == [f"episode_{i:06d}.h5" for i in range(2)]
    split = split_episodes(store)
    assert set(split["train"]) | set(split["validation"]) == set(eps)
    assert len(split["validation"]) == 1 and not set(split["train"]) & set(split["validation"])
    assert not check_gate(store, report, steps=None, split=split, target="drawer-open", consumer="snerl")["smoke"]
    bound = gate.bind_store_episodes(report, {"drawer-open": eps}, store_identities(store))
    assert bound["tasks"]["drawer-open"]["store_episodes"] == eps
    assert set(bound["calibration_hold"]["drawer-open"]["episodes"]) == set(eps)
    ending = {"schema": outcomes.REPORT_SCHEMA, "episodes": {
        f"drawer-open/episode_{i:06d}": {"path": f"/relocated/drawer-open/episode_{i:06d}.h5",
            "decision": "needs_image_review", "flags": [{"code": f"review_{i}", "severity": "review"}]}
        for i in range(2)}}
    labels = outcomes.outcome_labels(eps, ending_report=ending, store=store)
    assert outcomes.load_outcomes(labels) == dict.fromkeys(eps, True)
    assert [labels["episodes"][ep]["ending_flags"] for ep in eps] == [["review_0"], ["review_1"]]
    locator, gate_path, ending_path = (tmp_path / name for name in ("store.yaml", "gate.json", "ending.json"))
    locator.write_text(yaml.safe_dump({"kind": "compact", "paths": list(map(str, paths))}))
    gate_path.write_text(json.dumps(report))
    ending_path.write_text(json.dumps(ending))
    out = tmp_path / "inputs"
    result = subprocess.run([sys.executable, "-I", str(REPO / "scripts/realworld_learning.py"), "inputs",
                             "--store", str(locator), "--gate-report", str(gate_path), "--task", "drawer-open",
                             "--out-dir", str(out), "--ending-report", str(ending_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((out / "outcomes.json").read_text()) == labels
    assert json.loads((out / "split_seed0.json").read_text()) == split
    saved = json.loads((out / "gate_report.json").read_text())
    assert saved == bound
    assert not check_gate(store, saved, steps=None, consumer="snerl")["smoke"]
    # FQL's trainer gate uses the same provenance and store keys, including geometric calibration evidence.
    dataset = tmp_path / "gated_dataset.h5"
    with h5py.File(dataset, "w") as f:
        f.attrs.update(episodes=json.dumps(eps), condition="snerl", raw_identities=json.dumps(store_identities(store)),
                       calibration=json.dumps({ep: {"calibrate": store.upstream(ep)["calibrate"],
                                                     "cameras": store.cameras(ep)} for ep in eps}))
        for ep in eps:
            f.create_group(f"episodes/{ep}/sync").attrs["task"] = store.task(ep)
    for evidence in (report, saved):
        assert not gate.check_dataset_gate(dataset, evidence, steps=None, split=split)["smoke"]


@pytest.mark.parametrize("attribute,value", [("raw_recording", "episode_999999.h5"),
                                             ("raw_recording", ""), ("raw_sha256", "f" * 64)])
def test_pipeline_names_cannot_hide_wrong_raw_identity(pipeline_named_store, attribute, value):
    from s4d.rwlearn.baseline_data import check_gate

    store, report, paths = pipeline_named_store
    store.close()
    with h5py.File(paths[0], "r+") as f:
        f.attrs[attribute] = value
    with pytest.warns(UserWarning, match="MOCK"):
        changed = TS.CompactStore(paths)
    try:
        with pytest.raises(RuntimeError, match="raw identity"):
            check_gate(changed, report, steps=None, consumer="e2e_cnn")
    finally:
        changed.close()


@pytest.mark.parametrize("problem", ["missing", "mismatch", "duplicate", "sha256", "missing_path"])
def test_pipeline_outcome_flags_require_unambiguous_raw_identity(pipeline_named_store, problem):
    from s4d.rwlearn import outcomes

    store, _, _ = pipeline_named_store
    records = {f"drawer-open/raw_{i}": {"path": f"episode_{i:06d}.h5", "decision": "needs_image_review",
                                       "flags": []} for i in range(2)}
    if problem == "missing":
        records.pop("drawer-open/raw_0")
    elif problem == "mismatch":
        records["drawer-open/raw_0"]["path"] = "episode_999999.h5"
    elif problem == "missing_path":
        records["drawer-open/raw_0"].pop("path")
    elif problem == "duplicate":
        records["drawer-open/alias"] = dict(records["drawer-open/raw_0"])
    else:
        records["drawer-open/raw_0"]["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="raw identity"):
        outcomes.outcome_labels(store.episodes(), ending_report={"episodes": records}, store=store)


def test_store_layout_versions_and_idempotence(built):
    rec = built["rec"]
    s = rec["summary"]
    assert rec["stage"] == TS.STAGE and s["format"] == TS.FORMAT and s["schema_version"] == TS.SCHEMA_VERSION
    assert s["cameras"] == ["rig0", "rig1"] and s["excluded_cameras"] == {"rig2": "no metric depth"}
    assert s["semantic"] and s["mock"] and s["file_sha256"] == rio.sha256_file(built["tmp"] / "store" / TS.FILE)
    fields = TS.manifest_fields(rec)
    assert fields["schema_version"] == TS.SCHEMA_VERSION and fields["code_commit"] and fields["build_id"]
    assert set(rec["inputs"]) == {"ingest", "sensor_depth", "calibrate", "track2d", "lift3d", "track_motion",
                                  "sam3_masks"}
    assert TS.write_episode(built["root"], built["ep"], built["tmp"] / "store", task="synthetic",
                            depth_stage="sensor_depth", allow_mock=True)["skipped"]
    with h5py.File(built["tmp"] / "store" / TS.FILE, "r") as f:
        g = f["cameras/rig0"]
        n = g["timestamps_ns"].shape[0]
        assert g["rgb"].shape == (n, 128, 128, 3) and g["rgb"].chunks == (1, 128, 128, 3)
        assert g["depth"].dtype == np.float32 and g["depth"].compression == "gzip" and g["depth"].shuffle
        assert g["semantic"].shape == (n, 128, 128) and g["class_present"].shape == (n, 4)
        assert "xyz" not in g and not any("window" in k for k in f)  # no dense XYZ, no window copies
        assert f.attrs["MOCK_MODEL_OUTPUTS"]
    assert set(s["bytes"]["stored"]) == {"rgb", "depth", "semantic", "tracks", "robot", "sync", "camera_meta"}


def test_accepted_episode_is_byte_identical_to_the_pre_change_writer(built, tmp_path, monkeypatch):
    """Run the frozen release writer and this writer on the SAME synthetic inputs, with volatile provenance pinned."""
    monkeypatch.setattr(TS, "_code_commit", lambda: "2a35cfb" + "0" * 33)
    monkeypatch.setattr(TS.time, "time", lambda: 1793923200.0)
    namespace = dict(vars(TS))
    oracle = REPO / "tests/fixtures/realworld/train_store_v2_writer.py.txt"
    exec(compile(oracle.read_text(), str(oracle), "exec"), namespace)
    kw = dict(task="synthetic", depth_stage="sensor_depth", allow_mock=True)
    old = namespace["write_episode"](built["root"], built["ep"], tmp_path / "old", **kw)
    new = TS.write_episode(built["root"], built["ep"], tmp_path / "new", **kw)
    assert (tmp_path / "old" / TS.FILE).read_bytes() == (tmp_path / "new" / TS.FILE).read_bytes()
    assert old["fingerprint"] == new["fingerprint"] and old["config"] == new["config"]
    assert "layout" not in new["summary"]
    # An already exported accepted episode is reused in place, without any re-export.
    before = (tmp_path / "old" / TS.FILE).read_bytes()
    assert TS.write_episode(built["root"], built["ep"], tmp_path / "old", **kw)["skipped"]
    assert (tmp_path / "old" / TS.FILE).read_bytes() == before


@pytest.mark.parametrize("status,all_cameras", [("rejected", True), ("held", True), ("missing", True),
                                               ("rejected", False)])
def test_unaccepted_calibration_keeps_every_rgb_robot_timeline(built, tmp_path, status, all_cameras):
    import shutil

    from s4d.realworld.pipeline import load_meta, read_frames
    from s4d.rwlearn.store import sync_index
    from s4d.rwlearn.transitions import timelines

    root, ep = tmp_path / "root", built["ep"]
    shutil.copytree(built["root"], root)
    meta = load_meta(root, ep)
    cal_dir = rio.stage_dir(root, ep, "calibrate")
    for rig in ("rig0", "rig1", "rig2") if all_cameras else ("rig0",):
        path = cal_dir / f"{rig}__left.json"
        if status == "missing":
            path.unlink()
        else:
            cal = json.loads(path.read_text())
            cal.update(status=status, reasons=["motion coverage below 0.05 m"])
            path.write_text(json.dumps(cal))
    if all_cameras:
        # Calibration rejection may leave no lifted/fused products or complete geometry-stage records.
        for stage in ("sensor_depth", "track2d", "lift3d", "track_motion"):
            shutil.rmtree(rio.stage_dir(root, ep, stage))
    out = tmp_path / "store"
    # Even if the upstream record ids stayed the same, the new layout marker invalidates a store that omitted views.
    shutil.copytree(built["tmp"] / "store", out)
    rec = TS.write_episode(root, ep, out, task="synthetic", depth_stage="sensor_depth", allow_mock=True)
    assert not rec["skipped"] and rec["fingerprint"] != built["rec"]["fingerprint"]
    assert rec["summary"]["layout"] == TS.CALIBRATION_OPTIONAL_LAYOUT
    assert TS.manifest_fields(rec)["calibration_status"]["rig0"] == status
    with pytest.warns(UserWarning, match="MOCK"):
        store = TS.CompactStore([out])
    key = store.episodes()[0]
    ef = store.episode(key)
    assert ef.cameras == ["rig0", "rig1", "rig2"]
    for cam in ef.cameras:
        stream = f"{cam}/left"
        stamps = meta.stream_timestamps_ns[stream]
        np.testing.assert_array_equal(ef.timestamps(cam), stamps)
        expected = read_frames(root, ep, stream, list(range(len(stamps))))
        box = TS.square_box(*expected.shape[1:3])
        np.testing.assert_array_equal(ef.rgb(cam, np.arange(len(stamps))),
                                      np.stack([CR.crop_rgb(im, box, 128) for im in expected]))
        assert ef.robot(cam)["ok"].all()
        assert len(ef.robot(cam)["joint_positions"]) == len(stamps)
        g = ef.file[f"cameras/{cam}"]
        expected_status = status if all_cameras or cam == "rig0" else "accepted"
        assert json.loads(g.attrs["calibration"])["status"] == expected_status
        if all_cameras or cam == "rig0":
            assert not any(field in g for field in ("w2c", "tracks", "depth"))
        elif cam == "rig1":
            assert all(field in g for field in ("w2c", "tracks", "depth"))
    assert "gripper_visible" not in ef.sync()
    assert len(timelines(store, key, multiview=False)) == 3
    for access in (lambda: store.sync_sets(key), lambda: sync_index(store, key),
                   lambda: timelines(store, key, multiview=True), lambda: ef.w2c("rig0"),
                   lambda: ef.tracks("rig0"), lambda: ef.track_index("rig0"),
                   lambda: RealworldWindowDataset(store, num_views=1)):
        with pytest.raises(ValueError, match="geometry held"):
            access()
    store.close()


def test_frames_depth_semantics_and_calibration_match_the_preprocessing_outputs(built):
    from s4d.realworld.pipeline import accepted_calibration, load_meta, read_frames
    from s4d.realworld.stages.depth import load_view_depth

    root, ep = built["root"], built["ep"]
    meta = load_meta(root, ep)
    ef = TS.EpisodeFile(built["tmp"] / "store")
    for cam in ef.cameras:
        stream = f"{cam}/left"
        n = len(meta.stream_timestamps_ns[stream])
        H, W = built["adapter"].hw
        box = TS.square_box(H, W)
        idx = list(range(n))
        np.testing.assert_array_equal(ef.timestamps(cam), meta.stream_timestamps_ns[stream])
        np.testing.assert_array_equal(ef.rgb(cam, idx), np.stack([CR.crop_rgb(x, box, 128) for x in
                                                                   read_frames(root, ep, stream, idx)]))
        depth, valid, K = load_view_depth(root, ep, stream, idx, "sensor_depth")
        ref = np.stack([CR.crop_depth(x, box, 128) for x in np.where(valid, depth, 0)])
        got = ef.depth(cam, idx)
        np.testing.assert_array_equal(got, ref)  # lossless
        np.testing.assert_array_equal(ef.valid(cam, idx), ref > 0)
        with h5py.File(rio.stage_dir(root, ep, "sam3_masks") / f"{cam}__left.h5", "r") as f:
            sem = f["semantic"][:]
        np.testing.assert_array_equal(ef.semantic(cam, idx), np.stack([TS.nearest_resample(s, box) for s in sem]))
        np.testing.assert_array_equal(ef.intrinsics_native(cam), K)
        np.testing.assert_allclose(ef.intrinsics(cam), TS.box_affine(box).apply_K(K), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(ef.w2c(cam), accepted_calibration(root, ep, stream))
        np.testing.assert_array_equal(ef.segments(cam), np.zeros(n))  # one recording segment


def test_tracks_robot_state_and_sync_are_aligned_with_their_sources(built):
    from s4d.realworld.pipeline import load_meta, robot_state_at
    from s4d.realworld.stages.depth import frame_sets
    from s4d.realworld.stages.motion import MOVING, load_motion
    from s4d.realworld.tracking import VALID

    root, ep = built["root"], built["ep"]
    meta = load_meta(root, ep)
    ef = TS.EpisodeFile(built["tmp"] / "store")
    total = 0
    for cam in ef.cameras:
        tr = ef.tracks(cam)
        names = json.loads(ef.file[f"cameras/{cam}/tracks"].attrs["chunk_names"])
        for c, name in enumerate(names):
            with np.load(rio.stage_dir(root, ep, "lift3d") / name) as z:
                lift = {k: z[k] for k in z.files}
            mo = load_motion(root, ep, name)
            keep = np.flatnonzero(mo["class"] == MOVING)
            rows = slice(tr["chunk_offset"][c], tr["chunk_offset"][c + 1])
            L = len(lift["frame_indices"])
            np.testing.assert_array_equal(tr["chunk_frames"][c, :L], lift["frame_indices"])
            np.testing.assert_array_equal(ef.timestamps(cam)[tr["chunk_frames"][c, :L]], lift["timestamps_ns"])
            np.testing.assert_array_equal(tr["point_id"][rows], lift["point_ids"][keep])
            np.testing.assert_array_equal(tr["xyz"][rows, :L], lift["xyz_base"][:, keep].transpose(1, 0, 2))  # bitwise
            np.testing.assert_array_equal(tr["status"][rows, :L], lift["status"][:, keep].T)
            np.testing.assert_array_equal(tr["segment"][rows, :L], mo["segment"][:, keep].T)
            assert (tr["chunk_frames"][c, L:] == -1).all() and not np.isfinite(tr["xyz"][rows, L:]).any()
            total += len(keep)
        assert (np.isfinite(tr["xyz"]).all(-1) == (tr["status"] == VALID)).all()
        q, g, ok, _ = robot_state_at(meta, ef.timestamps(cam), ef.max_sync_ns)
        r = ef.robot(cam)
        np.testing.assert_array_equal(r["joint_positions"], q)
        np.testing.assert_array_equal(r["ok"], ok)
    assert total == sum(built["rec"]["summary"]["moving_tracks"].values()) > 0
    sets = frame_sets(meta, [f"{c}/left" for c in ef.cameras], ef.max_sync_ns, allow_unmatched=True)
    frames = ef.sync()["frames"]
    np.testing.assert_array_equal(frames[(frames >= 0).all(1)], sets["index"])


def test_window_loader_reproduces_the_former_export_contract(built):
    """Same windows, cameras, frames, gripper/probe/robot targets, K and poses as the former window export at the same
    square crop; track and gripper weights equal the export's times the crop membership of both pair endpoints."""
    from s4d.realworld.dataset import RealworldCacheDataset

    ds = _dataset(built)
    with pytest.warns(UserWarning, match="MOCK"):
        old = RealworldCacheDataset(built["tmp"] / "old", split="train", image_height=128, image_width=128)
    man = old.manifest
    assert [(w["canonical_indices"], w["stride"]) for w in man["windows"]] == \
        [(ds.window(i)["canonical_indices"], ds.window(i)["stride"]) for i in range(len(ds))]
    assert ds.capacity == man["track_capacity"]
    ef = ds.store.episode(ds.episodes[0])
    names = {c: json.loads(ef.file[f"cameras/{c}/tracks"].attrs["chunk_names"]) for c in ef.cameras}
    removed = 0
    for i in range(len(ds)):
        a, b, w = ds[i], old[i], man["windows"][i]
        for v, cam in enumerate(ef.cameras):
            assert names[cam][ds.window(i)["chunks"][v]] == w["cameras"][f"{cam}/left"]["chunk"]
            assert list(a["meta"]["frame_indices"][v]) == w["cameras"][f"{cam}/left"]["frames"]
        for key in ("gripper_points", "probe_state", "w2c", "c2w", "track_count"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        torch.testing.assert_close(a["K"], b["K"], rtol=0, atol=1e-5)
        assert torch.equal(a["depth"], b["depth"])  # same nearest sampling, lossless storage
        K, w2c = a["K"].numpy(), a["w2c"].numpy()
        src, disp, wo = b["track_src"].numpy(), b["track_disp"].numpy(), b["track_weight"].numpy()
        grip = a["gripper_points"].numpy()
        expect = wo.copy()
        expect_g = b["gripper_weight"].numpy().copy()
        for p, (s, t) in enumerate(PAIR_TIMES):
            for v in range(len(ef.cameras)):
                expect[p, v] *= inside_image(src[p, v], K[v], w2c[v], 128) & \
                    inside_image(src[p, v] + disp[p, v], K[v], w2c[v], 128)
                expect_g[p, v] *= inside_image(grip[s], K[v], w2c[v], 128) & inside_image(grip[t], K[v], w2c[v], 128)
        np.testing.assert_array_equal(a["track_weight"].numpy(), expect)
        np.testing.assert_array_equal(a["gripper_weight"].numpy(), expect_g)
        on = a["track_weight"].numpy() > 0
        np.testing.assert_array_equal(a["track_src"].numpy()[on], src[on])
        np.testing.assert_array_equal(a["track_disp"].numpy()[on], disp[on])
        removed += int((wo > 0).sum() - on.sum())
        assert torch.equal(a["motion_score"], b["motion_score"])
        validate_batch({k: v if k == "meta" else v.unsqueeze(0) for k, v in a.items()})
    assert ds.counts["dropped_too_few_views"] == 0
    # crop-dependent visibility: with every camera's principal point moved 47 px left (the moving tracks straddle the
    # stored image border) the pair weights are exactly the export's times the membership of both endpoints
    for cam in ef.cameras:
        K = ef.intrinsics(cam).copy()
        K[0, 2] -= 47.0
        ef._cache[("K", cam)] = K
    removed = supervised = 0
    for i in range(len(ds)):
        a, b = ds[i], old[i]
        K, w2c = a["K"].numpy(), a["w2c"].numpy()
        src, disp, wo = b["track_src"].numpy(), b["track_disp"].numpy(), b["track_weight"].numpy()
        expect = wo.copy()
        for p in range(len(PAIR_TIMES)):
            for v in range(len(ef.cameras)):
                expect[p, v] *= inside_image(src[p, v], K[v], w2c[v], 128) & \
                    inside_image(src[p, v] + disp[p, v], K[v], w2c[v], 128)
        np.testing.assert_array_equal(a["track_weight"].numpy(), expect)
        removed += int((wo > 0).sum() - (expect > 0).sum())
        supervised += int((wo > 0).sum())
    assert 0 < removed < supervised


def test_loader_options_splits_views_and_workers(built):
    full = _dataset(built)
    with pytest.warns(UserWarning):
        val = RealworldWindowDataset([built["tmp"] / "store"], split="validation", validation_tail_frames=4)
        train = RealworldWindowDataset([built["tmp"] / "store"], split="train", validation_tail_frames=4)
    tv = {tuple(val.window(i)["canonical_indices"]) for i in range(len(val))}
    tt = {tuple(train.window(i)["canonical_indices"]) for i in range(len(train))}
    assert tv and tt and not ({f for w in tv for f in w} & {f for w in tt for f in w})  # frame-disjoint
    with pytest.warns(UserWarning):
        one = RealworldWindowDataset([built["tmp"] / "store"], split="all", num_views=1, max_tracks=4)
    a, b = one[(0, 0)], one[(0, 1)]
    assert a["images"].shape[1] == 1 and a["track_src"].shape[2] == 4 and one.epoch_tagged_indices
    assert one[(0, 0)]["meta"]["camera_ids"] == a["meta"]["camera_ids"]  # deterministic per (index, epoch)
    assert {one[(0, e)]["meta"]["camera_ids"] for e in range(8)} | {b["meta"]["camera_ids"]} == {("rig0",), ("rig1",)}
    with pytest.raises(ValueError, match="exceeds the store's bound"):
        with pytest.warns(UserWarning):
            RealworldWindowDataset([built["tmp"] / "store"], max_sync_ns=10 ** 12)
    loader = torch.utils.data.DataLoader(full, batch_size=3, num_workers=2, collate_fn=collate,
                                         multiprocessing_context="spawn")
    batches = list(loader)
    assert sum(x["images"].shape[0] for x in batches) == len(full)
    for x, i in zip(batches[0]["images"], range(3)):
        assert torch.equal(x, full[i]["images"])


def test_store_lru_bounds_handles_and_cached_arrays(built):
    from s4d.realworld.pipeline import load_meta

    default = TS.write_episode(built["root"], built["ep"], built["tmp"] / "store_default_task",
                               depth_stage="sensor_depth", allow_mock=True)  # task = the recording's dataset folder
    assert default["summary"]["task"] == Path(load_meta(built["root"], built["ep"]).provenance.source_root).name
    rec2 = TS.write_episode(built["root"], built["ep"], built["tmp"] / "store2", task="synthetic2",
                            depth_stage="sensor_depth", allow_mock=True)
    assert rec2["summary"]["file_sha256"] != built["rec"]["summary"]["file_sha256"]  # task attr differs
    with pytest.warns(UserWarning):
        store = TS.CompactStore([built["tmp"] / "store", built["tmp"] / "store2"], max_open=1)
    a, b = store.episodes()
    store.rgb(a, "rig0", [0])
    store.episode(a).tracks("rig0")
    assert store.episode(a)._cache and store.episode(a)._f is not None
    store.rgb(b, "rig0", [0])
    ea = store._files[a]
    assert ea._f is None and not ea._cache  # evicted: handle closed, cached arrays dropped
    ea.rgb("rig0", [0])  # a RETAINED reader reopening goes through the LRU too: b is evicted, one handle stays open
    assert list(store._open) == [a] and store._files[b]._f is None and not store._files[b]._cache
    import pickle

    store.episode(a).tracks("rig0")
    clone = pickle.loads(pickle.dumps(store))  # a worker starts with no handles and no cached arrays
    assert all(ef._f is None and not ef._cache for ef in clone._files.values()) and not clone._open
    clone.rgb(a, "rig0", [0])
    clone.rgb(b, "rig0", [0])
    assert list(clone._open) == [b] and clone._files[a]._f is None
    ds = RealworldWindowDataset(store, split="all")  # the MOCK warning came with the store
    assert len(ds) == 2 * len(_dataset(built)) and ds[len(ds) - 1]["meta"]["task_name"] == "synthetic2"


def test_frame_store_interface_for_baselines_and_fql(built):
    """The ``FrameStore`` protocol of the real-world learning adapters (per-episode, per-camera frame access)."""
    with pytest.warns(UserWarning, match="MOCK"):
        store = TS.CompactStore(built["tmp"] / "store")
    for name in ("episodes", "task", "cameras", "timestamps", "segments", "rgb", "depth", "semantic", "intrinsics",
                 "w2c", "raw_recording"):
        assert callable(getattr(store, name)), name
    assert store.image_hw == (128, 128) and store.has_semantic
    (ep,) = store.episodes()
    assert ep == f"synthetic/{built['ep']}" and store.task(ep) == "synthetic"
    assert store.cameras(ep) == ["rig0", "rig1"]
    cam = "rig1"
    n = len(store.timestamps(ep, cam))
    seq = [n - 4, n - 3, n - 2, n - 1]  # a single-viewpoint sequence + next observation
    rgb, depth, sem = store.rgb(ep, cam, seq), store.depth(ep, cam, seq), store.semantic(ep, cam, seq)
    assert rgb.shape == (4, 128, 128, 3) and rgb.dtype == np.uint8
    assert depth.shape == (4, 128, 128) and depth.dtype == np.float32 and (depth >= 0).all()
    assert sem.dtype == np.uint8 and sem.max() < len(TS.SEMANTIC_CLASSES)
    assert store.intrinsics(ep, cam).shape == (3, 3) and store.w2c(ep, cam).shape == (4, 4)
    assert store.segments(ep, cam).shape == (n,)
    from s4d.realworld.pipeline import load_meta

    source = load_meta(built["root"], built["ep"]).provenance.source_files
    assert store.raw_recording(ep) == (str(source[0].path) if source else "")
    with pytest.raises(IndexError):
        store.rgb(ep, cam, [n])
    # multiview RGB-D + cameras (3D-MVP / SNeRL scene) at one synchronised set, lifted to the base frame
    ef = store.episode(ep)
    s = int(np.flatnonzero((ef.sync()["frames"] >= 0).all(1))[0])
    for v, c in enumerate(store.cameras(ep)):
        f = int(ef.sync()["frames"][s, v])
        d = store.depth(ep, c, [f])[0]
        K, c2w = store.intrinsics(ep, c), np.linalg.inv(store.w2c(ep, c))
        ys, xs = np.nonzero(d > 0)
        z = d[ys, xs]
        cam_pts = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * z, (ys + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        world = cam_pts @ c2w[:3, :3].T + c2w[:3, 3]
        assert np.isfinite(world).all() and len(world) > 0
    import pickle

    clone = pickle.loads(pickle.dumps(store))
    np.testing.assert_array_equal(clone.rgb(ep, cam, [0]), store.rgb(ep, cam, [0]))


def test_online_snerl_dino_targets_equal_the_cached_teacher_path(built, monkeypatch):
    """SNeRL targets are computed online from the stored 128x128 RGB with the Meta-World path (uint8 / 255 ->
    ``dino_target``) and equal the cached-teacher implementation (``dino_snerl_worker`` token grid + the consumer's
    upsampling) on the same image; the pinned weights are sha256-checked."""
    import importlib.util

    import torch.nn.functional as F

    from s4d.baselines.snerl.model import dino_target, to_device_scene
    from s4d.baselines.snerl.upstream import dino_extractor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU only
    ef = TS.EpisodeFile(built["tmp"] / "store")
    image = ef.rgb("rig0", [3])[0]
    report = TS.verify_snerl_teacher("cpu", image)
    assert report["weights_sha256_ok"] and report["online_equals_cached_path"]
    assert report["token_grid"] == [31, 31] and report["dim"] == 384 and report["target_shape"] == [128, 128, 384]
    spec = importlib.util.spec_from_file_location("dino_snerl_worker", REPO / "s4d/realworld/workers/dino_snerl_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    target = to_device_scene({"target_rgb": image}, torch.device("cpu"))["target_rgb"]
    _, out, meta = worker.handle({"params": {"output": "tokens"}}, {"images": target[None].numpy()})
    assert meta["verified_against_consumer"] and meta["device"] == "cpu"
    tokens = torch.from_numpy(out["tokens"])
    cached = F.interpolate(tokens.permute(0, 3, 1, 2), (128, 128), mode="bilinear").permute(0, 2, 3, 1)[0]
    online = dino_target(dino_extractor("cpu"), target)
    assert torch.equal(online, cached)


def test_our_model_forward_and_losses_on_cpu(built, monkeypatch):
    from s4d.config import load_config
    from s4d.losses import rgb
    from s4d.model import render
    from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
    from s4d.model.encoder import Encoder, EncoderConfig
    from s4d.train.loop import Model, forward_losses
    from tests.test_review_options import _fake_rasterizer

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())
    ds = _dataset(built)
    batch = collate([ds[0], ds[len(ds) - 1]])
    cfg = load_config([REPO / "configs/droid/pretrain.yaml"], ["model.anchor_stats=null"])
    cfg["train"]["bf16"] = False
    cfg.setdefault("aug", {})["synth_views"] = 0
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    torch.manual_seed(0)
    enc = Encoder(EncoderConfig(image_height=128, image_width=128, patch_size=16, width=16, depth=1, heads=4,
                                num_slots=1, slot_dim=8, drop_path=0, mask_ratio=0.6))
    model = Model(enc, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group, dynamic=group,
                                                     num_slots=1))).train()
    out = forward_losses(model, batch, cfg, 10, source=torch.tensor([0, 1]))
    out["total"].backward()
    assert torch.isfinite(out["total"])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any("motion" in k for k in out["losses"])


def test_visualization_geometry_store_is_compact_native_depth(built):
    from s4d.realworld.stages.depth import load_view_depth
    from s4d.realworld.vis_store import FILE, native_depth, write_native_geometry

    root, ep = built["root"], built["ep"]
    rec = write_native_geometry(root, ep, built["tmp"] / "vis", depth_stage="sensor_depth")
    assert rec["summary"]["cameras"] == ["rig0", "rig1"]  # every view the depth stage produced (rig2 unverified)
    depth, valid, K = load_view_depth(root, ep, "rig1/left", [0, 5], "sensor_depth")
    got, K2 = native_depth(built["tmp"] / "vis", "rig1", [0, 5])
    ref = np.where(valid, depth, 0)
    assert got.shape == ref.shape and ((got > 0) == (ref > 0)).all()
    assert np.abs(got - ref).max() <= 0.5e-4 + 1e-6  # display precision (0.1 mm steps)
    np.testing.assert_array_equal(K2, K)
    with h5py.File(built["tmp"] / "vis" / FILE, "r") as f:
        assert f["cameras/rig1/depth"].dtype == np.uint16 and "w2c" in f["cameras/rig1"]


def test_store_records_m5_sets_and_the_loader_filters_by_robot_motion(built, tmp_path):
    """Criterion M5 (coordinator decision 2026-11-06): mutual nearest sets without a time cap, the per-set robot-surface
    motion stored once, the loader's configurable threshold (default 5 mm), counts recorded in the store and the stage
    summary; the train_data version is bumped so every episode is re-exported."""
    import shutil

    assert TS.WRITER_VERSION == "2" and TS.SCHEMA_VERSION == "1.1.0" and TS.MAX_SET_MOTION_M == 0.005
    s = built["rec"]["summary"]
    assert s["sync"]["criterion"] == "M5" and s["sync"]["max_set_motion_m"] == 0.005
    assert TS.manifest_fields(built["rec"])["sync"] == s["sync"]
    path = built["tmp"] / "store" / TS.FILE
    with h5py.File(path, "r") as f:
        sg = f["sync"]
        disp, frames = sg["surface_disp_m"][:], sg["frames"][:]
        assert disp.dtype == np.float64 and disp.shape == (len(frames),)  # counts and loaders compare the same values
        assert sg.attrs["criterion"] == "M5" and sg.attrs["cross_camera_cap_ns"] == -1
        assert json.loads(sg.attrs["counts"]) == {k: v for k, v in s["sync"].items() if k != "criterion"}
        assert int(f.attrs["max_sync_ns"]) == int(sg.attrs["robot_sync_max_ns"])  # robot <-> camera tolerance only
    complete = (frames >= 0).all(1)
    assert s["sync"]["complete_sets"] == int(complete.sum())
    assert s["sync"]["accepted"] == int((complete & (disp <= 0.005)).sum())
    full = _dataset(built)
    assert full.counts["sync_criterion"] == "M5" and full.counts["accepted_sets"] <= full.counts["complete_sets"]
    moving = disp[complete & np.isfinite(disp)]
    assert (moving > 0).any()  # the synthetic robot moves
    try:  # a threshold below the smallest moving set keeps only the still sets (possibly none: no window at all)
        strict = _dataset(built, max_set_motion_m=float(moving[moving > 0].min()) / 2)
        assert strict.counts["accepted_sets"] < full.counts["accepted_sets"] and len(strict) < len(full)
    except ValueError as exc:
        assert "empty split" in str(exc)
    with pytest.raises(ValueError, match="only for stores without"):
        _dataset(built, sync_criterion="time")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    shutil.copy(path, legacy / TS.FILE)
    with h5py.File(legacy / TS.FILE, "a") as f:
        del f["sync/surface_disp_m"]
    with pytest.raises(ValueError, match="re-export"):
        with pytest.warns(UserWarning):
            RealworldWindowDataset([legacy / TS.FILE], split="all")
    with pytest.warns(UserWarning):
        old = RealworldWindowDataset([legacy / TS.FILE], split="all", sync_criterion="time")
    assert len(old) > 0 and old.counts["sync_criterion"] == "time"
