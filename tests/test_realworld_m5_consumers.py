"""Non-training consumers of the M5 cross-camera sets (2026-11-06): the viewer and the contract checks take their
multi-view sets from the training store (one source of truth with the loaders); ``train_data.max_sync_ns`` is only the
robot <-> camera tolerance; the contract checks report a single-view family that needs no cross-camera set."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from s4d.realworld import pipeline
from s4d.realworld.depth_stages import configured_sets_policy, episode_frame_sets, store_sets
from s4d.realworld.train_store import FILE, FORMAT, MAX_SET_MOTION_M, SCHEMA_VERSION
from s4d.realworld.viewer_data import describe, load_episode

PERIOD = 66_666_667
FIXTURE = Path(__file__).parent / "fixtures" / "rwlearn" / "raw_mug_hanging_ep3_actions.npz"


def write_store(root, ep, *, offsets=(0, 30_500_000, 5_000_000), disp=None, schema=SCHEMA_VERSION, n=6):
    """Minimal compact store: three free-running cameras ``a``, ``b``, ``c`` (camera b ~half a period off, beyond any
    former 30 ms cap), mutual-nearest sets for every reference frame, per-set robot motion ``disp``."""
    d = Path(root) / "train_data" / ep
    d.mkdir(parents=True, exist_ok=True)
    cams = ["a", "b", "c"]
    with h5py.File(d / FILE, "w") as f:
        f.attrs.update(format=FORMAT, schema_version=schema, episode_id=ep, task="t", cameras=json.dumps(cams),
                       reference_camera="a", image_size=128, max_sync_ns=30_000_000, raw_recording="")
        for c, off in zip(cams, offsets):
            g = f.create_group(f"cameras/{c}")
            g.attrs["stream_id"] = c
            g.create_dataset("timestamps_ns", data=np.arange(n, dtype=np.int64) * PERIOD + off)
        s = f.create_group("sync")
        s.create_dataset("frames", data=np.tile(np.arange(n, dtype=np.int32)[:, None], (1, 3)))
        s.create_dataset("error_ns", data=np.tile(np.asarray(offsets, np.int64), (n, 1)))
        if disp is not None:
            s.create_dataset("surface_disp_m", data=np.asarray(disp, np.float64))
    return d / FILE


def test_policy_is_m5_with_a_store_and_never_uses_the_robot_tolerance_as_a_cross_camera_cap(tmp_path):
    cfg = {"stages": {"train_data": {"max_sync_ns": 30_000_000}}}
    assert configured_sets_policy(tmp_path, "ep", cfg) == {}  # no store: strict, the robot tolerance is no cap
    assert configured_sets_policy(tmp_path, "ep", {"stages": {"export": {"max_sync_ns": 20_000_000}}}) == \
        {"max_sync_ns": 20_000_000, "allow_unmatched": True}  # legacy window-export roots keep their explicit cap
    write_store(tmp_path, "ep", disp=np.zeros(6))
    assert configured_sets_policy(tmp_path, "ep", cfg) == {"criterion": "M5", "max_set_motion_m": MAX_SET_MOTION_M}
    cfg["stages"]["train_data"]["max_set_motion_m"] = 0.008  # never diverges from the loaders' threshold
    assert configured_sets_policy(tmp_path, "ep", cfg)["max_set_motion_m"] == MAX_SET_MOTION_M


def test_store_sets_accept_by_robot_motion_whatever_the_phase_offset(tmp_path):
    disp = [0.001, 0.0049, 0.006, 0.002, np.inf, 0.005]
    write_store(tmp_path, "ep", disp=disp)
    s = store_sets(tmp_path, "ep")
    assert s["criterion"] == "M5" and s["streams"] == ["a", "b", "c"]
    np.testing.assert_array_equal(s["reference_rows"], [0, 1, 3, 5])  # 6 mm and the missing robot state rejected
    np.testing.assert_array_equal(s["index"], np.tile([[0], [1], [3], [5]], (1, 3)))
    assert (s["spread_ns"] == 30_500_000).all()  # camera b sits 30.5 ms off: still one set (old cap: none)
    assert s["counts"] == {"reference_frames": 6, "complete_sets": 6, "accepted": 4, "rejected_motion": 1,
                           "rejected_no_robot_state": 1, "max_set_motion_m": MAX_SET_MOTION_M}
    assert s["unmatched"]["b"] == [2, 4]
    # caller's stream order (columns follow it) and threshold
    t = store_sets(tmp_path, "ep", ["c", "a"], max_set_motion_m=0.008)
    assert t["streams"] == ["c", "a"] and len(t["index"]) == 5
    np.testing.assert_array_equal(t["stamps_ns"][:, 0] - t["stamps_ns"][:, 1], 5_000_000)
    with pytest.raises(ValueError, match="not in the training store"):
        store_sets(tmp_path, "ep", ["a", "d"])


def test_store_sets_refuse_a_store_without_per_set_motion(tmp_path):
    write_store(tmp_path, "ep", disp=None, schema="1.0.0")
    with pytest.raises(ValueError, match="re-export train_data"):
        store_sets(tmp_path, "ep")
    with pytest.raises(ValueError, match="re-export train_data"):
        episode_frame_sets(None, tmp_path, "ep", ["a", "b", "c"],
                           {"stages": {"export": {"max_sync_ns": 30_000_000}}})


@pytest.fixture
def viewer_meta(monkeypatch):
    ref = np.arange(6, dtype=np.int64) * PERIOD
    stamps = {"a": ref, "b": ref + 30_500_000, "c": ref + 5_000_000}
    meta = SimpleNamespace(stream_timestamps_ns=stamps,
                           views=lambda: [SimpleNamespace(stream_id=s, kind="rgbd") for s in stamps])
    monkeypatch.setattr(pipeline, "load_meta", lambda *_: meta)
    monkeypatch.setattr(pipeline, "accepted_calibration", lambda *_: np.eye(4))
    return meta


def test_viewer_uses_the_store_m5_sets(tmp_path, viewer_meta):
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({"stages": {"train_data": {"max_sync_ns": 30_000_000}}}))
    with pytest.raises(ValueError, match="do not pair 1:1"):  # without a store: strict pairing, camera b is 30.5 ms off
        load_episode(tmp_path, "ep")
    write_store(tmp_path, "ep", disp=[0.001, 0.001, 0.009, 0.001, 0.001, 0.001])
    ve = load_episode(tmp_path, "ep")
    assert ve.n_sets == 5
    d = describe(ve)
    assert d["criterion"] == "M5" and d["max_set_motion_m"] == MAX_SET_MOTION_M
    assert d["unmatched_frames"] == {"a": [2], "b": [2], "c": [2]}
    s = episode_frame_sets(viewer_meta, tmp_path, "ep", ["a", "b", "c"])
    np.testing.assert_array_equal(s["index"], ve.sets["index"])


def test_singleview_family_needs_no_cross_camera_set():
    """The single-view family on the real action fixture (mug-hanging episode 3; three cameras): per-camera frames
    and transitions; identical whatever the cross-camera sets (none are read)."""
    from ops.inference_server.contract_checks import singleview_family
    from s4d.rwlearn import actions as A
    from s4d.rwlearn.store import MemoryStore

    raw = A.raw_from_arrays(np.load(FIXTURE))
    n = 40
    cams = {s: {"t": raw.cameras[s]["t"][:n], "segment": raw.cameras[s]["segment"][:n],
                "rgb": np.zeros((n, 8, 8, 3), np.uint8), "depth": np.zeros((n, 8, 8), np.float32),
                "K": np.eye(3), "w2c": np.eye(4)} for s in raw.serials}
    store = MemoryStore({"mug-hanging/episode_000003": {"task": "mug-hanging", "raw": str(FIXTURE), "cameras": cams}},
                        (8, 8))
    out = singleview_family(store, "mug-hanging/episode_000003")
    assert out["passed"] and not out["problems"] and sorted(out["cameras"]) == raw.serials
    for row in out["cameras"].values():
        assert row["frames"] == n and row["transitions"] > 0 and "robot_ok_fraction" not in row
    # one camera without any frame fails the family (never silently skipped)
    cams[raw.serials[0]] = {**cams[raw.serials[0]], "t": raw.cameras[raw.serials[0]]["t"][:0],
                            "segment": raw.cameras[raw.serials[0]]["segment"][:0]}
    bad = singleview_family(store, "mug-hanging/episode_000003")
    assert not bad["passed"] and not bad["cameras"][raw.serials[0]]["passed"]
    # a recording camera missing from the store fails the family (never silently skipped)
    full = singleview_family(store, "mug-hanging/episode_000003", expected_cameras=raw.serials + ["999"])
    assert not full["passed"] and "missing from the training store" in full["problems"][0]
    # an unexpected error is a tooling error, never a family verdict
    from s4d.rwlearn import transitions as T

    def boom(*a, **k):
        raise ValueError("shapes do not broadcast")
    T_orig = T.episode_transitions
    try:
        T.episode_transitions = boom
        with pytest.raises(ValueError, match="broadcast"):
            singleview_family(store, "mug-hanging/episode_000003")
    finally:
        T.episode_transitions = T_orig
    # a recording outside the action contract is a recorded problem of this family
    broken = A.raw_from_arrays(np.load(FIXTURE))
    broken.teleop = {}
    worse = singleview_family(store, "mug-hanging/episode_000003", raw=broken)
    assert not worse["passed"] and worse["problems"][0].startswith("action contract")


@pytest.fixture
def contract_episode(tmp_path, monkeypatch):
    """A real single-view action timeline; every complete cross-camera set exceeds M5."""
    from ops.inference_server import contract_checks as cc
    from s4d.realworld import io as rio
    from s4d.rwlearn import actions as A

    raw = A.raw_from_arrays(np.load(FIXTURE))
    n = 40
    path = write_store(tmp_path, "ep", disp=np.full(n, 0.006), n=n)
    with h5py.File(path, "r+") as f:
        f.attrs.update(cameras=json.dumps(raw.serials), reference_camera=raw.serials[0], raw_recording=str(FIXTURE))
        for old, cam in zip(("a", "b", "c"), raw.serials):
            f.move(f"cameras/{old}", f"cameras/{cam}")
            g = f[f"cameras/{cam}"]
            g.attrs["stream_id"] = f"{cam}/color"
            g["timestamps_ns"][:] = raw.cameras[cam]["t"][:n]
            g.create_dataset("segment", data=raw.cameras[cam]["segment"][:n])
            g.create_dataset("robot/ok", data=np.ones(n, bool))
            g.create_dataset("robot/sync_err_ns", data=np.zeros(n, np.int64))
            for key, value in {"chunk_frames": np.empty((0, 3), np.int64),
                               "chunk_query": np.empty(0, np.int64), "chunk_offset": np.zeros(1, np.int64)}.items():
                g.create_dataset(f"tracks/{key}", data=value)
    (path.parent / rio.RECORD).write_text("{}")
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.safe_dump({"output_root": str(tmp_path), "baselines": ["snerl"]}))
    meta = SimpleNamespace(views=lambda: [SimpleNamespace(rig_id=c, stream_id=f"{c}/color", kind="rgbd")
                                         for c in raw.serials],
                           stream=lambda _: SimpleNamespace(intrinsics=SimpleNamespace(K=lambda: np.eye(3))))
    monkeypatch.setattr(pipeline, "load_meta", lambda *_: meta)
    monkeypatch.setattr(cc, "local_head", lambda: "test-commit")
    return SimpleNamespace(cc=cc, path=path, root=tmp_path,
                           argv=["--pipeline", str(cfg), "--server", str(tmp_path / "absent-server.yaml"),
                                 "--episode", "ep"],
                           report=tmp_path / "reports/ep/contracts/contract_checks.json")


@pytest.mark.parametrize("legacy", [False, True], ids=["empty-m5", "train-data-v1"])
def test_contract_records_independent_families_without_a_server(contract_episode, legacy, monkeypatch):
    case = contract_episode
    if legacy:
        with h5py.File(case.path, "r+") as f:
            del f["sync/surface_disp_m"]
            f.attrs["schema_version"] = "1.0.0"

    def no_remote(*a, **kw):
        pytest.fail("a single-view check or empty/refused M5 split must not load a server")

    monkeypatch.setattr(case.cc.Server, "load", no_remote)
    assert case.cc.main(case.argv) == 0
    report = json.loads(case.report.read_text())
    assert report["singleview"]["passed"] is True
    assert all(c["robot_ok_fraction"] == 1.0 and c["transitions"] > 0
               for c in report["singleview"]["cameras"].values())
    assert report["multiview"]["passed"] is False and report["dataset"] is None
    problem = report["multiview"]["problems"][0]
    assert ("re-export train_data" if legacy else "empty split train") in problem
    assert report["forward"] == [] and report["snerl"] is None


def test_multiview_tooling_error_is_raised_and_preserves_singleview_report(contract_episode, monkeypatch):
    from s4d.realworld import train_windows

    case = contract_episode

    def broken(*a, **kw):
        raise ValueError("unexpected loader shape")

    monkeypatch.setattr(train_windows, "RealworldWindowDataset", broken)
    with pytest.raises(ValueError, match="unexpected loader shape"):
        case.cc.main(case.argv)
    report = json.loads(case.report.read_text())
    assert report["singleview"]["passed"] is True and report["multiview"]["passed"] is False
    assert "unexpected loader shape" in report["multiview"]["problems"][0]


@pytest.mark.parametrize("total,loss,passed", [(1.0, 1.0, True), ("nan", 1.0, False), (1.0, "nan", False)])
def test_contract_snerl_uses_m5_and_checks_losses(contract_episode, monkeypatch, total, loss, passed):
    import torch
    from s4d.data import contract
    from s4d.realworld import io as rio, train_windows

    case = contract_episode
    with h5py.File(case.path, "r+") as f:
        f["sync/surface_disp_m"][[1, 3, 5]] = 0.001
        camera_names = json.loads(f.attrs["cameras"])
    masks = rio.stage_dir(case.root, "ep", "sam3_masks")
    masks.mkdir(parents=True)
    for cam in camera_names:
        with h5py.File(masks / f"{rio.safe_name(cam + '/color')}.h5", "w") as f:
            f.create_dataset("semantic", data=np.broadcast_to(np.arange(40)[:, None, None], (40, 8, 8)))
    monkeypatch.setattr(pipeline, "read_frames", lambda root, ep, s, idx:
                        np.broadcast_to(np.array(idx, np.uint8)[:, None, None, None], (len(idx), 8, 8, 3)))
    monkeypatch.setattr(pipeline, "accepted_calibration", lambda *_: np.eye(4))
    monkeypatch.setattr(case.cc, "axes_lookat", lambda _: np.zeros(3))

    class Dataset:
        cams = [camera_names]
        counts = {}
        store = SimpleNamespace(image_hw=(8, 8), close=lambda: None)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {}

    monkeypatch.setattr(train_windows, "RealworldWindowDataset", lambda *a, **kw: Dataset())
    monkeypatch.setattr(contract, "collate", lambda _: {"x": torch.zeros(1), "meta": {}})
    monkeypatch.setattr(case.cc, "batch_checks", lambda _: {})
    server = SimpleNamespace(config={}, verify_container=lambda: None, verify_source=lambda _: None,
                             audit_start=lambda: {}, audit_end=lambda _: {"ok": True}, stage_weights=lambda **kw: {})
    monkeypatch.setattr(case.cc.Server, "load", lambda _: server)
    monkeypatch.setattr(case.cc, "pool_config", lambda *a: {})
    monkeypatch.setattr(case.cc, "spec", lambda *a, **kw: None)
    monkeypatch.setattr(case.cc, "RemoteWorkerPool", lambda *a, **kw: SimpleNamespace(close=lambda: None))

    def kernel(name, cfg, arrays, **kw):
        if name == "forward_check":
            return {}, {"total": total, "losses": {"rgb": loss}, "metrics": {"psnr_moving": "nan"},
                        "finite": False}, None
        assert name == "snerl_check"
        np.testing.assert_array_equal(arrays["images"][:, 0, 0, 0, 0], [1, 3, 5])
        np.testing.assert_array_equal(arrays["semantic"][:, 0, 0, 0], [1, 3, 5])
        return {"images128": arrays["images"]}, {
            "scenes": [{"finite": True, "dino_worker_equals_consumer": True}] * 3}, None

    monkeypatch.setattr(case.cc, "RemoteRunner", lambda *a: SimpleNamespace(kernel=kernel, ack=lambda _: None))
    assert case.cc.main(case.argv + ["--snerl-scenes", "3"]) == 0
    report = json.loads(case.report.read_text())
    assert report["singleview"]["passed"] is True and report["multiview"]["passed"] is passed
    assert report["snerl"]["set_criterion"] == "M5"
    assert report["snerl"]["scene_frames"] == [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
