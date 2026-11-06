"""Exact compact-store -> recorder frame identity, including the panda adapter's unpaired colour drop rule."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from s4d.realworld.adapters.panda_teleop import PandaTeleopH5Adapter
from s4d.realworld.train_store import recording_segments
from s4d.rwlearn import actions as A
from s4d.rwlearn import fql_data as FD
from s4d.rwlearn import transitions as T
from s4d.rwlearn.pixel import OfflinePixelTransitions
from s4d.rwlearn.store import MemoryStore
from tests import test_realworld_rgbd as RGBD
from tests.test_rwlearn_fql_pipeline import store_on_fixture

panda_file = RGBD.panda_file


@pytest.fixture
def recording(panda_file):
    """The adapter's synthetic JPEG/depth recording, with 15 Hz cameras and fresh 100 Hz robot signals."""
    path = panda_file.path
    with h5py.File(path, "r+") as f:
        t0 = float(f["cameras/cam_101/t"][0])
        robot_t = t0 + np.arange(-1, 41) * 0.01
        n = len(robot_t)
        pose = np.tile([0.1, 0.2, 0.3, 0, 0, 0, 1], (n, 1))
        pose[:, 0] += (robot_t - t0) * 0.01
        command = pose.copy()
        command[:, 0] += 0.01
        teleop = {"max_position_lead": 0.04, "max_rotation_lead_deg": 15,
                  "max_linear_speed": 0.1, "max_angular_speed": 0.5}
        f.attrs.update(episode_index=9, task_name=path.parent.name, config_yaml=json.dumps({"teleop": teleop}))
        arrays = {"robot/t": robot_t, "robot/flange_pose": pose, "robot/q": np.zeros((n, 7)),
                  "robot/dq": np.zeros((n, 7)), "robot/tau_ext": np.zeros((n, 7)),
                  "robot/joint_frames": np.tile([0, 0, 0, 0, 0, 0, 1], (n, 7, 1)),
                  "action/flange_pose": command, "action/flange_target": command,
                  "action/gripper": np.linspace(0.1, 0.9, n), "action/engaged": np.ones(n, bool),
                  "action/sent": np.ones(n, bool), "action/lead_limited": np.zeros(n, bool),
                  "action/segment": np.zeros(n, int), "gripper/t": robot_t,
                  "gripper/position": np.linspace(3, 230, n), "gripper/segment": np.zeros(n, int)}
        for key, data in arrays.items():
            if key in f:
                del f[key]
            f[key] = data
        for serial in ("101", "202"):
            g = f[f"cameras/cam_{serial}"]
            t = t0 + np.arange(6) / 15 + (0.0001 if serial == "202" else 0)
            g["t"][:] = t
            g["t_recv"][:] = t + 0.065
            g["stereo/t"][:] = t + 0.00004
            g["stereo/t_recv"][:] = t + 0.06504
        f["segments/t_end"][:] = t0 + 0.4
    return panda_file


def normalized_store(recording):
    meta = recording.adapter.read_meta("panda")
    cams = {}
    for view in meta.rgbd:
        cam = view.rig_id
        times = meta.stream_timestamps_ns[view.color_stream]
        cams[cam] = {"t": times, "segment": recording_segments(meta, times),
                     "rgb": np.stack([rgb for _, _, rgb in recording.adapter.iter_frames("panda", view.color_stream)]),
                     "K": np.eye(3), "w2c": np.eye(4)}
    return MemoryStore({"panda": {"task": "test", "raw": str(recording.path), "cameras": cams}}, (24, 32))


@pytest.mark.parametrize("dropped", [2, 5])
def test_builder_maps_adapter_drop_actions_rewards_and_pixels(recording, tmp_path, dropped):
    with h5py.File(recording.path, "r+") as f:
        RGBD._drop_records(f["cameras/cam_202/stereo"], dropped)
    store = normalized_store(recording)
    path = tmp_path / "fql.h5"
    FD.build_fql_dataset(store, "e2e_cnn", None, path, episodes=["panda"], assume_final_success=True)
    raw = A.load_raw(recording.path)
    keep = np.delete(np.arange(6), dropped)
    # Pairing evidence survives the compact raw-fixture format too.
    np.testing.assert_array_equal(A.raw_from_arrays(A.raw_to_arrays(raw)).cameras["202"]["paired_colour"], keep)
    with h5py.File(path, "r") as f:
        prov = json.loads(f.attrs["frame_mapping"])["panda"]
        assert prov["202"] == {"raw_count": 6, "store_count": 5, "rule": "stereo/color_index",
                               "dropped_raw_indices": [dropped]}
        assert prov["101"]["dropped_raw_indices"] == []
        for cam, indices in (("101", np.arange(6)), ("202", keep)):
            g = f[f"episodes/panda/{cam}"]
            np.testing.assert_array_equal(g["frames"][:, 0], np.arange(len(indices)))
            np.testing.assert_array_equal(g["raw_frames"][:, 0], indices)
            times = raw.cameras[cam]["t"][indices]
            state_rows = np.searchsorted(raw.robot_t, times, side="right") - 1
            grip_rows = np.searchsorted(raw.gripper_t, times, side="right") - 1
            ok = g["transition_ok"][:]
            assert ok.any()
            np.testing.assert_array_equal(g["times"][:], times)
            np.testing.assert_allclose(g["proprio"][:, :3], raw.measured[state_rows, :3], atol=1e-7)
            np.testing.assert_allclose(g["proprio"][:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (len(indices), 1)))
            np.testing.assert_allclose(g["proprio"][:, -1], (raw.gripper_position[grip_rows] - 3) / 227, atol=1e-7)
            np.testing.assert_allclose(g["measured_pose"][:], A.pose7_to_matrix(raw.measured[state_rows]))
            ks = np.flatnonzero(ok)
            command_rows = np.searchsorted(raw.robot_t, times[ks + 1], side="left") - 1
            np.testing.assert_allclose(g["action_phys"][:][ok, :3],
                                       raw.command[command_rows, :3] - raw.measured[state_rows[ks], :3], atol=1e-7)
            np.testing.assert_allclose(g["action_phys"][:][ok, 3:6], 0)
            np.testing.assert_allclose(g["action_phys"][:][ok, 6], raw.gripper_command[command_rows], atol=1e-7)
            if cam == "101" or dropped == 2:
                assert g["success"][-1] and g["reward"][-2] == 0 and g["mask"][-2] == 0
            else:
                assert not g["success"][:].any() and g["mask"][-2] == 1
            if cam == "202" and dropped == 2:
                assert not ok[1] and g["run"][1] != g["run"][2]
    mapping = {cam: A.camera_frame_indices(raw, cam, store.timestamps("panda", cam), store.segments("panda", cam))
               for cam in store.cameras("panda")}
    tl, = T.timelines(store, "panda", True, raw_frames=mapping)
    for v, cam in enumerate(tl.cameras):
        np.testing.assert_array_equal(tl.raw_frames[:, v], mapping[cam][tl.frames[:, v]])
    if dropped == 2:
        assert np.any(tl.frames[:, 1] != tl.raw_frames[:, 1])
    tr = T.episode_transitions(raw, tl, A.contract_for(raw))
    assert tr["transition_ok"].any()
    assert tr["label"] == ("success" if dropped == 2 else "success_frame_unreachable")
    pixels = OfflinePixelTransitions(path, store, p_aug=None)
    np.testing.assert_array_equal(pixels.frames.numpy(), np.concatenate([store.rgb("panda", cam, np.arange(
        len(store.timestamps("panda", cam)))) for cam in ("101", "202")]))
    from ops.inference_server.contract_checks import singleview_family

    report = singleview_family(store, "panda")
    assert report["passed"]
    assert report["cameras"]["202"]["label"] == ("success" if dropped == 2 else "success_frame_unreachable")


@pytest.mark.parametrize("case, message", [
    ("missing", "no exact raw timestamp match"), ("past_end", "no exact raw timestamp match"),
    ("duplicate", "strictly increasing"), ("out_of_order", "strictly increasing"),
    ("raw_duplicate", "strictly increasing"), ("unexplained_drop", "not explained by stereo/color_index"),
    ("segment", "store segments differ")])
def test_builder_refuses_invalid_frame_identity(recording, tmp_path, case, message):
    store = normalized_store(recording)
    cam = store._eps["panda"]["cameras"]["202"]
    if case == "missing":
        cam["t"][2] += 1  # exact means even a 1 ns difference fails
    elif case == "past_end":
        cam["t"][-1] += 1
    elif case == "duplicate":
        cam["t"][2] = cam["t"][1]
    elif case == "out_of_order":
        cam["t"][[1, 2]] = cam["t"][[2, 1]]
    elif case == "raw_duplicate":
        with h5py.File(recording.path, "r+") as f:
            t = f["cameras/cam_202/t"]
            t[2] = t[1]
    elif case == "unexplained_drop":
        cam["t"], cam["segment"] = np.delete(cam["t"], 2), np.delete(cam["segment"], 2)
    else:
        cam["segment"][2] += 1
    path = tmp_path / "bad.h5"
    with pytest.raises(ValueError, match=message):
        FD.build_fql_dataset(store, "e2e_cnn", None, path, episodes=["panda"], assume_final_success=True)
    assert not path.exists()


def test_legacy_fixture_without_pairing_refuses_interior_omissions(tmp_path):
    store = store_on_fixture(n=8, hw=8)
    ep = store.episodes()[0]
    cam = store.cameras(ep)[0]
    c = store._eps[ep]["cameras"][cam]
    c["t"], c["segment"] = np.delete(c["t"], 2), np.delete(c["segment"], 2)
    with pytest.raises(ValueError, match="only a raw prefix is allowed without pairing metadata"):
        FD.build_fql_dataset(store, "e2e_cnn", None, tmp_path / "bad.h5", episodes=[ep], assume_final_success=True)


# Collect only where the recording is available; an available recording MUST pass, never skip or xfail.
REAL_RECORDING = Path("/home/ws/data/real_world/drawer-open/episode_000009.h5")
if REAL_RECORDING.is_file():
    def test_real_drawer_episode_9_adapter_timeline_maps_exactly(record_property):
        cam = "135122073201"
        adapter = PandaTeleopH5Adapter({"episodes": {"drawer-open-000009": str(REAL_RECORDING)}})
        meta = adapter.read_meta("drawer-open-000009")
        # This is the normalized colour timeline that ingest and train_store.write_episode copy.
        times = meta.stream_timestamps_ns[f"{cam}/color"]
        raw = A.load_raw(REAL_RECORDING)
        indices = A.camera_frame_indices(raw, cam, times, recording_segments(meta, times))
        with h5py.File(REAL_RECORDING, "r") as f:
            expected = f[f"cameras/cam_{cam}/stereo/color_index"][()]
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(raw.cameras[cam]["t"][indices], times)
        dropped = np.setdiff1d(np.arange(len(raw.cameras[cam]["t"])), indices)
        assert len(dropped) == 1 and len(times) == len(raw.cameras[cam]["t"]) - 1
        record_property("mapping", json.dumps({"camera": cam, "raw_frames": len(raw.cameras[cam]["t"]),
                                              "store_frames": len(times), "dropped_raw_indices": dropped.tolist()}))
