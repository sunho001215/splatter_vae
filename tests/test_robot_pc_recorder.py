"""Session recorder tests: synthetic arrays and local files only; no devices, sockets, GPU work or skips."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from ops.robot_pc.recorder import SessionRecorder

REPO = Path(__file__).resolve().parents[1]
HW = (24, 32)
K = [[100.0, 0.0, 15.5], [0.0, 101.0, 11.5], [0.0, 0.0, 1.0]]
POSE = [0.4, 0.1, 0.5, 0.0, 0.0, 0.0, 1.0]


def cameras(*serials: str) -> dict:
    return {s: {"native_hw": list(HW), "depth": s == "rgbd", "K_native": K,
                "calibration": {"id": s, "aligned_to": "RGB"}} for s in serials}


def rgb(value: int, hw: tuple = HW) -> np.ndarray:
    image = np.full((*hw, 3), value, np.uint8)
    image[..., 0], image[..., 2] = value + 1, value + 2
    return image


def summary_file(rec: SessionRecorder) -> dict:
    return json.loads((rec.session_dir / "summary.json").read_text())


def run_python(source: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(source), *args], cwd=REPO,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True,
                          timeout=15, check=True)


def test_round_trip_every_stream_exact_values_and_summary(tmp_path):
    meta = {"policy_id": "fixture", "checkpoint_sha256": "012abc", "catalog": {"eligible": True},
            "task": "drawer-open", "config": {"rate": 15}, "code_commit": "abcdef", "route": "local fixture",
            "mode": "dry_run", "operator_notes": "original timestamps; 눈"}
    rows = [
        {"t_step_ns": 400, "observation_time_ns": 300, "reference_id": "r0", "end_ids": {"rgbd": 10},
         "bound_state": {"pose7": POSE, "gripper_origin": "measured: fixture register", "gripper_register": 123},
         "action_norm": [0.1] * 7, "flange_pose": POSE, "execution_flange_pose": POSE, "gripper_closure": 0.5,
         "validation": {"accepted": True}, "timings": {"validation_ms": 0.125}, "status": "ok", "code": None},
        {"t_step_ns": 390, "observation_time_ns": 299, "reference_id": "r1", "status": "skipped",
         "code": "no_state", "gripper_origin": "measured: fixture register"},
    ]
    commands = [{"t_ns": 401, "reference_id": "r0", "observation_time_ns": 300, "goal_pose7": POSE,
                 "gripper_closure": 0.5}, {"reference_id": "r1", "observation_time_ns": 299}]
    native = [rgb(i * 10) for i in range(3)]
    depth = [np.full(HW, 1000 + i, np.uint16) for i in range(3)]
    obs = [rgb(i * 10, (8, 8)) for i in range(3)]
    obs_depth = [np.full((8, 8), 0.125 * i, np.float32) for i in range(3)]
    times, ids, scales = [300, 300, 290], [10, 10, 8], [0.001, None, 0.002]
    with SessionRecorder(tmp_path, meta=meta, cameras=cameras("rgbd", "rgb", "empty"),
                         clock=lambda: 9001, video=False) as rec:
        for i, (t, frame_id) in enumerate(zip(times, ids)):
            rec.frame("rgbd", frame_id, t, native[i], depth[i] if i != 1 else None, scales[i],
                      {"device_timestamp": 1.25 + i} if i != 1 else None)
            rec.frame("rgb", frame_id, t, native[i], None, None, None)
            rec.observation("rgbd", frame_id, t, obs[i], obs_depth[i] if i != 0 else None)
            rec.observation("rgb", frame_id, t, obs[i], None)
            rec.robot_state(t, [7, 7, 5][i], np.arange(7) + i / 8, np.arange(7) - i / 8,
                            -np.arange(7) - i / 8, np.asarray(POSE) + i / 8, np.asarray(POSE) - i / 8)
            rec.gripper(t, 123 + i, {"moving": bool(i % 2), "fault": i})
            rec.executed(t, "track", np.asarray(POSE) + i / 8, None if i == 0 else i / 8,
                         i == 2, f"{'sent' if i == 2 else 'dry run'} {i}")
            rec.event(t, f"event {i}: 눈")
        for row, cmd in zip(rows, commands):
            rec.step(row)
            rec.command(cmd)
    result = rec.close()
    assert result == summary_file(rec)
    assert re.fullmatch(r"inference_\d{8}_\d{6}", rec.session_dir.name)
    assert rec.session_dir.parent == tmp_path / "inference_outputs"
    assert not (rec.session_dir / "inference.h5.partial").exists()
    assert not (rec.session_dir / "summary.json.tmp").exists()
    assert result["status"] == "complete" and result["error"] is None
    assert result["meta"] == meta
    assert result["start_unix"] <= result["end_unix"]
    assert result["start_monotonic_ns"] == result["end_monotonic_ns"] == 9001
    assert result["h5_size_bytes"] == (rec.session_dir / "inference.h5").stat().st_size > 0
    assert not any(result["dropped"].values())
    assert result["video"] == "disabled"
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert f.attrs["format"] == "s4d_robot_pc_inference_session"
        assert f.attrs["format_version"] == 1
        assert f.attrs["created_unix"] == result["start_unix"]
        assert f.attrs["status"] == "complete" and f.attrs["error"] == ""
        assert json.loads(f.attrs["meta_json"]) == meta
        for serial in ("rgbd", "rgb"):
            g = f[f"cameras/{serial}"]
            np.testing.assert_array_equal(g["t_ns"], times)
            np.testing.assert_array_equal(g["frame_id"], ids)
            np.testing.assert_array_equal(g["rgb"], native)
            np.testing.assert_array_equal(g.attrs["K_native"], K)
            assert json.loads(g.attrs["calibration_json"]) == cameras(serial)[serial]["calibration"]
            assert g["rgb"].compression == "gzip" and g["rgb"].compression_opts == 1
            assert g["rgb"].chunks == (1, *HW, 3)
            assert g["rgb"].dtype == np.uint8
            assert g["t_ns"].dtype == g["frame_id"].dtype == np.int64
            assert g["t_ns"].maxshape == (None,)
            assert result["counts"][f"cameras/{serial}"] == len(g["rgb"]) == 3
            o = f[f"observations/{serial}"]
            np.testing.assert_array_equal(o["t_ns"], times)
            np.testing.assert_array_equal(o["frame_id"], ids)
            np.testing.assert_array_equal(o["rgb"], obs)
            assert o["rgb"].chunks == (1, 8, 8, 3)
            assert result["counts"][f"observations/{serial}"] == 3
        assert [json.loads(s) for s in f["cameras/rgbd/meta_json"].asstr()[:]] == [
            {"device_timestamp": 1.25}, None, {"device_timestamp": 3.25}]
        assert list(f["cameras/rgb/meta_json"].asstr()[:]) == ["null"] * 3
        g, o = f["cameras/rgbd"], f["observations/rgbd"]
        np.testing.assert_array_equal(g["depth"], [depth[0], depth[2]])
        np.testing.assert_array_equal(g["depth_frame_index"], [0, 2])
        np.testing.assert_array_equal(g["depth_scale_m"], [0.001, 0.002])
        assert g.attrs["depth_scale_m"] == 0.001
        assert g["depth"].dtype == np.uint16 and g["depth"].chunks == (1, *HW)
        np.testing.assert_array_equal(o["depth"], obs_depth[1:])
        np.testing.assert_array_equal(o["depth_frame_index"], [1, 2])
        assert o["depth"].dtype == np.float32
        assert "depth" not in f["cameras/rgb"] and "depth" not in f["observations/rgb"]
        np.testing.assert_array_equal(f["robot/t_ns"], times)
        np.testing.assert_array_equal(f["robot/seq"], [7, 7, 5])
        for name, expected in {
            "q": [np.arange(7) + i / 8 for i in range(3)],
            "dq": [np.arange(7) - i / 8 for i in range(3)],
            "tau_ext": [-np.arange(7) - i / 8 for i in range(3)],
            "flange_pose": [np.asarray(POSE) + i / 8 for i in range(3)],
            "ee_pose": [np.asarray(POSE) - i / 8 for i in range(3)],
        }.items():
            np.testing.assert_array_equal(f[f"robot/{name}"], expected)
        np.testing.assert_array_equal(f["gripper/t_ns"], times)
        np.testing.assert_array_equal(f["gripper/position"], [123, 124, 125])
        assert [json.loads(s) for s in f["gripper/status_json"].asstr()[:]] == [
            {"moving": bool(i % 2), "fault": i} for i in range(3)]
        np.testing.assert_array_equal(f["steps/t_step_ns"], [400, 390])
        np.testing.assert_array_equal(f["steps/observation_time_ns"], [300, 299])
        assert list(f["steps/status"].asstr()[:]) == ["ok", "no_state"]
        assert [json.loads(s) for s in f["steps/json"].asstr()[:]] == rows
        np.testing.assert_array_equal(f["commands/t_ns"], [401, 9001])
        assert [json.loads(s) for s in f["commands/json"].asstr()[:]] == commands
        np.testing.assert_array_equal(f["executed/t_ns"], times)
        assert list(f["executed/kind"].asstr()[:]) == ["track"] * 3
        np.testing.assert_array_equal(f["executed/pose7"], [np.asarray(POSE) + i / 8 for i in range(3)])
        np.testing.assert_array_equal(f["executed/gripper_closure"], [np.nan, 0.125, 0.25])
        np.testing.assert_array_equal(f["executed/sent"], [False, False, True])
        assert f["executed/sent"].dtype == np.bool_
        assert list(f["executed/detail"].asstr()[:]) == ["dry run 0", "dry run 1", "sent 2"]
        np.testing.assert_array_equal(f["events/t_ns"], times)
        assert list(f["events/text"].asstr()[:]) == [f"event {i}: 눈" for i in range(3)]
        for stream in ("robot", "gripper", "executed", "events"):
            assert result["counts"][stream] == 3
        assert result["counts"]["steps"] == result["counts"]["commands"] == 2
        assert result["counts"]["cameras/empty"] == result["counts"]["observations/empty"] == 0
        assert f["cameras/empty/rgb"].shape == (0, *HW, 3)
        assert f["observations/empty/rgb"].shape == (0, 0, 0, 3)


def test_empty_streams_and_idempotent_close(tmp_path):
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), video=False)
    result = rec.close("stopped", "operator stopped")
    path = rec.session_dir / "summary.json"
    stamp = path.stat().st_mtime_ns
    assert rec.close("error", "ignored second close") is result
    assert path.stat().st_mtime_ns == stamp
    assert summary_file(rec) == result
    assert result["status"] == "stopped" and result["error"] == "operator stopped"
    assert not any(result["counts"].values())
    rec.event(1, "after close is not accepted")
    assert summary_file(rec) == result
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert f.attrs["status"] == "stopped" and f.attrs["error"] == "operator stopped"
        lengths = []
        f.visititems(lambda _, value: lengths.append(len(value)) if isinstance(value, h5py.Dataset) else None)
        assert lengths and not any(lengths)
        assert f["cameras/rgbd/depth"].shape == (0, *HW)
        assert "depth" not in f["observations/rgbd"]


def test_nonblocking_full_queue_counts_each_stream(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_item

    def slow(self, f, stream, item):
        entered.set()
        assert release.wait(5), "test did not release slow writer"
        original(self, f, stream, item)

    monkeypatch.setattr(SessionRecorder, "_write_item", slow)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), queue_size=1, video=False)
    try:
        rec.event(1, "in flight")
        assert entered.wait(5)
        rec.event(2, "queued")
        before = time.monotonic()
        for _ in range(4):
            rec.frame("rgbd", 1, 3, rgb(1), None, None, {})
            rec.observation("rgbd", 1, 3, rgb(1, (8, 8)), None)
            rec.robot_state(3, 1, POSE, POSE, POSE, POSE, POSE)
            rec.gripper(3, 20, {})
            rec.step({"t_step_ns": 3, "observation_time_ns": 3, "status": "ok"})
            rec.command({"t_ns": 3})
            rec.executed(3, "track", POSE, None, False, "dry run")
            rec.event(3, "dropped")
        assert time.monotonic() - before < 0.5
        assert dict(rec.dropped) == dict.fromkeys(rec.dropped, 4)
    finally:
        release.set()
        result = rec.close()
    assert result["status"] == "complete"
    assert result["counts"]["events"] == 2
    assert sum(result["counts"].values()) == 2
    assert result["dropped"] == dict(rec.dropped)
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert list(f["events/text"].asstr()[:]) == ["in flight", "queued"]


def test_enqueued_inputs_are_snapshots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_item

    def slow(self, f, stream, item):
        entered.set()
        assert release.wait(5)
        original(self, f, stream, item)

    monkeypatch.setattr(SessionRecorder, "_write_item", slow)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), video=False)
    image, z = rgb(3), np.full(HW, 123, np.uint16)
    metadata = {"nested": [1, 2]}
    try:
        rec.event(0, "gate")
        assert entered.wait(5)
        rec.frame("rgbd", 1, 2, image, z, 0.001, metadata)
        image.fill(0)
        z.fill(0)
        metadata["nested"].append(3)
    finally:
        release.set()
        rec.close()
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        np.testing.assert_array_equal(f["cameras/rgbd/rgb"][0], rgb(3))
        np.testing.assert_array_equal(f["cameras/rgbd/depth"][0], np.full(HW, 123, np.uint16))
        assert json.loads(f["cameras/rgbd/meta_json"].asstr()[0]) == {"nested": [1, 2]}


def test_writer_exception_finalizes_and_counts_unwritten_rows(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_item

    def failing(self, f, stream, item):
        if item[1] == "boom":
            entered.set()
            assert release.wait(5)
            raise RuntimeError("injected writer failure")
        original(self, f, stream, item)

    monkeypatch.setattr(SessionRecorder, "_write_item", failing)
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    try:
        rec.event(1, "preserved")
        rec.event(2, "boom")
        assert entered.wait(5)
        rec.event(3, "queued after failing item")
    finally:
        release.set()
    rec._thread.join(timeout=5)  # a failure must finalize even before close is called
    assert not rec._thread.is_alive()
    rec.event(4, "refused after failure")
    result = rec.close()
    assert result["status"] == "error" and "injected writer failure" in result["error"]
    assert result["counts"]["events"] == 1 and result["dropped"]["events"] == 2
    assert result == summary_file(rec)
    assert not (rec.session_dir / "inference.h5.partial").exists()
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert f.attrs["status"] == "error" and f.attrs["error"] == result["error"]
        assert list(f["events/text"].asstr()[:]) == ["preserved"]


def test_failed_row_is_rolled_back(tmp_path):
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    rec.robot_state(1, 1, POSE, POSE, POSE, POSE, POSE)
    rec.robot_state(2, 2, POSE, [1.0], POSE, POSE, POSE)  # h5py would otherwise broadcast one value to seven
    result = rec.close()
    assert result["status"] == "error" and result["counts"]["robot"] == 1
    assert result["dropped"]["robot"] == 1
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert {len(ds) for ds in f["robot"].values()} == {1}
        np.testing.assert_array_equal(f["robot/t_ns"], [1])


@pytest.mark.parametrize("error", [ValueError("policy failed"), KeyboardInterrupt()])
def test_context_manager_finalizes_and_propagates_errors(tmp_path, error):
    with pytest.raises(type(error)):
        with SessionRecorder(tmp_path, meta={}, cameras={}, video=False) as rec:
            rec.event(44, "before error")
            raise error
    result = rec.close()
    assert result["status"] == "error" and type(error).__name__ in result["error"]
    assert result["counts"]["events"] == 1
    assert summary_file(rec) == result


def test_module_imports_without_torch_cv2_or_h5py(tmp_path):
    result = run_python("""
        import sys
        sys.modules.update(torch=None, cv2=None, h5py=None)
        from ops.robot_pc.recorder import SessionRecorder
        assert all(sys.modules[name] is None for name in ('torch', 'cv2', 'h5py'))
        print(SessionRecorder.__name__)
    """)
    assert result.stdout.strip() == "SessionRecorder"


def test_h5py_import_failure_returns_error_summary(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    result = rec.close()
    assert result["status"] == "error" and "h5py" in result["error"]
    assert result["h5_size_bytes"] == 0
    assert summary_file(rec) == result


def test_cv2_blocked_continues_h5_in_subprocess(tmp_path):
    run_python("""
        import json
        import sys
        import numpy as np
        sys.modules['cv2'] = None
        sys.modules['torch'] = None
        from ops.robot_pc.recorder import SessionRecorder
        cams = {'one': {'native_hw': [24, 32], 'depth': False, 'K_native': np.eye(3).tolist()}}
        with SessionRecorder(sys.argv[1], meta={}, cameras=cams) as rec:
            for i in range(3):
                rec.frame('one', i, 10 + i, np.full((24, 32, 3), i, np.uint8), None, None, None)
        result = rec.close()
        assert result['status'] == 'complete'
        assert result['video'].startswith('unavailable:') and 'cv2' in result['video']
        assert result['videos']['one']['frames'] == 0
        assert result['videos']['one']['file'] is None
        assert result['counts']['cameras/one'] == 3
    """, str(tmp_path))
    path = next((tmp_path / "inference_outputs").glob("*/inference.h5"))
    with h5py.File(path, "r") as f:
        assert len(f["cameras/one/rgb"]) == 3
        assert f.attrs["status"] == "complete"


def test_video_writer_contract_and_row_mapping(tmp_path, monkeypatch):
    writers, threads = [], []

    class VideoWriter:
        def __init__(self, path, fourcc, fps, wh):
            self.path, self.fourcc, self.fps, self.wh = path, fourcc, fps, wh
            self.frames, self.released = [], False
            Path(path).touch()
            writers.append(self)
            threads.append(threading.current_thread().name)

        def isOpened(self):
            return True

        def write(self, frame):
            assert frame.flags.c_contiguous
            self.frames.append(frame.copy())
            threads.append(threading.current_thread().name)

        def release(self):
            self.released = True
            threads.append(threading.current_thread().name)

    def fourcc(*chars):
        assert chars == tuple("mp4v")
        return 1234

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoWriter=VideoWriter, VideoWriter_fourcc=fourcc))
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("one", "two", "unused"), video_fps=12.5)
    assert not writers  # lazy until the first native frame
    for serial in ("one", "two"):
        for i in range(4):
            rec.frame(serial, i, 10 - i, rgb(i), None, None, None)
    result = rec.close()
    assert result["status"] == "complete" and result["video"] == "available"
    assert len(writers) == 2
    assert set(threads) == {"session-recorder"}
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        for serial, writer in zip(("one", "two"), writers):
            assert writer.released and writer.wh == HW[::-1] and writer.fps == 12.5 and writer.fourcc == 1234
            assert Path(writer.path).name == f"camera_{serial}.mp4"
            assert result["videos"][serial]["frames"] == len(writer.frames) == len(f[f"cameras/{serial}/rgb"]) == 4
            assert result["videos"][serial]["mapping"] == f"MP4 frame k = cameras/{serial}/rgb[k]"
            for i, frame in enumerate(writer.frames):
                np.testing.assert_array_equal(frame, f[f"cameras/{serial}/rgb"][i][..., ::-1])
    assert result["videos"]["unused"]["frames"] == 0
    assert not (rec.session_dir / "camera_unused.mp4").exists()


def test_mp4_frame_count_or_documented_unavailable_path(tmp_path, monkeypatch):
    try:
        import cv2
    except ImportError:
        cv2 = None
        monkeypatch.setitem(sys.modules, "cv2", None)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("one"))
    for i in range(5):
        rec.frame("one", i, 30 - i, rgb(i * 10), None, None, None)
    result = rec.close()
    assert result["status"] == "complete"
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        rows = len(f["cameras/one/rgb"])
    assert rows == 5
    if cv2 is None:
        assert result["video"].startswith("unavailable:") and "cv2" in result["video"]
        assert result["videos"]["one"]["frames"] == 0
        print("OpenCV absent in this venv: exercised documented video-unavailable path; HDF5 complete.")
    else:
        assert result["video"] == "available"
        reader = cv2.VideoCapture(result["videos"]["one"]["file"])  # decode a local file, never a device
        try:
            assert reader.isOpened()
            decoded = 0
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                assert frame.shape == (*HW, 3)
                decoded += 1
        finally:
            reader.release()
        assert decoded == result["videos"]["one"]["frames"] == rows


def test_sigterm_during_close_with_full_queue_does_not_deadlock(tmp_path):
    run_python("""
        import signal
        import sys
        import threading
        import time
        from ops.robot_pc.recorder import SessionRecorder
        entered = threading.Event()
        original = SessionRecorder._write_item
        def slow(self, f, stream, item):
            entered.set()
            time.sleep(0.15)
            original(self, f, stream, item)
        SessionRecorder._write_item = slow
        rec = SessionRecorder(sys.argv[1], meta={}, cameras={}, queue_size=1, video=False)
        rec.event(1, 'in flight')
        assert entered.wait(5)
        rec.event(2, 'full queue')
        rec.event(3, 'dropped')
        nested = []
        def stop(signum, frame):
            nested.append(rec.close('stopped'))
        signal.signal(signal.SIGTERM, stop)
        timer = threading.Timer(0.03, lambda: signal.raise_signal(signal.SIGTERM))
        timer.start()
        result = rec.close('stopped')
        timer.join()
        assert nested and nested[0] is result
        assert result['status'] == 'stopped'
        assert result['counts']['events'] == 2 and result['dropped']['events'] == 1
        assert not (rec.session_dir / 'inference.h5.partial').exists()
    """, str(tmp_path))
    path = next((tmp_path / "inference_outputs").glob("*/summary.json"))
    assert json.loads(path.read_text())["status"] == "stopped"


def test_same_second_session_never_overwrites_existing_data(tmp_path, monkeypatch):
    import ops.robot_pc.recorder as module

    monkeypatch.setattr(module.time, "time", lambda: 1_700_000_000.0)
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    rec.event(1, "original")
    rec.close()
    with pytest.raises(FileExistsError):
        SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert list(f["events/text"].asstr()[:]) == ["original"]
