"""Session recorder tests: synthetic arrays and local files only; no devices, sockets or GPU work."""

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
    from ops.robot_pc.switch_controller import T_FLANGE_EE

    meta = {"policy": "fixture", "mode": "dry_run", "config": {"rate": 15}, "notes": "눈"}
    resolved = {**meta, "task": "drawer-open"}
    names = {"rgbd": "left", "rgb": "right", "empty": "unused"}
    command = {"reference_id": "r0", "observation_time_ns": 300, "goal_pose7": POSE, "gripper_closure": 0.5}
    rows = [{"t_step_ns": 400, "observation_time_ns": 300, "reference_id": "r0", "status": "ok", "code": None,
             "command": command, "client_ms": {"e2e_ms": 2.5}, "server_ms": {"total_ms": 1.5}},
            {"t_step_ns": 390, "observation_time_ns": 299, "reference_id": "r1", "status": "skipped",
             "code": "no_state"}]
    native = [rgb(i * 10) for i in range(3)]
    depth = [np.full(HW, 1000 + i, np.uint16) for i in range(3)]
    obs = [rgb(i * 10, (128, 128)) for i in range(3)]
    obs_depth = [np.full((128, 128), 0.125 * i, np.float32) for i in range(3)]
    times, ids = [300, 300, 290], [10, 10, 8]
    statuses = [{"pos_request_echo": 100 + i, "current": i + 1, "object": i, "fault": i,
                 "status": 3, "activated": 1} for i in range(3)]
    with SessionRecorder(tmp_path, meta=meta, cameras=cameras(*names), camera_names=names,
                         files={"config.json": resolved}, clock=lambda: 9001, video=False) as rec:
        for i, (t, number) in enumerate(zip(times, ids)):
            # The capture's local id deliberately differs: RealSense metadata is authoritative.
            rec.frame("rgbd", i, t, native[i], depth[i] if i != 1 else None,
                      [0.001, None, 0.002][i], {"color_frame_number": number})
            rec.frame("rgb", i, t, native[i], None, None, {"color_frame_number": number})
            rec.observation("rgbd", number, t, obs[i], obs_depth[i] if i else None)
            rec.observation("rgb", number, t, obs[i], None)
            rec.robot_state(t, [7, 7, 5][i], np.arange(7) + i / 8, np.arange(7) - i / 8,
                            -np.arange(7) - i / 8, np.asarray(POSE) + i / 8, np.asarray(POSE) - i / 8)
            rec.gripper(t, 123 + i, statuses[i])
            rec.executed(t, "track", np.asarray(POSE) + i / 8, None if i == 0 else i / 8, i == 2, f"detail {i}")
            rec.event(t, f"event {i}: 눈")
        rec.gripper_command(401, 117, "target")
        rec.gripper_command(402, 123, "hold", "stop")
        for row in rows:
            rec.step(row)
    result = rec.close()
    assert result == summary_file(rec)
    assert re.fullmatch(r"inference_\d{8}_\d{6}", rec.session_dir.name)
    assert rec.session_dir.parent == tmp_path / "inference_outputs"
    assert not (rec.session_dir / "inference.h5.partial").exists()
    assert not (rec.session_dir / "summary.json.tmp").exists()
    assert result["status"] == "complete" and result["error"] is None
    assert result["meta"] == meta and result["start_unix"] <= result["end_unix"]
    assert result["start_monotonic_ns"] == result["end_monotonic_ns"] == 9001
    assert result["h5_size_bytes"] == (rec.session_dir / "inference.h5").stat().st_size > 0
    assert not any(result["dropped"].values()) and result["video"] == "disabled"
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert set(f) == {"robot", "gripper", "gripper_commands", "cameras", "observations", "action", "policy", "events"}
        assert f.attrs["format"] == "s4d_robot_pc_inference" and f.attrs["format_version"] == 2
        assert f.attrs["created_unix"] == result["start_unix"] and f.attrs["created_monotonic"] == 9001 / 1e9
        assert f.attrs["status"] == "complete" and f.attrs["error"] == ""
        assert json.loads(f.attrs["config_json"]) == json.loads((rec.session_dir / "config.json").read_text()) == resolved
        assert json.loads(f.attrs["camera_mapping_json"]) == names
        np.testing.assert_array_equal(f["robot"].attrs["T_flange_controller_ee"], T_FLANGE_EE)
        for serial, name in (("rgbd", "left"), ("rgb", "right")):
            g, o = f[f"cameras/{name}"], f[f"observations/{name}"]
            assert g.attrs["serial"] == serial
            np.testing.assert_array_equal(g.attrs["K_native"], K)
            assert json.loads(g.attrs["calibration_json"]) == cameras(serial)[serial]["calibration"]
            for group, images in ((g, native), (o, obs)):
                np.testing.assert_array_equal(group["t_ns"], times)
                np.testing.assert_array_equal(group["frame_number"], ids)
                np.testing.assert_array_equal(group["rgb"], images)
                assert group["rgb"].dtype == np.uint8
                assert result["counts"][group.name.lstrip("/")] == 3
            assert g["rgb"].compression == "lzf" and g["rgb"].chunks == (1, *HW, 3)
            assert o["rgb"].compression == "gzip" and o["rgb"].chunks == (1, 128, 128, 3)
            assert [json.loads(x) for x in g["meta_json"].asstr()[:]] == [{"color_frame_number": n} for n in ids]
        g, o = f["cameras/left"], f["observations/left"]
        np.testing.assert_array_equal(g["depth"], [depth[0], depth[2]])
        np.testing.assert_array_equal(g["depth_frame_index"], [0, 2])
        np.testing.assert_array_equal(g["depth_scale_m"], [0.001, 0.002])
        assert g.attrs["depth_scale_m"] == 0.001 and g["depth"].dtype == np.uint16
        np.testing.assert_array_equal(o["depth"], obs_depth[1:])
        np.testing.assert_array_equal(o["depth_frame_index"], [1, 2])
        assert o["depth"].dtype == np.float32
        assert "depth" not in f["cameras/right"] and "depth" not in f["observations/right"]
        assert "depth" not in f["cameras/unused"]
        assert f["observations/unused/rgb"].shape == (0, 0, 0, 3)
        np.testing.assert_array_equal(f["robot/seq"], [7, 7, 5])
        for field, expected in {"q": [np.arange(7) + i / 8 for i in range(3)],
                                "dq": [np.arange(7) - i / 8 for i in range(3)],
                                "tau_ext": [-np.arange(7) - i / 8 for i in range(3)],
                                "flange_pose": [np.asarray(POSE) + i / 8 for i in range(3)],
                                "ee_pose": [np.asarray(POSE) - i / 8 for i in range(3)]}.items():
            np.testing.assert_array_equal(f[f"robot/{field}"], expected)
        assert "status_json" not in f["gripper"]
        np.testing.assert_array_equal(f["gripper/position"], [123, 124, 125])
        for field in statuses[0]:
            np.testing.assert_array_equal(f[f"gripper/{field}"], [s[field] for s in statuses])
        np.testing.assert_array_equal(f["gripper_commands/t_ns"], [401, 402])
        np.testing.assert_array_equal(f["gripper_commands/register"], [117, 123])
        assert list(f["gripper_commands/kind"].asstr()[:]) == ["target", "hold"]
        assert list(f["gripper_commands/reason"].asstr()[:]) == ["", "stop"]
        np.testing.assert_array_equal(f["policy/t_ns"], [400, 390])
        np.testing.assert_array_equal(f["policy/t_obs_ns"], [300, 299])
        assert list(f["policy/status"].asstr()[:]) == ["ok", "no_state"]
        np.testing.assert_array_equal(f["policy/command"], [1, 0])
        assert f["policy/command"].dtype == np.uint8
        np.testing.assert_array_equal(f["policy/goal_flange_pose"], [POSE, [np.nan] * 7])
        np.testing.assert_array_equal(f["policy/gripper"], [0.5, np.nan])
        assert [json.loads(x) for x in f["policy/row_json"].asstr()[:]] == rows
        assert json.loads(f["policy/latency_json"].asstr()[0]) == {"client_ms": rows[0]["client_ms"],
                                                                        "server_ms": rows[0]["server_ms"]}
        np.testing.assert_array_equal(f["action/flange_pose"], [np.asarray(POSE) + i / 8 for i in range(3)])
        np.testing.assert_array_equal(f["action/gripper"], [np.nan, 0.125, 0.25])
        np.testing.assert_array_equal(f["action/sent"], [0, 0, 1])
        assert f["action/sent"].dtype == np.uint8
        assert list(f["action/detail"].asstr()[:]) == [f"detail {i}" for i in range(3)]
        assert list(f["events/text"].asstr()[:]) == [f"event {i}: 눈" for i in range(3)]
        for stream, count in result["counts"].items():
            group = f[stream]
            assert len(group["t_ns"]) == count and group["t_ns"].dtype == np.int64
            assert group["t"].dtype == np.float64
            np.testing.assert_array_equal(group["t"], group["t_ns"][:].astype(np.float64) / 1e9)
            if "/" not in stream:
                for ds in group.values():
                    assert ds.chunks == (1024, *ds.shape[1:])
        np.testing.assert_array_equal(f["policy/t_obs"], f["policy/t_obs_ns"][:].astype(np.float64) / 1e9)


def test_empty_streams_and_idempotent_close(tmp_path):
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), video=False)
    assert len(rec._slots["robot"]) == 65536
    assert len(rec._slots["cameras/cam_rgbd"]) == 256
    assert len(rec._slots["policy"]) == 8192
    assert rec._slots["robot"] is rec._slots["action"] is rec._slots["gripper"]
    assert rec._slots["policy"] is rec._slots["gripper_commands"] is rec._slots["events"]
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
        assert "depth" not in f["cameras/cam_rgbd"]
        assert "depth" not in f["observations/cam_rgbd"]


def test_nonblocking_full_queue_counts_each_stream(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_batch

    def slow(self, f, stream, items):
        entered.set()
        assert release.wait(5), "test did not release slow writer"
        original(self, f, stream, items)

    monkeypatch.setattr(SessionRecorder, "_write_batch", slow)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), queue_size=1, video=False)
    try:
        rec.event(1, "in flight")
        assert entered.wait(5)
        rec.event(2, "queued")
        rec.robot_state(2, 0, POSE, POSE, POSE, POSE, POSE)
        rec.frame("rgbd", 0, 2, rgb(0), None, None, {})
        before = time.monotonic()
        for _ in range(4):
            rec.frame("rgbd", 1, 3, rgb(1), None, None, {})
            rec.observation("rgbd", 1, 3, rgb(1, (8, 8)), None)
            rec.robot_state(3, 1, POSE, POSE, POSE, POSE, POSE)
            rec.gripper(3, 20, {})
            rec.step({"t_step_ns": 3, "observation_time_ns": 3, "status": "ok"})
            rec.gripper_command(3, 20, "target")
            rec.executed(3, "track", POSE, None, False, "dry run")
            rec.event(3, "dropped")
        assert time.monotonic() - before < 0.5
        assert dict(rec.dropped) == dict.fromkeys(rec.dropped, 4)
    finally:
        release.set()
        result = rec.close()
    assert result["status"] == "complete"
    assert result["counts"]["events"] == 2
    assert result["counts"]["robot"] == result["counts"]["cameras/cam_rgbd"] == 1
    assert sum(result["counts"].values()) == 4
    assert result["dropped"] == dict(rec.dropped)
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert list(f["events/text"].asstr()[:]) == ["in flight", "queued"]


def test_enqueued_inputs_are_snapshots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_batch

    def slow(self, f, stream, items):
        entered.set()
        assert release.wait(5)
        original(self, f, stream, items)

    monkeypatch.setattr(SessionRecorder, "_write_batch", slow)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), video=False)
    image, z = rgb(3), np.full(HW, 123, np.uint16)
    metadata = {"nested": [1, 2]}
    q, dq, tau = np.arange(7.0), list(range(7)), tuple(np.array(i) for i in range(7))
    step = {"t_step_ns": 3, "observation_time_ns": 2, "status": "ok", "nested": [1, 2]}
    command = {"t_ns": 4, "nested": [1, 2]}
    try:
        rec.event(0, "gate")
        assert entered.wait(5)
        rec.frame("rgbd", 1, 2, image, z, 0.001, metadata)
        rec.robot_state(2, 1, q, dq, tau, POSE, POSE)
        step["command"] = command
        rec.step(step)
        image.fill(0)
        z.fill(0)
        q.fill(-1)
        dq[:] = [-1] * 7
        for value in tau:
            value.fill(-1)
        step["nested"].append(3)
        command["nested"].append(3)
        metadata["nested"].append(3)
    finally:
        release.set()
        rec.close()
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        np.testing.assert_array_equal(f["cameras/cam_rgbd/rgb"][0], rgb(3))
        np.testing.assert_array_equal(f["cameras/cam_rgbd/depth"][0], np.full(HW, 123, np.uint16))
        assert json.loads(f["cameras/cam_rgbd/meta_json"].asstr()[0]) == {"nested": [1, 2]}
        for field in ("q", "dq", "tau_ext"):
            np.testing.assert_array_equal(f[f"robot/{field}"][0], np.arange(7))
        assert json.loads(f["policy/row_json"].asstr()[0])["nested"] == [1, 2]
        assert json.loads(f["policy/row_json"].asstr()[0])["command"]["nested"] == [1, 2]


def test_writer_exception_finalizes_and_counts_unwritten_rows(tmp_path, monkeypatch):
    entered, release, preserved = threading.Event(), threading.Event(), threading.Event()
    original = SessionRecorder._write_batch

    def failing(self, f, stream, items):
        if any(item[1] == "boom" for item in items):
            entered.set()
            assert release.wait(5)
            raise RuntimeError("injected writer failure")
        original(self, f, stream, items)
        preserved.set()

    monkeypatch.setattr(SessionRecorder, "_write_batch", failing)
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    try:
        rec.event(1, "preserved")
        assert preserved.wait(5)
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


def test_failed_row_is_rolled_back(tmp_path, monkeypatch):
    written = threading.Event()
    original = SessionRecorder._write_batch

    def write(self, f, stream, items):
        original(self, f, stream, items)
        written.set()

    monkeypatch.setattr(SessionRecorder, "_write_batch", write)
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    rec.robot_state(1, 1, POSE, POSE, POSE, POSE, POSE)
    assert written.wait(5)
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
        assert result['videos']['cam_one']['frames'] == 0
        assert result['videos']['cam_one']['file'] is None
        assert result['counts']['cameras/cam_one'] == 3
    """, str(tmp_path))
    path = next((tmp_path / "inference_outputs").glob("*/inference.h5"))
    with h5py.File(path, "r") as f:
        assert len(f["cameras/cam_one/rgb"]) == 3
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
            assert Path(writer.path).name == f"cam_{serial}.mp4"
            assert result["videos"][f"cam_{serial}"]["frames"] == len(writer.frames) == 4
            assert len(f[f"cameras/cam_{serial}/rgb"]) == 4
            assert result["videos"][f"cam_{serial}"]["mapping"] == f"MP4 frame k = cameras/cam_{serial}/rgb[k]"
            for i, frame in enumerate(writer.frames):
                np.testing.assert_array_equal(frame, f[f"cameras/cam_{serial}/rgb"][i][..., ::-1])
    assert result["videos"]["cam_unused"]["frames"] == 0
    assert not (rec.session_dir / "cam_unused.mp4").exists()


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
        rows = len(f["cameras/cam_one/rgb"])
    assert rows == 5
    if cv2 is None:
        assert result["video"].startswith("unavailable:") and "cv2" in result["video"]
        assert result["videos"]["cam_one"]["frames"] == 0
        print("OpenCV absent in this venv: exercised documented video-unavailable path; HDF5 complete.")
    else:
        assert result["video"] == "available"
        reader = cv2.VideoCapture(result["videos"]["cam_one"]["file"])  # decode a local file, never a device
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
        assert decoded == result["videos"]["cam_one"]["frames"] == rows


def test_sigterm_during_close_with_full_queue_does_not_deadlock(tmp_path):
    run_python("""
        import signal
        import sys
        import threading
        import time
        from ops.robot_pc.recorder import SessionRecorder
        entered = threading.Event()
        original = SessionRecorder._write_batch
        def slow(self, f, stream, items):
            entered.set()
            time.sleep(0.15)
            original(self, f, stream, items)
        SessionRecorder._write_batch = slow
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


def test_stream_batches_resize_and_write_each_column_once(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_batch = SessionRecorder._write_batch
    original_resize, original_write = h5py.Dataset.resize, h5py.Dataset.__setitem__
    resizes, writes = {}, {}

    def gate(self, f, stream, items):
        if stream == "events":
            entered.set()
            assert release.wait(5)
        original_batch(self, f, stream, items)

    def resize(ds, *args, **kwargs):
        if ds.name.startswith(("/robot/", "/action/")):
            resizes[ds.name] = resizes.get(ds.name, 0) + 1
        return original_resize(ds, *args, **kwargs)

    def write(ds, key, value):
        if ds.name.startswith(("/robot/", "/action/")):
            assert isinstance(key, slice) and key.start == 0
            assert key.stop == (257 if ds.name.startswith("/robot/") else 129)
            writes[ds.name] = writes.get(ds.name, 0) + 1
        return original_write(ds, key, value)

    monkeypatch.setattr(SessionRecorder, "_write_batch", gate)
    monkeypatch.setattr(h5py.Dataset, "resize", resize)
    monkeypatch.setattr(h5py.Dataset, "__setitem__", write)
    rec = SessionRecorder(tmp_path, meta={}, cameras={}, video=False)
    try:
        rec.event(0, "gate")
        assert entered.wait(5)
        for i in range(257):
            rec.robot_state(i, i, POSE, POSE, POSE, POSE, POSE)
            if i % 2 == 0:
                rec.executed(i, "track", POSE, None, False, "dry run")
    finally:
        release.set()
        result = rec.close()
    assert result["status"] == "complete" and not any(result["dropped"].values())
    assert len(resizes) == len(writes) == 15
    assert set(resizes.values()) == set(writes.values()) == {1}
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        np.testing.assert_array_equal(f["robot/t_ns"], np.arange(257))
        np.testing.assert_array_equal(f["action/t_ns"], np.arange(0, 257, 2))


def test_high_rate_and_image_overflow_cannot_starve_low_rate_streams(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = SessionRecorder._write_batch

    def gate(self, f, stream, items):
        if stream == "events" and items[0][1] == "gate":
            entered.set()
            assert release.wait(5)
        original(self, f, stream, items)

    monkeypatch.setattr(SessionRecorder, "_write_batch", gate)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("one"), queue_size=16, video=False)
    try:
        rec.event(0, "gate")
        assert entered.wait(5)
        for i in range(64):
            rec.robot_state(i, i, POSE, POSE, POSE, POSE, POSE)
            rec.executed(i, "track", POSE, None, False, "dry run")
            rec.gripper(i, 100, {})
            rec.frame("one", i, i, rgb(1), None, None, None)
            rec.observation("one", i, i, rgb(1, (8, 8)), None)
        for i in range(7):
            rec.step({"t_step_ns": i, "observation_time_ns": i, "status": "ok"})
            rec.gripper_command(i, 20, "target")
        rec.event(1, "reserved event")
    finally:
        release.set()
        result = rec.close()
    assert result["status"] == "complete"
    assert result["counts"]["policy"] == result["counts"]["gripper_commands"] == 7
    assert result["counts"]["events"] == 2
    assert all(result["dropped"][s] == 0 for s in ("policy", "gripper_commands", "events"))
    assert sum(result["counts"][s] for s in ("robot", "action", "gripper")) == 16
    assert sum(result["dropped"][s] for s in ("robot", "action", "gripper")) == 3 * 64 - 16
    assert sum(result["counts"][s] for s in ("cameras/cam_one", "observations/cam_one")) == 16
    assert sum(result["dropped"][s] for s in ("cameras/cam_one", "observations/cam_one")) == 2 * 64 - 16
    assert result["dropped"] == summary_file(rec)["dropped"]


@pytest.mark.parametrize("stream,field", [("robot", "dq"), ("cameras/cam_rgbd", "meta_json"),
                                         ("observations/cam_rgbd", "depth")])
def test_mid_batch_failure_rolls_back_all_fields_and_counts_pending_rows(tmp_path, monkeypatch, stream, field):
    written, entered, release = threading.Event(), threading.Event(), threading.Event()
    original_batch, original_write = SessionRecorder._write_batch, h5py.Dataset.__setitem__
    failed = []

    def gate(self, f, path, items):
        if path == "events":
            entered.set()
            assert release.wait(5)
        original_batch(self, f, path, items)
        if path == stream:
            written.set()

    def write(ds, key, values):
        original_write(ds, key, values)
        if ds.name == f"/{stream}/{field}" and len(values) > 1:
            failed.append(len(values))
            raise OSError("injected failure after partial batch write")

    monkeypatch.setattr(SessionRecorder, "_write_batch", gate)
    monkeypatch.setattr(h5py.Dataset, "__setitem__", write)
    rec = SessionRecorder(tmp_path, meta={}, cameras=cameras("rgbd"), video=False)

    def enqueue(i):
        if stream == "robot":
            rec.robot_state(i, i, POSE, POSE, POSE, POSE, POSE)
        elif stream.startswith("cameras/"):
            rec.frame("rgbd", i, i, rgb(i), np.full(HW, i, np.uint16), 0.001, {})
        else:
            rec.observation("rgbd", i, i, rgb(i, (8, 8)), np.full((8, 8), i, np.float32))

    try:
        enqueue(0)
        assert written.wait(5)
        rec.event(0, "gate")
        assert entered.wait(5)
        for i in range(1, 6):
            enqueue(i)
        rec.gripper_command(6, 20, "target")  # already drained into another stream's pending batch at failure
    finally:
        release.set()
        result = rec.close()
    assert failed == [5]
    assert result["status"] == "error" and "partial batch write" in result["error"]
    assert result["counts"][stream] == 1 and result["dropped"][stream] == 5
    assert result["counts"]["gripper_commands"] == 0 and result["dropped"]["gripper_commands"] == 1
    assert sum(result["dropped"].values()) == 6
    assert result == summary_file(rec)
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        assert {len(ds) for ds in f[stream].values()} == {1}
        np.testing.assert_array_equal(f[f"{stream}/t_ns"], [0])
        if stream != "robot":
            np.testing.assert_array_equal(f[f"{stream}/depth_frame_index"], [0])


def test_five_second_live_throughput_and_producer_latency(tmp_path):
    """Three paced producers at 2x real load, including two noisy native cameras and optional MP4.

    A burst of one second's 1x work calibrates the actual writer, storage and encoder on this CPU.
    Only a slower machine reduces the multiplier, targeting 90% utilization and never going below 1x.
    The live phase always lasts five seconds; the close tail is measured separately so a large queue
    cannot disguise an unsustainable writer. Native images use random bytes, not easy-to-compress zeros.
    """
    native_hw, obs_hw = (480, 640), (128, 128)
    configs = {s: {"native_hw": list(native_hw), "depth": False, "K_native": K} for s in ("one", "two")}
    rng = np.random.default_rng(104)
    # Cycle unrelated frames so an installed MP4 encoder also sees realistic motion/entropy.
    native = {s: rng.integers(0, 256, (4, *native_hw, 3), dtype=np.uint8) for s in configs}
    observations = {s: rng.integers(0, 256, (*obs_hw, 3), dtype=np.uint8) for s in configs}
    pose = np.asarray(POSE)
    calibration_start = time.perf_counter()
    with SessionRecorder(tmp_path / "calibration", meta={}, cameras=configs) as calibration:
        for i in range(1000):
            calibration.robot_state(i, i, pose, pose, pose, pose, pose)
            if i % 2 == 0:
                calibration.executed(i, "track", pose, 0.5, False, "dry run")
        for i in range(15):
            for serial in configs:
                calibration.frame(serial, i, i, native[serial][i % 4], None, None, {})
                calibration.observation(serial, i, i, observations[serial], None)
            calibration.step({"t_step_ns": i, "observation_time_ns": i, "status": "ok"})
            calibration.gripper_command(i, 20, "target")
    calibration_seconds = time.perf_counter() - calibration_start
    assert calibration.close()["status"] == "complete"
    assert not any(calibration.dropped.values())
    factor = min(2.0, max(1.0, 0.9 / calibration_seconds))
    robot_rate = 2 * int(1000 * factor / 2)
    policy_rate = int(15 * factor)
    duration = 5
    robot_rows, policy_rows = duration * robot_rate, duration * policy_rate
    rec = SessionRecorder(tmp_path / "live", meta={"load_multiplier": factor}, cameras=configs)
    latencies, errors = [], []
    stop, ready = threading.Event(), threading.Barrier(4)
    start = time.perf_counter() + 0.1

    def pace(i, rate):
        delay = start + i / rate - time.perf_counter()
        if delay > 0:
            stop.wait(delay)
        if stop.is_set():
            raise RuntimeError("throughput producer timed out")
        return int((start + i / rate) * 1e9)

    def control():
        joints = np.arange(7.0)
        for i in range(robot_rows):
            t_ns = pace(i, robot_rate)
            joints[0] = i
            before = time.perf_counter_ns()
            rec.robot_state(t_ns, i, joints, joints, joints, joints, joints)
            latencies.append(time.perf_counter_ns() - before)
            if i % 2 == 0:
                rec.executed(t_ns, "track", pose, 0.5, False, "dry run")

    def images():
        for i in range(policy_rows):
            t_ns = pace(i, policy_rate)
            for serial in configs:
                frame = native[serial][i % 4]
                frame[0, 0, 0] = i % 256
                observations[serial][0, 0, 0] = i % 256
                rec.frame(serial, i, t_ns, frame, None, None, {"device_timestamp": i})
                rec.observation(serial, i, t_ns, observations[serial], None)

    def policy():
        for i in range(policy_rows):
            t_ns = pace(i, policy_rate)
            rec.step({"t_step_ns": t_ns, "observation_time_ns": t_ns, "status": "ok", "action": POSE})
            rec.gripper_command(t_ns, 117, "target")
            rec.event(t_ns, f"step {i}")

    def producer(work):
        try:
            ready.wait(timeout=5)
            work()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=producer, args=(work,), daemon=True) for work in (control, images, policy)]
    try:
        for thread in threads:
            thread.start()
        ready.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=10)
        assert not any(thread.is_alive() for thread in threads), "live workload did not finish in real time"
        assert not errors, errors
        # Hold the five-second window even though the final samples arrive just before its boundary.
        remaining = start + duration - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    finally:
        stop.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        close_start = time.perf_counter()
        result = rec.close()
        close_seconds = time.perf_counter() - close_start

    expected = dict.fromkeys(result["counts"], 0)
    expected.update(robot=robot_rows, action=robot_rows // 2,
                    policy=policy_rows, gripper_commands=policy_rows, events=policy_rows)
    for serial in configs:
        expected[f"cameras/cam_{serial}"] = expected[f"observations/cam_{serial}"] = policy_rows
    p99_us = float(np.percentile(latencies, 99)) / 1000
    print(f"Recorder throughput: calibration={calibration_seconds:.3f}s, "
          f"robot={robot_rate}/s, executed={robot_rate // 2}/s, "
          f"native RGB={2 * policy_rate}/s (two cameras), observations={2 * policy_rate}/s, "
          f"steps=commands={policy_rate}/s; drops={sum(result['dropped'].values())}; "
          f"robot call p99={p99_us:.1f}us, max={max(latencies) / 1000:.1f}us; "
          f"close={close_seconds:.3f}s; video={result['video']}")
    assert result["status"] == "complete", result["error"]
    assert result["counts"] == expected
    assert not any(result["dropped"].values()), result["dropped"]
    assert p99_us < 500
    assert close_seconds < 2, "writer did not keep up with the live workload"
    assert time.perf_counter() - calibration_start < 20
    assert result == summary_file(rec)
    with h5py.File(rec.session_dir / "inference.h5", "r") as f:
        for stream, count in expected.items():
            group = f[stream]
            assert {len(ds) for ds in group.values()} == {count}
            for name, ds in group.items():
                if name.endswith("_ns"):
                    assert np.all(np.diff(ds[:]) >= 0), f"nonmonotonic {stream}/{name}"
        np.testing.assert_array_equal(f["robot/seq"], np.arange(robot_rows))
        np.testing.assert_array_equal(f["robot/q"][:, 0], np.arange(robot_rows))
        for serial in configs:
            for kind in ("cameras", "observations"):
                np.testing.assert_array_equal(f[f"{kind}/cam_{serial}/frame_number"], np.arange(policy_rows))
                np.testing.assert_array_equal(f[f"{kind}/cam_{serial}/rgb"][:, 0, 0, 0],
                                              np.arange(policy_rows) % 256)
            if result["videos"][f"cam_{serial}"]["status"] == "available":
                assert result["videos"][f"cam_{serial}"]["frames"] == policy_rows


@pytest.mark.parametrize("damage", [None, "seconds", "policy_goal", "video_count", "frame_gap", "frame_reverse"])
def test_v2_named_camera_policy_round_trip_verifier(tmp_path, monkeypatch, damage):
    from ops.robot_pc import switch_controller as sc
    from ops.robot_pc.gripper import closure_to_position
    from scripts.robot_pc.verify_session import verify

    # Exercise encoder/decoder row mapping even on CPU environments without OpenCV.
    # The optional real-codec test above independently decodes a real MP4 when cv2 is installed.
    videos = {}
    class Writer:
        def __init__(self, path, *args):
            self.frames = videos.setdefault(path, [])
            Path(path).touch()

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            pass

    class Capture:
        def __init__(self, path):
            self.frames = iter(videos.get(path, []))

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoWriter=Writer, VideoCapture=Capture,
                                                          VideoWriter_fourcc=lambda *args: 0))
    meta = {"policy": "fixture", "mode": "dry_run", "config": {"executor": {"rate_hz": 500}}}
    cmd = {"reference_id": "r", "observation_time_ns": 1_060_000_000, "goal_pose7": POSE,
           "measured_pose7": POSE, "gripper_closure": 0.5, "gripper_register": closure_to_position(0.5),
           "translation_m": 0.0, "rotation_deg": 0.0}
    with SessionRecorder(tmp_path, meta=meta, cameras=cameras("device"), camera_names={"device": "left"}) as rec:
        for i in range(200):
            rec.robot_state(10**9 + i * 10**6, i, np.ones(7), np.zeros(7), np.zeros(7), POSE, sc.ee_from_flange(POSE))
        for i in range(10):
            rec.gripper(10**9 + i * 20_000_000, 117, {"fault": 0, "status": 3, "activated": 1})
        for i in range(3):
            t = 10**9 + i * 66_666_667
            rec.frame("device", i, t, rgb(20), None, None, {"color_frame_number": 42 + i})
            rec.observation("device", 42 + i, t, rgb(20, (128, 128)), None)
        rec.step({"t_step_ns": 1_070_000_000, "observation_time_ns": 1_060_000_000, "status": "ok",
                  "command": cmd, "execution_flange_pose": POSE,
                  "client_ms": {"e2e_ms": 10}, "server_ms": {"total_ms": 5}})
        rec.executed(1_070_000_000, "track", POSE, 0.5, False, "dry run")
        rec.executed(1_072_000_000, "track", POSE, 0.5, False, "dry run")
        rec.gripper_command(1_070_000_000, 117, "target")
        rec.gripper_command(1_080_000_000, 117, "hold", "stop")
    if damage == "video_count":
        summary = summary_file(rec)
        summary["videos"]["left"]["frames"] += 1
        (rec.session_dir / "summary.json").write_text(json.dumps(summary))
    elif damage:
        with h5py.File(rec.session_dir / "inference.h5", "r+") as f:
            if damage == "seconds":
                f["policy/t_obs"][0] += 0.001
            elif damage == "policy_goal":
                f["policy/goal_flange_pose"][0, 0] += 0.01
            else:
                f["cameras/left/frame_number"][2] = 51 if damage == "frame_gap" else 35
    report = verify(rec.session_dir)
    assert report["ok"] is (damage is None), report["failures"]
    camera = report["checks"]["cameras_video"]["per_camera"]["left"]
    assert camera["serial"] == "device"
    assert camera["frame_number_gap_max"] == (8 if damage in ("frame_gap", "frame_reverse") else 1)
    if damage is None:
        assert camera["frames"] == camera["mp4_frames"] == 3
        assert report["checks"]["executed"]["sent_true"] == 0
    else:
        expected = {"seconds": "time_columns", "policy_goal": "policy_columns", "video_count": "cameras_video",
                    "frame_gap": "cameras_video", "frame_reverse": "cameras_video"}[damage]
        assert expected in report["failures"]
