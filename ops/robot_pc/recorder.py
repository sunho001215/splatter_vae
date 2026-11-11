"""Asynchronous, local-only session recording. Only the writer imports h5py and (for video) cv2.

Enqueue methods snapshot their inputs and never wait for disk or queue space. ``close`` is the intentional
blocking boundary: stop producers first, then drain and finalize. It takes no producer/queue locks, so a caller's
SIGTERM handler may call it, including while another close is waiting. Install signal/atexit handlers in the caller.

Missing depth is not padded: ``depth_frame_index`` maps each depth row to its RGB row. Native depth scales are
stored per depth row (NaN means unspecified); the attribute is the first supplied scale. Empty observation images
have unknown spatial dimensions until the first observation. Existing session directories are never overwritten.

Native camera RGB uses lossless LZF: gzip level 1 on noisy 480x640 frames consumes the writer's CPU budget
before two 15 Hz cameras plus MP4 encoding have sufficient headroom. Depth and policy images retain gzip 1.
Numeric columns use 1024-row chunks and all streams are appended in batches, coalesced for at most 10 ms.
"""

from __future__ import annotations

import copy
import json
import math
import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _dataset(group: Any, name: str, dtype: Any, shape: tuple = (), *, image: bool = False,
             native_rgb: bool = False) -> Any:
    options = {"compression": "lzf"} if native_rgb else (
        {"compression": "gzip", "compression_opts": 1} if image else {})
    return group.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype,
                                chunks=(1, *shape) if image else (1024, *shape), **options)


def _append(group: Any, values: dict) -> None:
    """Append column batches atomically, including sparse depth columns with their own row counts."""
    datasets = {key: group[key] for key in values}
    columns = {}
    # Validate and stack before resizing; never let h5py broadcast a malformed row across its fields.
    for key, rows in values.items():
        ds = datasets[key]
        column = np.asarray(rows, dtype=ds.dtype)
        if column.shape != (len(rows), *ds.shape[1:]):
            raise ValueError(f"{ds.name}: expected batch shape {(len(rows), *ds.shape[1:])}, got {column.shape}")
        columns[key] = column
    sizes = {key: len(ds) for key, ds in datasets.items()}
    try:
        for key, column in columns.items():
            ds, start = datasets[key], sizes[key]
            ds.resize(start + len(column), axis=0)
            ds[start:start + len(column)] = column
    except BaseException:
        for key, size in sizes.items():
            datasets[key].resize(size, axis=0)
        raise


def _snapshot(value: Any) -> Any:
    """Copy reusable buffers once; scalar sequences need no deepcopy traversal or memo table."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(v) if isinstance(v, (np.ndarray, dict, tuple, list)) else v for v in value)
    return value


def _image(value: Any, dtype: Any, shape: tuple) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"expected {np.dtype(dtype)} image {shape}, got {array.dtype} {array.shape}")
    return array


def _next_run_dir(parent: Path, prefix: str) -> Path:
    """``<prefix>/runNNN`` (``prefix`` may be nested, e.g. ``drawer-open/e2e_cnn/seed0/right``): 1 + the largest
    existing index in that folder, created exclusively (a concurrent run that takes the index first moves this one
    to the next)."""
    import re

    parent = parent.joinpath(*prefix.split("/"))
    parent.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"run(\d{3,})$")
    used = [int(m.group(1)) for p in parent.iterdir() if (m := pat.match(p.name))]
    n = max(used, default=0) + 1
    while True:
        d = parent / f"run{n:03d}"
        try:
            d.mkdir()
            return d
        except FileExistsError:
            n += 1


class SessionRecorder:
    """One session under ``root/inference_outputs/inference_YYYYMMDD_HHMMSS``.

    ``dropped`` uses HDF5 group paths, e.g. ``cameras/left``, ``observations/left``, ``robot``. Queue overflow
    drops the incoming item. A writer error also counts queued items it cannot write, then finalizes automatically.
    Timestamp columns retain input nanoseconds alongside float64 host CLOCK_MONOTONIC seconds.
    All file access, compression and video conversion belongs to one thread.
    Independent queue budgets reserve 65536 high-rate, 256 image and 8192 low-rate slots by default.
    ``queue_size`` caps each pool independently, so an explicit small value can exercise overflow behavior.
    """

    def __init__(self, root: str | Path, *, meta: dict, cameras: dict[str, dict],
                 clock: Callable[[], int] = time.monotonic_ns, queue_size: int = 65536,
                 video: bool = True, video_fps: float = 15.0, run_prefix: str | None = None,
                 camera_names: dict | None = None, files: dict | None = None):
        """``run_prefix`` (e.g. ``drawer-open/e2e_cnn/seed0/right``): the session folder is
        ``inference_outputs/<run_prefix>/runNNN`` with NNN the next free index (exclusive mkdir, never reused);
        without it ``inference_YYYYMMDD_HHMMSS``. ``camera_names`` {serial: name} names HDF5 camera groups
        and ``<name>.mp4`` (default name: ``cam_<serial>``). ``files`` {filename: JSON-able}: written into
        the folder at start (resolved config, mapping)."""
        if not isinstance(queue_size, int) or queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError("video_fps must be finite and positive")
        self.meta, self.cameras = copy.deepcopy(meta), copy.deepcopy(cameras)
        _json(self.meta)
        for serial, config in self.cameras.items():
            if not serial or serial in (".", "..") or any(c in serial for c in "/\\\x00"):
                raise ValueError(f"invalid camera serial {serial!r}")
            hw = config["native_hw"]
            if len(hw) != 2 or any(not isinstance(n, int) or n <= 0 for n in hw):
                raise ValueError("native_hw must contain positive integer H, W")
            if np.asarray(config["K_native"]).shape != (3, 3):
                raise ValueError("K_native must be 3x3")
            _json(config.get("calibration"))
        self.clock, self.video_fps = clock, float(video_fps)
        self._start_unix, self._start_ns = time.time(), int(clock())
        parent = Path(root).expanduser().resolve() / "inference_outputs"
        parent.mkdir(parents=True, exist_ok=True)
        self.camera_names = {s: str((camera_names or {}).get(s, f"cam_{s}")) for s in self.cameras}
        if len(set(self.camera_names.values())) != len(self.camera_names):
            raise ValueError("camera names must be unique")
        self.config = copy.deepcopy((files or {}).get("config.json", self.meta))
        _json(self.config)
        for v in self.camera_names.values():
            if not v or any(c in v for c in "/\\\x00") or v in (".", ".."):
                raise ValueError(f"invalid camera name {v!r}")
        if run_prefix is None:
            name = time.strftime("inference_%Y%m%d_%H%M%S", time.localtime(self._start_unix))
            self.session_dir = parent / name
            self.session_dir.mkdir()  # exclusive: a same-second collision must not clobber a prior run
        else:
            parts = run_prefix.split("/") if run_prefix else [""]
            if any(not x or x.startswith(".") or any(c in x for c in "\\\x00") for x in parts):
                raise ValueError(f"invalid run prefix {run_prefix!r}")
            self.session_dir = _next_run_dir(parent, run_prefix)
        for fname, obj in {"config.json": self.config, **(files or {})}.items():
            if "/" in fname or fname.startswith("."):
                raise ValueError(f"invalid file name {fname!r}")
            (self.session_dir / fname).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
        self._partial = self.session_dir / "inference.h5.partial"
        self._h5_path = self.session_dir / "inference.h5"
        streams = ["robot", "gripper", "gripper_commands", "policy", "action", "events"]
        streams += [f"{kind}/{name}" for kind in ("cameras", "observations")
                    for name in self.camera_names.values()]
        self.dropped: Counter = Counter(dict.fromkeys(streams, 0))
        self._counts = dict.fromkeys(streams, 0)
        # SimpleQueue's C put is reentrant, unlike Queue's mutex. Atomic deque reservations bound its capacity
        # without a producer lock that a signal-handler close could leave held while waiting for the writer.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        high = deque([None] * queue_size)
        images = deque([None] * min(queue_size, 256))
        low = deque([None] * min(queue_size, 8192))
        self._slots = {stream: images if "/" in stream else high if stream in ("robot", "action", "gripper")
                       else low for stream in streams}
        self._accepting = True
        self._requests: dict = {}
        self._failure: str | None = None
        self._summary: dict | None = None
        self._video_status = "available" if video else "disabled"
        self._video_enabled, self._cv2 = bool(video), None
        self._videos = {s: {"file": None, "frames": 0, "fps": self.video_fps,
                            "mapping": f"MP4 frame k = cameras/{s}/rgb[k]", "status": "empty" if video else "disabled"}
                        for s in self.camera_names.values()}
        self._video_writers: dict = {}
        self._thread = threading.Thread(target=self._run, name="session-recorder", daemon=True)
        self._thread.start()

    def _enqueue(self, stream: str, payload: Any) -> None:
        if not self._accepting:
            return
        slots = self._slots[stream]
        try:
            slots.popleft()
        except IndexError:
            self.dropped.update((stream,))
            return
        try:
            snapshot = _snapshot(payload)
            if self._accepting:
                self._queue.put((stream, snapshot))
                return
        except BaseException:
            slots.append(None)
            raise
        slots.append(None)

    def frame(self, serial: str, frame_id: int, t_ns: int, rgb: np.ndarray, depth: np.ndarray | None,
              depth_scale_m: float | None, meta: dict | None) -> None:
        """Record native RGB and optional aligned z16 depth; never acquire or synthesize a frame."""
        self._enqueue(f"cameras/{self.camera_names[serial]}", (frame_id, t_ns, rgb, depth, depth_scale_m, meta))

    def observation(self, serial: str, frame_id: int, t_ns: int, rgb128: np.ndarray,
                    depth128: np.ndarray | None) -> None:
        """Record precisely the square RGB/float-depth observation supplied to the policy."""
        self._enqueue(f"observations/{self.camera_names[serial]}", (frame_id, t_ns, rgb128, depth128))

    def robot_state(self, t_ns: int, seq: int, q7: ArrayLike, dq7: ArrayLike, tau7: ArrayLike,
                    flange_pose7: ArrayLike, ee_pose7: ArrayLike) -> None:
        """Record every measured robot row, including repeated or non-increasing timestamps/sequences."""
        self._enqueue("robot", (t_ns, seq, q7, dq7, tau7, flange_pose7, ee_pose7))

    def gripper(self, t_ns: int, register: int, status: dict) -> None:
        """Record a measured raw register, separately from commanded closure."""
        self._enqueue("gripper", (t_ns, register, status))

    def step(self, row: dict) -> None:
        """Record a complete policy row with its original step/observation times and status code."""
        self._enqueue("policy", row)

    def gripper_command(self, t_ns: int, register: int, kind: str, reason: str = "") -> None:
        """Record an actual serial write, classifying physical holds separately from targets."""
        if kind not in ("target", "hold"):
            raise ValueError("gripper command kind must be target or hold")
        self._enqueue("gripper_commands", (t_ns, register, kind, reason))

    def executed(self, t_ns: int, kind: str, pose7: ArrayLike, gripper_closure: float | None,
                 sent: bool, detail: str) -> None:
        """Record an executor setpoint and whether it actually reached the controller transport."""
        self._enqueue("action", (t_ns, kind, pose7, gripper_closure, sent, detail))

    def event(self, t_ns: int, text: str) -> None:
        """Record an event with its caller-supplied timestamp."""
        self._enqueue("events", (t_ns, text))

    def close(self, status: str = "complete", error: str | None = None) -> dict:
        """Stop accepting, drain, and return the atomically published summary; repeated calls return the same result.

        No sentinel enqueue or caller-held lock: even a full queue and reentrant signal-handler close can drain.
        The writer uses timed reads, and always closes HDF5 in a finally path. An I/O operation itself must finish
        before close can return; forcibly closing a file from a second thread would corrupt it.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("close must be called outside the writer thread")
        self._accepting = False
        self._requests.setdefault("close", (status, error))
        while self._summary is None:
            time.sleep(0.01)  # no application mutex held; nested close may wait for this same writer
        return self._summary

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self.close("error" if exc is not None else "complete", None if exc is None else f"{exc_type.__name__}: {exc}")
        return False

    def _initialize(self, f: Any, string: Any) -> None:
        from ops.robot_pc.switch_controller import T_FLANGE_EE

        f.attrs.update(format="s4d_robot_pc_inference", format_version=2, created_unix=self._start_unix,
                       created_monotonic=self._start_ns / 1e9, config_json=_json(self.config),
                       camera_mapping_json=_json(self.camera_names), meta_json=_json(self.meta),
                       status="recording", error="")
        f.create_group("cameras")
        f.create_group("observations")
        for serial, config in self.cameras.items():
            name = self.camera_names[serial]
            g = f.create_group(f"cameras/{name}")
            hw = tuple(config["native_hw"])
            g.attrs.update(serial=serial, K_native=np.asarray(config["K_native"], np.float64),
                           calibration_json=_json(config.get("calibration")), depth_scale_m=np.nan)
            for col in ("t_ns", "frame_number"):
                _dataset(g, col, np.int64)
            _dataset(g, "t", np.float64)
            _dataset(g, "rgb", np.uint8, (*hw, 3), image=True, native_rgb=True)
            _dataset(g, "meta_json", string)
            obs = f.create_group(f"observations/{name}")
            obs.attrs["serial"] = serial
            for col in ("t_ns", "frame_number"):
                _dataset(obs, col, np.int64)
            _dataset(obs, "t", np.float64)
            obs.create_dataset("rgb", shape=(0, 0, 0, 3), maxshape=(None, None, None, 3), dtype=np.uint8,
                               chunks=(1, 128, 128, 3), compression="gzip", compression_opts=1)
        schema = {
            "robot": {"t_ns": np.int64, "seq": np.int64,
                      **{k: (np.float64, (7,)) for k in ("q", "dq", "tau_ext", "flange_pose", "ee_pose")}},
            "gripper": {"t_ns": np.int64, **{k: np.int64 for k in (
                "position", "pos_request_echo", "current", "object", "fault", "status", "activated")}},
            "gripper_commands": {"t_ns": np.int64, "register": np.int64, "kind": string, "reason": string},
            "policy": {"t_ns": np.int64, "t_obs_ns": np.int64, "t_obs": np.float64, "status": string,
                       "command": np.uint8, "goal_flange_pose": (np.float64, (7,)), "gripper": np.float64,
                       "latency_json": string, "row_json": string},
            "action": {"t_ns": np.int64, "kind": string, "flange_pose": (np.float64, (7,)),
                       "gripper": np.float64, "sent": np.uint8, "detail": string},
            "events": {"t_ns": np.int64, "text": string},
        }
        for path, fields in schema.items():
            group = f.create_group(path)
            for key, spec in {"t": np.float64, **fields}.items():
                dtype, shape = spec if isinstance(spec, tuple) else (spec, ())
                _dataset(group, key, dtype, shape)
        f["robot"].attrs["T_flange_controller_ee"] = T_FLANGE_EE
        f["robot/t"].attrs["source"] = "host CLOCK_MONOTONIC receipt time; t_ns / 1e9"
        f["policy/status"].attrs["source"] = "row.code when non-null; otherwise row.status (text)"

    def _write_batch(self, f: Any, stream: str, items: list) -> None:
        """Writer-only dispatch: one append per stream, preserving its enqueue order."""
        g = f[stream]
        if stream.startswith("cameras/"):
            fields = {key: [] for key in ("t_ns", "frame_number", "rgb", "meta_json")}
            depth_fields = {key: [] for key in ("depth", "depth_frame_index", "depth_scale_m")}
            rgb_shape, start, first_scale = g["rgb"].shape[1:], len(g["rgb"]), None
            for offset, (frame_number, t_ns, rgb, depth, scale, meta) in enumerate(items):
                rgb = _image(rgb, np.uint8, rgb_shape)
                fields["t_ns"].append(t_ns)
                fields["frame_number"].append((meta or {}).get("color_frame_number", frame_number))
                fields["rgb"].append(rgb)
                fields["meta_json"].append(_json(meta))
                if depth is not None:
                    if "depth" not in g:
                        _dataset(g, "depth", np.uint16, rgb_shape[:2], image=True)
                        _dataset(g, "depth_frame_index", np.int64)
                        _dataset(g, "depth_scale_m", np.float64)
                    depth_fields["depth"].append(_image(depth, np.uint16, rgb.shape[:2]))
                    depth_fields["depth_frame_index"].append(start + offset)
                    depth_fields["depth_scale_m"].append(np.nan if scale is None else scale)
                if scale is not None and (first_scale is None or math.isnan(first_scale)):
                    first_scale = float(scale)
            if depth_fields["depth"]:
                fields.update(depth_fields)
            fields["t"] = np.asarray(fields["t_ns"], np.float64) / 1e9
            _append(g, fields)
            self._counts[stream] += len(items)
            if first_scale is not None and math.isnan(g.attrs["depth_scale_m"]):
                g.attrs["depth_scale_m"] = first_scale
            for rgb in fields["rgb"]:
                self._write_video(stream.split("/", 1)[1], rgb)
            return
        elif stream.startswith("observations/"):
            fields = {key: [] for key in ("t_ns", "frame_number", "rgb")}
            depth_fields = {key: [] for key in ("depth", "depth_frame_index")}
            start = len(g["rgb"])
            rgb_shape = g["rgb"].shape[1:]
            for offset, (frame_number, t_ns, rgb, depth) in enumerate(items):
                if rgb.ndim != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[0] == 0:
                    raise ValueError("observation RGB must be a nonempty square HxWx3 image")
                rgb = _image(rgb, np.uint8, (rgb.shape[0], rgb.shape[0], 3))
                if rgb_shape[0] == 0:
                    rgb_shape = rgb.shape
                _image(rgb, np.uint8, rgb_shape)
                fields["t_ns"].append(t_ns)
                fields["frame_number"].append(frame_number)
                fields["rgb"].append(rgb)
                if depth is not None:
                    depth_fields["depth"].append(_image(depth, np.float32, rgb.shape[:2]))
                    depth_fields["depth_frame_index"].append(start + offset)
            if g["rgb"].shape[1] == 0:
                del g["rgb"]
                _dataset(g, "rgb", np.uint8, rgb_shape, image=True)
            if depth_fields["depth"]:
                if "depth" not in g:
                    _dataset(g, "depth", np.float32, rgb_shape[:2], image=True)
                    _dataset(g, "depth_frame_index", np.int64)
                fields.update(depth_fields)
        elif stream == "robot":
            fields = dict(zip(("t_ns", "seq", "q", "dq", "tau_ext", "flange_pose", "ee_pose"), zip(*items)))
        elif stream == "gripper":
            times, registers, statuses = zip(*items)
            fields = {"t_ns": times, "position": registers}
            # -1 marks a field absent from a partial offline status, never a fabricated healthy value.
            fields.update({key: [s.get(key, -1) for s in statuses] for key in (
                "pos_request_echo", "current", "object", "fault", "status", "activated")})
        elif stream == "gripper_commands":
            fields = dict(zip(("t_ns", "register", "kind", "reason"), zip(*items)))
        elif stream == "policy":
            fields = {"t_ns": [row["t_step_ns"] for row in items],
                      "t_obs_ns": [row["observation_time_ns"] for row in items],
                      "status": [str(row["status"] if row.get("code") is None else row["code"]) for row in items],
                      "command": [bool(row.get("command")) for row in items],
                      "goal_flange_pose": [(row.get("command") or {}).get("goal_pose7", [np.nan] * 7)
                                           for row in items],
                      "gripper": [(row.get("command") or {}).get("gripper_closure", row.get("gripper_closure", np.nan))
                                  for row in items],
                      "latency_json": [_json({k: row[k] for k in ("client_ms", "server_ms", "timings") if k in row})
                                       for row in items],
                      "row_json": [_json(row) for row in items]}
            fields["t_obs"] = np.asarray(fields["t_obs_ns"], np.float64) / 1e9
        elif stream == "action":
            times, kinds, poses, closures, sent, details = zip(*items)
            fields = {"t_ns": times, "kind": kinds, "flange_pose": poses,
                      "gripper": [np.nan if c is None else c for c in closures], "sent": sent, "detail": details}
        elif stream == "events":
            times, texts = zip(*items)
            fields = {"t_ns": times, "text": texts}
        fields["t"] = np.asarray(fields["t_ns"], np.float64) / 1e9
        _append(g, fields)
        self._counts[stream] += len(items)

    def _write_video(self, serial: str, rgb: np.ndarray) -> None:
        info = self._videos[serial]
        if not self._video_enabled or info["status"].startswith("unavailable:"):
            return
        try:
            if self._cv2 is None:
                try:
                    import cv2
                except Exception as exc:
                    self._video_enabled = False
                    self._video_status = f"unavailable: {type(exc).__name__}: {exc}"
                    for entry in self._videos.values():
                        entry["status"] = self._video_status
                    return
                self._cv2 = cv2
            if serial not in self._video_writers:
                path = self.session_dir / f"{serial}.mp4"
                writer = self._cv2.VideoWriter(str(path), self._cv2.VideoWriter_fourcc(*"mp4v"), self.video_fps,
                                               (rgb.shape[1], rgb.shape[0]))
                self._video_writers[serial] = writer
                if not writer.isOpened():
                    raise RuntimeError("cv2.VideoWriter could not open mp4v")
                info.update(file=str(path), status="available")
            self._video_writers[serial].write(np.ascontiguousarray(rgb[..., ::-1]))
            info["frames"] += 1
        except Exception as exc:
            info["status"] = f"unavailable: {type(exc).__name__}: {exc}"
            self._video_status = info["status"]

    def _fail(self, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self._failure = f"{self._failure}; {message}" if self._failure else message
        self._accepting = False

    def _result(self) -> tuple[str, str | None]:
        status, error = self._requests.get("close", ("complete", None))
        if self._failure:
            return "error", f"{error}; {self._failure}" if error else self._failure
        broken = [s for s, v in self._videos.items() if str(v.get("status", "")).startswith("unavailable")
                  and self._counts.get(f"cameras/{s}", 0) > 0]
        if self._video_enabled and broken and status == "complete":  # frames recorded but an MP4 is missing
            return "error", f"MP4 unavailable for {broken}: {self._videos[broken[0]]['status']}"
        return status, error

    def _run(self) -> None:
        f = None
        pending: dict[str, list] = {}
        try:
            import h5py

            f = h5py.File(self._partial, "w")
            self._initialize(f, h5py.string_dtype("utf-8"))
            last_flush = time.monotonic()
            while True:
                try:
                    stream, item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if "close" in self._requests:
                        break
                else:
                    self._slots[stream].append(None)
                    pending[stream] = [item]
                    # Bound both work and image memory. Waiting here coalesces sparse arrivals without
                    # slowing producers; close drains the available queue immediately instead.
                    deadline = time.monotonic() + 0.01
                    size, images = 1, int("/" in stream)
                    while size < 4096 and images < 16:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            if "close" in self._requests:
                                stream, item = self._queue.get_nowait()
                            else:
                                stream, item = self._queue.get(timeout=remaining)
                        except queue.Empty:
                            break
                        self._slots[stream].append(None)
                        pending.setdefault(stream, []).append(item)
                        size += 1
                        images += int("/" in stream)
                    for stream in list(pending):
                        items = pending.pop(stream)
                        before = self._counts[stream]
                        try:
                            self._write_batch(f, stream, items)
                        except BaseException:
                            self.dropped.update({stream: len(items) - (self._counts[stream] - before)})
                            raise
                if time.monotonic() - last_flush >= 1.0:
                    f.flush()
                    last_flush = time.monotonic()
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._accepting = False
            for stream, items in pending.items():
                self.dropped.update({stream: len(items)})
            pending.clear()
            while not self._queue.empty():
                stream, _ = self._queue.get_nowait()
                self._slots[stream].append(None)
                self.dropped.update((stream,))
            for writer in self._video_writers.values():
                try:
                    writer.release()
                except BaseException as exc:
                    self._fail(exc)
            if f is not None:
                try:
                    status, error = self._result()
                    f.attrs.update(status=status, error=error or "")
                    f.flush()
                except BaseException as exc:
                    self._fail(exc)
                finally:
                    try:
                        f.close()
                    except BaseException as exc:
                        self._fail(exc)
            try:
                if self._partial.exists():
                    self._partial.replace(self._h5_path)
            except BaseException as exc:
                self._fail(exc)
            self._publish_summary()

    def _publish_summary(self) -> None:
        end_ns, size = None, 0
        try:
            end_ns = int(self.clock())
            size = self._h5_path.stat().st_size if self._h5_path.exists() else 0
        except BaseException as exc:
            self._fail(exc)
        status, error = self._result()
        summary = {"session_dir": str(self.session_dir), "status": status, "error": error,
                   "start_unix": self._start_unix, "end_unix": time.time(),
                   "start_monotonic_ns": self._start_ns, "end_monotonic_ns": end_ns,
                   "counts": self._counts.copy(), "dropped": dict(self.dropped),
                   "video": self._video_status, "videos": self._videos, "meta": self.meta,
                   "h5_size_bytes": size}
        try:
            temporary = self.session_dir / "summary.json.tmp"
            temporary.write_text(_json(summary) + "\n", encoding="utf-8")
            temporary.replace(self.session_dir / "summary.json")
        except BaseException as exc:
            self._fail(exc)
            summary["status"], summary["error"] = self._result()
        finally:
            self._summary = summary
