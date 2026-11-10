"""Asynchronous, local-only session recording. Only the writer imports h5py and (for video) cv2.

Enqueue methods snapshot their inputs and never wait for disk or queue space. ``close`` is the intentional
blocking boundary: stop producers first, then drain and finalize. It takes no producer/queue locks, so a caller's
SIGTERM handler may call it, including while another close is waiting. Install signal/atexit handlers in the caller.

Missing depth is not padded: ``depth_frame_index`` maps each depth row to its RGB row. Native depth scales are
stored per depth row (NaN means unspecified); the attribute is the first supplied scale. Empty observation images
have unknown spatial dimensions until the first observation. Existing session directories are never overwritten.
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


def _dataset(group: Any, name: str, dtype: Any, shape: tuple = (), *, image: bool = False) -> Any:
    options = {"compression": "gzip", "compression_opts": 1} if image else {}
    return group.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype,
                                chunks=(1, *shape) if image else True, **options)


def _append(group: Any, values: dict) -> None:
    """Append measured fields together, rolling back incomplete rows on a write failure."""
    sizes = {key: len(group[key]) for key in values}
    try:
        for key, value in values.items():
            ds = group[key]
            if np.shape(value) != ds.shape[1:]:
                raise ValueError(f"{ds.name}: expected row shape {ds.shape[1:]}, got {np.shape(value)}")
            ds.resize(sizes[key] + 1, axis=0)
            ds[sizes[key]] = value
    except BaseException:
        for key, size in sizes.items():
            group[key].resize(size, axis=0)
        raise


def _image(value: Any, dtype: Any, shape: tuple) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"expected {np.dtype(dtype)} image {shape}, got {array.dtype} {array.shape}")
    return array


class SessionRecorder:
    """One session under ``root/inference_outputs/inference_YYYYMMDD_HHMMSS``.

    ``dropped`` uses HDF5 group paths, e.g. ``cameras/123``, ``observations/123``, ``robot``. Queue overflow
    drops the incoming item. A writer error also counts queued items it cannot write, then finalizes automatically.
    Timestamp columns retain input nanoseconds; commands without ``t_ns`` use the recorder's enqueue clock,
    explicitly labelled in the HDF5. All file access, compression and video conversion belongs to one thread.
    """

    def __init__(self, root: str | Path, *, meta: dict, cameras: dict[str, dict],
                 clock: Callable[[], int] = time.monotonic_ns, queue_size: int = 2048,
                 video: bool = True, video_fps: float = 15.0):
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
        name = time.strftime("inference_%Y%m%d_%H%M%S", time.localtime(self._start_unix))
        parent = Path(root).expanduser().resolve() / "inference_outputs"
        parent.mkdir(parents=True, exist_ok=True)
        self.session_dir = parent / name
        self.session_dir.mkdir()  # exclusive: a same-second collision must not clobber a prior run
        self._partial = self.session_dir / "inference.h5.partial"
        self._h5_path = self.session_dir / "inference.h5"
        streams = ["robot", "gripper", "steps", "commands", "executed", "events"]
        streams += [f"{kind}/{serial}" for kind in ("cameras", "observations") for serial in self.cameras]
        self.dropped: Counter = Counter(dict.fromkeys(streams, 0))
        self._counts = dict.fromkeys(streams, 0)
        # SimpleQueue's C put is reentrant, unlike Queue's mutex. Atomic deque reservations bound its capacity
        # without a producer lock that a signal-handler close could leave held while waiting for the writer.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._slots = deque([None] * queue_size)
        self._accepting = True
        self._requests: dict = {}
        self._failure: str | None = None
        self._summary: dict | None = None
        self._video_status = "available" if video else "disabled"
        self._video_enabled, self._cv2 = bool(video), None
        self._videos = {s: {"file": None, "frames": 0, "fps": self.video_fps,
                            "mapping": f"MP4 frame k = cameras/{s}/rgb[k]", "status": "empty" if video else "disabled"}
                        for s in self.cameras}
        self._video_writers: dict = {}
        self._thread = threading.Thread(target=self._run, name="session-recorder", daemon=True)
        self._thread.start()

    def _enqueue(self, stream: str, payload: Any) -> None:
        if not self._accepting:
            return
        try:
            self._slots.popleft()
        except IndexError:
            self.dropped.update((stream,))
            return
        try:
            snapshot = copy.deepcopy(payload)
            if self._accepting:
                self._queue.put((stream, snapshot))
                return
        except BaseException:
            self._slots.append(None)
            raise
        self._slots.append(None)

    def frame(self, serial: str, frame_id: int, t_ns: int, rgb: np.ndarray, depth: np.ndarray | None,
              depth_scale_m: float | None, meta: dict | None) -> None:
        """Record native RGB and optional aligned z16 depth; never acquire or synthesize a frame."""
        self._enqueue(f"cameras/{serial}", (frame_id, t_ns, rgb, depth, depth_scale_m, meta))

    def observation(self, serial: str, frame_id: int, t_ns: int, rgb128: np.ndarray,
                    depth128: np.ndarray | None) -> None:
        """Record precisely the square RGB/float-depth observation supplied to the policy."""
        self._enqueue(f"observations/{serial}", (frame_id, t_ns, rgb128, depth128))

    def robot_state(self, t_ns: int, seq: int, q7: ArrayLike, dq7: ArrayLike, tau7: ArrayLike,
                    flange_pose7: ArrayLike, ee_pose7: ArrayLike) -> None:
        """Record every measured robot row, including repeated or non-increasing timestamps/sequences."""
        self._enqueue("robot", (t_ns, seq, q7, dq7, tau7, flange_pose7, ee_pose7))

    def gripper(self, t_ns: int, register: int, status: dict) -> None:
        """Record a measured raw register, separately from commanded closure."""
        self._enqueue("gripper", (t_ns, register, status))

    def step(self, row: dict) -> None:
        """Record a complete policy row with its original step/observation times and status code."""
        self._enqueue("steps", row)

    def command(self, cmd: dict) -> None:
        """Record a validated command; absent command time means enqueue time, never observation time."""
        self._enqueue("commands", (cmd["t_ns"] if "t_ns" in cmd else int(self.clock()), cmd))

    def executed(self, t_ns: int, kind: str, pose7: ArrayLike, gripper_closure: float | None,
                 sent: bool, detail: str) -> None:
        """Record an executor setpoint and whether it actually reached the controller transport."""
        self._enqueue("executed", (t_ns, kind, pose7, gripper_closure, sent, detail))

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
        f.attrs.update(format="s4d_robot_pc_inference_session", format_version=1, created_unix=self._start_unix,
                       meta_json=_json(self.meta), status="recording", error="")
        f.create_group("cameras")
        f.create_group("observations")
        for serial, config in self.cameras.items():
            g = f.create_group(f"cameras/{serial}")
            hw = tuple(config["native_hw"])
            g.attrs.update(K_native=np.asarray(config["K_native"], np.float64),
                           calibration_json=_json(config.get("calibration")), depth_scale_m=np.nan)
            for name in ("t_ns", "frame_id"):
                _dataset(g, name, np.int64)
            _dataset(g, "rgb", np.uint8, (*hw, 3), image=True)
            _dataset(g, "meta_json", string)
            if config["depth"]:
                _dataset(g, "depth", np.uint16, hw, image=True)
                _dataset(g, "depth_frame_index", np.int64)
                _dataset(g, "depth_scale_m", np.float64)
            obs = f.create_group(f"observations/{serial}")
            for name in ("t_ns", "frame_id"):
                _dataset(obs, name, np.int64)
            obs.create_dataset("rgb", shape=(0, 0, 0, 3), maxshape=(None, None, None, 3), dtype=np.uint8,
                               chunks=(1, 1, 1, 3), compression="gzip", compression_opts=1)
        schema = {
            "robot": {"t_ns": np.int64, "seq": np.int64,
                      **{k: (np.float64, (7,)) for k in ("q", "dq", "tau_ext", "flange_pose", "ee_pose")}},
            "gripper": {"t_ns": np.int64, "position": np.int64, "status_json": string},
            "steps": {"t_step_ns": np.int64, "observation_time_ns": np.int64, "status": string, "json": string},
            "commands": {"t_ns": np.int64, "json": string},
            "executed": {"t_ns": np.int64, "kind": string, "pose7": (np.float64, (7,)),
                         "gripper_closure": np.float64, "sent": np.bool_, "detail": string},
            "events": {"t_ns": np.int64, "text": string},
        }
        for path, fields in schema.items():
            group = f.create_group(path)
            for key, spec in fields.items():
                dtype, shape = spec if isinstance(spec, tuple) else (spec, ())
                _dataset(group, key, dtype, shape)
        f["commands/t_ns"].attrs["source"] = "cmd.t_ns when supplied; otherwise recorder enqueue clock"
        f["steps/status"].attrs["source"] = "row.code when non-null; otherwise row.status (text)"

    def _write_item(self, f: Any, stream: str, item: Any) -> None:
        """Writer-only dispatch; kept separate for fault injection and slow-writer tests."""
        g = f[stream]
        if stream.startswith("cameras/"):
            frame_id, t_ns, rgb, depth, scale, meta = item
            rgb = _image(rgb, np.uint8, g["rgb"].shape[1:])
            fields = {"t_ns": t_ns, "frame_id": frame_id, "rgb": rgb, "meta_json": _json(meta)}
            if depth is not None:
                if "depth" not in g:
                    raise ValueError(f"depth supplied for RGB-only {stream}")
                fields.update(depth=_image(depth, np.uint16, rgb.shape[:2]), depth_frame_index=len(g["rgb"]),
                              depth_scale_m=np.nan if scale is None else scale)
            if scale is not None and math.isnan(g.attrs["depth_scale_m"]):
                g.attrs["depth_scale_m"] = float(scale)
            _append(g, fields)
            self._write_video(stream.split("/", 1)[1], rgb)
        elif stream.startswith("observations/"):
            frame_id, t_ns, rgb, depth = item
            if rgb.ndim != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[0] == 0:
                raise ValueError("observation RGB must be a nonempty square HxWx3 image")
            rgb = _image(rgb, np.uint8, (rgb.shape[0], rgb.shape[0], 3))
            if g["rgb"].shape[1] == 0:
                del g["rgb"]
                _dataset(g, "rgb", np.uint8, rgb.shape, image=True)
            _image(rgb, np.uint8, g["rgb"].shape[1:])
            fields = {"t_ns": t_ns, "frame_id": frame_id, "rgb": rgb}
            if depth is not None:
                depth = _image(depth, np.float32, rgb.shape[:2])
                if "depth" not in g:
                    _dataset(g, "depth", np.float32, depth.shape, image=True)
                    _dataset(g, "depth_frame_index", np.int64)
                fields.update(depth=depth, depth_frame_index=len(g["rgb"]))
            _append(g, fields)
        elif stream == "robot":
            _append(g, dict(zip(("t_ns", "seq", "q", "dq", "tau_ext", "flange_pose", "ee_pose"), item)))
        elif stream == "gripper":
            t_ns, register, status = item
            _append(g, {"t_ns": t_ns, "position": register, "status_json": _json(status)})
        elif stream == "steps":
            code = item.get("code")
            _append(g, {"t_step_ns": item["t_step_ns"], "observation_time_ns": item["observation_time_ns"],
                        "status": str(item["status"] if code is None else code), "json": _json(item)})
        elif stream == "commands":
            _append(g, {"t_ns": item[0], "json": _json(item[1])})
        elif stream == "executed":
            t_ns, kind, pose, closure, sent, detail = item
            _append(g, {"t_ns": t_ns, "kind": kind, "pose7": pose,
                        "gripper_closure": np.nan if closure is None else closure, "sent": sent, "detail": detail})
        elif stream == "events":
            _append(g, {"t_ns": item[0], "text": item[1]})
        self._counts[stream] += 1

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
                path = self.session_dir / f"camera_{serial}.mp4"
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
        return status, error

    def _run(self) -> None:
        f = None
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
                    self._slots.append(None)
                    try:
                        self._write_item(f, stream, item)
                    except BaseException:
                        self.dropped.update((stream,))
                        raise
                if time.monotonic() - last_flush >= 1.0:
                    f.flush()
                    last_flush = time.monotonic()
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._accepting = False
            while not self._queue.empty():
                stream, _ = self._queue.get_nowait()
                self._slots.append(None)
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
