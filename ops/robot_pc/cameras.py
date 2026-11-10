"""Camera acquisition of the robot PC (CPU only).

* ``LiveCameras``: one RealSense pipeline per selected serial (``ops.policy_serving.client.RealSenseCapture``: colour
  rgb8 at the trained 640x480 / 15 fps, + native z16 depth ``rs.align``-ed to colour with the validated native scale for
  3D-MVP, the exact training crop + 128 px resize on this CPU, the recorder's capture time ``DeviceClock``), each read
  by its own thread into a bounded queue. The device-busy guard runs first (``devices.require_free``).
* ``RecordedCameras``: the recorded colour JPEG decoded as the training adapter does and the recorded native depth
  aligned like the live path (``s4d.rwlearn.deploy.replay.RecordedEpisode``, default ``rs_align`` emulation).

Frame ids are the cameras' own acquisition counters (gaps stay gaps); history spacing is never re-timed here: the
deploy history refuses irregular histories.
"""

from __future__ import annotations

import collections
import threading
import time

import numpy as np

from ops.robot_pc import devices


class RecordedCameras:
    def __init__(self, recording, cameras, *, depth: bool, alignment: str = "rs_align"):
        self.rec, self.cameras, self.depth, self.alignment = recording, list(cameras), bool(depth), alignment

    def timestamps_ns(self, cam: str) -> np.ndarray:
        return self.rec.timestamps_ns(cam)

    def native_hw(self, cam: str) -> tuple:
        return self.rec.native_hw(cam)

    def K_native(self, cam: str) -> np.ndarray:
        return self.rec.K_native(cam)

    def native(self, cam: str, index: int) -> dict:
        """Recorded frame ``index`` before the client preprocessing (+ ``acquire_ms``: replay-only JPEG decode and
        rs.align emulation, NOT a live cost)."""
        t0 = time.monotonic_ns()
        out = self.rec.native(cam, index, depth=self.depth, alignment=self.alignment)
        out["acquire_ms"] = (time.monotonic_ns() - t0) / 1e6
        return out


class LiveCameras:
    """Live RealSense acquisition of the session cameras; ``start`` -> ``intrinsics`` -> ``bind`` (session K / native
    correction) -> reader threads -> ``drain``. Never touches a device when the busy guard refuses."""

    def __init__(self, serials, *, native_hw=(480, 640), fps: int = 15, depth: bool = False, rs_module=None,
                 check_devices: bool = True, queue: int = 8, keep_native: bool = False, allow_controller: bool = False):
        if check_devices:
            self.device_scan = devices.require_free(allow_controller=allow_controller)
        if rs_module is None:
            import pyrealsense2 as rs_module
        self.rs, self.serials, self.native_hw = rs_module, [str(s) for s in serials], tuple(native_hw)
        self.fps, self.depth, self._queue_len, self.keep_native = int(fps), bool(depth), int(queue), bool(keep_native)
        self.captures, self._queues, self._threads, self.errors = {}, {}, [], collections.Counter()
        self._stop = threading.Event()
        self._started = False

    def start(self, corrections: dict | None = None) -> dict:
        """Start every pipeline (colour [+ depth]); returns {serial: K_native (continuous colour intrinsics)}."""
        from ops.policy_serving.client import RealSenseCapture
        from s4d.rwlearn.deploy.realsense import color_intrinsics

        Ks = {}
        try:
            for s in self.serials:
                cap = RealSenseCapture(s, native_hw=self.native_hw, fps=self.fps, depth=self.depth,
                                       correction=(corrections or {}).get(s), rs_module=self.rs,
                                       keep_native=self.keep_native)
                self.captures[s] = cap
                vp = cap.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
                Ks[s] = color_intrinsics(vp.get_intrinsics())
        except Exception:
            self.close()
            raise
        return Ks

    def bind(self, K_native: dict) -> None:
        """Lock each stream to the session intrinsics (checked on every frame by RealSenseCapture) and start reading."""
        for s, cap in self.captures.items():
            cap.K_native = np.asarray(K_native[s], np.float64)
            self._queues[s] = collections.deque(maxlen=self._queue_len)
            t = threading.Thread(target=self._reader, args=(s, cap), name=f"rs-{s}", daemon=True)
            self._threads.append(t)
            t.start()
        self._started = True

    def _reader(self, serial: str, cap) -> None:
        while not self._stop.is_set():
            try:
                out = cap.read(timeout_ms=1000)
            except Exception as exc:  # noqa: BLE001 - counted and surfaced by drain(); the loop decides
                self.errors[f"{serial}: {type(exc).__name__}: {exc}"[:200]] += 1
                time.sleep(0.01)
                continue
            out["arrival_ns"] = time.monotonic_ns()
            self._queues[serial].append(out)

    def drain(self, serial: str) -> list:
        q, out = self._queues[serial], []
        while q:
            out.append(q.popleft())
        return out

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for cap in self.captures.values():
            try:
                cap.stop()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
        self.captures.clear()
