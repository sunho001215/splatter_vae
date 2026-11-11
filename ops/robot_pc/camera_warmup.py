"""Colour-stream warmup on preprocessed RGB frames. numpy only; no device or command access.

Capture timestamps determine stability; the waiter's clock only bounds waiting and camera silence.
Warmup frames are drained and discarded before the policy builds its consecutive-frame history.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class WarmupConfig:
    window_s: float = 1.0
    min_s: float = 1.5
    rb_tolerance: float = 0.02
    luma_tolerance: float = 0.03
    timeout_s: float = 10.0
    on_timeout: str = "proceed"
    enabled: bool = True

    def __post_init__(self):
        for key in ("window_s", "min_s", "rb_tolerance", "luma_tolerance", "timeout_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"camera_warmup.{key} must be finite and nonnegative")
            object.__setattr__(self, key, float(value))
        if self.window_s == 0:
            raise ValueError("camera_warmup.window_s must be positive")
        if self.on_timeout not in ("proceed", "refuse"):
            raise ValueError("camera_warmup.on_timeout must be proceed or refuse")
        if not isinstance(self.enabled, bool):
            raise ValueError("camera_warmup.enabled must be a boolean")

    @classmethod
    def from_config(cls, d: dict | None = None) -> WarmupConfig:
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise ValueError("camera_warmup must be a mapping")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown camera_warmup keys {sorted(unknown)}")
        return cls(**d)


@dataclass
class _CameraWindow:
    frames_seen: int = 0
    first_ns: int | None = None
    last_ns: int | None = None
    first_rb: float | None = None
    first_luma: float | None = None
    window: deque = field(default_factory=deque)
    settle_s: float | None = None


def _relative_range(values) -> float | None:
    if not values or any(v is None for v in values):
        return None
    mean = float(np.mean(values))
    return float((max(values) - min(values)) / mean) if mean > 0 else None


class ColourStabilityGate:
    def __init__(self, serials, cfg: WarmupConfig, *, fps: float):
        serials = list(serials)
        if not serials or len(set(serials)) != len(serials):
            raise ValueError("warmup needs nonempty, unique camera serials")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("warmup fps must be finite and positive")
        self.cfg = cfg
        self.required_frames = int(np.ceil(0.8 * fps * cfg.window_s))
        self._cameras = {s: _CameraWindow() for s in serials}

    def feed(self, serial: str, capture_ns: int, rgb: np.ndarray) -> None:
        c = self._cameras[serial]
        if not isinstance(capture_ns, (int, np.integer)) or capture_ns < 0:
            raise ValueError("warmup capture_ns must be a nonnegative integer")
        capture_ns = int(capture_ns)
        if c.last_ns is not None and capture_ns <= c.last_ns:
            raise ValueError("warmup capture timestamps must increase")
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
            raise ValueError("warmup expects uint8 HxWx3 RGB")
        r, g, b = rgb.mean(axis=(0, 1), dtype=np.float64)
        rb = float(r / b) if b > 0 else None
        luma = float(0.299 * r + 0.587 * g + 0.114 * b)
        if c.first_ns is None:
            c.first_ns, c.first_rb, c.first_luma = capture_ns, rb, luma
        c.last_ns = capture_ns
        c.frames_seen += 1
        c.window.append((capture_ns, rb, luma))
        cutoff = capture_ns - self.cfg.window_s * 1e9
        while c.window[0][0] < cutoff:
            c.window.popleft()
        if c.settle_s is None and self._result(c)["stable"]:
            c.settle_s = (capture_ns - c.first_ns) / 1e9

    def _result(self, c: _CameraWindow) -> dict:
        rb_range = _relative_range([f[1] for f in c.window])
        luma_range = _relative_range([f[2] for f in c.window])
        elapsed = 0.0 if c.first_ns is None else (c.last_ns - c.first_ns) / 1e9
        stable = elapsed >= self.cfg.min_s and len(c.window) >= self.required_frames and \
            rb_range is not None and rb_range <= self.cfg.rb_tolerance and \
            luma_range is not None and luma_range <= self.cfg.luma_tolerance
        return {"frames_seen": c.frames_seen, "elapsed_s": elapsed,
                "first_rb": c.first_rb, "last_rb": c.window[-1][1] if c.window else None,
                "first_luma": c.first_luma, "last_luma": c.window[-1][2] if c.window else None,
                "window_frames": len(c.window), "rb_range": rb_range, "luma_range": luma_range,
                "stable": stable, "settle_s": c.settle_s}

    @property
    def stable(self) -> bool:
        return all(r["stable"] for r in self.result().values())

    def result(self) -> dict:
        """JSON-able per-camera statistics; settle_s is capture time elapsed at the first stable window."""
        return {s: self._result(c) for s, c in self._cameras.items()}


def wait_until_stable(cameras, serials, cfg: WarmupConfig, *, cancel, clock=time.monotonic_ns,
                      sleep=time.sleep, fps: float) -> dict:
    gate = ColourStabilityGate(serials, cfg, fps=fps)
    start = clock()

    def finish(status, **detail):
        return {"status": status, "waited_s": (clock() - start) / 1e9, "cameras": gate.result(), **detail}

    if cancel.is_set():
        return finish("cancelled")
    if not cfg.enabled or cfg.timeout_s == 0:
        return finish("disabled")
    errors = dict(cameras.errors)
    last_frame = dict.fromkeys(gate.result(), start)
    while True:
        if cancel.is_set():
            return finish("cancelled")
        for s in last_frame:
            frames = cameras.drain(s)
            if frames:
                last_frame[s] = clock()
            for frame in frames:
                gate.feed(s, frame["capture_ns"], frame["rgb"])
        # A reader error takes precedence over stability or an ordinary timeout.
        current_errors = dict(cameras.errors)
        if any(n > errors.get(k, 0) for k, n in current_errors.items()):
            return finish("no_frames", reason="camera read error", errors=current_errors)
        if cancel.is_set():
            return finish("cancelled")
        now = clock()
        silent = [s for s, last in last_frame.items() if now - last >= 2_000_000_000]
        if silent:
            return finish("no_frames", reason="no frame for 2 s", serials=silent)
        if gate.stable:
            return finish("stable")
        if now - start >= cfg.timeout_s * 1e9:
            return finish("timeout")
        sleep(min(0.01, max(0.0, cfg.timeout_s - (now - start) / 1e9)))
