"""Colour warmup with deterministic CPU frames and clocks; no hardware, sockets or real sleeps."""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from ops.robot_pc import config as C
from ops.robot_pc.camera_warmup import ColourStabilityGate, WarmupConfig, wait_until_stable

FPS = 15


def rgb(rb=1.15, level=100.0):
    # Fractional channel means avoid uint8 quantization hiding the measured small variations.
    out = np.empty((128, 128, 3), np.uint8)
    for channel, value in enumerate((rb * level, level, level)):
        whole = int(value)
        out[..., channel] = whole
        out.reshape(-1, 3)[:round((value - whole) * 128**2), channel] = whole + 1
    return out


class FakeClock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += round(seconds * 1e9)


class FakeCameras:
    def __init__(self, clock, streams):
        self.clock, self.streams, self.errors = clock, streams, Counter()
        self.next = dict.fromkeys(streams, 0)

    def drain(self, serial):
        out = []
        stream = self.streams[serial]
        while self.next[serial] < len(stream):
            t, image = stream[self.next[serial]]
            if t > self.clock():
                break
            self.next[serial] += 1
            out.append({"capture_ns": t, "rgb": image})
        return out


def stream(seconds, image=lambda t: rgb()):
    return [(round(i * 1e9 / FPS), image(i / FPS)) for i in range(round(seconds * FPS) + 1)]


def test_stable_stream_waits_for_minimum_and_reports_json_statistics():
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    for t, frame in stream(2):
        gate.feed("a", t, frame)
        assert gate.stable == (t >= 1.5e9)
    r = gate.result()["a"]
    assert r["frames_seen"] == 31 and r["elapsed_s"] == 2
    assert r["first_rb"] == pytest.approx(1.15) and r["last_rb"] == r["first_rb"]
    assert r["first_luma"] == pytest.approx(104.485) and r["last_luma"] == r["first_luma"]
    assert r["rb_range"] == r["luma_range"] == 0
    assert 1.5 <= r["settle_s"] <= 1.5 + 1 / FPS
    json.dumps(gate.result(), allow_nan=False)


def test_measured_cyan_ramp_never_passes_during_transient_then_settles():
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    for t, frame in stream(4.1, lambda s: rgb(0.55 + 0.6 * min(s / 3, 1))):
        gate.feed("a", t, frame)
        if t <= 3e9:
            assert not gate.stable
        if gate.stable:
            break
    r = gate.result()["a"]
    assert gate.stable and 3 < r["settle_s"] <= 4.1
    assert r["first_rb"] == pytest.approx(0.55) and r["last_rb"] == pytest.approx(1.15)


def test_measured_steady_p95_variation_passes():
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    for t, frame in stream(2, lambda s: rgb(1.15 * (1 + 0.006 * np.sin(2 * np.pi * s)),
                                          100 * (1 + 0.0103 * np.cos(2 * np.pi * s)))):
        gate.feed("a", t, frame)
    r = gate.result()["a"]
    assert 0.011 < r["rb_range"] < 0.013
    assert 0.020 < r["luma_range"] < 0.022
    assert gate.stable


def test_luma_drift_alone_does_not_pass():
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    for t, frame in stream(3, lambda s: rgb(level=60 + 15 * s)):
        gate.feed("a", t, frame)
        assert not gate.stable
    r = gate.result()["a"]
    assert r["rb_range"] < 0.001 and r["luma_range"] > 0.03


def test_multi_camera_requires_all_current_windows_to_be_stable():
    gate = ColourStabilityGate(["a", "b"], WarmupConfig(), fps=FPS)
    for t, frame in stream(2):
        gate.feed("a", t, frame)
    assert gate.result()["a"]["stable"] and not gate.stable
    for t, frame in stream(3, lambda s: rgb(0.55 + 0.2 * s)):
        gate.feed("b", t, frame)
        assert not gate.stable
    for i in range(46, 62):
        gate.feed("b", round(i * 1e9 / FPS), rgb())
    assert gate.stable
    gate.feed("a", 3_000_000_000, rgb(0.6))
    assert not gate.stable  # stability is not latched once a camera has passed


def test_window_coverage_and_expiry_use_capture_times():
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    for i in range(12):
        gate.feed("a", i * 100_000_000, rgb())
    gate.feed("a", 1_500_000_000, rgb())
    assert not gate.stable and gate.result()["a"]["window_frames"] == 8
    for i in range(1, 6):
        gate.feed("a", 1_500_000_000 + i * 1_000_000, rgb())
    assert gate.stable  # ceil(0.8 * 15 * 1) = 12 frames


@pytest.mark.parametrize("frame", [np.zeros((4, 4, 3), np.uint8), rgb(rb=0)])
def test_dark_frames_do_not_pass_or_emit_nonfinite_json(frame):
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    for t, _ in stream(2):
        gate.feed("a", t, frame)
    assert not gate.stable
    json.dumps(gate.result(), allow_nan=False)


@pytest.mark.parametrize("options", [{"surprise": 1}, {"window_s": 0}, {"window_s": -1}, {"min_s": -1},
                                     {"rb_tolerance": -1}, {"luma_tolerance": -1}, {"timeout_s": -1},
                                     {"timeout_s": float("nan")}, {"min_s": float("inf")}, {"min_s": True},
                                     {"enabled": "false"}, {"on_timeout": "ignore"}, {"window_s": "1"}, []])
def test_invalid_config_is_refused(options):
    with pytest.raises(ValueError, match="camera_warmup"):
        WarmupConfig.from_config(options)


def test_loader_accepts_block_and_defaults_for_old_installed_configs(tmp_path):
    p = tmp_path / "robot_pc.yaml"
    p.write_text("root: .\n")
    assert C.load(p)["camera_warmup"] == asdict(WarmupConfig())
    p.write_text(yaml.safe_dump({"camera_warmup": {"timeout_s": 4, "on_timeout": "refuse"}}))
    assert C.load(p)["camera_warmup"] == asdict(WarmupConfig(timeout_s=4, on_timeout="refuse"))
    p.write_text("camera_warmup:\n  typo: 1\n")
    with pytest.raises(ValueError, match="unknown camera_warmup"):
        C.load(p)


def test_waiter_discards_frames_and_finishes_after_minimum():
    clock = FakeClock()
    cameras = FakeCameras(clock, {"a": stream(3)})
    r = wait_until_stable(cameras, ["a"], WarmupConfig(), cancel=threading.Event(),
                          clock=clock, sleep=clock.sleep, fps=FPS)
    assert r["status"] == "stable" and 1.5 <= r["waited_s"] < 1.6
    assert r["cameras"]["a"]["frames_seen"] == cameras.next["a"]
    assert cameras.drain("a") == []
    clock.sleep(1 / FPS)
    assert cameras.drain("a")[0]["capture_ns"] > r["cameras"]["a"]["elapsed_s"] * 1e9


@pytest.mark.parametrize("on_timeout", ["proceed", "refuse"])
def test_timeout_is_reported_for_either_caller_policy(on_timeout):
    clock = FakeClock()
    cameras = FakeCameras(clock, {"a": stream(2, lambda s: rgb(0.55 + 0.2 * s))})
    r = wait_until_stable(cameras, ["a"], WarmupConfig(timeout_s=1, on_timeout=on_timeout),
                          cancel=threading.Event(), clock=clock, sleep=clock.sleep, fps=FPS)
    assert r["status"] == "timeout" and r["waited_s"] == 1


@pytest.mark.parametrize("streams", [{"a": []}, {"a": stream(0.4)}, {"a": stream(4), "b": []}])
def test_camera_silence_is_a_refusal_even_if_another_camera_streams(streams):
    clock = FakeClock()
    cameras = FakeCameras(clock, streams)
    r = wait_until_stable(cameras, list(streams), WarmupConfig(min_s=3), cancel=threading.Event(),
                          clock=clock, sleep=clock.sleep, fps=FPS)
    assert r["status"] == "no_frames" and 2 <= r["waited_s"] <= 2.4


def test_reader_errors_take_precedence_over_stability_and_timeout():
    clock = FakeClock()
    cameras = FakeCameras(clock, {"a": stream(2)})
    cameras.errors["old"] = 1  # unchanged errors from before the wait are not new failures

    def sleep(seconds):
        clock.sleep(seconds)
        if clock() >= 1_540_000_000:
            cameras.errors["new"] += 1

    r = wait_until_stable(cameras, ["a"], WarmupConfig(timeout_s=1.54), cancel=threading.Event(),
                          clock=clock, sleep=sleep, fps=FPS)
    assert r["status"] == "no_frames" and r["errors"]["new"] == 1


@pytest.mark.parametrize("already_cancelled", [False, True])
def test_cancellation(already_cancelled):
    clock, cancel = FakeClock(), threading.Event()
    cameras = FakeCameras(clock, {"a": []})
    if already_cancelled:
        cancel.set()

    def sleep(seconds):
        clock.sleep(seconds)
        cancel.set()

    r = wait_until_stable(cameras, ["a"], WarmupConfig(), cancel=cancel, clock=clock, sleep=sleep, fps=FPS)
    assert r["status"] == "cancelled" and r["waited_s"] <= 0.01


@pytest.mark.parametrize("cfg", [WarmupConfig(enabled=False), WarmupConfig(timeout_s=0)])
def test_disabled_gate_never_reads_cameras(cfg):
    r = wait_until_stable(None, ["a"], cfg, cancel=threading.Event(), fps=FPS)
    assert r["status"] == "disabled" and r["cameras"]["a"]["frames_seen"] == 0


def _frames_with_mean(mean_rgb, n=100):
    """A uint8 frame of n pixels whose per-channel mean equals ``mean_rgb`` to 1/n (the gate reads only means)."""
    lo = np.floor(mean_rgb)
    ups = np.rint((mean_rgb - lo) * n).astype(int)
    px = np.repeat(lo[None], n, 0)
    for c in range(3):
        px[: ups[c], c] += 1
    return px.reshape(10, n // 10, 3).astype(np.uint8)


def test_real_recorded_transient():
    """The worst recorded start-up transient of the 2026-11-11 robot trials (R/B 0.53 -> 1.14; fixture source in the
    npz): never stable during the cast, stable at the time the gate gave on the full recorded frames (5.80 s)."""
    z = np.load(Path(__file__).parent / "fixtures/robot_pc/warmup_transient.npz")
    assert "reviwo/seed0/left/run002" in str(z["source"])
    gate = ColourStabilityGate(["a"], WarmupConfig(), fps=FPS)
    settle = None
    for t, m in zip(z["capture_ns"], z["mean_rgb"]):
        gate.feed("a", int(t), _frames_with_mean(m))
        if gate.stable and settle is None:
            settle = int(t) / 1e9
        if int(t) < 3_000_000_000:
            assert not gate.stable  # R/B still rising through 1.06
    assert settle is not None and abs(settle - 5.80) < 0.07
    r = gate.result()["a"]
    assert r["rb_range"] <= 0.02 and 1.13 < r["last_rb"] < 1.15
