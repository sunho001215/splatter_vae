"""Held-out replay check: frame 0 may differ only when its stored kinematics explain it (Meta-World window-open)."""

from __future__ import annotations

import types

import numpy as np
import pytest

hr = pytest.importorskip("s4d.data.metaworld.heldout_render")


class Episode(dict):
    def __init__(self, length, rgb, depth, xpos):
        super().__init__(rgb=rgb, depth=depth, xpos=xpos)
        self.attrs = {"length": length}


def run(monkeypatch, *, mismatch_t, fraction, deviation_m, deviation_t=0):
    length, cams, n = 30, 2, 32
    rgb = np.zeros((length, cams, n, n, 3), np.uint8)
    depth = np.full((length, cams, n, n), 5000, np.uint16)
    xpos = np.zeros((length, 4, 3))
    scene = types.SimpleNamespace(model=types.SimpleNamespace(nbody=4, body_mocapid=np.array([-1, -1, -1, 0])),
                                  data=types.SimpleNamespace(xpos=np.zeros((4, 3))), cams=list(range(cams)))
    current = {}

    def set_state(_scene, _episode, t):
        current["t"] = t
        scene.data.xpos = xpos[t].copy()
        if t == deviation_t:
            scene.data.xpos[1, 0] += deviation_m  # recomputed kinematics differ from the stored ones

    def render(_scene, _cams):
        out = rgb[current["t"]].copy()
        if current["t"] == mismatch_t:
            flat = out.reshape(-1, 3)
            flat[: int(fraction * flat.shape[0])] = 200
        return out, depth[current["t"]].copy()

    monkeypatch.setattr(hr, "_set_goal", lambda *_: None)
    monkeypatch.setattr(hr, "_set_state", set_state)
    monkeypatch.setattr(hr, "_render", render)
    return hr.replay_check(scene, Episode(length, rgb, depth, xpos), np.arange(cams))


def test_exact_replay_passes_and_checks_frame_one(monkeypatch):
    check = run(monkeypatch, mismatch_t=-1, fraction=0.0, deviation_m=0.0)
    assert check["passed"] and [r["t"] for r in check["rows"]] == [0, 1, 10, 20, 29]


def test_frame_zero_mismatch_explained_by_stale_reset_kinematics_passes(monkeypatch):
    assert run(monkeypatch, mismatch_t=0, fraction=0.003, deviation_m=5e-4)["passed"]


@pytest.mark.parametrize("deviation_m", [0.0, 2e-3])
def test_frame_zero_mismatch_without_bounded_kinematic_explanation_fails(monkeypatch, deviation_m):
    assert not run(monkeypatch, mismatch_t=0, fraction=0.003, deviation_m=deviation_m)["passed"]


def test_later_frames_stay_strict(monkeypatch):
    assert not run(monkeypatch, mismatch_t=1, fraction=0.003, deviation_m=5e-4, deviation_t=1)["passed"]
    assert not run(monkeypatch, mismatch_t=0, fraction=0.02, deviation_m=5e-4)["passed"]
