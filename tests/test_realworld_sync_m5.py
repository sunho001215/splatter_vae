"""Cross-camera sync criterion M5 (coordinator decision 2026-11-06 from PREP2's review
/home/ws/ws/realworld-ops/pre/reviews/sync-criterion-20261106.md): mutual nearest frames without a time cap, a set
accepted iff the robot's URDF surface moved <= 5 mm across its member timestamps; single-view consumers never depend
on cross-camera sync. Synthetic constant-offset data (0, 20, 30, 33 ms), the 5 mm boundary, the rwlearn multi-view
index, single-view independence, and agreement with PREP2's per-episode counts on real recordings (local raw data and
URDF only; skipped elsewhere)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s4d.realworld import train_store as TS
from s4d.realworld.train_windows import enumerate_windows
from s4d.rwlearn.store import MemoryStore, sync_index
from s4d.rwlearn.transitions import timelines

MS = 1_000_000
PERIOD = int(66.715 * MS)
CAMS = ("135122070988", "135122073201", "135122076054")


def stamps(n, offset_ms, jitter_ms=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return (np.arange(n) * PERIOD + int(offset_ms * MS) + rng.uniform(-jitter_ms, jitter_ms, n) * MS).astype(np.int64)


def all_eligible_windows(ref_t, frames, matched, robot_ok=None):
    n, V = frames.shape
    segs = [np.zeros(int(frames[:, v].max(initial=0)) + 1, int) for v in range(V)]
    chunks = [(np.arange(len(s))[None], np.zeros(1, int)) for s in segs]  # one query window with every frame
    rows, _ = enumerate_windows(ref_t, np.zeros(n, int), np.ones(n, bool) if robot_ok is None else robot_ok, frames,
                                matched, segs, chunks, strides=(1, 2), max_gap_factor=1.5, min_views=V)
    return len(rows)


@pytest.mark.parametrize("offset_ms", [0.0, 20.0, 30.0, 33.0, -33.0])
def test_mutual_nearest_pairs_every_constant_offset_below_half_a_period(offset_ms):
    n = 60
    ref, other, third = stamps(n, 0), stamps(n, offset_ms, seed=1), stamps(n, -offset_ms / 2, seed=2)
    frames, err = TS.mutual_sync([ref, other, third])  # M5 pairing: no time cap
    inner = slice(1, n - 1)  # the ends may lose a partner to the shift
    assert (frames[inner] >= 0).all() and (np.diff(frames[inner], axis=0) == 1).all()
    assert abs(err[inner, 1].max() - abs(offset_ms) * MS) < 0.25 * MS
    windows = all_eligible_windows(ref, frames, frames >= 0)
    assert windows >= 2 * n - 10
    # the former raw time bounds lose whole episodes at large constant offsets
    capped20, _ = TS.mutual_sync([ref, other, third], 20 * MS)
    capped30, _ = TS.mutual_sync([ref, other, third], 30 * MS)
    if abs(offset_ms) > 20.2:
        assert all_eligible_windows(ref, capped20, capped20 >= 0) == 0
    if abs(offset_ms) > 30.2:
        assert all_eligible_windows(ref, capped30, capped30 >= 0) == 0
    with pytest.raises(ValueError, match="half the frame period"):
        TS.mutual_sync([ref, other], PERIOD // 2)


def test_dropped_frames_never_pair_across_a_period():
    ref = stamps(30, 0)
    other = np.delete(stamps(30, 31.0, seed=3), [10, 11])
    frames, err = TS.mutual_sync([ref, other])
    assert (frames[:, 1] >= 0).sum() == 28 and (err[frames[:, 1] >= 0, 1] < 33.4 * MS).all()
    used = frames[frames[:, 1] >= 0, 1]
    assert len(set(used.tolist())) == len(used)  # one-to-one


def test_set_surface_motion_and_the_5mm_boundary():
    cloud = np.zeros((5, 3))  # exact arithmetic at the boundary (a real surface: tests below on recordings)

    def surface(state):  # a rigid translation of a fixed cloud by the state's 3-vector
        return cloud + np.asarray(state, float)

    frames = np.array([[0, 0, 0], [1, 0, -1], [2, 2, 2], [3, 3, 3]])
    states = [[(0, 0, 0)] * 4,  # camera 0
              [(0, 0, 0.005), (0, 0, 0), (0, 0, 0), (0, 0, 0)],  # camera 1: frame 0 is 5 mm off
              [(0, 0, 0), (0, 0, 0), (0.0050001, 0, 0), (0, 0, 0)]]  # camera 2: frame 2 is just above 5 mm off
    ok = [np.ones(4, bool), np.ones(4, bool), np.array([True, True, True, False])]
    disp = TS.set_surface_motion(surface, frames, states, ok)
    assert disp[0] == pytest.approx(0.005, abs=1e-12)  # exactly 5 mm: accepted (<=)
    assert disp[1] == pytest.approx(0.005, abs=1e-12)  # camera 2 missing: the pair (0, 1) only
    assert disp[2] == pytest.approx(0.0050001, abs=1e-9) and np.isinf(disp[3])  # invalid robot state: never accepted
    c = TS.sync_counts(frames, disp, 0.005)
    assert c == {"reference_frames": 4, "complete_sets": 3, "accepted": 1, "rejected_motion": 1,
                 "rejected_no_robot_state": 1, "max_set_motion_m": 0.005}


# ---------------------------------------------------------------------------------------------- rwlearn consumers
def synthetic(offsets_ms=(0.0, 20.0, 33.0), n=40, disp=None, raw=None):
    rng = np.random.default_rng(0)
    cams = {}
    for v, c in enumerate(CAMS):
        cams[c] = {"t": stamps(n, offsets_ms[v], seed=v), "segment": np.zeros(n, int),
                   "rgb": rng.integers(0, 256, (n, 16, 16, 3), dtype=np.uint8),
                   "depth": np.ones((n, 16, 16), np.float32), "K": np.eye(3), "w2c": np.eye(4)}
    ep = {"task": "mug-hanging", "raw": raw, "cameras": cams}
    if disp is not None:
        frames, _ = TS.mutual_sync([cams[c]["t"] for c in CAMS])
        ep["sync_sets"] = {"cameras": list(CAMS), "frames": frames, "surface_disp_m": np.full(n, disp, np.float32)}
    return MemoryStore({"mug-hanging/episode_000045": ep}, (16, 16))


def test_multiview_index_uses_the_stored_motion_and_its_threshold():
    ep = "mug-hanging/episode_000045"
    s = sync_index(synthetic(disp=0.004), ep)
    assert s.criterion == "M5" and len(s.times) >= 38  # a 33 ms constant offset is no obstacle
    assert len(sync_index(synthetic(disp=0.0051), ep).times) == 0
    assert len(sync_index(synthetic(disp=0.0051), ep, max_set_motion_m=0.006).times) >= 38
    legacy = sync_index(synthetic(), ep)  # no robot motion in a synthetic store: the former time bound (30 ms)
    assert legacy.criterion == "time" and len(legacy.times) == 0


def test_compact_store_without_motion_refuses_multiview(tmp_path):
    import h5py

    path = tmp_path / "ep" / TS.FILE
    path.parent.mkdir()
    with h5py.File(path, "w") as f:
        f.attrs.update(format=TS.FORMAT, schema_version="1.0.0", episode_id="episode_000045", task="mug-hanging",
                       cameras=json.dumps(list(CAMS)), reference_camera=CAMS[0], image_size=16,
                       max_sync_ns=30 * MS, raw_recording="")
        for v, c in enumerate(CAMS):
            g = f.create_group(f"cameras/{c}")
            g.create_dataset("timestamps_ns", data=stamps(10, 10.0 * v))
            g.create_dataset("segment", data=np.zeros(10, np.int16))
        f.create_dataset("sync/frames", data=np.zeros((10, 3), np.int32))
        f.create_dataset("sync/error_ns", data=np.zeros((10, 3), np.int64))
    store = TS.CompactStore([path])
    with pytest.raises(ValueError, match="re-export train_data"):
        sync_index(store, store.episodes()[0])
    # single-view consumers do not touch the cross-camera sets at all
    tls = timelines(store, store.episodes()[0], multiview=False)
    assert [len(t.times) for t in tls] == [10, 10, 10]


def test_single_view_consumers_are_independent_of_cross_camera_sync():
    """Every set rejected (multi-view empty): the single-view timelines (FQL single view, e2e CNN + FQL) and DynaMo's
    per-camera windows are exactly those of a store whose sets are all accepted."""
    from s4d.rwlearn.baseline_data import RealworldDynaMoWindows

    ep = "mug-hanging/episode_000045"
    good, bad = synthetic(disp=0.001), synthetic(disp=0.02)
    assert len(sync_index(bad, ep).times) == 0 and len(sync_index(good, ep).times) > 0
    for a, b in zip(timelines(good, ep, multiview=False), timelines(bad, ep, multiview=False)):
        assert a.name == b.name and np.array_equal(a.times, b.times) and np.array_equal(a.frames, b.frames)
    assert len(timelines(bad, ep, multiview=True)[0].times) == 0
    assert len(RealworldDynaMoWindows(good, [ep])) == len(RealworldDynaMoWindows(bad, [ep])) > 0


# ---------------------------------------------------------------------------------------------- real recordings
RAW = Path("/home/ws/data/real_world")
URDF = Path("/home/ws/ws/validation-20261031/panda_assets/panda_robotiq_2f85_droid_iso50.urdf")
PREP2 = Path("/home/ws/ws/realworld-ops/pre/reviews/sync-criterion-20261106.json")


def raw_sets(path, geom):
    """Sets and windows of one raw recording with this module's M5 functions (the stage's pairing, motion and window
    rule; every frame in one TrackEverything window, as PREP2's count): colour frames with a stereo record, robot q
    and the Robotiq register interpolated to every member timestamp (the adapter's conventions)."""
    import h5py

    with h5py.File(path, "r") as f:
        t, seg = [], []
        for c in CAMS:
            g = f[f"cameras/cam_{c}"]
            paired = np.asarray(g["stereo/color_index"][()], np.int64)
            t.append(np.rint(np.asarray(g["t"][()], np.float64)[paired] * 1e9).astype(np.int64))
            seg.append(np.asarray(g["segment"][()])[paired])
        rt = np.rint(f["robot/t"][()] * 1e9).astype(np.int64)
        rq = np.asarray(f["robot/q"][()], np.float64)
        grip = np.interp(rt, np.rint(f["gripper/t"][()] * 1e9), np.asarray(f["gripper/position"][()], np.float64))
    frames, _ = TS.mutual_sync(t)
    states = [[(np.array([np.interp(x, rt, rq[:, j]) for j in range(7)]), float(np.interp(x, rt, grip))) for x in tc]
              for tc in t]
    ok = [(tc >= rt[0]) & (tc <= rt[-1]) for tc in t]
    disp = TS.set_surface_motion(lambda st: geom.surface.points(geom.q(st[0], st[1]))[0], frames, states, ok)
    inside = ok[0]
    acc = (frames >= 0).all(1) & (disp <= TS.MAX_SET_MOTION_M) & inside
    segs = [np.asarray(s) for s in seg]
    chunks = [(np.arange(len(tc))[None], np.zeros(1, int)) for tc in t]
    rows, _ = enumerate_windows(t[0], segs[0], inside, frames, (frames >= 0) & acc[:, None], segs, chunks,
                                strides=(1, 2), max_gap_factor=1.5, min_views=3)
    return {"sets": int(acc.sum()), "windows": len(rows), "complete": int((frames >= 0).all(1).sum()),
            "disp_max_mm": float(np.nanmax(np.where(np.isfinite(disp), disp, np.nan)) * 1e3)}


@pytest.mark.skipif(not (RAW.is_dir() and URDF.is_file() and PREP2.is_file()), reason="local raw data / URDF only")
@pytest.mark.parametrize("task,ep", [("mug-hanging", 45), ("mug-hanging", 50), ("mug-hanging", 1),
                                     ("mug-hanging", 55), ("drawer-open", 11), ("drawer-open", 0)])
def test_agreement_with_prep2_on_real_recordings(task, ep):
    from s4d.realworld.pipeline import RobotGeometry

    desc = SimpleNamespace(urdf_path=str(URDF), gripper_model="robotiq_2f_85", gripper_joint="robotiq_85_left_knuckle_joint",
                           gripper_state_semantics="robotiq_position_register", gripper_base_frame=None, mesh_root=None,
                           flange_frame="panda_link8", joint_names=[f"panda_joint{i}" for i in range(1, 8)])
    geom = RobotGeometry(SimpleNamespace(robot=desc))  # the stage's surface sampling (300 points per link)
    got = raw_sets(RAW / task / f"episode_{ep:06d}.h5", geom)
    ref = next(r for r in json.loads(PREP2.read_text())["per_episode"] if r["task"] == task and r["ep"] == ep)
    assert got["complete"] == ref["sets"]["any"]
    assert abs(got["sets"] - ref["sets"]["M5"]) <= 2 and abs(got["windows"] - ref["windows"]["M5"]) <= 6, (got, ref)
    assert got["windows"] > 0
