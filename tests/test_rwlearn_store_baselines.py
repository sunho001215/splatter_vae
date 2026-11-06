"""Store-backed real-world baseline adapters (s4d/rwlearn/store.py, baseline_data.py, configs.py) on a synthetic
in-memory store with the real rig's quirks: three cameras with constant offsets (0, +6.6, -22.9 ms), a dropped frame,
a recording pause (two segments) and per-episode calibrations. Pixel (0, 0) encodes (frame, camera, episode)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import yaml

from s4d.baselines.data import GL_FROM_CV
from s4d.rwlearn import baseline_data as BD
from s4d.rwlearn import configs as C
from s4d.rwlearn.store import FrameStore, MemoryStore, history, open_store, split_episodes, sync_index

MS = 1_000_000
OFFSETS_MS = (0.0, 6.6, -22.9)
CAMS = ("135122070988", "135122073201", "135122076054")


def look_at_w2c(centre, target=(0.33, -0.03, 0.35)):
    z = np.asarray(target, float) - np.asarray(centre, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = centre
    return np.linalg.inv(c2w)


def synthetic_store(episodes=3, frames=24, hw=(32, 32), pause_at=12, drop=(2, 7), seed=0, raw=None) -> MemoryStore:
    rng = np.random.default_rng(seed)
    eps = {}
    for e in range(episodes):
        cams = {}
        for v, cam in enumerate(CAMS):
            t = np.arange(frames) * 66.7 * MS + OFFSETS_MS[v] * MS + e * 10**10
            t[pause_at:] += 1500 * MS  # recording pause
            seg = (np.arange(frames) >= pause_at).astype(int)
            keep = np.ones(frames, bool)
            if drop is not None and v == drop[0]:
                keep[drop[1]] = False
            rgb = rng.integers(0, 256, (frames, *hw, 3), dtype=np.uint8)
            rgb[:, 0, 0] = np.stack([np.arange(frames), np.full(frames, v), np.full(frames, e)], 1)
            K = np.array([[hw[1] * 1.27, 0, hw[1] / 2 + 0.4], [0, hw[0] * 1.27, hw[0] / 2 - 0.3], [0, 0, 1.0]])
            centre = np.array([0.9 + 0.1 * v, -0.6 + 0.6 * v, 0.9]) + 0.01 * e
            cams[cam] = {"t": t[keep].astype(np.int64), "segment": seg[keep], "rgb": rgb[keep],
                         "depth": rng.uniform(0.6, 2.5, (frames, *hw)).astype(np.float32)[keep],
                         "semantic": rng.integers(0, 4, (frames, *hw), dtype=np.uint8)[keep],
                         "K": K, "w2c": look_at_w2c(centre)}
        eps[f"drawer-open/episode_{e:06d}"] = {"task": "drawer-open", "raw": raw, "cameras": cams}
    return MemoryStore(eps, hw)


def test_memory_store_implements_the_protocol_and_registry():
    store = synthetic_store()
    assert isinstance(store, FrameStore)
    assert open_store(store) is store
    with pytest.raises(KeyError, match="no FrameStore opener"):
        open_store({"kind": "nope"})
    with pytest.raises(ValueError, match="no training-store episodes"):  # the merged compact store refuses an empty set
        open_store({"kind": "compact", "paths": []})


def test_sync_index_matches_within_tolerance_and_never_duplicates_frames():
    store = synthetic_store()
    ep = store.episodes()[0]
    s = sync_index(store, ep, max_sync_ns=30 * MS)
    assert s.cameras == list(CAMS)
    for v in range(3):
        assert (np.diff(s.frames[:, v]) > 0).all()  # strictly increasing native frames per camera
        err = np.abs(store.timestamps(ep, CAMS[v])[s.frames[:, v]] - s.times)
        assert err.max() <= 30 * MS
    assert len(s.times) == 24 - 1  # the reference step whose camera-2 frame was dropped is gone
    assert np.unique(s.run).tolist() == [0, 1, 2]  # pause + dropped step split the steps into runs
    assert len(sync_index(store, ep, max_sync_ns=20 * MS).times) == 0  # -22.9 ms camera needs > 22.9 ms
    assert history(s.run, int(np.flatnonzero(s.run == 2)[0]) + 1, 3)[0] == int(np.flatnonzero(s.run == 2)[0])


def test_sincro_windows_carry_their_episode_cameras_in_reference_conventions():
    store = synthetic_store()
    eps = store.episodes()
    ds = BD.RealworldSinCroWindows(store, eps, temporal_stride=1, frame_spacing=1, num_views=3)
    for i in (0, len(ds) // 2, len(ds) - 1):
        x = ds[i]
        ep, start = ds.indices[i]
        assert x["images"].shape == (3, 32, 3, 32, 3) and x["images"].max() <= 1
        s = ds.sync[ep]
        assert len(set(s.run[start:start + 3])) == 1  # inside one run
        frames = (x["images"][:, 0, :, 0, 0] * 255).round().long()  # (T, V) frame ids from pixel (0, 0)
        expected = np.stack([store.rgb(ep, CAMS[v], s.frames[start:start + 3, v])[:, 0, 0, 0] for v in range(3)], 1)
        np.testing.assert_array_equal(frames.numpy(), expected)
        K = store.intrinsics(ep, CAMS[0]).copy()
        K[:2, 2] -= 0.5
        np.testing.assert_allclose(x["K"][0].numpy(), K, atol=1e-5)
        np.testing.assert_allclose(x["c2w"][1].numpy(), np.linalg.inv(store.w2c(ep, CAMS[1])) @ GL_FROM_CV, atol=1e-5)
    assert not torch.equal(ds[0]["c2w"], ds[len(ds) - 1]["c2w"])  # per-episode calibrations


def test_reviwo_mvp_snerl_and_dynamo_samples():
    store = synthetic_store()
    eps = store.episodes()
    rv = BD.RealworldReViWoStates(store, eps, 3)
    y = rv[3]
    assert y["images"].shape == (3, 3, 32, 32) and -1 <= float(y["images"].min()) and float(y["images"].max()) <= 1
    mv = BD.RealworldMVPTimeSteps(store, eps, 3)
    m = mv[(5, 123)]
    ep, t = mv.indices[5]
    s = mv.sync[ep]
    np.testing.assert_array_equal(m["depth"][2].numpy(), store.depth(ep, CAMS[2], [s.frames[t, 2]])[0])
    np.testing.assert_allclose(m["K"].numpy(), np.stack([store.intrinsics(ep, c) for c in CAMS]), atol=1e-6)
    with pytest.raises(ValueError, match="fuses all"):
        BD.RealworldMVPTimeSteps(store, eps, 2)
    sn = BD.RealworldSNeRLScenes(store, eps, seed=0, num_inputs=3, lookat=C.SNERL_LOOKAT, scale=C.SNERL_POSE_SCALE)
    a, b = sn.draw(7), sn.draw(7)
    assert a == b and sorted(a[2]) == [0, 1, 2]  # all three cameras are inputs (rig of exactly three)
    q = sn[7]
    ep, t, inputs, target = a
    s = sn.sync[ep]
    np.testing.assert_array_equal(q["target_semantic"], store.semantic(ep, CAMS[target], [s.frames[t, target]])[0])
    assert q["obs_c2w"].shape == (3, 4, 4)
    K, c2w = sn.camera(ep, target)
    np.testing.assert_allclose(c2w, np.linalg.inv(store.w2c(ep, CAMS[target])))
    dy = BD.RealworldDynaMoWindows(store, eps, window=5, frame_step=1)
    for i in range(0, len(dy), 9):
        ep, cam, frames = dy.window_frames(i)
        w = dy[i]
        assert w.shape == (5, 1, 3, 32, 32) and w.dtype == torch.uint8
        native = store.rgb(ep, cam, frames)[:, 0, 0, 0]
        np.testing.assert_array_equal(w[:, 0, 0, 0, 0].numpy(), native)
        runs = BD.camera_runs(store, ep, cam)
        assert len(set(runs[frames])) == 1  # official slicer per (episode, camera, run)


def test_pickled_dataset_reopens_a_registered_store(tmp_path):
    import pickle

    from s4d.rwlearn.store import register_store

    store = synthetic_store()
    register_store("unit-test-memory", lambda spec: store)
    ds = BD.RealworldReViWoStates({"kind": "unit-test-memory"}, store.episodes(), 3)
    assert ds.store is None  # released after indexing: workers never inherit handles
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.store is None and torch.equal(clone[2]["images"], ds[2]["images"])
    sn = BD.RealworldSNeRLScenes({"kind": "unit-test-memory"}, store.episodes(), seed=0)
    assert sn.store is None and sn[0]["obs_rgb"].shape == (3, 32, 32, 3)
    sn.close()
    assert sn.store is None


def test_sincro_per_sample_cameras_equal_the_reference_for_one_rig_and_differ_otherwise(tmp_path):
    import inspect

    import s4d.baselines.sincro.nerf as sincro_nerf
    from s4d.baselines.sincro import training
    from s4d.baselines.sincro.training import DatasetConfig, ExperimentConfig, SimpleArgs, SinCroModelConfig, TrainConfig

    ref_src = inspect.getsource(training.forward_sincro_batch)
    assert BD._REFERENCE_RAY_BLOCK in ref_src  # the replaced block is verbatim reference code
    tiny = {"img_size": 32, "patch_size": 16, "embed_dim": 32, "vit_depth": 1, "vit_num_heads": 2, "vit_mlp_dim": 64,
            "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_dim": 64, "decoder_output_dim": 16,
            "netdepth": 2, "netwidth": 32, "netdepth_fine": 2, "netwidth_fine": 32, "N_rand": 64, "N_samples": 8,
            "N_importance": 8, "multires": 4, "multires_views": 2, "chunk": 1024, "netchunk": 4096,
            "precrop_iters": 1, "num_views": 6, "num_ref_views": 2}  # encoder num_view (config)
    model_cfg = SinCroModelConfig(**tiny)
    (tmp_path / "nerf").mkdir()
    args = SimpleArgs(model_cfg, TrainConfig(), DatasetConfig(batch_size=2), ExperimentConfig(str(tmp_path), "nerf"))
    sincro_nerf.device = torch.device("cpu")
    torch.manual_seed(0)
    kwargs, _, _, _, _, latent = sincro_nerf.create_nerf(args, args.basedir, args.expname)
    kwargs.update(near=model_cfg.near, far=model_cfg.far)
    store = synthetic_store()
    ds = BD.RealworldSinCroWindows(store, store.episodes(), frame_spacing=1, num_views=3)
    same = {k: torch.stack([ds[0][k], ds[1][k]]) for k in ("images", "K", "c2w")}  # one episode: one rig

    def run(fn, batch):
        torch.manual_seed(1)
        np.random.seed(1)
        loss, stats = fn(batch, latent, kwargs, args, model_cfg, global_step=5)
        return loss.item(), stats

    assert run(training.forward_sincro_batch, same) == run(BD.sincro_forward_per_sample_cameras, same)
    mixed = {k: torch.stack([ds[0][k], ds[len(ds) - 1][k]]) for k in ("images", "K", "c2w")}  # two episodes
    a, b = run(training.forward_sincro_batch, mixed), run(BD.sincro_forward_per_sample_cameras, mixed)
    assert a != b and math.isfinite(b[0])


def test_multiview_observation_time_is_the_latest_constituent_frame():
    from s4d.rwlearn.transitions import timelines

    store = synthetic_store()
    ep = store.episodes()[0]
    s = sync_index(store, ep)
    frame_t = np.stack([store.timestamps(ep, c)[s.frames[:, v]] for v, c in enumerate(CAMS)], 1)
    np.testing.assert_array_equal(s.latest_times, frame_t.max(1))
    assert (s.latest_times - s.times).max() == 6.6 * MS  # camera 1 is 6.6 ms after the reference
    (tl,) = timelines(store, ep, multiview=True)
    np.testing.assert_array_equal(tl.times, s.latest_times)


def test_gate_evidence_required_beyond_smoke_runs():
    store = synthetic_store()
    assert BD.check_gate(store, None, steps=100, consumer="dynamo")["smoke"]
    with pytest.raises(RuntimeError, match="no passed full-data gate"):
        BD.check_gate(store, None, steps=300_000, consumer="dynamo")
    report = {"passed": True, "tasks": {"drawer-open": {"passed": True, "store_episodes": store.episodes()}}}
    assert not BD.check_gate(store, report, steps=None, consumer="dynamo")["smoke"]
    eps = store.episodes()
    good = {"train": eps[:2], "validation": eps[2:]}
    assert not BD.check_gate(store, report, steps=None, split=good, target="drawer-open", consumer="dynamo",
                             caps={"max_episodes": None, "max_frames_per_demo": 3000})["smoke"]
    for split in ({"train": eps[:1], "validation": eps[2:]}, {"train": eps, "validation": eps[2:]},
                  {"train": eps[:2], "validation": eps[2:] * 2}):
        with pytest.raises(RuntimeError, match="partition"):
            BD.check_gate(store, report, steps=None, split=split, consumer="dynamo")
    with pytest.raises(RuntimeError, match="target mug-hanging"):
        BD.check_gate(store, report, steps=None, target="mug-hanging", consumer="dynamo")
    with pytest.raises(RuntimeError, match="unknown target"):  # task-specific encoders only: no joint target
        BD.check_gate(store, report, steps=None, target="joint", consumer="dynamo")
    for caps in ({"max_episodes": 2}, {"max_frames_per_demo": 10}):
        with pytest.raises(RuntimeError, match="truncate"):
            BD.check_gate(store, report, steps=None, caps=caps, consumer="dynamo")
        BD.check_gate(store, report, steps=100, caps=caps, consumer="dynamo")  # smoke-length runs may cap
    BD.check_gate(store, report, steps=None, caps={"max_episodes": 3, "num_episodes": 50}, consumer="dynamo")
    report["tasks"]["drawer-open"]["store_episodes"] = store.episodes()[:-1]
    with pytest.raises(RuntimeError, match="differ"):
        BD.check_gate(store, report, steps=None, consumer="dynamo")
    with pytest.raises(ValueError, match="not a real-world data interface"):
        BD.realworld_classes({"source": "metaworld"}, 10, store, consumer="dynamo")


def test_trainer_hook_binds_the_gate_to_the_calling_trainer():
    """Codex review: a geometric trainer must not be downgraded by a config that declares a non-geometric consumer."""
    store = synthetic_store()
    for e in store._eps.values():
        e["upstream"] = {"calibrate": "cal-1"}
    report = {"passed": True, "tasks": {"drawer-open": {"passed": True, "store_episodes": store.episodes()}}}
    iface = {"source": "realworld", "gate_report": report, "target": "drawer-open"}
    assert BD.realworld_classes({**iface, "consumer": "dynamo"}, 10**6, store, consumer="dynamo")["dynamo"]
    with pytest.raises(ValueError, match="conflicts with the trainer"):
        BD.realworld_classes({**iface, "consumer": "dynamo"}, 10**6, store, consumer="snerl")
    with pytest.raises(RuntimeError, match="calibration hold"):  # no declared consumer: the trainer's identity counts
        BD.realworld_classes(iface, 10**6, store, consumer="snerl")
    with pytest.raises(RuntimeError, match="calibration hold"):
        BD.realworld_classes(iface, 100, store, consumer="sincro")
    assert BD.realworld_classes({**iface, "smoke_without_calibration": True}, 100, store, consumer="sincro")["sincro"]


def _hold(store, build="cal-1", cams=CAMS, passed=True):
    return {"passed": passed, "episodes": {ep: {"calibrate_build_id": build, "cameras": sorted(cams)}
                                           for ep in store.episodes()} if passed else {}}


@pytest.mark.parametrize("consumer", ["snerl", "mvp3d", "sincro", None, "unknown-consumer"])
def test_geometry_dependent_consumers_need_the_validated_calibration(consumer):
    """Calibration hold (user decision 2026-11-06, item 2): geometry-dependent (and unclassified) consumers need the
    released hold AND a store built from each episode's validated calibrate build with only validated cameras."""
    store = synthetic_store()
    for e in store._eps.values():
        e["upstream"] = {"calibrate": "cal-1"}
    base = {"passed": True, "tasks": {"drawer-open": {"passed": True, "store_episodes": store.episodes()}}}
    with pytest.raises(RuntimeError, match="calibration hold"):  # no hold section: held
        BD.check_gate(store, base, steps=None, consumer=consumer)
    with pytest.raises(RuntimeError, match="calibration hold"):  # short runs too, unless explicitly opted in
        BD.check_gate(store, base, steps=100, consumer=consumer)
    assert BD.check_gate(store, base, steps=100, consumer=consumer, smoke_without_calibration=True)["smoke"]
    with pytest.raises(RuntimeError, match="calibration hold"):
        BD.check_gate(store, {**base, "calibration_hold": {"drawer-open": _hold(store, passed=False)}}, steps=None,
                      consumer=consumer)
    ok = {**base, "calibration_hold": {"drawer-open": _hold(store)}}
    res = BD.check_gate(store, ok, steps=None, consumer=consumer)
    assert not res["smoke"] and res["geometry_dependent"]
    with pytest.raises(RuntimeError, match="store built from calibrate"):
        BD.check_gate(store, {**base, "calibration_hold": {"drawer-open": _hold(store, build="cal-2")}}, steps=None,
                      consumer=consumer)
    with pytest.raises(RuntimeError, match="no validated calibration"):
        BD.check_gate(store, {**base, "calibration_hold": {"drawer-open": _hold(store, cams=CAMS[:2])}}, steps=None,
                      consumer=consumer)
    for e in store._eps.values():
        e.pop("upstream")
    with pytest.raises(RuntimeError, match="store built from calibrate None"):
        BD.check_gate(store, ok, steps=None, consumer=consumer)


@pytest.mark.parametrize("consumer", ["dynamo", "reviwo", "e2e_cnn"])
def test_non_geometric_consumers_are_not_held_by_the_calibration(consumer):
    store = synthetic_store()
    base = {"passed": True, "tasks": {"drawer-open": {"passed": True, "store_episodes": store.episodes()}},
            "calibration_hold": {"drawer-open": {"passed": False, "episodes": {}}}}
    res = BD.check_gate(store, base, steps=None, consumer=consumer)
    assert not res["smoke"] and not res["geometry_dependent"]


def test_split_is_per_task_seeded_and_disjoint():
    store = synthetic_store(episodes=20)
    s = split_episodes(store, 0.1, 0)
    assert len(s["validation"]) == 2 and len(s["train"]) == 18
    assert not set(s["train"]) & set(s["validation"]) and split_episodes(store, 0.1, 0) == s
    assert split_episodes(store, 0.1, 1)["validation"] != s["validation"]


def test_committed_configs_are_the_derived_ones_and_change_only_the_documented_keys():
    for path, text in C.derive_all().items():
        assert path.read_text() == text, f"{path} is stale; run scripts/realworld_learning.py configs"
    for b in C.BASELINES:
        base = yaml.safe_load((C.REPO / f"configs/baselines/{b}/drawer-open.yaml").read_text())
        for target in C.TARGETS:
            cfg = yaml.safe_load((C.REPO / f"configs/realworld/baselines/{b}/{target}.yaml").read_text())
            changed = {k for k, *_ in C.overrides(b, target)}

            def flat(d, prefix=""):
                out = {}
                for k, v in d.items():
                    out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
                return out

            fb, fc = flat(base), flat(cfg)
            diff = {k for k in set(fb) | set(fc) if fb.get(k) != fc.get(k)}
            assert diff <= changed, (b, target, diff - changed)
            assert fc["data_interface.source"] == "realworld" and target in fc["dataset.hdf5_path"]
    assert C.SNERL_LOOKAT == [0.334, -0.031, 0.353] and C.SNERL_POSE_SCALE == 3.79
