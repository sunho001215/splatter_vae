"""Offline FQL pipeline on frozen baseline encoders (s4d/rwlearn/fql_data.py, inference.py): a synthetic 128 px store
on the REAL camera timeline of the action fixture (mug-hanging episode_000003, two segments), frozen random-init
exports of a single-view frame encoder (DynaMo), a multi-view encoder (SNeRL) and a sequence encoder (SinCro); dataset
build, camera-randomised sampling invariants, FQL updates with the per-network adapter heads, checkpoint ->
``OfflinePolicy`` -> robot command (no robot I/O)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from s4d.rwlearn import actions as A
from s4d.rwlearn import fql_data as FD
from s4d.rwlearn.fql import FQLAgent
from s4d.rwlearn.gate import QA_FAMILIES
from s4d.rwlearn.inference import OfflinePolicy, checkpoint_payload, replay
from s4d.rwlearn.store import MemoryStore
from tests.test_rwlearn_gate import gate_identities

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "rwlearn" / "raw_mug_hanging_ep3_actions.npz"
N_FRAMES = 18  # per camera; the first segment of the fixture has ~114 frames, the test uses its start


def look_at_w2c(centre, target=(0.33, -0.03, 0.35)):
    z = np.asarray(target, float) - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    c2w[:3, 3] = centre
    return np.linalg.inv(c2w)


def store_on_fixture(n=N_FRAMES, hw=128, seed=0) -> MemoryStore:
    raw = A.raw_from_arrays(np.load(FIXTURE))
    rng = np.random.default_rng(seed)
    cams = {}
    for v, serial in enumerate(raw.serials):
        t, seg = raw.cameras[serial]["t"][:n], raw.cameras[serial]["segment"][:n]
        K = np.array([[160.0, 0, 64.4], [0, 160.0, 65.1], [0, 0, 1]])
        w2c = look_at_w2c(np.array([1.0 + 0.1 * v, -0.6 + 0.6 * v, 0.9]))
        cams[serial] = {"t": t, "segment": seg, "rgb": rng.integers(0, 256, (n, hw, hw, 3), dtype=np.uint8),
                        "depth": rng.uniform(0.6, 1.4, (n, hw, hw)).astype(np.float32),
                        "semantic": rng.integers(0, 4, (n, hw, hw), dtype=np.uint8), "K": K, "w2c": w2c}
    return MemoryStore({"mug-hanging/episode_000003": {"task": "mug-hanging", "raw": str(FIXTURE), "cameras": cams}},
                       (hw, hw))


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    d = tmp_path_factory.mktemp("exports")
    from s4d.baselines.dynamo.model import build_encoder, encoder_kwargs
    from s4d.baselines.dynamo.model import save_export as save_dynamo

    cfg = yaml.safe_load((REPO / "configs/realworld/baselines/dynamo/drawer-open.yaml").read_text())
    torch.manual_seed(0)
    enc = build_encoder(encoder_kwargs(cfg)).eval()
    save_dynamo(d / "dynamo.pt", enc, encoder_kwargs(cfg), 1, 128, 0, True)

    from s4d.baselines.snerl import model as M

    nerf = yaml.safe_load((REPO / "configs/realworld/baselines/snerl/drawer-open.yaml").read_text())["nerf"]
    nerf.update(netdepth=2, netwidth=32, netdepth_fine=2, netwidth_fine=32, N_samples=8, N_importance=8)
    model = M.create(nerf, "cpu")
    M.save_export(d / "snerl.pt", model.encoder, 0, 3, lookat=(0.334, -0.031, 0.353), scale=3.79, random_init=True)

    from s4d.baselines.sincro.encoder import build_encoder as build_sincro
    from s4d.baselines.sincro.encoder import save_export as save_sincro
    from s4d.baselines.sincro.training import SinCroModelConfig

    tiny = {"img_size": 128, "patch_size": 16, "embed_dim": 32, "vit_depth": 1, "vit_num_heads": 2, "vit_mlp_dim": 64,
            "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_dim": 64, "decoder_output_dim": 16,
            "num_views": 6, "num_ref_views": 2}
    mc = asdict(SinCroModelConfig(**tiny))
    save_sincro(d / "sincro.pt", build_sincro(mc, "cpu").eval(), mc, 1, 0)

    from s4d.baselines.mvp3d.model import MVP3DConfig
    from s4d.baselines.mvp3d.model import build_encoder as build_mvp
    from s4d.baselines.mvp3d.model import save_export as save_mvp
    from s4d.rwlearn.configs import MVP3D_BOUNDS

    tiny_mvp = MVP3DConfig(img_size=24, img_patch_size=6, im_channels=16, attn_dim=32, attn_heads=2, attn_dim_head=16,
                           depth=4, decoder_dim=32, decoder_heads=2, decoder_dim_head=16)
    save_mvp(d / "mvp3d.pt", build_mvp(tiny_mvp).eval(), asdict(tiny_mvp), {"bounds": MVP3D_BOUNDS, "radius": 0.012},
             0, random_init=True)
    return d


@pytest.mark.parametrize("condition,timeline_names", [
    ("dynamo", None), ("snerl", ["sync"]), ("sincro", None), ("mvp3d_single", None), ("mvp3d_multiview", ["sync"])])
def test_dataset_build_sampling_and_policy_round_trip(tmp_path, exports, condition, timeline_names):
    torch.manual_seed(0)
    store = store_on_fixture()
    ep = store.episodes()[0]
    path = tmp_path / f"{condition}.h5"
    info = FD.build_fql_dataset(store, condition, exports / f"{FD.CONDITIONS[condition].encoder_type}.pt", path,
                                episodes=[ep], assume_final_success=True)
    assert info["transitions"] > 0
    with h5py.File(path, "r") as f:
        names = sorted(f["episodes"][ep])
        assert names == (timeline_names or sorted(store.cameras(ep)))
        g = f["episodes"][ep][names[0]]
        ok = g["transition_ok"][:].astype(bool)
        assert g["features"].shape[0] == len(g["times"]) and g["features"].dtype == np.float16
        r, m = g["reward"][:][ok], g["mask"][:][ok]
        assert (r == -1).all() and (m == 1).all()  # the slice ends before the recording: no success label
        assert g.attrs["label"] == "success_frame_unreachable"
        assert np.abs(g["action"][:][ok]).max() <= 1.0
    data = FD.OfflineTransitions(path)
    gen = torch.Generator().manual_seed(0)
    batch = data.sample(64, gen)
    feat = batch["observations"]["feat"]
    assert feat.shape == ((64, data.feat_dim) if data.sequence else (64, FD.FRAME_STACK, data.feat_dim))
    # viewpoint fixed per sample: the observation history and its next observation lie in ONE timeline (camera)
    spans = [(base, base + n) for *_, base, n in data.timelines]
    idx = torch.randint(data.size, (200,), generator=gen)
    for o, nx in zip(data.obs_idx[idx], data.next_idx[idx]):
        tl_o = {i for i, (a, b) in enumerate(spans) for x in o.tolist() if a <= x < b}
        tl_n = {i for i, (a, b) in enumerate(spans) for x in nx.tolist() if a <= x < b}
        assert len(tl_o) == 1 and tl_o == tl_n
        if not data.sequence:
            assert nx[-1] == o[-1] + 1 and (nx[:-1] >= o[:-1]).all()  # next history = history shifted by one step
    if len(data.timelines) > 1:  # single-view conditions: every camera is sampled
        hit = {i for x in data.obs_k[torch.randint(data.size, (400,), generator=gen)].tolist()
               for i, (a, b) in enumerate(spans) if a <= x < b}
        assert hit == set(range(len(spans)))

    enc = data.encoder_factory(32)
    agent = FQLAgent.create(0, enc().out_dim, A.ACTION_DIM, {"actor_hidden_dims": (32, 32), "value_hidden_dims": (32, 32),
                                                              "batch_size": 64}, encoder=enc)
    for _ in range(3):
        info = agent.update(data.sample(64, gen))
    assert all(np.isfinite(v) for v in info.values())
    assert agent.critic.encoder is not agent.actor_bc_flow.encoder  # one adapter per network (official)
    ckpt = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(agent, data, condition, 3), ckpt)
    policy = OfflinePolicy(ckpt)
    # the cached features equal the frozen encoder on the store frames (same preprocessing at train and act time)
    with h5py.File(path, "r") as f:
        name = sorted(f["episodes"][ep])[0]
        g = f["episodes"][ep][name]
        cams = __import__("json").loads(g.attrs["cameras"])
        frames, run = g["frames"][:], g["run"][:]
        k = int(np.flatnonzero(g["transition_ok"][:])[3])
        hist = [max(k - j, int(np.flatnonzero(run == run[k])[0])) for j in (2, 1, 0)]
        rgb = np.stack([np.stack([store.rgb(ep, c, [frames[h, v]])[0] for v, c in enumerate(cams)]) for h in hist])
        obs = {"rgb": rgb}
        if FD.CONDITIONS[condition].depth:
            obs["depth"] = np.stack([np.stack([store.depth(ep, c, [frames[h, v]])[0] for v, c in enumerate(cams)])
                                     for h in hist])
        if FD.CONDITIONS[condition].cameras:
            obs["K"] = np.stack([store.intrinsics(ep, c) for c in cams])
            obs["c2w"] = np.stack([np.linalg.inv(store.w2c(ep, c)) for c in cams])
        live = policy.features(obs)
        cached = torch.from_numpy(g["features"][:][[k] if data.sequence else hist].astype(np.float32))
        torch.testing.assert_close(live.squeeze(0).float(), cached.reshape(live.squeeze(0).shape), atol=2e-3,
                                   rtol=2e-3)
        pose = A.matrix_to_pose7(g["measured_pose"][k])
        grip = A.GRIPPER_OPEN_REGISTER + A.GRIPPER_SPAN * float(g["proprio"][k, -1])
    out = policy.act(obs, pose, grip, noise=torch.zeros(1, A.ACTION_DIM))
    assert out["flange_pose"].shape == (7,) and 3 <= int(out["gripper_register"]) <= 230
    goal = A.pose7_to_matrix(out["flange_pose"])
    np.testing.assert_allclose(A.pose_delta(A.pose7_to_matrix(pose), goal), out["delta_phys"], atol=1e-9)
    assert np.abs(out["delta_phys"][:3]).max() <= policy.contract.pos_scale_m + 1e-12
    rep = replay(policy, path, ep, name)
    assert rep["transitions"] == int(ok.sum()) and np.isfinite(rep["position_error_mm_median"])


def test_export_mismatch_is_refused(tmp_path, exports):
    store = store_on_fixture(n=8)
    ep = store.episodes()[0]
    path = tmp_path / "d.h5"
    with pytest.raises(ValueError, match="no outcome label"):
        FD.build_fql_dataset(store, "dynamo", exports / "dynamo.pt", path, episodes=[ep])
    with pytest.raises(ValueError, match="JSON booleans"):
        FD.build_fql_dataset(store, "dynamo", exports / "dynamo.pt", path, episodes=[ep], outcomes={ep: "false"})
    FD.build_fql_dataset(store, "dynamo", exports / "dynamo.pt", path, episodes=[ep], outcomes={ep: False})
    with h5py.File(path, "r") as f:
        assert all(g.attrs["label"] == "failure" and not g["success"][:].any() for g in f["episodes"][ep].values())
    from s4d.rwlearn.gate import check_dataset_gate

    assert check_dataset_gate(path, None, steps=100)["smoke"]
    with pytest.raises(RuntimeError, match="no passed full-data gate"):
        check_dataset_gate(path, None, steps=1_000_000)
    report = {"tasks": {"mug-hanging": {"passed": True, **gate_identities(store)}},
              "families": {"mug-hanging": {f: {"passed": True} for f in QA_FAMILIES}}}
    assert not check_dataset_gate(path, report, steps=None, split={"train": [ep], "validation": []})["smoke"]
    with pytest.raises(RuntimeError, match="partition"):
        check_dataset_gate(path, report, steps=None, split={"train": [], "validation": []})
    data = FD.OfflineTransitions(path)
    enc = data.encoder_factory(16)
    agent = FQLAgent.create(0, enc().out_dim, A.ACTION_DIM, {"actor_hidden_dims": (8,), "value_hidden_dims": (8,)},
                            encoder=enc)
    torch.save(checkpoint_payload(agent, data, "dynamo", 0), tmp_path / "p.pt")
    with pytest.raises(ValueError, match="differs"):
        OfflinePolicy(tmp_path / "p.pt", export_path=exports / "snerl.pt")


def test_default_adapter_input_is_unchanged():
    """``feature_norm`` none (every condition but 3D-MVP): no buffers, the head reads the stored features as before."""
    torch.manual_seed(3)
    enc = FD.ObsEncoder(12, False, proj=8)
    assert enc.feature_norm is None and enc.stats_sha256() is None
    assert set(enc.state_dict()) == {f"head.{k}" for k in enc.head.state_dict()}
    feat, proprio = torch.randn(5, FD.FRAME_STACK, 12).half(), torch.randn(5, A.PROPRIO_DIM).double()
    assert torch.equal(enc({"feat": feat, "proprio": proprio}),
                       torch.cat([enc.head(feat.float()), proprio.float()], dim=-1))
    ok = {"kind": "train_standardize", "dim": 12, "mean": np.zeros(12), "std": np.ones(12)}
    for bad in ({**ok, "dim": 13}, {**ok, "kind": "other"}, {**ok, "std": np.ones(11)}):
        with pytest.raises(ValueError):
            FD.ObsEncoder(12, False, proj=8, feature_stats=bad)


def test_feature_standardisation_numerics():
    """The 3D-MVP regime (offset ~400, per-dim sd 0.05..2, float16 storage) is recovered to float32 precision; a
    constant dimension is floored (no division by zero); fewer than two rows or all-constant features are refused."""
    rng = np.random.default_rng(0)
    offset = np.concatenate([np.full(8, 400.0), np.full(7, 10.0), [400.0]]) * rng.choice([-1, 1], 16)
    sd = np.concatenate([rng.uniform(0.5, 2.0, 8), rng.uniform(0.05, 0.2, 7), [0.0]])
    x64 = offset + rng.normal(size=(5000, 16)) * sd
    x = torch.from_numpy(x64).half()
    stats = FD.fit_feature_standardization(x, chunk=777)
    ref = x.double().numpy()
    floor = FD.STD_FLOOR_REL * ref.std(0).max()
    np.testing.assert_allclose(stats["mean"], ref.mean(0), rtol=1e-7)
    np.testing.assert_allclose(stats["std"], np.maximum(ref.std(0), floor), rtol=1e-6)
    assert stats["floored_dims"] == 1 and stats["std"][-1] == np.float32(floor)
    z = FD.ObsEncoder(16, True, proj=8, feature_stats=stats)
    standardised = (x.float() - z.feat_mean) / z.feat_std
    np.testing.assert_allclose(standardised[:, :15].double().mean(0).numpy(), 0, atol=1e-4)
    np.testing.assert_allclose(standardised[:, :15].double().std(0, correction=0).numpy(), 1, atol=1e-4)
    with pytest.raises(ValueError, match="two stored rows"):
        FD.fit_feature_standardization(x[:1])
    with pytest.raises(ValueError, match="constant"):
        FD.fit_feature_standardization(torch.ones(10, 4).half())


@pytest.mark.parametrize("condition", ["dynamo", "sincro", "mvp3d_single", "mvp3d_multiview"])
def test_feature_standardisation_is_fitted_on_the_given_rows_and_travels_with_the_policy(tmp_path, exports,
                                                                                         condition):
    """``encoder.feature_norm=train_standardize``: per-dimension statistics of the stored rows of the given (training)
    timelines, float64, std floored; buffers of EVERY network's encoder (target critic included), constant under the
    updates, recorded in the checkpoint meta and the deploy contract, rebuilt and sha256-verified by
    ``OfflinePolicy``; the adapter sees exactly (f - mu) / sigma and starts from the same initial weights."""
    from s4d.rwlearn.deploy.contracts import read_training_contract

    torch.manual_seed(0)
    store = store_on_fixture()
    ep = store.episodes()[0]
    path = tmp_path / f"{condition}.h5"
    FD.build_fql_dataset(store, condition, exports / f"{FD.CONDITIONS[condition].encoder_type}.pt", path,
                         episodes=[ep], assume_final_success=True)
    data = FD.OfflineTransitions(path, [ep])
    stats = FD.fit_feature_standardization(data.features, chunk=7)
    with h5py.File(path, "r") as f:
        x = np.concatenate([g["features"][:] for g in f["episodes"][ep].values()]).astype(np.float64)
    sd = x.std(0)
    np.testing.assert_allclose(stats["mean"], x.mean(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(stats["std"], np.maximum(sd, FD.STD_FLOOR_REL * sd.max()), rtol=1e-5, atol=1e-8)
    assert stats["mean"].dtype == stats["std"].dtype == np.float32
    assert (stats["rows"], stats["dim"]) == (len(x), data.feat_dim) and stats["kind"] == "train_standardize"
    assert stats["sha256"] == FD.feature_stats_sha256(stats["mean"], stats["std"])
    record = {k: v for k, v in stats.items() if k not in ("mean", "std")}

    cfg = {"actor_hidden_dims": (32, 32), "value_hidden_dims": (32, 32), "batch_size": 64}
    agent = FQLAgent.create(0, 32 + A.PROPRIO_DIM, A.ACTION_DIM, cfg, encoder=data.encoder_factory(32, stats))
    raw = FQLAgent.create(0, 32 + A.PROPRIO_DIM, A.ACTION_DIM, cfg, encoder=data.encoder_factory(32))
    for name, module in agent.modules().items():  # the buffers consume no randomness: same initial parameters
        sd_norm, sd_raw = module.state_dict(), raw.modules()[name].state_dict()
        assert set(sd_norm) - set(sd_raw) == {"encoder.feat_mean", "encoder.feat_std"}
        assert all(torch.equal(sd_norm[k], v) for k, v in sd_raw.items())
    obs = data.sample(16, torch.Generator().manual_seed(1))["observations"]
    z = (obs["feat"].float() - torch.from_numpy(stats["mean"])) / torch.from_numpy(stats["std"])
    assert torch.equal(agent.critic.encoder(obs), raw.critic.encoder({"feat": z, "proprio": obs["proprio"]}))
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        info = agent.update(data.sample(64, gen))
    assert all(np.isfinite(v) for v in info.values())
    assert all(m.encoder.stats_sha256() == stats["sha256"] for m in agent.modules().values())

    ckpt = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(agent, data, condition, 3), ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["meta"]["feature_norm"] == record
    assert read_training_contract(payload["meta"], path)["feature_norm"] == record  # dataset source: the ckpt's
    policy = OfflinePolicy(ckpt)
    agent.train_mode(False)
    probe = {"feat": obs["feat"][:2].float(), "proprio": obs["proprio"][:2]}
    noise = torch.randn(2, A.ACTION_DIM, generator=torch.Generator().manual_seed(2))
    assert torch.equal(policy.agent.sample_actions(probe, noise), agent.sample_actions(probe, noise))
    rep = replay(policy, path, ep, sorted(store.cameras(ep))[0] if not FD.CONDITIONS[condition].multiview else "sync")
    assert np.isfinite(rep["position_error_mm_median"])

    payload["agent"]["modules"]["actor_onestep_flow"]["encoder.feat_mean"][0] += 1
    torch.save(payload, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="standardisation differs"):
        OfflinePolicy(tmp_path / "tampered.pt")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload["meta"].pop("feature_norm")
    torch.save(payload, tmp_path / "unrecorded.pt")
    with pytest.raises(RuntimeError, match="Unexpected key"):  # buffers without the meta record never load silently
        OfflinePolicy(tmp_path / "unrecorded.pt")
    payload = checkpoint_payload(raw, data, condition, 0)
    payload["meta"]["feature_norm"] = record
    torch.save(payload, tmp_path / "unbuffered.pt")
    with pytest.raises(RuntimeError, match="Missing key"):  # a meta record without the buffers neither
        OfflinePolicy(tmp_path / "unbuffered.pt")


def test_fql_build_preserves_raw_identity_across_pipeline_names(tmp_path):
    import json

    from s4d.rwlearn import gate, outcomes
    from s4d.rwlearn.store import split_episodes
    from tests.test_rwlearn_gate import entry

    store = store_on_fixture(n=8, hw=32)
    old = store.episodes()[0]
    ep = "mug-hanging/mug-hanging-000003"
    store._eps[ep] = store._eps.pop(old)
    raw_sha = FD.sha256_file(FIXTURE)
    store._eps[ep]["raw_sha256"] = raw_sha
    rec = entry("mug-hanging", 3, sha=raw_sha)
    rec["filename"] = FIXTURE.name
    manifest = {"schema": gate.SCHEMA, "schema_version": gate.SCHEMA_VERSION,
                "episodes": {"mug-hanging/ingest-record-3": rec},
                "counts": {"mug-hanging": {"complete": True, "target": 1}}}
    report = gate.evaluate(manifest, ["mug-hanging"], target=1)
    assert report["passed"]
    path = tmp_path / "e2e_identity.h5"
    FD.build_fql_dataset(store, "e2e_cnn", None, path, episodes=[ep], outcomes=outcomes.outcome_labels([ep]))
    with h5py.File(path, "r") as f:
        identity = json.loads(f.attrs["raw_identities"])[ep]
        assert identity["raw_recording"] == str(FIXTURE) and identity["raw_sha256"] == raw_sha
    assert not gate.check_dataset_gate(path, report, steps=None, split=split_episodes(store))["smoke"]
    for value in ({}, {ep: {**identity, "raw_recording": "other.npz"}},
                  {ep: {**identity, "raw_sha256": "f" * 64}}):
        with h5py.File(path, "r+") as f:
            f.attrs["raw_identities"] = json.dumps(value)
        with pytest.raises(RuntimeError, match="raw identity"):
            gate.check_dataset_gate(path, report, steps=None)


# ---------------------------------------------------------------------------------------------- end-to-end CNN + FQL
def test_e2e_cnn_dataset_pixel_sampling_joint_training_and_policy(tmp_path):
    """e2e_cnn: transitions without features; frames read once per camera from the store; the camera is random per
    sample and fixed through the 3-frame history and the paired next observation; the IMPALA encoders are trained
    jointly (parameters move); checkpoint -> OfflinePolicy.act -> robot command."""
    from s4d.rwlearn import outcomes as O
    from s4d.rwlearn import pixel as PX

    store = store_on_fixture(hw=32)
    ep = store.episodes()[0]
    path = tmp_path / "e2e.h5"
    with pytest.raises(ValueError, match="takes no encoder export"):
        FD.build_fql_dataset(store, "e2e_cnn", FIXTURE, path, episodes=[ep], assume_final_success=True)
    with pytest.raises(ValueError, match="needs its encoder export"):
        FD.build_fql_dataset(store, "dynamo", None, path, episodes=[ep], assume_final_success=True)
    doc = O.outcome_labels([ep])
    info = FD.build_fql_dataset(store, "e2e_cnn", None, path, episodes=[ep], outcomes=doc)
    assert info["transitions"] > 0
    with h5py.File(path, "r") as f:
        assert f.attrs["condition"] == "e2e_cnn" and f.attrs["export_sha256"] == ""
        assert sorted(f["episodes"][ep]) == sorted(store.cameras(ep))
        assert all("features" not in g for g in f["episodes"][ep].values())
        assert __import__("json").loads(f.attrs["outcomes"])["document"]["schema"] == "rw-outcomes"
        assert f.attrs["extra_inputs"] == "none"
    with pytest.raises(ValueError, match="OfflinePixelTransitions"):
        FD.OfflineTransitions(path)
    data = PX.OfflinePixelTransitions(path, store, p_aug=None)
    assert np.abs(data.actions.numpy()).max() <= 1 - 1e-5 + 1e-7  # official action clipping
    gen = torch.Generator().manual_seed(0)
    b = data.sample(48, gen)
    px = b["observations"]["pixels"]
    assert px.shape == (48, 32, 32, 9) and px.dtype == torch.uint8
    assert b["next_observations"]["pixels"].shape == px.shape
    spans = [(base, base + n, name) for _ep, name, base, n in data.timelines]
    idx = torch.randint(data.size, (300,), generator=gen)
    hit = set()
    for i in idx.tolist():
        o, n = data.obs_idx[i].tolist(), data.next_idx[i].tolist()
        (cam_o,) = {name for a, z, name in spans for x in o if a <= x < z}
        (cam_n,) = {name for a, z, name in spans for x in n if a <= x < z}
        assert cam_o == cam_n  # one camera for the whole history and its next observation
        assert n[-1] == o[-1] + 1 and n[:2] == o[1:]  # next history = history shifted by one step
        hit.add(cam_o)
    assert hit == set(store.cameras(ep))  # random camera per sample: every camera drawn
    # the stacked pixels are exactly the store frames of that camera, oldest first (official channel order)
    i = int(idx[0])
    one = data.batch(torch.tensor([i]))["observations"]["pixels"][0].numpy()
    a, _, cam = next(s for s in spans if s[0] <= int(data.obs_idx[i, 0]) < s[1])
    frames = store.rgb(ep, cam, (data.obs_idx[i] - a).numpy())
    np.testing.assert_array_equal(one, np.concatenate(list(frames), axis=-1))

    aug = PX.OfflinePixelTransitions(path, store, p_aug=1.0)
    ab = aug.sample(8, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    assert ab["observations"]["pixels"].shape == (8, 32, 32, 9)

    enc = data.encoder_factory()
    agent = FQLAgent.create(0, enc().out_dim, A.ACTION_DIM, {"actor_hidden_dims": (32, 32), "value_hidden_dims": (32, 32),
                                                              "batch_size": 16}, encoder=enc)
    assert agent.critic.encoder.out_dim == 512 + A.PROPRIO_DIM and agent.critic.encoder is not agent.actor_bc_flow.encoder
    before = {n: [p.detach().clone() for p in getattr(agent, n).encoder.parameters()]
              for n in ("critic", "actor_bc_flow", "actor_onestep_flow")}
    for _ in range(2):
        out = agent.update(aug.sample(16, gen, gen))
    assert all(np.isfinite(v) for v in out.values())
    for n, ps in before.items():  # trained jointly, never frozen: every network's own encoder moves
        assert any(not torch.equal(p0, p1) for p0, p1 in zip(ps, getattr(agent, n).encoder.parameters())), n
    ckpt = tmp_path / "policy.pt"
    torch.save(checkpoint_payload(agent, data, "e2e_cnn", 2), ckpt)
    policy = OfflinePolicy(ckpt)
    bad = torch.load(ckpt, map_location="cpu", weights_only=False)
    bad["meta"]["feature_norm"] = {"kind": "train_standardize"}
    torch.save(bad, tmp_path / "declared.pt")
    with pytest.raises(ValueError, match="no frozen features"):  # never a declared but unapplied standardisation
        OfflinePolicy(tmp_path / "declared.pt")
    assert policy.meta["encoder_config"]["stack_sizes"] == [16, 32, 32]
    with h5py.File(path, "r") as f:
        g = f["episodes"][ep][cam]
        k = int(np.flatnonzero(g["transition_ok"][:])[2])
        pose = A.matrix_to_pose7(g["measured_pose"][k])
        grip = A.GRIPPER_OPEN_REGISTER + A.GRIPPER_SPAN * float(g["proprio"][k, -1])
        run = g["run"][:]
    hist = [max(k - j, int(np.flatnonzero(run == run[k])[0])) for j in (2, 1, 0)]
    rgb = store.rgb(ep, cam, hist)[:, None]
    act = policy.act({"rgb": rgb}, pose, grip, noise=torch.zeros(1, A.ACTION_DIM))
    assert act["flange_pose"].shape == (7,) and 3 <= int(act["gripper_register"]) <= 230
    # same observation through the training batch path and the policy path -> same action
    proprio = data.proprio[a + k][None]
    direct = agent.sample_actions({"pixels": PX.stack_frames(torch.from_numpy(rgb[:, 0])[None]), "proprio": proprio},
                                  torch.zeros(1, A.ACTION_DIM))
    np.testing.assert_allclose(direct.numpy()[0], act["action_norm"], atol=1e-5)
    rep = replay(policy, path, ep, cam, store=store)
    with h5py.File(path, "r") as f:
        assert rep["transitions"] == int(f["episodes"][ep][cam]["transition_ok"][:].sum())
    with pytest.raises(ValueError, match="needs the training store"):
        replay(policy, path, ep, cam)
    # the pixels are pinned: changed RGB content behind the same store is refused (loader and replay)
    store._eps[ep]["cameras"][cam]["rgb"][0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="rgb_sha256"):
        PX.OfflinePixelTransitions(path, store)
    with pytest.raises(ValueError, match="rgb_sha256"):
        replay(policy, path, ep, cam, store=store)


def test_geometric_condition_dataset_needs_the_validated_calibration(tmp_path, exports):
    from s4d.rwlearn.gate import check_dataset_gate

    store = store_on_fixture(n=8)
    ep = store.episodes()[0]
    store._eps[ep]["upstream"] = {"calibrate": "cal-1"}
    path = tmp_path / "snerl.h5"
    FD.build_fql_dataset(store, "snerl", exports / "snerl.pt", path, episodes=[ep], assume_final_success=True)
    with h5py.File(path, "r") as f:
        cal = __import__("json").loads(f.attrs["calibration"])
        assert cal[ep] == {"calibrate": "cal-1", "cameras": sorted(store.cameras(ep))}
        assert "camera poses" in f.attrs["extra_inputs"] and "+2 RGB views" in f.attrs["extra_inputs"]
    base = {"tasks": {"mug-hanging": {"passed": True, **gate_identities(store)}},
            "families": {"mug-hanging": {f: {"passed": True} for f in QA_FAMILIES}}}
    with pytest.raises(RuntimeError, match="calibration hold released"):
        check_dataset_gate(path, base, steps=None)
    hold = lambda build, cams=None: {**base, "calibration_hold": {"mug-hanging": {  # noqa: E731
        "passed": True, "episodes": {ep: {"calibrate_build_id": build, "cameras": cams or sorted(store.cameras(ep))}}}}}
    res = check_dataset_gate(path, hold("cal-1"), steps=None)
    assert res["geometry_dependent"] and not res["smoke"]
    with pytest.raises(RuntimeError, match="store built from calibrate"):
        check_dataset_gate(path, hold("cal-0"), steps=None)
    with pytest.raises(RuntimeError, match="no validated calibration"):
        check_dataset_gate(path, hold("cal-1", sorted(store.cameras(ep))[:2]), steps=None)
    with pytest.raises(RuntimeError, match="calibration hold released"):  # short runs are held too (geometric)
        check_dataset_gate(path, base, steps=100)
    assert check_dataset_gate(path, base, steps=100, smoke_without_calibration=True)["smoke"]  # explicit opt-in only
