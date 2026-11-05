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
from s4d.rwlearn.inference import OfflinePolicy, checkpoint_payload, replay
from s4d.rwlearn.store import MemoryStore

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
    report = {"tasks": {"mug-hanging": {"passed": True, "store_episodes": [ep]}}}
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
