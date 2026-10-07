"""SinCro and ReViWo baselines: data adapters and camera conventions, training steps, exports, DrM end to end."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from s4d.baselines.data import ReViWoStates, SinCroWindows, reference_cameras, split_episodes
from s4d.baselines.reviwo.model import build_model
from s4d.baselines.reviwo.model import save_export as save_reviwo
from s4d.baselines.reviwo.training import ReViWoTrainConfig, compute_reviwo_loss
from s4d.baselines.sincro.encoder import build_encoder
from s4d.baselines.sincro.encoder import save_export as save_sincro
from s4d.baselines.sincro.nerf_helpers import get_rays
from s4d.data.metaworld.cameras import LOOKAT, camera_rig
from s4d.rl.agent import DrMAgent
from s4d.rl.evaluate import policy_inputs
from s4d.rl.replay import Replay, replay_iterator

REPO = Path(__file__).resolve().parents[1]
SIZE = 32
TINY_SINCRO = {
    "img_size": SIZE,
    "patch_size": 16,
    "embed_dim": 32,
    "vit_depth": 1,
    "vit_num_heads": 2,
    "vit_mlp_dim": 64,
    "decoder_depth": 1,
    "decoder_num_heads": 2,
    "decoder_mlp_dim": 64,
    "decoder_output_dim": 16,
    "netdepth": 2,
    "netwidth": 32,
    "netdepth_fine": 2,
    "netwidth_fine": 32,
    "N_rand": 64,
    "N_samples": 8,
    "N_importance": 8,
    "multires": 4,
    "multires_views": 2,
    "chunk": 1024,
    "netchunk": 4096,
    "precrop_iters": 1,
}
TINY_REVIWO = {
    "img_size": SIZE,
    "patch_size": 16,
    "fusion_style": "plus",
    "use_latent_vq": True,
    "is_latent_ae": False,
    "use_view_vq": True,
    "is_view_ae": False,
    **{
        part: {"n_layer": 1, "n_head": 2, "n_embed": 32, "dropout": 0.1, "bias": False, "mask_rate": 0}
        for part in ("view_encoder", "latent_encoder", "decoder")
    },
    "view_codebook": {"embed_dim": 8, "n_embed": 4, "beta": 0.25},
    "latent_codebook": {"embed_dim": 8, "n_embed": 8, "beta": 0.25},
}


def write_dataset(tmp_path: Path, episodes: int = 3, length: int = 12) -> tuple[Path, Path]:
    """Our HDF5 layout with the real camera rig; pixel value = frame index, so spacing is observable."""
    rig = camera_rig(SIZE, SIZE)
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as f:
        cams = f.create_group("cameras")
        for key in ("K", "c2w", "w2c", "is_train"):
            cams.create_dataset(key, data=rig[key])
        group = f.create_group("episodes")
        for e in range(episodes):
            ep = group.create_group(f"ep{e:03d}")
            ep.attrs["length"] = length
            frames = (np.arange(length, dtype=np.uint8) + 20 * e)[:, None, None, None, None]
            ep.create_dataset("rgb", data=np.broadcast_to(frames, (length, 10, SIZE, SIZE, 3)).copy())
    manifest = tmp_path / "split.json"
    names = [f"ep{e:03d}" for e in range(episodes)]
    manifest.write_text(json.dumps({"train": names[:-1], "validation": names[-1:]}))
    return path, manifest


def test_sincro_windows_shapes_range_and_two_step_spacing(tmp_path):
    path, manifest = write_dataset(tmp_path)
    train = split_episodes(manifest, "train")
    ds = SinCroWindows(path, train, sequence_length=3, temporal_stride=3, frame_spacing=2, num_views=6)
    assert len(ds) == 2 * len(range(0, 12 - 4, 3))  # starts 0, 3, 6 in each of two episodes
    sample = ds[1]
    assert sample["images"].shape == (3, SIZE, 6, SIZE, 3) and sample["K"].shape == (6, 3, 3)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    frames = (sample["images"][:, 0, 0, 0, 0] * 255).round().int().tolist()
    assert frames == [3, 5, 7], "window start 3, frames 2 simulator steps apart"
    for i in range(len(ds)):  # windows never cross episodes (episode e stores 20e + t)
        first, last = ((ds[i]["images"][[0, -1], 0, 0, 0, 0] * 255).round().int() % 20).tolist()
        assert last - first == 4 and last < 12
    capped = SinCroWindows(path, train, max_episodes=1, max_frames_per_demo=8)
    assert [s for _, s in capped.indices] == [0, 3] and {ep for ep, _ in capped.indices} == {"ep000"}


def test_reviwo_states_shapes_and_range(tmp_path):
    path, manifest = write_dataset(tmp_path)
    ds = ReViWoStates(path, split_episodes(manifest, "validation"), num_views=6)
    assert len(ds) == 12
    sample = ds[5]["images"]
    assert sample.shape == (6, 3, SIZE, SIZE)
    assert torch.allclose(sample, torch.full_like(sample, (40 + 5) / 255 * 2 - 1))
    assert len(ReViWoStates(path, split_episodes(manifest, "train"), max_frames_per_demo=5)) == 2 * 5


def project_opencv(K: np.ndarray, w2c: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    cam = w2c[:3, :3] @ x + w2c[:3, 3]
    return np.array([K[0, 0] * cam[0] / cam[2] + K[0, 2], K[1, 1] * cam[1] / cam[2] + K[1, 2]]), float(cam[2])


def test_reference_camera_convention_matches_our_projection(tmp_path):
    """Reference K/c2w through SinCro's own get_rays agree with our OpenCV cameras, and the rig's look-at point
    projects to the image centre in both conventions."""
    path, _ = write_dataset(tmp_path)
    K_ref, c2w_gl = reference_cameras(path)
    rig = camera_rig(SIZE, SIZE)
    lookat = np.array(LOOKAT)
    pixels = [(0, 0), (SIZE - 1, 0), (7, 20), (SIZE // 2, SIZE // 2), (SIZE - 1, SIZE - 1)]
    for v in range(6):
        K, w2c = rig["K"][v], rig["w2c"][v]
        uv, depth = project_opencv(K, w2c, lookat)
        assert depth > 0 and np.allclose(uv, [SIZE / 2, SIZE / 2], atol=1e-6)
        cam_gl = np.linalg.inv(c2w_gl[v].astype(np.float64)) @ np.append(lookat, 1.0)
        assert cam_gl[2] < 0, "reference cameras look down -z"
        u_ref = K_ref[v][0, 2] + K_ref[v][0, 0] * cam_gl[0] / -cam_gl[2]
        v_ref = K_ref[v][1, 2] - K_ref[v][1, 1] * cam_gl[1] / -cam_gl[2]
        assert np.allclose([u_ref, v_ref], [(SIZE - 1) / 2, (SIZE - 1) / 2], atol=1e-4)
        rays_o, rays_d = get_rays(SIZE, SIZE, torch.from_numpy(K_ref[v]), torch.from_numpy(c2w_gl[v, :3, :4]), "cpu")
        for i, j in pixels:  # a world point on the reference ray of pixel (i, j) lies at its centre (i+.5, j+.5)
            x = (rays_o[j, i] + 0.8 * rays_d[j, i]).double().numpy()
            uv, depth = project_opencv(K, w2c, x)
            assert depth > 0 and np.allclose(uv, [i + 0.5, j + 0.5], atol=1e-3), (v, i, j, uv)


def sincro_args(tmp_path: Path):
    from s4d.baselines.sincro.training import DatasetConfig, ExperimentConfig, SimpleArgs, SinCroModelConfig, TrainConfig

    model_cfg = SinCroModelConfig(**TINY_SINCRO)
    (tmp_path / "nerf").mkdir(exist_ok=True)
    args = SimpleArgs(model_cfg, TrainConfig(), DatasetConfig(batch_size=2), ExperimentConfig(str(tmp_path), "nerf"))
    return model_cfg, args


def test_sincro_training_step_is_finite_and_updates_every_network(tmp_path):
    import s4d.baselines.sincro.nerf as sincro_nerf
    from s4d.baselines.sincro.training import forward_sincro_batch

    path, manifest = write_dataset(tmp_path)
    ds = SinCroWindows(path, split_episodes(manifest, "train"))
    torch.manual_seed(0)
    np.random.seed(0)
    sincro_nerf.device = torch.device("cuda")
    model_cfg, args = sincro_args(tmp_path)
    kwargs, _, _, _, optimizer, latent = sincro_nerf.create_nerf(args, args.basedir, args.expname)
    kwargs.update(near=model_cfg.near, far=model_cfg.far)
    batch = {k: torch.stack([ds[0][k], ds[3][k]]).cuda() for k in ("images", "K", "c2w")}
    before = {n: [p.clone() for p in m.parameters()] for n, m in (("enc", latent), ("fn", kwargs["network_fn"]))}
    loss, stats = forward_sincro_batch(batch, latent, kwargs, args, model_cfg, global_step=5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert math.isfinite(loss.item()) and all(math.isfinite(v) for v in stats.values())
    assert stats["img_loss0"] > 0, "the coarse (rgb0) and fine losses are both included"
    for name, module in (("enc", latent), ("fn", kwargs["network_fn"])):
        assert any(not torch.equal(a, b) for a, b in zip(before[name], module.parameters())), name


def test_reviwo_training_step_is_finite(tmp_path):
    torch.manual_seed(0)
    cfg = ReViWoTrainConfig(**yaml.safe_load((REPO / "configs/baselines/reviwo/hammer.yaml").read_text())["train"])
    cfg.camera_num = 6
    device = torch.device(cfg.device)  # the reference config's "cuda:0"; the reference loss asserts an exact match
    model = build_model(TINY_REVIWO, SIZE).to(device)
    images = torch.rand(2, 6, 3, SIZE, SIZE) * 2 - 1
    losses = compute_reviwo_loss(model, {"images": images}, cfg, device)
    losses["loss"].backward()
    assert all(math.isfinite(v.item()) for v in losses.values())
    assert all(p.grad is not None for p in model.latent_encoder.parameters() if p.requires_grad)
    assert not model.latent_output_head.init_kmeans, "codebooks are k-means initialised on the first batch"


def test_exports_rebuild_identical_frozen_encoders(tmp_path):
    from s4d.baselines.reviwo.model import load_export as load_reviwo
    from s4d.baselines.sincro.encoder import load_export as load_sincro
    from s4d.baselines.sincro.training import SinCroModelConfig

    torch.manual_seed(0)
    model_cfg = asdict(SinCroModelConfig(**TINY_SINCRO))
    encoder = build_encoder(model_cfg, "cpu").eval()
    save_sincro(tmp_path / "sincro.pt", encoder, model_cfg, 2, 7)
    payload, loaded = load_sincro(tmp_path / "sincro.pt")
    assert payload["frame_spacing"] == 2 and payload["step"] == 7
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    model = build_model(TINY_REVIWO, SIZE)
    model.encode(torch.rand(12, 3, SIZE, SIZE))  # k-means initialisation, as in training
    model.eval()
    save_reviwo(tmp_path / "reviwo.pt", model, TINY_REVIWO, SIZE, 3)
    _, reloaded = load_reviwo(tmp_path / "reviwo.pt")
    x = torch.rand(4, 3, SIZE, SIZE) * 2 - 1
    assert torch.equal(model.encode(x)[2], reloaded.encode(x)[2])
    assert not any(getattr(m, "init_kmeans", False) for m in reloaded.modules())


def drm_cfg(encoder_type: str, export: Path) -> dict:
    base = yaml.safe_load((REPO / "configs/rl/base.yaml").read_text())
    return {
        "env": {"frame_stack": 3, "image_size": SIZE},
        "vision": {"encoder_type": encoder_type, "export_path": str(export)},
        "agent": {**base["agent"], "feature_dim": 8, "hidden_dim": 16, "dormant_perturb_interval": 4},
    }


@pytest.mark.parametrize("encoder_type", ["sincro", "reviwo"])
def test_drm_end_to_end_with_frozen_baseline_encoders(tmp_path, encoder_type):
    from s4d.baselines.sincro.training import SinCroModelConfig

    device = torch.device("cuda")
    torch.manual_seed(0)
    if encoder_type == "sincro":
        model_cfg = asdict(SinCroModelConfig(**TINY_SINCRO))
        save_sincro(tmp_path / "export.pt", build_encoder(model_cfg, "cpu"), model_cfg, 2, 0)
    else:
        model = build_model(TINY_REVIWO, SIZE)
        model.encode(torch.rand(12, 3, SIZE, SIZE))
        save_reviwo(tmp_path / "export.pt", model, TINY_REVIWO, SIZE, 0)
    agent = DrMAgent(drm_cfg(encoder_type, tmp_path / "export.pt"), 4, 4, device)
    enc = agent.encoder
    assert not agent.augment_pixels and agent.encoder.replay_atom_is_feature
    assert not any(p.requires_grad for p in enc.backbone.parameters())
    assert enc.replay_atom_is_stack_feature == (encoder_type == "sincro")
    replay = Replay(None, enc.replay_atom_shape, np.float16, 4, 4, 1000, enc.replay_atom_frame_stack, 10)
    rng = np.random.default_rng(0)

    def atom(stack):
        features = policy_inputs(agent, stack)[0]
        return (features if enc.replay_atom_is_stack_feature else features[-1]).cpu().numpy()

    for _ in range(3):
        replay.add_initial(atom(rng.integers(0, 255, (1, 9, SIZE, SIZE), dtype=np.uint8)), rng.normal(size=4))
        for t in range(12):
            stack = rng.integers(0, 255, (1, 9, SIZE, SIZE), dtype=np.uint8)
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, atom(stack), rng.normal(size=4), t == 11)
    frozen = copy.deepcopy(enc.backbone.state_dict())
    actor = [p.clone() for p in agent.actor.parameters()]
    batches = replay_iterator(replay, 16, 0.97, seed=0, device=device)
    perturbed = []
    for step in range(2, 10, 2):
        metrics = agent.update(batches, step)
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        perturbed.append("perturb_factor" in metrics)
    assert perturbed == [False, True, False, True]
    assert any(not torch.equal(a, b) for a, b in zip(actor, agent.actor.parameters()))
    for key, value in enc.backbone.state_dict().items():
        assert torch.equal(value, frozen[key]), f"frozen encoder changed: {key}"
    action = agent.act(policy_inputs(agent, np.zeros((2, 9, SIZE, SIZE), np.uint8)), np.zeros((2, 4)), 5000, True)
    assert action.shape == (2, 4)


def run(script: str, *args: str) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]}
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / script), *args], env=env, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]


def tiny_config(tmp_path: Path, name: str, path: Path, manifest: Path) -> Path:
    cfg = yaml.safe_load((REPO / f"configs/baselines/{name}/hammer.yaml").read_text())
    cfg["dataset"].update(hdf5_path=str(path), num_workers=0, batch_size=2)
    cfg["data_interface"]["split_manifest"] = str(manifest)
    if name == "sincro":
        cfg["model"].update(TINY_SINCRO)
        cfg["train"].update(max_global_steps=3, eval_every=2, save_every=2, i_print=1)
    else:
        cfg["reviwo"] = TINY_REVIWO
        cfg["train"].update(max_global_steps=3, eval_every=2, save_every=2, log_every=1)
    out = tmp_path / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


@pytest.mark.parametrize("name", ["sincro", "reviwo"])
def test_pretraining_entry_points_run_export_and_resume(tmp_path, name):
    path, manifest = write_dataset(tmp_path)
    config = tiny_config(tmp_path, name, path, manifest)
    root = tmp_path / "runs"  # conftest keeps tmp_path inside the repository, as the scripts require
    run(f"scripts/train_{name}.py", "--config", str(config), "--name", name, "--output-root", str(root), "--no-wandb")
    run_dir = root / name
    assert json.loads((run_dir / "completion.json").read_text())["steps"] == 3
    assert any((run_dir / "eval").iterdir()), "validation output written"
    export = torch.load(run_dir / "encoder.pt", map_location="cpu", weights_only=True)
    assert export["step"] == 3
    steps = "train.max_global_steps=5"
    run(
        f"scripts/train_{name}.py",
        "--config",
        str(config),
        "--name",
        name,
        "--output-root",
        str(root),
        "--no-wandb",
        "--set",
        steps,
    )
    assert json.loads((run_dir / "completion.json").read_text())["steps"] == 5, "resumed from step 3"
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert {3, 4} <= {r["step"] for r in rows}
