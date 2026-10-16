"""Baseline representation evaluation (``s4d/baselines/representation.py``): our evaluator's metrics on the RL-time
features of frozen baseline exports. CPU only; synthetic data and randomly initialised encoders, not evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from s4d.baselines import representation as R
from s4d.config import load_config
from s4d.data.metaworld import cameras as C
from s4d.diag.local_log import RunLogger
from s4d.model.decoder import DecoderConfig, GaussianDecoder, GroupConfig
from s4d.model.encoder import Encoder, EncoderConfig
from s4d.rl import encoders as RL
from s4d.rl.env import HELDOUT_PATHS
from s4d.rl.evaluate import nearest_training_camera
from s4d.train import evaluate as E
from s4d.train.loop import Model

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
EPISODES, LENGTH = 66, 6  # 64 distinct episodes for the retrieval protocol, 3 validation episodes at train_ratio 0.96


# --------------------------------------------------------------------------------------------- synthetic data
def write_task(root: Path, task: str = "hammer", size: int = 16, episodes: int = EPISODES, seed: int = 0) -> dict:
    """Our HDF5 layout with the real 10-camera rig plus the near/trajectory held-out sets of the validation
    episodes (real set cameras and bases); random pixels and a random-walk probe state."""
    rng = np.random.default_rng(seed)
    rig = C.camera_rig(size, size)
    names = [f"ep{e:03d}" for e in range(episodes)]
    with h5py.File(root / f"{task}.hdf5", "w") as f:
        f.attrs.update({"task": task, "depth_unit_m": 1e-4, "dt_seconds": 0.0125, "background_body": 0,
                        "body_names": json.dumps(["world", "box"])})
        cams = f.create_group("cameras")
        for key in ("K", "c2w", "w2c", "is_train"):
            cams.create_dataset(key, data=rig[key])
        for name in names:
            g = f.create_group(f"episodes/{name}")
            g.attrs["length"] = LENGTH
            obs = np.zeros((LENGTH, 39), dtype=np.float32)
            obs[:, :11] = np.cumsum(rng.normal(size=(LENGTH, 11)), 0)
            g.create_dataset("rgb", data=rng.integers(0, 256, (LENGTH, 10, size, size, 3), dtype=np.uint8))
            g.create_dataset("depth", data=rng.integers(5000, 15000, (LENGTH, 10, size, size), dtype=np.uint16))
            g.create_dataset("body_id", data=np.ones((LENGTH, 6, size, size), dtype=np.uint16))
            g.create_dataset("xpos", data=np.zeros((LENGTH, 2, 3), dtype=np.float32))
            g.create_dataset("xquat", data=np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (LENGTH, 2, 1)))
            g.create_dataset("obs", data=obs)
    train_ratio = 0.96 if episodes >= 25 else 0.5
    n_val = episodes - max(1, min(episodes - 1, round(episodes * train_ratio)))
    import random

    shuffled = names[:]
    random.Random(0).shuffle(shuffled)
    validation = sorted(shuffled[len(shuffled) - n_val:])
    sets = [cam for name in C.HELDOUT_SETS for cam in C.heldout_camera_set(name)]
    held_rig = C.orbit_rig([(c["azimuth"], c["elevation"], c["distance"]) for c in sets], size, size)
    (root / "heldout_sets").mkdir(exist_ok=True)
    with h5py.File(root / "heldout_sets" / f"{task}.hdf5", "w") as f:
        f.attrs.update({"complete": True, "depth_unit_m": 1e-4, "sets": json.dumps(list(C.HELDOUT_SETS))})
        cams = f.create_group("cameras")
        cams.create_dataset("names", data=np.asarray([c["name"] for c in sets], dtype=h5py.string_dtype()))
        cams.create_dataset("set", data=np.asarray([c["set"] for c in sets], dtype=h5py.string_dtype()))
        cams.create_dataset("base", data=np.asarray([c["base"] for c in sets]))
        for key in ("K", "c2w", "w2c"):
            cams.create_dataset(key, data=held_rig[key])
        for name in validation:
            g = f.create_group(f"episodes/{name}")
            g.attrs["length"] = LENGTH
            g.create_dataset("rgb", data=rng.integers(0, 256, (LENGTH, len(sets), size, size, 3), dtype=np.uint8))
            g.create_dataset("depth", data=rng.integers(5000, 15000, (LENGTH, len(sets), size, size), dtype=np.uint16))
    return {"validation": validation, "bases": [c["base"] for c in sets], "train_ratio": train_ratio}


def our_config(root: Path, task: str = "hammer", train_ratio: float = 0.96, batch_size: int = 4) -> dict:
    cfg = load_config([REPO / "configs/metaworld/base.yaml"])
    cfg["wandb"]["enabled"] = False
    cfg["data"].update(root=str(root), task=task, train_ratio=train_ratio, strides=[2], val_strides=[2])
    cfg["train"].update(num_workers=0, batch_size=batch_size)
    cfg["eval"]["heldout_sets"] = True
    return cfg


def first_batch(root: Path, cfg: dict) -> dict:
    from train import build_data

    _, val_loaders, _, _ = build_data(cfg)
    return next(iter(val_loaders[2]))


# --------------------------------------------------------------------------------------------- fake backbones
def frame_feature(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) in [0,1] -> (N,12): per-sample exact, so batch composition cannot change a value."""
    return F.adaptive_avg_pool2d(x, 2).flatten(1)


class FakeFrameEncoder(torch.nn.Module):
    is_trainable = False

    def __init__(self, cfg):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return frame_feature(x)


class FakeMultiViewEncoder(FakeFrameEncoder):
    """Per view: pooled RGB, mean depth and the camera centre, concatenated in input order."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.observation = {"views": int(cfg["env"]["views"]), "depth": True, "cameras": True}

    def forward(self, obs):
        n, v = obs["rgb"].shape[:2]
        rgb = frame_feature(obs["rgb"].flatten(0, 1)).view(n, v, -1)
        return torch.cat((rgb, obs["depth"].mean((-1, -2))[..., None], obs["c2w"][..., :3, 3]), -1).flatten(1)


def fake_adapter(monkeypatch, backbone, size: int = 16, **env) -> RL.VisionEncoderAdapter:
    monkeypatch.setitem(RL.BACKBONES, "fake", backbone)
    return RL.VisionEncoderAdapter({"env": {"frame_stack": 3, "image_size": size, **env},
                                    "vision": {"encoder_type": "fake"}, "agent": {"feature_dim": 8}})


# --------------------------------------------------------------------------------------------- metric identity
def _model():
    encoder = Encoder(EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4,
                                    num_slots=1, slot_dim=8, drop_path=0))
    group = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
    return Model(encoder, GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=group, dynamic=group)))


def _synthetic_render(gs, xyz, w2c, K, height, width, near, far):
    b, times = xyz.shape[:2]
    shape = (b, times, w2c.shape[1], 1, height, width)
    return {"rgb": torch.full((b, times, w2c.shape[1], 3, height, width), 0.5), "depth": torch.ones(shape),
            "alpha": torch.full(shape, 0.9)}


def our_states(images: torch.Tensor) -> torch.Tensor:
    """Our evaluator's state of every (window, camera) for the fake backbone: the fp16 per-frame features of the three
    frames concatenated, (B,V,1,3*12)."""
    B, T, V = images.shape[:3]
    f = frame_feature(images.float().div(255.0).flatten(0, 2)).half().float()
    return f.view(B, T, V, -1).transpose(1, 2).reshape(B, V, 1, -1)


def test_metrics_and_protocol_equal_our_evaluator_on_the_same_states(tmp_path, monkeypatch):
    """Our evaluator (model forward faked so that its states are the fake backbone's stacked features) and the
    baseline evaluator (the fake backbone through the RL adapter) give identical retrieval, probe and state metrics."""
    from train import build_data

    write_task(tmp_path)
    cfg = our_config(tmp_path)
    _, val_loaders, probe_loaders, n_train = build_data(cfg)
    assert n_train == 6

    def fake_forward(m, batch, cfg_, step, *, source, mask_ratio, return_renders=False):
        B = batch["images"].shape[0]
        gs = m.decoder(torch.zeros(B, 1, 8))
        return {"gs": gs, "slots": our_states(batch["images"]), "losses": {"total": torch.tensor(1.0)},
                "metrics": {"psnr": torch.tensor(20.0)}, "source": torch.zeros(B, dtype=torch.long),
                **_synthetic_render(gs, gs.xyz_sequence(), batch["w2c"], batch["K"], 16, 16, 0.05, 3.0),
                "pred_disp": batch["motion3d"].clone(), "pairs": ((0, 1, 0), (1, 2, 1), (0, 2, 0)),
                "used": torch.ones(gs.opacity.shape, dtype=torch.bool)}

    monkeypatch.setattr(E, "forward_losses", fake_forward)
    monkeypatch.setattr(E, "render_rgbd", _synthetic_render)
    monkeypatch.setattr(E, "encode_states", lambda m, x: our_states(x))
    monkeypatch.setattr(E.Evaluator, "panels", lambda *a, **k: None)
    ours = E.Evaluator(cfg, val_loaders, probe_loaders, RunLogger(tmp_path / "run"), torch.device("cpu"), n_train)(
        _model(), 0, full=True
    )
    adapter = fake_adapter(monkeypatch, FakeFrameEncoder)
    _, val_loaders, probe_loaders, _ = build_data(cfg)  # fresh loaders: the probe draw advances their generator
    evaluator = R.BaselineEvaluator(cfg, val_loaders, probe_loaders, torch.device("cpu"), n_train, chunk=7)
    theirs, num_batches = evaluator(adapter)
    assert num_batches == {2: len(val_loaders[2])}
    expected = {"metric/retrieval_top1_train", "metric/retrieval_top1_heldout", "metric/effective_rank_retrieval",
                "metric/retrieval_top1_train_val", "metric/effective_rank_val", "metric/state_std_mean_val"}
    expected |= {f"metric/retrieval_top1_{g}_val" for g in ("heldout", "near", "traj")}
    expected |= {f"metric/r2_{t}_val_{g}" for t in ("hand_pos", "obj_pos", "hand_vel")
                 for g in ("traincams", "heldout", "near", "traj")}
    assert {f"{k}@s2" for k in expected} <= set(theirs)
    assert theirs["metric/retrieval_protocol_available@s2"] is True
    for key, value in theirs.items():
        assert key in ours, key
        assert ours[key] == value, key
    assert not any(k.startswith(("metric/psnr", "metric/cd_", "metric/oracle", "loss/")) for k in theirs)


def test_feature_choices_select_frames_of_the_cached_history(tmp_path, monkeypatch):
    write_task(tmp_path, episodes=4)
    cfg = our_config(tmp_path, train_ratio=0.5)
    batch = first_batch(tmp_path, cfg)
    adapter = fake_adapter(monkeypatch, FakeFrameEncoder)
    B, T, V = batch["images"].shape[:3]
    full = our_states(batch["images"])[:, :, 0].view(B, V, T, -1)
    for feature, expected in (("stack", full.flatten(2)), ("newest", full[:, :, -1]), ("middle", full[:, :, 1])):
        evaluator = R.BaselineEvaluator(cfg, {}, None, torch.device("cpu"), 6, feature=feature)
        assert torch.equal(evaluator.states(adapter, batch), expected), feature
        held = evaluator.states(adapter, batch, "near")  # single camera: column j is held-out camera j alone
        expected_near = our_states(batch["near_images"])[:, :, 0].view(B, -1, T, 12)
        assert held.shape[:2] == (B, 12)
        if feature == "stack":
            assert torch.equal(held, expected_near.flatten(2))
    with pytest.raises(ValueError, match="unknown feature"):
        R.BaselineEvaluator(cfg, {}, None, torch.device("cpu"), 6, feature="mean")


# --------------------------------------------------------------------------------------------- multi-view columns
def test_multiview_columns_follow_the_rl_evaluation_design(tmp_path, monkeypatch):
    info = write_task(tmp_path, episodes=4)
    cfg = our_config(tmp_path, train_ratio=0.5)
    batch = first_batch(tmp_path, cfg)
    assert R.input_columns(6, 3).tolist() == [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 0], [4, 5, 1], [5, 0, 2]]
    assert R.input_columns(6, 1).tolist() == [[i] for i in range(6)]
    assert R.input_columns(6, 3, torch.tensor([4, 0])).tolist() == [[6, 5, 1], [7, 1, 3]]
    # nearest training camera: RL's rule for the far cameras, and the stored base camera of every near/trajectory camera
    far = R.nearest_training_cameras(batch["c2w"][0], torch.linalg.inv(batch["eval_w2c"][0]))
    assert far.tolist() == [nearest_training_camera(HELDOUT_PATHS[f"eval{j}"]) for j in range(4)] == [4, 5, 2, 3]
    held = torch.cat((batch["near_c2w"][0], batch["traj_c2w"][0]))
    assert R.nearest_training_cameras(batch["c2w"][0], held).tolist() == info["bases"]

    adapter = fake_adapter(monkeypatch, FakeMultiViewEncoder, views=3, depth=True, cameras=True)
    evaluator = R.BaselineEvaluator(cfg, {}, None, torch.device("cpu"), 6, chunk=5)
    B, T = batch["images"].shape[:2]

    def direct(cams: list[tuple[str, int]]) -> torch.Tensor:
        """Stacked features of one input camera list, straight from the fake backbone."""
        def get(group, key):
            return batch[f"{group}_{key}" if group else key]
        frames = []
        for t in range(T):
            obs = {"rgb": torch.stack([get(g, "images")[:, t, j] for g, j in cams], 1).float() / 255,
                   "depth": torch.stack([get(g, "depth")[:, t, j, 0] for g, j in cams], 1),
                   "c2w": torch.stack([get(g, "c2w")[:, j] if g != "eval" else torch.linalg.inv(batch["eval_w2c"][:, j])
                                       for g, j in cams], 1)}
            frames.append(FakeMultiViewEncoder.forward(adapter.backbone, obs).half().float())
        return torch.cat(frames, 1)

    train = evaluator.states(adapter, batch)
    assert train.shape == (B, 6, 3 * 3 * 16)
    for i, triple in enumerate(R.input_columns(6, 3).tolist()):
        assert torch.equal(train[:, i], direct([("", k) for k in triple])), i
    eval_states = evaluator.states(adapter, batch, "eval")
    assert eval_states.shape == (B, 4, 3 * 3 * 16)
    for j, base in enumerate(far.tolist()):
        triple = R.input_columns(6, 3)[base].tolist()
        assert torch.allclose(eval_states[:, j], direct([("eval", j), ("", triple[1]), ("", triple[2])]), atol=1e-3)
    near = evaluator.states(adapter, batch, "near")
    for j in (0, 5, 11):
        triple = R.input_columns(6, 3)[info["bases"][j]].tolist()
        assert torch.equal(near[:, j], direct([("near", j), ("", triple[1]), ("", triple[2])])), j


# --------------------------------------------------------------------------------------------- real exports
def test_dynamo_export_through_the_rl_adapter(tmp_path):
    from s4d.baselines.dynamo.model import build_encoder, encoder_kwargs, save_export

    write_task(tmp_path, episodes=4)
    batch = first_batch(tmp_path, our_config(tmp_path, train_ratio=0.5))
    cfg = yaml.safe_load((REPO / "configs/baselines/dynamo/hammer.yaml").read_text())
    torch.manual_seed(0)
    encoder = build_encoder(encoder_kwargs(cfg)).eval()
    save_export(tmp_path / "encoder_init.pt", encoder, encoder_kwargs(cfg), 2, 16, 0, True)
    adapter, rl_cfg = R.load_adapter("dynamo", "hammer", tmp_path / "encoder_init.pt")
    assert rl_cfg["vision"]["export_path"] == str(tmp_path / "encoder_init.pt") and adapter.observation is None
    assert adapter.backbone.random_init and R.input_views(adapter) == 1
    evaluator = R.BaselineEvaluator(our_config(tmp_path, train_ratio=0.5), {}, None, torch.device("cpu"), 6)
    states = evaluator.states(adapter, batch, "eval")
    B, T = batch["images"].shape[:2]
    assert states.shape == (B, 4, 3 * 512)
    with torch.no_grad():
        expected = encoder(batch["eval_images"][:, :, 2].flatten(0, 1).float() / 255).view(B, T * 512)
    torch.testing.assert_close(states[:, 2], expected, atol=2e-2, rtol=2e-3)  # fp16 replay cache
    with pytest.raises(ValueError, match="not a frozen baseline"):
        R.load_adapter("cnn", "hammer", tmp_path / "encoder_init.pt")


@pytest.mark.parametrize("encoder", ["mvp3d_single", "mvp3d_multiview"])
def test_mvp3d_conditions_through_the_rl_adapter(tmp_path, encoder):
    from s4d.baselines.mvp3d.model import SCENE_BOUNDS, MVP3DConfig, build_encoder, save_export
    from s4d.baselines.mvp3d.renderer import render_rgbd

    tiny = MVP3DConfig(img_size=24, img_patch_size=6, im_channels=16, attn_dim=32, attn_heads=2, attn_dim_head=16,
                       depth=4, decoder_dim=32, decoder_heads=2, decoder_dim_head=16)
    write_task(tmp_path, episodes=4, size=32)
    cfg = our_config(tmp_path, train_ratio=0.5, batch_size=2)
    batch = first_batch(tmp_path, cfg)
    torch.manual_seed(0)
    enc = build_encoder(tiny).eval()
    save_export(tmp_path / "encoder.pt", enc, asdict(tiny), {"bounds": list(SCENE_BOUNDS), "radius": 0.012}, 3)
    adapter, _ = R.load_adapter(encoder, "hammer", tmp_path / "encoder.pt")
    views = 3 if encoder == "mvp3d_multiview" else 1
    assert adapter.observation == {"views": views, "depth": True, "cameras": True}
    evaluator = R.BaselineEvaluator(cfg, {}, None, torch.device("cpu"), 6, chunk=4)
    states = evaluator.states(adapter, batch)
    B, T = batch["images"].shape[:2]
    assert states.shape == (B, 6, T * 5 * tiny.attn_dim)
    cams = R.input_columns(6, views)[4].tolist()
    t = 1
    with torch.no_grad():
        images = render_rgbd(batch["images"][:, t, cams].float() / 255, batch["depth"][:, t, cams, 0], batch["K"][:, cams],
                             batch["c2w"][:, cams], SCENE_BOUNDS, tiny.img_size, 0.012)
        expected = enc.global_feature(enc(images))
    D = 5 * tiny.attn_dim
    torch.testing.assert_close(states[:, 4, t * D : (t + 1) * D], expected, atol=2e-2, rtol=2e-3)
    assert evaluator.states(adapter, batch, "traj").shape == (B, 12, T * D)


def test_snerl_export_through_the_rl_adapter(tmp_path):
    from s4d.baselines.snerl import model as M
    from s4d.baselines.snerl.data import normalise_c2w

    write_task(tmp_path, episodes=2, size=128)
    cfg = our_config(tmp_path, train_ratio=0.5, batch_size=1)
    batch = first_batch(tmp_path, cfg)
    torch.manual_seed(0)
    M.save_export(tmp_path / "encoder.pt", M.build_encoder(), 0, 3, random_init=True)
    adapter, _ = R.load_adapter("snerl", "hammer", tmp_path / "encoder.pt")
    assert adapter.observation == {"views": 3, "depth": False, "cameras": True}
    evaluator = R.BaselineEvaluator(cfg, {}, None, torch.device("cpu"), 6, feature="newest", chunk=3)
    states = evaluator.states(adapter, batch, "eval")
    assert states.shape == (1, 4, 63)
    base = int(R.nearest_training_cameras(batch["c2w"][0], torch.linalg.inv(batch["eval_w2c"][0]))[0])
    triple = R.input_columns(6, 3)[base].tolist()
    rgb = torch.cat((batch["eval_images"][:, -1, :1], batch["images"][:, -1, triple[1:]]), 1).float() / 255
    c2w = torch.cat((torch.linalg.inv(batch["eval_w2c"][:, :1]), batch["c2w"][:, triple[1:]]), 1)
    with torch.no_grad():
        expected = M.encode_batch(adapter.backbone.encoder, rgb, normalise_c2w(c2w, adapter.backbone.lookat,
                                                                                adapter.backbone.scale).flatten(-2))
    torch.testing.assert_close(states[:, 0], expected, atol=2e-2, rtol=2e-3)


# --------------------------------------------------------------------------------------------- entry point
def test_entry_point_writes_both_arms_with_provenance(tmp_path, monkeypatch):
    from s4d.baselines.dynamo.model import build_encoder, encoder_kwargs, save_export

    info = write_task(tmp_path)
    cfg = our_config(tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    dyn = yaml.safe_load((REPO / "configs/baselines/dynamo/hammer.yaml").read_text())
    torch.manual_seed(0)
    encoder = build_encoder(encoder_kwargs(dyn)).eval()
    save_export(tmp_path / "encoder_init.pt", encoder, encoder_kwargs(dyn), 2, 16, 0, True)
    with torch.no_grad():
        for p in encoder.parameters():
            p.mul_(0.5)
    save_export(tmp_path / "encoder.pt", encoder, encoder_kwargs(dyn), 2, 16, 7, False)
    spec = importlib.util.spec_from_file_location("eval_baseline_representation",
                                                  REPO / "scripts/eval_baseline_representation.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["eval_baseline_representation.py", "--task", "hammer", "--encoder", "dynamo",
                                      "--export", str(tmp_path / "encoder.pt"), "--output", str(out), "--config",
                                      str(tmp_path / "config.yaml"), "--device", "cpu", "--cpu-threads", "2"])
    script.main()
    combined = json.loads((out / "summary.json").read_text())
    assert set(combined) == {"dynamo/pretrained", "dynamo/random_init"}
    pre, init = (json.loads((out / "dynamo" / arm / "summary.json").read_text()) for arm in script.ARMS)
    assert pre["provenance"]["export_step"] == 7 and not pre["provenance"]["random_init"]
    assert init["provenance"]["random_init"] and init["provenance"]["export"].endswith("encoder_init.pt")
    split = pre["provenance"]["split"]
    assert split["validation_episodes"] == info["validation"] and split["val_windows"] == {"2": 3 * (LENGTH - 4)}
    assert len(pre["provenance"]["export_sha256"]) == 64 and pre["provenance"]["git_commit"]
    for record in (pre, init):
        assert record["metric/retrieval_protocol_available@s2"] is True
        assert np.isfinite(record["metric/r2_hand_pos_val_near@s2"])
    assert pre["metric/effective_rank_val@s2"] != init["metric/effective_rank_val@s2"]
    assert (out / "provenance.json").is_file()

