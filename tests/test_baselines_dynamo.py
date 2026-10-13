"""DynaMo baseline: the pinned official code against our adapters.

Official definitions that cannot be imported (``datasets/core.py`` and ``train.py`` import hydra, accelerate and
wandb at module level) are taken verbatim from the pinned submodule with ``ast`` and executed; nothing of theirs is
rewritten. ``accelerate`` is not installed: where ``train.py`` calls it, ``SingleProcessAccelerator`` provides the
calls with their meaning for the official single-process fp32 launch (``python3 train.py``, no accelerate config).
"""

from __future__ import annotations

import ast
import contextlib
import copy
import difflib
import json
import math
import os
import subprocess
import sys
import textwrap
import types
import typing
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from s4d.baselines.data import split_episodes
from s4d.baselines.dynamo import official
from s4d.baselines.dynamo.data import DynaMoWindows, official_slices, to_official_obs
from s4d.baselines.dynamo.model import build_encoder, build_modules, encoder_kwargs, load_export, save_export
from s4d.baselines.dynamo.training import EpochLog, EpochSampler, epoch_lr, evaluate, start_epoch, train_step
from s4d.data.metaworld.cameras import camera_rig

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/dynamo"
SIZE = 32
TASKS = ["hammer", "pick-place", "bin-picking", "coffee-push", "door-open", "peg-insert-side", "peg-unplug-side",
         "stick-push"]
TYPING = {name: getattr(typing, name) for name in ("Any", "Callable", "Dict", "List", "Optional", "Sequence")}
TINY = {"n_layer": 1, "n_head": 2, "n_embd": 16, "dropout": 0.0}


def task_config(task: str = "hammer") -> dict:
    return yaml.safe_load((REPO / f"configs/baselines/dynamo/{task}.yaml").read_text())


def tiny_config(**train) -> dict:
    """The hammer config with small transformers (the encoder is the fixed-size official ResNet-18)."""
    cfg = task_config()
    cfg["projector"].update(TINY, output_dim=4)
    cfg["ssl"].update(TINY)
    cfg["train"].update(train)
    return cfg


def write_dataset(tmp_path: Path, lengths=(14, 12, 16), seed: int = 0) -> tuple[Path, Path]:
    """Our HDF5 layout (real camera rig, 6 training + 4 held-out cameras); random pixels, except that pixel (0, 0)
    stores (frame, camera, episode) so windows can be traced."""
    rng = np.random.default_rng(seed)
    rig = camera_rig(SIZE, SIZE)
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as f:
        cams = f.create_group("cameras")
        for key in ("K", "c2w", "w2c", "is_train"):
            cams.create_dataset(key, data=rig[key])
        group = f.create_group("episodes")
        for e, length in enumerate(lengths):
            rgb = rng.integers(0, 256, (length, 10, SIZE, SIZE, 3), dtype=np.uint8)
            rgb[:, :, 0, 0, 0] = np.arange(length)[:, None]
            rgb[:, :, 0, 0, 1] = np.arange(10)[None, :]
            rgb[:, :, 0, 0, 2] = e
            ep = group.create_group(f"ep{e:03d}")
            ep.attrs["length"] = length
            ep.create_dataset("rgb", data=rgb, chunks=(1, 1, SIZE, SIZE, 3), compression="gzip")
    manifest = tmp_path / "split.json"
    names = [f"ep{e:03d}" for e in range(len(lengths))]
    manifest.write_text(json.dumps({"train": names[:-1], "validation": names[-1:]}))
    return path, manifest


def official_source(relative: str, qualname: str) -> str:
    """Verbatim source of a top-level definition or a ``Class.method`` of a pinned upstream file."""
    source = (UPSTREAM / relative).read_text()
    nodes, node = ast.parse(source).body, None
    for part in qualname.split("."):
        node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
        nodes = node.body
    return textwrap.dedent(ast.get_source_segment(source, node, padded=True))


def official_namespace(*definitions: tuple[str, str], **names) -> dict:
    namespace = {"__name__": "dynamo_official_excerpt", **names}
    for relative, qualname in definitions:
        exec(compile(official_source(relative, qualname), f"{UPSTREAM / relative}::{qualname}", "exec"), namespace)
    return namespace


class SingleProcessAccelerator:
    """What train.py's ``accelerate.Accelerator`` calls do with one process and mixed_precision "no"."""

    is_main_process = True

    def autocast(self):
        return contextlib.nullcontext()

    def backward(self, loss, **kwargs):
        loss.backward(**kwargs)

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def gather_for_metrics(self, x):
        return x


def official_trainer_class() -> type:
    """``train.py::Trainer`` with its verbatim ``train``, ``eval``, ``adjust_lr`` and ``log_append``."""
    import einops
    import tqdm

    eval_mode = official_namespace(("utils/inference.py", "eval_mode"), torch=torch)["eval_mode"]
    reduce_dict = official_namespace(("utils/__init__.py", "reduce_dict"), **TYPING)["reduce_dict"]
    utils = types.SimpleNamespace(inference=types.SimpleNamespace(eval_mode=eval_mode), reduce_dict=reduce_dict)
    ns = official_namespace(
        *(("train.py", f"Trainer.{m}") for m in ("train", "eval", "adjust_lr", "log_append")),
        tqdm=tqdm, torch=torch, np=np, einops=einops, utils=utils,
    )
    return type("OfficialTrainer", (), {m: ns[m] for m in ("train", "eval", "adjust_lr", "log_append")})


# --------------------------------------------------------------------------------------------- third-party code


def test_patched_tree_is_the_pinned_commit_minus_accelerate_only():
    status = subprocess.run(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True, check=True).stdout
    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert status == "" and head == "62874039e5d3fc663b0fccd2083b2dd2c8d9935e"
    tree = official.tree()
    assert tree != UPSTREAM, "the patched copy is used, not the checkout"
    changed = []
    for path in sorted(p for p in UPSTREAM.rglob("*") if p.is_file() and ".git" not in p.relative_to(UPSTREAM).parts):
        relative = path.relative_to(UPSTREAM)
        if (tree / relative).read_bytes() != path.read_bytes():
            changed.append(str(relative))
    assert changed == ["models/ssl/dynamo.py"]
    before = (UPSTREAM / "models/ssl/dynamo.py").read_text().splitlines()
    after = (tree / "models/ssl/dynamo.py").read_text().splitlines()
    diff = [line for line in difflib.ndiff(before, after) if line[:1] in "+-"]
    assert all(line.startswith("- ") for line in diff), "the patch only removes lines"
    removed = "\n".join(line[2:] for line in diff)
    assert "Accelerator" not in "\n".join(after) and "accelerate" not in "\n".join(after)
    assert removed.count("accelerator") == 2 and "from accelerate import Accelerator" in removed
    module = official.module("ssl.dynamo")
    assert Path(module.__file__).resolve() == (tree / "models/ssl/dynamo.py").resolve()
    for name, loaded in list(sys.modules.items()):  # upstream modules exist only under the unique package name
        location = getattr(loaded, "__file__", None) or ""
        if location.startswith((str(tree), str(UPSTREAM))):
            assert name.startswith(official.PACKAGE + "."), name


# ------------------------------------------------------------------------------------------------- data adapter


def test_windows_match_the_official_trajectory_slicer(tmp_path):
    """Our windows are the official ``TrajectorySlicerDataset`` (frame_step path) over per-camera trajectories."""
    path, manifest = write_dataset(tmp_path, lengths=(14, 9, 12, 16))
    episodes = split_episodes(manifest, "train")
    ns = official_namespace(
        ("datasets/core.py", "TrajectoryDataset"),
        ("datasets/core.py", "TrajectorySlicerDataset"),
        abc=__import__("abc"), np=np, torch=torch, Dataset=torch.utils.data.Dataset, utils=None, **TYPING,
    )

    class Trajectories(ns["TrajectoryDataset"]):  # the official interface over our file: obs / 255.0 on the CPU
        def __init__(self):
            self.file = h5py.File(path, "r")
            self.items = [(ep, cam) for ep in episodes for cam in range(6)]

        def __len__(self):
            return len(self.items)

        def get_seq_length(self, idx):
            return int(self.file["episodes"][self.items[idx][0]].attrs["length"])

        def get_frames(self, idx, frames):
            ep, cam = self.items[idx]
            obs = torch.from_numpy(self.file["episodes"][ep]["rgb"][list(frames), cam]).permute(0, 3, 1, 2)
            return obs[:, None] / 255.0, torch.zeros(len(frames), 4), torch.ones(len(frames))

        def __getitem__(self, idx):
            return self.get_frames(idx, range(self.get_seq_length(idx)))

    reference = ns["TrajectorySlicerDataset"](Trajectories(), window=5, frame_step=2)
    ours = DynaMoWindows(path, episodes, window=5, frame_step=2)
    assert ours.slices == reference.slices and len(ours) == len(reference) == 6 * (5 + 0 + 3)
    for index in range(len(ours)):
        obs, _, mask = reference[index]
        sample = ours[index]
        assert sample.dtype == torch.uint8 and sample.shape == (5, 1, 3, SIZE, SIZE) and bool(mask.all())
        assert torch.equal(to_official_obs(sample[None], "cpu")[0], obs), index
    for lengths, window, step in (([3, 9, 10, 40], 5, 2), ([5, 6, 7], 5, 1), ([2, 8], 2, 3)):
        fake = type("L", (), {"__len__": lambda s, n=len(lengths): n, "get_seq_length": lambda s, i, ls=lengths: ls[i]})()
        reference_slices = ns["TrajectorySlicerDataset"](fake, window, frame_step=step).slices
        assert official_slices(lengths, window, step) == reference_slices


def test_windows_are_single_camera_two_step_and_split_respecting(tmp_path):
    path, manifest = write_dataset(tmp_path)
    train = DynaMoWindows(path, split_episodes(manifest, "train"))
    val = DynaMoWindows(path, split_episodes(manifest, "validation"))
    assert train.cameras == list(range(6)), "the six training cameras only"
    assert len(train) == 6 * ((14 - 9) + (12 - 9)) and len(val) == 6 * (16 - 9)
    seen = set()
    for dataset, episodes in ((train, {0, 1}), (val, {2})):
        for index in range(len(dataset)):
            tag = dataset[index][:, 0, :, 0, 0].int()  # (T, 3): frame, camera, episode at pixel (0, 0)
            ep, cam, frames = dataset.window_frames(index)
            assert tag[:, 0].tolist() == frames and frames == list(range(frames[0], frames[0] + 9, 2))
            assert set(tag[:, 1].tolist()) == {cam} and cam < 6 and set(tag[:, 2].tolist()) == {int(ep[2:])}
            assert int(ep[2:]) in episodes
            seen.add((int(ep[2:]), cam, frames[0]))
    assert len(seen) == len(train) + len(val)
    sample = train[3]
    with h5py.File(path, "r") as f:
        ep, cam, frames = train.window_frames(3)
        expected = torch.from_numpy(f["episodes"][ep]["rgb"][frames, cam]).permute(0, 3, 1, 2)[:, None]
    assert torch.equal(sample, expected)
    obs = to_official_obs(sample[None], "cpu")
    assert obs.dtype == torch.float32 and 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0
    assert torch.equal(obs, sample[None].float() / 255.0), "exactly the official CPU obs / 255.0"
    if torch.cuda.is_available():
        assert torch.equal(to_official_obs(sample[None], "cuda").cpu(), obs)
    capped = DynaMoWindows(path, split_episodes(manifest, "train"), max_episodes=1)
    assert len(capped) == 6 * (14 - 9)


def test_epoch_sampler_is_a_resumable_permutation():
    sampler = EpochSampler(37, seed=42)
    sampler.set_epoch(3)
    full = list(sampler)
    assert sorted(full) == list(range(37)) and len(sampler) == 37
    sampler.set_epoch(3, start=16)
    assert list(sampler) == full[16:] and len(sampler) == 21
    sampler.set_epoch(4)
    assert list(sampler) != full


# -------------------------------------------------------------------------------------------- encoder and loss


def test_encoder_is_the_official_resnet18_trunk():
    import torchvision

    torch.manual_seed(0)
    encoder = build_encoder(encoder_kwargs(task_config())).eval()
    assert isinstance(encoder, official.resnet18())
    assert sum(p.numel() for p in encoder.parameters()) == 11_176_512
    reference = torchvision.models.resnet18(weights=None)
    names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
    state = {}
    for key, value in encoder.state_dict().items():
        index, rest = key.split(".", 2)[1:]
        state[f"{names[index]}.{rest}"] = value
    reference.fc = torch.nn.Identity()
    reference.load_state_dict(state, strict=True)
    reference.eval()
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = torch.rand(6, 3, 128, 128)
    with torch.no_grad():
        out = encoder(x)
        assert out.shape == (6, 512)
        assert torch.allclose(out, reference(normalize(x)), atol=1e-6, rtol=1e-5)
        assert torch.allclose(encoder(x[0]), out[0], atol=1e-5, rtol=1e-5)  # a 3-D frame (batch size changes rounding)
        assert torch.equal(encoder(x.view(2, 3, 1, 3, 128, 128)).view(6, 512), out)


def test_objective_is_the_dynamo_loss_on_the_official_modules(tmp_path):
    """Our training batch through the official DynaMoSSL equals the paper's objective computed by hand:
    1 - cos(f(s_t, z_{t+1}), sg(s_{t+1})) + 0.04 * mean over the d(d-1) off-diagonal covariances squared."""
    path, manifest = write_dataset(tmp_path)
    dataset = DynaMoWindows(path, split_episodes(manifest, "train"))
    torch.manual_seed(42)
    modules = build_modules(tiny_config(), "cpu")
    obs = to_official_obs(torch.stack([dataset[i] for i in (0, 7, 20, 31)]), "cpu")
    obs_enc, obs_proj, loss, components = modules.ssl.forward(obs)
    assert obs_enc.shape == (4, 5, 1, 512) and obs_proj.shape == (4, 5, 1, 4)
    s, z = obs_enc[:, :, 0], obs_proj[:, :, 0]
    pred = modules.ssl.forward_dynamics(torch.cat([s[:, :-1], z[:, 1:]], dim=-1))
    dynamics = 1 - F.cosine_similarity(pred, s[:, 1:].detach(), dim=-1).mean()
    cov = torch.cov(s.reshape(-1, 512).T)
    off = cov[~torch.eye(512, dtype=torch.bool)]
    covariance = 0.04 * off.square().sum() / (512 * 511)
    assert torch.allclose(components["dynamics_loss_0_0"], dynamics, atol=1e-6)
    assert torch.allclose(components["covariance_loss"], covariance, rtol=1e-4)
    assert torch.allclose(loss, dynamics + covariance, atol=1e-6)
    shifted = s.clone()
    shifted[:, -1] += 1.0  # the inverse model is causal: z_t sees s_0..s_t only
    moved = modules.projector(shifted[:, :, None])
    assert torch.allclose(moved[:, :-1], obs_proj[:, :-1], atol=1e-6, rtol=0)
    assert not torch.allclose(moved[:, -1], obs_proj[:, -1], atol=1e-3)
    # gradients: the target is stop-gradient (SimSiam), so the hand-written loss with a detached target matches
    params = list(modules.encoder.parameters()) + list(modules.projector.parameters())
    grads = torch.autograd.grad(loss.sum(), params, retain_graph=True)
    manual = torch.autograd.grad((dynamics + covariance).sum(), params)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-4) for a, b in zip(grads, manual))


def test_training_step_and_schedule_equal_the_official_trainer(tmp_path):
    """Three epochs (lr 0, warmup, cosine) of the verbatim ``Trainer.train`` loop and ``Trainer.eval`` against
    ``start_epoch`` / ``train_step`` / ``evaluate`` from the same initial weights: bitwise equal parameters,
    optimiser states and epoch logs."""
    path, manifest = write_dataset(tmp_path)
    dataset = DynaMoWindows(path, split_episodes(manifest, "train"))
    batches = [torch.stack([dataset[i] for i in idx]) for idx in ((0, 9, 33), (5, 17, 40, 2), (12,))]
    official_obs = [(b.float() / 255.0, None, None) for b in batches]  # upstream: obs / 255.0 in the dataset
    train_cfg = {"use_lr_scheduling": True, "ssl_lr": 1e-4, "warmup_epochs": 2, "num_epochs": 4, "clip_grad_norm": 0.1}
    cfg = tiny_config(**train_cfg)
    torch.manual_seed(42)
    ours = build_modules(cfg, "cpu")
    torch.manual_seed(42)
    theirs = build_modules(cfg, "cpu")
    trainer = official_trainer_class()()
    trainer.__dict__.update(
        cfg=types.SimpleNamespace(**train_cfg, eval_offline=False), accelerator=SingleProcessAccelerator(),
        encoder=theirs.encoder, projector=theirs.projector, ssl=theirs.ssl, encoder_optim=theirs.encoder_optim,
        projector_optim=theirs.projector_optim, train_loader=official_obs, test_loader=official_obs[:2],
        log_components=OrderedDict(),
    )
    log = EpochLog()
    initial = copy.deepcopy(ours.encoder.state_dict())
    for epoch in range(3):
        trainer.epoch = epoch
        trainer.train()
        trainer.eval()
        lr = start_epoch(ours, train_cfg, epoch)
        log.append("metrics", 1, {"lr": lr})
        for batch in batches:
            obs = to_official_obs(batch, "cpu")
            log.append("ssl_train", len(obs), train_step(ours, obs, train_cfg["clip_grad_norm"]))
        evaluate(ours, batches[:2], log, lambda b: to_official_obs(b, "cpu"))
        assert lr == epoch_lr(train_cfg, epoch) == [0.0, 5e-5, 1e-4][epoch]
        assert ours.forward_dynamics_optim.param_groups[0]["lr"] == 1e-4, "upstream never schedules it"
        for a, b in zip(ours.state_dict().values(), theirs.state_dict().values()):
            assert_equal_nested(a, b)
        assert log.components == trainer.log_components
        assert {"metrics/lr", "ssl_train/total_loss", "ssl_train/covariance_loss", "ssl_eval/dynamics_loss_0_0",
                "metrics/obs_enc_mean_std", "metrics/obs_proj_mean_norm"} <= set(log.components)
        assert log.components["ssl_train/total_loss"][0] == 8, "every window of the epoch is counted"
        log.components, trainer.log_components = OrderedDict(), OrderedDict()
        if epoch == 0:  # lr 0: only BatchNorm running statistics and the forward model move
            for key, value in ours.encoder.state_dict().items():
                assert torch.equal(value, initial[key]) == ("running" not in key and "num_batches" not in key), key
    assert not any(torch.equal(v, initial[k]) for k, v in ours.encoder.state_dict().items() if k.endswith("conv1.weight"))


def assert_equal_nested(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal_nested(x, y)
    elif torch.is_tensor(a):
        assert torch.equal(a, b)
    else:
        assert a == b


# --------------------------------------------------------------------------------------------- configs and size


def test_configs_carry_the_official_simsiam_hyperparameters():
    up = lambda rel: yaml.safe_load((UPSTREAM / "configs" / rel).read_text())  # noqa: E731
    kitchen, pusht = up("train_sim_kitchen.yaml"), up("train_pusht.yaml")
    ssl_up, proj_up, enc_up = up("ssl/dynamo_sim_kitchen.yaml"), up("projector/inverse_dynamics_sim_kitchen.yaml"), up(
        "encoder/resnet18_random.yaml"
    )
    for name in ("ssl/dynamo_pusht.yaml", "ssl/dynamo_libero.yaml"):
        assert up(name) == ssl_up, "the SimSiam setting is shared by Kitchen, Push-T and LIBERO"
    reference = task_config()
    for task in TASKS:
        cfg = task_config(task)
        assert cfg["dataset"]["hdf5_path"].endswith(f"/{task}.hdf5")
        assert cfg["data_interface"]["split_manifest"].endswith(f"/splits/{task}_seed0.json")
        for section in ("encoder", "projector", "ssl", "train"):
            assert cfg[section] == reference[section], (task, section)
    for key, value in cfg["ssl"].items():
        assert ssl_up[key] == value, key
    assert {k: v for k, v in cfg["projector"].items() if k != "output_dim"} == {
        k: v for k, v in proj_up.items() if k not in ("_target_", "window_size", "input_dim", "output_dim")
    }
    assert cfg["projector"]["output_dim"] == 39, "chosen: the Meta-World observation size"
    assert cfg["encoder"] == {k: v for k, v in enc_up.items() if k != "_target_"}
    for key in ("ssl_lr", "ssl_weight_decay", "betas", "clip_grad_norm", "seed", "sync_bn", "use_lr_scheduling",
                "warmup_epochs", "num_epochs", "save_every_epochs"):
        assert cfg["train"][key] == (float(kitchen[key]) if key == "ssl_lr" else kitchen[key]), key
    assert cfg["dataset"]["window_size"] == pusht["window_size"] == 5
    assert cfg["dataset"]["batch_size"] == kitchen["batch_size"] and cfg["data_interface"]["frame_spacing"] == 2
    rl = yaml.safe_load((REPO / "configs/rl/encoders/dynamo.yaml").read_text())
    assert rl["vision"]["encoder_type"] == "dynamo" and rl["agent"]["feature_dim"] == 256


def test_rl_time_encoder_size():
    from torch.utils.flop_counter import FlopCounterMode

    encoder = build_encoder(encoder_kwargs(task_config())).eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        out = encoder(torch.rand(1, 3, 128, 128))  # one RL frame
    assert out.shape == (1, 512)
    assert sum(p.numel() for p in encoder.parameters()) == 11_176_512
    assert counter.get_total_flops() == 1_184_366_592  # multiply-add = 2 FLOPs (convolutions only)


# ----------------------------------------------------------------------------------------- exports and RL side


def test_exports_rebuild_the_trained_encoder_and_its_random_init(tmp_path):
    path, manifest = write_dataset(tmp_path)
    dataset = DynaMoWindows(path, split_episodes(manifest, "train"))
    cfg = tiny_config()
    torch.manual_seed(42)
    modules = build_modules(cfg, "cpu")
    kwargs = encoder_kwargs(cfg)
    save_export(tmp_path / "init.pt", modules.encoder, kwargs, 2, SIZE, 0, True)
    initial = copy.deepcopy(modules.encoder.state_dict())
    obs = to_official_obs(torch.stack([dataset[i] for i in range(4)]), "cpu")
    for _ in range(2):
        train_step(modules, obs, 0.1)
    save_export(tmp_path / "encoder.pt", modules.encoder, kwargs, 2, SIZE, 2, False)
    payload, loaded = load_export(tmp_path / "encoder.pt")
    assert payload["frame_spacing"] == 2 and payload["step"] == 2 and not payload["random_init"]
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert not any(isinstance(m, torch.nn.SyncBatchNorm) for m in loaded.modules()), "plain BatchNorm at RL time"
    x = obs.flatten(0, 2)
    modules.encoder.eval()
    with torch.no_grad():
        assert torch.equal(loaded(x), modules.encoder(x))
    for key, value in modules.encoder.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
    rng = torch.get_rng_state()
    init_payload, init = load_export(tmp_path / "init.pt")
    assert torch.equal(torch.get_rng_state(), rng), "loading leaves the global RNG alone"
    assert init_payload["random_init"] and init_payload["step"] == 0
    for key, value in init.state_dict().items():
        assert torch.equal(value, initial[key])
    torch.manual_seed(42)
    fresh = build_modules(cfg, "cpu")
    for key, value in fresh.encoder.state_dict().items():
        assert torch.equal(value, initial[key]), "the random-init export is the seed's initialisation"


def test_drm_with_the_frozen_dynamo_encoder(tmp_path):
    from s4d.rl.agent import DrMAgent
    from s4d.rl.encoders import BACKBONES, FrameMLPStackHead
    from s4d.rl.evaluate import policy_inputs
    from s4d.rl.replay import Replay, replay_iterator

    assert BACKBONES["dynamo"].__name__ == "DynaMoFrameEncoder"
    device = torch.device("cuda")
    torch.manual_seed(0)
    encoder = build_encoder(encoder_kwargs(task_config()))
    encoder.train()
    with torch.no_grad():  # non-trivial BatchNorm running statistics, as after pretraining
        for _ in range(3):
            encoder(torch.rand(16, 3, SIZE, SIZE))
    save_export(tmp_path / "export.pt", encoder, encoder_kwargs(task_config()), 2, SIZE, 5, False)
    base = yaml.safe_load((REPO / "configs/rl/base.yaml").read_text())
    cfg = {
        "env": {"frame_stack": 3, "image_size": SIZE},
        "vision": {"encoder_type": "dynamo", "export_path": str(tmp_path / "export.pt")},
        "agent": {**base["agent"], "feature_dim": 8, "hidden_dim": 16, "dormant_perturb_interval": 4},
    }
    agent = DrMAgent(cfg, 4, 4, device)
    enc = agent.encoder
    assert not agent.augment_pixels and enc.replay_atom_is_feature and not enc.replay_atom_is_stack_feature
    assert enc.replay_atom_shape == (512,) and isinstance(enc.proj_head, FrameMLPStackHead)
    assert not enc.backbone.training and not any(p.requires_grad for p in enc.backbone.parameters())
    stack = np.random.default_rng(0).integers(0, 255, (2, 9, SIZE, SIZE), dtype=np.uint8)
    features = policy_inputs(agent, stack)
    assert features.shape == (2, 3, 512) and features.dtype == torch.float16
    frames = torch.from_numpy(stack).float().div(255).view(6, 3, SIZE, SIZE)
    encoder.eval()
    with torch.no_grad():
        expected = encoder(frames).view(2, 3, 512)
    assert torch.allclose(features.float().cpu(), expected, rtol=1e-2, atol=1e-2)  # fp16 cache, TF32 convolutions
    single = policy_inputs(agent, stack[:1])  # eval-mode BatchNorm: a frame's feature ignores the rest of the batch
    assert torch.allclose(single[0].float(), features[0].float(), rtol=1e-3, atol=1e-3)
    replay = Replay(None, enc.replay_atom_shape, np.float16, 4, 4, 1000, enc.replay_atom_frame_stack, 10)
    rng = np.random.default_rng(0)
    atom = lambda s: policy_inputs(agent, s)[0][-1].cpu().numpy()  # noqa: E731
    for _ in range(3):
        replay.add_initial(atom(rng.integers(0, 255, (1, 9, SIZE, SIZE), dtype=np.uint8)), rng.normal(size=4))
        for t in range(12):
            s = rng.integers(0, 255, (1, 9, SIZE, SIZE), dtype=np.uint8)
            replay.add(rng.uniform(-1, 1, 4), rng.normal(), 1.0, atom(s), rng.normal(size=4), t == 11)
    frozen = copy.deepcopy(enc.backbone.state_dict())
    batches = replay_iterator(replay, 16, 0.97, seed=0, device=device)
    for step in range(2, 10, 2):
        metrics = agent.update(batches, step)
        assert all(math.isfinite(v) for v in metrics.values()), metrics
    for key, value in enc.backbone.state_dict().items():
        assert torch.equal(value, frozen[key]), f"frozen encoder changed: {key}"
    assert not enc.backbone.training


# ------------------------------------------------------------------------------------------------- entry point


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]}
    result = subprocess.run([sys.executable, "-I", str(REPO / "scripts/train_dynamo.py"), *args], env=env,
                            capture_output=True, text=True, timeout=900)
    if check:
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result


def test_entry_point_trains_exports_and_resumes_losslessly(tmp_path):
    path, manifest = write_dataset(tmp_path)  # 48 training windows: 3 steps per epoch at batch 20 (last batch 8)
    cfg = tiny_config(warmup_epochs=1, num_epochs=4, save_every_epochs=1, save_every=3, log_every=1)
    cfg["dataset"].update(hdf5_path=str(path), num_workers=2, batch_size=20)
    cfg["data_interface"]["split_manifest"] = str(manifest)
    config = tmp_path / "dynamo.yaml"
    config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "runs"  # conftest keeps tmp_path inside the repository, as the script requires
    common = ["--config", str(config), "--output-root", str(root), "--no-wandb"]
    run(*common, "--name", "straight", "--set", "train.max_global_steps=10")
    run(*common, "--name", "resumed", "--set", "train.max_global_steps=5")  # stops mid-epoch 1
    refused = run(*common, "--name", "resumed", "--set", "train.max_global_steps=10", check=False)
    assert refused.returncode != 0 and "--resume auto" in refused.stderr
    run(*common, "--name", "resumed", "--resume", "auto", "--set", "train.max_global_steps=10")
    a, b = (torch.load(root / n / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
            for n in ("straight", "resumed"))
    assert (a["global_step"], a["epoch"], a["step_in_epoch"]) == (b["global_step"], b["epoch"], b["step_in_epoch"])
    assert (a["global_step"], a["epoch"], a["step_in_epoch"]) == (10, 3, 1)
    assert_equal_nested(a["modules"], b["modules"])
    assert a["epoch_log"] == b["epoch_log"]

    def epochs(name):
        rows = [json.loads(line) for line in (root / name / "metrics.jsonl").read_text().splitlines()]
        return [r for r in rows if "epoch" in r]

    assert epochs("straight") == epochs("resumed") and [r["epoch"] for r in epochs("straight")] == [0, 1, 2]
    first = epochs("straight")[0]
    assert first["metrics/lr"] == 0.0 and epochs("straight")[1]["metrics/lr"] == 1e-4
    for key in ("ssl_train/total_loss", "ssl_train/covariance_loss", "ssl_eval/total_loss", "metrics/obs_enc_mean_std"):
        assert math.isfinite(first[key]), key
    assert json.loads((root / "straight/completion.json").read_text())["steps"] == 10
    load = lambda n, f: torch.load(root / n / f, map_location="cpu", weights_only=True)  # noqa: E731
    exports = {n: load(n, "encoder.pt") for n in ("straight", "resumed")}
    assert exports["straight"]["step"] == 10 and exports["straight"]["frame_spacing"] == 2
    assert_equal_nested(exports["straight"]["state_dict"], exports["resumed"]["state_dict"])
    assert sorted(p.name for p in (root / "straight/exports").iterdir()) == [f"encoder_epoch0{e}.pt" for e in range(3)]
    init_a, init_b = (load(n, "encoder_init.pt") for n in ("straight", "resumed"))
    assert init_a["random_init"] and init_a["step"] == 0
    assert_equal_nested(init_a["state_dict"], init_b["state_dict"])
    torch.manual_seed(42)
    fresh = build_modules(cfg, "cpu").encoder.state_dict()
    assert_equal_nested(dict(init_a["state_dict"]), dict(fresh))
