"""Differential P1 checks against an immutable copy of the immediate-parent launcher.

Command definitions are AST-loaded: no bootstrap, GPU guard or live data is executed.
The evidence harness reuses these fixtures and captures the complete comparison values.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import io
import json
import math
import pickle
import random
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from s4d.config import load_config
from s4d.data.hardneg import DEFAULTS, HardNegativeSampler, robot_state
from s4d.diag import _encp_repr
from s4d.diag.probes import cross_view_retrieval, r2_score, retrieval_top1, ridge_fit, ridge_predict
from s4d.losses.invariance import state_statistics
from s4d.train.evaluate import frame_interval
from s4d.train.loop import hardneg_metrics

REPO = Path(__file__).resolve().parents[1]
LAUNCHERS = ("scripts/encp_repr_diagnostics.py", "runs/setup/tools/encp_repr_diagnostics.py")
ORACLE = REPO / "tests/fixtures/encp_repr_diagnostics_447be17.py.txt"
CONSTANTS = {"GROUPS", "TARGETS", "OPEN", "NEAR_M", "STRIDE", "MIN_GAP", "MAX_GAP"}


def definitions(source, launcher):
    """Execute only definitions/constants with explicit safe dependencies, never command imports."""
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) or (
        isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in CONSTANTS
                                        for t in n.targets))]
    # Tuple assignments (OPEN/NEAR_M and STRIDE/gaps) are equally explicit constants.
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
              and all(isinstance(t, ast.Name) and t.id in CONSTANTS for t in n.targets[0].elts)]
    module = types.ModuleType(launcher.removesuffix(".py").replace("/", "."))
    module.__dict__.update(
        argparse=argparse, json=json, math=math, Path=Path, sys=sys, torch=torch, F=F, h5py=h5py, np=np,
        _encp_repr=_encp_repr, DEFAULTS=DEFAULTS, HardNegativeSampler=HardNegativeSampler, robot_state=robot_state,
        ridge_fit=ridge_fit, ridge_predict=ridge_predict, r2_score=r2_score, retrieval_top1=retrieval_top1,
        cross_view_retrieval=cross_view_retrieval, state_statistics=state_statistics, frame_interval=frame_interval,
        REPO=REPO, hardneg_metrics=hardneg_metrics, load_config=load_config,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), launcher, "exec"), module.__dict__)
    return module


def packed(value):
    """Retain order, types, shapes, dtypes and every tensor/array value; no tolerances or masking."""
    if torch.is_tensor(value):
        return {"type": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": packed(value.detach().cpu().tolist())}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": packed(value.tolist())}
    if isinstance(value, dict):
        return {k: packed(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "values": [packed(v) for v in value]}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    return value


def rng_state():
    return {"python": packed(random.getstate()), "numpy": packed(np.random.get_state()),
            "torch_cpu_hex": torch.get_rng_state().numpy().tobytes().hex(), "cuda": "disabled",
            "threads": torch.get_num_threads()}


def episode_fixture(path):
    gen = np.random.default_rng(41)
    with h5py.File(path, "w") as f:
        f.create_group("cameras")["is_train"] = [True, False, True]
        episodes = f.create_group("episodes")
        for name in ("moving", "still"):
            ep = episodes.create_group(name)
            ep["rgb"] = gen.integers(0, 256, (28, 3, 4, 4, 3), dtype=np.uint8)
            obs = np.zeros((28, 11), dtype=np.float32)
            obs[:, 3] = 1
            if name == "moving":
                obs[:, 0] = np.arange(28) * 0.003
                obs[14:, 3] = 0.3
            obs[:, 4:7] = obs[:, :3] + 0.05
            ep["obs"] = obs
    return path


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = types.SimpleNamespace(num_frames=3)

    def policy_state(self, images):
        assert not self.training and not torch.is_grad_enabled()
        return images.mean((-1, -2)).flatten(1) + self.anchor


class TorchCPU:
    """Only main's hard-coded CUDA device is rebound for the synthetic CPU replay."""
    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, *args):
        assert args == ("cuda", 0)
        return torch.device("cpu")


def numeric_capture(module, path):
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    out = {"rng_before": rng_state()}
    gen = torch.Generator().manual_seed(23)
    state = torch.rand(20, 3, 11, generator=gen)
    state[::2, 2, 3] = 1
    state[1::2, 2, 3] = 0.2
    dt = torch.linspace(0.01, 0.04, 20)
    x = torch.randn(20, 9, generator=gen, dtype=torch.float64)
    groups = [f"ep{i // 4}" for i in range(20)]
    before = packed((state, dt, x))
    y = module.targets(state, dt)
    out["targets"] = packed(y)
    out["phase"] = packed(module.phase(state))
    losses = []

    def mse_loss(*args, **kwargs):
        value = F.mse_loss(*args, **kwargs)
        losses.append([torch.is_grad_enabled(), float(value.detach()).hex()])
        return value

    module.F = types.SimpleNamespace(mse_loss=mse_loss, normalize=F.normalize)
    predict, info = module.mlp_probe(x, y["hand_vel"], groups, seed=7, device="cpu")
    out["mlp"] = {"predictions": packed(predict(x[:7])), "fit": packed(info), "losses": losses[:],
                  "metadata": [predict.__module__, predict.__qualname__, str(inspect.signature(predict))]}
    assert not predict(x[:7]).requires_grad
    # The original predictor consults its launcher's torch at prediction time.
    calls = []

    class TorchPrediction:
        def no_grad(self):
            calls.append("no_grad")
            return torch.no_grad()

    module.torch = TorchPrediction()
    predict(x[:1])
    assert calls == ["no_grad"]
    module.torch = torch
    losses.clear()
    sets = {"train": (x, y, module.phase(state)),
            "approach_only": (x[:4], {k: v[:4] for k, v in y.items()}, torch.ones(4, dtype=torch.bool)),
            "contact_only": (x[:4], {k: v[:4] for k, v in y.items()}, torch.zeros(4, dtype=torch.bool))}
    out["report"] = packed(module.probe_report(x, y, groups, sets, "cpu"))
    out["report_losses"] = losses[:]
    module.F = F
    out["inputs_unchanged"] = before == packed((state, dt, x))
    assert out["inputs_unchanged"]
    draws, metrics = [], []

    class Sampler(HardNegativeSampler):
        def sample(self, episode, t0, stride, generator=None):
            pre = generator.get_state().numpy().tobytes().hex()
            result = super().sample(episode, t0, stride, generator)
            post = generator.get_state().numpy().tobytes().hex()
            draws.append([episode, t0, stride, result, pre, post])
            return result

    def measured(*args):
        assert not torch.is_grad_enabled()
        result = hardneg_metrics(*args)
        metrics.append(packed((args, result)))
        return result

    module.HardNegativeSampler, module.hardneg_metrics = Sampler, measured
    module.load_encoder = lambda _: FakeEncoder()
    arms = [module.Arm("a=fixture-a.pt", "cpu"), module.Arm("b=fixture-b.pt", "cpu")]
    out["within_episode"] = packed(module.within_episode(arms, path, ["moving", "still"], "cpu"))
    out["draws"], out["hardneg_inputs_outputs"] = draws, metrics
    out["rng_after"] = rng_state()
    return out


def generator_capture(module, path):
    transcript, handles = [], []

    class Reader:
        def __init__(self, node):
            self.node = node

        def __getitem__(self, key):
            transcript.append([self.node.name, repr(key)])
            result = self.node[key]
            return Reader(result) if isinstance(result, (h5py.Group, h5py.Dataset)) else result

    @contextlib.contextmanager
    def opened(filename, mode):
        assert filename == path
        transcript.append(["open", mode])
        with h5py.File(filename, mode) as f:
            handles.append(f)
            try:
                yield Reader(f)
            finally:
                transcript.append(["close"])

    module.h5py = types.SimpleNamespace(File=opened)
    result = module.episode_windows(path, ["moving", "still"])
    assert inspect.isgenerator(result) and transcript == []
    first = packed(next(result))
    assert handles[-1].id.valid
    result.close()
    assert not handles[-1].id.valid
    all_rows = packed(list(module.episode_windows(path, ["still", "moving"])))
    empty = list(module.episode_windows(path, []))
    with pytest.raises(KeyError) as exc:
        list(module.episode_windows(path, ["missing"]))
    assert not handles[-1].id.valid
    return {"first": first, "all": all_rows, "empty": empty, "transcript": transcript,
            "error": [type(exc.value).__name__, str(exc.value)]}


def main_capture(module, path, outpath, optional):
    gen = torch.Generator().manual_seed(101)
    with h5py.File(path, "r") as f:
        obs = torch.from_numpy(f["episodes/moving/obs"][:])
        images = torch.from_numpy(f["episodes/moving/rgb"][:]).permute(0, 1, 4, 2, 3)
    idx = torch.arange(10)[:, None] + torch.tensor([0, 2, 4])[None]
    batch = {"images": images[idx, :2], "probe_state": obs[idx],
             "meta": {"episode": [f"ep{i // 2}" for i in range(10)], "stride": [2] * 10}}
    if optional:
        for key in ("eval_images", "near_images", "traj_images"):
            batch[key] = torch.randint(256, (10, 3, 1, 3, 4, 4), generator=gen, dtype=torch.uint8)
    module.load_encoder = lambda _: FakeEncoder()
    module.torch = TorchCPU()
    module.GPU_MAPPING = [{"fixture": "cpu"}]
    module.time = types.SimpleNamespace(time=iter([100.0, 101.25]).__next__)
    module.load_config = lambda _: {"data": {"root": str(path.parent), "train_ratio": 0.8},
                                    "train": {}, "eval": {}}
    configs = []

    def build_data(cfg):
        configs.append(json.loads(json.dumps(cfg)))
        return None, {2: [batch]}, {2: [batch]}, 2

    module.build_data, module.fork_safe_iter = build_data, iter
    module.split_episodes = lambda *args: (["fit"], ["moving", "still"])
    original = sys.argv
    sys.argv = [str(REPO / module.__name__.replace(".", "/")) + ".py", "--task", "fixture",
                "--arm", "a=fixture-a.pt", "--arm", "b=fixture-b.pt", "--out", str(outpath)]
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            module.main()
    finally:
        sys.argv = original
    configs[0]["data"]["root"] = "<fixture-root>"  # Only this synthetic root varies in pytest tmp dirs.
    return {"json_bytes": outpath.read_text(), "stdout": stream.getvalue(), "config": configs}


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])
    torch.set_num_threads(previous)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_calculations_and_rng_are_exact(launcher, tmp_path):
    path = episode_fixture(tmp_path / "fixture.hdf5")
    baseline = definitions(ORACLE.read_text(), launcher)
    candidate = definitions((REPO / launcher).read_text(), launcher)
    assert numeric_capture(baseline, path) == numeric_capture(candidate, path)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_episode_generator_reads_close_and_errors_are_exact(launcher, tmp_path):
    path = episode_fixture(tmp_path / "fixture.hdf5")
    assert generator_capture(definitions(ORACLE.read_text(), launcher), path) == generator_capture(
        definitions((REPO / launcher).read_text(), launcher), path)


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("optional", [False, True])
def test_main_json_and_stdout_are_exact(launcher, optional, tmp_path):
    path = episode_fixture(tmp_path / "fixture.hdf5")
    out = tmp_path / "output.json"
    assert main_capture(definitions(ORACLE.read_text(), launcher), path, out, optional) == main_capture(
        definitions((REPO / launcher).read_text(), launcher), path, out, optional)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_metadata_pickle_and_caller_patches(launcher, tmp_path, monkeypatch):
    baseline = definitions(ORACLE.read_text(), launcher)
    module = definitions((REPO / launcher).read_text(), launcher)
    for name in ("targets", "mlp_probe", "probe_report", "episode_windows", "within_episode", "Arm", "phase", "main"):
        before, after = getattr(baseline, name), getattr(module, name)
        assert (str(inspect.signature(before)), before.__module__, before.__qualname__, before.__doc__) == (
            str(inspect.signature(after)), after.__module__, after.__qualname__, after.__doc__)
        assert inspect.isgeneratorfunction(before) == inspect.isgeneratorfunction(after)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert pickle.loads(pickle.dumps(module.Arm)) is module.Arm
    for name in ("targets", "mlp_probe", "probe_report", "episode_windows", "within_episode"):
        assert pickle.loads(pickle.dumps(getattr(module, name))) is getattr(module, name)
    # Patched caller helpers/constants are resolved on each invocation.
    module.TARGETS = ("rel_pos",)
    x, y = torch.randn(6, 3), {"rel_pos": torch.randn(6, 3)}
    calls = []
    module.mlp_probe = lambda *args, **kw: (lambda q: q.double(), calls.append(kw) or {"patched": True})
    report = module.probe_report(x, y, [0, 0, 1, 1, 2, 2], {"q": (x, y, torch.ones(6, dtype=torch.bool))}, "cpu")
    assert list(report["mlp_fit"]) == ["rel_pos"] and calls == [{"device": "cpu"}]
    path = episode_fixture(tmp_path / "fixture.hdf5")
    rows = list(module.episode_windows(path, ["moving"]))
    episode_calls = []
    module.episode_windows = lambda *args: episode_calls.append(args) or iter(rows)
    module.STRIDE, module.MIN_GAP, module.MAX_GAP = 1, 2, 5
    module.DEFAULTS = {"max_gap": 5, "min_hand_m": 100, "min_gripper": 100}
    module.load_encoder = lambda _: FakeEncoder()
    result, draws = module.within_episode([module.Arm("a=fake.pt", "cpu")], path, ["moving"], "cpu")
    assert draws == 26 and result["a"]["hardneg_valid_fraction"] == 0
    assert episode_calls == [(path, ["moving"])]
    other_path = LAUNCHERS[1 - LAUNCHERS.index(launcher)]
    other = definitions((REPO / other_path).read_text(), other_path)
    assert other.TARGETS != module.TARGETS and other.STRIDE == 2
    assert other.GROUPS is not module.GROUPS


def test_oracle_is_the_pinned_parent_source():
    # Fixed digest of scripts/encp_repr_diagnostics.py at 447be17; the oracle is never regenerated from the helper.
    expected = "fc6cdefb978c8011008f767c6b8e076eb75b4fa2f0cd432206f0058aed2a88ac"
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest() == expected
