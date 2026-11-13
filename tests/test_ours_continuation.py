"""CPU continuation tests; run with --noconftest (the repository conftest requires a GPU)."""

import copy
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from s4d.config import load_config
from s4d.train import loop
from s4d.train.checkpoint import load_checkpoint
from s4d.train.continuation import file_sha, stage_checkpoint
from s4d.train.ddp import DistContext
from s4d.train.schedule import lr_function, training_end

ROOT = Path(__file__).resolve().parents[1]


def real_config():
    return load_config(
        [ROOT / "configs/realworld/ours_ext/baseline200k.yaml", ROOT / "configs/realworld/ours_ext/continuation.yaml"]
    )


def test_legacy_schedule_is_exact_over_every_original_step():
    cfg = real_config()
    f = lr_function(cfg)
    legacy = copy.deepcopy(cfg)
    legacy["train"]["steps"] = 200000
    for key in ("original_steps", "lr_schedule", "restarts"):
        legacy["train"].pop(key)
    old = lr_function(legacy)
    for s in range(200000):
        expected = (
            (s + 1) / 10000
            if s < 10000
            else 0.02 + (1 - 0.02) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 10000) / 190000)))
        )
        assert old(s) == f(s) == expected
    explicit = copy.deepcopy(legacy)
    explicit["train"]["lr_schedule"] = "cosine"
    assert all(lr_function(explicit)(s) == old(s) for s in (0, 1, 9999, 10000, 199999, 200000))


def test_cycle_endpoints_and_no_boundary_jump():
    cfg = real_config()
    f = lr_function(cfg)
    for step in (200000, 300000, 400000, 450000):
        assert f(step) * cfg["train"]["lr"] == 1e-5
    for step in (202500, 302500):
        assert f(step) * cfg["train"]["lr"] == 2.5e-4
    assert f(200001) > f(200000)
    assert f(299999) > f(300000)
    assert f(300001) > f(300000)
    cfg["train"]["stop_step"] = 300000
    assert training_end(cfg) == 300000 and cfg["train"]["steps"] == 400000


@pytest.mark.parametrize(
    "change",
    [
        {"lr_schedule": "bogus"},
        {"original_steps": None},
        {"original_steps": 10000},
        {"restarts": []},
        {"restarts": [{"start": 200001, "end": 400000, "warmup": 2500, "peak_lr": 0.00025}]},
        {"restarts": [{"start": 200000, "end": 400000, "warmup": 200000, "peak_lr": 0.00025}]},
        {"restarts": [{"start": 200000, "end": 300000, "warmup": 2500, "peak_lr": 0.00025}]},
        {"restarts": [{"start": 200000, "end": 400000, "warmup": 0, "peak_lr": 0.00025}]},
        {"restarts": [{"start": 200000, "end": 400000, "warmup": 2500, "peak_lr": 1e-5}]},
        {"restarts": [{"start": 200000, "end": 400000, "warmup": 2500, "peak_lr": float("nan")}]},
        {"restarts": [None]},
        {"stop_step": 400001},
        {"stop_step": -1},
        {"stop_step": True},
    ],
)
def test_invalid_schedule_refused(change):
    cfg = real_config()
    cfg["train"].update(change)
    with pytest.raises(ValueError):
        lr_function(cfg)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = nn.Linear(2, 3), nn.Linear(3, 1)


class Windows(torch.utils.data.Dataset):
    def __len__(self):
        return 17  # 8 batches per epoch: resume at 9 crosses an epoch boundary

    def __getitem__(self, i):
        return {"x": torch.tensor([float(i), 1.0]), "meta": {"index": i}}


class Log:
    def __init__(self):
        self.lr = []

    def text(self, _):
        pass

    def scalars(self, step, values):
        if "train/lr" in values:
            self.lr.append((step, values["train/lr"]))


class Eval:
    def __init__(self):
        self.probe_loaders = {1: SimpleNamespace(generator=torch.Generator().manual_seed(0))}
        self.calls = []

    def __call__(self, model, step):
        # Evaluation may use all three RNGs but must not perturb the next training update.
        self.calls.append((step, torch.rand(2, generator=self.probe_loaders[1].generator).tolist()))
        torch.rand(4)
        random.random()
        np.random.rand()


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_actual_loop_two_jobs_matches_uninterrupted_optimizer_rng_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "build_model", lambda cfg: Tiny())
    monkeypatch.setattr(loop, "validate_batch", lambda batch: None)
    monkeypatch.setattr(
        loop,
        "collate",
        lambda samples: {
            "x": torch.stack([s["x"] for s in samples]),
            "meta": {"index": [s["meta"]["index"] for s in samples]},
        },
    )
    seen = []

    def forward(model, batch, cfg, step):
        seen.append((step, batch["meta"]["index"]))
        target = torch.randn(len(batch["x"]), 1) + random.random() + np.random.rand()
        loss = (model.decoder(model.encoder(batch["x"])) - target).square().mean()
        return {
            "total": loss,
            "losses": {"total": loss, "motion": loss, "infonce": loss},
            "metrics": {"psnr": loss.detach()},
        }

    monkeypatch.setattr(loop, "forward_losses", forward)
    base = {
        "train": dict(
            steps=6,
            warmup_steps=2,
            lr=0.0005,
            min_lr=0.00001,
            batch_size=2,
            num_workers=0,
            log_every=1,
            eval_every=3,
            save_every=3,
            seed=7,
            eval_at_start=True,
        )
    }
    ctx = DistContext(0, 0, 1, torch.device("cpu"))
    loop.train(base, tmp_path / "base", Windows(), ctx, Log(), Eval())
    checkpoint = tmp_path / "base/checkpoints/step_0000006.pt"
    cfg = copy.deepcopy(base)
    cfg["train"].update(
        steps=12,
        original_steps=6,
        lr_schedule="cosine_restarts",
        restarts=[dict(start=6, end=9, warmup=1, peak_lr=0.00025), dict(start=9, end=12, warmup=1, peak_lr=0.00025)],
    )
    log_full, ev_full = Log(), Eval()
    seen.clear()
    loop.train(cfg, tmp_path / "full", Windows(), ctx, log_full, ev_full, str(checkpoint))
    full_stream = seen.copy()
    seen.clear()
    cfg_a = copy.deepcopy(cfg)
    cfg_a["train"]["stop_step"] = 9
    log_a, ev_a, log_b, ev_b = Log(), Eval(), Log(), Eval()
    loop.train(cfg_a, tmp_path / "a", Windows(), ctx, log_a, ev_a, str(checkpoint))
    a_checkpoint = tmp_path / "a/checkpoints/step_0000009.pt"
    a = torch.load(a_checkpoint, weights_only=False)
    # Loading the checkpoint restores the exact AdamW moments, counters, scheduler state, and all RNGs.
    model = Tiny()
    opt, sch = loop.build_optimizer(model, cfg)
    assert load_checkpoint(a_checkpoint, model.encoder, model.decoder, opt, sch) == 9
    assert_tree_equal(opt.state_dict(), a["optimizer"])
    assert_tree_equal(sch.state_dict(), a["scheduler"])
    assert all(int(s["step"]) == 9 and s["exp_avg"].abs().sum() > 0 for s in opt.state.values())
    assert torch.equal(torch.get_rng_state(), a["rng"]["torch"])
    assert random.getstate() == a["rng"]["python"]
    np.testing.assert_array_equal(np.random.get_state()[1], a["rng"]["numpy"][1])
    loop.train(cfg, tmp_path / "b", Windows(), ctx, log_b, ev_b, str(a_checkpoint))
    assert seen == full_stream
    assert log_a.lr + log_b.lr == log_full.lr
    assert ev_a.calls + ev_b.calls == ev_full.calls  # no eval_at_start replay on either resume
    full = torch.load(tmp_path / "full/checkpoints/step_0000012.pt", weights_only=False)
    split = torch.load(tmp_path / "b/checkpoints/step_0000012.pt", weights_only=False)
    for key in ("encoder", "decoder", "optimizer", "scheduler"):
        assert_tree_equal(full[key], split[key])
    ev_retry = Eval()
    loop.train(cfg, tmp_path / "b", Windows(), ctx, Log(), ev_retry, str(tmp_path / "b/checkpoints/step_0000012.pt"))
    assert ev_retry.calls == [ev_full.calls[-1]]  # final evaluation is recovered after an endpoint-save failure


def test_copy_never_writes_source_and_rejects_corruption(tmp_path):
    source, copied = tmp_path / "old.pt", tmp_path / "new/seed.pt"
    source.write_bytes(b"full-state-source")
    before = source.stat().st_mtime_ns
    digest = file_sha(source)
    stage_checkpoint(source, copied, digest)
    stage_checkpoint(source, copied, digest)
    assert source.stat().st_mtime_ns == before and file_sha(copied) == digest
    copied.write_bytes(b"bad")
    with pytest.raises(ValueError):
        stage_checkpoint(source, copied, digest)
    with pytest.raises(ValueError):
        stage_checkpoint(source, source, digest)
    with pytest.raises(ValueError):
        stage_checkpoint(source, tmp_path / "other.pt", "0" * 64)


def test_resume_validation_rejects_reset_moments_and_wrong_schedule(tmp_path):
    from s4d.train.checkpoint import save_checkpoint
    from s4d.train.continuation import validate_resume_state

    cfg = real_config()
    model = Tiny()
    opt, sch = loop.build_optimizer(model, cfg)
    model.decoder(model.encoder(torch.ones(2, 2))).sum().backward()
    opt.step()
    sch.step()
    # Use the real-scale epoch/counters without doing 200,000 CPU updates.
    sch.last_epoch = 200000
    for moments in opt.state.values():
        moments["step"].fill_(200000)
    baseline = copy.deepcopy(cfg)
    baseline["train"]["steps"] = 200000
    for key in ("lr_schedule", "original_steps", "restarts"):
        baseline["train"].pop(key)
    path = save_checkpoint(tmp_path / "source.pt", 200000, model.encoder, model.decoder, opt, sch, baseline)
    state = torch.load(path, weights_only=False)
    validate_resume_state(state, cfg, lower=200000, upper=300000, exact_step=200000)
    reset = copy.deepcopy(state)
    next(iter(reset["optimizer"]["state"].values()))["step"].zero_()
    with pytest.raises(ValueError, match="counters"):
        validate_resume_state(reset, cfg, lower=200000, upper=300000)
    wrong = copy.deepcopy(state)
    wrong["config"]["train"]["steps"] = 400000
    with pytest.raises(ValueError, match="original cosine"):
        validate_resume_state(wrong, cfg, lower=200000, upper=300000)


def test_legacy_stop_semantics_and_isolated_wrapper_help(tmp_path):
    import subprocess
    import sys

    assert training_end({"train": {"steps": 10, "stop_step": None}}) == 10
    assert training_end({"train": {"steps": 10, "stop_step": 0}}) == 10
    assert training_end({"train": {"steps": 10, "stop_step": 12}}) == 10
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/train_ours_continuation.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "--source-checkpoint" in result.stdout
