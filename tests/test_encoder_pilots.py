"""Encoder pilots on chlrm (user directive 2026-10-25): hard negatives (hn), patch-local conv stem (cs), head dim 64 (h2),
SinCro-shaped body (w4), width 256 at depth 8 (w8); every new flag defaults to off."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from s4d.config import load_config
from s4d.data.contract import ContractError, collate, validate_batch
from s4d.data.hardneg import HardNegativeSampler, candidate_starts, robot_moved, robot_state
from s4d.data.metaworld.dataset import MetaworldWindowDataset
from s4d.losses.invariance import multi_positive_info_nce
from s4d.model import encoder as encoder_mod
from s4d.model.encoder import DEFAULT_ONLY_FIELDS, Encoder, EncoderConfig, PatchCNNStem, load_encoder, visible_count
from s4d.train.loop import build_model, build_optimizer, forward_losses
from tests import _enc_golden as golden

REPO = Path(__file__).resolve().parents[1]
VARIANTS = REPO / "configs/metaworld/variants"
CHLRM = [REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/tasks/hammer.yaml",
         VARIANTS / "chamfer_lean.yaml", VARIANTS / "chamfer_lean_randmask.yaml"]
ARMS = {"hn": "enc_hn.yaml", "cs": "enc_cs.yaml", "h2": "enc_h2.yaml", "w4": "enc_w4.yaml", "w8": "enc_w8.yaml"}
SINCRO_PARAMS, SINCRO_FLOPS = 9_064_704, 1_786_078_891
EXPECTED_PARAMS = {"chlrm": 1_726_592, "hn": 1_726_592, "h2": 1_726_592, "cs": 1_885_856, "w4": 3_439_616,
                   "w8": 6_598_656}
OFF = {"data.hardneg": False, "loss.hardneg_weight": 1.0, "train.encoder_lr_mult": 1.0, "model.encoder.stem": "patchify"}


def _arm_cfg(arm: str | None) -> dict:
    return load_config(CHLRM + ([VARIANTS / ARMS[arm]] if arm else []))


# ---------------------------------------------------------------- flags off: bit for bit the pre-change training path
@pytest.mark.parametrize("explicit", [False, True])
def test_flags_off_reproduce_the_pre_change_training_bit_for_bit(tmp_path, explicit):
    """Against tests/data/enc_flags_off_golden.json, written by the unchanged sources (commit ffd4fb8 = main d9fb142):
    every hash and float bit pattern must match in the environment that wrote it. Elsewhere (another CPU may round
    BLAS sums differently) the platform-independent parts must match: window count, optimizer groups, parameter
    shapes and count, the masks drawn and the random-number consumption of three training steps."""
    expected = json.loads(golden.GOLDEN.read_text())
    got = golden.fingerprints(tmp_path, OFF if explicit else None, {"hardneg": None} if explicit else None)
    if got["environment"] == expected["environment"]:
        assert got == expected
    else:
        for key in ("dataset_len", "optimizer_groups", "full_encoder_params", "rng_after_steps"):
            assert got[key] == expected[key], key
        assert {k: v[0] for k, v in got["full_encoder_state"].items()} == \
            {k: v[0] for k, v in expected["full_encoder_state"].items()}
        assert got["full_encoder_train"]["visible"] == expected["full_encoder_train"]["visible"]


def test_resolved_chlrm_config_has_no_new_keys():
    cfg = _arm_cfg(None)
    assert "hardneg" not in cfg["data"] and "hardneg_weight" not in cfg["loss"]
    assert "encoder_lr_mult" not in cfg["train"] and "stem" not in cfg["model"]["encoder"]


def test_variant_configs_change_only_their_registered_keys():
    ref = _arm_cfg(None)
    changes = {
        "hn": {("data", "hardneg"): True, ("data", "hardneg_max_gap"): 40, ("data", "hardneg_min_hand_m"): 0.01,
               ("data", "hardneg_min_gripper"): 0.1, ("loss", "hardneg_weight"): 1.0},
        "cs": {("model", "encoder", "stem"): "patch_cnn"},
        "h2": {("model", "encoder", "heads"): 2},
        "w4": {("model", "encoder", "width"): 256, ("model", "encoder", "depth"): 4, ("model", "encoder", "heads"): 4},
        "w8": {("model", "encoder", "width"): 256, ("model", "encoder", "heads"): 4},  # depth 8 is chlrm's
    }

    def flat(d, prefix=()):
        out = {}
        for k, v in d.items():
            out.update(flat(v, prefix + (k,)) if isinstance(v, dict) else {prefix + (k,): v})
        return out

    base = flat(ref)
    for arm, wanted in changes.items():
        cfg = flat(_arm_cfg(arm))
        diff = {k: cfg.get(k) for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
        assert diff == wanted, arm
    assert ref["model"]["encoder"]["mask_mode"] == "random" and ref["model"]["encoder"]["mask_ratio"] == 0.75


# ---------------------------------------------------------------- hn: sampler, robot-state rule, dataset, loss
def test_candidate_starts_respect_episode_stride_and_gap_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        length, stride = int(rng.integers(20, 120)), int(rng.choice([2, 4, 6]))
        t0 = int(rng.integers(0, length - 2 * stride))
        starts = candidate_starts(t0, stride, length, 40)
        gaps = np.abs(starts - t0)
        assert ((gaps >= 2 * stride + 2) & (gaps <= 40)).all()
        assert (starts >= 0).all() and (starts + 2 * stride <= length - 1).all()
        for s in starts:  # the two windows share no frame
            assert not {t0, t0 + stride, t0 + 2 * stride} & {s, s + stride, s + 2 * stride}
        everything = [s for s in range(length - 2 * stride) if 2 * stride + 2 <= abs(s - t0) <= 40]
        assert starts.tolist() == everything


def _robot(length=80, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros((length, 4))
    states[:, 0] = np.cumsum(rng.uniform(0, 0.002, length))  # slow hand drift
    states[:, 3] = (np.arange(length) // 25) % 2  # gripper toggles every 25 frames
    return states


def test_sampler_draws_uniformly_among_valid_candidates_only():
    sampler = HardNegativeSampler({"ep": _robot()}, 40, 0.01, 0.1)
    starts, valid = sampler.candidates("ep", 30, 2)
    assert valid.any() and not valid.all()
    gen = torch.Generator().manual_seed(0)
    draws = [sampler.sample("ep", 30, 2, gen) for _ in range(20000)]
    assert set(draws) == set(starts[valid].tolist())
    counts = np.bincount(draws)[starts[valid]]
    from scipy.stats import chisquare

    assert chisquare(counts).pvalue > 1e-3  # uniform over the valid candidates


def test_robot_rule_thresholds_hand_and_gripper():
    a = np.array([0.0, 0.0, 0.0, 0.5])
    for b, moved in (([0.0099, 0, 0, 0.5], False), ([0.0101, 0, 0, 0.5], True), ([0, 0, 0, 0.59], False),
                     ([0, 0, 0, 0.601], True), ([0.006, 0.006, 0.006, 0.5], True)):
        assert bool(robot_moved(a, np.array(b), 0.01, 0.1)) is moved, b
        assert bool(robot_moved(torch.tensor(a), torch.tensor(b), 0.01, 0.1)) is moved, b


def test_the_rule_applies_to_droid_robot_states():
    """DROID probe_state = cartesian_position (xyz, Euler xyz) + gripper_position: only xyz and the gripper decide."""
    a = torch.tensor([0.40, 0.00, 0.20, 0.1, 0.2, 0.3, 0.50])
    for change, moved in (({0: 0.405}, False), ({0: 0.411}, True), ({6: 0.59}, False), ({6: 0.61}, True),
                          ({3: 2.0, 4: -1.0, 5: 0.7}, False)):  # orientation alone never counts
        b = a.clone()
        for i, v in change.items():
            b[i] = v
        got = robot_moved(robot_state(a, "droid"), robot_state(b, "droid"), 0.01, 0.1)
        assert bool(got) is moved, change


def test_hand_displacement_is_between_the_two_last_frames():
    states = np.zeros((60, 4))
    states[:, 0] = np.arange(60) * 0.001  # 1 mm per frame
    sampler = HardNegativeSampler({"ep": states}, 40, 0.01, 0.1)
    assert sampler.hand_displacement("ep", 10, 30, 2) == pytest.approx(0.020)  # frames 14 and 34
    assert sampler.hand_displacement("ep", 30, 10, 4) == pytest.approx(0.020)  # frames 38 and 18


def test_hardneg_metrics_on_constructed_states():
    from s4d.train.loop import hardneg_metrics

    V = 3
    a = torch.eye(4)[:V][None].repeat(2, 1, 1)  # anchors: camera c -> e_c
    a[:, :, 3] = 1.0  # shared component: own views are similar (cos 0.5)
    n = a.clone()
    n[0] = torch.tensor([[1.0, 0, 0, 0.0]] * V)  # sample 0: every negative view equals e_0
    n[1] = -a[1]  # sample 1: negatives opposite
    valid = torch.tensor([True, True])
    m = hardneg_metrics(a, n, valid, torch.tensor([0.02, 0.04]))
    # sample 0: anchor cam 0 vs own cam b: 0.5; vs negative cam b: cos(e0+e3, e0) = 0.707 -> negative wins (pairs 0->1, 0->2)
    #           anchors cams 1, 2 vs negative e0: 0 < 0.5 -> own wins (4 pairs); sample 1: all own wins (6 pairs)
    assert float(m["hardneg_within_episode_top1"]) == pytest.approx(10 / 12)
    assert float(m["hardneg_gallery_top1"]) == pytest.approx(5 / 6)  # only sample 0, camera 0 fails
    cos0 = torch.tensor([0.7071068, 0.0, 0.0]).mean()
    assert float(m["hardneg_cosine"]) == pytest.approx(float((cos0 + torch.tensor(-1.0)) / 2), abs=1e-5)
    assert float(m["hardneg_hand_disp_mean"]) == pytest.approx(0.03)
    none = hardneg_metrics(a, n, torch.tensor([False, False]), torch.tensor([-1.0, -1.0]))
    assert float(none["hardneg_valid_fraction"]) == 0 and torch.isnan(none["hardneg_gallery_top1"])


def test_robot_state_columns_are_the_proprio_and_the_droid_robot_fields():
    probe = torch.arange(11.0)[None]
    assert robot_state(probe, "metaworld").tolist() == [[0.0, 1.0, 2.0, 3.0]]
    droid = torch.arange(7.0)[None]  # cartesian_position (xyz, Euler xyz), gripper_position
    assert robot_state(droid, "droid").tolist() == [[0.0, 1.0, 2.0, 6.0]]


def test_filter_reads_robot_state_only(tmp_path):
    """Changing every object (and other non-robot) observation leaves every decision unchanged; moving the robot
    across a threshold changes it."""
    path = golden.write_fixture(tmp_path / "f.hdf5")
    rule = {"max_gap": 40, "min_hand_m": 0.01, "min_gripper": 0.1}
    ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6), hardneg=rule)
    before = {(e, t, s): ds.hardneg.candidates(e, t, s)[1].tolist() for e, t, (s,) in ds.samples}
    import h5py

    with h5py.File(path, "r+") as f:
        for ep in ("ep000", "ep001"):
            obs = f["episodes"][ep]["obs"]
            obs[:, 4:] = np.random.default_rng(5).normal(size=obs[:, 4:].shape) * 10  # object and other state
            f["episodes"][ep]["xpos"][...] = np.random.default_rng(6).normal(size=f["episodes"][ep]["xpos"].shape)
    ds2 = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6), hardneg=rule)
    assert {(e, t, s): ds2.hardneg.candidates(e, t, s)[1].tolist() for e, t, (s,) in ds2.samples} == before
    with h5py.File(path, "r+") as f:
        obs = f["episodes"]["ep000"]["obs"]
        frozen = obs[...]
        frozen[:, 0:3] = 0.0
        frozen[:, 3] = 0.5  # robot still: nothing is valid
        obs[...] = frozen
    still = MetaworldWindowDataset(path, ["ep000"], strides=(2, 4, 6), hardneg=rule)
    assert not any(still.hardneg.candidates(e, t, s)[1].any() for e, t, (s,) in still.samples)
    with h5py.File(path, "r+") as f:
        obs = f["episodes"]["ep000"]["obs"]
        moved = obs[...]
        moved[20:, 3] = 0.61  # gripper change of 0.11 >= 0.1 from frame 20 on
        obs[...] = moved
    some = MetaworldWindowDataset(path, ["ep000"], strides=(2, 4, 6), hardneg=rule)
    assert any(some.hardneg.candidates(e, t, s)[1].any() for e, t, (s,) in some.samples)


def test_dataset_adds_the_negative_window_of_the_same_episode_and_stride(tmp_path):
    path = golden.write_fixture(tmp_path / "f.hdf5")
    plain = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6))
    ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6),
                                hardneg={"max_gap": 40, "min_hand_m": 0.01, "min_gripper": 0.1})
    import h5py

    f = h5py.File(path, "r")
    torch.manual_seed(0)
    n_valid = 0
    for i in range(0, len(ds), 5):
        s, p = ds[i], plain[i]
        assert all(torch.equal(s[k], p[k]) for k in p if torch.is_tensor(p[k]))  # the window itself is unchanged
        ep, t0, (stride,) = ds.samples[i]
        if bool(s["hardneg_valid"]):
            n_valid += 1
            tn = int(s["hardneg_t0"])
            assert 2 * stride + 2 <= abs(tn - t0) <= 40
            rgb = np.stack([f["episodes"][ep]["rgb"][tn + k * stride, :2] for k in range(3)])
            assert torch.equal(s["hardneg_images"], torch.from_numpy(rgb).permute(0, 1, 4, 2, 3))
            assert float(s["hardneg_hand_disp"]) == pytest.approx(
                ds.hardneg.hand_displacement(ep, t0, tn, stride), abs=1e-6)
        else:
            assert int(s["hardneg_t0"]) == -1 and not s["hardneg_images"].any()
    assert n_valid > 0
    batch = collate([ds[0], ds[1]])
    validate_batch(batch)
    bad = dict(batch)
    del bad["hardneg_t0"]
    with pytest.raises(ContractError):
        validate_batch(bad)


def test_negatives_enter_only_the_denominator():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8)
    neg = torch.randn(5, 8, requires_grad=True)
    base, _ = multi_positive_info_nce(x, 0.1, distributed=False)
    none_valid, _ = multi_positive_info_nce(x, 0.1, distributed=False, negatives=neg,
                                            negatives_valid=torch.zeros(5, dtype=torch.bool))
    assert torch.equal(base, none_valid)  # no valid extra column: exactly the loss without them
    valid = torch.tensor([True, False, True, True, False])
    loss, _ = multi_positive_info_nce(x, 0.1, distributed=False, negatives=neg, negatives_valid=valid)
    # manual reference: the positives are untouched, the valid negatives add exp(cos / t) to every denominator
    local = F.normalize(x.reshape(12, 8), dim=-1)
    n = F.normalize(neg, dim=-1)[valid]
    logits = local @ local.t() / 0.1
    sample = torch.arange(12) // 3
    pos = (sample[:, None] == sample[None]) & ~torch.eye(12, dtype=torch.bool)
    den = ~torch.eye(12, dtype=torch.bool)
    log_pos = torch.logsumexp(logits.masked_fill(~pos, -1e30), 1)
    log_den = torch.logsumexp(torch.cat((logits.masked_fill(~den, -1e30), local @ n.t() / 0.1), 1), 1)
    assert torch.allclose(loss, (log_den - log_pos).mean(), atol=1e-6)
    assert loss > base
    loss.backward()
    assert neg.grad[valid].abs().sum() > 0 and neg.grad[~valid].abs().sum() == 0
    weighted, _ = multi_positive_info_nce(x, 0.1, distributed=False, negatives=neg.detach(), negatives_valid=valid,
                                          negatives_weight=2.0)
    log_den2 = torch.logsumexp(torch.cat((logits.masked_fill(~den, -1e30), local @ n.t() / 0.1 + np.log(2.0)), 1), 1)
    assert torch.allclose(weighted, (log_den2 - log_pos).mean(), atol=1e-6)


def _hn_batch(tmp_path, n=2):
    path = golden.write_fixture(tmp_path / "f.hdf5")
    ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6),
                                hardneg={"max_gap": 40, "min_hand_m": 0.01, "min_gripper": 0.1})
    torch.manual_seed(3)
    picks = [i for i in range(len(ds)) if bool(ds[i]["hardneg_valid"])][:n]
    torch.manual_seed(3)
    return collate([ds[i] for i in picks])


def _small(arm=None):
    cfg = golden.small_cfg(*([ARMS[arm]] if arm else []))
    return cfg


def test_hn_changes_only_infonce_and_reuses_the_anchor_masking(tmp_path, monkeypatch):
    """Negatives are encoded with the anchors' scheme (fresh uniform masks of the same count over the 3 x N temporal
    tokens), never decoded, rendered or used in consistency: every other loss is bit-identical to the same step
    without them, and the rasterizer sees the same calls."""
    saved = golden._patched()
    try:
        batch = _hn_batch(tmp_path)
        plain_batch = {k: v for k, v in batch.items() if not k.startswith("hardneg_")}
        calls_seen, masks = {}, {}
        results = {}
        for name, cfg, b in (("off", _small(), plain_batch), ("hn", _small("hn"), batch)):
            from s4d.model import render
            from tests.test_review_options import _fake_rasterizer

            calls = []
            render._rasterize = _fake_rasterizer(calls)
            torch.manual_seed(0)
            model = build_model(cfg).train()
            seen = []
            original = encoder_mod.sample_random_tokens

            def recording(rows, tokens, keep, device, generator=None, _seen=seen, _orig=original):
                _seen.append((rows, tokens, keep))
                return _orig(rows, tokens, keep, device, generator)

            monkeypatch.setattr(encoder_mod, "sample_random_tokens", recording)
            torch.manual_seed(1)
            results[name] = forward_losses(model, b, cfg, 100, source=torch.tensor([0, 1]))
            calls_seen[name], masks[name] = calls, seen
            monkeypatch.setattr(encoder_mod, "sample_random_tokens", original)
        off, hn = results["off"]["losses"], results["hn"]["losses"]
        assert set(off) == set(hn)
        for key in off:
            if key in ("infonce", "total"):
                continue
            assert torch.equal(off[key], hn[key]), key
        assert not torch.equal(off["infonce"], hn["infonce"])
        assert calls_seen["off"] == calls_seen["hn"]
        B, V = batch["images"].shape[0], batch["images"].shape[2]
        N = 16  # 16 x 16 images, patch 4
        anchor = (B * V, 3 * N, visible_count(3 * N, 0.75))
        assert masks["off"] == [anchor] and masks["hn"] == [anchor, anchor]
        m = results["hn"]["metrics"]
        assert 0.0 <= float(m["hardneg_within_episode_top1"]) <= 1.0 and float(m["hardneg_valid_fraction"]) == 1.0
        assert -1.0 <= float(m["hardneg_cosine"]) <= 1.0
    finally:
        golden._restore(saved)


def test_hn_gradients_reach_the_encoder_through_the_negatives(tmp_path):
    saved = golden._patched()
    try:
        batch = _hn_batch(tmp_path)
        cfg = _small("hn")
        torch.manual_seed(0)
        model = build_model(cfg).train()
        outputs = []
        original = model.encoder.forward

        def recording(images, motion_score=None, mask_ratio=None):
            res = original(images, motion_score, mask_ratio)
            outputs.append(res["slots"])
            return res

        model.encoder.forward = recording
        out = forward_losses(model, batch, cfg, 100, source=torch.tensor([0, 1]))
        del model.encoder.forward
        assert len(outputs) == 2  # anchors, then the negatives (synthetic views use forward_visible)
        (grad,) = torch.autograd.grad(out["total"], [outputs[1]])
        assert grad.abs().sum() > 0  # the loss depends on the negative states, so their encoder receives gradients
        with pytest.raises(ValueError, match="hard-negative"):
            forward_losses(model, {k: v for k, v in batch.items() if not k.startswith("hardneg_")}, cfg, 100)
        with pytest.raises(ValueError, match="hard-negative"):
            forward_losses(model, batch, _small(), 100)
        model.eval()  # validation: no negatives are used even when the batch carries them
        forward_losses(model, {k: v for k, v in batch.items() if not k.startswith("hardneg_")}, cfg, 100)
    finally:
        golden._restore(saved)


def test_hn_statistics_summarise_kept_pairs():
    sampler = HardNegativeSampler({"ep": _robot()}, 40, 0.01, 0.1)
    samples = [("ep", t, s) for s in (2, 4, 6) for t in range(80 - 12)]
    stats = sampler.statistics(samples)
    assert 0 < stats["kept_fraction"] < 1 and stats["candidate_pairs"] > stats["kept_pairs"] > 0
    assert sum(stats["hand_displacement_m_histogram"]["counts"]) == stats["kept_pairs"]
    assert 0 < stats["valid_negative_fraction"] <= 1


# ---------------------------------------------------------------- cs: patch-local stem
def test_conv_stem_is_patch_local_and_keeps_the_token_layout():
    torch.manual_seed(0)
    stem = PatchCNNStem(16, 32).eval()
    x = torch.rand(2, 3, 64, 48)
    base = stem(x)
    assert base.shape == (2, 32, 4, 3)
    for r, c in ((0, 0), (37, 21), (63, 47), (16, 16)):
        y = x.clone()
        y[1, 1, r, c] += 0.5
        diff = (stem(y) - base).abs().sum(1)  # (2, 4, 3)
        changed = (diff > 0).nonzero().tolist()
        assert changed == [[1, r // 16, c // 16]], (r, c, changed)
    enc = Encoder(EncoderConfig(stem="patch_cnn", mask_mode="random", mask_ratio=0.75)).eval()
    img = torch.rand(1, 3, 3, 128, 128)
    tokens = enc.patch_embed(img.flatten(0, 1)).flatten(2).transpose(1, 2)
    img2 = img.clone()
    img2[0, 1, :, 40, 100] += 0.3  # frame 1, patch row 2, column 6
    tokens2 = enc.patch_embed(img2.flatten(0, 1)).flatten(2).transpose(1, 2)
    changed = ((tokens2 - tokens).abs().sum(-1) > 0).nonzero().tolist()
    assert changed == [[1, 2 * 8 + 6]]
    with pytest.raises(ValueError, match="patch_size"):
        PatchCNNStem(14, 32)
    with pytest.raises(ValueError, match="stem"):
        Encoder(EncoderConfig(stem="overlapping"))


# ---------------------------------------------------------------- shapes, sizes, costs, export / RL adapter
@pytest.mark.parametrize("arm", [None, *ARMS])
def test_encoder_sizes_shapes_and_costs(arm):
    from torch.utils.flop_counter import FlopCounterMode

    cfg = _arm_cfg(arm)
    model = build_model(cfg)
    enc = model.encoder.eval()
    params = sum(p.numel() for p in enc.parameters())
    assert params == EXPECTED_PARAMS[arm or "chlrm"] and params <= SINCRO_PARAMS
    x = torch.zeros(1, 3, 3, 128, 128)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        slots = enc(x, mask_ratio=0.0)["slots"]
    assert counter.get_total_flops() / 3 <= SINCRO_FLOPS  # three-frame observation / 3, multiply-add = 2
    assert slots.shape == (1, 1, 256) and enc.state_dim == 256
    assert model.decoder.cfg.dim == 128 and enc.slot_proj.in_features == enc.cfg.width
    if arm in ("w4", "w8"):
        assert enc.cfg.width // enc.cfg.heads == 64 and enc.blocks[0].mlp.fc1.out_features == 1024
    if arm == "h2":
        assert enc.cfg.width // enc.cfg.heads == 64


@pytest.mark.parametrize("arm", [None, *ARMS])
def test_export_and_rl_adapter_keep_the_256_feature(tmp_path, arm):
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import export_encoder

    from s4d.config import dump_config

    cfg = _arm_cfg(arm)
    model = build_model(cfg)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    dump_config(cfg, run / "config.yaml")
    torch.save({"encoder": model.encoder.state_dict(), "step": 7}, run / "checkpoints/latest.pt")
    out = export_encoder.export(run, None, tmp_path / "encoder.pt")
    payload = torch.load(out, map_location="cpu", weights_only=True)
    keys = set(payload["encoder_config"])
    old_fields = {f.name for f in fields(EncoderConfig)} - set(DEFAULT_ONLY_FIELDS)
    assert keys == (old_fields | {"stem"} if arm == "cs" else old_fields)  # unchanged arms: readable by older code
    enc = load_encoder(out)
    assert enc.state_dim == 256
    from s4d.rl.encoders import Splatter4DStateEncoder

    adapter = Splatter4DStateEncoder({"vision": {"export_path": str(out)}})
    assert adapter.frame_mode == "joint"
    assert adapter(torch.rand(2, 3, 3, 128, 128)).shape == (2, 256)


def test_encoder_lr_multiplier_scales_only_the_encoder_groups():
    cfg = _small()
    torch.manual_seed(0)
    model = build_model(cfg)
    ref, _ = build_optimizer(model, cfg)
    cfg["train"]["encoder_lr_mult"] = 0.6
    opt, sched = build_optimizer(model, cfg)
    lr = float(cfg["train"]["lr"])
    enc_ids = {id(p) for p in model.encoder.parameters()}
    for group in opt.param_groups:
        in_enc = {id(p) in enc_ids for p in group["params"]}
        assert len(in_enc) == 1
        assert group["initial_lr"] == pytest.approx(lr * (0.6 if in_enc.pop() else 1.0))
    assert sum(len(g["params"]) for g in opt.param_groups) == sum(len(g["params"]) for g in ref.param_groups)
    assert len(ref.param_groups) == 2 and len(opt.param_groups) == 4
