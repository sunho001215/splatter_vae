"""Variant future_k2: action-conditioned future prediction with a fixed history stride (registered 2026-10-20).

Invariants: the encoder sees only the history window (t-4, t-2, t) and its history-only masking score, identical for
every future stride k2 in 1..6; the targets (t, t+k2, t+2*k2), their depth and 3D motion supervise only; the recorded
actions[t:t+2*k2] and k2 reach only the dynamic displacement heads; current-time Gaussians ignore actions and k2; the
first displacement ignores second-interval actions; padding beyond a prefix is removed; data alignment matches the
collector (action[i]: observation i -> i+1); encoder-side metadata (meta stride, probes) is the history's; exports
record history stride 2 and pass the RL spacing check. The original method's paths are covered by its own tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from s4d.config import load_config
from s4d.data.contract import ContractError, collate, validate_batch
from s4d.data.metaworld.dataset import MAX_ACTIONS, MetaworldWindowDataset
from s4d.losses import rgb
from s4d.model.decoder import ActionPrefixEncoder, DecoderConfig, GaussianDecoder, GroupConfig
from s4d.model.encoder import Encoder, EncoderConfig
from s4d.model.gaussians import DYNAMIC_GROUP
from s4d.train.loop import Model, build_model, forward_losses

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
GROUP = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
STATIC_FIELDS = ("xyz", "scales", "quats", "opacity", "rgb")
K2 = (1, 2, 3, 4, 5, 6)


def _decoder(**kw):
    cfg = DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=GROUP, dynamic=GROUP, num_slots=2, action_dim=4,
                        max_actions=12, action_hidden=32, action_embed=8, **kw)
    decoder = GaussianDecoder(cfg)
    with torch.no_grad():  # leave persistence: non-zero displacement heads so action effects are visible
        for head in (decoder.groups[1].motion01, decoder.groups[1].motion12):
            head[2].weight.normal_(std=0.5)
            head[2].bias.normal_(std=0.1)
    return decoder


def _actions(k2, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(len(k2), 12, 4)
    for b, k in enumerate(k2.tolist()):
        a[b, : 2 * k] = torch.rand(2 * k, 4, generator=g) * 2 - 1
    return a


def test_prefix_encoder_is_61_128_64_and_removes_values_beyond_the_prefix():
    enc = ActionPrefixEncoder(action_dim=4, max_actions=12, hidden=128, embed=64)
    assert [m.in_features for m in enc.mlp if isinstance(m, torch.nn.Linear)] == [61, 128]
    assert enc.mlp[2].out_features == 64
    actions = torch.arange(48, dtype=torch.float32).view(1, 12, 4) + 1
    for n in (1, 3, 12):
        x = enc.prefix(actions, torch.tensor([n]))
        assert x.shape == (1, 61) and torch.equal(x[0, : 4 * n], actions[0, :n].flatten()) and not bool(x[0, 4 * n : 48].any())
        assert x[0, 48:60].tolist() == [1.0] * n + [0.0] * (12 - n) and float(x[0, 60]) == pytest.approx(n / 12)


def test_current_time_gaussians_ignore_future_actions_and_the_future_stride():
    decoder = _decoder()
    slots = torch.randn(6, 2, 8)
    k2 = torch.tensor(K2)
    a = decoder(slots, _actions(k2, 0), k2)
    b = decoder(slots, _actions(k2.flip(0), 1), k2.flip(0))  # other horizons and other actions, same encoding
    for name in STATIC_FIELDS:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    dyn = a.group == DYNAMIC_GROUP
    assert not torch.allclose(a.delta01[:, dyn], b.delta01[:, dyn])
    assert not a.delta01[:, ~dyn].any() and not a.delta12[:, ~dyn].any()
    fresh = GaussianDecoder(decoder.cfg)  # zero-initialised output layers: exactly zero displacement at step 0
    out = fresh(slots, _actions(k2), k2)
    assert not out.delta01.any() and not out.delta12.any()
    assert not decoder(slots).delta01.any() and torch.equal(decoder(slots).xyz, a.xyz)  # panels: persistence


@pytest.mark.parametrize("k", K2)
def test_first_displacement_ignores_second_interval_actions(k):
    decoder = _decoder()
    slots, k2 = torch.randn(2, 2, 8), torch.tensor([k, k])
    base = _actions(k2, 0)
    changed = base.clone()
    changed[:, k : 2 * k] = torch.rand(2, k, 4) * 2 - 1  # second interval only
    a, b = decoder(slots, base, k2), decoder(slots, changed, k2)
    torch.testing.assert_close(a.delta01, b.delta01, rtol=0, atol=0)
    assert not torch.allclose(a.delta12, b.delta12)


def test_values_beyond_the_valid_prefix_do_not_change_any_output():
    decoder = _decoder()
    slots, k2 = torch.randn(3, 2, 8), torch.tensor([1, 3, 5])
    clean = _actions(k2, 0)
    dirty = clean.clone()
    for b, k in enumerate(k2.tolist()):
        dirty[b, 2 * k :] = 7.0
    a, b = decoder(slots, clean, k2), decoder(slots, dirty, k2)
    for name in (*STATIC_FIELDS, "delta01", "delta12"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_action_inputs_are_rejected_where_they_do_not_belong():
    plain = GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=GROUP, dynamic=GROUP, num_slots=2))
    slots, k2 = torch.randn(1, 2, 8), torch.tensor([2])
    with pytest.raises(ValueError, match="not action-conditioned"):
        plain(slots, _actions(k2), k2)
    decoder = _decoder()
    with pytest.raises(ValueError, match="together"):
        decoder(slots, _actions(k2))
    with pytest.raises(ValueError, match="positive"):
        decoder(slots, torch.zeros(1, 12, 4), torch.tensor([0]))
    with pytest.raises(ValueError, match="group split"):
        GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=GROUP, dynamic=GROUP, num_slots=2,
                                      action_dim=4, single_group=True))
    assert set(plain.state_dict()) == set(GaussianDecoder(plain.cfg).state_dict())  # frozen-method checkpoints load


# ------------------------------------------------------------------------------------------ data alignment
def _write_episode_file(path: Path, T: int = 40) -> Path:
    V, H, W = 3, 16, 16  # two training cameras, one evaluation camera
    K = np.tile(np.array([[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float32), (V, 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (V, 1, 1))
    c2w[:, 0, 3] = [0.0, 0.05, -0.05]
    rgb_ = np.zeros((T, V, H, W, 3), dtype=np.uint8)
    rgb_[..., 0] = np.arange(T, dtype=np.uint8)[:, None, None, None]  # frame index in the red channel
    depth = np.full((T, V, H, W), 10000, dtype=np.uint16)
    body = np.zeros((T, 2, H, W), dtype=np.uint16)
    body[..., H // 2 :, :] = 1
    xpos = np.zeros((T, 2, 3), dtype=np.float32)
    xpos[:, 1, 0] = np.arange(T) * 0.004  # the lower half moves 4 mm per simulator step
    quat = np.zeros((T, 2, 4), dtype=np.float32)
    quat[..., 0] = 1
    action = np.stack([np.arange(T, dtype=np.float32) + 0.25 * j for j in range(4)], 1)  # action[i] encodes i
    with h5py.File(path, "w") as f:
        f.attrs.update({"task": "synthetic", "depth_unit_m": 1e-4, "dt_seconds": 0.05, "background_body": 65535,
                        "body_names": json.dumps(["world", "mover"]), "complete": True})
        cams = f.create_group("cameras")
        for key, values in {"K": K, "c2w": c2w, "w2c": np.linalg.inv(c2w), "is_train": np.array([True, True, False])}.items():
            cams.create_dataset(key, data=values)
        ep = f.create_group("episodes").create_group("ep000")
        ep.attrs["length"] = T
        for key, values in {"rgb": rgb_, "depth": depth, "body_id": body, "xpos": xpos, "xquat": quat,
                            "obs": np.tile(np.arange(T, dtype=np.float32)[:, None], (1, 39)), "action": action}.items():
            ep.create_dataset(key, data=values)
    return path


def test_fixed_history_and_sampled_future_strides_align_for_every_k2(tmp_path):
    path = _write_episode_file(tmp_path / "ep.hdf5")
    T = 40
    future = MetaworldWindowDataset(path, ["ep000"], objective="future")
    assert future.history_stride == 2 and future.future_strides == K2 and future.strides == ()
    # eligibility: history t-4 >= 0 and the longest target t+12 <= T-1, one sample per k2 for every anchor
    assert len(future) == 6 * (T - 4 - 12)
    counts = {k: sum(1 for s in future.samples if s[2] == (k,)) for k in K2}
    assert len(set(counts.values())) == 1  # k2 exactly uniform, no episode-boundary bias
    reference = {k: MetaworldWindowDataset(path, ["ep000"], strides=(k,)) for k in K2}
    history_ref = MetaworldWindowDataset(path, ["ep000"], strides=(2,))
    for index in range(len(future)):
        _, t, (k,) = future.samples[index]
        s = future[index]
        assert s["meta"]["history_t_indices"] == [t - 4, t - 2, t]  # encoder frames fixed for every k2
        assert s["meta"]["t_indices"] == [t, t + k, t + 2 * k]
        assert s["meta"]["stride"] == 2 and s["meta"]["history_stride"] == 2 and s["meta"]["future_stride"] == k
        assert s["history_images"][:, 0, 0, 0, 0].tolist() == [t - 4, t - 2, t]
        assert s["images"][:, 0, 0, 0, 0].tolist() == [t, t + k, t + 2 * k]
        assert int(s["future_stride"]) == k and s["actions"].shape == (MAX_ACTIONS, 4)
        assert s["actions"][: 2 * k, 0].tolist() == list(range(t, t + 2 * k)) and not s["actions"][2 * k :].any()
        assert s["probe_state"][:, 0].tolist() == [t - 4, t - 2, t]  # probes describe what the encoder sees
        h = history_ref[history_ref.samples.index(("ep000", t - 4, (2,)))]
        assert torch.equal(h["motion_score"], s["history_motion_score"]) and torch.equal(h["images"], s["history_images"])
        r = reference[k][reference[k].samples.index(("ep000", t, (k,)))]
        for key in ("images", "depth", "motion3d", "motion_weight", "motion_score"):
            assert torch.equal(r[key], s[key]), (k, key)  # every supervision field = the target window
    batch = collate([future[0], future[7]])
    assert batch["future_stride"].dtype == torch.int64 and batch["history_images"].shape == batch["images"].shape
    # per-k2 evaluation loaders with the training maximum share the identical anchors (same encoder windows)
    anchors = {k: [s[:2] for s in MetaworldWindowDataset(path, ["ep000"], objective="future", future_strides=(k,),
                                                         eligibility_future=6).samples] for k in K2}
    assert all(a == anchors[1] for a in anchors.values()) and len(anchors[1]) == T - 16
    with pytest.raises(ValueError, match="eligibility_future"):
        MetaworldWindowDataset(path, ["ep000"], objective="future", future_strides=(5,), eligibility_future=4)


def test_contract_and_dataset_reject_invalid_future_settings(tmp_path):
    path = _write_episode_file(tmp_path / "ep.hdf5")
    future = MetaworldWindowDataset(path, ["ep000"], objective="future", future_strides=(1,))
    batch = collate([future[0], future[1]])
    batch["actions"][0, 2 * int(batch["future_stride"][0])] = 1.0
    with pytest.raises(ContractError, match="beyond"):
        validate_batch(batch)
    with pytest.raises(ValueError, match="history stride"):
        MetaworldWindowDataset(path, ["ep000"], objective="future", history_stride=4)
    with pytest.raises(ValueError, match="future strides"):
        MetaworldWindowDataset(path, ["ep000"], objective="future", future_strides=(0, 2))
    with pytest.raises(ValueError, match="future strides"):
        MetaworldWindowDataset(path, ["ep000"], objective="future", future_strides=(7,))


def test_config_validation_separates_the_objectives():
    from train import check_objective

    variant = load_config([REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/variants/future_k2.yaml"])
    assert check_objective(variant) == "future"
    assert check_objective(load_config([REPO / "configs/metaworld/base.yaml"])) == "reconstruct"
    for key, value in (("data.history_stride", 4), ("data.future_strides", [0, 1]), ("data.val_future_strides", [7]),
                       ("model.decoder.action_dim", 0), ("loss.self_render", 0.5)):
        bad = load_config([REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/variants/future_k2.yaml"],
                          [f"{key}={json.dumps(value)}"])
        with pytest.raises(ValueError):
            check_objective(bad)
    with pytest.raises(ValueError):
        check_objective(load_config([REPO / "configs/metaworld/base.yaml"], ["model.decoder.action_dim=4"]))
    assert not (REPO / "configs/metaworld/variants/action_future.yaml").exists()  # the superseded variant is retired


# ------------------------------------------------------------------------------------------ forward_losses
def _model():
    encoder = Encoder(
        EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8,
                      drop_path=0)
    )
    return Model(encoder, _decoder())


def _future_batch(batch, seed=0, k2=(1, 6)):
    g = torch.Generator().manual_seed(seed)
    B, T, V, _, H, W = batch["images"].shape
    batch = dict(batch)
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8, generator=g)
    batch["history_images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8, generator=g)
    batch["history_depth"] = torch.ones(B, T, V, 1, H, W)
    batch["history_motion_score"] = torch.rand(B, T, V, 1, H, W, generator=g)
    batch["future_stride"] = torch.tensor(k2)
    batch["actions"] = _actions(batch["future_stride"], seed)
    return batch


def _cfg(**aug):
    cfg = load_config([REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/variants/future_k2.yaml"])
    cfg["train"]["bf16"] = False
    cfg["aug"] = {**cfg["aug"], **aug}
    return cfg


def _fake(monkeypatch):
    from s4d.model import render
    from tests.test_review_options import _fake_rasterizer

    monkeypatch.setattr(render, "_rasterize", _fake_rasterizer([]))
    monkeypatch.setattr(rgb, "_ssim_map", lambda x, y: 1 - (x - y).square())


def test_variant_config_builds_an_action_conditioned_model_with_the_unchanged_encoder():
    cfg = _cfg()
    model = build_model(cfg)
    assert model.decoder.action_encoder is not None and cfg["data"]["objective"] == "future"
    frozen = build_model(load_config([REPO / "configs/metaworld/base.yaml"]))
    assert {k: v.shape for k, v in model.encoder.state_dict().items()} == {
        k: v.shape for k, v in frozen.encoder.state_dict().items()
    }
    assert not any("time" in k or "horizon" in k for k in model.encoder.state_dict())  # no time input to the encoder


def test_encoder_sees_only_the_history_and_its_score_and_positives_come_from_the_history(batch, monkeypatch):
    from s4d.train import loop

    _fake(monkeypatch)
    model = _model().train()
    seen, synth = [], []
    original_forward = model.encoder.forward

    def recording(images, score, **kw):
        seen.append((images.detach().clone(), None if score is None else score.detach().clone()))
        return original_forward(images, score, **kw)

    monkeypatch.setattr(model.encoder, "forward", recording)
    original_synth = loop.synthesize_views

    def recording_synth(b, cams, near, far):
        synth.append(b["images"])
        return original_synth(b, cams, near, far)

    monkeypatch.setattr(loop, "synthesize_views", recording_synth)
    fb = _future_batch(batch)
    out = forward_losses(model, fb, _cfg(), 100, source=torch.tensor([0, 1]))
    B, T, V, _, H, W = fb["images"].shape
    hist = (fb["history_images"].float() / 255).permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 3, H, W)
    torch.testing.assert_close(seen[0][0], hist)
    torch.testing.assert_close(seen[0][1], fb["history_motion_score"].permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 1, H, W))
    assert len(synth) == 2 and synth[0] is fb["images"] and synth[1] is fb["history_images"]  # targets, positives
    assert torch.isfinite(out["total"])


def test_future_stride_targets_and_actions_cannot_change_the_encoding(batch, monkeypatch):
    _fake(monkeypatch)
    model = _model().eval()
    a = _future_batch(batch, 0, k2=(1, 6))
    b = dict(a)
    b.update(images=torch.randint(0, 256, a["images"].shape, dtype=torch.uint8), depth=a["depth"] * 1.5,
             motion3d=a["motion3d"] + 0.01, motion_score=torch.rand_like(a["motion_score"]))
    b["future_stride"] = torch.tensor([4, 2])
    b["actions"] = _actions(b["future_stride"], 5)
    with torch.no_grad():
        oa = forward_losses(model, a, _cfg(), 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
        ob = forward_losses(model, b, _cfg(), 100, source=torch.tensor([0, 1]), mask_ratio=0.0)
    assert torch.equal(oa["slots"], ob["slots"])
    for name in STATIC_FIELDS:
        assert torch.equal(getattr(oa["gs"], name), getattr(ob["gs"], name)), name
    assert not torch.equal(oa["gs"].delta01, ob["gs"].delta01)


def test_gradients_reach_the_encoder_and_the_action_branch(batch, monkeypatch):
    _fake(monkeypatch)
    model = _model().train()
    out = forward_losses(model, _future_batch(batch), _cfg(), 100, source=torch.tensor([0, 1]))
    out["total"].backward()
    for name, module in (("encoder", model.encoder), ("action encoder", model.decoder.action_encoder),
                         ("motion01", model.decoder.groups[1].motion01), ("motion12", model.decoder.groups[1].motion12)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        assert any(float(g.abs().sum()) > 0 for g in grads), name


def test_objective_and_self_render_are_checked(batch, monkeypatch):
    _fake(monkeypatch)
    model = _model().train()
    with pytest.raises(ValueError, match="objective"):
        forward_losses(model, batch, _cfg(), 100, source=torch.tensor([0, 1]))  # future config, no history
    cfg = _cfg()
    cfg["loss"]["self_render"] = 0.5
    with pytest.raises(ValueError, match="self_render"):
        forward_losses(model, _future_batch(batch), cfg, 100, source=torch.tensor([0, 1]))


def test_evaluator_encodes_history_and_never_falls_back_to_target_frames(tmp_path):
    from s4d.train.evaluate import encoder_images

    path = _write_episode_file(tmp_path / "ep.hdf5")
    future = MetaworldWindowDataset(path, ["ep000"], with_eval=True, objective="future", future_strides=(3,))
    batch = collate([future[0], future[1]])
    for key in ("images", "eval_images"):
        assert encoder_images(batch, key) is batch[f"history_{key}"]
        assert not torch.equal(encoder_images(batch, key), batch[key])
    del batch["history_eval_images"]
    with pytest.raises(KeyError, match="history_eval_images"):
        encoder_images(batch, "eval_images")
    original = MetaworldWindowDataset(path, ["ep000"], strides=(2,), with_eval=True)
    plain = collate([original[0]])
    assert encoder_images(plain, "eval_images") is plain["eval_images"]


def test_real_export_records_history_stride_two_and_stays_rgb_only_compatible(tmp_path, monkeypatch):
    import importlib
    from dataclasses import asdict

    from s4d.config import dump_config
    from s4d.model.encoder import load_encoder
    from s4d.rl.env import check_pretraining_spacing
    from s4d.train.checkpoint import save_checkpoint

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    exporter = importlib.import_module("scripts.export_encoder")
    enc = EncoderConfig(image_height=32, image_width=32, patch_size=8, width=16, depth=1, heads=4, slot_dim=12)
    dec = asdict(DecoderConfig(slot_dim=12, dim=16, depth=1, heads=4, scene=GROUP, dynamic=GROUP, action_dim=4,
                               action_hidden=32, action_embed=8))
    dec.pop("slot_dim")
    data = load_config([REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/variants/future_k2.yaml"])["data"]
    config = {"model": {"encoder": asdict(enc), "decoder": dec}, "data": data}
    model = build_model(config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dump_config(config, tmp_path / "config.yaml")
    ckpt = save_checkpoint(tmp_path / "checkpoints" / "step_0000010.pt", 10, model.encoder, model.decoder, opt,
                           torch.optim.lr_scheduler.StepLR(opt, step_size=1), config)
    out = tmp_path / "encoder.pt"
    monkeypatch.setattr(sys, "argv", ["export_encoder.py", "--run", str(tmp_path), "--checkpoint", str(ckpt), "--out", str(out)])
    exporter.main()
    payload = torch.load(out, map_location="cpu", weights_only=True)
    assert payload["frame_strides"] == [2] and payload["history_stride"] == 2
    assert payload["pretraining_objective"] == "future" and payload["pretraining_future_strides"] == list(K2)
    assert not any(k.startswith(("action", "groups")) for k in payload["state_dict"])  # encoder only
    check_pretraining_spacing(payload["frame_strides"], int(payload["encoder_config"]["num_frames"]), 2)  # RL: repeat 2
    encoder = load_encoder(out)  # the original RGB-only loader and interface
    slots = encoder(torch.rand(2, 3, 3, 32, 32), None, mask_ratio=0.0)["slots"]
    assert slots.shape[0] == 2 and torch.isfinite(slots).all()
