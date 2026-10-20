"""Variant action_future: action-conditioned future prediction from an RGB history (registered 2026-10-20).

Invariants: the encoder sees only the history window (t-2k, t-k, t) and its history-only masking score; the targets
(t, t+k, t+2k), their depth and 3D motion supervise only; the recorded actions[t:t+2k] reach only the dynamic
displacement heads; current-time Gaussians ignore actions; the first displacement ignores second-interval actions;
padding beyond a prefix is removed; data alignment matches the collector (action[i]: observation i -> i+1).
"""

from __future__ import annotations

import json
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
GROUP = GroupConfig(parents=2, children=2, offset_scale=0.2, child_radius=0.05, scale_min=0.001, scale_max=0.08)
STATIC_FIELDS = ("xyz", "scales", "quats", "opacity", "rgb")


def _decoder(**kw):
    cfg = DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=GROUP, dynamic=GROUP, num_slots=2, action_dim=4,
                        max_actions=12, action_hidden=32, action_embed=8, **kw)
    decoder = GaussianDecoder(cfg)
    with torch.no_grad():  # leave persistence: non-zero displacement heads so action effects are visible
        for head in (decoder.groups[1].motion01, decoder.groups[1].motion12):
            head[2].weight.normal_(std=0.5)
            head[2].bias.normal_(std=0.1)
    return decoder


def _actions(B, count, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(B, 12, 4)
    for b in range(B):
        a[b, : int(count[b])] = torch.rand(int(count[b]), 4, generator=g) * 2 - 1
    return a


def test_prefix_encoder_is_61_128_64_for_four_dimensional_actions_and_removes_values_beyond_the_prefix():
    enc = ActionPrefixEncoder(action_dim=4, max_actions=12, hidden=128, embed=64)
    assert [m.in_features for m in enc.mlp if isinstance(m, torch.nn.Linear)] == [61, 128]
    assert enc.mlp[2].out_features == 64
    actions = torch.arange(48, dtype=torch.float32).view(1, 12, 4) + 1
    x = enc.prefix(actions, torch.tensor([3]))
    assert x.shape == (1, 61)
    assert torch.equal(x[0, :12], actions[0, :3].flatten()) and not bool(x[0, 12:48].any())  # forbidden values removed
    assert x[0, 48:60].tolist() == [1.0] * 3 + [0.0] * 9 and float(x[0, 60]) == pytest.approx(3 / 12)
    with pytest.raises(ValueError):
        enc.prefix(actions, torch.tensor([13]))


def test_current_time_gaussians_ignore_future_actions_and_heads_start_as_persistence():
    decoder = _decoder()
    slots, count = torch.randn(2, 2, 8), torch.tensor([4, 12])
    a, b = decoder(slots, _actions(2, count, 0), count), decoder(slots, _actions(2, count, 1), count)
    for name in STATIC_FIELDS:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    dyn = a.group == DYNAMIC_GROUP
    assert not torch.allclose(a.delta01[:, dyn], b.delta01[:, dyn]) and not torch.allclose(a.delta12[:, dyn], b.delta12[:, dyn])
    assert not a.delta01[:, ~dyn].any() and not a.delta12[:, ~dyn].any()  # the scene group never moves
    fresh = GaussianDecoder(decoder.cfg)  # zero-initialised output layers: exactly zero displacement at step 0
    out = fresh(slots, _actions(2, count, 0), count)
    assert not out.delta01.any() and not out.delta12.any()
    no_actions = decoder(slots)  # diagnostic panels: persistence, same current-time Gaussians
    assert not no_actions.delta01.any() and torch.equal(no_actions.xyz, a.xyz)


def test_first_displacement_ignores_second_interval_actions():
    decoder = _decoder()
    slots, count = torch.randn(3, 2, 8), torch.tensor([4, 8, 12])
    base = _actions(3, count, 0)
    changed = base.clone()
    for b, n in enumerate(count.tolist()):
        changed[b, n // 2 : n] = torch.rand(n // 2, 4) * 2 - 1  # second interval only
    a, b = decoder(slots, base, count), decoder(slots, changed, count)
    torch.testing.assert_close(a.delta01, b.delta01, rtol=0, atol=0)
    assert not torch.allclose(a.delta12, b.delta12)


def test_values_beyond_the_valid_prefix_do_not_change_any_output():
    decoder = _decoder()
    slots, count = torch.randn(2, 2, 8), torch.tensor([4, 8])
    clean = _actions(2, count, 0)
    dirty = clean.clone()
    for b, n in enumerate(count.tolist()):
        dirty[b, n:] = 7.0  # forbidden values in the padding
    a, b = decoder(slots, clean, count), decoder(slots, dirty, count)
    for name in (*STATIC_FIELDS, "delta01", "delta12"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_action_inputs_are_rejected_where_they_do_not_belong():
    plain = GaussianDecoder(DecoderConfig(slot_dim=8, dim=16, depth=1, heads=4, scene=GROUP, dynamic=GROUP, num_slots=2))
    slots, count = torch.randn(1, 2, 8), torch.tensor([4])
    with pytest.raises(ValueError, match="not action-conditioned"):
        plain(slots, _actions(1, count), count)
    decoder = _decoder()
    with pytest.raises(ValueError, match="together"):
        decoder(slots, _actions(1, count))
    with pytest.raises(ValueError, match="2k"):
        decoder(slots, _actions(1, torch.tensor([3])), torch.tensor([3]))
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
    rgb = np.zeros((T, V, H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(T, dtype=np.uint8)[:, None, None, None]  # frame index in the red channel
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
        for key, values in {"rgb": rgb, "depth": depth, "body_id": body, "xpos": xpos, "xquat": quat,
                            "obs": np.tile(np.arange(T, dtype=np.float32)[:, None], (1, 39)), "action": action}.items():
            ep.create_dataset(key, data=values)
    return path


def test_future_windows_align_history_targets_and_actions_for_every_stride(tmp_path):
    path = _write_episode_file(tmp_path / "ep.hdf5")
    T = 40
    future = MetaworldWindowDataset(path, ["ep000"], strides=(2, 4, 6), objective="future")
    original = MetaworldWindowDataset(path, ["ep000"], strides=(2, 4, 6))
    assert len(original) == 3 * (T - 12) and len(future) == 3 * (T - 24)  # eligibility: full history + targets
    for index in range(len(future)):
        _, t, (k,) = future.samples[index]
        s = future[index]
        assert s["meta"]["t_indices"] == [t, t + k, t + 2 * k]
        assert s["meta"]["history_t_indices"] == [t - 2 * k, t - k, t]
        assert s["images"][:, 0, 0, 0, 0].tolist() == [t, t + k, t + 2 * k]  # targets
        assert s["history_images"][:, 0, 0, 0, 0].tolist() == [t - 2 * k, t - k, t]  # encoder input
        n = int(s["action_count"])
        assert n == 2 * k and s["actions"].shape == (MAX_ACTIONS, 4)
        assert s["actions"][:n, 0].tolist() == list(range(t, t + n))  # actions[t:t+2k], in order
        assert not s["actions"][n:].any()
        assert s["probe_state"][:, 0].tolist() == [t - 2 * k, t - k, t]  # probes describe what the encoder sees
        ref = original[original.samples.index(("ep000", t - 2 * k, (k,)))]
        assert torch.equal(ref["motion_score"], s["history_motion_score"])  # masking score from the history alone
        target_ref = original[original.samples.index(("ep000", t, (k,)))]
        for key in ("images", "depth", "motion3d", "motion_weight", "motion_score"):
            assert torch.equal(target_ref[key], s[key]), key  # every supervision field = the target window
    batch = collate([future[0], future[1]])
    assert batch["action_count"].dtype == torch.int64 and batch["history_images"].shape == batch["images"].shape


def test_contract_rejects_actions_beyond_the_prefix(tmp_path):
    path = _write_episode_file(tmp_path / "ep.hdf5")
    future = MetaworldWindowDataset(path, ["ep000"], strides=(2,), objective="future")
    batch = collate([future[0], future[1]])
    batch["actions"][0, int(batch["action_count"][0])] = 1.0
    with pytest.raises(ContractError, match="beyond"):
        validate_batch(batch)
    with pytest.raises(ValueError):
        MetaworldWindowDataset(path, ["ep000"], strides=(8,), objective="future")  # 2k would exceed 12 actions


# ------------------------------------------------------------------------------------------ forward_losses
def _model():
    encoder = Encoder(
        EncoderConfig(image_height=16, image_width=16, patch_size=4, width=16, depth=1, heads=4, num_slots=2, slot_dim=8,
                      drop_path=0)
    )
    return Model(encoder, _decoder())


def _future_batch(batch, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, T, V, _, H, W = batch["images"].shape
    batch = dict(batch)
    batch["images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8, generator=g)
    batch["history_images"] = torch.randint(0, 256, batch["images"].shape, dtype=torch.uint8, generator=g)
    batch["history_depth"] = torch.ones(B, T, V, 1, H, W)
    batch["history_motion_score"] = torch.rand(B, T, V, 1, H, W, generator=g)
    batch["action_count"] = torch.tensor([4, 8])
    batch["actions"] = _actions(B, batch["action_count"], seed)
    return batch


def _cfg(**aug):
    cfg = load_config([REPO / "configs/metaworld/base.yaml", REPO / "configs/metaworld/variants/action_future.yaml"])
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


def test_future_targets_and_actions_cannot_change_the_encoding(batch, monkeypatch):
    _fake(monkeypatch)
    model = _model().eval()
    a = _future_batch(batch, 0)
    b = dict(a)
    b.update(images=torch.randint(0, 256, a["images"].shape, dtype=torch.uint8), depth=a["depth"] * 1.5,
             motion3d=a["motion3d"] + 0.01, motion_score=torch.rand_like(a["motion_score"]),
             actions=_actions(2, a["action_count"], 5))
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
    future = MetaworldWindowDataset(path, ["ep000"], strides=(2,), with_eval=True, objective="future")
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
