"""SNeRL-code SAC (s4d/rl/snerl_sac.py): unchanged upstream import, adapters, config resolution and agent updates."""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest
import torch

from s4d.rl import snerl_sac
from s4d.rl.protocol import resolve_config

# third_party/snerl @ d01533b: the agent, encoders and utilities run exactly as pinned
UPSTREAM_SHA256 = {
    "curl_sac.py": "863fb2e9da6a6151eb6c48edfc8eeecac920cda082941d7b64c58217037d297d",
    "encoder.py": "c28270c82505c193686882d7a660de1f59b4e86a82b0cbe3528b754098f1ca1e",
    "utils.py": "f670d050fe34f2efb3caef71abf9265dd4d24a34f5053c10ae34cd16f82c7975",
}


def test_upstream_files_are_the_pinned_ones_and_import_in_isolation():
    for name, digest in UPSTREAM_SHA256.items():
        assert hashlib.sha256((snerl_sac.UPSTREAM / name).read_bytes()).hexdigest() == digest
    up = snerl_sac.load_upstream()
    assert set(up) == {"encoder", "utils", "curl_sac"}
    for name in ("encoder", "utils", "curl_sac", "gym", "cv2", "skimage"):  # nothing generic left behind
        module = sys.modules.get(name)
        assert module is None or not str(getattr(module, "__file__", "") or "").startswith(str(snerl_sac.UPSTREAM))
        assert not getattr(module, "__s4d_stub__", False)
    assert up["curl_sac"].utils is up["utils"]


def test_algorithm_config_replaces_drm_sections_and_keeps_the_protocol():
    cfg = resolve_config("sweep-into", "cnn", 2000, ["train.batch_size=64"], algorithm="snerl_sac")
    drm = resolve_config("sweep-into", "cnn", 2000, [])
    assert cfg["algorithm"] == "snerl_sac" and drm["algorithm"] == "drm"
    assert cfg["env"] == drm["env"] and cfg["eval"] == drm["eval"] and cfg["vision"] == drm["vision"]
    assert "nstep" not in cfg["train"] and "expectile" not in cfg["agent"] and "max_perturb_factor" not in cfg["agent"]
    assert cfg["train"]["batch_size"] == 64 and cfg["train"]["replay_size"] == 100000
    assert cfg["agent"]["hidden_dim"] == 1024 and cfg["agent"]["init_temperature"] == 0.1
    assert cfg["agent"]["image_size"] == cfg["agent"]["pre_transform_image_size"] == cfg["env"]["image_size"] == 128


def _register():
    up = snerl_sac.load_upstream()
    snerl_sac._register(up)
    return up


@pytest.mark.parametrize("mode", ["stack", "frame"])
def test_frozen_feature_head_shapes_logits_and_proprio(mode):
    up = _register()
    snerl_sac._CONTEXT.clear()
    snerl_sac._CONTEXT.update(proprio_dim=4, feature=(mode, 32))
    actor_enc = up["encoder"].make_encoder("frozen", (32,), 189, 4, 32, output_logits=False, multiview=1, frame_stack=3)
    critic_enc = up["encoder"].make_encoder("frozen", (32,), 189, 4, 32, output_logits=True, multiview=1, frame_stack=3)
    x = torch.randn(5, 32) if mode == "stack" else torch.randn(5, 3, 32)
    proprio = torch.randn(5, 4)
    a, c = actor_enc(snerl_sac.Packed(x, proprio)), critic_enc(snerl_sac.Packed(x, proprio))
    assert a.shape == c.shape == (5, 193) and actor_enc.feature_dim == 193
    assert torch.equal(a[:, -4:], proprio) and a[:, :-4].abs().max() <= 1.0  # tanh for the actor
    assert actor_enc.fc.out_features == (63 if mode == "frame" else 189)


def test_pixel_encoder_accepts_128_pixels_and_curl_sees_pixels_only():
    up = _register()
    snerl_sac._CONTEXT.clear()
    snerl_sac._CONTEXT.update(proprio_dim=4)
    enc = up["encoder"].make_encoder("pixel", (9, 128, 128), 189, 4, 32, output_logits=True, multiview=1, frame_stack=3)
    assert enc.fc.in_features == 32 * 57 * 57
    pixels = torch.randint(0, 255, (2, 9, 128, 128)).float()
    assert enc(snerl_sac.Packed(pixels, torch.zeros(2, 4))).shape == (2, 193)
    assert enc(pixels).shape == (2, 189)
    cropped = up["utils"].random_crop(pixels, 128)  # README size: the CURL random crop is the identity
    assert torch.equal(cropped, pixels)


@pytest.mark.parametrize("encoder", ["cnn"])
def test_agent_updates_and_state_round_trip(tmp_path, encoder):
    device = torch.device("cuda")
    cfg = resolve_config("hammer", encoder, 2000, ["train.log_every_steps=1"], algorithm="snerl_sac")
    sac = snerl_sac.SNeRLSAC(cfg, 4, 4, device)
    assert hasattr(sac.agent, "CURL")  # end-to-end pixel encoder: CURL as upstream

    def batches():
        while True:
            obs = torch.randint(0, 255, (8, 9, 128, 128), dtype=torch.uint8)
            yield (obs, torch.randn(8, 4), torch.rand(8, 4) * 2 - 1, torch.randn(8, 1), torch.ones(8, 1),
                   obs.clone(), torch.randn(8, 4))

    stream = batches()
    before = [p.detach().clone() for p in sac.agent.critic.parameters()]
    metrics = {}
    for step in range(4):
        metrics = sac.update(stream, step)
    assert "train/curl_loss" in metrics and "train_critic/loss" in metrics
    assert any(not torch.equal(b, p) for b, p in zip(before, sac.agent.critic.parameters()))
    state = sac.state_dict()
    other = snerl_sac.SNeRLSAC(cfg, 4, 4, device)
    other.load_state_dict(state)
    x = torch.randint(0, 255, (3, 9, 128, 128), dtype=torch.uint8, device=device)
    proprio = np.zeros((3, 4), np.float32)
    assert np.allclose(sac.act(x, proprio, 0, eval_mode=True), other.act(x, proprio, 0, eval_mode=True))
    other.load_policy(sac.policy_state())
