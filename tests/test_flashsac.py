"""FlashSAC (s4d/rl/flashsac.py): pinned upstream import, the paper's vision configuration on the shared protocol, the
visual adapter per family, GPU updates (encoder gradients, frozen backbones untouched, no augmentation), reward
normalisation, state round trip, deterministic evaluation and the end-to-end CLI."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from s4d.rl import flashsac
from s4d.rl.protocol import resolve_config

REPO = Path(__file__).resolve().parents[1]
# third_party/flashsac @ 87edc90: the agent, networks, layers, updates, reward normalisation and schedules run exactly
# as pinned (types.py, agents/__init__.py, base_agent.py and torch_buffer.py are imported on the way)
UPSTREAM_SHA256 = {
    "flash_rl/agents/flashSAC/agent.py": "e1bb9c49cdf1b72507705403801a5dc993471a89f69dbcee973f14949895595a",
    "flash_rl/agents/flashSAC/network.py": "4068098466f195fac2c7d91e3cb1abe0c4daf42c4d1cd85f3940dca9edfd4c98",
    "flash_rl/agents/flashSAC/layer.py": "7af7b2954c22a5ae56595b8d3139017e2a44b17dbd63c2f54c4e65dbd17ef7cc",
    "flash_rl/agents/flashSAC/update.py": "6134612e4ad43089dd44ce06cb6644546e864a3de13bb12525cb631e65aa30ef",
    "flash_rl/agents/utils/network.py": "02b1e8c27ae2d75ba39969a5ea6b21e1bc596dbab71fdd6c5dc9cd5d7aab1d69",
    "flash_rl/agents/utils/reward_normalization.py": "b673f5fba40468f6bb183d3cc82daa05111e95407ccb6b73de854f338c7b70b8",
    "flash_rl/agents/utils/scheduler.py": "b3732509dbe01a01d76b3018d062386287bf7cb669973e2a465192c584030811",
    "flash_rl/agents/utils/distribution.py": "30c2987d80238ccc9c59dd6f7a9dd4056ed01773901d3d08a0db8c40f61b76ed",
    "flash_rl/types.py": "0c1944097870ab47f5180a09ab979923a4de22b39b4a1314f35c4135bc278cd1",
    "flash_rl/agents/__init__.py": "921bf41fd77b2efdd7714dc966f69a0fbb3208f6feb4cb9088f7b1289cca81b8",
    "flash_rl/agents/base_agent.py": "d58f98e02eed8887c16f5f7d83d97f16e4a7ac0d104dc40d1809bf3832a72dfc",
    "flash_rl/buffers/torch_buffer.py": "663ec003992ebda3c029c963eb93db8b902bc666cdd76f451e7cb348d37439a9",
    "LICENSE": "5e7a0ca5f1955091e476af95148f32b48b58feb851a36e09b5c89a0c05bd4421",  # MIT, Holiday Robotics 2026
}
CUDA = torch.device("cuda")


# ------------------------------------------------------------------------------------------------ upstream import
def test_upstream_files_are_the_pinned_ones_and_import_in_isolation():
    root = flashsac.upstream_dir()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert head.stdout.strip() == flashsac.PINNED_COMMIT
    for name, digest in UPSTREAM_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    up = flashsac.load_upstream()
    assert set(up) == set(flashsac.UPSTREAM_MODULES)
    for key, module in up.items():
        assert Path(module.__file__).resolve().is_relative_to(root.resolve()), key
    for name in flashsac.STUB_MODULES:  # stubs never outlive the import
        assert not getattr(sys.modules.get(name), "__s4d_stub__", False)
    assert up["agent"].update_critic is up["update"].update_critic and up["agent"].Network is up["nets"].Network
    assert flashsac.load_upstream()["agent"] is up["agent"]
    assert not list(root.rglob("__pycache__"))  # no bytecode written into the pinned checkout
    assert "torch" in up["agent"].__dict__ and "jax" not in up["agent"].__dict__  # the PyTorch agent


def test_missing_jax_and_omegaconf_are_stubbed_only_during_the_import():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        sys.modules["jax"] = None  # an import of these raises: treated as not installed
        sys.modules["omegaconf"] = None
        from s4d.rl import flashsac
        up = flashsac.load_upstream()
        assert sys.modules["jax"] is None and sys.modules["omegaconf"] is None and "jax.numpy" not in sys.modules
        assert sys.modules["flash_rl.types"].jnp.__s4d_stub__
        try:
            sys.modules["flash_rl.agents"].create_agent(None, None, {{}}, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the omegaconf stub must refuse create_agent")
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


# ------------------------------------------------------------------------------------------------ configuration
def test_config_is_the_paper_vision_setting_on_the_shared_protocol():
    cfg = resolve_config("hammer", "cnn", 2000, [], algorithm="flashsac")
    drm = resolve_config("hammer", "cnn", 2000, [])
    assert cfg["algorithm"] == "flashsac" and drm["algorithm"] == "drm"
    assert cfg["env"] == drm["env"] and cfg["eval"] == drm["eval"] and cfg["vision"] == drm["vision"]
    env = cfg["env"]
    assert (env["image_size"], env["frame_stack"], env["action_repeat"], env["max_episode_steps"]) == (128, 3, 2, 250)
    assert env["proprio_indices"] == [0, 1, 2, 3]
    a, t = cfg["agent"], cfg["train"]
    assert "nstep" not in t and "expectile" not in a and "feature_dim" in a
    # Table 11 (vision): batch 256, UTD 0.5, 3-step returns, 1M replay, encoder output 50
    assert (a["sample_batch_size"], t["updates_per_interaction_step"], a["n_step"]) == (256, 0.5, 3)
    assert (a["buffer_max_length"], a["feature_dim"], t["num_train_steps"]) == (1000000, 50, 1000000)
    up = flashsac.load_upstream()
    fields = set(up["agent"].FlashSACConfig.__dataclass_fields__)
    assert set(a) - fields == {"feature_dim"}  # every other agent entry is an upstream FlashSACConfig field
    # differences from the upstream code defaults are exactly the documented ones
    upstream = yaml.safe_load((flashsac.upstream_dir() / "configs/agent/flashSAC.yaml").read_text())
    base = yaml.safe_load((flashsac.upstream_dir() / "configs/flashSAC_base.yaml").read_text())
    def value(v):  # PyYAML reads upstream's "3e-4" as a string
        try:
            return float(v) if isinstance(v, str) else v
        except ValueError:
            return v

    differs = {k for k in a if k in upstream and not str(upstream[k]).startswith("$") and value(upstream[k]) != a[k]}
    assert differs == {"sample_batch_size", "use_amp", "use_compile"}
    assert (a["gamma"], a["n_step"]) == (base["gamma"], 3) and base["n_step"] == 1
    fcfg = flashsac.flashsac_config(up, cfg, CUDA)
    assert (fcfg.critic_min_v, fcfg.critic_max_v, fcfg.critic_num_bins) == (-5.0, 5.0, 101)
    assert (fcfg.learning_rate_warmup_step, fcfg.learning_rate_decay_step) == (0, 500000)  # rate x 1M steps x UTD
    assert fcfg.buffer_max_length == 1 and fcfg.buffer_device_type == "cpu"  # upstream's own buffer is unused


# ------------------------------------------------------------------------------------------------ families
def write_export(path: Path) -> Path:
    from s4d.model.encoder import Encoder, EncoderConfig

    cfg = EncoderConfig(image_height=128, image_width=128, width=32, depth=1, heads=2, slot_dim=16, num_frames=3)
    torch.manual_seed(0)
    encoder = Encoder(cfg)
    torch.save({"format": "splatter4d-encoder-v1", "encoder_config": asdict(cfg), "state_dict": encoder.state_dict(),
                "step": 0, "frame_strides": [2, 4, 6]}, path)
    return path


class FrameFeatures:
    """Stand-in for a per-frame frozen backbone on single-camera pixels (ReViWo / DynaMo / 3D-MVP B layout)."""

    def __init__(self, cfg, device):
        self.atom_shape, self.atom_dtype, self.atom_frame_stack, self.mode, self.dim = (5,), np.float16, 3, "frame", 5
        self.weight = torch.randn(48, 5, generator=torch.Generator().manual_seed(0)).to(device)

    def __call__(self, obs):
        b = obs.shape[0]
        pooled = F.avg_pool2d(obs.float().reshape(b * 3, 3, 128, 128) / 255.0, 32).reshape(b, 3, 48)
        return (pooled @ self.weight).half()


class ViewFeatures(FrameFeatures):
    """Stand-in for a per-frame frozen backbone on the multi-view dict observation (SNeRL / 3D-MVP A layout)."""

    def __call__(self, obs):
        assert isinstance(obs, dict) and obs["rgb"].shape[1:3] == (3, 3) and obs["c2w"].shape[-2:] == (4, 4)
        b = obs["rgb"].shape[0]
        pooled = F.avg_pool2d(obs["rgb"][:, :, 0].float().reshape(b * 3, 3, 128, 128) / 255.0, 32).reshape(b, 3, 48)
        return (pooled @ self.weight + obs["c2w"][:, :, 0, 0, 3:4]).half()


FAMILIES = {"cnn": "cnn", "stack": "splatter4d", "frame": "reviwo", "views": "snerl"}


def make_agent(kind: str, tmp_path: Path, monkeypatch, extra=()):
    import s4d.rl.snerl_sac

    sets = ["wandb.enabled=false", *extra]
    if kind == "stack":  # the real FrozenFeatures path (VisionEncoderAdapter) on a small splatter4d export
        sets.append(f"vision.export_path={write_export(tmp_path / 'encoder.pt')}")
    if kind == "frame":
        monkeypatch.setattr(s4d.rl.snerl_sac, "FrozenFeatures", FrameFeatures)
    if kind == "views":
        monkeypatch.setattr(s4d.rl.snerl_sac, "FrozenFeatures", ViewFeatures)
    cfg = resolve_config("hammer", FAMILIES[kind], 2000, sets, algorithm="flashsac")
    torch.manual_seed(0)
    return cfg, flashsac.FlashSAC(cfg, 4, 4, CUDA)


def observation(agent, n: int, seed: int = 0):
    """A batched raw observation of the agent's family (pixels or the multi-view dict)."""
    g = torch.Generator().manual_seed(seed)
    if agent.cfg["env"].get("views", 1) == 3:
        rgb = torch.randint(0, 256, (n, 3, 3, 3, 128, 128), dtype=torch.uint8, generator=g)
        return {"rgb": rgb.numpy(), "c2w": torch.randn(n, 3, 3, 4, 4, generator=g).numpy(),
                "K": torch.eye(3).expand(n, 3, 3, 3, 3).numpy()}
    return torch.randint(0, 256, (n, 9, 128, 128), dtype=torch.uint8, generator=g).numpy()


def inputs_of(agent, n: int, seed: int = 0):
    from s4d.rl.evaluate import policy_inputs

    return policy_inputs(agent, observation(agent, n, seed))


def batch_stream(agent, b: int = 8, seed: int = 0):
    """Replay batches as ``replay_iterator`` yields them (atoms already stacked; 3-step rewards, discount 0.99^3)."""
    k = seed
    while True:
        obs, nxt = inputs_of(agent, b, k), inputs_of(agent, b, k + 1)
        g = torch.Generator().manual_seed(k)
        yield (obs, torch.randn(b, 4, generator=g), torch.rand(b, 4, generator=g) * 2 - 1,
               torch.rand(b, 1, generator=g) * 3, torch.full((b, 1), 0.99**3), nxt, torch.randn(b, 4, generator=g))
        k += 2


@pytest.mark.parametrize("kind", ["cnn", "stack", "frame", "views"])
def test_adapter_shapes_per_family(tmp_path, monkeypatch, kind):
    cfg, agent = make_agent(kind, tmp_path, monkeypatch)
    up = agent.up
    enc = agent.feature_encoder
    assert enc.bottleneck.out_features == 50 and agent.feature_dim == 50
    critic, target = agent.agent._critic.network, agent.agent._target_critic.network
    assert isinstance(critic, flashsac.EncodedCritic) and critic.encoder is enc
    core = up["network"].FlashSACDoubleCritic
    assert isinstance(critic.core, core) and isinstance(target.core, core)
    assert critic.core.embedder.w.weight.shape == (2, 256, 54 + 4)  # [50 features, 4 proprio] + 4-d action
    actor = agent.agent._actor.network
    assert isinstance(actor, up["network"].FlashSACActor) and actor.embedder.w.w.weight.shape == (128, 54)
    assert agent.agent._cfg.temp_target_entropy == pytest.approx(flashsac.target_entropy(4, 0.15))
    x = inputs_of(agent, 2)
    if kind == "cnn":
        from s4d.rl.encoders import ConvNet

        assert agent.use_pixels and agent.features is None and isinstance(enc.backbone, ConvNet)
        assert [m.out_channels for m in enc.backbone.convnet if isinstance(m, torch.nn.Conv2d)] == [32] * 4
        assert enc.bottleneck.in_features == 32 * 57 * 57 and x.dtype == torch.uint8 and x.shape == (2, 9, 128, 128)
    else:
        assert not agent.use_pixels and enc.backbone is None and x.dtype == torch.float16
        if kind == "stack":  # the joint feature of the whole history: one atom per step
            assert agent.features.atom_frame_stack == 1 and x.shape == (2, *agent.features.atom_shape)
            assert enc.bottleneck.in_features == agent.features.dim
        else:  # per-frame features of the 3 frames, concatenated
            assert x.shape == (2, 3, 5) and enc.bottleneck.in_features == 15
    obs = agent.policy_observation(x, np.zeros((2, 4), np.float32))
    assert obs.shape == (2, 54) and obs.dtype == torch.float32
    for eval_mode in (True, False):
        action = agent.act(x, np.zeros((2, 4), np.float32), step=0, eval_mode=eval_mode)
        assert action.shape == (2, 4) and np.all(np.abs(action) <= 1)


def _hashes(module: torch.nn.Module) -> dict[str, str]:
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in module.state_dict().items()}


@pytest.fixture
def no_augmentation(monkeypatch):
    """Every stochastic image augmentation of the repository (and the sampling ops they are built on) raises."""
    calls = []

    def forbid(name):
        def raiser(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not run in FlashSAC")
        return raiser

    monkeypatch.setattr("s4d.rl.agent.RandomShiftsAug.forward", forbid("RandomShiftsAug"))
    monkeypatch.setattr("s4d.train.augment.random_shift", forbid("random_shift"))
    monkeypatch.setattr("s4d.train.augment.random_resized_crop", forbid("random_resized_crop"))
    monkeypatch.setattr(F, "grid_sample", forbid("grid_sample"))
    monkeypatch.setattr(F, "affine_grid", forbid("affine_grid"))
    return calls


@pytest.mark.parametrize("kind", ["cnn", "stack", "frame", "views"])
def test_gpu_update_trains_encoder_actor_critic_and_never_the_frozen_backbone(tmp_path, monkeypatch, no_augmentation,
                                                                              kind):
    cfg, agent = make_agent(kind, tmp_path, monkeypatch)
    a = agent.agent
    frozen = None
    if kind == "stack":
        backbone = agent.features.adapter.backbone
        frozen = _hashes(backbone)
        assert not any(p.requires_grad for p in backbone.parameters())
        optimised = {id(p) for net in (a._actor, a._critic, a._temperature) for g in net.optimizer.param_groups
                     for p in g["params"]}
        assert not optimised & {id(p) for p in agent.features.adapter.parameters()}
    groups = a._critic.optimizer.param_groups
    assert len(groups) == 2 and {id(p) for p in groups[1]["params"]} == {id(p) for p in agent.feature_encoder.parameters()}
    before = {
        "encoder": [p.detach().clone() for p in agent.feature_encoder.parameters()],
        "actor": [p.detach().clone() for p in a._actor.network.parameters()],
        "critic": [p.detach().clone() for p in a._critic.network.core.parameters()],
        "target": [p.detach().clone() for p in a._target_critic.network.parameters()],
        "temperature": [p.detach().clone() for p in a._temperature.network.parameters()],
    }
    for r in (1.0, 3.0, 2.0):
        agent.observe_reward(r, False)
    stream = batch_stream(agent)
    for _ in range(2):  # update 0 updates the actor (period 2), update 1 does not
        agent.update(stream)
    metrics = agent.metrics()
    assert {"actor/loss", "critic/loss", "temperature/value"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())
    # the encoder (CNN convolutions + bottleneck, or a frozen family's bottleneck) receives gradients and moves
    for p, b in zip(agent.feature_encoder.parameters(), before["encoder"]):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0 and not torch.equal(p.detach(), b)
    for name, module in (("actor", a._actor.network), ("critic", a._critic.network.core),
                         ("target", a._target_critic.network), ("temperature", a._temperature.network)):
        assert any(not torch.equal(b, p.detach()) for b, p in zip(before[name], module.parameters())), name
    if frozen is not None:
        assert _hashes(agent.features.adapter.backbone) == frozen
    assert groups[0]["lr"] == groups[1]["lr"]  # the encoder follows the critic's schedule
    x = inputs_of(agent, 2)
    agent.act(x, np.zeros((2, 4), np.float32), step=0, eval_mode=False)
    assert no_augmentation == []


def test_actor_loss_never_reaches_the_encoder_and_critic_paths_agree(tmp_path, monkeypatch):
    cfg, agent = make_agent("cnn", tmp_path, monkeypatch)
    a = agent.agent
    captured = {}
    original = agent.up["agent"]._update_networks

    def capture(batch, **kwargs):
        captured.update(batch)
        return original(batch=batch, **kwargs)

    monkeypatch.setattr(agent.up["agent"], "update_critic", lambda **kwargs: {})  # actor + temperature only
    monkeypatch.setattr(agent.up["agent"], "_update_networks", capture)
    for r in (1.0, 2.0):
        agent.observe_reward(r, False)
    before = [p.detach().clone() for p in agent.feature_encoder.parameters()]
    actor_before = [p.detach().clone() for p in a._actor.network.parameters()]
    stream = batch_stream(agent)
    raw = next(batch_stream(agent))
    agent.update(stream)
    assert all(p.grad is None and torch.equal(p.detach(), b) for p, b in zip(agent.feature_encoder.parameters(), before))
    assert any(not torch.equal(p.detach(), b) for p, b in zip(a._actor.network.parameters(), actor_before))
    b = captured["action"].shape[0]
    assert not captured["actor_observation"].requires_grad and captured["actor_observation"].shape == (b, 54)
    assert torch.equal(captured["observation"][:, -1], torch.arange(b, device=CUDA, dtype=torch.float32))
    assert torch.equal(captured["next_observation"][:, -1], torch.arange(b, 2 * b, device=CUDA, dtype=torch.float32))
    assert torch.equal(captured["terminated"], torch.zeros(b, device=CUDA))  # time limit: bootstrapped through
    scale = agent.reward_scale()
    assert torch.allclose(captured["reward"], raw[3].reshape(-1).to(CUDA) / scale)  # upstream reward normalisation
    # the gradient path of the online critic computes the same values as the packed no-grad features
    critic = a._critic.network
    packed = torch.cat([captured["observation"], captured["next_observation"]])
    actions = torch.rand(2 * b, 4, device=CUDA) * 2 - 1
    with torch.no_grad():
        constant, _ = critic(packed, actions, training=False)
    obs = raw[0].to(CUDA)
    critic.grad_features = agent.feature_encoder(obs)
    try:
        graded, _ = critic(packed, actions, training=False)
    finally:
        critic.grad_features = None
    assert graded.requires_grad and torch.allclose(graded, constant, atol=1e-5)
    with pytest.raises(RuntimeError):
        critic(packed, actions, training=False)  # outside an update the gradient path does not exist


def test_reward_normaliser_sees_raw_agent_step_rewards_and_resets_at_the_time_limit(tmp_path, monkeypatch):
    cfg, agent = make_agent("frame", tmp_path, monkeypatch)
    norm = agent.agent.reward_normalizer
    rewards = [1.0, 2.0, 4.0]
    for r in rewards:
        agent.observe_reward(r, False)
    expected = 0.0
    for r in rewards:
        expected = 0.99 * expected + r
    assert float(norm.G_r) == pytest.approx(expected) and float(norm.G_r_max) == pytest.approx(expected)
    agent.observe_reward(0.5, True)  # the time limit (a truncation) ends the discounted return (upstream: done)
    agent.observe_reward(1.5, False)
    assert float(norm.G_r) == pytest.approx(0.99 * 0.5 + 1.5)
    scale = agent.reward_scale()
    assert scale >= float(norm.G_r_max) / 5.0 and torch.allclose(
        norm.normalize_rewards(torch.ones(3, device=CUDA)).float(), torch.full((3,), 1 / scale, device=CUDA))


def test_state_round_trip_resumes_exactly_and_evaluation_is_deterministic(tmp_path, monkeypatch):
    from s4d.lifecycle_cleanup import TRAINERS, torch_policy_check
    from s4d.rl.algorithms import evaluation_policy

    cfg, agent = make_agent("frame", tmp_path, monkeypatch)
    for r in (1.0, 2.0, 0.5):
        agent.observe_reward(r, False)
    stream = batch_stream(agent)
    agent.update(stream)
    x, proprio = inputs_of(agent, 3, seed=9), np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    agent.act(x[:1], proprio[:1], step=1, eval_mode=False)  # advance the noise-repetition state
    buffer = io.BytesIO()
    torch.save(agent.state_dict(), buffer)
    buffer.seek(0)
    other = flashsac.FlashSAC(cfg, 4, 4, CUDA)
    other.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=False))
    mine, theirs = agent.state_dict(), other.state_dict()

    def flat(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from flat(v, f"{prefix}{k}/")
        else:
            yield prefix, tree

    left, right = dict(flat(mine)), dict(flat(theirs))
    assert left.keys() == right.keys()
    for key, value in left.items():
        if torch.is_tensor(value):
            assert torch.equal(value.cpu(), right[key].cpu()), key
        else:
            assert value == right[key], key
    # deterministic evaluation: tanh(mean), no noise state touched, identical after a reload
    noise = agent.agent._cached_noise.clone(), int(agent.agent._cur_noise_repeat_count)
    first = agent.act(x, proprio, step=0, eval_mode=True)
    assert np.array_equal(first, agent.act(x, proprio, step=0, eval_mode=True))
    assert np.array_equal(first, other.act(x, proprio, step=0, eval_mode=True))
    assert torch.equal(agent.agent._cached_noise, noise[0]) and int(agent.agent._cur_noise_repeat_count) == noise[1]
    # training continues identically after the reload
    batch = next(stream)
    torch.manual_seed(3)
    agent.update(iter([batch]))
    torch.manual_seed(3)
    other.update(iter([batch]))
    for p, q in zip(agent.agent._critic.network.parameters(), other.agent._critic.network.parameters()):
        assert torch.allclose(p, q, atol=1e-6)
    # the evaluation snapshot (encoder + actor) through the evaluator's dispatch
    snapshot = tmp_path / "step_0010000.pt"
    torch.save({"step": 10000, "algorithm": "flashsac", "policy": agent.policy_state()}, snapshot)
    keys = list(TRAINERS["train_flashsac.py"].policy_keys)
    assert keys == ["encoder", "actor"] and torch_policy_check(str(snapshot), 10000, keys)["torch_tensors"] > 0
    policy = evaluation_policy(cfg, 4, 4, CUDA)
    assert isinstance(policy, flashsac.FlashSAC)
    policy.train(False)
    policy.load_policy(torch.load(snapshot, map_location="cuda", weights_only=True)["policy"])
    assert np.array_equal(policy.act(x, proprio, step=0, eval_mode=True), agent.act(x, proprio, step=0, eval_mode=True))


def test_training_actions_use_upstream_noise_repetition(tmp_path, monkeypatch):
    cfg, agent = make_agent("frame", tmp_path, monkeypatch)
    a = agent.agent
    x, proprio = inputs_of(agent, 1), np.zeros((1, 4), np.float32)
    a._cur_noise_repeat_count.fill_(0)
    first = agent.act(x, proprio, step=1, eval_mode=False)
    n = int(a._cur_noise_repeat_n)
    assert 1 <= n <= 16 and int(a._cur_noise_repeat_count) == 1
    repeats = [agent.act(x, proprio, step=2 + i, eval_mode=False) for i in range(n - 1)]
    assert all(np.array_equal(first, r) for r in repeats)  # the same noise is held for n steps
    assert not np.array_equal(first, agent.act(x, proprio, step=0, eval_mode=True))


def test_trainer_registration():
    from s4d.lifecycle_cleanup import DEFAULT_REPLAY_PATTERNS, TRAINERS
    from s4d.remote_launch import validate_output_arguments
    from scripts import jobs

    assert "train_flashsac.py" in jobs.RL_TRAINERS
    assert TRAINERS["train_flashsac.py"].policy_keys == ("encoder", "actor")
    assert TRAINERS["train_flashsac.py"].replay_patterns == DEFAULT_REPLAY_PATTERNS  # Replay memmaps / RAM snapshot
    state = {"script": "scripts/train_flashsac.py", "required_results": ["final.json", "train.jsonl"]}
    assert {"final.json", "exit.json", "provenance.json"} <= set(jobs.remote_check_names(state))
    validate_output_arguments("scripts/train_flashsac.py", ["--run-dir", "runs/ds-x"], "ds-x")
    with pytest.raises(ValueError):
        validate_output_arguments("scripts/train_flashsac.py", ["--run-dir", "runs/other"], "ds-x")


# ------------------------------------------------------------------------------------------------ end to end
TINY = [
    "agent.buffer_min_length=100",
    "agent.sample_batch_size=16",
    "agent.buffer_max_length=1000",
    "train.log_every_steps=50",
    "eval.every_steps=125",
    "train.checkpoint_every_steps=125",
    "eval.episodes_per_train_camera=1",
    "eval.episodes_per_heldout_camera=1",
    "eval.episodes_per_trajectory=1",
    "env.image_size=64",
    "wandb.enabled=false",
]


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0], "MUJOCO_GL": "egl"}
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / script), *args], env=env, capture_output=True, text=True, timeout=1200
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return result


def test_cnn_training_evaluation_and_resume(tmp_path):
    run_dir = tmp_path / "flashsac_cli"
    common = ["--task", "hammer", "--encoder", "cnn", "--seed", "3", "--run-dir", str(run_dir)]
    run("scripts/train_flashsac.py", *common, "--set", "train.num_train_steps=250", *TINY)
    final = json.loads((run_dir / "final.json").read_text())
    assert final["completed"] and final["steps"] == 250 and final["updates"] == 75  # UTD 0.5 from 100 transitions
    assert sorted(p.name for p in (run_dir / "snapshots").iterdir()) == ["step_0000125.pt", "step_0000250.pt"]
    records = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    assert records[-1]["step"] == 250 and "train/critic_loss" in records[-1]
    assert records[-1]["rolling_return"] > 0 and records[-1]["reward_scale"] > 0  # raw returns, internal scaling
    assert (run_dir / "replay" / "atom.memmap").is_file(), "CNN replay is a disk memmap"
    run("scripts/eval_rl.py", "--run-dir", str(run_dir), "--train-job", "none", "--workers", "2",
        "--envs-per-worker", "3", "--poll-seconds", "1")
    evals = [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [125, 250] and all(e["episodes"] == 12 for e in evals)
    (run_dir / "final.json").unlink()
    run("scripts/train_flashsac.py", *common, "--set", "train.num_train_steps=375", *TINY)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert {"event": "resumed", "step": 250} == {k: events[0][k] for k in ("event", "step")}
    assert json.loads((run_dir / "final.json").read_text())["updates"] == 75 + 63  # 0.5 per step, counter resumed
