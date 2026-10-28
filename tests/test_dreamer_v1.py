"""Dreamer v1 pilot (s4d/rl/dreamer_v1.py): unchanged upstream import, adapters, replay, updates, state and recurrent
acting."""

from __future__ import annotations

import fnmatch
import hashlib
import subprocess
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from s4d.rl import dreamer_v1
from s4d.rl.protocol import resolve_config

REPO = Path(__file__).resolve().parents[1]
# third_party/dreamer-pytorch @ b51af83: the models, imagination/return utilities, preprocessing and replay sampling run
# exactly as pinned
UPSTREAM_SHA256 = {
    "models.py": "da7d1bed1e8a42169d357d3252572faaf65f5cdc00194858b1a6ed60eb42235e",
    "utils.py": "2ef824c59cf39687e33be78791581a1435a6e4835a8f7940e4d1fbecb73faa1d",
    "env.py": "339ce5c726e96b600602fe490fb3bbf4cc727c7b577ad407f395189444203b8c",
    "memory.py": "1048ca64fb8e3cd79b10c1ea76bafed22bcc5b5729b60fdd492e078473887a53",
}
SMALL = ["agent.belief_size=16", "agent.state_size=4", "agent.hidden_size=16", "agent.embedding_size=32",
         "agent.planning_horizon=4", "train.batch_size=3", "train.chunk_size=6", "train.replay_size=200"]


def test_upstream_files_are_the_pinned_ones_and_import_in_isolation():
    root = dreamer_v1.upstream_dir()
    for name, digest in UPSTREAM_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    up = dreamer_v1.load_upstream()
    assert set(up) == {"env", "models", "utils", "memory"}
    for name in ("env", "models", "utils", "memory", "cv2", "plotly"):  # nothing generic left behind
        module = sys.modules.get(name)
        assert module is None or not str(getattr(module, "__file__", "") or "").startswith(str(root))
        assert not getattr(module, "__s4d_stub__", False)
    assert all(sys.modules[f"dreamer_upstream.{name}"] is module for name, module in up.items())
    assert up["memory"].preprocess_observation_ is up["env"].preprocess_observation_  # memory.py's `from env import`
    assert dreamer_v1.load_upstream()["models"] is up["models"]
    assert not list(root.rglob("__pycache__"))  # no bytecode written into the pinned checkout


def test_missing_cv2_and_plotly_are_stubbed_only_during_the_import():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        sys.modules["cv2"] = None  # an import of these raises: treated as not installed
        sys.modules["plotly"] = None
        from s4d.rl import dreamer_v1
        up = dreamer_v1.load_upstream()
        assert sys.modules["cv2"] is None and sys.modules["plotly"] is None
        assert "plotly.graph_objs" not in sys.modules and "plotly.graph_objs.scatter" not in sys.modules
        assert up["utils"].cv2.__s4d_stub__ and up["env"].cv2.__s4d_stub__
        try:
            up["utils"].Scatter()
        except RuntimeError:
            pass
        else:
            raise AssertionError("the plotly stub must refuse to plot")
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def test_algorithm_config_replaces_drm_sections_and_keeps_the_protocol():
    cfg = resolve_config("hammer", "cnn", 0, ["train.batch_size=16"], algorithm="dreamer_v1")
    drm = resolve_config("hammer", "cnn", 0, [])
    assert cfg["algorithm"] == "dreamer_v1" and drm["algorithm"] == "drm"
    assert cfg["env"] == drm["env"] and cfg["eval"] == drm["eval"] and cfg["vision"] == drm["vision"]
    assert "nstep" not in cfg["train"] and "expectile" not in cfg["agent"]
    a, t = cfg["agent"], cfg["train"]
    # SNeRL Table 3
    assert (a["belief_size"], a["state_size"], a["action_noise"], a["model_learning_rate"]) == (128, 30, 0.3, 1e-3)
    assert a["actor_learning_rate"] == a["value_learning_rate"] == 5e-5 and a["adam_epsilon"] == 1e-7
    assert (a["discount"], a["planning_horizon"], t["replay_size"]) == (0.99, 15, 100000)
    assert t["batch_size"] == 16  # command-line override (32 in the file)
    # dreamer-pytorch defaults and the protocol scaling
    assert (a["embedding_size"], a["hidden_size"], a["free_nats"], a["disclam"], a["grad_clip_norm"]) == (
        1024, 200, 3, 0.95, 100.0)
    assert (t["chunk_size"], t["seed_episodes"], a["bit_depth"], a["observation_size"]) == (50, 5, 5, 64)
    horizon = cfg["env"]["max_episode_steps"] // cfg["env"]["action_repeat"]
    assert horizon == 125 and t["collect_interval"] / horizon == pytest.approx(100 / 500)
    assert t["num_train_steps"] == 500000 and t["checkpoint_every_steps"] % horizon == 0


def write_export(path: Path) -> Path:
    from s4d.model.encoder import Encoder, EncoderConfig

    cfg = EncoderConfig(image_height=128, image_width=128, width=32, depth=1, heads=2, slot_dim=16, num_frames=3)
    torch.manual_seed(0)
    encoder = Encoder(cfg)
    torch.save({"format": "splatter4d-encoder-v1", "encoder_config": asdict(cfg), "state_dict": encoder.state_dict(),
                "step": 0, "frame_strides": [2, 4, 6]}, path)
    return path


class FrameFeatures:
    """Stand-in for a per-frame frozen backbone (the SNeRL / ReViWo / DynaMo / 3D-MVP layout): (B, 3, 5) fp16."""

    def __init__(self, cfg, device):
        self.atom_shape, self.atom_dtype, self.atom_frame_stack, self.mode, self.dim = (5,), np.float16, 3, "frame", 5
        self.weight = torch.randn(48, 5, generator=torch.Generator().manual_seed(0)).to(device)

    def __call__(self, obs):
        b = obs.shape[0]
        pooled = F.avg_pool2d(obs.float().reshape(b * 3, 3, 128, 128) / 255.0, 32).reshape(b, 3, 48)
        return (pooled @ self.weight).half()


def make_cfg(encoder: str, tmp_path: Path, extra=()) -> dict:
    sets = list(SMALL) + list(extra)
    if encoder == "splatter4d":
        sets.append(f"vision.export_path={write_export(tmp_path / 'encoder.pt')}")
    return resolve_config("hammer", encoder, 0, sets, algorithm="dreamer_v1")


@pytest.fixture
def per_frame(monkeypatch):
    import s4d.rl.snerl_sac

    monkeypatch.setattr(s4d.rl.snerl_sac, "FrozenFeatures", FrameFeatures)


def make_agent(encoder, tmp_path, request, learn=True):
    if encoder == "reviwo":
        request.getfixturevalue("per_frame")
    cfg = make_cfg(encoder, tmp_path)
    cls = dreamer_v1.DreamerAgent if learn else dreamer_v1.DreamerPolicy
    return cfg, cls(cfg, 4, 4, torch.device("cuda"))


def pixels(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 9, 128, 128), dtype=torch.uint8, generator=g).cuda()


def policy_input(agent, x: torch.Tensor) -> torch.Tensor:
    return x if agent.use_pixels else agent.extract_cacheable_feature(x)


@pytest.mark.parametrize("encoder", ["cnn", "splatter4d", "reviwo"])
def test_adapter_shapes_for_frozen_and_cnn_arms(tmp_path, request, encoder):
    cfg, agent = make_agent(encoder, tmp_path, request)
    models = agent.models
    emb = cfg["agent"]["embedding_size"]
    t = agent.transition_model
    assert isinstance(t, models.TransitionModel) and t.fc_embed_belief_posterior.in_features == 16 + emb + 4
    assert isinstance(agent.observation_model, models.VisualObservationModel)  # the same 64-px decoder for every arm
    decoded = agent.observation_model(torch.zeros(2, 16, device="cuda"), torch.zeros(2, 4, device="cuda"))
    assert decoded.shape == (2, 3, 64, 64)
    reference = models.VisualEncoder(emb, "relu")
    if encoder == "cnn":
        enc = agent.obs_encoder
        assert isinstance(enc, models.VisualEncoder) and agent.use_pixels and agent.atom_stack == 3
        assert enc.conv1.weight.shape == (32, 9, 4, 4) and reference.conv1.weight.shape == (32, 3, 4, 4)
        assert enc.conv1.stride == reference.conv1.stride == (2, 2)
        for name in ("conv2", "conv3", "conv4", "fc"):  # nothing else changed
            assert [p.shape for p in getattr(enc, name).parameters()] == [
                p.shape for p in getattr(reference, name).parameters()]
        # upstream's FreezeParameters list (pre-scripting module objects) shares the adapted convolution's weights
        assert enc.modules[0].weight is enc.conv1.weight and enc.modules[0].weight.shape[1] == 9
        x = pixels(2)
        assert agent.encoder_input(x).shape == (2, 9, 64, 64)
    else:
        enc = agent.obs_encoder
        assert isinstance(enc, models.SymbolicEncoder) and not agent.use_pixels
        if encoder == "splatter4d":  # the joint feature of the whole history: one atom per step
            assert agent.atom_stack == 1 and agent.atom_shape == (16,) and enc.fc1.in_features == 16
            assert not any(p.requires_grad for p in agent.features.adapter.backbone.parameters())
        else:  # per-frame features: the 3-frame stack, flattened
            assert agent.atom_stack == 3 and agent.atom_shape == (5,) and enc.fc1.in_features == 15
        assert enc.fc3.out_features == emb
        x = policy_input(agent, pixels(2))
        assert agent.encoder_input(x).shape == (2, enc.fc1.in_features)
    agent.begin_episodes(2)
    for eval_mode in (True, False):
        action = agent.act(x, np.zeros((2, 4), np.float32), step=0, eval_mode=eval_mode)
        assert action.shape == (2, 4) and np.all(np.abs(action) <= 1)


def test_pixel_rows_and_acting_stacks_are_the_same_frames():
    g = torch.Generator().manual_seed(3)
    frames = torch.randint(0, 256, (4, 3, 3, 128, 128), dtype=torch.uint8, generator=g)
    cpu = dreamer_v1.downsample(frames, 64)
    assert torch.equal(cpu, dreamer_v1.downsample(frames.cuda(), 64).cpu())
    manual = frames.float().reshape(4, 3, 3, 64, 2, 64, 2).mean(dim=(4, 6)).round()
    assert torch.equal(cpu, manual.to(torch.uint8))
    with pytest.raises(ValueError):
        dreamer_v1.downsample(frames[..., :100, :100], 64)


def make_replay(tmp_path, pixel: bool, size: int = 50, stack: int = 3, mode: str = "w+"):
    cls = dreamer_v1.replay_class()
    return cls(tmp_path / "replay", size, (3, 64, 64), None if pixel else (2,), np.float16, stack, 4, 4, 5,
               torch.device("cpu"), mode)


def fill(replay, episodes: int, length: int, reward_offset: float = 0.0) -> None:
    for _ in range(episodes):
        for t in range(length):
            g = replay.steps
            atom = None if replay.pixel_atoms else np.full(2, g, np.float16)
            replay.append(np.full((3, 64, 64), g % 256, np.uint8), atom, np.full(4, g, np.float32),
                          np.full(4, g / 1000, np.float32), g + reward_offset, t == length - 1, t)


def check_batch(replay, batch, length: int, low: int = 0):
    frames, atoms, stack_index, proprio, actions, rewards, nonterminals = (x.numpy() for x in batch)
    L, n = proprio.shape[:2]
    g = proprio[..., 0].astype(np.int64)
    assert np.array_equal(g, g[:1] + np.arange(L)[:, None])  # consecutive rows: never across the write index
    assert g.min() >= low and g.max() < replay.steps
    assert np.array_equal(rewards, g.astype(np.float32))  # rewards of this run's history only (no crash rows)
    assert np.allclose(actions[..., 0], g / 1000)
    assert np.array_equal(nonterminals[..., 0] == 0, (g + 1) % length == 0)  # the time limit closes every episode
    stacked = atoms[stack_index, np.arange(n)[None, :, None]]  # (L, n, S, ...)
    ep = g % length
    expected = np.stack([g - np.minimum(k, ep) for k in range(replay.atom_stack - 1, -1, -1)], axis=-1)
    if replay.pixel_atoms:
        assert frames.size == 0 and np.array_equal(stacked[..., 0, 0, 0], expected % 256)
        assert np.array_equal(atoms[replay.atom_stack - 1:][..., 0, 0, 0], g % 256)  # the targets
    else:
        assert np.array_equal(stacked[..., 0], expected) and np.array_equal(frames[..., 0, 0, 0], g % 256)


@pytest.mark.parametrize("pixel", [True, False])
def test_replay_chunks_stacks_and_crash_resume(tmp_path, pixel):
    np.random.seed(0)
    replay = make_replay(tmp_path, pixel)
    assert sorted(p.name for p in (tmp_path / "replay").iterdir()) == sorted(
        f"{name}.memmap" for name in dreamer_v1.REPLAY_FIELDS if not (pixel and name == "features"))
    fill(replay, episodes=2, length=20)
    for _ in range(20):
        check_batch(replay, replay.sample(4, 5), 20)
    fill(replay, episodes=1, length=20)  # 60 rows in 50 slots: wrapped
    assert replay.full and replay.idx == 10
    for _ in range(20):
        check_batch(replay, replay.sample(4, 5), 20, low=10)
    meta = replay.checkpoint()
    fill(replay, episodes=1, length=15, reward_offset=0.5)  # a crashed process writes past its checkpoint
    reopened = make_replay(tmp_path, pixel, mode="r+")
    reopened.restore(meta)
    assert (reopened.idx, reopened.full, reopened.steps) == (10, True, 60)
    for _ in range(50):  # rows 10..24 were overwritten by the crash: only 25..59 (and their own stack frames) remain
        check_batch(reopened, reopened.sample(4, 5), 20, low=25)
    fill(reopened, episodes=1, length=20)
    for _ in range(50):
        check_batch(reopened, reopened.sample(4, 5), 20, low=30)


def fill_random(replay, episodes: int, length: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        for t in range(length):
            atom = None if replay.pixel_atoms else rng.normal(size=replay.atom_shape).astype(replay.features.dtype)
            replay.append(rng.integers(0, 256, (3, 64, 64), dtype=np.uint8), atom, rng.normal(size=4),
                          rng.uniform(-1, 1, 4), rng.normal(), t == length - 1, t)


@pytest.mark.parametrize("encoder", ["cnn", "splatter4d", "reviwo"])
def test_world_model_actor_and_value_update_on_gpu(tmp_path, request, encoder):
    _, agent = make_agent(encoder, tmp_path, request)
    replay = agent.make_replay(tmp_path / "replay")
    fill_random(replay, episodes=3, length=20)
    groups = {"world_model": agent.param_list, "actor": list(agent.actor_model.parameters()),
              "value": list(agent.value_model.parameters()), "encoder": list(agent.obs_encoder.parameters())}
    before = {k: [p.detach().clone() for p in params] for k, params in groups.items()}
    frozen = None if agent.features is None or not hasattr(agent.features, "adapter") else [
        p.detach().clone() for p in agent.features.adapter.parameters()]
    np.random.seed(0)
    for _ in range(2):
        metrics = agent.update(replay)
    assert set(metrics) >= {"observation_loss", "reward_loss", "kl_loss", "actor_loss", "value_loss"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["kl_loss"] >= 3.0 - 1e-5  # free nats
    for k, params in groups.items():
        assert any(not torch.equal(b, p) for b, p in zip(before[k], params)), k
    if frozen is not None:
        assert all(torch.equal(b, p) for b, p in zip(frozen, agent.features.adapter.parameters()))


def test_state_round_trip_and_policy_snapshot(tmp_path, request):
    from s4d.lifecycle_cleanup import TRAINERS, torch_policy_check

    cfg, agent = make_agent("cnn", tmp_path, request)
    replay = agent.make_replay(tmp_path / "replay")
    fill_random(replay, episodes=2, length=20)
    np.random.seed(0)
    agent.update(replay)
    other = dreamer_v1.DreamerAgent(cfg, 4, 4, torch.device("cuda"))
    other.load_state_dict(agent.state_dict())
    for name, module in agent._networks().items():
        mine, theirs = module.state_dict(), other._networks()[name].state_dict()
        assert list(mine) == list(theirs) and all(torch.equal(mine[k], theirs[k]) for k in mine), name
    for name in ("model_optimizer", "actor_optimizer", "value_optimizer"):
        mine, theirs = agent._modules()[name].state_dict()["state"], other._modules()[name].state_dict()["state"]
        assert mine.keys() == theirs.keys() and all(torch.equal(mine[k]["exp_avg"], theirs[k]["exp_avg"]) for k in mine)
    x, proprio = pixels(3), np.zeros((3, 4), np.float32)

    def acts(policy):
        policy.begin_episodes(3)
        torch.manual_seed(7)
        return [policy.act(x, proprio, step=0, eval_mode=True) for _ in range(2)]

    reference = acts(agent)
    assert all(np.array_equal(a, b) for a, b in zip(reference, acts(other)))
    snapshot = tmp_path / "step_0001000.pt"
    torch.save({"step": 1000, "algorithm": "dreamer_v1", "policy": agent.policy_state()}, snapshot)
    keys = list(TRAINERS["train_dreamer.py"].policy_keys)
    assert torch_policy_check(str(snapshot), 1000, keys)["torch_tensors"] > 0
    policy = dreamer_v1.DreamerPolicy(cfg, 4, 4, torch.device("cuda"))
    assert not hasattr(policy, "observation_model") and not hasattr(policy, "model_optimizer")
    policy.load_policy(torch.load(snapshot, map_location="cuda", weights_only=True)["policy"])
    assert all(np.array_equal(a, b) for a, b in zip(reference, acts(policy)))


def test_recurrent_eval_act_with_index_subsets(tmp_path, request):
    cfg, agent = make_agent("cnn", tmp_path, request, learn=False)
    twin = dreamer_v1.DreamerPolicy(cfg, 4, 4, torch.device("cuda"))
    twin.load_policy(agent.policy_state())
    x, proprio = pixels(3, seed=1), np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    agent.begin_episodes(3)
    assert agent.belief.abs().sum() == 0 and agent.prev_action.abs().sum() == 0
    agent.act(x, proprio, step=0, eval_mode=True, index=[0, 1, 2])
    middle = (agent.belief[1].clone(), agent.posterior_state[1].clone(), agent.prev_action[1].clone())
    twin.begin_episodes(2)  # the same two episodes on their own
    for name in ("belief", "posterior_state", "prev_action"):
        getattr(twin, name)[:] = getattr(agent, name)[[0, 2]]
    torch.manual_seed(11)
    subset = agent.act(x[[0, 2]], proprio[[0, 2]], step=0, eval_mode=True, index=[0, 2])
    torch.manual_seed(11)
    alone = twin.act(x[[0, 2]], proprio[[0, 2]], step=0, eval_mode=True)
    assert np.array_equal(subset, alone)
    assert torch.equal(agent.belief[[0, 2]], twin.belief) and torch.equal(agent.prev_action[[0, 2]], twin.prev_action)
    assert all(torch.equal(a, b) for a, b in zip(middle, (agent.belief[1], agent.posterior_state[1],
                                                          agent.prev_action[1])))  # finished episode untouched
    with pytest.raises(ValueError):
        agent.act(x, proprio, step=0, eval_mode=True, index=[0, 1])


def test_evaluation_suite_drives_the_recurrent_policy(tmp_path, request):
    from s4d.rl.algorithms import evaluation_policy
    from s4d.rl.evaluate import evaluation_suite

    cfg = make_cfg("cnn", tmp_path)
    agent = evaluation_policy(cfg, 4, 4, torch.device("cuda"))
    assert isinstance(agent, dreamer_v1.DreamerPolicy) and agent.recurrent
    agent.train(False)
    calls = []
    act = agent.act

    def recording_act(inputs, proprio, step, eval_mode, index=None):
        calls.append((len(inputs), list(index), eval_mode, agent.belief.shape[0]))
        return act(inputs, proprio, step, eval_mode, index=index)

    agent.act = recording_act

    class Pool:  # EnvPool.run's protocol: waves of 4, episodes ending at different steps
        size = 4

        def run(self, policy, episodes):
            rng = np.random.default_rng(0)
            for start in range(0, len(episodes), self.size):
                wave = list(range(start, min(start + self.size, len(episodes))))
                policy.begin_wave(len(wave))
                active, t = list(range(len(wave))), 0
                while active:
                    policy.active = list(active)
                    obs = rng.integers(0, 256, (len(active), 9, 128, 128), dtype=np.uint8)
                    assert policy(obs, np.zeros((len(active), 4), np.float32)).shape == (len(active), 4)
                    t += 1
                    active = [k for k in active if t < 2 + k]
            return [0.0] * len(episodes), [1.0] * len(episodes)

    eval_cfg = {"train_episodes_total": 6, "episodes_per_heldout_camera": 0, "episodes_per_trajectory": 0}
    metrics = evaluation_suite(Pool(), agent, 0, 0, 0, eval_cfg)
    assert metrics["episodes"] == 6 and metrics["train_cameras_success"] == 0.0
    assert calls[:5] == [(4, [0, 1, 2, 3], True, 4)] * 2 + [(3, [1, 2, 3], True, 4), (2, [2, 3], True, 4),
                         (1, [3], True, 4)]
    assert (2, [0, 1], True, 2) in calls  # second wave: two episodes, fresh state


def test_trainer_registration_matches_the_replay_and_snapshot_layout(tmp_path):
    from s4d.lifecycle_cleanup import DEFAULT_REPLAY_PATTERNS, TRAINERS
    from s4d.remote_launch import validate_output_arguments
    from scripts import jobs

    trainer = TRAINERS["train_dreamer.py"]
    assert trainer.policy_keys == ("world_model", "actor") and trainer.replay_patterns == DEFAULT_REPLAY_PATTERNS
    replay = make_replay(tmp_path, pixel=False)
    names = [p.name for p in (tmp_path / "replay").iterdir()]
    assert names and all(any(fnmatch.fnmatchcase(n, pat) for pat in trainer.replay_patterns) for n in names)
    assert replay.nbytes() > 0
    state = {"script": "scripts/train_dreamer.py", "required_results": ["final.json", "train.jsonl"]}
    assert {"final.json", "exit.json", "provenance.json"} <= set(jobs.remote_check_names(state))
    validate_output_arguments("scripts/train_dreamer.py", ["--run-dir", "runs/ds-x"], "ds-x")
    with pytest.raises(ValueError):
        validate_output_arguments("scripts/train_dreamer.py", ["--run-dir", "runs/other"], "ds-x")
