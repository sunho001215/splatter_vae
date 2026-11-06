"""GPU entry-point tests of the real-world offline FQL tooling (marker ``gpu``): ``scripts/realworld_fql_dataset.py``
and ``scripts/train_realworld_fql.py`` as subprocesses on a compact training store written from the action fixture's
real camera timeline (mug-hanging episode_000003), for the end-to-end CNN condition (``e2e_cnn``) and a frozen-encoder
condition (``dynamo``): a short run, a lossless resume and the exported ``policy.pt`` through ``OfflinePolicy``. These
need a CUDA device and run in the coordinator's release chain on the GPU; without CUDA they are skipped.

Lossless resume is proven with ``train.deterministic=true`` (bitwise-reproducible kernels): 2 + 2 steps must equal 4
uninterrupted steps bit for bit in every parameter, target network, Adam moment and generator (sampling and
augmentation), so any state the checkpoint failed to restore would show. Under the default fast kernels two
UNINTERRUPTED GPU runs of the CNN condition already differ (cuDNN convolution / max-pool backward are
nondeterministic; measured: 126 of 132 tensors differ after 4 steps, up to 2 % relative), so the default-kernel run
checks only what the hardware can give: the restored step and generator states, finite values."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from s4d.rwlearn import actions as A

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test: runs in the release chain")]

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "rwlearn" / "raw_mug_hanging_ep3_actions.npz"
N_FRAMES, HW = 24, 128


def write_compact_store(root: Path, episodes=("episode_000003", "episode_000903")) -> Path:
    """Minimal ``s4d.realworld.train`` episode files (RGB + timestamps + segments + cameras) on the fixture timeline."""
    from s4d.realworld import train_store as TS

    raw = A.raw_from_arrays(np.load(FIXTURE))
    rng = np.random.default_rng(0)
    paths = []
    for eid in episodes:
        d = root / eid
        d.mkdir(parents=True)
        with h5py.File(d / TS.FILE, "w") as f:
            f.attrs.update(format=TS.FORMAT, schema_version=TS.SCHEMA_VERSION, episode_id=eid, task="mug-hanging",
                           cameras=json.dumps(raw.serials), reference_camera=raw.serials[0], image_size=HW,
                           max_sync_ns=22_000_000, raw_recording=str(FIXTURE), MOCK_MODEL_OUTPUTS=True,
                           upstream=json.dumps({"calibrate": "cal-test"}))
            for serial in raw.serials:
                g = f.create_group(f"cameras/{serial}")
                g.create_dataset("timestamps_ns", data=raw.cameras[serial]["t"][:N_FRAMES])
                g.create_dataset("segment", data=raw.cameras[serial]["segment"][:N_FRAMES].astype(np.int16))
                g.create_dataset("rgb", data=rng.integers(0, 256, (N_FRAMES, HW, HW, 3), dtype=np.uint8),
                                 chunks=(1, HW, HW, 3))
                g.create_dataset("depth", data=rng.uniform(0.6, 1.4, (N_FRAMES, HW, HW)).astype(np.float32),
                                 chunks=(1, HW, HW))
                g.create_dataset("K", data=np.array([[160.0, 0, 64.4], [0, 160.0, 65.1], [0, 0, 1]]))
                g.create_dataset("w2c", data=np.eye(4))
        paths.append(str(d / TS.FILE))
    locator = root / "store.yaml"
    locator.write_text(yaml.safe_dump({"kind": "compact", "paths": paths}))
    return locator


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]}
    res = subprocess.run([sys.executable, "-I", str(REPO / script), *args], env=env, capture_output=True, text=True,
                         timeout=1200)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-4000:]
    return res


def dynamo_export(path: Path) -> Path:
    from s4d.baselines.dynamo.model import build_encoder, encoder_kwargs
    from s4d.baselines.dynamo.model import save_export as save_dynamo

    cfg = yaml.safe_load((REPO / "configs/realworld/baselines/dynamo/mug-hanging.yaml").read_text())
    torch.manual_seed(0)
    save_dynamo(path, build_encoder(encoder_kwargs(cfg)).eval(), encoder_kwargs(cfg), 1, HW, 0, True)
    return path


@pytest.mark.parametrize("condition", ["e2e_cnn", "dynamo"])
def test_dataset_build_training_lossless_resume_and_policy(tmp_path, condition):
    locator = write_compact_store(tmp_path / "store")
    from s4d.rwlearn import outcomes as O
    from s4d.rwlearn.store import open_store

    store = open_store(locator)
    eps = store.episodes()
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(json.dumps(O.outcome_labels(eps)))
    data = tmp_path / f"{condition}.h5"
    build = ["build", "--smoke", "--store", str(locator), "--condition", condition, "--out", str(data),
             "--outcomes", str(outcomes), "--device", "cpu" if condition == "e2e_cnn" else "cuda"]
    if condition != "e2e_cnn":
        build += ["--export", str(dynamo_export(tmp_path / "dynamo.pt"))]
    run("scripts/realworld_fql_dataset.py", *build)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [eps[0]], "validation": [eps[1]]}))
    root = tmp_path / "runs"
    sets = [f"condition={condition}", f"data.dataset={data}", f"data.split_manifest={split}",
            "agent.batch_size=16", "agent.actor_hidden_dims=[32,32]", "agent.value_hidden_dims=[32,32]",
            "train.log_interval=1", "train.eval_interval=2", "train.save_interval=2", "train.val_batches=1",
            "train.device=cuda"]
    if condition == "e2e_cnn":
        sets.append(f"data.store={locator}")
    common = ["--config", "configs/realworld/fql/base.yaml", "--output-root", str(root), "--no-wandb"]
    det = [*sets, "train.deterministic=true"]
    run("scripts/train_realworld_fql.py", *common, "--name", "straight", "--set", *det, "train.offline_steps=4")
    run("scripts/train_realworld_fql.py", *common, "--name", "resumed", "--set", *det, "train.offline_steps=2")
    run("scripts/train_realworld_fql.py", *common, "--name", "resumed", "--set", *det, "train.offline_steps=4")
    a, b = (torch.load(root / r / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
            for r in ("straight", "resumed"))
    assert a["step"] == b["step"] == 4
    for name, mod in a["agent"]["modules"].items():  # parameters incl. encoders and target networks
        for k, v in mod.items():
            assert torch.equal(v, b["agent"]["modules"][name][k]), (name, k)
    for p_id, st in a["agent"]["optimizer"]["state"].items():  # Adam moments and step counts
        for k, v in st.items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(b["agent"]["optimizer"]["state"][p_id][k])), (p_id, k)
    for key in ("sample_gen", "aug_gen"):  # data cursor (sampling RNG) and augmentation RNG
        assert torch.equal(a[key], b[key]), key
    assert torch.equal(a["agent"]["generator"], b["agent"]["generator"])  # the agent's noise generator
    rows = lambda r: [json.loads(x) for x in (root / r / "metrics.jsonl").read_text().splitlines()]  # noqa: E731
    loss = lambda rs: {x["step"]: x["training/critic/critic_loss"] for x in rs if "training/critic/critic_loss" in x}  # noqa: E731
    assert loss(rows("straight")) == loss(rows("resumed")) and set(loss(rows("straight"))) == {1, 2, 3, 4}
    # default (fast, nondeterministic) kernels: the resume restores step and generators; values stay finite
    run("scripts/train_realworld_fql.py", *common, "--name", "fast", "--set", *sets, "train.offline_steps=2")
    run("scripts/train_realworld_fql.py", *common, "--name", "fast", "--set", *sets, "train.offline_steps=4")
    fast = torch.load(root / "fast" / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert fast["step"] == 4 and all(torch.equal(fast[k], a[k]) for k in ("sample_gen", "aug_gen"))
    assert all(torch.isfinite(v).all() for m in fast["agent"]["modules"].values() for v in m.values()
               if v.is_floating_point())
    val = [x for x in rows("straight") if "validation/onestep_pos_err_mm" in x]
    assert val and all(np.isfinite(x["validation/critic/critic_loss"]) for x in val)
    assert a["agent"]["config"]["alpha"] == 10.0
    if condition == "e2e_cnn":  # the jointly trained CNN is part of the agent (no export); policy meta has its config
        assert any(k.startswith("encoder.stacks.") for k in a["agent"]["modules"]["critic"])
        assert a["meta"]["encoder_config"]["stack_sizes"] == [16, 32, 32] and a["meta"]["export_sha256"] == ""
    from s4d.rwlearn.inference import OfflinePolicy

    policy = OfflinePolicy(root / "straight" / "policy.pt", device="cuda")
    serial = store.cameras(eps[0])[0]
    rgb = store.rgb(eps[0], serial, [4, 5, 6])[:, None]
    with h5py.File(data, "r") as f:
        g = f["episodes"][eps[0]][serial]  # both conditions are single-view: one timeline per camera
        pose = A.matrix_to_pose7(g["measured_pose"][6])
        grip = A.GRIPPER_OPEN_REGISTER + A.GRIPPER_SPAN * float(g["proprio"][6, -1])
    out = policy.act({"rgb": rgb}, pose, grip, noise=torch.zeros(1, A.ACTION_DIM))
    assert out["flange_pose"].shape == (7,) and np.isfinite(out["flange_pose"]).all()
