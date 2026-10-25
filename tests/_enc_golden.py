"""Flags-off reference for the encoder pilots (2026-10-25): fingerprints of the chlrm training path computed by the code
BEFORE the pilot features existed (main d9fb142 sources = pool release f3538b1 for s4d/ and the training script), stored
in tests/data/enc_flags_off_golden.json and compared bit for bit by tests/test_encoder_pilots.py.

Covered: Meta-World window samples (a deterministic fixture file), three chlrm training steps (small sizes, the test
suite's synthetic rasterizer, CPU, one thread: forward, backward, clipping, AdamW) and the full-size chlrm encoder
(width 128, depth 8, 8 heads, 16 px patches, 75 % random temporal-token masking) in training and policy mode.

    python -I tests/_enc_golden.py --write   # only at the pre-change commit
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests/data/enc_flags_off_golden.json"
VARIANTS = REPO / "configs/metaworld/variants"


def _sha(t) -> str:
    if torch.is_tensor(t):
        t = t.detach().cpu().contiguous()
        return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()


def environment() -> dict:
    cpu = ""
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"torch": torch.__version__, "machine": platform.machine(), "cpu": cpu}


def write_fixture(path: Path) -> Path:
    """Two episodes, 40 frames, two training cameras and one evaluation camera at 16 x 16, a moving body and a moving
    hand (deterministic)."""
    rng = np.random.default_rng(1234)
    T, V, H, W = 40, 3, 16, 16
    K = np.tile(np.array([[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]], dtype=np.float32), (V, 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (V, 1, 1))
    c2w[:, 0, 3] = [0.0, 0.05, -0.05]
    with h5py.File(path, "w") as f:
        f.attrs.update({"task": "fixture", "depth_unit_m": 1e-4, "dt_seconds": 0.0125, "background_body": 65535,
                        "body_names": json.dumps(["world", "hand", "object"]), "complete": True})
        cams = f.create_group("cameras")
        cams.create_dataset("K", data=K)
        cams.create_dataset("c2w", data=c2w)
        cams.create_dataset("w2c", data=np.linalg.inv(c2w))
        cams.create_dataset("is_train", data=np.array([True, True, False]))
        cams.create_dataset("names", data=np.asarray(["train0", "train1", "eval0"], dtype=h5py.string_dtype()))
        eps = f.create_group("episodes")
        for e in range(2):
            ep = eps.create_group(f"ep{e:03d}")
            ep.attrs["length"] = T
            body = np.zeros((T, V - 1, H, W), dtype=np.uint16)  # training cameras only
            body[..., H // 2:, :] = 1
            body[..., :4, :4] = 2
            xpos = np.zeros((T, 3, 3), dtype=np.float32)
            xpos[:, 1, 0] = np.arange(T) * 0.004 * (1 + e)
            xpos[:, 2, 1] = np.sin(np.arange(T) / 7.0) * 0.01
            quat = np.zeros((T, 3, 4), dtype=np.float32)
            quat[..., 0] = 1
            obs = rng.normal(size=(T, 39)).astype(np.float32) * 0.01
            obs[:, 0] = np.arange(T) * 0.004 * (1 + e)  # hand x
            obs[:, 3] = (np.arange(T) % 20 < 10).astype(np.float32)  # gripper opening in [0, 1]
            for key, value in {
                "rgb": rng.integers(0, 256, size=(T, V, H, W, 3), dtype=np.uint8),
                "depth": (8000 + rng.integers(0, 4000, size=(T, V, H, W))).astype(np.uint16),
                "body_id": body, "xpos": xpos, "xquat": quat, "obs": obs,
                "qpos": np.zeros((T, 3), dtype=np.float32), "qvel": np.zeros((T, 3), dtype=np.float32),
                "action": rng.normal(size=(T, 4)).astype(np.float32), "reward": np.zeros(T, dtype=np.float32),
                "success": np.zeros(T, dtype=np.uint8),
            }.items():
                ep.create_dataset(key, data=value)
    return path


def small_cfg(*variants: str) -> dict:
    from s4d.config import load_config

    files = [REPO / "configs/metaworld/base.yaml", VARIANTS / "chamfer_lean.yaml",
             VARIANTS / "chamfer_lean_randmask.yaml"] + [VARIANTS / v for v in variants]
    cfg = load_config(files)
    cfg["train"]["bf16"] = False
    cfg["model"]["encoder"].update(image_height=16, image_width=16, patch_size=4, width=16, depth=2, heads=4,
                                   slot_dim=8)
    cfg["model"]["decoder"].update(dim=16, depth=1, heads=4)
    for group in ("scene", "dynamic"):
        cfg["model"]["decoder"][group].update(parents=4, children=2)
    cfg["loss"]["chamfer_points"] = 64
    return cfg


def set_dotted(cfg: dict, key: str, value) -> None:
    *parents, leaf = key.split(".")
    node = cfg
    for name in parents:
        node = node.setdefault(name, {})
    node[leaf] = value


def _patched():
    """The suite's synthetic rasterizer and SSIM stand-in (tests/test_review_options.py)."""
    from s4d.losses import rgb
    from s4d.model import render
    from tests.test_review_options import _fake_rasterizer

    saved = (render._rasterize, rgb._ssim_map)
    render._rasterize = _fake_rasterizer([])
    rgb._ssim_map = lambda x, y: 1 - (x - y).square()
    return saved


def _restore(saved) -> None:
    from s4d.losses import rgb
    from s4d.model import render

    render._rasterize, rgb._ssim_map = saved


def fingerprints(tmp: Path, overrides: dict | None = None, dataset_kwargs: dict | None = None) -> dict:
    """``overrides``: dotted config keys applied to the small chlrm config (the flags-off settings of new keys);
    ``dataset_kwargs``: extra keyword arguments of the training dataset (the flags-off value of new options)."""
    from s4d.data.contract import collate
    from s4d.data.metaworld.dataset import MetaworldWindowDataset
    from s4d.model.encoder import Encoder, EncoderConfig
    from s4d.train.loop import build_model, build_optimizer, forward_losses

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    saved = _patched()
    try:
        out: dict = {"environment": environment()}
        path = write_fixture(tmp / "fixture.hdf5")
        ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6), seed=0, **(dataset_kwargs or {}))
        out["dataset_len"] = len(ds)
        picks = [0, 7, len(ds) // 2, len(ds) - 1]
        samples = [ds[i] for i in picks]
        out["samples"] = [{k: (_sha(v) if torch.is_tensor(v) else repr(v)) for k, v in sorted(s.items())}
                          for s in samples]

        cfg = small_cfg()
        for key, value in (overrides or {}).items():
            set_dotted(cfg, key, value)
        torch.manual_seed(0)
        model = build_model(cfg).train()
        optimizer, scheduler = build_optimizer(model, cfg)
        out["optimizer_groups"] = [{"n": len(g["params"]), "lr": g["lr"], "wd": g["weight_decay"]}
                                   for g in optimizer.param_groups]
        out["init"] = _sha(torch.cat([p.detach().flatten() for p in model.parameters()]))
        batch = collate([ds[i] for i in picks[:2]])
        steps = []
        for step in range(3):
            res = forward_losses(model, batch, cfg, step, source=torch.tensor([0, 1]))
            optimizer.zero_grad(set_to_none=True)
            res["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            steps.append({"total": float(res["total"].detach()).hex(), "grad_norm": float(norm).hex(),
                          "losses": {k: float(v).hex() for k, v in sorted(res["losses"].items())},
                          "params": _sha(torch.cat([p.detach().flatten() for p in model.parameters()]))})
        out["steps"] = steps
        out["rng_after_steps"] = _sha(torch.get_rng_state())

        torch.manual_seed(1)
        enc = Encoder(EncoderConfig(mask_mode="random", mask_ratio=0.75)).train()
        out["full_encoder_params"] = sum(p.numel() for p in enc.parameters())
        out["full_encoder_state"] = {k: [list(v.shape), _sha(v)] for k, v in sorted(enc.state_dict().items())}
        x = torch.rand(2, 3, 3, 128, 128, generator=torch.Generator().manual_seed(2))
        res = enc(x)
        res["slots"].float().square().sum().backward()
        out["full_encoder_train"] = {"slots": _sha(res["slots"]), "visible": _sha(res["visible"]),
                                     "grads": _sha(torch.cat([p.grad.flatten() for p in enc.parameters()
                                                              if p.grad is not None]))}
        out["full_encoder_policy"] = _sha(enc.policy_state(x))
        return out
    finally:
        _restore(saved)
        torch.set_num_threads(threads)


if __name__ == "__main__":
    import tempfile

    sys.path.insert(0, str(REPO))
    if "--write" not in sys.argv:
        raise SystemExit("usage: python -I tests/_enc_golden.py --write (pre-change commit only)")
    with tempfile.TemporaryDirectory(dir=REPO / ".pytest_tmp" if (REPO / ".pytest_tmp").is_dir() else None) as tmp:
        result = fingerprints(Path(tmp))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", GOLDEN)
