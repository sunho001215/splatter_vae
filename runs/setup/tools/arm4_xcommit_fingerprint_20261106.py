"""Four-arm campaign (arm4, user directive 2026-11-06): cross-commit training-path fingerprints (CPU, read-only).

Answers "did any default drift between the reference runs' code and the release the new arms run on?" by running the
SAME resolved configuration (passed in as JSON) through the training path of two source trees and comparing every
fingerprint bit for bit: the deterministic fixture's window samples, the initial parameters, and three training steps
(forward_losses, backward, gradient clipping, AdamW + schedule) with every loss term, the gradient norm and the
parameters after each step. Small sizes, one CPU thread, the test suite's synthetic differentiable rasterizer and SSIM
stand-in (the real rasterizer needs a GPU; the stand-in is identical code for both trees because it lives here).

A source tree is a directory holding s4d/ (e.g. `git archive <commit> s4d | tar -x -C <dir>`). Run once per tree:

    nice -n 15 .venv/bin/python -I arm4_xcommit_fingerprint_20261106.py <src_root> <cfg.json> <out.json>

Nothing here imports campaign drivers or the scheduler; no file outside <out.json> (and a fixture in a temporary
directory) is written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path


def _sha(t) -> str:
    import numpy as np
    import torch
    if torch.is_tensor(t):
        t = t.detach().cpu().contiguous()
        return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(t).tobytes()).hexdigest()


def write_fixture(path: Path) -> Path:
    """tests/_enc_golden.py write_fixture (HEAD b5ab9a6), copied so both trees read the same file: two episodes, 40
    frames, two training cameras and one evaluation camera at 16 x 16 (32 x 32 with --size 32), a moving body and hand."""
    import h5py
    import numpy as np
    size = int(SIZE)
    rng = np.random.default_rng(1234)
    T, V, H, W = 40, 3, size, size
    K = np.tile(np.array([[float(size), 0.0, size / 2], [0.0, float(size), size / 2], [0.0, 0.0, 1.0]], dtype=np.float32),
                (V, 1, 1))
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
            body = np.zeros((T, V - 1, H, W), dtype=np.uint16)
            body[..., H // 2:, :] = 1
            body[..., :4, :4] = 2
            xpos = np.zeros((T, 3, 3), dtype=np.float32)
            xpos[:, 1, 0] = np.arange(T) * 0.004 * (1 + e)
            xpos[:, 2, 1] = np.sin(np.arange(T) / 7.0) * 0.01
            quat = np.zeros((T, 3, 4), dtype=np.float32)
            quat[..., 0] = 1
            obs = rng.normal(size=(T, 39)).astype(np.float32) * 0.01
            obs[:, 0] = np.arange(T) * 0.004 * (1 + e)
            obs[:, 3] = (np.arange(T) % 20 < 10).astype(np.float32)
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


def fake_rasterizer(calls):
    """tests/test_review_options.py _fake_rasterizer (HEAD b5ab9a6), copied (identical for both trees)."""
    import torch

    def fake(**kw):
        means, opacity, colors = kw["means"], kw["opacities"], kw["colors"]
        lead = means.shape[:-2]
        V, H, W = kw["viewmats"].shape[-3], kw["height"], kw["width"]
        alpha = opacity.mean(-1)[..., None, None].expand(*lead, V, 1)
        depth = means[..., 2].mean(-1).abs().add(0.5)[..., None, None].expand(*lead, V, 1)
        if colors is None:
            values = depth
        else:
            per_camera = colors.dim() == means.dim() + 1
            values = (colors.mean(-2) if per_camera else colors.mean(-2)[..., None, :]) * alpha
            if kw["render_mode"] == "RGB+ED":
                values = torch.cat((values, depth), -1)
        calls.append((kw["render_mode"], V, torch.is_grad_enabled()))
        output = values[..., None, None, :].expand(*lead, V, H, W, values.shape[-1])
        return output, alpha[..., None, None, :].expand(*lead, V, H, W, 1), {}

    return fake


def small(cfg: dict) -> dict:
    """The resolved configuration at test size (the golden harness's sizes; patch 16 on 32 x 32 for the patch_cnn
    stem, which needs patch_size 16). Only sizes change; every loss weight, flag and schedule key is kept."""
    cfg = copy.deepcopy(cfg)
    cfg["train"]["bf16"] = False
    enc = cfg["model"]["encoder"]
    patch = 16 if enc.get("stem") == "patch_cnn" else 4
    enc.update(image_height=int(SIZE), image_width=int(SIZE), patch_size=patch, width=16, depth=2, heads=4, slot_dim=8)
    cfg["model"]["decoder"].update(dim=16, depth=1, heads=4)
    for group in ("scene", "dynamic"):
        cfg["model"]["decoder"][group].update(parents=4, children=2)
    if "chamfer_points" in cfg["loss"]:
        cfg["loss"]["chamfer_points"] = 64
    cfg["model"]["anchor_stats"] = None
    return cfg


def fingerprints(cfg: dict, tmp: Path, steps: int) -> dict:
    import torch
    from s4d.data.contract import collate
    from s4d.data.metaworld.dataset import MetaworldWindowDataset
    from s4d.losses import rgb
    from s4d.model import render
    from s4d.train.loop import build_model, build_optimizer, forward_losses

    torch.set_num_threads(1)
    calls = []
    render._rasterize = fake_rasterizer(calls)
    rgb._ssim_map = lambda x, y: 1 - (x - y).square()
    out: dict = {"torch": torch.__version__}
    path = write_fixture(tmp / "fixture.hdf5")
    ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6), seed=0)
    out["dataset_len"] = len(ds)
    picks = [0, 7, len(ds) // 2, len(ds) - 1]
    out["samples"] = [{k: (_sha(v) if torch.is_tensor(v) else repr(v)) for k, v in sorted(ds[i].items())} for i in picks]
    cfg = small(cfg)
    torch.manual_seed(0)
    model = build_model(cfg).train()
    optimizer, scheduler = build_optimizer(model, cfg)
    out["optimizer_groups"] = [{"n": len(g["params"]), "lr": g["lr"], "wd": g["weight_decay"]}
                               for g in optimizer.param_groups]
    out["init"] = _sha(torch.cat([p.detach().flatten() for p in model.parameters()]))
    batch = collate([ds[i] for i in picks[:2]])
    rows = []
    for step in range(steps):
        res = forward_losses(model, batch, cfg, step, source=torch.tensor([0, 1]))
        optimizer.zero_grad(set_to_none=True)
        res["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        rows.append({"total": float(res["total"].detach()).hex(), "grad_norm": float(norm).hex(),
                     "losses": {k: float(v).hex() for k, v in sorted(res["losses"].items())},
                     "params": _sha(torch.cat([p.detach().flatten() for p in model.parameters()]))})
    out["steps"] = rows
    out["render_calls"] = [list(c) for c in calls]
    out["rng_after_steps"] = _sha(torch.get_rng_state())
    return out


SIZE = 16
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--size" in sys.argv:
        SIZE = int(sys.argv[sys.argv.index("--size") + 1])
        args = [a for a in args if a != str(SIZE)]
    src, cfg_path, out_path = Path(args[0]).resolve(), Path(args[1]), Path(args[2])
    sys.path.insert(0, str(src))
    import s4d  # noqa: E402

    assert Path(s4d.__file__).resolve().parent == src / "s4d", s4d.__file__
    with tempfile.TemporaryDirectory() as tmp:
        result = fingerprints(json.loads(cfg_path.read_text()), Path(tmp), 3)
    result["source_root"] = str(src)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out_path)
