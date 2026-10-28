"""Option-off reference for the self-render replacement variant (2026-10-29): fingerprints of the csw4 training path
(base + chamfer_lean + chamfer_lean_randmask + enc_cs + enc_w4) computed by the code BEFORE ``aug.synth_source`` and
``aug.synth_warmup_steps`` existed (branch selfrender-20261029 at 795bec3, unchanged s4d/ sources), stored in
tests/data/selfrender_off_golden.json and compared bit for bit by tests/test_selfrender_replace.py.

Covered: three csw4 training steps at small sizes that keep the patch-local CNN stem (32 x 32 images, 16 px patches,
4 patches per frame, 75 % random temporal-token masking: 3 of 12 tokens kept), the point-cloud synthetic views (two
jittered cameras, coverage-masked tokens, InfoNCE/consistency positives and synthetic render targets), Chamfer, M3D;
the suite's synthetic rasterizer, CPU, one thread: forward, backward, clipping, AdamW. Every loss, metric, the gradient
norm, the parameters after each step and the random-number state are fingerprinted.

    python -I tests/_selfrender_golden.py --write   # only at the pre-change commit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests/data/selfrender_off_golden.json"
CSW4 = ["chamfer_lean.yaml", "chamfer_lean_randmask.yaml", "enc_cs.yaml", "enc_w4.yaml"]


def small_csw4_cfg(*extra_variants: str) -> dict:
    """The csw4 configuration at test sizes (patch 16 is required by the patch-local stem)."""
    from tests import _enc_golden as golden

    cfg = golden.small_cfg(*(CSW4[2:] + list(extra_variants)))  # small_cfg already loads chamfer_lean + randmask
    cfg["model"]["encoder"].update(image_height=32, image_width=32, patch_size=16, width=32, depth=2, heads=4,
                                   slot_dim=8)
    return cfg


def fixture_batch(tmp: Path) -> dict:
    """Two fixture windows (tests/_enc_golden.py) upsampled 2x to 32 x 32 (pixel replication, intrinsics scaled)."""
    from s4d.data.contract import collate
    from s4d.data.metaworld.dataset import MetaworldWindowDataset
    from tests import _enc_golden as golden

    path = golden.write_fixture(tmp / "fixture.hdf5")
    ds = MetaworldWindowDataset(path, ["ep000", "ep001"], strides=(2, 4, 6), seed=0)
    batch = collate([ds[0], ds[len(ds) // 2]])
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() >= 2 and tuple(value.shape[-2:]) == (16, 16):
            value = value.repeat_interleave(2, -2).repeat_interleave(2, -1).contiguous()
        out[key] = value
    K = out["K"].clone()
    K[..., :2, :] *= 2.0
    out["K"] = K
    return out


def fingerprints(tmp: Path, overrides: dict | None = None) -> dict:
    """``overrides``: dotted config keys applied to the small csw4 config (the option-off settings of new keys)."""
    from s4d.train.loop import build_model, build_optimizer, forward_losses
    from tests import _enc_golden as golden

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    saved = golden._patched()
    try:
        out: dict = {"environment": golden.environment()}
        batch = fixture_batch(tmp)
        out["batch"] = {k: golden._sha(v) for k, v in sorted(batch.items()) if torch.is_tensor(v)}
        cfg = small_csw4_cfg()
        for key, value in (overrides or {}).items():
            golden.set_dotted(cfg, key, value)
        torch.manual_seed(0)
        model = build_model(cfg).train()
        optimizer, scheduler = build_optimizer(model, cfg)
        out["optimizer_groups"] = [{"n": len(g["params"]), "lr": g["lr"], "wd": g["weight_decay"]}
                                   for g in optimizer.param_groups]
        out["init"] = golden._sha(torch.cat([p.detach().flatten() for p in model.parameters()]))
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
                          "metrics": {k: float(v).hex() for k, v in sorted(res["metrics"].items())},
                          "params": golden._sha(torch.cat([p.detach().flatten() for p in model.parameters()]))})
        out["steps"] = steps
        out["rng_after_steps"] = golden._sha(torch.get_rng_state())
        return out
    finally:
        golden._restore(saved)
        torch.set_num_threads(threads)


if __name__ == "__main__":
    import tempfile

    sys.path.insert(0, str(REPO))
    if "--write" not in sys.argv:
        raise SystemExit("usage: python -I tests/_selfrender_golden.py --write (pre-change commit only)")
    with tempfile.TemporaryDirectory(dir=REPO / ".pytest_tmp" if (REPO / ".pytest_tmp").is_dir() else None) as tmp:
        result = fingerprints(Path(tmp))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", GOLDEN)
