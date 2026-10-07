"""One native gsplat rasterization on the visible GPU: forward values and backward gradients.

Run once per authorized GPU: ``CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/check_native_render.py``.
Writes ``docs/native_render/<uuid>.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402

from s4d.model.gaussians import DYNAMIC_GROUP, GaussianSet  # noqa: E402
from s4d.model.render import render_rgbd, require_prebuilt_renderer  # noqa: E402


def main() -> None:
    binary = require_prebuilt_renderer()
    torch.manual_seed(0)
    n = 4096
    xyz = torch.randn(1, n, 3, device="cuda") * 0.2 + torch.tensor([0.0, 0.0, 2.0], device="cuda")
    gs = GaussianSet(
        xyz.requires_grad_(),
        # Anisotropic scales and random rotations, so that every attribute (including rotation) has a gradient.
        (torch.rand(1, n, 3, device="cuda") * 0.03 + 0.005).requires_grad_(),
        torch.nn.functional.normalize(torch.randn(1, n, 4, device="cuda"), dim=-1).requires_grad_(),
        torch.full((1, n), 0.8, device="cuda", requires_grad=True),
        torch.rand(1, n, 3, device="cuda").requires_grad_(),
        torch.zeros(1, n, 3, device="cuda"),
        torch.zeros(1, n, 3, device="cuda"),
        torch.full((n,), DYNAMIC_GROUP, device="cuda", dtype=torch.long),
    )
    w2c = torch.eye(4, device="cuda").reshape(1, 1, 4, 4)
    K = torch.tensor([[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]], device="cuda").reshape(1, 1, 3, 3)
    torch.cuda.synchronize()
    start = time.time()
    out = render_rgbd(gs, gs.xyz[:, None], w2c, K, 128, 128, 0.05, 5.0)
    (out["rgb"].mean() + out["depth"].mean()).backward()
    torch.cuda.synchronize()
    out = {k: v.detach() for k, v in out.items()}
    valid = out["alpha"] > 0.5
    report = {
        "gpu_mapping": GPU_MAPPING,
        "binary": binary,
        "torch": torch.__version__,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "seconds_first_call": time.time() - start,
        "alpha_max": float(out["alpha"].max()),
        "covered_fraction": float(valid.float().mean()),
        "median_depth_where_opaque": float(out["depth"][valid].median()),
        "all_finite": all(bool(torch.isfinite(v).all()) for v in out.values()),
        "grad_abs_mean": {
            name: float(getattr(gs, name).grad.abs().mean()) for name in ("xyz", "scales", "quats", "opacity", "rgb")
        },
    }
    report["passed"] = (
        report["all_finite"]
        and report["alpha_max"] > 0.9
        and 1.5 < report["median_depth_where_opaque"] < 2.5
        and all(v > 0 for v in report["grad_abs_mean"].values())
    )
    out_dir = REPO / "docs/native_render"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{GPU_MAPPING[0]['uuid']}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        raise SystemExit("native rasterization check failed")


if __name__ == "__main__":
    main()
