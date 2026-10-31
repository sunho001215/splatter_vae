"""FoundationStereo worker (runs in the separate ``foundation_stereo`` conda env of the pinned upstream checkout).

UNVERIFIED: written against NVlabs/FoundationStereo ``scripts/run_demo.py`` at the pinned commit (see
configs/realworld/upstream_pins.yaml) but never executed here (no checkpoint / env on this host).

task ``stereo_disparity``: inputs ``left``, ``right`` (N,H,W,3) uint8 *rectified* pairs (infrared pairs are
replicated to 3 channels by the caller *as model input only*, recorded in the stage record); output ``disparity``
(N,H,W) float32, left-reference (``x_left - x_right``) on the given grid. Right-reference disparity is obtained by
the caller with flip-and-swap (s4d.realworld.stereo), never by swapping inputs. No resizing happens here: the caller
resizes both eyes identically and tracks the intrinsics (``scale`` of run_demo.py is therefore not used).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402


def handle(request, inputs):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    repo = Path(request["repo_dir"]).resolve()
    sys.path.insert(0, str(repo))
    from core.foundation_stereo import FoundationStereo  # upstream
    from core.utils.utils import InputPadder  # upstream

    params = request["params"]
    ckpt = Path(request["checkpoint"])
    cfg = OmegaConf.load(ckpt.parent / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = int(params.get("valid_iters", 32))
    cfg["hiera"] = int(params.get("hiera", 0))
    args = OmegaConf.create(cfg)
    model = FoundationStereo(args)
    # torch >= 2.6 defaults to weights_only=True; the official checkpoint also pickles numpy scalars (global_step,
    # epoch), which upstream's torch 2.4 loaded with the old default. Its sha256 is recorded in the identity below.
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    model.cuda().eval()
    out = []
    for left, right in zip(inputs["left"], inputs["right"]):
        H, W = left.shape[:2]
        img0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            if args.hiera:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
            else:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        out.append(padder.unpad(disp.float()).cpu().numpy().reshape(H, W))
    identity = {"name": "FoundationStereo", "repo": "NVlabs/FoundationStereo", "commit": git_commit(str(repo)),
                "checkpoint": str(ckpt), "checkpoint_sha256": sha256(str(ckpt)), "is_mock": False}
    meta = {"reference": "left", "valid_iters": int(args.valid_iters), "hiera": int(args.hiera),
            "torch": torch.__version__}
    return identity, {"disparity": np.stack(out).astype(np.float32)}, meta


if __name__ == "__main__":
    main(handle)
