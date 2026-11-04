"""AllTracker worker (runs in the separate ``alltracker`` conda env of the pinned upstream checkout).

UNVERIFIED: written against aharley/alltracker ``demo.py`` at the pinned commit; never executed here.

task ``track``: input ``rgbs`` (T,H,W,3) uint8 frames of ONE camera (H, W multiples of 8; the caller resizes and
tracks the intrinsics), ``query_frame`` in params (tracks are produced for frames >= query_frame).
outputs ``flows`` (T',2,H,W) float32 (pixel offsets from the query grid, integer-centre convention as upstream),
``visconf`` (T',2,H,W) float32 sigmoids (channel 0 visibility, channel 1 confidence).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402

_MODELS: dict[tuple, object] = {}


def handle(request, inputs):
    import numpy as np
    import torch

    repo = Path(request["repo_dir"]).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from nets.alltracker import Net  # upstream

    params = request["params"]
    ckpt = Path(request["checkpoint"])  # alltracker.pth downloaded once into the alltracker env, never here
    key = (str(repo), str(ckpt.resolve()), int(params.get("window_len", 16)))
    if key not in _MODELS:
        model = Net(key[2])
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"], strict=True)
        _MODELS[key] = model.cuda().eval()
    model = _MODELS[key]
    rgbs = torch.from_numpy(inputs["rgbs"]).permute(0, 3, 1, 2).float()[None].cuda()  # 1,T,3,H,W in 0..255
    q = int(params.get("query_frame", 0))
    if rgbs.shape[-1] % 8 or rgbs.shape[-2] % 8:
        raise ValueError("AllTracker needs H, W divisible by 8 (resize upstream and track intrinsics)")
    with torch.no_grad():
        flows, visconf, _, _ = model.forward_sliding(rgbs[:, q:], iters=int(params.get("inference_iters", 4)), sw=None,
                                                     is_training=False)
    identity = {"name": "AllTracker", "repo": "aharley/alltracker", "commit": git_commit(str(repo)),
                "checkpoint": str(ckpt), "checkpoint_sha256": sha256(str(ckpt)), "is_mock": False}
    outputs = {"flows": flows[0].float().cpu().numpy().astype(np.float32),
               "visconf": visconf[0].float().cpu().numpy().astype(np.float32)}
    meta = {"query_frame": q, "pixel_convention": "integer_centers", "torch": torch.__version__,
            "inference_iters": int(params.get("inference_iters", 4)), "window_len": int(params.get("window_len", 16))}
    return identity, outputs, meta


if __name__ == "__main__":
    main(handle)
