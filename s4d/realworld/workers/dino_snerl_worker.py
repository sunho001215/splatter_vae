"""DINO teacher features exactly as the repository's SNeRL baseline consumes them.

Runs in the repository venv (the consumer's own environment: third_party/dino at the pinned submodule commit and
the sha256-checked official ``dino_deitsmall8_pretrain.pth``) and calls the consumer's functions directly:
``s4d.baselines.snerl.upstream.dino_extractor`` (ViTExtractor('dino_vits8', stride 4)) and
``s4d.baselines.snerl.model.dino_target`` (scalar normalisation (x-0.45)/0.225, layer 11, facet "key", no bin,
no CLS, bilinear upsampling of the 31x31x384 token grid to the image size). SNeRL itself computes these online
from the training images; this cache is optional (inspection / offline reuse) and is keyed by the image hash.

task ``dino_snerl``: input ``images`` (N,H,W,3) float32 in [0,1] at the consumer resolution (128x128);
output ``features`` (N,H,W,384) float32.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main  # noqa: E402


def handle(request, inputs):
    import numpy as np
    import torch

    from s4d.baselines.snerl.model import dino_target
    from s4d.baselines.snerl.upstream import DINO_WEIGHTS_SHA256, dino_extractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = dino_extractor(device)
    feats = []
    for image in inputs["images"]:
        feats.append(dino_target(extractor, torch.from_numpy(np.asarray(image, np.float32)).to(device)).cpu().numpy())
    identity = {"name": "DINO ViT-S/8 (SNeRL teacher)", "repo": "facebookresearch/dino",
                "commit": git_commit(str(REPO / "third_party" / "dino")), "checkpoint": "dino_deitsmall8_pretrain.pth",
                "checkpoint_sha256": DINO_WEIGHTS_SHA256, "is_mock": False}
    meta = {"layer": 11, "facet": "key", "bin": False, "stride": 4, "normalisation": "(x-0.45)/0.225",
            "upsampling": "bilinear to image size", "consumer": "s4d.baselines.snerl.model.dino_target"}
    return identity, {"features": np.stack(feats).astype(np.float32)}, meta


if __name__ == "__main__":
    main(handle)
