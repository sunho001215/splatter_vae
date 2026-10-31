"""DINO teacher features exactly as the repository's SNeRL baseline consumes them.

Runs in the repository venv (the consumer's own environment: third_party/dino at the pinned submodule commit and
the sha256-checked official ``dino_deitsmall8_pretrain.pth``) and calls the consumer's functions directly:
``s4d.baselines.snerl.upstream.dino_extractor`` (ViTExtractor('dino_vits8', stride 4)) and
``s4d.baselines.snerl.model.dino_target`` (scalar normalisation (x-0.45)/0.225, layer 11, facet "key", no bin,
no CLS, bilinear upsampling of the 31x31x384 token grid to the image size). SNeRL itself computes these online
from the training images; this cache is optional (inspection / offline reuse) and is keyed by the image hash.

task ``dino_snerl``: input ``images`` (N,H,W,3) float32 in [0,1] at the consumer resolution (128x128).
params ``output``: ``"tokens"`` (default) returns ``tokens`` (N,h,w,384) float32 - the descriptor grid *before*
the consumer's bilinear upsampling - and verifies on the first image that F.interpolate(tokens) equals
``dino_target`` exactly (``meta.verified_against_consumer``); ``"upsampled"`` returns ``features`` (N,H,W,384).
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
    import torch.nn.functional as F

    extractor = dino_extractor(device)
    mode = request["params"].get("output", "tokens")
    feats, verified = [], None
    for k, image in enumerate(inputs["images"]):
        target = torch.from_numpy(np.asarray(image, np.float32)).to(device)
        if mode == "upsampled":
            feats.append(dino_target(extractor, target).cpu().numpy())
            continue
        with torch.no_grad():  # the first three lines of dino_target, verbatim
            x = (target - 0.45) / 0.225
            tok = extractor.extract_descriptors(x.unsqueeze(dim=0).permute(0, 3, 1, 2), 11, "key", False)
            tok = tok.view(1, *extractor.num_patches, -1)
        if k == 0:
            up = F.interpolate(tok.permute(0, 3, 1, 2), tuple(target.shape[:2]), mode="bilinear").permute(0, 2, 3, 1)[0]
            verified = bool(torch.equal(up, dino_target(extractor, target)))
            if not verified:
                raise RuntimeError("token path disagrees with s4d.baselines.snerl.model.dino_target")
        feats.append(tok[0].cpu().numpy())
    identity = {"name": "DINO ViT-S/8 (SNeRL teacher)", "repo": "facebookresearch/dino",
                "commit": git_commit(str(REPO / "third_party" / "dino")), "checkpoint": "dino_deitsmall8_pretrain.pth",
                "checkpoint_sha256": DINO_WEIGHTS_SHA256, "is_mock": False}
    meta = {"layer": 11, "facet": "key", "bin": False, "stride": 4, "normalisation": "(x-0.45)/0.225",
            "upsampling": "bilinear (F.interpolate) to image size", "output": mode,
            "consumer": "s4d.baselines.snerl.model.dino_target", "verified_against_consumer": verified,
            "torch": torch.__version__, "device": device}
    key = "tokens" if mode == "tokens" else "features"
    return identity, {key: np.stack(feats).astype(np.float32)}, meta


if __name__ == "__main__":
    main(handle)
