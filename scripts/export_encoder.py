"""Export the encoder (weights + config) to encoder.pt for downstream policies.

python scripts/export_encoder.py --run outputs/metaworld-hammer-full-0 [--checkpoint ...] [--out encoder.pt]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import REPO, guard_gpus

guard_gpus()

import torch  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.model.encoder import Encoder, EncoderConfig  # noqa: E402

EXPORT_FORMAT = "splatter4d-encoder-v1"


def export(run_dir: Path, checkpoint: Path | None, out: Path) -> Path:
    if REPO.resolve() not in out.resolve().parents:
        raise ValueError("encoder exports must remain inside the new repository")
    cfg = load_config([run_dir / "config.yaml"])
    settings = dict(cfg["model"]["encoder"])
    backbone = settings.pop("backbone", "scratch")
    encoder_cfg = EncoderConfig.dinov2_vits14(**settings) if backbone == "dinov2_vits14" else EncoderConfig(**settings)
    encoder_cfg.backbone, encoder_cfg.pretrained_path = "scratch", None
    encoder = Encoder(encoder_cfg)
    ckpt = checkpoint or run_dir / "checkpoints" / "latest.pt"
    # Own training checkpoints include NumPy/Python RNG state. Do not use untrusted checkpoints.
    training = torch.load(ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(training["encoder"], strict=True)
    step = int(training["step"])
    payload = {
        "format": EXPORT_FORMAT,
        "encoder_config": asdict(encoder.cfg),
        "state_dict": encoder.state_dict(),
        "step": step,
        "source_checkpoint": str(ckpt),
        "run_name": cfg.get("run", {}).get("name"),
    }
    payload["encoder_config"]["pretrained_path"] = None  # exported weights already contain the backbone
    payload["encoder_config"]["backbone"] = (
        "scratch" if payload["encoder_config"]["backbone"] == "dinov2_vits14" else payload["encoder_config"]["backbone"]
    )
    torch.save(payload, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "encoder.pt"
    print(f"exported {export(run_dir, Path(args.checkpoint) if args.checkpoint else None, out)}")


if __name__ == "__main__":
    main()
