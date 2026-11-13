"""Continue only from a verified full-state checkpoint; never fall back to scratch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import guard_gpus

guard_gpus()

from s4d.config import load_config  # noqa: E402
from s4d.train.continuation import file_sha, stage_checkpoint, validate_resume_state  # noqa: E402
from s4d.train.schedule import training_end  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--source-checkpoint", type=Path, required=True)
    p.add_argument("--source-sha256")  # B binds to the immutable dependency checkpoint at launch
    p.add_argument("--baseline-config", type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--start-step", type=int, required=True)
    p.add_argument("--stop-step", type=int, choices=(300000, 400000), required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    a = p.parse_args()
    if file_sha(a.baseline_config) != a.baseline_sha256:
        raise ValueError("saved 200k baseline config changed")
    if any(
        x.partition("=")[0] not in ("data.store", "data.split_manifest", "data.gate_report", "model.anchor_stats")
        for x in a.set
    ):
        raise ValueError("only verified input path overrides are allowed")
    cfg = load_config(
        [a.baseline_config, "configs/realworld/ours_ext/continuation.yaml"], [f"train.stop_step={a.stop_step}", *a.set]
    )
    training_end(cfg)
    from s4d.train.continuation import science_config

    if science_config(cfg) != science_config(load_config([a.baseline_config])):
        raise ValueError("input overrides may only map the same data into the verified local mirror")
    import torch

    from ops.resources.gpu_guard import validate_runtime_path

    # Use actual CPU parameters to check every AdamW group/id/shape before staging or accepting a resume.
    from s4d.train.loop import build_model, build_optimizer

    validation_model = build_model(cfg)
    validation_optimizer, _ = build_optimizer(validation_model, cfg)
    source = torch.load(a.source_checkpoint, map_location="cpu", weights_only=False)
    validate_resume_state(
        source, cfg, optimizer=validation_optimizer, lower=a.start_step, upper=a.stop_step, exact_step=a.start_step
    )
    run = validate_runtime_path(a.output_root / a.name)
    digest = a.source_sha256 or file_sha(a.source_checkpoint)
    copied = stage_checkpoint(a.source_checkpoint, run / "checkpoints" / a.source_checkpoint.name, digest)
    (run / "resume_source.json").write_text(
        json.dumps(
            {"source": str(a.source_checkpoint), "sha256": digest, "copy": str(copied), "start_step": a.start_step}, indent=2
        )
    )
    latest = run / "checkpoints/latest.pt"
    resume = latest if latest.exists() else copied
    state = torch.load(resume, map_location="cpu", weights_only=False)
    validate_resume_state(state, cfg, optimizer=validation_optimizer, lower=a.start_step, upper=a.stop_step)
    del state, source, validation_optimizer, validation_model, _
    sys.argv = [
        "train.py",
        "--config",
        str(a.baseline_config),
        "configs/realworld/ours_ext/continuation.yaml",
        "--name",
        a.name,
        "--output-root",
        str(a.output_root),
        "--resume",
        str(resume),
        "--set",
        f"train.stop_step={a.stop_step}",
        *a.set,
    ]
    import train

    try:
        train.main()
    finally:
        if file_sha(a.source_checkpoint) != digest or file_sha(copied) != digest:
            raise ValueError("source/copy checksum changed during continuation")


if __name__ == "__main__":
    main()
