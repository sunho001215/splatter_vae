"""Prepare the run directory of a full-split evaluation job: a copy of the pretraining run's config.yaml with the
evaluation's run name and without the pretraining GPU record (``scripts/evaluate.py --run`` reads it).

    python analysis/eval_config.py runs/pretrain/<pretraining run> runs/<evaluation job> [...pairs]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def prepare(source: Path, target: Path) -> None:
    cfg = yaml.safe_load((source / "config.yaml").read_text())
    cfg.setdefault("run", {})["name"] = target.name
    cfg["run"].pop("gpus", None)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "config.yaml"
    if out.exists() and yaml.safe_load(out.read_text()) != cfg:
        raise FileExistsError(f"{out} exists with a different configuration")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or len(args) % 2:
        raise SystemExit(__doc__)
    for src, dst in zip(args[::2], args[1::2]):
        prepare(Path(src), Path(dst))
        print(f"{dst}/config.yaml <- {src}/config.yaml")
