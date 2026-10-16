"""3D-MVP encoder size table: the original RVT-1 / 3D-MVP configurations (computed on the meta device) and the
size-matched one (measured on CPU), against SinCro's RL-time encoder, plus the RL-time cost of one frame.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/mvp3d_size.py [--export runs/pretrain/<name>/encoder.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

guard_gpus()

import torch  # noqa: E402

from s4d.baselines.mvp3d.model import MVP3DConfig, build_encoder, load_export  # noqa: E402
from s4d.baselines.mvp3d.size import rl_cost, size_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=None, help="measure this encoder export (default: a fresh size-matched one)")
    ap.add_argument("--out", default=str(REPO / "docs/baselines_mvp3d_size.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    table = size_table()
    if args.export:
        payload, encoder = load_export(args.export)
        img_size = int(payload["model_cfg"]["img_size"])
    else:
        torch.manual_seed(0)
        encoder, img_size = build_encoder(MVP3DConfig()).eval(), MVP3DConfig().img_size
    table["rl_time_per_frame"] = rl_cost(encoder, img_size)
    for name, row in table["rows"].items():
        print(f"{name}: {row['encoder_params']:,} params, {row['encoder_flops_per_frame']:,} FLOPs/frame, "
              f"decoder {row['decoder_params']:,} params")
    print(json.dumps(table["rl_time_per_frame"], indent=1))
    Path(args.out).write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
