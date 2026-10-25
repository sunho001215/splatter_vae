"""Before training (encoder pilot hn, 2026-10-25): how the robot-state rule acts on the TRAINING episodes of each task.

For every training sample (episode, first frame t0, stride s) of the pretraining dataset (strides 2/4/6, the same
episode split), the gap-eligible candidate windows (2s + 2 <= |t0' - t0| <= 40) and the fraction the rule keeps (hand
moved >= 1 cm or gripper changed >= 0.1 between the windows' last frames), the fraction of samples with a valid
negative, and the hand displacement among kept pairs. Reads robot state only (obs[:, 0:4]); thresholds are the
registered ones and are not tuned on this or any result.

    python -I scripts/hardneg_report.py --tasks hammer pick-place bin-picking --out runs/setup/encp-hardneg-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402

from s4d.data.hardneg import DEFAULTS, HardNegativeSampler, robot_state  # noqa: E402
from s4d.data.metaworld.dataset import TRAIN_STRIDES, split_episodes  # noqa: E402

ROOT = Path("/home/ws/data/metaworld/splatter4d_v1")


def report(task: str, root: Path = ROOT, rule: dict = DEFAULTS) -> dict:
    path = root / f"{task}.hdf5"
    train, _ = split_episodes(path, 0.96, 0)
    with h5py.File(path, "r") as f:
        robot = {ep: robot_state(f["episodes"][ep]["obs"][:, :4], "metaworld") for ep in train}
        lengths = {ep: int(f["episodes"][ep].attrs["length"]) for ep in train}
    span = 2 * max(TRAIN_STRIDES)  # the dataset's anchor rule: every stride fits
    samples = [(ep, t0, s) for ep in train for t0 in range(lengths[ep] - span) for s in TRAIN_STRIDES]
    sampler = HardNegativeSampler(robot, **rule)
    out = {"task": task, "data": str(path), "train_episodes": len(train), **sampler.statistics(samples)}
    out["by_stride"] = {str(s): {k: v for k, v in sampler.statistics([x for x in samples if x[2] == s]).items()
                                 if k in ("candidate_pairs", "kept_fraction", "valid_negative_fraction",
                                          "hand_displacement_m_quantiles")} for s in TRAIN_STRIDES}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["hammer", "pick-place", "bin-picking"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {"rule": DEFAULTS, "tasks": {t: report(t) for t in args.tasks}}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for t, r in result["tasks"].items():
        q = r["hand_displacement_m_quantiles"]
        print(f"{t}: kept {r['kept_fraction']:.3f} of {r['candidate_pairs']} pairs; valid negative for "
              f"{r['valid_negative_fraction']:.3f} of {r['samples']} samples; hand disp (m) median {q['0.5']:.3f} "
              f"[5%: {q['0.05']:.3f}, 95%: {q['0.95']:.3f}]")


if __name__ == "__main__":
    main()
