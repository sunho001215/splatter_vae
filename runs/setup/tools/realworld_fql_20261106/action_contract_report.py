"""Evidence for the real-world action contract (s4d/rwlearn/actions.py) over every local raw recording (read-only):
validity counts, round trip policy action -> recorded sent command, normalised ranges, command-vs-measured
agreement, lead saturation, gripper usage, recording segments. Writes a JSON report (docs/realworld_fql_action_report.json).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> python fql_..._report.py --out docs/realworld_fql_action_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from s4d.rwlearn import actions as A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ws/data/real_world")
    ap.add_argument("--tasks", nargs="+", default=["drawer-open", "mug-hanging"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"root": args.root, "tasks": {}}
    for task in args.tasks:
        acts, rows, reasons, rt, segs, lead_lim, cmp = [], [], {}, 0.0, [], [], []
        for path in sorted(glob.glob(f"{args.root}/{task}/episode_*.h5")):
            raw = A.read_raw_episode(path)
            A.check_contract_preconditions(raw)
            c = A.contract_for(raw)
            segs.append(len(raw.segments["t_start"]))
            lead_lim.append(float(raw.lead_limited.mean()))
            for serial in raw.serials:
                cam = raw.cameras[serial]
                tab = A.step_table(raw, cam["t"], cam["segment"], nominal_period_s=c.nominal_period_s)
                ok = np.flatnonzero(tab.transition_ok)
                a = tab.action_phys[ok]
                acts.append(a)
                out = A.to_robot_command(A.matrix_to_pose7(tab.measured_pose[ok]), c.normalise(a), c)
                rt = max(rt, float(np.abs(A.pose7_to_matrix(out["flange_pose"]) - tab.command_pose[ok]).max()))
                meas = A.pose_delta(tab.measured_pose[ok], tab.measured_pose[ok + 1])
                cos = np.sum(meas[:, :3] * a[:, :3], 1) / np.maximum(
                    np.linalg.norm(meas[:, :3], axis=1) * np.linalg.norm(a[:, :3], axis=1), 1e-12)
                moving = np.linalg.norm(meas[:, :3], axis=1) > 2e-3
                cmp.append((cos[moving], np.linalg.norm(meas[:, :3], axis=1), np.linalg.norm(a[:, :3], axis=1)))
                for k, v in tab.reasons.items():
                    reasons[k] = reasons.get(k, 0) + int(v)
                rows.append(len(ok))
        a = np.concatenate(acts)
        cosines = np.concatenate([x[0] for x in cmp])
        meas_n, cmd_n = np.concatenate([x[1] for x in cmp]), np.concatenate([x[2] for x in cmp])
        report["tasks"][task] = {
            "episodes": len(segs), "camera_timelines": len(rows), "valid_transitions": int(sum(rows)),
            "invalid_reasons": reasons, "recording_segments_per_episode": dict(zip(*map(
                lambda x: x.tolist(), np.unique(segs, return_counts=True)))),
            "round_trip_max_abs_error": rt, "contract": c.to_json(),
            "normalised_abs_max_per_dim": np.abs(c.normalise(a)).max(0).round(4).tolist(),
            "command_offset_m_percentiles": np.percentile(cmd_n, [5, 50, 95, 100]).round(4).tolist(),
            "measured_motion_per_step_m_percentiles": np.percentile(meas_n, [5, 50, 95, 100]).round(4).tolist(),
            "rotation_increment_deg_percentiles": np.degrees(np.percentile(np.linalg.norm(a[:, 3:6], axis=1),
                                                                           [50, 95, 100])).round(2).tolist(),
            "cosine_command_offset_vs_measured_motion_when_moving": {
                "median": float(np.median(cosines)), "p10": float(np.percentile(cosines, 10))},
            "lead_limited_fraction_mean": float(np.mean(lead_lim)),
            "gripper_command": {"fraction_closed_gt_0.5": float((a[:, 6] > 0.5).mean()),
                                "fraction_in_transition": float(((a[:, 6] > 0.05) & (a[:, 6] < 0.95)).mean())},
        }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({t: {k: v for k, v in r.items() if k != "contract"} for t, r in report["tasks"].items()}, indent=1))


if __name__ == "__main__":
    main()
