"""Apply the pre-registered decision rules of the mid-campaign review to full-split evaluations (docs/EXPERIMENT_LOG.md).

Analysis only (not used by any job). Reads ``runs/<job>/eval/step_*/summary.json`` written by ``scripts/evaluate.py``.

    python analysis/review_rules.py item2 <screen-name>          # e.g. crop -> s2-screen-crop-{hammer,pick-place}
    python analysis/review_rules.py item4 <screen-name>          # e.g. g4a-declr3-dec4x256 vs g-ref
    python analysis/review_rules.py table <job> [<job> ...]       # metrics side by side
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRIDES = (2, 6)
TASKS = ("hammer", "pick-place")


def summary(job: str) -> dict:
    files = sorted((REPO / "runs" / job / "eval").glob("step_*/summary.json"))
    if not files:
        raise FileNotFoundError(f"no evaluation summary for {job}")
    return json.loads(files[-1].read_text())


def value(s: dict, name: str, stride: int) -> float:
    v = s.get(f"metric/{name}@s{stride}")
    return float("nan") if v is None else float(v)


def mean_strides(s: dict, name: str) -> float:
    return sum(value(s, name, k) for k in STRIDES) / len(STRIDES)


def sym_p90(s: dict, kind: str, stride: int) -> float:
    return 0.5 * (value(s, f"cd_{kind}_p2g_p90", stride) + value(s, f"cd_{kind}_g2p_p90", stride))


def item2(screen: str, base_jobs=("fulleval2-s1-base-{task}-100k",)) -> bool:
    """2d: on both tasks (mean of strides 2 and 6) trajectory-set retrieval, hand-position R2 and hand-velocity R2 each
    improve by >= 0.10; training-camera moving PSNR not lower by more than 0.5 dB; CD-render (trajectory, symmetric
    p90) not higher by more than 5 %."""
    adopt = True
    for task in TASKS:
        s = summary(f"fulleval2-s2-screen-{screen}-{task}-100k")
        b = summary(base_jobs[0].format(task=task))
        checks = {
            "retrieval_traj": mean_strides(s, "retrieval_top1_traj_val") - mean_strides(b, "retrieval_top1_traj_val") >= 0.10,
            "r2_hand_pos_traj": mean_strides(s, "r2_hand_pos_val_traj") - mean_strides(b, "r2_hand_pos_val_traj") >= 0.10,
            "r2_hand_vel_traj": mean_strides(s, "r2_hand_vel_val_traj") - mean_strides(b, "r2_hand_vel_val_traj") >= 0.10,
            "psnr_moving": mean_strides(s, "psnr_moving") - mean_strides(b, "psnr_moving") >= -0.5,
            "cd_render_traj": sum(sym_p90(s, "render_traj", k) for k in STRIDES)
            <= 1.05 * sum(sym_p90(b, "render_traj", k) for k in STRIDES),
        }
        print(task, json.dumps(checks))
        adopt = adopt and all(checks.values())
    print(f"item 2d, screen {screen}: {'ADOPT' if adopt else 'reject'}")
    return adopt


def item4(screen: str, reference: str = "g-ref") -> bool:
    """Promote if better than the reference on >= 3 of 5 metrics by the margins and worse on none beyond them."""
    s, r = summary(f"fulleval2-screen-{screen}-6k"), summary(f"fulleval2-screen-{reference}-6k")
    rows = []
    for name, margin, higher in (("psnr", 0.5, True), ("psnr_moving", 0.5, True), ("rel_epe_02", 0.03, False)):
        d = mean_strides(s, name) - mean_strides(r, name)
        rows.append((name, d, (d > margin) if higher else (d < -margin), (d < -margin) if higher else (d > margin)))
    for kind in ("centers", "motion_dyn"):
        sv = sum(sym_p90(s, kind, k) for k in STRIDES)
        rv = sum(sym_p90(r, kind, k) for k in STRIDES)
        rel = sv / rv - 1.0 if rv else math.nan
        rows.append((f"cd_{kind}_sym_p90", rel, rel < -0.10, rel > 0.10))
    for name, d, better, worse in rows:
        print(f"{name:22s} delta {d:+.4f} better={better} worse={worse}")
    promote = sum(b for *_, b, _ in rows) >= 3 and not any(w for *_, w in rows)
    print(f"item 4, screen {screen}: {'PROMOTE' if promote else 'no'}")
    return promote


TABLE = (
    "psnr", "psnr_moving", "psnr_near", "psnr_near_covered", "psnr_traj", "psnr_traj_covered", "psnr_heldout",
    "psnr_heldout_covered", "oracle_coverage_traj", "oracle_psnr_covered_traj", "rel_epe_02", "retrieval_top1_train",
    "retrieval_top1_train_val", "retrieval_top1_near_val", "retrieval_top1_traj_val", "retrieval_top1_heldout_val",
    "r2_hand_pos_val_traincams", "r2_hand_vel_val_traincams", "r2_hand_pos_val_near", "r2_hand_vel_val_near",
    "r2_hand_pos_val_traj", "r2_hand_vel_val_traj", "r2_obj_pos_val_traj", "cd_centers_p2g_p50", "cd_centers_p2g_p90",
    "cd_centers_g2p_p50", "cd_centers_g2p_p90", "cd_centers_dyn_p2g_p90", "cd_centers_dyn_g2p_p90",
    "cd_render_traj_p2g_p50", "cd_render_traj_p2g_p90", "cd_render_traj_g2p_p50", "cd_render_traj_g2p_p90",
    "cd_motion_dyn_p2g_p50", "cd_motion_dyn_p2g_p90", "cd_motion_dyn_g2p_p90",
)


def table(jobs: list[str]) -> None:
    data = {j: summary(j) for j in jobs}
    print(f"{'metric (s2 / s6)':28s}" + "".join(f"{j[-30:]:>32s}" for j in jobs))
    for name in TABLE:
        row = f"{name:28s}"
        for j in jobs:
            a, b = value(data[j], name, 2), value(data[j], name, 6)
            row += f"{f'{a:.3f} / {b:.3f}':>32s}"
        print(row)


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    {"item2": lambda: item2(rest[0]), "item4": lambda: item4(rest[0]), "table": lambda: table(rest)}[command]()
