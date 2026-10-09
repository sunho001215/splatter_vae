"""Apply the pre-registered decision rules of the mid-campaign review to full-split evaluations (docs/EXPERIMENT_LOG.md).

Analysis only (not used by any job). Reads ``runs/<job>/eval/step_*/summary.json`` written by ``scripts/evaluate.py``.

    python analysis/review_rules.py item2 <screen-name>          # e.g. crop -> s2-screen-crop-{hammer,pick-place}
    python analysis/review_rules.py item4 <screen-name>          # e.g. g4a-declr3-dec4x256 vs g-ref
    python analysis/review_rules.py table <job> [<job> ...]       # metrics side by side
    python analysis/review_rules.py proxies <run-prefix>          # item 5: last-5 mean and recovery, seeds 1000/1001
    python analysis/review_rules.py s1                            # directives S1 (R vs V1 inv-only vs V2 + self-render)
    python analysis/review_rules.py item1                         # directive item 1 (D0-D4)
    python analysis/review_rules.py item2m3d [<m2d stem>]         # directive item 2 (M3D vs M2D, representation part)
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


def main() -> None:
    command, *rest = sys.argv[1:]
    commands = {"item2": lambda: item2(rest[0]), "item4": lambda: item4(rest[0]), "table": lambda: table(rest),
                "proxies": lambda: proxies(rest[0]), "s1": s1, "item1": directive_item1, "item2m3d": directive_item2}
    commands[command]()


def proxy(run: str) -> dict:
    """Item 5 quantities of one RL proxy run: last-5 mean training-camera success at the end, and recovery after the
    perturbations at 100k / 200k (success at 150k / 250k minus the best success at or before 100k / 200k)."""
    rows = [json.loads(line) for line in (REPO / "runs" / run / "eval.jsonl").read_text().splitlines()]
    by_step = {int(r["step"]): r for r in rows}
    tr = {s: float(r["train_cameras_success"]) for s, r in by_step.items()}
    steps = sorted(tr)
    out = {
        "final_step": steps[-1],
        "last5_train": sum(tr[s] for s in steps[-5:]) / 5,
        "last5_heldout": sum(float(by_step[s].get("heldout_cameras_success", math.nan)) for s in steps[-5:]) / 5,
        "last5_traj": sum(float(by_step[s].get("trajectories_success", math.nan)) for s in steps[-5:]) / 5,
        "peak_train": max(tr.values()),
    }
    for at, before in ((150000, 100000), (250000, 200000)):
        if at in tr:
            out[f"recovery_{at // 1000}k"] = tr[at] - max(tr[s] for s in steps if s <= before)
    return out


def proxies(prefix: str, seeds=(1000, 1001)) -> None:
    rows = {seed: proxy(f"{prefix}-s{seed}") for seed in seeds}
    for seed, r in rows.items():
        print(seed, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))
    keys = rows[seeds[0]].keys()
    print("mean", json.dumps({k: round(sum(r[k] for r in rows.values()) / len(rows), 3) for k in keys if k != "final_step"}))


# ------------------------------------------------------------------------------------------ directives of 2026-10-09
# Two-seed standard: margin = max(|reference seed 0 - reference seed 1|, floor), "better/worse beyond the margin"
# compares the two-seed means. Motion metrics (relative EPE overall and per bin, EPE, CD-motion, dynamic share,
# hand-velocity R2) are read at stride 6, every other metric as the mean of strides 2 and 6; pair 0->2 throughout.
# name: (reader, column, floor, higher_is_better, relative_margin)
METRICS = {
    "psnr": ("mean", "psnr", 0.2, True, False),
    "psnr_moving": ("mean", "psnr_moving", 0.2, True, False),
    "psnr_traj": ("mean", "psnr_traj", 0.2, True, False),
    "retrieval_traj": ("mean", "retrieval_top1_traj_val", 0.03, True, False),
    "r2_pos_traj": ("mean", "r2_hand_pos_val_traj", 0.05, True, False),
    "r2_vel_traj": ("s6", "r2_hand_vel_val_traj", 0.05, True, False),
    "r2_vel_train": ("s6", "r2_hand_vel_val_traincams", 0.05, True, False),
    "rel_epe": ("s6", "rel_epe_02", 0.03, False, False),
    "relepe_bin1": ("s6", "relepe_bin1_02", 0.03, False, False),
    "relepe_bin2": ("s6", "relepe_bin2_02", 0.03, False, False),
    "relepe_bin3": ("s6", "relepe_bin3_02", 0.03, False, False),
    "epe_mm": ("s6", "epe_moving_mm_02", 1.0, False, False),
    "dyn_share": ("s6", "dyn_alpha_share_moving", 0.05, True, False),
    "utilisation": ("mean", "utilisation", 0.02, True, False),
    "hidden": ("mean", "hidden_fraction", 0.02, False, False),
    "floater": ("mean", "floater_fraction", 0.02, False, False),
    "cd_render_traj": ("sym_mean", "render_traj", 0.05, False, True),
    "cd_centers_vis": ("sym_mean", "centers_vis", 0.05, False, True),
    "cd_motion_dyn": ("sym_s6", "motion_dyn", 0.05, False, True),
}


def read(s: dict, metric: str) -> float:
    reader, column = METRICS[metric][:2]
    if reader == "mean":
        return mean_strides(s, column)
    if reader == "s6":
        return value(s, column, 6)
    if reader == "sym_mean":
        return sum(sym_p90(s, column, k) for k in STRIDES) / len(STRIDES)
    return sym_p90(s, column, 6)


def two_seed(jobs: list[str], metric: str) -> list[float]:
    return [read(summary(j), metric) for j in jobs]


def compare(variant: list[str], reference: list[str], metric: str, margin_from: list[str] | None = None) -> dict:
    """Two-seed means of variant and reference; the margin comes from ``margin_from`` (default: the reference)."""
    _, _, floor, higher, relative = METRICS[metric]
    v, r = two_seed(variant, metric), two_seed(reference, metric)
    m0, m1 = two_seed(margin_from or reference, metric)
    vm, rm = sum(v) / len(v), sum(r) / len(r)
    if relative:
        margin = max(abs(m0 - m1) / (0.5 * (m0 + m1)), floor)
        delta = vm / rm - 1.0
    else:
        margin = max(abs(m0 - m1), floor)
        delta = vm - rm
    signed = delta if higher else -delta
    return {"variant": vm, "reference": rm, "delta": delta, "margin": margin, "better": signed > margin,
            "worse": signed < -margin, "seeds_variant": v, "seeds_reference": r}


def show(title: str, rows: dict) -> None:
    print(title)
    for (task, metric), c in rows.items():
        print(f"  {task:10s} {metric:15s} var {c['variant']:8.4f} ref {c['reference']:8.4f} delta {c['delta']:+8.4f} "
              f"margin {c['margin']:.4f} {'BETTER' if c['better'] else 'worse' if c['worse'] else '='}")


def margins(reference: dict, names) -> None:
    """Print the margins of the reference's two seeds (written into EXPERIMENT_LOG before comparisons)."""
    for task in TASKS:
        for name in names:
            c = compare(reference[task], reference[task], name)
            print(f"  margin {task:10s} {name:15s} {c['margin']:.4f}  seeds {[round(x, 4) for x in c['seeds_reference']]}")


def s1_jobs(variant: str) -> dict:
    return {t: [f"fulleval-s1v-{variant}-seed{k}-{t}-100k" for k in (0, 1)] for t in TASKS}


def s1() -> str:
    """S1 rule (EXPERIMENT_LOG 2026-10-09). Returns 'V1', 'V2' or 'R'."""
    R = s1_jobs("synth")
    names = ("dyn_share", "rel_epe", "retrieval_traj", "r2_pos_traj", "r2_vel_traj", "psnr", "psnr_moving", "cd_render_traj")
    margins(R, names)
    qualifies = {}
    for label, variant in (("V1", "synthinv"), ("V2", "synthsr")):
        V = s1_jobs(variant)
        rows = {(t, n): compare(V[t], R[t], n) for t in TASKS for n in names}
        show(f"{label} ({variant}) vs R", rows)
        a = rows[("pick-place", "dyn_share")]["variant"] >= 0.5 and rows[("pick-place", "rel_epe")]["better"]
        b = not any(rows[(t, n)]["worse"] for t in TASKS for n in ("retrieval_traj", "r2_pos_traj", "r2_vel_traj"))
        c = not any(rows[(t, n)]["worse"] for t in TASKS for n in ("psnr", "psnr_moving", "cd_render_traj"))
        qualifies[label] = a and b and c
        print(f"  {label}: (a) {a} (b) {b} (c) {c} -> {'qualifies' if qualifies[label] else 'does not qualify'}")
    if qualifies["V1"] and qualifies["V2"]:
        tie = ("retrieval_traj", "r2_pos_traj", "r2_vel_traj", "psnr_moving", "rel_epe", "cd_render_traj")
        rows = {(t, n): compare(s1_jobs("synthsr")[t], s1_jobs("synthinv")[t], n, margin_from=R[t]) for t in TASKS for n in tie}
        show("V2 vs V1 (margins from R)", rows)
        wins = sum(c["better"] for c in rows.values())
        choice = "V2" if wins >= 7 else "V1"
        print(f"  V2 better than V1 on {wins} of 12 -> {choice}")
    else:
        choice = "V1" if qualifies["V1"] else "V2" if qualifies["V2"] else "R"
    print(f"S1 decision: {choice}")
    return choice


def d_jobs(variant: str) -> dict:
    return {t: [f"fulleval-i1-{variant}-seed{k}-{t}-100k" for k in (0, 1)] for t in TASKS}


ITEM1_GUARDS = ("psnr", "psnr_moving", "psnr_traj", "cd_render_traj", "rel_epe", "retrieval_traj", "r2_pos_traj", "r2_vel_traj")


def directive_item1() -> str:
    """Item-1 rule. Returns the adopted variant ('d0'..'d4')."""
    D0 = d_jobs("d0")
    margins(D0, ("utilisation", "hidden", "cd_centers_vis") + ITEM1_GUARDS)

    def passes(variant: str) -> bool:
        V = d_jobs(variant)
        rows = {(t, n): compare(V[t], D0[t], n) for t in TASKS for n in ("utilisation", "hidden", "cd_centers_vis") + ITEM1_GUARDS}
        show(f"{variant} vs d0", rows)
        i = all(rows[(t, n)]["better"] for t in TASKS for n in ("utilisation", "hidden", "cd_centers_vis"))
        ii = not any(rows[(t, n)]["worse"] for t in TASKS for n in ITEM1_GUARDS)
        print(f"  {variant}: (i) {i} (ii) {ii}")
        return i and ii

    if passes("d2"):
        beats = {}
        for variant in ("d3", "d4"):
            rows = {(t, n): compare(d_jobs(variant)[t], d_jobs("d2")[t], n, margin_from=D0[t]) for t in TASKS for n in ITEM1_GUARDS}
            show(f"{variant} vs d2 (margins from d0)", rows)
            beats[variant] = sum(c["better"] for c in rows.values()) >= 3 and not any(c["worse"] for c in rows.values())
            print(f"  {variant} beats d2: {beats[variant]}")
        choice = "d1" if all(beats.values()) else "d3" if beats["d3"] else "d4" if beats["d4"] else "d2"
    else:
        choice = "d1" if passes("d1") else "d0"
    print(f"item 1 decision: {choice}")
    return choice


ITEM2_MOTION = ("relepe_bin1", "relepe_bin2", "relepe_bin3", "epe_mm", "cd_motion_dyn", "dyn_share", "r2_vel_train", "r2_vel_traj")


def directive_item2(m2d: str | None = None) -> bool:
    """Item-2 representation part (the RL part is checked with ``proxies``). ``m2d``: the item-1 winner's job stem."""
    m2d = m2d or (sys.argv[2] if len(sys.argv) > 2 else "d2")
    M2 = d_jobs(m2d)
    M3 = {t: [f"fulleval-i2-m3d-seed{k}-{t}-100k" for k in (0, 1)] for t in TASKS}
    margins(M2, ITEM2_MOTION + ("psnr", "psnr_moving"))
    rows = {(t, n): compare(M3[t], M2[t], n) for t in TASKS for n in ITEM2_MOTION + ("psnr", "psnr_moving")}
    show("M3D vs M2D", rows)
    per_task = {t: sum(rows[(t, n)]["better"] for n in ITEM2_MOTION) for t in TASKS}
    worse_guard = any(rows[(t, n)]["worse"] for t in TASKS for n in ("psnr", "psnr_moving"))
    ok = sum(per_task.values()) >= 9 and min(per_task.values()) >= 3 and not worse_guard
    print(f"item 2 representation part: better on {per_task} (need >= 9, >= 3 per task), guard worse {worse_guard} -> {ok}")
    return ok


if __name__ == "__main__":
    main()
