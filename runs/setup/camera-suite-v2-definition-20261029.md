# Camera suite v2 (`cameras_v2`): frozen definition

Frozen 2026-10-28T17:56:06+00:00 before any policy evaluation. Evaluation only: training cameras, training data, pretraining and RL training are unchanged; the original held-out cameras eval0..3 and their `eval.jsonl` history are kept and reported separately as an **occlusion-heavy stress test**. Inputs to the selection: MuJoCo geometry, segmentation and depth on representative scripted-expert / random trajectories only.

## Selection rule history

1. Registered rule `runs/setup/camera-suite-v2-rule-20261029.json` (sha256 `983a71690902f9cb`, registered before any candidate was measured): 0/138 candidates eligible. 0 eligible: step 4 required matching the least observable training camera on ~15 criteria of every task; the step-7 fallback needed exceptions on 6 of 12 tasks.
2. Amendment A1 `runs/setup/camera-suite-v2-rule-amendment-A1-20261029.json` (registered before its outcome was computed): 0/138 eligible. 0 eligible: the per-task score floors (each task's minimum over the training cameras, 0.62-0.92) are not met on all tasks by any admissible viewpoint.
3. Amendment A2 `runs/setup/camera-suite-v2-rule-amendment-A2-20261029.json` (registered before its outcome was computed; used): the 4 admissible cameras (>= 25 deg from every training camera, >= 30 deg pairwise) maximising the smallest worst-task score (set minimum 0.555); task exceptions only where a chosen camera falls outside the training rig's global worst case (obj_hidden_fraction_max 0.2692, ee_hidden_fraction_max 0.1121, goal_hidden_fraction_max 0.3978, score_min 0.6163).
4. Amendment A3 `runs/setup/camera-suite-v2-rule-amendment-A3-20261029.json` (moving-path validation, registered 2026-10-28T17:55:36+00:00, before any policy evaluation on these paths): the per-target hidden-run limit of step 8 is not monotonic in amplitude (registered 10% reductions of circular fail at 27/16.2, 24/14.4, 21/12.6, 16/9.6, 14/8.4, 12/7.2 deg and pass at 18/10.8) and also fails the unchanged 1x circular path; the directive's criteria (collision, clipping, prolonged loss of the entire workspace) are all met. The per-target hidden runs stay reported (below); the literal step-8 outcome (lateral 0.216 / 0.288 m, circular 18/10.8 deg for both levels) is a documented, non-adopted alternative and not an evaluation group. The 3x3 median of the near-plane check is diagnostic-only: it never touches policy observations or evaluation renders.

## Held-out cameras `heldout_v2` (common rig)

| camera | tilt | azimuth | separation from train0..5 (deg) | min | separation from eval0..3 (deg) | multi-view slot |
|---|---|---|---|---|---|---|
| vis0 | 55 | 140 | 74, 27, 65, 36, 76, 26 | 26.0 | 53, 20, 69, 52 | train5 |
| vis1 | 25 | 200 | 108, 26, 86, 36, 124, 44 | 25.7 | 105, 67, 75, 42 | train1 |
| vis2 | 40 | 300 | 44, 79, 27, 56, 71, 105 | 27.3 | 83, 100, 15, 42 | train2 |
| vis3 | 25 | 340 | 26, 108, 36, 86, 44, 124 | 25.7 | 67, 105, 42, 75 | train0 |

Pairwise separation (deg): vis0-vis1 52.7, vis0-vis2 83.5, vis0-vis3 98.2, vis1-vis2 81.3, vis1-vis3 116.8, vis2-vis3 36.5.

## Task-specific exceptions (frozen; applied identically to every method)

| task | group | common-rig camera | replaced by | separation from replaced | why: common camera on that task, every frame (selection used every 5th) |
|---|---|---|---|---|---|
| door-open | vis1 | (25, 200) | (25, 120) | 71 deg | object hidden 0% (0%), gripper hidden 28% (27%), goal hidden 7% (7%), score 0.73 (0.74) |
| door-open | vis2 | (40, 300) | (50, 80) | 83 deg | object hidden 23% (24%), gripper hidden 23% (22%), goal hidden 0% (0%), score 0.62 (0.62) |
| window-open | vis3 | (25, 340) | (40, 260) | 67 deg | object hidden 0% (0%), gripper hidden 13% (13%), goal hidden 49% (49%), score 0.55 (0.56) |

## Visibility (every frame of 10 expert + 5 random episodes per task, 12 tasks)

visibility = mean visible share of the arm-free footprint (goal: of 5 goal-location points); hidden = share of frames with nothing visible; arm = mean share hidden by the robot's proximal links; min / max over tasks. score = mean over object, gripper, goal of (visibility - hidden).

| camera | set | worst-task score | mean score | obj vis (min) | obj hidden (max) | obj arm (mean / max) | gripper vis (min) | gripper hidden (max) | gripper arm (mean / max) | goal vis (min) | goal hidden (max) | workspace visible (mean) | within training envelope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vis0 | v2 held-out (as evaluated) | 0.817 (sweep-into) | 0.888 | 0.803 | 2.9% | 10.1% / 19.7% | 0.698 | 0.6% | 14.2% / 30.2% | 0.850 | 8.8% | 0.795 | yes (confirmed) |
| vis1 | v2 held-out (as evaluated) | 0.908 (bin-picking) | 0.956 | 0.954 | 1.4% | 1.7% / 4.6% | 0.745 | 1.5% | 10.4% / 25.5% | 0.960 | 3.1% | 0.811 | yes (confirmed) |
| vis2 | v2 held-out (as evaluated) | 0.661 (door-open) | 0.885 | 0.751 | 11.4% | 6.9% / 23.5% | 0.655 | 1.6% | 14.2% / 34.5% | 0.620 | 32.2% | 0.779 | yes (confirmed) |
| vis3 | v2 held-out (as evaluated) | 0.695 (peg-unplug-side) | 0.914 | 0.800 | 17.8% | 3.4% / 16.8% | 0.707 | 1.1% | 7.7% / 29.3% | 0.652 | 34.8% | 0.793 | yes (confirmed) |
| vis1 | v2 common camera on all tasks | 0.730 (door-open) | 0.940 | 0.954 | 1.4% | 1.7% / 4.6% | 0.623 | 28.3% | 11.8% / 37.7% | 0.926 | 7.4% | 0.793 | no (exceptions) |
| vis2 | v2 common camera on all tasks | 0.618 (door-open) | 0.882 | 0.751 | 22.7% | 5.6% / 23.5% | 0.554 | 23.1% | 16.7% / 44.6% | 0.806 | 19.4% | 0.752 | no (exceptions) |
| vis3 | v2 common camera on all tasks | 0.554 (window-open) | 0.896 | 0.800 | 17.8% | 3.4% / 16.8% | 0.740 | 13.5% | 7.1% / 26.0% | 0.508 | 49.2% | 0.805 | no (exceptions) |
| train0 | training | 0.616 (stick-push) | 0.875 | 0.737 | 24.9% | 3.6% / 26.3% | 0.632 | 4.4% | 11.0% / 36.8% | 0.577 | 36.5% | 0.854 | no |
| train1 | training | 0.717 (door-open) | 0.932 | 0.954 | 0.9% | 1.8% / 4.6% | 0.581 | 9.2% | 10.1% / 41.9% | 0.798 | 14.1% | 0.834 | yes |
| train2 | training | 0.780 (hammer) | 0.918 | 0.892 | 8.6% | 4.0% / 10.5% | 0.825 | 0.4% | 7.6% / 17.5% | 0.751 | 23.2% | 0.814 | yes |
| train3 | training | 0.744 (door-open) | 0.897 | 0.862 | 9.7% | 3.2% / 9.6% | 0.521 | 3.0% | 17.8% / 47.9% | 0.794 | 19.2% | 0.783 | yes |
| train4 | training | 0.723 (window-open) | 0.921 | 0.846 | 6.9% | 3.6% / 15.4% | 0.609 | 2.8% | 9.0% / 39.1% | 0.622 | 37.8% | 0.807 | yes |
| train5 | training | 0.923 (window-open) | 0.964 | 0.889 | 0.6% | 3.2% / 11.1% | 0.847 | 0.0% | 6.6% / 15.3% | 0.982 | 1.6% | 0.831 | yes |
| eval0 | v1 held-out (stress test) | -0.380 (bin-picking) | 0.178 | 0.165 | 66.0% | 57.3% / 83.5% | 0.304 | 49.8% | 51.7% / 69.6% | 0.151 | 77.1% | 0.540 | no |
| eval1 | v1 held-out (stress test) | 0.531 (stick-push) | 0.901 | 0.818 | 1.7% | 5.1% / 18.2% | 0.753 | 2.0% | 10.1% / 24.7% | 0.333 | 66.7% | 0.860 | no |
| eval2 | v1 held-out (stress test) | 0.257 (window-open) | 0.809 | 0.677 | 26.0% | 5.8% / 32.3% | 0.494 | 14.2% | 19.5% / 50.6% | 0.210 | 79.0% | 0.783 | no |
| eval3 | v1 held-out (stress test) | 0.256 (coffee-push) | 0.809 | 0.470 | 47.7% | 5.0% / 42.5% | 0.601 | 34.2% | 8.4% / 36.4% | 0.000 | 100.0% | 0.758 | no |

eval0..3 against the same envelope (tasks outside it):

- eval0: hammer (ee 0.26, score 0.51); bin-picking (obj 0.66, ee 0.48, goal 0.73, score -0.38); door-open (ee 0.30, score 0.40); peg-insert-side (ee 0.26, score 0.33); peg-unplug-side (ee 0.14, goal 0.44, score 0.14); pick-place (obj 0.48, ee 0.24, score 0.24); stick-push (ee 0.31); coffee-push (obj 0.52, goal 0.68, score -0.13); drawer-open (ee 0.50, goal 0.77, score -0.23); window-open (ee 0.27, score 0.34); soccer (obj 0.48, ee 0.31, goal 0.48, score 0.02); sweep-into (obj 0.37, ee 0.19, goal 0.45, score 0.22)
- eval1: stick-push (goal 0.67, score 0.53)
- eval2: coffee-push (goal 0.47, score 0.49); window-open (ee 0.14, goal 0.79, score 0.26)
- eval3: stick-push (goal 1.00, score 0.30); coffee-push (obj 0.48, goal 0.47, score 0.26); window-open (ee 0.34, score 0.53)

Per-task scores (as evaluated for vis; reference cameras for comparison):

| camera | hammer | bin-picking | door-open | peg-insert-side | peg-unplug-side | pick-place | stick-push | coffee-push | drawer-open | window-open | soccer | sweep-into |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vis0 | 0.98 | 0.88 | 0.84 | 0.85 | 0.90 | 0.88 | 0.93 | 0.94 | 0.95 | 0.83 | 0.87 | 0.82 |
| vis1 | 0.99 | 0.91 | 0.93 | 0.92 | 0.98 | 0.99 | 0.93 | 0.99 | 0.96 | 0.95 | 0.95 | 0.97 |
| vis2 | 0.96 | 0.89 | 0.66 | 0.92 | 0.98 | 0.99 | 0.84 | 0.89 | 0.77 | 0.89 | 0.91 | 0.94 |
| vis3 | 0.99 | 0.94 | 0.99 | 0.79 | 0.69 | 0.99 | 0.91 | 0.99 | 0.98 | 0.77 | 0.97 | 0.97 |
| train0 | 0.94 | 0.94 | 0.85 | 0.71 | 0.88 | 0.99 | 0.62 | 1.00 | 0.92 | 0.83 | 0.87 | 0.97 |
| train1 | 0.99 | 0.99 | 0.72 | 0.85 | 0.93 | 0.99 | 0.89 | 0.99 | 0.92 | 0.96 | 0.96 | 0.98 |
| train2 | 0.78 | 0.93 | 0.94 | 0.94 | 0.95 | 0.98 | 0.79 | 0.94 | 0.95 | 0.98 | 0.90 | 0.95 |
| train3 | 0.99 | 0.83 | 0.74 | 0.82 | 0.95 | 0.92 | 0.95 | 0.92 | 0.90 | 0.95 | 0.91 | 0.88 |
| train4 | 0.95 | 0.94 | 0.96 | 0.82 | 0.93 | 0.98 | 0.93 | 0.98 | 0.97 | 0.72 | 0.91 | 0.96 |
| train5 | 1.00 | 0.94 | 0.94 | 0.97 | 0.98 | 0.98 | 0.97 | 0.97 | 0.96 | 0.92 | 0.97 | 0.97 |
| eval0 | 0.51 | -0.38 | 0.40 | 0.33 | 0.14 | 0.24 | 0.67 | -0.13 | -0.23 | 0.34 | 0.02 | 0.22 |
| eval1 | 0.99 | 0.90 | 0.91 | 0.94 | 0.98 | 0.90 | 0.53 | 0.96 | 0.95 | 0.87 | 0.94 | 0.94 |
| eval2 | 0.98 | 0.86 | 0.89 | 0.90 | 0.97 | 0.98 | 0.81 | 0.49 | 0.83 | 0.26 | 0.88 | 0.85 |
| eval3 | 0.99 | 0.96 | 0.92 | 0.94 | 0.98 | 0.98 | 0.30 | 0.26 | 0.94 | 0.53 | 0.93 | 0.98 |

## Moving paths (train1 base, 72-pose period, starting phase pose 0 at reset)

Increasing the amplitude at a fixed period increases the camera speed too: the 2x/3x levels are combined range-and-speed stress tests. Episode: 126 renders (reset + 125 agent steps of 2 simulator steps, 12.5 ms each); pose = render index mod 72 (1.75 periods per episode).

| path | amplitude | azimuth range | tilt range | distance range | max / mean translation per agent step | max / mean rotation per agent step | per sim step (max) | max speed | validation |
|---|---|---|---|---|---|---|---|---|---|
| traj_lateral | +/-0.12 m | 170.5..189.5 | 44.6..45.0 | 1.000..1.007 | 6.8 / 6.6 mm | 0.55 / 0.53 deg | 3.4 mm, 0.27 deg | 0.27 m/s, 22 deg/s | pass |
| traj_circular | +/-10 deg az, +/-6 deg tilt | 170.0..190.0 | 39.0..51.0 | 1.000..1.000 | 11.8 / 9.9 mm | 0.87 / 0.71 deg | 5.9 mm, 0.44 deg | 0.47 m/s, 35 deg/s | pass |
| traj_lateral_2x | +/-0.24 m | 161.5..198.5 | 43.5..45.0 | 1.000..1.028 | 13.5 / 13.1 mm | 1.10 / 1.03 deg | 6.8 mm, 0.55 deg | 0.54 m/s, 44 deg/s | pass |
| traj_circular_2x | +/-20 deg az, +/-12 deg tilt | 160.0..200.0 | 33.0..57.0 | 1.000..1.000 | 25.5 / 19.8 mm | 1.74 / 1.42 deg | 12.8 mm, 0.87 deg | 1.02 m/s, 70 deg/s | pass |
| traj_lateral_3x | +/-0.36 m | 153.3..206.7 | 41.8..45.0 | 1.000..1.061 | 20.3 / 19.7 mm | 1.64 / 1.50 deg | 10.1 mm, 0.82 deg | 0.81 m/s, 66 deg/s | pass |
| traj_circular_3x | +/-30 deg az, +/-18 deg tilt | 150.0..210.0 | 27.0..63.0 | 1.000..1.000 | 40.7 / 29.4 mm | 2.62 / 2.13 deg | 20.3 mm, 1.31 deg | 1.63 m/s, 105 deg/s | pass |

Hidden-run diagnostics (reported, not a gate after A3): per level and target, the longest run of consecutive renders with the target fully hidden (seconds at 25 ms per render) and its task, and the largest hidden-frame share above train1 over tasks.

| path | object: longest run (task) | gripper: longest run (task) | goal: longest run (task) | max hidden share above train1 (obj / gripper / goal) |
|---|---|---|---|---|
| traj_lateral | 3 (0.08 s, soccer; train1 4) | 10 (0.25 s, door-open; train1 17) | 20 (0.50 s, peg-insert-side; train1 21) | +0.1% / +0.3% / +0.7% |
| traj_circular | 4 (0.10 s, coffee-push; train1 0) | 14 (0.35 s, door-open; train1 17) | 30 (0.75 s, peg-unplug-side; train1 22) | +0.2% / +1.5% / +1.1% |
| traj_lateral_2x | 3 (0.08 s, coffee-push; train1 0) | 24 (0.60 s, door-open; train1 17) | 22 (0.55 s, door-open; train1 27) | +0.4% / +2.6% / +0.6% |
| traj_circular_2x | 13 (0.33 s, soccer; train1 4) | 25 (0.62 s, door-open; train1 17) | 28 (0.70 s, peg-unplug-side; train1 22) | +0.6% / +3.5% / +1.2% |
| traj_lateral_3x | 15 (0.38 s, coffee-push; train1 0) | 25 (0.62 s, door-open; train1 17) | 25 (0.62 s, door-open; train1 27) | +0.8% / +3.6% / +0.6% |
| traj_circular_3x | 13 (0.33 s, soccer; train1 4) | 19 (0.48 s, door-open; train1 17) | 15 (0.38 s, peg-unplug-side; train1 22) | +1.8% / +3.2% / +1.6% |

Lateral paths repeat their first pose at the loop boundary (the reference construction `linspace(0, 4, 72)` ends where it starts): a one-step pause, never a jump. Circular paths step evenly across the boundary.

## Evaluation

`scripts/eval_rl.py --suites v2` writes `eval_cameras_v2.jsonl` (never `eval.jsonl`). Groups: `ref_train0..5` and `ref_traj_{lateral,circular}` (matched references, `--v2-reference`), `vis0..3` (with the task exceptions above), `traj_{lateral,circular}_{2x,3x}`; 20 episodes each; paired seeds `default_rng([run_seed, eval_index, 7, 20261029])` shared by every v2 group, so every method with the same run seed sees identical initial conditions and camera paths. `--snapshots last5` re-evaluates 960k..1M of a finished run (final = 1M). Multi-view methods: a vis camera replaces its nearest training camera's slot; trajectories replace train1's slot; per-frame K and c2w come from the rendered pose.

