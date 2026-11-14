The task-aware length evaluator is `scripts/evaluate_ours_task_length.py`, backed by
`s4d/diag/ours_task_length_eval.py`. Its CLI and three result filenames match the
historical evaluator; its report protocol is `ours-task-length-v1`.

The historical `scripts/evaluate_ours_length.py` and `s4d/diag/ours_length_eval.py`
are intentionally byte-identical to main. The persisted placement of
`ourx-20261113-ours-drawer-open-eval200k` includes both files in its execution
scope. Editing either breaks the dsreset invariant. The isolated entry point
allows a future evaluation to opt into the correction without altering any live
or placed row's scope. Launch classification and strict input/output validation
recognize the new entry point with the same rules as the historical evaluator.

Compact-store locators are `task/episode`, but window metadata separates
`task_name` and `episode`. Grasp sites now use their canonical task/episode
identity. Missing observed closure edges produce an explicit unavailable site.
Gripper and future-displacement probes still run, while grasp-relative and phase
probes score their available samples independently. Unsupported probe scores are
null with explanations under `not_applicable`.

For old drawer-open, the fixed initial-grasp 35 mm ROI is retained. Other tasks
use their own `model.anchor_stats` dynamic mean +/- one standard deviation box,
projected through each calibrated camera. This is explicitly a task dynamic
workspace proxy, not a semantic handle annotation. Covering, pick-and-place,
and mug-hanging calibration stats all contain measured dynamic anchors with
`dynamic_fallback_to_scene=false`. The established SCENE_PARAMETERS supply only
rig look-at, pose scale and global bounds; those do not establish a small-detail
region. Scene fallbacks, wrong-task statistics and absent/invalid dynamic anchors
are never substituted. ROI metrics become null with a reason and zero support;
all other defined metrics continue. A projected box outside the image or behind
the camera also has no metric support.

`rwv2-20261113-eval400k-ours-covering-S` requires a future re-pin from its recorded
`f2e2f8bbc201e6f878d96a0fe13498a23b357c86` release to the candidate and a script
change to `scripts/evaluate_ours_task_length.py`. Re-pinning while retaining the
historical entry point will not apply this fix. Keep its checkpoint, calibrated
data, split, gate, workspace stats, and evaluation settings. Its exhausted retry
state will also need the usual explicit rerun preparation. This change does not
edit the queue, placements, pins, retry state or any source training run, and does
not launch a job.
