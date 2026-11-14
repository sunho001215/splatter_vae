Run these commands from the adopted main checkout **after its full release suite
has produced current, passing `docs/tests.json`**. The adopted HEAD must include
the coordinator, generator changes and placement helper. `--commit` is that HEAD,
not the earlier `2fa4ea1` on which this change was prepared.

```bash
cd /home/ws/ws/splatter4d
TASK_EVAL_COMMIT=$(git rev-parse HEAD)
TASK_EVAL_IMAGE=$(.venv/bin/python -c 'import json; print(json.load(open("runs/setup/role-release.json"))["image"])')

# Read-only preview. Review rows, holds, skipped IDs and queue_diff in stdout.
.venv/bin/python runs/setup/tools/register_ours_task_eval_20261115.py \
  --commit "$TASK_EVAL_COMMIT" --image "$TASK_EVAL_IMAGE" --release-source local

# Coordinator publication: installs holds first, then appends the queue rows.
.venv/bin/python runs/setup/tools/register_ours_task_eval_20261115.py \
  --commit "$TASK_EVAL_COMMIT" --image "$TASK_EVAL_IMAGE" --release-source local --execute
```

Covering S always becomes `rwv2-20261113-eval400k-ours-covering-S-v2`.
Every other registered, never-attempted non-drawer old-script row becomes its
original ID plus `-v2`. This includes pick-and-place S **if that evaluation row is
registered**. Its checked-in 400k spec currently has `include_eval: false`; the
coordinator does not invent an evaluation row for an absent source. The scan
also handles other campaign IDs and records attempted rows as skipped.

Existing repin tools preserve script and argv. The coordinator therefore never
edits old rows: failed covering S stays failed; never-launched originals enter
the permanent `ours-task-eval-superseded-20261115` exact-ID hold. New rows enter
`ours400k-await-local-placement-20261113`, with any previously held IDs retained.
Other pauses are preserved. The registry is never written. A pending journal
after interruption requires inspection; the installed holds remain active.

The dry-run record includes `placement_command`, containing **all** replacement
IDs selected from the queue. Run that exact command to preview placement, then
append `--execute` to place them and release only the moved IDs. For covering S
and a registered, never-launched pick-and-place S row, the commands are:

```bash
.venv/bin/python runs/setup/tools/place_ours400k_20261113.py \
  rwv2-20261113-eval400k-ours-covering-S-v2 \
  rwv2-20261113-eval400k-ours-pick-and-place-S-v2

.venv/bin/python runs/setup/tools/place_ours400k_20261113.py \
  rwv2-20261113-eval400k-ours-covering-S-v2 \
  rwv2-20261113-eval400k-ours-pick-and-place-S-v2 --execute
```

Use only IDs present in the plan. Dependencies that are still pending keep their
replacement held until explicit placement succeeds. The placement helper never
lifts the superseded-original hold. Publication and placement do not launch jobs.

The standalone 400k generator now emits `evaluate_ours_task_length.py`. The
dsreset registrar also upgrades any future non-drawer old-script length rows at
its generator boundary; its current paired graph emits policy artifact
evaluations and has no Ours length-evaluation stage to replace.

Offline regression command (from the candidate worktree, using its Python environment):

```bash
CUDA_VISIBLE_DEVICES=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd .venv/bin/python -m pytest -q \
  tests/test_ours_task_eval_registration.py \
  runs/setup/tools/test_register_ours400k_20261113.py \
  tests/test_dsreset_registration.py tests/test_ours_length_eval.py tests/test_realworld_local.py
```
