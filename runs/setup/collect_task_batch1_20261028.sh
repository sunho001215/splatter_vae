#!/bin/bash
# New-task batch 1 (user directive 2026-10-28): the established collection protocol of the eight campaign tasks
# (queue rows pilot-/collect-/split-/stats-/check-/heldout-sets-coffee-push), run directly per task on one local MIG.
# Usage: collect_task_batch1_20261028.sh <task> <MIG uuid>
set -euo pipefail
task=$1; export CUDA_VISIBLE_DEVICES=$2 MUJOCO_GL=egl
cd /home/ws/ws/splatter4d
root=/home/ws/data/metaworld/splatter4d_v1
py=".venv/bin/python -E -s"
step() { name=$1; shift; mkdir -p runs/$name-$task; echo "=== $(date '+%F %T') $name-$task: $*"; \
  $py "$@" > runs/$name-$task/console.log 2>&1; echo "=== $(date '+%F %T') $name-$task exit $?"; }
step pilot scripts/collect_metaworld.py --task $task --output $root/${task}_pilot.hdf5 --episodes 5
step collect scripts/collect_metaworld.py --task $task --output $root/$task.hdf5
step split scripts/split_metaworld.py --path $root/$task.hdf5
step stats scripts/compute_workspace_stats.py --regime metaworld --path $root/$task.hdf5 --out $root/workspace_stats/$task.json
step check scripts/check_metaworld.py --path $root/$task.hdf5 --out /home/ws/ws/splatter4d/docs/data_checks/$task
step heldout-sets scripts/render_heldout_sets.py --task $task
echo "=== $(date '+%F %T') $task batch complete"
