#!/usr/bin/env bash
# Prepare a rollback candidate; never adopt, stop jobs, or change topology.
set -euo pipefail
candidate_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
main_root="${1:-/home/ws/ws/splatter4d}"
python_bin=/home/ws/ws/splatter4d/.venv/bin/python
expansion_commit="$(git -C "$candidate_root" log -1 --format=%H -- ops/jobs/local_devices.py)"
"$python_bin" "$candidate_root/runs/setup/tools/apply_local_mig16_20261115.py" --root "$main_root" --rollback
rollback_tree="$main_root/.worktrees/rollback-local-mig16-$(date +%Y%m%d-%H%M%S)"
git -C "$main_root" worktree add -b "rollback/local-mig16-$(date +%Y%m%d-%H%M%S)" "$rollback_tree" HEAD
git -C "$rollback_tree" revert --no-edit "$expansion_commit"
printf 'Rollback candidate: %s\nValidate its full suite and use the normal adoption path.\n' "$rollback_tree"
