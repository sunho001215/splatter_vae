Temporary user authorization: 2026-11-15 15:00 KST, 16 MIGs on GPUs 4-7 and 225 GB.

From the candidate worktree, one step prepares rollback:

```bash
bash runs/setup/tools/rollback_local_mig16_20261115.sh /home/ws/ws/splatter4d
```

The queue tool first takes stage2-driver.lock then experiments/registry.lock, lists and refuses any
running/launching/reserving job or visible process on a GPU 6/7 MIG, and atomically removes only the eight
new local limits with a timestamped backup. Repeated application is harmless. No job row is rewritten.
It then creates a new worktree from current main and commits `git revert` of the expansion commit there.
Conflicts stop preparation for review. It never adopts the revert, kills jobs or changes MIG topology.
Run the full suite on an original GPU 4/5 MIG in that rollback candidate and adopt through the normal
reviewed evidence path. Existing GPU 6/7 jobs must finish before rollback; no automatic migration occurs.
The expansion remains authorized until the revert is adopted. GPU topology remains a coordinator operation.

Apply after adoption (dry-run is the default):

```bash
/home/ws/ws/splatter4d/.venv/bin/python runs/setup/tools/apply_local_mig16_20261115.py --root /home/ws/ws/splatter4d
/home/ws/ws/splatter4d/.venv/bin/python runs/setup/tools/apply_local_mig16_20261115.py --root /home/ws/ws/splatter4d --apply
```

Device eligibility intersects the planned release's literal local guard/config allowlist with the actual script
checkout's allowlist and current authorization. Old pinned rows use only their original devices; new pinned rows
may use all sixteen. Reservations, fair-share, placement, scheduler selection and the final launch guard use the
same check. Pinned entry points receive their own host config, avoiding old guards rejecting a new main config.

Adoption (coordinator only, after publishing the candidate as origin/main):

```bash
candidate=$(git -C /home/ws/ws/splatter4d/.worktrees/mig16 rev-parse HEAD)
base=736d821ee0e1e8cdc46cf1c88b2b52bb97c4f197
/home/ws/ws/splatter4d/.venv/bin/python /home/ws/ws/splatter4d/.worktrees/mig16/runs/setup/tools/merge_adopt_local_mig16_20261115.py "$candidate" "$base" /home/ws/ws/splatter4d/.worktrees/mig16
```

This is the existing local adoption flow with the candidate's expanded GPU authorization and no remote queue rewrite.
The older 2026-11-12 tool checks the pre-merge main guard (eight devices), so it cannot adopt evidence from GPU 6/7.
The new tool preserves its pushed-candidate, clean-source, complete-suite, exact-fingerprint and driver-lock gates.
It leaves all queue limits untouched; run the apply command above only after adoption. If main advances, integrate
it in the candidate and rerun the full suite before adoption. The branch began at ee85ad9 and incorporated main's
four already-adopted commits through 736d821 in the isolated worktree to keep the handoff fast-forwardable.
