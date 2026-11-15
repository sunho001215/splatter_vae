Temporary user authorization: 2026-11-15 ~20:00 KST, 16 MIGs on GPUs 4-7 and 400 decimal GB.

The host-local, git-ignored `runs/setup/local-memory-budget.json` is the single RAM control:
`{"ceiling_bytes": 400000000000}`. Only an integer in [150000000000, 400000000000], with exactly
that key, is accepted. Missing, unreadable or invalid data means 150 GB. No environment override exists.
Training admission subtracts 10 GB transient overhead and 2 GB staging: 388 GB when applied,
138 GB after rollback. Host MemAvailable must retain 60 GiB in both modes. Clean-cache accounting,
validation reservations, queue-specific preprocessing caps and other resource checks still apply.
The watchdog alert follows the current ceiling minus 3 GB (397 / 147 GB).

From this candidate, publish and adopt (coordinator only):

```bash
worktree=/home/ws/ws/splatter4d/.worktrees/ram400
python_bin=/home/ws/ws/splatter4d/.venv/bin/python
candidate=$(git -C "$worktree" rev-parse HEAD)
base=fe462a15b16552600610ca83cda54a4bb4d0a162
git -C "$worktree" push origin "$candidate:main"
"$python_bin" "$worktree/runs/setup/tools/merge_adopt_local_mig16_20261115.py" "$candidate" "$base" "$worktree"
```

The adoption helper requires complete passing native evidence in this worktree's `docs/tests.json`,
no failures/errors/skips, unchanged exact source fingerprints, authorized suite GPU, origin/main at the
candidate, and adopted main still at the base. It leaves the live queue and remote holds untouched.
If main advances, integrate it in this candidate and rerun the full suite.

After adoption, one command applies 400 GB plus all sixteen MIG limits:

```bash
/home/ws/ws/splatter4d/.venv/bin/python /home/ws/ws/splatter4d/runs/setup/tools/apply_local_mig16_20261115.py --root /home/ws/ws/splatter4d --apply-ram400
```

One command restores **150 GB + GPUs 4/5 (eight MIG limits)**, without a code release:

```bash
/home/ws/ws/splatter4d/.venv/bin/python /home/ws/ws/splatter4d/runs/setup/tools/apply_local_mig16_20261115.py --root /home/ws/ws/splatter4d --rollback
# Equivalent entry point:
bash /home/ws/ws/splatter4d/runs/setup/tools/rollback_local_mig16_20261115.sh /home/ws/ws/splatter4d
```

Rollback refuses and lists reserving/running/launching jobs or visible CUDA processes on GPU 6/7 MIGs.
Allow them to finish before retrying; no job is signalled or migrated. Both modes take stage2-driver.lock
then experiments/registry.lock. They validate before writing, preserve job rows and remote fields, and use
fsync plus atomic replacement; queue changes retain timestamped backups. Rollback lowers RAM before removing
limits; apply adds limits before raising RAM. If interrupted, rerun the same idempotent command.
Without mutation flags the tool previews adding limits without writing queue/control files (it takes locks).
Legacy `--apply` still adds limits only; use `--apply-ram400` to set the RAM policy too.
The control is deliberately not tracked: adoption cannot overwrite an operator's rollback or dirty source evidence.
Drivers read after acquiring admission locks; long-lived resource consumers reread on access. Local preprocessing
also intersects devices with effective local queue limits. No MIG topology or GPU authorization file changes.

Execution identity stays strict: the actual training import closure excludes memory_budget, memory_watch,
realworld_local, rwprep_mig and this JSON. No fingerprint exemption was added. `tests/test_memory_admission.py`
compares execution scope to fe462a1, proves placement against that pin, and checks all sixteen device admissions.
