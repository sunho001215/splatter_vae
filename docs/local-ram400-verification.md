Temporary local RAM400 control — user 2026-11-15 ~20:00 KST

Prepared from adopted main fe462a15b16552600610ca83cda54a4bb4d0a162, with no live writes or network.
The coordinator must adopt, then run `--apply-ram400`: 400,000,000,000 bytes, 388 GB training ceiling,
16 queue MIG limits on GPUs 4-7. `--rollback` restores 150,000,000,000 bytes, 138 GB training ceiling,
eight limits on GPUs 4/5. The existing shell rollback now invokes this runtime operation directly.
Exact push/adopt/apply/rollback commands: `runs/setup/tools/ROLLBACK-local-mig16-20261115.md`.

`runs/setup/local-memory-budget.json` is host-local and git-ignored so a code adoption cannot reset a rollback.
The reader accepts exactly `{"ceiling_bytes": INTEGER}` within inclusive bounds 150 GB through 400 GB;
missing, unreadable, malformed, non-integer or out-of-bounds input fails safe to 150 GB. Consumers reread on
attribute access. Driver caps are sampled after acquiring the same admission lock as apply/rollback.
Both mutation modes validate under admission then registry locks, preserve job rows/remote fields, and use
atomic replacement and fsync. Rollback refuses GPU 6/7 live jobs/processes before touching either file.
It lowers RAM before removing limits; apply adds limits before raising RAM. Repeating either command is harmless.
GPU topology and allowlist files remain unchanged; effective queue limits restrict placement/admission to eight
MIGs after rollback. Preprocessing now intersects local inventory with those same effective queue limits.

The 60 GiB host MemAvailable reserve, 10 GB transient overhead, 2 GB staging reserve, measured growth,
validation reservations, explicit preprocessing caps and verified clean-file-cache exclusion remain enforced.
The watchdog alert threshold follows the live control (397 GB at RAM400; 147 GB after rollback).

Consumer grep/review covered:
- ops/campaign/stage2_local_driver.py and baseline_local_driver.py (shared TRAINING_CEILING, validation subtraction).
- ops/jobs/realworld_local.py (_capacity) and runs/setup/tools/place_local_phase_a_20261115.py (delegates to it).
- ops/jobs/rwprep_mig.py (shared local training ceiling, explicit queue caps, effective local device limits).
- ops/resources/memory_watch.py (fits/staging/alerts) and runs/setup/tools compatibility entry points.
- Current operational docstrings and local policy/runbook documentation. Historical logs/reports retain their numbers.

Fingerprint proof: no guard exemption or fingerprint algorithm changed. Training scripts' actual static import
closure excludes memory_budget, memory_watch, realworld_local and rwprep_mig, and the runtime JSON is outside
execution identity. The regression compares combined execution scope against fe462a1, exercises the actual
placement_scope proof at that pin, and verifies all sixteen devices with local_devices.allows.
Read-only live audit: 97 placed/pinned local rows, 12 distinct pins; zero changed/missing execution files and
zero new scope mismatches. Nineteen recorded scopes fully match adopted main. Seventy-eight older scopes already
differ from main and retain their prior refusals; this change does not waive them. Evidence: local-ram400-pins.json.

Verification:
- Combined regression run: 284 passed (placement/capacity, invalid policy, staging, preprocessing, apply/rollback,
  both legacy/nested queue schemas, conflict and live-job refusal, fe462a1 pin, and adoption helper).
- Operational tool tests: 18 passed; recorded in local-ram400-tools.log / local-ram400-tools.junit.xml.
- Full native suite: 4,085 collected tests + 27 subtests = 4,112 passes; zero failures, errors or skips.
  Full-suite acceptance true; source fingerprints unchanged. Duration: 749.7 seconds.
  Authoritative evidence remains in docs/tests.json, docs/tests.log and docs/tests.junit.xml (not staged;
  the adoption helper explicitly permits docs-only evidence changes in this worktree).
  Command: CUDA_VISIBLE_DEVICES=MIG-0b9a2ef1-2f30-5730-ab39-c31e8e7242b6 HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline S4D_TEST_CPU_THREADS=32
  /home/ws/ws/splatter4d/.venv/bin/python scripts/run_tests.py --partition all --shards-cpu 4 --shards-gpu 1
- Local cached Git objects and the existing DINO checkpoint provisioned this candidate's dependencies offline.
- Before testing: counted budget 73,977,758,720 bytes; host available 403,033,714,688 bytes;
  the 50 GB suite allowance retains the 60 GiB host reserve. A later read-only measurement including
  running jobs' remaining growth projected 256,106,404,864 bytes with the suite allowance, within the
  user-authorized 400 GB temporary ceiling; host available was 397,228,945,408 bytes. See local-ram400-headroom.json.
- Zero introduced lint findings versus fe462a1; shell syntax and whitespace checks passed.
