# Coordinator commands — preparation only

No command below was used to push, adopt, publish a hold, register, place or launch
work during implementation. Source base is main `ee85ad9`. The candidate branch is
`rwv3-matrix-20261115`; evidence is the short detached worktree `/home/ws/ws/rwm`.
See VERIFICATION.md for final SHAs and the complete suite result.

## Preview before adoption

```bash
export M=/home/ws/ws/splatter4d
export W=$M/.worktrees/rwv3
export PY=$M/.venv/bin/python
export RWV3_CANDIDATE=$(git -C "$W" rev-parse HEAD)
export RWV3_EXPECTED_MAIN=$(git -C "$M" rev-parse ee85ad9)
export RWV3_IMAGE=$("$PY" -c 'import json; print(json.load(open("/home/ws/ws/splatter4d/runs/setup/role-release.json"))["image"])')
cd "$W"
CUDA_VISIBLE_DEVICES= "$PY" runs/setup/tools/register_ours_matrix_20261115.py \
 --repo "$M" --source-root "$W" --commit "$RWV3_CANDIDATE" --image "$RWV3_IMAGE" \
 > /tmp/ours-matrix-preview.json
```

Expect 84 matrix additions, plus one PNP continuation evaluator if still missing;
zero modifications/removals, `registered=false`, four verified local input closures,
seven deferred dependency IDs. If another worker has already added the task-aware
PNP evaluator, it is audited/reused and no second evaluator is emitted. The preview
uses live queue/registry snapshots and will become stale as normal work progresses.

## After the normal release workflow is authorized

The existing adoption helper requires the candidate already pushed under the
normal coordinator release procedure; no push is performed by this candidate.
It rejects changed main, dirty evidence outside docs, mismatched fingerprints or
incomplete tests. If main advances in tested sources, integrate and rerun the
full suite before proceeding; do not force the old expected-main pin.

```bash
cd "$M"
CUDA_VISIBLE_DEVICES= "$PY" -I "$M/runs/setup/tools/merge_adopt_local_mig_20261112.py" \
 "$RWV3_CANDIDATE" "$RWV3_EXPECTED_MAIN" /home/ws/ws/rwm --dry-run
# Coordinator's adoption step, after the reviewed release prerequisites:
CUDA_VISIBLE_DEVICES= "$PY" -I "$M/runs/setup/tools/merge_adopt_local_mig_20261112.py" \
 "$RWV3_CANDIDATE" "$RWV3_EXPECTED_MAIN" /home/ws/ws/rwm
```

## Register, then place exact IDs

After adoption, replan from main so the release pin uses main's accepted full-suite
evidence. The first Python block installs only this plan's exact-ID hold, under
the reviewed locks. It refuses an existing same-name hold rather than replacing
one. Preserve every other hold, especially discontinued tasks, old 200k-only rows,
remote boot/outage controls and any preparation priority hold.

```bash
cd "$M"
export RWV3_RELEASE=$(git rev-parse HEAD)
CUDA_VISIBLE_DEVICES= "$PY" runs/setup/tools/register_ours_matrix_20261115.py \
 --commit "$RWV3_RELEASE" --image "$RWV3_IMAGE" > /tmp/ours-matrix-register-plan.json
CUDA_VISIBLE_DEVICES= "$PY" - <<'PYCODE'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'runs/setup/tools'))
import register_ours_matrix_20261115 as reg
record=json.loads(Path('/tmp/ours-matrix-register-plan.json').read_text())
with reg.R.locks(reg.MAIN):
    path=reg.MAIN/'runs/setup/admission-pauses.json'
    doc=json.loads(path.read_bytes())
    reg.R.require(not any(h.get('name')==reg.LOCAL_PAUSE for h in doc['pauses']),
                  'existing matrix hold requires reconciliation')
    reg.R.require(reg.R.file_sha(reg.MAIN/'experiments/queue.yaml')==record['queue_sha256_before'],
                  'queue changed; replan')
    doc['pauses'].append(record['holds'][0])
    reg.R.atomic_write(path,reg.R.json_bytes(doc))
PYCODE
CUDA_VISIBLE_DEVICES= "$PY" runs/setup/tools/register_ours_matrix_20261115.py \
 --commit "$RWV3_RELEASE" --image "$RWV3_IMAGE" --execute \
 > /tmp/ours-matrix-registered.json

# Read-only first; an active exact-ID hold is allowed only for placement inspection.
CUDA_VISIBLE_DEVICES= "$PY" runs/setup/tools/place_ours_matrix_loop_20261115.py
# Coordinator placement loop; does not launch trainers. Existing scheduler/drivers do that.
CUDA_VISIBLE_DEVICES= "$PY" runs/setup/tools/place_ours_matrix_loop_20261115.py --execute --loop 300
```

The placement loop backfills independent chains, repeats normal dependency and
capacity checks, recovers placement-before-lift crashes and releases only IDs with
verified local ownership. Its default is one read-only pass. A supervisor owner
may use the exact final command above as a separate new finite service, with the
existing singleton/command-identity checks. Do not replace or restart live
controllers. It exits when all rows are placed or when
`$M/runs/setup/STOP-OURS-MATRIX-PLACE` exists; it never signals a trainer. Do not
remove an existing STOP marker automatically. GPU capacity is always checked by
the official placement/admission code; no running job is preempted.

The first four IDs, available immediately subject to normal capacity, are:

```text
rwv3-20261115-pre-ours-covering-S-orig-scratch400k
rwv3-20261115-pre-ours-covering-S-ch01-scratch400k
rwv3-20261115-pre-ours-pick-and-place-S-orig-scratch400k
rwv3-20261115-pre-ours-pick-and-place-S-ch01-scratch400k
```

## Dsreset publication integration

The pipeline worker retains its accepted preprocessing/calibration/QA/input
publication workflow and independent local phase-A qualification. Phase A alone
never releases learning. Before its final dsreset registration, produce a NEW
spec with `ours_input_barriers_only=true`, keeping the existing `reuse_ours_S=true`
proofs for covering/PNP and all other method selections:

```bash
export P=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline
"$PY" - <<'PYCODE'
import json,os
from pathlib import Path
p=Path(os.environ['P'])
src=p/'registration-spec.json'
out=p/'registration-spec-matrix.json'
doc=json.loads(src.read_bytes());doc['ours_input_barriers_only']=True
with out.open('x') as stream:json.dump(doc,stream,indent=2)
PYCODE
# Only after actual accepted paired inputs, raw proof, QA and reuse proofs exist:
CUDA_VISIBLE_DEVICES= "$PY" runs/setup/tools/register_dsreset_20261113.py \
 --spec "$P/registration-spec-matrix.json" --ingest-manifest "$P/ingest-snapshot/manifest.json" \
 --commit "$RWV3_RELEASE" --image "$RWV3_IMAGE" --release-source local \
 > /tmp/dsreset-matrix-publication-plan.json
# The pipeline coordinator then uses its existing dsreset hold/register/mirror/place/closeout route
# with THIS spec, adding --execute only to its reviewed final registration command.
```

The new boolean mode suppresses all NEW legacy Ours pre/pre400k/export/fqldata/FQL/
eval rows while retaining shared prep/calib/qa, per-variant Ours train-only stats
and every selected other method. With the standard full spec and the two S reuse
proofs, expect **415** additions instead of the obsolete 461. The matrix's missing
IDs are exactly the three QA/publication barriers and four new stats producers.
They become ready on normal publication/closeout and are picked up by the same
placement loop. No fake completion events or speculative data hashes are needed.
If a legacy dsreset graph was registered meanwhile, reconcile its unlaunched Ours
rows under the newer directive before enabling it; never stop healthy continuations
or edit placed scopes. Do not run both old and matrix publication registrations.

## Deployment handoff

The published matrix proof is
`$M/runs/setup/rw-registration-ours-matrix-20261115.json`. The Robot PC worker
should read each FQL row's `deployment_hook` and the proof's
`continuation_audit.<task>.policy_hooks` (36+6 policies), wait for FQL and the linked
evaluation's normal closeouts, validate policy hashes and deduplicate by complete
experiment identity before its established export/transfer/registration workflow.
No deployment, inference reload/restart or physical robot run is performed here.
