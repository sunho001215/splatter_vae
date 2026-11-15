# Drawer S identity repair, 2026-11-16

The failure is missing ingest identity, not a stale amendment, an SF failure-set
hash, or renumbering. `s4d/rwlearn/dsreset.py:338` matches each exclusion on task,
success directory label, filename and optional SHA, then requires exactly one
record with validation status `invalid` at line 349. The live catalog contains
**zero drawer-open-v2 records**. Episode 77 therefore fails first with zero
matches; 79 would fail identically. The active 98 hashes would also be absent.
The live worker's fixed task list omits drawer-v2
(`runs/setup/tools/realworld_ingest_20261109b/ingest_worker.py:53`), and its
discovery only scans files immediately inside each task directory at line 534.
Drawer-v2 has success/failure subdirectories. Registering hybrid jobs does not
register ingest records.

`smoke.json` binds the inspected live/catalog bytes and canonical S manifest.
The manifest amendment is exactly the released JSON, SHA256
`041042f1ff98d220b20dfd5ee159ed113eb79ad29a357134f246e8e5514ccbe7`.
Fresh reads reproduce the documented 77 missing-schema and 79 nonmonotonic-clock
rejections with the exact excluded hashes/sizes. S has 98 successes, split 88/10.
The exclusion hashes bind two rejected **success** files; neither failure membership
nor an SF manifest hash enters this check. No new membership decision is needed.

The repair is a new S snapshot, used consistently by learning-input generation,
offline staging and registration. `scripts/realworld_dsreset_snapshot.py:19`
now accepts `--success-only` and requires the released amendment and exact S raw
membership before reading dispatchers. It skips unselected dispatcher jobs,
rehashes raw files, preserves independent remote-verification requirements,
requires conclusive exclusion rejection, detects changing raw/manifests, and
binds provenance to the actual source/proof bytes read. It preserves the live
catalog. Missing prep, calibration, QA or measurement still blocks registration.
The gate's only change is a diagnostic naming the missing exclusion and match count.

38 relevant CPU tests pass, plus the actual read-only exclusion smoke and row-ID
overlap smoke. The first 40-test run had two environment failures:
`test_every_persisted_placement_scope_has_zero_candidate_changes` and
`test_every_live_or_placed_scope_has_same_differences_as_main`; their differences
are missing dino/dynamo/rvt/snerl submodule checkout identities. The normal pytest
conftest requires a visible GPU, unavailable here, so the CPU tests used
`CUDA_VISIBLE_DEVICES='' python -B -m pytest --noconftest`; no GPU mapping was
invented. `targeted.xml` records the 38-test rerun with those two deselected.
This is targeted evidence, not a passing complete-suite release claim. No full
suite, network, live writes, source adoption, queue change or raw change was run.

These commands are for the coordinator only. Source integration is local and
does not adopt a runtime release or update its pin. Registration/placement still
requires the established valid current-source release evidence; these commands
do not manufacture it or bypass that gate. No foreground full suite is requested.

```bash
set -euo pipefail
M=/home/ws/ws/splatter4d
W=$M/.worktrees/drwm
PY=$M/.venv/bin/python
C=$(git -C "$W" rev-parse HEAD)
# Apply this source fix locally; resolve any integration conflict before proceeding.
git -C "$M" cherry-pick -x "$C"
cd "$M"
export P=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline
export RUN=$P/drawer-s-20261116
MANIFEST=$P/drawer-canonical-manifests/S/manifest.json
LIVE=/home/ws/ws/realworld-ops/ingest/manifest.json
A=$M/docs/drawer-identity-20261116
RELEASE_COMMIT=$(git rev-parse HEAD)
IMAGE=sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a
```

Once the 98 hybrid closeouts exist, the independent RGB route from nosam commit
`07b0894c7eba374758b1d5e55de236272f87d422` remains applicable. Integrate that commit
locally if absent, retain the ordinary current-source release gate, then run:

```bash
git merge-base --is-ancestor 07b0894c7eba374758b1d5e55de236272f87d422 HEAD || \
  git cherry-pick -x 07b0894c7eba374758b1d5e55de236272f87d422
RELEASE_COMMIT=$(git rev-parse HEAD)
RGB=$M/runs/drawer-rgb-inputs-20261115/S
"$PY" -B scripts/realworld_rgb_inputs.py --manifest "$MANIFEST" --out "$RGB"
"$PY" -B scripts/realworld_rgb_inputs.py --manifest "$MANIFEST" --out "$RGB" --execute
RGBREG=("$PY" -B runs/setup/tools/register_drawer_rgb_20261115.py
  --manifest "$MANIFEST" --bundle "$RGB" --commit "$RELEASE_COMMIT" --image "$IMAGE")
"${RGBREG[@]}" > /tmp/drawer-rgb-dry.json
"${RGBREG[@]}" --execute > /tmp/drawer-rgb-registered.json
mapfile -t RGB_IDS < <("$PY" -B -c 'import json,sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1]))["rows"]))' /tmp/drawer-rgb-registered.json)
"$PY" -B runs/setup/tools/place_drawer_rgb_20261115.py "${RGB_IDS[@]}"
"$PY" -B runs/setup/tools/place_drawer_rgb_20261115.py "${RGB_IDS[@]}" --execute
```

This route verifies raw identities and all 98 qualified hybrid output inventories
and local closeouts before writing a new RGB bundle. It needs no geometric/SAM3
acceptance. It does not fix or populate the geometric ingest catalog. It uses the
unchanged RGB consumer gates and cannot qualify geometric consumers. If RGB
publication or those IDs already exist, inspect their existing proof and use
closeout/placement; neither registrar permits duplicate registration or repinning.

For geometric publication, wait for real S phase-B outputs and QA, the accepted
task/session rig + reference, compact files in the verified local mirror, and an
existing independent raw transfer proof. None of these currently exists as a
complete accepted drawer input set. The paths below are the established output
contract, not claims that those artifacts exist. Do not run historical SF producer
loops. Do not synthesize the proof map from local hashes or a VERIFIED marker.
If the independent proof is absent, this path stays blocked; this handoff does
not authorize network access to obtain it.

```bash
RIG=/home/ws/data/realworld-processed/dsreset-20261113/calibration/drawer-open-v2/rig-drawer-open-v2-accepted.json
test -s "$RIG"
test -s "${RIG%.json}.reference.npz"
test -s "$P/remote-raw-hashes.json"
test -s "$P/local-phaseB-drawer-open-v2/dispatcher.json"
"$PY" -B scripts/realworld_dsreset_snapshot.py --success-only \
  --source "$LIVE" --manifest "$MANIFEST" \
  --prep-config "$P/local-phaseB-drawer-open-v2/dispatcher.json" \
  --remote-hashes "$P/remote-raw-hashes.json" --out-dir "$RUN/snapshot"
INGEST=$RUN/snapshot/manifest.json
# Release writer rechecks actual calibrate build/rig/reference/fixedness evidence.
"$PY" -B -m s4d.realworld.calib.release plan --manifest "$INGEST" \
  --rig "drawer-open-v2=$RIG" --out "$RUN/calibration-plan.json"
"$PY" -B -m s4d.realworld.calib.release apply --plan "$RUN/calibration-plan.json" --state-dir "$RUN/snapshot"
"$PY" -B -m s4d.realworld.calib.release apply --plan "$RUN/calibration-plan.json" --state-dir "$RUN/snapshot" --apply
"$PY" -B scripts/realworld_dsreset_inputs.py --manifest "$MANIFEST" \
  --ingest-manifest "$INGEST" --out-dir "$RUN/learning-inputs/drawer-open-v2/S"
"$PY" -B scripts/realworld_dsreset_measure.py --inputs "$RUN/learning-inputs/drawer-open-v2/S" \
  --sample-native --rig "$RIG" --out "$RUN/measurement"
```

Both supplied specs select S, preserve the three shared barriers and Ours S stats,
and use a fresh append-only publication namespace. `spec-all-71.json` emits the
deferred 71 rows. **The 32 nosam rows are a subset of these 71 IDs**, not another
32 rows; `overlap.json` proves the disjoint 32+39 partition. If RGB was registered
first, use `spec-after-rgb-39.json` for the remaining methods. The following
read-only choice refuses partially registered RGB sets and arbitrary existing
IDs; the ordinary registrar subsequently refuses any other collision.

```bash
SPEC=$("$PY" -B - "$M" "$A" <<'PY'
import json,sys,yaml
from pathlib import Path
m,a=map(Path,sys.argv[1:])
rgb=set(json.loads((a/'overlap.json').read_text())['rgb_ids'])
rows={r['id']:r for r in yaml.safe_load((m/'experiments/queue.yaml').read_text())['jobs']}
present=rgb & rows.keys()
if present:
    assert present==rgb, 'partial RGB publication: inspect existing journal/proof'
    assert all(rows[k].get('registration_tool')=='runs/setup/tools/register_drawer_rgb_20261115.py' for k in rgb), 'existing non-RGB IDs: inspect; never repin'
print(a/('spec-after-rgb-39.json' if present else 'spec-all-71.json'))
PY
)
"$PY" -I -B runs/setup/tools/stage_dsreset_s_inputs_20261115.py --spec "$SPEC" --ingest-manifest "$INGEST"
"$PY" -I -B runs/setup/tools/stage_dsreset_s_inputs_20261115.py --spec "$SPEC" --ingest-manifest "$INGEST" --execute
RELEASE_COMMIT=$(git rev-parse HEAD)
REG=("$PY" -I -B runs/setup/tools/register_dsreset_20261113.py
  --spec "$SPEC" --ingest-manifest "$INGEST" --commit "$RELEASE_COMMIT" --image "$IMAGE" --release-source local)
"${REG[@]}" > /tmp/drawer-s-dry.json
"${REG[@]}" --execute > /tmp/drawer-s-registered.json
mapfile -t IDS < <("$PY" -B -c 'import json,sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1]))["rows"]))' /tmp/drawer-s-registered.json)
"$PY" -I -B runs/setup/tools/place_dsreset_s_20261115.py "${IDS[@]}"
"$PY" -I -B runs/setup/tools/place_dsreset_s_20261115.py "${IDS[@]}" --execute
```

If choosing all 71 at once, skip the RGB block. Never execute both overlapping
registrations. After real producers finish, run ordinary local closeout and repeat
the appropriate exact-ID placement pair. Keep cancellation holds in place. No
SF registration, ownership rewrite, queue repin, launch, deployment or network
command is included. On refusal, retain the holds and inspect the named missing
artifact/journal; do not edit acceptance fields to force a retry.
