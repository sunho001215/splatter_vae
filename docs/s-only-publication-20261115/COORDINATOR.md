S-only publication for covering and pick-and-place: 140 NEW queue rows (70 each),
all eight baseline conditions, 48 FQL policies and 48 artifact evaluations.
Ours scratch/continuation chains are unchanged; existing S statistics are reused
only after the existing raw/split/calibration/recipe proof. Shared readiness
barriers now read S. No new SF identity can be executed by this registrar.

Both fresh reports were measured on S TRAIN only: covering 88 and pick-and-place
87 episodes, eight evenly spaced frames/camera, native RealSense alignment and
accepted sensor-depth correction, ≥99% robot/object/fixture containment. Covering
replayed the measurement frames from hash-verified local recordings; pick-and-place
read the verified existing native overlays. Neither report reuses SF statistics.

43 targeted tests passed. The isolated local mirror smoke passed full readiness
checks for all families on covering 98/98 and pick-and-place 97/97 episodes.
`registration-dry-run.json` is a validated dry run; `drawer-preview.json` is ONLY
a deferred graph preview. `smoke.json` records an isolated mirror that was removed
after validation; its hashes describe the verified smoke, not a live publication.

Drawer's canonical 98-success manifest currently fails the live ingest exclusion
identity gate. None of its new learning rows has complete accepted local inputs.
Its 98 S hybrid phase-A rows are already registered; their further audit belongs
to the parallel nosam worktree. The deferred 71-row S graph retains the shared
prep → calibration → QA barriers and its new S-only stats dependency. Do not
execute its preview or invent acceptance evidence to register consumers early.

The adoption helper explicitly requires a passing complete suite for the candidate.
The full suite runs in the background at the end; do not wait on it here. Inspect
`background-suite.log` and `docs/tests.json`. Adoption refuses until that report
has `full_suite_passed=true`, no failures/errors/skips, and exact source hashes.
No push, adoption, staging into main, queue/pause/ownership mutation, placement,
training launch, or network action was performed by the implementer.

Run these commands as coordinator, in order. All paths are local except `push`;
only the coordinator executes that network command. Registrations are revalidated
against the then-current queue/registry/ingest under the existing integrity gates.
The offline stage command publishes new input files into the local mirror namespace;
it does not upload anything or claim the generated inputs exist remotely.

```bash
M=/home/ws/ws/splatter4d
W=$M/.worktrees/dsrs
A=$W/docs/s-only-publication-20261115
PY=$M/.venv/bin/python
C=$(git -C "$W" rev-parse HEAD)
BASE=20045e93100dea2fe30f8b909a1cd16180d6c987
IMAGE=sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a
INGEST=/home/ws/ws/realworld-ops/ingest/manifest.json
cd "$M"
git -C "$W" push origin "$C":refs/heads/dsrs-s-only-20261115
"$PY" -I runs/setup/tools/merge_adopt_local_mig16_20261115.py "$C" "$BASE" "$W" --dry-run
"$PY" -I runs/setup/tools/merge_adopt_local_mig16_20261115.py "$C" "$BASE" "$W"

# Read-only stage preview, then append-only offline input publication.
"$PY" -I -B runs/setup/tools/stage_dsreset_s_inputs_20261115.py --spec "$A/spec.json" --ingest-manifest "$INGEST"
"$PY" -I -B runs/setup/tools/stage_dsreset_s_inputs_20261115.py --spec "$A/spec.json" --ingest-manifest "$INGEST" --execute
# Validated dry run, then exact-ID held publication (pause installed before queue).
"$PY" -I -B runs/setup/tools/register_dsreset_20261113.py --spec "$A/spec.json" --ingest-manifest "$INGEST" --commit "$C" --image "$IMAGE" > /tmp/dsrs-registration-current.json
"$PY" -I -B runs/setup/tools/register_dsreset_20261113.py --spec "$A/spec.json" --ingest-manifest "$INGEST" --commit "$C" --image "$IMAGE" --execute

# IDs in dependency/priority order; exactly the new S publication, never an SF loop.
mapfile -t IDS < "$A/ids.txt"
"$PY" -I -B runs/setup/tools/place_dsreset_s_20261115.py "${IDS[@]}"
"$PY" -I -B runs/setup/tools/place_dsreset_s_20261115.py "${IDS[@]}" --execute
# Repeat closeout + exact-ID placement as producers finish. The helper calls
# realworld_local.apply and lifts only placement holds with matching local ownership/pins.
"$PY" -I -B ops/jobs/realworld_local.py closeout --all
"$PY" -I -B runs/setup/tools/place_dsreset_s_20261115.py "${IDS[@]}" --execute
```

The existing drivers admit placed rows under normal priority and capacity rules:
preprocessing/stats -4, Ours pretraining -3, Ours downstream -1, baseline
pretraining 0, baseline FQL/evaluation 1. Independent baseline pretrains can begin
as soon as the three small readiness checks close out; 3D-MVP also waits for its
native overlay producer. Descendants stay held until their inputs exist.

If main has moved from BASE, do not rewrite that guard: integrate the new main
and produce candidate evidence again. A pending registration journal requires
inspection before retrying; holds remain in force on an interrupted publication.
