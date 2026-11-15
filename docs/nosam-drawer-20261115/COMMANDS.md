# Coordinator commands (dry-run first; not executed here)

Adopt the commit into `/home/ws/ws/splatter4d` through the existing local release process. Do not use an older pin, rewrite ownership, or lift `s-only-cancelled-20261115-a9594590`. These commands do no network access. Actual registration is permitted only from adopted main with the existing valid release evidence. No full suite should run in the foreground; this patch does not override the trainer/release evidence gates.

The 98 canonical hybrid jobs must complete and pass the established local closeout before RGB publication succeeds. Preflight fails closed while any output is pending. It checks the existing immutable qualification, raw files and completion records; it cannot run SAM3 or phase B. The one-time CPU export writes a **new** directory, preserves every native RGB frame, and performs the established raw/action and M5 checks. Keep the output at this path: descriptors and checksums intentionally refuse relocation.

```bash
set -euo pipefail
cd /home/ws/ws/splatter4d
PY=/home/ws/ws/splatter4d/.venv/bin/python
MANIFEST=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/drawer-canonical-manifests/S/manifest.json
RGB=/home/ws/ws/splatter4d/runs/drawer-rgb-inputs-20261115/S
RELEASE_COMMIT=$(git rev-parse HEAD)
IMAGE=sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a

# Read-only preparation preflight; missing hybrid output / closeout is a hard refusal.
"$PY" -B scripts/realworld_rgb_inputs.py --manifest "$MANIFEST" --out "$RGB"
# Only after that succeeds: create new immutable local RGB inputs, no queue writes.
"$PY" -B scripts/realworld_rgb_inputs.py --manifest "$MANIFEST" --out "$RGB" --execute

REG=("$PY" -B runs/setup/tools/register_drawer_rgb_20261115.py
     --manifest "$MANIFEST" --bundle "$RGB" --commit "$RELEASE_COMMIT" --image "$IMAGE")
# Review 32 rows: S only, 98 hybrid deps, pretrain/export -3, dataset/FQL/eval -1.
"${REG[@]}" > /tmp/drawer-rgb-registration-dry.json
# Appends held rows only; no launch, network, pin update or SF hold change.
"${REG[@]}" --execute > /tmp/drawer-rgb-registration.json

mapfile -t RGB_IDS < <("$PY" -B -c 'import json,sys; print("\n".join(r["id"] for r in json.load(open(sys.argv[1]))["rows"]))' /tmp/drawer-rgb-registration.json)
# Official capacity/source/input/ownership placement planner, dry-run by default.
"$PY" -B runs/setup/tools/place_drawer_rgb_20261115.py "${RGB_IDS[@]}"
# Moves only eligible rows and releases only their exact placement-hold IDs; never launches.
"$PY" -B runs/setup/tools/place_drawer_rgb_20261115.py "${RGB_IDS[@]}" --execute
```

Repeat the final dry-run / execute pair after producers close out to place newly ready descendants. The official scheduler retains dependency, memory/CPU/GPU, priority, ownership, source-scope and completion gates. The helper never lifts another pause. Do not execute either historical paired registrar unchanged; this command selects only the four independent RGB conditions. Existing Ours scratch400k rows remain blocked on their geometric inputs, as documented in `AUDIT.md`.

For a graph-only preview **before** hybrid completion/publication, the same registrar without `--execute` works and reports `ready=false`; it writes stdout only. `registration-preview.json` records that read-only 32-row preview, pinned to the starting checkout for audit purposes. Regenerate it with the adopted `RELEASE_COMMIT` before coordinator publication; it is not release evidence.
