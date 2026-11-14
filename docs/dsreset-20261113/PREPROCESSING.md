# Immediate local MIG preprocessing

Route A uses a dedicated host-local queue (`ops/campaign/rwprep_mig.py`), adapted
from the established rwprep dispatcher. Standard campaign rows are not presently
feasible: `host_roles.load_roles` explicitly rejects local preprocessing and the
local input resolver has no upstream prep worker environment. The scheduler guard
regex requires a literal `guard_gpus()`/`enforce_allowed_gpus()` call; registering a
raw module command would bypass that contract. The dedicated route instead uses
fresh allowlisted MIG occupancy and the existing prep container isolation checks.
It never imports/executes campaign drivers. Training rows/registry are untouched.

The dispatcher takes `runs/setup/stage2-driver.lock` (both local drivers' lock),
uses memory_budget's 138 GB admission ceiling inside the 150 GB cap, reserves
running training growth and whole prep container limits, CPU quotas + load <=85%,
idle >=20%, PSI <=75, 60 GiB host reserve and >=15% disk free. One prep job owns
one otherwise idle MIG. Ready/running prep creates a durable priority hold read
by `admission_pause`: no NEW training admission until prep drains. Healthy jobs
continue. Missing rig/dependencies do not reserve. A stopped daemon's hold stays
closed until it is restarted; STOP drains workers without killing them. Three
attempts maximum; ambiguous vanished containers require diagnosis, never blind
relaunch. State and per-attempt logs/results persist; outputs use ordinary stage
fingerprints. No files/containers are removed automatically.

Phase A declares 16 GiB GPU (measured <=13.1 GiB including contexts), 48 GiB RAM,
4 CPUs. Thus RAM may limit concurrency before eight MIGs fill. Phase B declares
7 GiB GPU/16 GiB RAM and exports `S4D_TE_CUDA_MEMORY_CAP_GIB=5.5`; the adopted TE
worker chunks/evicts its cache (measured GPU <=5.61 GiB, RAM <=6.2 GB). The old
40 GiB TE profile MUST NOT run on these MIGs. Four concurrent LoMa processes also
need 40 GiB: split them into one-process jobs and use the bounded matching worker
rather than running the historical `run_many ... 4` command.

Coordinator commands (no operation was executed by the implementer):

```bash
cd /home/ws/ws/splatter4d
P=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline
# Use a clean standalone clone of the adopted candidate, with its own .git.
# A worktree .git pointer is not visible inside the read-only container mount.
git clone --no-hardlinks /home/ws/ws/splatter4d "$P/prep-code"
git -C "$P/prep-code" checkout --detach "$(git rev-parse HEAD)"
# Install the established image + pinned weights on LOCAL before registration.
# The local realworld-20261104/weights directory is currently EMPTY.
# On the remote host: docker image save s4d-rwprep-f7e075f51c79:20261106 > /home/compu/kaist/sunho/realworld/prep-image.tar
# Coordinator's established authenticated transfer: copy that tar and
# /home/compu/kaist/sunho/realworld/weights/ to $P/prep-image.tar and $P/weights/.
# Verify every weight against pre/tools/rwprep/configs/rwprep_server.yaml.
docker image load --input "$P/prep-image.tar"
PREP_IMAGE=$(docker image inspect --format '{{.Id}}' s4d-rwprep-f7e075f51c79:20261106)
.venv/bin/python scripts/realworld_prep_register.py \
 --plan "$P/prep-available-drawer-v2/prep-plan.json" \
 --plan "$P/covering-canonical-manifests/prep-plan.json" \
 --plan "$P/pick-and-place-canonical-manifests/prep-plan.json" \
 --out "$P/local-phaseA" --processed /home/ws/data/realworld-processed/dsreset-20261113 \
 --weights "$P/weights" --image "$PREP_IMAGE" --code "$P/prep-code"
# Recompile PNP from a complete immutable 147-episode plan if that plan is stale.
# The compiler skips explicitly reused success outputs (195 existing episodes).
.venv/bin/python ops/campaign/rwprep_mig.py --config "$P/local-phaseA/dispatcher.json"
# Then persistent supervisor on the Docker host (not inside the login container):
sudo systemd-run --unit=dsreset-prep --property=Restart=always \
 --property=RestartSec=15 --property=WorkingDirectory=/home/ws/ws/splatter4d \
 /home/ws/ws/splatter4d/.venv/bin/python /home/ws/ws/splatter4d/ops/campaign/rwprep_mig.py \
 --config "$P/local-phaseA/dispatcher.json" --apply --loop
# STOP admission, keep reconciliation and healthy workers:
touch "$P/local-phaseA/state/STOP"
```

For reboot persistence install the equivalent unit in /etc/systemd/system with
[Install] WantedBy=multi-user.target and `systemctl enable --now dsreset-prep`;
`systemd-run` alone survives logout/process restart but not reboot. Exact permanent
unit and downstream commands are in the pipeline RUNBOOK.

Remote: compile on the executing remote host with `--host remote`, its verified
raw/processed/tools/weights/assets/source paths, and the same 1g.24gb inventory.
Do not reuse local plan paths unchanged: compile host-specific configs. The same
load check intentionally refuses while remote load is ~9000. Do not change the
old whole-GPU dispatcher config or remove its STOP. Partition episode identities
between hosts before registration; never run the same output on both hosts.
Mirror the remote priority hold to the campaign control host before enabling
remote work; the campaign scheduler runs there, not on the remote Docker host.

Phase B: after accepted rig publication, compile a NEW queue using
`--phase B --rig-map <label-inclusive-id-to-accepted-rig.json>` and phase-A configs
as its prep-plan (same output roots). Mount rig documents via `--extra-readonly`.
Phase B runs rig-mode calibrate, capped TE, track_motion, train_data, contracts,
QA `--families`. Before learning gates sync current build-bound records and run
calibration_release plan/apply; no acceptance threshold is relaxed.
