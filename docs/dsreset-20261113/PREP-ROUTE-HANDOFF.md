# Container-only preprocessing handoff

Candidate branch: `dsreset-prep-route`, worktree
`/home/ws/ws/splatter4d/.worktrees/dsreset-prep-route`.
Frozen source: `fb3baee8c012e848d742081a01b88a03df469b0c`, descendant of requested
`228e9307f392f50ae72f4588f19cec321e4309f1` and current main `9c901bf`.
The final branch commit adds only documentation/evidence over this tested source.

FULL LOCAL SUITE PASSED: **4000 passed, 0 failed/error/skipped**, 3973 collected
plus 27 subtests, exit 0, 866.45 seconds, unchanged source fingerprints. Evidence
worktree `/home/ws/ws/splatter4d/.worktrees/dsr` remains at the frozen source HEAD;
only its docs contain generated evidence. Reports are `docs/tests.json`,
`docs/tests.log`, and `docs/tests.junit.xml`; copies are archived on this branch.
`expected_commit` is null because the runner environment did not set it. The
normal adoption helper explicitly accepts null or the evidence HEAD; the evidence
HEAD and exact source fingerprints were independently verified. No report field
was filled in or altered after execution. All test files are covered exactly.

The long development worktree path exceeded multiprocessing socket limits on the
initial attempt, so the full run used the short `dsr` evidence worktree. The failed
setup attempts are not final evidence. No source changes occurred after the source
commit or during the passing suite.

Focused route tests: 22 passed. Additional offline integration compiled the real
248-job membership, generated both publication archives, checked standalone Git
metadata/remote paths, and exercised the actual established launcher with a FAKE
transport. The generated launch command contains both scheduler/rwprep remote
locks and one MIG CDI device. Fake transport executed no SSH/Docker commands.
Audit: `prep-route-offline-evidence.json`, including external helper SHA256 pins.
The fake-image preview is not launchable operational state.

Exact coordinator procedure: `PREP-ROUTE.md`, also copied to the requested
`/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/PREP-ROUTE.md`. It replaces the
forbidden local-host setup with a campaign-container controller, checked remote
uploads, the installed remote prep image/weights/raw data, and the scheduler's
shared MIG capacity guard and boot hold. CPU admission uses our enforced campaign
quotas against the LIVE scheduler budget; global load/idle/PSI are excluded.
Existing RAM/disk/GPU limits are preserved. Priority -4 and separate durable
priority holds precede new training. The legacy whole-GPU dispatcher stays stopped.
A starts before optional submodule/LoMa staging; later calibration, bounded TE,
QA, snapshot/records, learning inputs and measurements use the same remote route.

Live limit: RemoteClient read-only probes returned RemoteUnreachable (including
its bounded ten-minute waiting probe). No live remote image/capacity/model smoke
was possible. R2 was not selected; the campaign venv lacks phase-A dependencies,
while auxiliary SAM3/FS environments pass imports but do not provide the existing
strict read-only/tmpfs worker envelope in this campaign container. No local model
inference or real preprocessing was launched. GPU use was the required suite.

No operational queue/registry/pauses/legacy STOP changes, policy deployment,
reviewed push, adoption, host operations or restart-policy changes were performed.
The coordinator must restore remote reachability, follow normal reviewed adoption,
prepare a fresh operational bundle and inspect its dry tick before its explicitly
identified first real launch command.
