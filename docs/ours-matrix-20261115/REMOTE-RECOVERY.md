# Remote reconciliation after VPN recovery

This checklist prepares recovery; it is not authorization to launch duplicate
jobs, stop healthy work, change pins, or mutate the Robot PC. No remote action
was performed by the matrix implementer.

1. Revalidate the remote host identity, boot identity, accepted 16 MIG inventory,
   current image digest, free disk/RAM, enforced CPU budget and boot launch holds.
   Use `RemoteClient` and the established read-only ledger/status tools. Keep
   remote W&B offline. Do not clear an outage or boot hold solely because SSH works.
2. Reconcile **all** registry attempts with container labels, job ID, attempt,
   source commit, fingerprints, start time, PID/container identity, exit records,
   required results, and `remote_verified`/backup hashes. A timeout, missing SSH
   reply or absent local backup does not establish failure. Keep `unknown` intents
   blocked until the existing remote ledger reconciliation proves their state.
   Healthy jobs keep their recorded scopes, ownership, priorities and devices.
3. Inventory the phase-A and phase-B dispatcher bundles, immutable source/config
   archives, prep image ID, exact helper hashes, staged weights, raw verified
   manifests, state/launch intents and output `_COMPLETE` fingerprints. The old
   `realworld-ops/pre/STOP` remains present. Check the independently qualified
   local phase-A bundle and accepted CPU equivalence before choosing remaining
   work. Reuse accepted local/remote outputs by task **and outcome label and raw
   SHA**, never episode number alone. Drawer77/79 stay excluded. Do not prepare a
   second dispatcher from the old inventory or restart already completed episodes.
4. Before publishing local outputs remotely, inventory exact local completed
   outputs and accepted stage evidence, and compare any destination's manifest
   and hashes. Upload to a unique staging directory through the existing checked
   archive publication route; verify bytes and required stage markers before
   atomic publication to the canonical location. An existing matching directory
   is a reuse; a conflicting directory requires reconciliation, never overwrite.
   Upload canonical compact episode outputs and their original metadata, accepted
   rigs/references, records, and the five-file learning-input bundles. Keep raw
   datasets read-only and preserve canonical paths and label-inclusive identities.
5. Validate paired membership, by-episode split, outcomes, QA/calibration and
   train-only statistics with the adopted dsreset checks. S success episodes,
   splits and rewards match the success subset of SF. Failure episodes receive
   no success reward and end with truncation; actions are ordinary transitions,
   not per-action negatives. Publish **both** variants atomically with the dsreset
   registrar's checks. The matrix's dependency IDs are the standard
   `rwv2-20261113-qa-ours-<task>` publication barriers and
   `rwv2-20261113-prep-ours-<task>-<S|SF>` statistics producers. Do not create fake
   `done` events for them. Complete the normal local mirror and producer closeout
   route before local placement; remote file presence alone does not release rows.
6. Inventory local matrix/continuation run directories and placement ownership
   before any remote training admission. Registered matrix rows are initially
   held by their exact-ID **local-release** hold. The matrix place helper moves
   only proven eligible rows locally and lifts only those IDs. Never register
   another copy remotely, remove this hold wholesale, or substitute the old
   200k+restart recipe for the new scratch matrix. Upload completed local
   checkpoints, encoder exports, fqldata and policies with source/attempt/result
   hashes into separately identified staging locations; do not claim remote
   execution or change the registry host/attempt to make an upload look executed.
7. Reconcile deployment manifests by task, S/SF, Original/Chamfer0.1,
   scratch/continuation, checkpoint and FQL seed. The deployment worker consumes
   `deployment_hook` records and the continuation hooks in the registrar proof,
   verifies FQL closeout plus the linked task evaluation, and deduplicates by policy
   SHA and complete experiment identity. Preserve existing production policies
   and operator workflows; do not restart/reload inference or initiate a robot
   trial. Physical execution needs the user's separate explicit authorization.
8. Run read-only remote admission and dispatcher ticks first. Let the coordinator
   select only missing, reconciled attempts under the ordinary admission locks
   and capacity budgets. Preserve the local phase-A worker's qualification result;
   an unfinished qualification is not a reason to relax its gate. Restore normal
   upload bandwidth controls via their recorded restoration mechanism if an
   uncapped window was used. Record the ledger, manifests, hashes, reused/missing
   IDs and selected route in a recovery report before enabling admission.
