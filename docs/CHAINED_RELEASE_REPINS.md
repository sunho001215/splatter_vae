Stopped remote pretraining jobs can retain their id, arguments, data, image and
checkpoint across successive release re-pins. Each chained move requires an
exact current-release X..target Y approval, byte-bound independent evidence for
every scope difference, target acceptance, and explicit resume authorization.
Plan is read-only. Apply locks and appends a registry event; it never launches.

`repin_chain` opts new events into continuity validation. It contains the exact
prior registry transitions (without nested chain records). The reducer rejects
gaps, revisited releases, stale attempts and incomplete records. Legacy events
and all legacy reducer output remain unchanged; history is exposed in reducer
state only after a new chained event. No execution-scope files are changed.

The immutable runtime in `ops/jobs/run_identity.py` still permits only one
release transition. The coordinator launcher handles subsequent transitions on
the host, after deterministic-container, release/image/acceptance and admission
checks, inside the existing admission lock, immediately before Docker launch.
It sends its own stdlib-only `repin_provenance.py` source; it neither imports nor
edits the target's runtime code. Supported trainers share the entry/trainer
provenance path `runs/<job_id>/provenance.json`: train, sincro, reviwo, snerl,
dynamo and mvp3d. Other trainer layouts refuse chained apply until supported.

Preparation validates job, invocation, data, image and the full runtime history
against registry history. It durably saves exact original provenance and the
recorded authorizations in `.release-repins/<binding-sha256>.json`, then atomically
prepares the target release fields and continuous history. Entries explicitly
say `prepared_by: coordinator`; `release_preparation` says runtime start has not
been asserted. This is an authorized launch view, not proof of target execution.
Existing history is never discarded. Data, numeric config and checkpoint bytes
are untouched. The immutable entry sees its authorized target identity; the
trainer still checks its computed config/data identity and writes actual launch
metadata. Attempts on the same target preserve history. Receipts permit retries
before or after the atomic replacement, including after entry/trainer metadata
updates. A failed launch may leave prepared provenance and its receipt; it does
not imply successful execution. The separate registry remains the execution
record. Missing/malformed provenance or inconsistent receipts fail closed.

No new target runtime guard is needed, but adoption of this coordinator code,
exact target/image acceptance, full-range patch approval, and the usual remote
admission checks are required before an operational resume. Scientific evidence
and a passing local suite do not substitute for those approvals.
