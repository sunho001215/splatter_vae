# Real-world policy serving over the network

Robot-side CPU client -> ONE persistent inference server (RTX 5090, `irsl`) running encoder + FQL for every retained
FQL condition (DynaMo, ReViWo, SinCro, SNeRL, 3D-MVP single / multi-view, e2e CNN; ours once trained) -> action back to
the client. Directive 2026-11-09 (network policy inference). **Dry-run only: nothing here commands a robot.**

| layer | code | owns |
|---|---|---|
| science | `s4d/rwlearn/deploy/` (docs/realworld_deployment.md) | input contracts (`InputSpec`), CPU crop/resize + native depth (`preprocess_frame`, `realsense`), trained history (`HistoryWindow`), encoder + FQL + action conversion (`DeployPolicy`), typed refusals (`errors`), recorded replay helpers (`replay`) |
| networking | `ops/policy_serving/` | wire protocol, persistent server (sessions, seq/ack, queue, staleness), CPU client (reconnect, resume, action expiry), bench |
| config | `configs/realworld/policy_serving.yaml` | bind address/port, limits, timeouts, GPU memory budget, catalog path (+ optional static policies) |
| catalog | `<checkpoint_root>/catalog.json` (`config.load_catalog`, written by the staging tool) | selectable policy ids `<condition>-<task>-s<seed>`, aliases, status deployable / restricted / quarantined + reasons, per-criterion results (`criteria`), usage restriction (`usage`, `usage_reasons`, `execution_eligible`, `execution_decision`), advertised input spec, staged paths + sha256 |
| demo | `scripts/policy_replay_demo.py` | recorded episode -> client -> server -> actions JSONL (+ direct equality, recorded commands) |
| staging | `ops/policy_serving/staging.py` + `criteria.py` | inventory of every real-world pretraining / FQL row (registry + queue read as data), the deployability criteria (realworld-ops/serving/POL-CRITERIA.md), staging of deployable policies to the inference host + catalog + reload (dry run by default) |
| tests | `tests/test_policy_serving.py`, `tests/test_policy_staging.py` | CPU loopback, labelled FIXTURE policies of every condition; criteria, inventory, catalog, dry-run staging |

`ops/policy_serving` imports only `s4d.rwlearn.deploy` (+ numpy/stdlib); it never reinterprets an input or output.
The client side (`protocol`, `client`, `bench`, `config`) imports without torch (tested).

## Protocol (version 1)

One TCP connection per client, `TCP_NODELAY`, kept open for the whole session. One message =

```
u32 little-endian header length | UTF-8 JSON header | payload (raw little-endian arrays, C order, table order, no gaps)
header: {"type", "v": 1, ...fields..., "arrays": [{"name", "dtype", "shape", "offset", "nbytes"}], "payload_nbytes"}
```

Wire dtypes: `|u1 |b1 <u2 <i4 <i8 <f4 <f8` (uint8 RGB, bool validity, float32 metres / normalized actions / noise,
float64 poses, int64 timestamps). Arrays travel at their exact dtype; no compression (128 px frames: 49 KB RGB,
+64 KB depth + 16 KB validity per RGB-D frame). Malformed tables, NaN / Infinity in the JSON, header >
`max_header_bytes` or payload > `max_payload_bytes` (64 KiB each until a HELLO bound a session) are fatal (`bad_request`,
connection closed: the stream cannot be re-synchronised). A message cut by the reader's deadline (the client's step
deadline, at most `io_timeout_s`) also makes the connection unusable: it is dropped and the client resumes.

| message | direction | fields |
|---|---|---|
| WELCOME | s -> c (on connect) | `server`, `nonce`, `auth_required`, `policies: {id or alias: {input_spec (unbound), info {condition, task, seed, run, status, reasons, criteria, usage, usage_reasons, execution_eligible, execution_decision, flags, checkpoint_sha256, resident, alias_of}}}`, `unavailable: {id: {status, reasons}}` |
| HELLO | c -> s | `policy_id`, `choices` {`camera_id` (single view), `native_hw` {cam: [H, W]}, `calibration` {cam: {K_native, c2w}}, `native_depth_correction` {cam: ...}}, `resume` {session_id, last_ack} or null, `auth` (HMAC), `takeover` |
| SPEC | s -> c | `session_id`, `resumed`, `input_spec` (bound `InputSpec.to_json()`), `limits`, `last_seq_received`, `redeliver` [seq] |
| STEP | c -> s | `session_id`, `seq`, `frames` [NEW frames only: `camera_id`, `frame_id` (camera acquisition counter), `capture_ns` (client monotonic), `pose7`, `gripper_register` (measured at that frame), `metadata` (`native_hw` [H, W] of the native colour frame, `K` at the training grid for calibrated conditions; depth: `depth_source`, `native_correction`), `arrays` {rgb, depth, valid}], `sets` [multi-view, optional and informational since 2026-11-12: `frame_ids` {cam: id}, `surface_disp_m` (M5; null = no valid robot state)], `request` {`end_ids` {cam: id}, `reference_id`, `pose7` + `pose_time_ns`, `gripper_register` + `gripper_time_ns` (the measured samples bound to the observation, client monotonic), `ttl_ms`, `pad_run_start`, validation `noise` (array, float32 (1, 7)) or `noise_seed`} |
| ACTION | s -> c | `seq`, `ack`, `status: ok`, `reference_id`, `outputs` + `output_arrays` (`DeployPolicy.infer` verbatim: `action_norm`, `delta_phys`, `flange_pose`, `execution_flange_pose`, `gripper_closure`, `gripper_register`, `measured_pose7`, `obs_ids`, `frame_timestamps_ns`, `metadata`, `timings`; multi-view: + `sync`, the server's sync measurement), `server_ms` {prep, queue, validation, encoder, fql, policy, infer, total} |
| ERROR | s -> c | `code`, `detail`, `seq`/`ack`, `history_reset`, `fatal` |
| RESET / RESET_OK | c -> s / s -> c | `seq` (same sequence space as STEP): explicit history reset, the session stays bound |
| PING / PONG | c -> s / s -> c | `t_client_ns` echoed, `t_server_ns` |
| BYE | c -> s | ends the session at once |

Error codes: `version`, `auth` (fatal); `bad_request` (malformed, wrong sizes/dtypes, a frame of another native grid or
intrinsics than the session's, or an input the deploy contract refuses); `session_unknown` (expired / ended / server
restarted); `session_busy`; `session_taken_over` (evicted by another client's takeover: stop); `incomplete_history`;
`irregular_spacing`; `unsynchronized` (multi-view run-start padding not step-aligned; cross-camera offsets and M5
motion are warnings since 2026-11-12, see "Multi-view sync"); `binding_refused` (pose / gripper sample not the
observation's, training ages 15 / 100 ms); `duplicate` (repeated seq whose response was evicted); `stale`;
`overloaded` (queue full, or the connection cap); `shutting_down`; `internal` (the worker stays alive).
Deploy refusals map by class (`s4d.rwlearn.deploy.errors`), never by message text.

**Validation mode.** `noise_seed` -> `np.random.default_rng(seed).standard_normal((1, 7), float32)`
(`protocol.validation_noise`, identical on client, server and in direct runs) or an explicit `noise` array. Without
either the server's FQL noise stream is used (production).

## Semantics

* **Policy registry (catalog).** The server offers the config's static `policies` plus every entry of the staged
  catalog (`--catalog`, default `server.catalog` under the checkpoint root; schema `s4d-policy-catalog/1`, validated
  by `config.load_catalog`). WELCOME advertises each offered id and alias with its unbound input spec WITHOUT loading
  it (the catalog's `input_spec`, else `server.describe_entry`: checkpoint meta + training contract). A policy is
  loaded on the first HELLO that selects it (sha256, training-contract checks, warm-up; the loaded spec must equal the
  advertised one, else `bad_request`), logged as a JSON `policy_loaded` line (load_s, warmup_ms, footprint), and stays
  resident afterwards. `--preload` loads + warms ids before READY. **Selection latency:** the first HELLO of a
  non-resident policy waits for its load (seconds): select before the control loop starts (or use a client
  `connect_timeout_s` that covers it, or preload).
* **GPU memory budget.** `gpu_memory_budget_mb` (resident footprint = parameters + buffers + optimizer state, estimate
  before a first load = staged file sizes) and optional `max_resident_policies`: a load evicts least-recently-used
  IDLE policies (`policy_evicted` line); a policy with a live session (or a HELLO in progress) is never evicted - the
  HELLO is then refused `overloaded` naming the policies in use. Pre-loaded constructor policies are pinned.
* **Catalog reload without restart.** SIGHUP, every new connection (WELCOME) and a HELLO naming an unknown id re-read
  the catalog when its file changed (`catalog_reloaded` line: added / removed / changed). An invalid catalog keeps the
  current registry. A removed or changed entry is no longer offered in its old form; its resident object keeps serving
  its live sessions and is unloaded when the last one ends; a HELLO for a changed id meanwhile gets `session_busy`
  (or evicts them with `takeover`), then loads the new entry.
* **Aliases and quarantine.** Aliases (e.g. the earlier `<condition>-<task>` ids -> `-s0`) select the same resident
  policy. Entries with status `quarantined` (and entries that cannot be described) are listed in `unavailable` and
  refused at HELLO (`bad_request`, with the reasons); `--allow-quarantined` serves them for pipeline checks only,
  labelled `status: quarantined` + reasons in every info.
* **Sessions.** HELLO binds the session once (`DeployPolicy.reset_session(concurrent=True)`: cameras, native grids,
  static calibration, native depth correction); sessions share the loaded model, at most `max_sessions_per_policy`
  (default 1 = one controller); another HELLO gets `session_busy` or, with `takeover: true`, evicts the least recently
  active one, whose client is told at once (`session_taken_over`). Sessions expire after `session_idle_timeout_s`
  without traffic (PING counts) or end on BYE / server shutdown (`end_session`: their windows are refused from then on).
* **Takeover is terminal for the evicted client.** `step()` then returns `StepResult(code="session_taken_over")`
  (`result.terminal`; `submit()` raises `SessionTakenOver`), also when a disconnected client tries to renew a lost
  session and finds the policy held by another client. It never re-HELLOs by itself: the robot loop must stop;
  `connect()` again only deliberately.
* **Native geometry.** Every frame declares its native colour grid (and, for calibrated conditions, its intrinsics at
  the training grid); client and server refuse frames that differ from the grid / calibration bound at HELLO (the
  server crops K from the session's grid, so another stream resolution would otherwise get a wrong K silently).
  `RealSenseCapture` checks the stream at start and on every frame.
* **History.** The client sends each frame once; the server assembles the trained history (3 frames at spacing 1,
  multi-view: each camera's history ending at its own end frame; misalignment is measured, not refused) ending at
  `end_ids`. A missing frame -> `incomplete_history`; frame ids or
  capture times off the trained spacing -> `irregular_spacing`; never re-spaced, padded or substituted.
  `pad_run_start: true` applies the training rule (frame-stack padding at a run start) and nothing else.
  RESET clears the server history; a frame refused by the contract (or a refused message that carried frames)
  also clears it (`history_reset: true`) and the client re-sends its buffered frames. Re-sending frames the server
  holds is idempotent (identical id + capture time: skipped); a re-sent buffer OLDER than frames already held (a
  pipelined request pushed newer ones after the reset), or a conflicting frame, rebuilds the session history from that
  message (counted as `history_rebuilds`), so a re-send can never dead-end.
* **Sequencing.** Per-session `seq` strictly increasing. A repeated seq returns the cached response (no
  re-inference); one still running is answered once when done; an evicted one -> `duplicate`. Responses are cached
  (`response_cache`) before delivery, so a resumed session gets the ones it missed (`last_ack`).
* **Staleness.** One worker thread (one GPU), bounded queue (`max_queue`). A session keeps at most one queued request:
  a newer one supersedes it (`stale`). A request not started within `ttl_ms` of its arrival -> `stale`. Queue full
  -> `overloaded`.
* **Client.** Reconnect with backoff (`reconnect_backoff_s`), resume (session id + last ack), re-send unacknowledged
  requests with their original seq; a lost session is renewed ONCE (pipelined `session_unknown` answers do not nest a
  second HELLO) and the in-flight steps return `session_unknown`. Responses for requests no longer awaited, from
  another session, or ACTIONs older than an ACTION already received are dropped. `step()` returns `timeout` after
  `step_timeout_s` and never raises for a lost connection or a takeover. Before sending, the client checks the training
  binding rule (pose / gripper sample at or before the observation, <= 15 / 100 ms old).
* **Timings.** `e2e_ms` = first preprocessing of a NEW frame of the step -> decoded action. Frames re-sent after a
  renewal / history reset are not this step's work: they are reported apart (`resent_frames`, `resent_wait_ms`).
  `capture_to_action_ms` / `check_action` ages start at the capture time.
* **Actions are bound.** Every action carries the `reference_id`, frame ids and observation time it was computed
  from, and its goal `flange_pose` is relative to the measured pose of THAT observation. The robot executes the
  absolute goal (through its own safety layer, which stays on the robot PC with the high-rate control) or nothing:
  `client.check_action(result, reference_id=..., now_ns=time.monotonic_ns(), max_age_ms=...)` refuses actions of
  another pose, another session or older than `max_action_age_ms`; never re-apply `delta_phys` to a newer pose.
* **Clocks.** Every timestamp in a request (frames, pose/gripper sample times) is on the robot PC's CLOCK_MONOTONIC;
  the server compares them only with each other. Live RealSense: `client.DeviceClock` reproduces the recorder's
  training frame times exactly: `t = hw_timestamp_ms / 1e3 - (t_recv_unix - t_recv)`, i.e. the librealsense
  GLOBAL_TIME stamp minus the realtime-to-monotonic offset sampled at receipt, refreshed every frame, no pipeline
  latency added (arrival is ~65 ms later); other timestamp domains are refused. The recorder's source is not in this
  repository: the rule is read from its recordings (holds to <= 0.36 us on 5964 colour + stereo frames of 10 episodes
  of all 5 task folders; `tests/test_policy_serving.py` pins recorded rows). Server durations are server-clock
  differences.
* **Auth and limits.** Shared token: the server reads it from `--token-file` at launch; the client proves it with
  HMAC-SHA256(token, per-connection nonce). The token is never sent, stored in source/config or logged. It is REQUIRED
  whenever the server binds beyond loopback (the server refuses to start otherwise). Messages are capped at 64 KiB
  until a HELLO bound a session; at most `max_connections` connections (more are answered `overloaded` and closed).
* **GPU.** `--device cuda` serves only on `server.gpu_uuid`, enforced by `ops.resources.gpu_guard` before torch loads
  (`S4D_HOST_CONFIG` = absolute `configs/hosts/inference.yaml`, `CUDA_VISIBLE_DEVICES` = exactly that UUID, torch's
  device UUID checked); the server refuses to start otherwise. FIXTURE checkpoints are refused without
  `--allow-fixture`.

## Launch

Server (inside `shkim_rw`, the designated GPU only; conventions of `configs/realworld/inference_server.yaml`; the
deployment decides code path, checkpoint root and port):

```bash
# irsl host: worktree revision under ~/shkim/src (see realworld-ops/serving/SURVEY.md), checkpoints + deploy-contract
# sidecars (scripts/check_realworld_deploy.py contracts) copied into the container RAM root, e.g. /ram/serving/checkpoints
docker exec -d -w /opt/src/splatter4d-serve -e CUDA_VISIBLE_DEVICES=GPU-9bcf38b6-8240-e910-6548-ca715643dc11 \
  -e S4D_HOST_CONFIG=/opt/src/splatter4d-serve/configs/hosts/inference.yaml shkim_rw sh -c 'exec /opt/envs/s4d/bin/python -I -B \
  ops/policy_serving/server.py serve --config configs/realworld/policy_serving.yaml \
  --catalog catalog.json --preload dynamo-drawer-open-s0 --bind 0.0.0.0 --port 47100 \
  --checkpoint-root /ram/serving/checkpoints --token-file /ram/serving/token > /ram/serving/server.log 2>&1'
# a newly staged catalog: picked up by the next connection, or explicitly (graceful, sessions keep running):
docker exec shkim_rw pkill -HUP -f '^/opt/envs/s4d/bin/python -I -B ops/policy_serving/server.py serve'
```

The server advertises every offered catalog entry (`--policies ID,...` restricts the offer), loads `--preload` ids
(sha256 + training-contract checks, warm-up with synthetic labelled input), prints one JSON `ready` line (address,
port, offered ids, aliases, resident policies with load / warm-up timings and footprints, unavailable ids, catalog
path + sha256, budget) and serves until SIGINT/SIGTERM (graceful: queued requests answered `shutting_down`, the running
one finished). Other policies load on their first HELLO. Entry points are
run by path (`-I` drops the working directory from `sys.path`; each script adds its repository root itself).

Route: `shkim_rw` is bridge-only. Least invasive: an SSH local forward on the existing master to the container IP
(`ssh -O forward -L 127.0.0.1:47100:<docker inspect IP>:47100 irsl`), server bound to `0.0.0.0` inside the
container + token. A direct LAN route needs a published port (container recreate: coordinator/user decision).

Client (robot PC, CPU, numpy + Pillow + this repository; no torch):

```python
from ops.policy_serving.client import PolicyClient, check_action
c = PolicyClient("127.0.0.1", 47100, policy_id="dynamo-drawer-open", token=open(tok, "rb").read().strip(),
                 choices={"camera_id": cam, "native_hw": {cam: [480, 640]}})
c.connect()
c.add_frame_native(cam, frame_number, capture_ns, rgb_native, pose7=pose, gripper_register=grip)  # crop + resize here
r = c.step({cam: frame_number}, ref, pose7, grip, pose_time_ns=t_pose, gripper_time_ns=t_grip, pad_run_start=True)
ok, why = check_action(r, reference_id=ref, now_ns=time.monotonic_ns(), max_age_ms=200)
```

3D-MVP sessions add `calibration` + `native_depth_correction` (`s4d.rwlearn.deploy.realsense.load_rig_session`) and
pass native depth aligned to colour (`RealSenseCapture(..., depth=True)` = `rs.align` + the shared crop/resize).
Live multi-view: every camera's latest valid frame; cross-camera misalignment is a counted, logged warning (section
"Multi-view sync").

## Deployable policies: inventory, criteria, staging

`ops/policy_serving/staging.py` (read-only on runs, datasets, recordings, registry and queue; no `ops.campaign`):

```bash
# inventory (A) pretraining completion, (B) FQL completion, (C) classes; --evaluate runs criteria c-f on CPU
.venv/bin/python -I ops/policy_serving/staging.py inventory --evaluate --out-json INV.json --out-md INV.md
# stage deployable runs (dry run by default): checkpoint + encoder export + deploy-contract sidecar (+ config /
# completion / provenance; never datasets) -> irsl ~/shkim/serving/checkpoints (sha256, atomic, never overwriting a
# differing file) -> shkim_rw /ram/serving/checkpoints -> catalog.json (merged, atomic) -> SIGHUP (only a server
# whose process catches SIGHUP; older servers are never signalled)
.venv/bin/python -I ops/policy_serving/staging.py stage RUN... [--inventory INV.json] [--execute]
# monitoring hook for ONE newly completed FQL run (re-evaluates every criterion; stages only if deployable;
# exit 0 staged / 3 not deployable / 2 error)
.venv/bin/python -I ops/policy_serving/staging.py hook RUN [--execute]
```

Criteria (`criteria.py`, fixed in POL-CRITERIA.md before any result was read): (a) encoder pretraining completed at
its configured budget, (b) FQL completed at `train.offline_steps` and finally synced, (c) export sha256 chain
pretraining == FQL dataset == policy meta and the policy loads through `DeployPolicy`, (d) deploy-contract sidecar
(`scripts/check_realworld_deploy.py contracts`) accepted at load, (e1) frozen-feature conditioning of the FQL dataset
(offset/signal <= 10, LayerNorm unit std >= 0.5; amendment A2: a policy failing ONLY the LayerNorm-spread gate while
passing e2 is `restricted`: the failure is published, the policy is served for dry run / replay only), (e2)
held-out visual sensitivity through the deploy path (action change under visual-input replacement >= 0.2 x action
spread and >= 0.25 x the FQL-noise-only change; e2e_cnn: image replacement; depth conditions: native depth aligned as
the live client + the task's released rig; multi-view: recorded mutual-nearest sets, M5 not certified by the probe),
(f) finite metrics, no q_mean > 100,
late-phase q_mean <= +1. Explicit decisions (rwcal-20261109 3D-MVP: quarantined) and supersessions are recorded in
`staging.QUARANTINE` / `staging.SUPERSEDED`. Policy ids are `<condition>-<task>-s<seed>`; the pre-catalog ids
`<condition>-<task>` are aliases of a deployable `-s0`.

**Usage restriction (fail closed).** Every catalog entry carries `usage` (`dry_run_replay_only` / `none` /
`execution`), `usage_reasons` and `execution_eligible`; WELCOME and SPEC `info` publish them (`execution_eligible` is
true only if the entry says exactly true). **Execution eligibility (user decisions 2026-11-12 ~16:00, superseding
2026-11-10 21:10 / 2026-11-12 ~00:20; validator allowlist in `ops/policy_serving/config.py`):**
* end-to-end CNN entries (legacy E2E setting) that are `deployable` stay eligible (`staging.EXECUTION_DECISION`);
* legacy frozen-encoder entries (every other condition trained with alpha 10, normalize_q_loss false) are NOT
  eligible and the validator refuses re-enabling them (artefacts and results kept; `staging.py relabel` applies the
  rule to an existing catalog); the a100 comparison setting is never eligible; quarantined entries never;
* new-setting (a10nq: alpha 10 + normalize_q_loss true) entries of every condition, 3D-MVP included, become eligible
  automatically (`staging.AUTO_NQ_DECISION`) once `deployable` and once their deployment validation passed: a record
  `<validation dir>/<run>.json` (auto-eligibility watcher `realworld-ops/serving/tools/pol/stage_watch_v2.sh` +
  `tools/autoelig/validate_run.py`: offline deploy-vs-training equality, in-container server direct == network,
  robot-PC client replay == direct; never a robot command) bound to the run, the checkpoint sha256, the training
  contract sha256 and the input_spec digest, published as the entry's `deploy_validation`; the server additionally
  checks that the checkpoint's stored `agent.config` (alpha, normalize_q_loss) is the setting of the entry's identity
  and advertises nothing eligible otherwise;
* a `restricted` entry (its failed criterion stays FAILED and published, never waived) is eligible only within
  `config.RESTRICTED_EXECUTION_SCOPES` (currently DynaMo a10nq: its representation quality / failed LN-spread
  criterion alone must not block it) under a scoped user decision bound to its checkpoint, with every other check and
  the deployment validation; the server binds it only for a HELLO declaring `restricted-execution-notice/1`;
* `+bcflow` variants are eligible only on an explicitly eligible base under a decision naming bc_flow (never on an
  a10nq base: no deployment validation of their own).
Local FQL runs that exited 0 count as completed for staging (`completed_local`; results_synced is remote-only).
Deployment order for rule changes: server code -> robot-PC client -> catalog; roll back the catalog before older
server code. Withdrawn entries are never eligible. Execution on the robot PC additionally needs its own manual execution mode. A withdrawn policy's remaining
sessions see `status: withdrawn`, `usage: none`; non-finite numbers are refused in catalog entries (JSON wire safety). The robot-PC client refuses execution mode for any other policy. A
re-evaluation that no longer passes withdraws the run's catalog entry (quarantined, usage none). Label changes
(status / criteria / usage) refresh on reload without reloading the model; the catalog transaction is serialised by a
local lock and the container copy is verified byte-identical to the host copy.

**Experimental BC-flow actor (DIAG 2026-11-11).** A catalog entry may carry `actor: bc_flow`: the server loads the
same staged checkpoint with `DeployPolicy(actor="bc_flow")` (the checkpoint's BC flow instead of its one-step policy).
It never replaces a policy: the validator requires its own id `<base id>+bcflow`, the flag `experimental_bc_flow`,
no alias pointing at it, and execution eligibility only under a user decision whose text names bc_flow (never on a
restricted base). `staging.py bcflow-variant ID... [--decision-json F] [--execute]` derives such entries from staged
one-step entries (no files copied); every staging transaction re-derives them from their current base, so a withdrawn
base withdraws its variant. The robot-PC client selects a variant only with `--actor bc_flow` (an explicit id also needs it:
`--policy <id>+bcflow --actor bc_flow`). The validator also requires the variant to carry exactly its base entry's
artefacts, a status no better than the base's (quarantined without a servable base), eligibility only on a deployable
base, and a bc_flow decision bound to the checkpoint sha256 (a re-staged base drops the grant). Static config entries
and pre-loaded objects cannot select an actor.

```bash
.venv/bin/python -I ops/policy_serving/staging.py bcflow-variant reviwo-drawer-open-s0 [--decision-json D.json] [--execute]
```

## Dry-run demo and benchmark

```bash
# server on this machine (CPU, loopback) for a local check; any config / catalog with local paths works
.venv/bin/python -I ops/policy_serving/server.py serve --config CFG --catalog CATALOG --checkpoint-root ROOT \
    --policies dynamo-drawer-open-s0 --device cpu --port 0
.venv/bin/python -I scripts/policy_replay_demo.py --episode /home/ws/data/real_world/drawer-open/episode_000000.h5 \
    --policy dynamo-drawer-open --port P --steps 60 --noise-seed 0 --out actions.jsonl \
    --direct-checkpoint runs/rwng-20261107b-fql-dynamo-drawer-open-s0/policy.pt \
    --direct-export runs/rwng-20261107-dynamo-drawer-open-pre/encoder.pt \
    --direct-training runs/rwng-20261107b-fqldata-dynamo-drawer-open/dataset.h5
.venv/bin/python -I ops/policy_serving/bench.py --port P --policy dynamo-drawer-open --mode paced,max --steps 300 \
    --native-hw 480x640 --out bench.json
```

The demo reads the recording read-only, decodes and aligns frames exactly as the training adapter
(`s4d.rwlearn.deploy.replay.RecordedEpisode`), binds the measured state by the training rule, and writes per step the
action, the recorded command comparison (position / rotation / gripper error) and, with `--direct-*`, whether the
direct `DeployPolicy.infer` result on the identical frames + noise is bit-identical. Multi-view needs `--m5-store`
(the recording's released training-store episode, schema >= 1.1.0). The bench uses synthetic native frames and a
synthetic 15 Hz capture clock; it reports p50/p95/p99 of end-to-end (first preprocessing -> decoded action),
preprocessing, serialization, network (= round trip - server total) and server prep/queue/encoder/FQL/total, payload
bytes (first step = history fill vs steady state), achieved Hz and steps over the 66.7 ms budget.

## Tests

```bash
cd <worktree> && mkdir -p outputs && source scripts/gpu_env.sh && \
  .venv/bin/python -I runs/setup/tools/memory_watch.py fits 3000000000 && \
  CUDA_VISIBLE_DEVICES=$MIG4_3 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=outputs/pytest-net \
  tests/test_policy_serving.py
```

## Limits and open items

* `max_sessions_per_policy` (default 1) sessions share one loaded policy; inference is serialised per policy.
* A lazy load runs on the selecting connection's thread while the worker keeps serving other sessions (shared GPU:
  their latency can rise briefly during a load); eviction frees device memory once nothing references the object.
* The deploy history applies the 1 % timing tolerance to every interval of a window; training only splits runs above
  1.5 periods, so intervals of 1.01-1.5 periods are refused for two steps where training would have used them (safe,
  stricter than training; science decision, `s4d.rwlearn.deploy`).
* Bench `e2e_ms` excludes camera capture/transfer, rs.align and the colour-z conversion (its synthetic depth is already
  on the colour grid): live 3D-MVP end-to-end is larger by those stages.
* `ttl_ms` is counted from arrival on the server; the authoritative freshness check is `check_action` on the robot.
* The token authenticates the client but does not encrypt or integrity-protect payloads (the SSH tunnel does).
* (Resolved 2026-11-12) Live multi-view runs with the validated live M5 helper (docs/robot_pc.md); since the evening
  directive of 2026-11-12 its displacement is informational only (section "Multi-view sync").
* No compression, precision change or compilation: the benchmark decides first (directive item 3).
* The server trusts the client's clock only relatively (spacing, ages); `ttl_ms` is counted from arrival on the server.

## Multi-view sync (user directive 2026-11-12, evening)

A multi-view session or observation is never refused only because the cameras' timestamps are misaligned. The
client steps on every camera's latest valid frame (`ops/robot_pc/syncsets.LatestFrames`); the server assembles each
camera's trained history ending at its end frame as sent and measures the window (`ObservationWindow.sync_report`):
per-camera capture times (client monotonic clock), ages relative to the observation time (newest end frame), pairwise
offsets, the per-step spread, the largest offset from the reference camera vs the training mutual-nearest bound (half
a frame period: `mismatch`) and any declared M5 evidence (`motion_over_bound` above 5 mm). Every multi-view
observation logs one `sync session=...` server line (INFO; WARNING with the reason on a mismatch / M5 motion) and
increments `sync_observations` / `sync_mismatch` / `sync_motion_over_bound` (server and session counters, reported in
the session-end log line); the report is returned as `outputs["sync"]` (network outputs only; the direct
`DeployPolicy.infer` outputs are unchanged and still bit-identical). Kept unchanged: per-camera history checks (trained
spacing, timestamp regularity, frame validity / native correction), missing frames (`incomplete_history`), run-start
padding alignment (`unsynchronized`), calibration / intrinsics checks, binding ages, ttl and the client's action-age
bound. Older clients that declare M5 sets keep working (their sets only annotate the matching steps).

