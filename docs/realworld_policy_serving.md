# Real-world policy serving over the network

Robot-side CPU client -> ONE persistent inference server (RTX 5090, `irsl`) running encoder + FQL for every retained
FQL condition (DynaMo, ReViWo, SinCro, SNeRL, 3D-MVP single / multi-view, e2e CNN; ours once trained) -> action back to
the client. Directive 2026-11-09 (network policy inference). **Dry-run only: nothing here commands a robot.**

| layer | code | owns |
|---|---|---|
| science | `s4d/rwlearn/deploy/` (docs/realworld_deployment.md) | input contracts (`InputSpec`), CPU crop/resize + native depth (`preprocess_frame`, `realsense`), trained history (`HistoryWindow`), encoder + FQL + action conversion (`DeployPolicy`), typed refusals (`errors`), recorded replay helpers (`replay`) |
| networking | `ops/policy_serving/` | wire protocol, persistent server (sessions, seq/ack, queue, staleness), CPU client (reconnect, resume, action expiry), bench |
| config | `configs/realworld/policy_serving.yaml` | bind address/port, limits, timeouts, served checkpoints (paths relative to `checkpoint_root`) |
| demo | `scripts/policy_replay_demo.py` | recorded episode -> client -> server -> actions JSONL (+ direct equality, recorded commands) |
| tests | `tests/test_policy_serving.py` | CPU loopback, labelled FIXTURE policies of every condition |

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
| WELCOME | s -> c (on connect) | `server`, `nonce`, `auth_required`, `policies: {id: {input_spec (unbound), info}}` |
| HELLO | c -> s | `policy_id`, `choices` {`camera_id` (single view), `native_hw` {cam: [H, W]}, `calibration` {cam: {K_native, c2w}}, `native_depth_correction` {cam: ...}}, `resume` {session_id, last_ack} or null, `auth` (HMAC), `takeover` |
| SPEC | s -> c | `session_id`, `resumed`, `input_spec` (bound `InputSpec.to_json()`), `limits`, `last_seq_received`, `redeliver` [seq] |
| STEP | c -> s | `session_id`, `seq`, `frames` [NEW frames only: `camera_id`, `frame_id` (camera acquisition counter), `capture_ns` (client monotonic), `pose7`, `gripper_register` (measured at that frame), `metadata` (`native_hw` [H, W] of the native colour frame, `K` at the training grid for calibrated conditions; depth: `depth_source`, `native_correction`), `arrays` {rgb, depth, valid}], `sets` [multi-view: `frame_ids` {cam: id}, `surface_disp_m` (M5; null = no valid robot state)], `request` {`end_ids` {cam: id}, `reference_id`, `pose7` + `pose_time_ns`, `gripper_register` + `gripper_time_ns` (the measured samples bound to the observation, client monotonic), `ttl_ms`, `pad_run_start`, validation `noise` (array, float32 (1, 7)) or `noise_seed`} |
| ACTION | s -> c | `seq`, `ack`, `status: ok`, `reference_id`, `outputs` + `output_arrays` (`DeployPolicy.infer` verbatim: `action_norm`, `delta_phys`, `flange_pose`, `execution_flange_pose`, `gripper_closure`, `gripper_register`, `measured_pose7`, `obs_ids`, `frame_timestamps_ns`, `metadata`, `timings`), `server_ms` {prep, queue, validation, encoder, fql, policy, infer, total} |
| ERROR | s -> c | `code`, `detail`, `seq`/`ack`, `history_reset`, `fatal` |
| RESET / RESET_OK | c -> s / s -> c | `seq` (same sequence space as STEP): explicit history reset, the session stays bound |
| PING / PONG | c -> s / s -> c | `t_client_ns` echoed, `t_server_ns` |
| BYE | c -> s | ends the session at once |

Error codes: `version`, `auth` (fatal); `bad_request` (malformed, wrong sizes/dtypes, a frame of another native grid or
intrinsics than the session's, or an input the deploy contract refuses); `session_unknown` (expired / ended / server
restarted); `session_busy`; `session_taken_over` (evicted by another client's takeover: stop); `incomplete_history`;
`irregular_spacing`; `unsynchronized` (mutual-nearest or M5); `binding_refused` (pose / gripper sample not the
observation's, training ages 15 / 100 ms); `duplicate` (repeated seq whose response was evicted); `stale`;
`overloaded` (queue full, or the connection cap); `shutting_down`; `internal` (the worker stays alive).
Deploy refusals map by class (`s4d.rwlearn.deploy.errors`), never by message text.

**Validation mode.** `noise_seed` -> `np.random.default_rng(seed).standard_normal((1, 7), float32)`
(`protocol.validation_noise`, identical on client, server and in direct runs) or an explicit `noise` array. Without
either the server's FQL noise stream is used (production).

## Semantics

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
  multi-view: synchronised steps) ending at `end_ids`. A missing frame (or set) -> `incomplete_history`; frame ids or
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
  --policies dynamo-drawer-open,mvp3d_single-drawer-open --bind 0.0.0.0 --port 47100 \
  --checkpoint-root /ram/serving/checkpoints --token-file /ram/serving/token > /ram/serving/server.log 2>&1'
```

The server loads every listed policy once (sha256 + training-contract checks), warms up (`warmup_iterations`,
synthetic labelled input), prints one JSON `ready` line (address, port, policies, load and warm-up timings) and serves
until SIGINT/SIGTERM (graceful: queued requests answered `shutting_down`, the running one finished). Entry points are
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
Live multi-view needs the M5 robot-surface motion per set (`add_sync_set`): see the open items.

## Dry-run demo and benchmark

```bash
# server on this machine (CPU, loopback) for a local check; any config with local paths works
.venv/bin/python -I ops/policy_serving/server.py serve --config CFG --policies dynamo-drawer-open --device cpu --port 0
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
* The deploy history applies the 1 % timing tolerance to every interval of a window; training only splits runs above
  1.5 periods, so intervals of 1.01-1.5 periods are refused for two steps where training would have used them (safe,
  stricter than training; science decision, `s4d.rwlearn.deploy`).
* Bench `e2e_ms` excludes camera capture/transfer, rs.align and the colour-z conversion (its synthetic depth is already
  on the colour grid): live 3D-MVP end-to-end is larger by those stages.
* `ttl_ms` is counted from arrival on the server; the authoritative freshness check is `check_action` on the robot.
* The token authenticates the client but does not encrypt or integrity-protect payloads (the SSH tunnel does).
* Live multi-view (SNeRL, 3D-MVP multi-view) must NOT run yet: the robot PC would have to compute each synchronised
  set's M5 robot-surface displacement with the training rule (URDF surface FK) and no validated helper exists
  (science API R3). Recorded replay takes it from the training store; single-view conditions need no M5.
* No compression, precision change or compilation: the benchmark decides first (directive item 3).
* The server trusts the client's clock only relatively (spacing, ages); `ttl_ms` is counted from arrival on the server.
