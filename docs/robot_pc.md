# Robot-PC client: deployment package, control path WITHOUT actuation, live M5 helper

Directive I (2026-11-10). The robot-side PC (Ubuntu 24.04, hostname `irsl` / recordings `irsl-MS-7C79`, 192.168.50.20,
`/home/irsl/workspace`; called **robotpc** here, never confused with the 5090 server whose ssh alias is also `irsl`)
acquires camera frames and the robot state, sends observations to the policy server on the RTX 5090
(`docs/realworld_policy_serving.md`) over one persistent connection, validates the returned action and hands it to a
command sink. **The robot is never operated in this revision**: nothing here commands, moves, initialises, homes or
switches controllers of any robot, arm or gripper. Science (preprocessing, history, contracts, M5 rule) comes from
`s4d/rwlearn/deploy/` (`docs/realworld_deployment.md`).

## Safety guarantees (and the tests proving them, `tests/test_robot_pc.py`)

| guarantee | mechanism | test |
|---|---|---|
| no live command path exists | `sinks.LiveSink` refuses without an `ExecutionAuthorization` and then ALWAYS raises `Unbound` (controller binding UNBOUND until the robot PC's teleop / switch_controller code is inspected read-only) | `test_execution_mode_needs_all_three_explicit_conditions` |
| manual execution needs 3 explicit conditions | `safety.authorize_execution`: `--execute` flag + typed `CONFIRMATION` + config `execution.armed: true`; the authorization object cannot be built otherwise | same test; `test_execute_script_refuses_even_fully_authorized` (exit 3, no socket) |
| replay / dry run / bench never build a live sink or a live state source, and talk only to the policy server | MockSink / DryRunSink only; `LiveRobotState` is an UNBOUND skeleton (raises) | runtime spies in `test_single_view_replay_end_to_end`, `test_multiview_rgbd_replay_with_the_live_m5_helper`, `test_replay_episode_composition_and_cli_with_a_config`, `test_live_dry_run_logs_only_and_refuses_busy_devices` (0 constructions, every TCP connect = the server) |
| no robot transport in the code | no zmq / serial / ROS / Franka / Modbus import anywhere in `ops/robot_pc`, `scripts/robot_pc`; only `sinks.py` / `execute.py` mention the live sink | `test_no_robot_io_in_the_robot_pc_sources` (AST scan) |
| live acquisition cannot interfere with a recording | `devices.require_free` before any camera opens: refuses on a recorder / teleop process, a `/dev/video*` / USB holder, or the collector IPC `/tmp/teleop_collector_requests.ipc`; librealsense refuses a streaming device as last check | `test_device_busy_guard`, dry-run test |
| torch-free robot side | the whole client stack, `ops/robot_pc`, `m5_live` import with `torch` blocked | `test_robot_pc_stack_imports_with_torch_blocked` |
| actions bound to their observation | validation refuses another reference id / session / frame ids / sample times / echoed measured pose; executor only adopts newer observations, never re-derives the goal from a newer pose | `test_action_validation_accepts_bound_actions_and_names_every_refusal`, `test_pose_limits_and_high_rate_executor` |

## Robot-PC folder layout

```
/home/irsl/workspace/s4d-robot-pc/            (ROOT; nothing outside it is written)
  releases/s4d-robot-pc-<short>/              extracted bundle: src/ (repo subset at <commit>), assets/, wheels/, setup.sh, BUNDLE.json
  src -> releases/<current>/src               switched atomically by setup.sh
  assets -> releases/<current>/assets         m5/ (M5 surface asset + validation), urdf/ (unmodified composed URDF + manifest), rig/ (per-task rig documents + reference npz)
  venv/                                       python3.12 venv (python3 -m venv --without-pip; pip from the bundled wheel; pinned wheels offline, --require-hashes)
  config/robot_pc.yaml                        edit here (copied once; later templates land as robot_pc.yaml.new)
  secrets/token                               0600, copied by the USER (never in the bundle, repo, logs)
  logs/  runs/                                selfcheck.json, replay / dry-run / bench outputs
```

Packages (pinned to the training lock): numpy 1.26.4, Pillow 11.3.0 (bit-exact RGB resize), PyYAML 6.0.3, scipy
1.15.3 and h5py 3.16.0 (the authoritative rig loader, recorded replay and the M5 rule import them), optional
pyrealsense2 2.59.1.11053 (= the recorder's librealsense 2.59.1). No torch.

## Install (user actions marked USER)

```bash
# 5090 (staged by the coordinator): ~/shkim/robot-pc/s4d-robot-pc-<short>.tar.gz (+ .sha256)
# USER, on robotpc: fetch the bundle (scp from irsl@143.248.142.98:shkim/robot-pc/ or a USB stick), then
mkdir -p ~/workspace/s4d-robot-pc/releases
sha256sum -c s4d-robot-pc-<short>.tar.gz.sha256
tar -C ~/workspace/s4d-robot-pc/releases -xzf s4d-robot-pc-<short>.tar.gz
bash ~/workspace/s4d-robot-pc/releases/s4d-robot-pc-<short>/setup.sh          # venv, links, config, self-check
# USER: the server token (from the 5090: ~/shkim/serving/token) -> robotpc, never printed:
install -m 600 /dev/null ~/workspace/s4d-robot-pc/secrets/token && cat > ~/workspace/s4d-robot-pc/secrets/token
# USER: the route = the robot PC's own SSH local forward to the shkim_rw container (its IP: docker inspect on the 5090)
ssh -N -L 127.0.0.1:47100:<shkim_rw container IP>:47100 irsl@143.248.142.98
```

Git-clone alternative (only if robotpc can reach a remote holding the commit): `git clone --no-checkout <url> src &&
git -C src checkout --detach <commit>`, then the wheels / assets of the bundle are still needed (assets are not in git).

## Commands (on robotpc; `cd ~/workspace/s4d-robot-pc`; every command is dry: MockSink / DryRunSink)

```bash
venv/bin/python -I src/scripts/robot_pc/selfcheck.py --server            # packages, assets, rig, safety, WELCOME
venv/bin/python -I src/scripts/robot_pc/policies.py                       # offered policy ids + requirements
# recorded replay (recordings of this PC: config output_dir /data):
venv/bin/python -I src/scripts/robot_pc/replay.py --episode <episode.h5> --policy reviwo-drawer-open-s0 \
    --pace realtime --out runs/replay.jsonl --summary runs/replay.json
# latency benchmark (replay path, 15 Hz paced + max rate; label the route):
venv/bin/python -I src/scripts/robot_pc/bench.py --episode <episode.h5> --policy reviwo-drawer-open-s0 \
    --steps 150 --route "robotpc -> 5090 (ssh -L)" --out runs/bench-reviwo.json
# live dry run (cameras live; robot state: --state live is UNBOUND -> refused; --state static = LATENCY ONLY):
venv/bin/python -I src/scripts/robot_pc/dry_run.py --policy reviwo-drawer-open-s0 --state static --steps 300 \
    --route "robotpc -> 5090 (ssh -L)" --log runs/dryrun.jsonl --summary runs/dryrun.json
venv/bin/python -I src/scripts/robot_pc/execute.py --execute             # future mode: REFUSED (exit 3)
```

Development host only: `scripts/robot_pc/m5_asset.py export|validate` (M5 surface asset, validation) and
`scripts/robot_pc/make_bundle.py` (the tarball).

## Modules

| module | role |
|---|---|
| `ops/robot_pc/safety.py` | execution modes, `authorize_execution`, `CONTROLLER_BINDING = "UNBOUND"` |
| `ops/robot_pc/sinks.py` | MockSink, DryRunSink (JSONL, never sends), LiveSink (refuses, then Unbound) |
| `ops/robot_pc/validation.py`, `poses.py` | action validation -> `Command` (absolute goal bound to its observation, expiry) |
| `ops/robot_pc/executor.py` | local high-rate loop design (500 Hz setpoints, speed + lead limits, hold on expiry) |
| `ops/robot_pc/robot_state.py` | read-only state: recorded, static latency-only, live UNBOUND skeleton |
| `ops/robot_pc/cameras.py`, `devices.py` | live RealSense acquisition (threads, training preprocessing) / recorded frames; busy guard |
| `ops/robot_pc/selection.py` | WELCOME -> policy, cameras, HELLO choices (rig document, native depth correction) |
| `ops/robot_pc/loop.py`, `replay.py`, `dryrun.py` | one step (bind -> server -> validate -> sink) + latency summary; drivers |
| `ops/robot_pc/config.py`, `cli.py` | `configs/robot_pc/robot_pc.yaml` reader; entry-point plumbing |
| `s4d/rwlearn/deploy/m5_live.py` | live M5: `SurfaceMotion`, `StateTimeline`, `LiveSets`, `LiveM5`, `export_surface` |

## What is inferred from the recordings (READ-ONLY evidence; the robot PC's code was never seen)

All from `/home/ws/data/real_world/{drawer-open,mug-hanging}/episode_*.h5` attrs / datasets (200 episodes, identical
configs: software 2.0.0, host irsl-MS-7C79) and `s4d/realworld/adapters/panda_teleop.py`:

* endpoints (`config_yaml.robot`): `state_endpoint tcp://127.0.0.1:28400`, `command_bind tcp://127.0.0.1:28401`,
  `action_bind tcp://127.0.0.1:28402`, `request_endpoint ipc:///tmp/teleop_collector_requests.ipc`,
  `controller_mode 3`, `command_timeout_s 0.05`, `state_timeout_s 0.05`; the gripper is driven over `/dev/ttyUSB0`
  (Robotiq, speed 146, force 0) by the collector itself (a second serial reader would interfere).
* events: `ARMED` -> `requested switch_controller mode 3; enabling in 0.5 s` -> `motion ENABLED` -> `RECORDING` ->
  `motion DISABLED`; i.e. the collector switches the controller itself. We never do.
* robot state: `robot/seq` advances ~10 per 10 ms row -> the state publisher runs at ~1 kHz; recorded rows 100 Hz
  (`robot_rate_hz`), `t_recv - t` 6 us; fields `q` (rad), `dq`, `flange_pose` [x y z qx qy qz qw] of panda_link8 in
  panda_link0 (m; `conventions_json`), `joint_frames`, `tau_ext`, `wrench_ext`, `t_send_unix`.
* frames: flange = panda_link8 (link7 + 0.107 m), `T_flange_tcp` = identity (TCP == flange), controller `tcp` body =
  flange rotated -90 deg about z (`T_flange_controller_ee`); base = panda_link0 == switch_controller `world`.
* commands: `action/flange_pose` (sent, engaged) at the 500 Hz teleop loop, lead-limited to <= 0.040 m around the
  measured pose (max over all episodes: exactly 0.040), `max_rotation_lead_deg 20`, `max_linear_speed 0.3`,
  `max_angular_speed 1.5`; gripper command closure c in [0, 1], register = 3 + 227 c; measured register
  `gripper/position` 0..255 at ~60 Hz (dt p50 16.0 ms).
* cameras: D435I 640x480 colour (bgr8 -> JPEG q90) + z16 depth at 15 fps, librealsense 2.59.1; frame time = device
  global-time stamp on the host monotonic clock; frames reach the host **~65 ms** after that stamp (`t_recv - t` p50
  64.9-65.2 ms on all three cameras).

**Controller binding: UNBOUND.** The live robot-state source and the live sink stay unbound until the teleop /
switch_controller code on robotpc has been read (read-only, never modified) to fix message formats, ownership of the
command endpoints and the safe hand-over (the collector owns the controller switch and the gripper port).

## Control path design (no actuation)

1. **Binding** (training rule, checked by the client and the server): pose = latest robot sample at or before the
   observation time (<= 15 ms old), register = latest register sample (<= 100 ms old); the action's goal is relative to
   THAT pose and is executed as an absolute goal or not at all.
2. **Validation** (`configs/robot_pc/robot_pc.yaml` limits): finite, |action_norm| <= 1, unit quaternion,
   |goal - measured| <= 0.04 m / 20 deg, goal inside the task's workspace box (recorded flange range +- 5 cm), closure in
   [0, 1] and register = round(3 + 227 c), observation age <= 200 ms.
3. **Local high-rate loop** (`executor.py`): 500 Hz setpoints moving toward the bound goal at <= 0.3 m/s, 1.5 rad/s and
   clamped to the lead envelope around the CURRENT measured pose (the recorder's own limits); a command is replaced only
   by one of a newer observation; after its expiry the loop holds at the measured pose.
4. **Stale / timeout / disconnect / reset**: `PolicyClient` (reconnect + resume, re-send with the original seq,
   `session_unknown` renewal, terminal `session_taken_over`); a step without a valid action produces no command.

## Live M5 (multi-view) helper

`m5_live.LiveM5` reproduces the training store's M5 evidence online: mutual-nearest sets (`train_store.mutual_sync`,
emitted as soon as later frames cannot change them), member states by `pipeline.robot_state_at` with the adapter's
gripper interpolation, displacement by `train_store.set_surface_motion` over the training store's legacy surface
(`RobotGeometry(meta)`: 5400 points, 18 links, seed 0), sampled once on the development host and shipped as
`assets/m5/panda_robotiq_2f85_droid_iso50.m5.npz` next to the unmodified URDF (sha256 ca60bab9...); the robot PC poses
it with the authoritative `URDFModel.fk`.

Validation (criterion pre-registered in `realworld-ops/serving/ROB-STATUS.md`: |diff| <= 1e-9 m, identical inf pattern,
decisions and `sync/frames`): release store drawer-open-000000 (179 sets) and mug-hanging-000001 (207 sets + 1
incomplete), both the store-trajectory path and the LIVE path (raw recording replayed in arrival order): **max |diff|
0.0 m (bitwise), frames / decisions / inf identical**; sets are emitted with no wait beyond the frames' own arrival.
Only those two release episodes are available locally; neither contains a rejected (> 5 mm) or invalid set (the unit
test covers both cases against the authoritative functions). Evidence: `realworld-ops/serving/evidence/
m5-live-validation-20261110.json`.

## Latency

Measured routes and numbers: `realworld-ops/serving/ROB-BENCH.md`. **Not yet measured (need robotpc access)**: the
robotpc -> 5090 route itself (RTT, transport), robot-PC CPU preprocessing / native librealsense rs.align, live camera
capture -> action (expected >= 65 ms by the recorded camera latency alone), live robot-state age, the 3-camera
multi-view live dry run, and setup on the real Ubuntu 24.04 python3.
