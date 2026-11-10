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
| no live command path exists | `safety.EXECUTION_PERMITTED = False` (Directive I): `LiveSink` refuses at construction AND on every send, `execute.py` refuses before any network activity; the package creates no command socket (the controller binding is the RobotAction MESSAGE FORMAT only) | `test_execution_mode_needs_all_three_explicit_conditions`, `test_live_sink_is_bound_to_the_action_format_but_refused_in_this_revision`, `test_execute_script_refuses_even_fully_authorized` (exit 3, no socket) |
| manual execution needs 4 explicit conditions (for a later directive) | `safety.authorize_execution`: `--execute` + typed `CONFIRMATION` + config `execution.armed: true` + the policy's catalog entry `execution_eligible: true` (missing = NOT eligible; DynaMo is dry-run/replay only by user decision); the authorization object is only issued by that function | same tests |
| replay / dry run / bench never build a live sink or connect to the controller | MockSink / DryRunSink only; `--state live` is refused before anything connects (no read-only gripper source); `safety.admit_endpoint` refuses every connection to ports 28400-28404 (switch_controller channels) | runtime spies in the replay / dry-run tests (0 LiveSink / LiveStateReader constructions, every TCP connect = the policy server) |
| no robot transport in the code | no serial / ROS / Franka / Modbus import; zmq only in `switch_controller.py`, which never binds, never creates a PUB and never sends (read-only SUB, approval-gated, unused here) | `test_no_robot_io_in_the_robot_pc_sources` (AST + text scan), `test_live_state_reader_is_a_read_only_subscriber` |
| no output can open a device | `safety.safe_output`: logs / rows / summaries only to regular files outside /dev, /proc, /sys, never through a symlink | used by DryRunSink, JsonlRows, write_json, selfcheck |
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
1.15.3 and h5py 3.16.0 (the authoritative rig loader, recorded replay and the M5 rule import them), pyrealsense2
2.59.1.11053 (= the recorder's librealsense 2.59.1), pyzmq 27.2.0 + robot-msg-irsl 0.4.0 (the collector's exact pins;
only for the approval-gated state reader). No torch.

**Installed 2026-11-10** on robotpc at `/home/irsl/workspace/s4d-robot-pc` (Ubuntu 24.04.5, python3 3.12.3): releases
5fe4a05, 9eaa2b7, 7cba254 (active); venv via `python3 -m venv --without-pip` + the bundled pip wheel (no
python3-venv / apt needed), wheels offline with `--require-hashes`; selfcheck all OK (packages, no torch, 315 bundle
files, active release, M5 helper, both rig documents, not armed / not permitted, token 0600). The token was copied by
the USER (an agent copy was refused by the permission policy).

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

`scripts/policy_replay_demo.py` (the network demo with the torch DeployPolicy direct-equality check, 306 lines) stays
as is: it is not small and its direct-equality path does not belong on the robot PC; `scripts/robot_pc/replay.py` is
the robot-PC replay.

Development host only: `scripts/realworld_m5_asset.py export|validate` (M5 surface asset, validation),
`scripts/robot_pc/make_bundle.py` (the tarball) and the ONE-command robot-PC benchmark over the `robotpc` ssh master
(REPLAY ONLY, no camera opened; GPU snapshot of the 5090 before/after, LOCAL and robot-PC PING, one policy per
condition x task (6 today), 150 frames 15 Hz paced + max, 10 warm-up steps discarded, about 3-5 min; refuses before
starting and before every policy if the robot PC runs a recorder / teleop / switch_controller process, holds a camera
or has the collector IPC; contended / uncontended results in separate directories):

```bash
cd /home/ws/ws/splatter4d/.worktrees/rwdeploy3   # or the merged main checkout
.venv/bin/python -I scripts/robot_pc/route_bench.py --label uncontended   # --label contended while the GPU is shared
#   -> /home/ws/ws/realworld-ops/serving/rob-bench/<label>-<YYYYmmdd-HHMMSS>/{summary.json,table.md,<policy>.json}
```

**Direct route (live since 2026-11-10 14:50; the LOCAL relay is removed).** A dedicated robot-PC key
(`secrets/ssh_5090_ed25519`, 0600) whose 5090 authorized_keys entry is
`restrict,port-forwarding,permitopen="172.17.0.2:47100",command="/bin/echo tunnel-only"` (set up by the coordinator;
pinned host keys `secrets/known_hosts_5090`). `scripts/robot_pc/tunnel.py start|stop|status` manages exactly the
verified command (`ops/robot_pc/tunnel.py`): status = pid alive + listener + WELCOME; start refuses if the key / host
keys are missing, the key is not 0600, 127.0.0.1:47100 is already bound or a tunnel pid is live, and stops ssh again
if the listener / WELCOME does not come up; stop signals only the pid from `logs/tunnel-direct-<port>.pid` after
checking its command line (never `pkill -f`). If the shkim_rw IP changes, the permitopen entry must be updated first.

## Modules

| module | role |
|---|---|
| `ops/robot_pc/safety.py` | execution modes, `authorize_execution` (--execute + operator ENTER + catalog `execution_eligible`), `authorize_hardware_test` (policy-free gripper / hold tests only) |
| `ops/robot_pc/sinks.py`, `live.py` | MockSink, DryRunSink (never sends), LiveSink (ControllerLink + expiring gripper target); `LiveSession` (live cameras + state -> policy -> 500 Hz executor -> sink, recorded), `MotionShutdown` (executor join -> gripper clear -> controller release before any I/O close) |
| `ops/robot_pc/controller.py` | commander side of switch_controller (execution only): binds :28401/:28402, re-sent mode 3, RobotAction, release = fresh measured-ee hold + bounded still wait + IDLE |
| `ops/robot_pc/gripper.py` | Robotiq 2F-85 Modbus RTU single owner (flock + TIOCEXCL + holder recheck), 50 Hz measured gPO, expiring targets, fault inhibit |
| `ops/robot_pc/recorder.py` | session recorder: inference_outputs/inference_YYYYMMDD_HHMMSS/{inference.h5, camera_<serial>.mp4, summary.json}; batched, zero-drop |
| `ops/robot_pc/conservative.py`, `hwtests.py` | tighten-only `slow` profile + start box; policy-free P3 tests (gripper open/close, zero-motion hold) |
| `scripts/robot_pc/run.py`, `hwtest.py`, `verify_session.py` | live launcher (dry run default; `--execute`), hardware tests (`--rehearse` = read-only), read-back verification of a recorded session |
| `ops/robot_pc/validation.py`, `poses.py` | action validation -> `Command` (absolute goal bound to its observation, expiry) |
| `ops/robot_pc/executor.py` | local high-rate loop design (500 Hz setpoints, speed + lead limits, hold on expiry) |
| `ops/robot_pc/robot_state.py` | read-only state: recorded, static latency-only, live UNBOUND skeleton |
| `ops/robot_pc/cameras.py`, `devices.py` | live RealSense acquisition (threads, training preprocessing) / recorded frames; busy guard |
| `ops/robot_pc/selection.py` | WELCOME -> policy, cameras, HELLO choices (rig document, native depth correction) |
| `ops/robot_pc/loop.py`, `replay.py`, `dryrun.py` | one step (bind -> server -> validate -> sink) + latency summary; drivers |
| `ops/robot_pc/config.py`, `cli.py` | `configs/robot_pc/robot_pc.yaml` reader; entry-point plumbing |
| `ops/robot_pc/switch_controller.py` | the robot PC's controller interface: frames, RobotAction format, approval-gated read-only RobotState reader + the collector's state-health rules |
| `ops/robot_pc/tunnel.py` | the direct robot PC -> 5090 `ssh -N -L` (start / stop / status by pid file) |
| `s4d/rwlearn/deploy/m5_live.py` | live M5: `SurfaceMotion`, `StateTimeline`, `LiveSets`, `LiveM5`, `export_surface` |

## Evidence: recordings and the robot PC's code (READ-ONLY)

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

**Read-only inspection of the robot PC's code (2026-11-10, cat/grep only; nothing built, run or modified)**:
`panda-teleop/teleop_collector/{README.md, config/default.yaml, config/site.yaml, teleop_collector/robot.py,
teleop.py, app.py, requirements.txt}`, `panda-teleop/switch_controller/docs/communication.md`:

* protocol: `robot-msg-irsl==0.4.0` (robot_msg protocol 2) over ZeroMQ (pyzmq 27.2.0).
* RobotState: switch_controller **binds a PUB** on `tcp://*:28400`, ~1 kHz (state publishing "requested at 1,000 Hz"),
  per-peer HWM 1; consumers connect a SUB with `ZMQ_CONFLATE=1`. Fields `sequence`, `q`, `dq`, `tau_ext`,
  `base_wrench_ext`, `joint_frames`, `ee_frames` (normalized xyzw), `metadata.send_time` (unix, serialization time).
  A SUB cannot send application messages to a PUB: reading it cannot change the controller.
* Commands: the COMMANDER (the collector) binds PUBs on :28401 (ControllerCommand `ctrl_mode`; 3 = task impedance,
  requested on the first pedal press, IDLE on exit) and :28402 (RobotAction `frame_id="world"`,
  `real_time_action.ee_pose_rt=[ee pose]`, `target_dt` = `command_timeout_s` 0.05); the controller's SUBs connect to
  them. The controller holds the measured pose when no fresh action arrives within `target_dt` and rejects targets
  > 0.2 m / 0.5 rad from the measured pose.
* Frames: switch_controller's ee = flange (panda_link8) rotated -90 deg about z (`T_FLANGE_EE = make_T(rot_z(-90), 0)`);
  `flange = ee @ inv(T_FLANGE_EE)`, `ee_cmd = flange_cmd @ T_FLANGE_EE` (tcp_offset null). Quaternions w >= 0.
* Gripper: Robotiq 2F-85 over Modbus RTU (`/dev/ttyUSB0`, 115200 8N1, slave 9) driven AND read by the collector's
  gripper thread; closure 0..1 -> position 3..230, speed 146, force 0. **No gripper state is published**: a live
  register needs the collector (or Modbus requests to the gripper) -> no read-only source exists.
* Safety behaviour of the collector: dead-man pedal, re-anchoring, speed (0.3 m/s site, 1.5 rad/s) AND acceleration
  (4 m/s^2, 15 rad/s^2) smoothing, lead limits 0.04 m / 20 deg, stale-state stop (`state_timeout_s` 0.05).

**Controller adapter** (`ops/robot_pc/switch_controller.py`): the frame conversion and `robot_action` (bytes identical
to the collector's construction, checked with the real robot-msg-irsl 0.4.0; RobotState decode -> flange pose exact
up to input rounding), and `LiveStateReader` (read-only SUB, `approved=True` required, drops non-increasing sequences;
cannot detect a frozen publisher). Not connected in this directive. `LiveSink` formats executor setpoints as these
RobotActions for a caller-supplied publisher, but is refused (`EXECUTION_PERMITTED = False`) and no publisher exists.

## LIVE integration design (Directive J, 2026-11-10; P0 read-only inspection of panda-teleop)

Evidence (read-only; teleop_collector = `TC`, switch_controller = `SC`):

| topic | fact | source |
|---|---|---|
| state | SC binds RobotState PUB `tcp://*:28400`, ~1 kHz; consumers connect a CONFLATE SUB | SC docs/communication.md "Topology"; TC robot.py:4, :74 |
| commands | the COMMANDER binds PUBs `:28401` (ControllerCommand `ctrl_mode`) and `:28402` (RobotAction); SC's SUBs connect to `127.0.0.1` | TC robot.py:87 (`_bound_pub`), :122 (`send_ee_pose`), :151 (`request_mode`); SC config/panda_real.yaml:31-35 |
| action | `RobotAction(frame_id="world", real_time_action.ee_pose_rt=[ee pose], target_dt)`; SC holds the measured pose when no action within target_dt, rejects targets > 0.2 m / 0.5 rad | TC robot.py:1-13, :122; SC communication.md "Motion actions" |
| modes | 0 STOP (software cancel), 1 IDLE (backend IdleControl), 3 task impedance (the collector's mode); a selection stops the old worker first | SC docs/controllers.md "Selection", table |
| engage | TC requests mode 3 on the first enable press (re-sent on every engage), waits `CONTROLLER_SETTLE_S` 0.5 s, engages at the measured pose | TC app.py:53, :305-330 |
| disengage / exit | final command at the MEASURED pose (immediate hold), then on exit wait until still and request IDLE (mode 1) | TC app.py:347-363, :785-797 |
| health | state older than 50 ms, invalid, or q/dq/tau bit-identical > 0.25 s blocks motion | TC app.py:54, :243-275 |
| frames | SC ee = flange rotated -90 deg about z; recorded flange = ee @ inv(T) | TC teleop.py:134-139 |
| gripper | Robotiq 2F-85, Modbus RTU `/dev/ttyUSB0` 115200 8N1 slave 9, status FC03 @0x07D0, command FC16 @0x03E8, 50 Hz worker thread that is the ONLY master, activates an unactivated gripper (fingers move), closure c <-> register 3 + 227 c | TC gripper.py:1-27, :72-108, :154-219 |
| recording | one writer owns the HDF5, `.h5.partial` renamed on finish, extendible chunked datasets, rows batched; leftovers trimmed on restart | TC recorder.py:3-5, :77, :139-224 |
| launch | SC container `panda_robot --backend real` starts in IDLE and publishes state; `launch_real.sh` refuses without the RT kernel | SC docs/panda-docker.md "Launch" |

Live verification (P2, READ-ONLY SUB, 2026-11-10 ~19:00, controller up, robot at rest): 1000.0 Hz, 0 sequence gaps,
0 decode errors, send -> receive 0.06 ms on the same host; flange pose from the ee frame == URDF FK of panda_link8 for
the live joints (2e-15 m, 2e-13 deg); `state_blockers` empty (`realworld-ops/serving/robj-evidence/`).

**Design.**
* **State**: `switch_controller.LiveStateReader` (SUB, never binds/sends) as the live robot source; the collector's
  health rules gate bindings and the executor.
* **Gripper (measured + command, no second Modbus master)**: `ops/robot_pc/gripper.py` ports the collector's Robotiq
  RTU protocol and 50 Hz worker into OUR process, which becomes the single master only while the collector is NOT
  running: it refuses known port holders or a teleop collector, takes the lock-file `flock`, opens with pyserial
  `exclusive=True` (an advisory flock), explicitly applies `ioctl(TIOCEXCL)`, then rechecks existing holders and closes
  if any are found. Unreadable `/proc/<pid>/fd` entries/directories are counted as `unverifiable` and exposed in
  status/logs; that count alone does not refuse ownership. TIOCEXCL blocks later unprivileged opens; privileged
  processes can bypass it, and inaccessible root fds prevent proving exclusive ownership. Dry run /
  validation = status reads ONLY (no activation, no command: fingers cannot move); execution additionally sends the
  closure command (deadband / keepalive as the collector) and NEVER activates (an unactivated gripper is reported and
  commands are withheld - activation stays the operator's / collector's action). Targets carry the accepted command's
  monotonic expiry; expired/cleared targets cannot be written or replayed on reconnect, including keepalives. Faults
  inhibit writes. Execution preflight requires activated gSTA=3, fault=0, no communication error and a status <100 ms
  old; it waits up to 1 s for readiness (initial idle-bus fault 0x09 can clear). Execution stops on a fault/inactive
  status or no fresh status for 200 ms; dry run logs/records health only. Measured samples (gPO with host time)
  feed the binding; commands are recorded separately and can never be read back as state (`gripper_origin` guard).
  Blocker found: `/dev/ttyUSB0` is root:dialout 0660 and the robot-PC user is not in `dialout` (the collector reaches it
  from docker): user decision (group membership, or a container with `--device`).
* **Commander**: `ops/robot_pc/controller.py` binds the two PUBs exactly as the collector (so the collector and this
  launcher can never command at the same time: the second bind fails), waits for SC's subscriber TCP connections
  (monitor ACCEPTED), and repeats mode 3 requests every ~100 ms during the 0.5 s settle (send count logged). This
  recovers requests dropped during subscription propagation; TCP connection and PUB send do not prove subscription
  readiness or mode acceptance. RobotState has **no mode field**: the operator confirms acceptance on the controller
  UI. The executor's existing leash/rate-conflict latch detects non-following. Either subscriber DISCONNECTED event
  latches a session stop, even if reconnected before the next tick. Release sends one hold at the measured ee pose as
  received only with a finite sample <50 ms old, waits at least target_dt=50 ms and at most 1 s for fresh |dq|<0.02,
  then requests IDLE + 50 ms sleep whenever a mode was requested. Hold/wait failures cannot skip IDLE; close always runs.
* **Safety** (simplified, directive item 4): one arm/start action (`--execute`, then ENTER) and an immediate stop
  (Ctrl-C, `s`, SIGTERM, or any failure); the hardware e-stop stays the operator's. Kept: the collector's rate / lead
  limits, task workspace boxes, stale / frozen / invalid state stop, expired action -> nothing sent (SC holds),
  communication failure -> stop, controller-compatibility checks (state live, subscribers connected; mode acceptance
  confirmed on the UI), and execution only for policies the catalog marks `execution_eligible`. One cancellation
  event covers startup and streaming, checked before engagement, after ENTER, after subscriber wait, after engage,
  and before streaming; release is registered immediately on link construction. The idempotent stop path runs on a
  separate shutdown thread: stop/join the executor (timeout logged), clear gripper target and stop/join its worker,
  then controller release. Only afterwards close recorder, client, cameras and state reader. Release does not depend
  on an outstanding policy call or recorder/disk progress. Worker exceptions latch a stop; main-loop exceptions stop
  then propagate. Consecutive failures count steps without an accepted command, including validation rejection.
* **Acquisition**: RealSense RGB (+ native depth aligned to RGB only when the policy needs it); never IR. LiveSession
  and the launcher refuse policies needing multiple cameras: live multiview sync sets are not implemented.
* **Recording**: `inference_outputs/inference_YYYYMMDD_HHMMSS/{inference.h5, camera_<serial>.mp4, summary.json}`
  (`ops/robot_pc/recorder.py`), non-blocking writer thread, `.partial` + rename on finalize. Executed rows mark
  `sent=True` only when the execution sink returns successfully; detail distinguishes the arm send from gripper
  target queueing and includes either component's error. Gripper queueing does not attest a serial write.

## Live robot state and measured gripper: UNRESOLVED interfaces

**Robot state (switch_controller RobotState).** Known from the code (read-only): PUB bound by switch_controller on
`tcp://*:28400` (~1 kHz), robot_msg protocol 2 (`robot-msg-irsl==0.4.0`), consumers connect a CONFLATE SUB; fields
sequence, q, dq, tau_ext, base_wrench_ext, joint_frames, ee_frames (switch_controller ee = flange rotated -90 deg about
z), metadata.send_time (unix serialization time, not a measurement age). `switch_controller.LiveStateReader` implements
that reader (never binds, never publishes; drops non-increasing sequences; refuses bindings when the state is older than
50 ms, invalid, or frozen > 0.25 s - the collector's rules) and is constructed only with `approved=True`.
**Not verified** (needs a RUNNING switch_controller and explicit user approval, never done here): that a SUB connection
cannot alter the controller in practice (ZeroMQ SUB->PUB carries only subscription frames, per the protocol), the real
message rate and receive latency, the frozen-state behaviour after STOP, the decode of a real frame, and coexistence
with the collector's own SUB (per-peer HWM 1). The dry run refuses `--state live` before anything connects.

**Measured gripper state.** No read-only source exists today: the Robotiq 2F-85 is driven AND read over Modbus RTU on
`/dev/ttyUSB0` by the teleop collector's gripper thread, and nothing publishes the register; a second reader would
have to send Modbus requests to the gripper (not allowed). The policy's gripper input must be the MEASURED register
(training proprio = `gripper/position`), never the commanded closure (`action/gripper`): every `BoundState` carries a
`gripper_origin`, `robot_state.require_measured_gripper` refuses anything that is not "measured ..." (the labelled
static latency-only state is the one explicit exception) and `PolicyLoop` applies it before every request
(`test_policy_gripper_input_is_the_measured_register_never_the_command`).

## Control path design (no actuation)

1. **Binding** (training rule, checked by the client and the server): pose = latest robot sample at or before the
   observation time (<= 15 ms old), register = latest register sample (<= 100 ms old); the action's goal is relative to
   THAT pose and is executed as an absolute goal or not at all.
2. **Validation** (`configs/robot_pc/robot_pc.yaml` limits): finite, |action_norm| <= 1, unit quaternion,
   |goal - measured| <= 0.04 m / 20 deg, goal inside the task's workspace box (recorded flange range +- 5 cm), closure in
   [0, 1] and register = round(3 + 227 c), observation age <= 200 ms.
3. **Local high-rate loop** (`executor.py`) = the collector's command shaping, ported from its code (read-only):
   `RateLimiter` (teleop.py:154-190: speed <= max_speed and <= the braking speed sqrt(2 a d), velocity change <= a dt)
   with 0.3 m/s (site.yaml teleop.max_linear_speed), 1.5 rad/s, 4 m/s^2 (MAX_LINEAR_ACCEL, teleop.py:39), 15 rad/s^2
   (MAX_ANGULAR_ACCEL, teleop.py:40); goal clamped to 2 x lead around the measured pose, limiter output to 1 x lead
   (0.04 m / 20 deg) with `constrain` (TeleopMapper.step, teleop.py:254-275; clamp_pose_lead :193-205); 500 Hz
   (teleop.rate_hz), elapsed-time steps. Workspace: the collector has NO workspace limits (README "Safety behaviour":
   "There are no workspace limits"); switch_controller rejects targets > 0.2 m / 0.5 rad from the measured pose
   (robot.py docstring), so the configured per-step limits must stay below that (enforced) and the task workspace
   boxes of section 2 are this package's own addition. State health = the collector's `state_ok` (app.py:243-275):
   stop when the measured state is older than `state_timeout_s` 0.05 s or q/dq/tau_ext are bit-identical for
   > 0.25 s (FROZEN_STATE_S, app.py:54: switch_controller republishes its last state when stopped); the stop is
   latched until `rearm()` (+ a new command). Expired command: hold at the measured pose, nothing of it applied.
   The read-only state reader refuses a binding under the same rules (`switch_controller.state_blockers`).
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
test covers both cases against the authoritative functions). The early-finality shortcuts assume every frame interval
>= 0.99 periods; the first shorter interval switches them off (exact waiting) and is counted. The live gripper
interpolation uses nanosecond times (the adapter used the recorder's float seconds): bitwise on both episodes, not
guaranteed bitwise in general (relative weight differences ~1e-9). Through the client path on the real multi-view
3D-MVP checkpoint (local CPU server): 179/179 sets, displacements == store (max diff 0.0)
(`evidence/rob-mv-local-replay-m5-20261110.json`).

Negative cases (`scripts/realworld_m5_asset.py negative`, evidence `evidence/m5-live-negative-20261110.json`): the two
release episodes perturbed IN MEMORY (files untouched) - fast joint motion (143 / 203 sets rejected > 5 mm), a camera
shifted by 0.45 / 0.55 periods, every 5th frame of a camera dropped (36 / 42 non-mutual incomplete sets), a 0.5 s robot
state gap (8 sets without valid state), the state stream ending at 70 % (53 / 62) or starting at 20 % (36 / 41), joints
frozen for 1 s, NaN joints (3): helper frames, inf pattern and 5 mm decisions == the training rule in every case
(finite |diff| 0.0 m); a repeated state time is refused. Frozen state is decision-equal by design (the M5 rule has no
freshness notion): stale / frozen state is refused by the state reader and the executor, not by M5.
**Validated**: the M5 rule (sets, states, displacement, decisions) on 2 release episodes + these perturbations.
**Not validated**: live 3-camera arrival on the robot PC (no multi-view policy is deployable), more release episodes. Evidence: `realworld-ops/serving/evidence/
m5-live-validation-20261110.json`.

## Latency

Numbers, routes and GPU snapshots: `realworld-ops/serving/ROB-BENCH.md` (+ `rob-bench/<label>-<time>/`,
`rob-dryrun*/`). All measured ON the robot PC (robot-PC monotonic clock), dry run / replay only.

* UNCONTENDED, direct route (GPU-9bcf38b6 idle, no foreign process before/after), 2026-11-10 14:39-14:45:
  * live camera dry run: 15.0 Hz; preprocessing -> action p50 / p95 / p99 10.3 / 11.6 / 12.0 ms (reviwo) and 8.8 / 9.9 /
    10.4 ms (e2e_cnn), 0 misses of 66.7 ms; capture -> validated command 72.0 / 73.2 / 73.8 and 73.8 / 74.8 / 75.9 ms:
    ALL miss 66.7 ms because the camera's own device-stamp -> host latency is 62.7-66.0 ms (recordings: 65 ms);
  * recorded replay (6 policies): preprocessing -> action p50 8.5-10.7 ms, p99 <= 22.7 ms, 0 misses; max rate 77-93 Hz;
    transport 5.6-6.2 ms per 50 KB request (PING 0.7 ms); server 1.5-3.9 ms.
* CONTENDED (another user's training at 97 %), relay route (earlier): preprocessing -> action p50 40-79 ms, server
  31-76 ms, DynaMo / SinCro over 66.7 ms; transport 7.3 ms.

**Not yet measured**: live multi-view / RGB-D (no deployable multi-view or 3D-MVP policy; native rs.align cost on the
robot PC unmeasured), live robot-state age (live state refused), setup.sh executed as one script on the robot PC
(its steps were run individually; the single-script remote execution was refused by the permission policy).
