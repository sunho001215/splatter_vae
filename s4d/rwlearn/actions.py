"""Robot action contract for offline RL on the ``panda_teleop_episode`` v2 recordings (Franka Panda + Robotiq 2F-85).

Evidence and the reasoning behind every choice: docs/realworld_fql.md ("Action contract"). Summary:

* source: the RECORDED COMMAND actually sent to the robot, ``action/flange_pose`` (rows at the 100 Hz robot loop, time
  ``robot/t``; ``action/sent`` and ``action/engaged`` must be 1). Inferred from the data, not documented by the file
  (open question in the design doc): ``action/flange_target`` is the operator's raw target and ``action/flange_pose``
  is that target limited to ``teleop.max_position_lead`` / ``max_rotation_lead_deg`` around the measured pose
  (|flange_pose - robot/flange_pose| <= 0.040 m on every row; rows with ``lead_limited`` = 1 have |target - pose|
  ~ 0.04 m). Measured poses (``robot/flange_pose``) are used only as the reference of the delta and as proprioception.
* point / frame: the flange (``panda_link8``) origin; ``T_flange_tcp`` must be the identity (checked per episode: the
  TCP IS the flange in these recordings, ``teleop.tcp_offset: null``). Increments are expressed in the robot BASE frame
  (``panda_link0``, +x forward, +y left, +z up).
* control interval: one policy step = one camera frame of the observing camera (15 Hz, ~66.7 ms). The commands of the
  ~6.7 robot-loop rows inside step k are aggregated by ZERO-ORDER HOLD AT THE STEP END: the last command issued
  strictly before the next frame time tau_{k+1}.
* delta: a_k = goal relative to the MEASURED flange pose at the observation time (latest robot sample <= tau_k):
  dp = p_goal - p_k (m), r = log(R_goal R_k^T) as a rotation vector (axis-angle, rad, base frame, left increment);
  hence R_goal = exp(r) R_k and p_goal = p_k + dp (robosuite OSC-delta semantics in the base frame).
* gripper: the recorded command ``action/gripper`` = closure c in [0, 1] (0 open, 1 closed; Robotiq position register
  = 3 + 227 c, file convention) at the same row; policy dimension g = 2c - 1. Kept continuous (exact round trip);
  ``binary_gripper`` thresholds at c = 0.5 for execution if wanted (open decision).
* normalisation to the policy's [-1, 1]^7: fixed physical bounds from the recorder's own teleop limits,
  pos_scale = max_position_lead + max_linear_speed * max_step_s, rot_scale = rad(max_rotation_lead_deg) +
  max_angular_speed * max_step_s, max_step_s = 1.5 x the nominal frame period. Every recorded command lies inside
  them (checked when building datasets); a policy output never maps outside the teleop's own command envelope.
* proprioception (policy input, all baselines, as the campaign's sim protocol gives every method proprio): measured
  flange position (3, m, base), orientation as the first two columns of R (6D, Zhou et al. 2019; continuous, unlike
  Euler/quaternion inputs), measured gripper closure (Robotiq ``gripper/position`` register -> (pos - 3) / 227,
  clipped to [0, 1]) -> 10 dims.

All times are int64 nanoseconds on the recorder's host CLOCK_MONOTONIC (``np.rint(seconds * 1e9)``, as the
preprocessing adapter does). Only ``h5py`` reads of the recording; the file is never written.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FORMAT = "panda_teleop_episode"
GRIPPER_OPEN_REGISTER, GRIPPER_SPAN = 3.0, 227.0  # conventions_json: "Robotiq position 3 + 227 * closure (DROID)"
ACTION_DIM, PROPRIO_DIM = 7, 10
NOMINAL_PERIOD_S = 1.0 / 15.0  # configured colour frame rate (config_yaml camera.fps 15); measured 14.99 fps
PERIOD_TOLERANCE = 0.01  # every camera's measured median frame period must lie within 1 % of the nominal one
MAX_GRIPPER_AGE_NS = 100_000_000  # gripper register samples arrive at ~50-60 Hz; older ones are not current state
ACTION_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")


def ns(seconds) -> np.ndarray:
    return np.rint(np.asarray(seconds, np.float64) * 1e9).astype(np.int64)


@dataclass
class ActionContract:
    """Everything a consumer needs to interpret a 7-d normalised action (stored in datasets and checkpoints)."""

    pos_scale_m: float
    rot_scale_rad: float
    nominal_period_s: float
    max_step_s: float
    source: str = "action/flange_pose (sent command, lead-limited) at the last robot-loop row before tau_{k+1}"
    reference: str = "robot/flange_pose measured at the latest robot sample <= tau_k (observation time)"
    point: str = "panda_link8 (flange) origin; T_flange_tcp == identity (TCP == flange)"
    frame: str = "panda_link0 base frame (+x forward, +y left, +z up); left increment R_goal = exp(r) R_k"
    rotation: str = "rotation vector (axis-angle, rad)"
    units: str = "m, rad; gripper closure in [0, 1]"
    gripper: str = "action/gripper closure c (0 open .. 1 closed), Robotiq register = 3 + 227 c; policy g = 2c - 1"
    normalisation: str = ("a_norm = [dp / pos_scale_m, r / rot_scale_rad, 2c - 1]; scales from the recorder's teleop "
                          "limits (max_position_lead + max_linear_speed * max_step_s, rad(max_rotation_lead_deg) + "
                          "max_angular_speed * max_step_s), max_step_s = 1.5 nominal frame periods")
    proprio: str = "[flange xyz (m, base), R[:, 0], R[:, 1] (6D), measured gripper closure (register - 3) / 227]"
    version: str = "rwlearn-action-v1"
    names: tuple = ACTION_NAMES
    lead_m: float | None = None  # recorder lead limits (execution envelope, ``execution_pose``)
    lead_rot_deg: float | None = None

    @classmethod
    def from_teleop(cls, teleop: dict, nominal_period_s: float = NOMINAL_PERIOD_S,
                    max_step_factor: float = 1.5) -> ActionContract:
        """Scales depend only on the teleop limits and the NOMINAL frame period (never on a measured period, so all
        episodes of a dataset share one contract exactly)."""
        step = float(nominal_period_s) * float(max_step_factor)
        return cls(pos_scale_m=float(teleop["max_position_lead"]) + float(teleop["max_linear_speed"]) * step,
                   rot_scale_rad=float(np.deg2rad(teleop["max_rotation_lead_deg"]))
                   + float(teleop["max_angular_speed"]) * step,
                   nominal_period_s=float(nominal_period_s), max_step_s=step,
                   lead_m=float(teleop["max_position_lead"]), lead_rot_deg=float(teleop["max_rotation_lead_deg"]))

    def to_json(self) -> dict:
        d = asdict(self)
        d["names"] = list(self.names)
        return d

    @classmethod
    def from_json(cls, d: dict) -> ActionContract:
        d = dict(d)
        d["names"] = tuple(d.get("names", ACTION_NAMES))
        return cls(**d)

    # ------------------------------------------------------------------------------------------- (de)normalisation
    def normalise(self, phys: np.ndarray) -> np.ndarray:
        """(..., 7) [dp (m), r (rad), c in [0,1]] -> (..., 7) policy actions (in [-1, 1] for recorded commands)."""
        phys = np.asarray(phys, np.float64)
        out = np.empty_like(phys)
        out[..., :3] = phys[..., :3] / self.pos_scale_m
        out[..., 3:6] = phys[..., 3:6] / self.rot_scale_rad
        out[..., 6] = 2.0 * phys[..., 6] - 1.0
        return out

    def denormalise(self, norm: np.ndarray) -> np.ndarray:
        norm = np.clip(np.asarray(norm, np.float64), -1.0, 1.0)
        out = np.empty_like(norm)
        out[..., :3] = norm[..., :3] * self.pos_scale_m
        out[..., 3:6] = norm[..., 3:6] * self.rot_scale_rad
        out[..., 6] = (norm[..., 6] + 1.0) / 2.0
        return out


# ----------------------------------------------------------------------------------------------- pose algebra
def pose7_to_matrix(p: np.ndarray) -> np.ndarray:
    """(..., 7) [x y z qx qy qz qw] -> (..., 4, 4)."""
    p = np.asarray(p, np.float64)
    T = np.zeros((*p.shape[:-1], 4, 4))
    T[..., :3, :3] = Rotation.from_quat(p[..., 3:7].reshape(-1, 4)).as_matrix().reshape(*p.shape[:-1], 3, 3)
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1.0
    return T


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    q = Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(*T.shape[:-2], 4)
    return np.concatenate([T[..., :3, 3], q], -1)


def pose_delta(ref: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """(..., 4, 4) x2 -> (..., 6) [p_goal - p_ref, rotvec(R_goal R_ref^T)] in the base frame."""
    dp = goal[..., :3, 3] - ref[..., :3, 3]
    R = goal[..., :3, :3] @ np.swapaxes(ref[..., :3, :3], -1, -2)
    r = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)
    return np.concatenate([dp, r], -1)


def apply_delta(ref: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Inverse of ``pose_delta``: (..., 4, 4), (..., 6) -> goal (..., 4, 4)."""
    delta = np.asarray(delta, np.float64)
    out = np.array(ref, np.float64, copy=True)
    Rd = Rotation.from_rotvec(delta[..., 3:6].reshape(-1, 3)).as_matrix().reshape(*delta.shape[:-1], 3, 3)
    out[..., :3, :3] = Rd @ ref[..., :3, :3]
    out[..., :3, 3] = ref[..., :3, 3] + delta[..., :3]
    return out


def rotation_6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two columns, column-major [R00 R10 R20 R01 R11 R21]."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def closure_from_register(register) -> np.ndarray:
    return np.clip((np.asarray(register, np.float64) - GRIPPER_OPEN_REGISTER) / GRIPPER_SPAN, 0.0, 1.0)


def register_from_closure(c) -> np.ndarray:
    return np.rint(GRIPPER_OPEN_REGISTER + GRIPPER_SPAN * np.clip(np.asarray(c, np.float64), 0.0, 1.0)).astype(int)


def measured_proprio(measured_pose7: np.ndarray, gripper_register: float) -> np.ndarray:
    """The shared 10-D inference input, in the same order as ``step_table``."""
    T = pose7_to_matrix(np.asarray(measured_pose7, np.float64))
    return np.concatenate([T[:3, 3], rotation_6d(T[:3, :3]), [closure_from_register(gripper_register)]])


# ----------------------------------------------------------------------------------------------- raw recording
@dataclass
class RawEpisode:
    """The arrays of one recording that the action contract needs (read-only copy)."""

    path: str
    task_name: str
    episode_index: int
    robot_t: np.ndarray  # (N,) ns, robot/t
    measured: np.ndarray  # (N, 7) robot/flange_pose
    command: np.ndarray  # (N, 7) action/flange_pose (sent, lead-limited)
    target: np.ndarray  # (N, 7) action/flange_target (operator target, before limiting)
    gripper_command: np.ndarray  # (N,) action/gripper closure
    engaged: np.ndarray  # (N,) bool
    sent: np.ndarray  # (N,) bool
    lead_limited: np.ndarray  # (N,) bool
    segment: np.ndarray  # (N,) int, action/segment
    gripper_t: np.ndarray  # (G,) ns
    gripper_position: np.ndarray  # (G,) raw register
    gripper_segment: np.ndarray  # (G,) recording segment of each gripper sample
    cameras: dict  # serial -> {"t": (F,) ns colour t, "segment": (F,), "paired_colour": stereo/color_index}
    segments: dict  # segments/* arrays (t_start/t_end in ns)
    teleop: dict  # config_yaml teleop block
    T_flange_tcp: np.ndarray
    status: str
    events: list = field(default_factory=list)

    @property
    def serials(self) -> list[str]:
        return sorted(self.cameras)


def read_raw_episode(path: str | Path) -> RawEpisode:
    import h5py
    import yaml

    from s4d.realworld.adapters.panda_teleop import _paired_colour

    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT or int(f.attrs.get("format_version", 0)) != 2:
            raise ValueError(f"{path}: not a {FORMAT} v2 recording")
        a, r, g = f["action"], f["robot"], f["gripper"]
        n = len(r["t"])
        for k in ("flange_pose", "flange_target", "gripper", "engaged", "sent", "segment", "lead_limited"):
            if len(a[k]) != n:
                raise ValueError(f"{path}: action/{k} has {len(a[k])} rows, robot/t has {n}")
        cams = {}
        for name in f["cameras"]:
            if name.startswith("cam_"):
                c = f["cameras"][name]
                cams[name[4:]] = {"t": ns(c["t"][()]), "segment": np.asarray(c["segment"][()], np.int64),
                                  "paired_colour": _paired_colour(c, name[4:])}
        cfg = yaml.safe_load(str(f.attrs["config_yaml"]))
        events = [(float(t), x.decode() if isinstance(x, bytes) else str(x))
                  for t, x in zip(f["events/t"][()], f["events/text"][()])] if "events" in f else []
        return RawEpisode(
            path=str(path), task_name=str(f.attrs.get("task_name")), episode_index=int(f.attrs["episode_index"]),
            robot_t=ns(r["t"][()]), measured=np.asarray(r["flange_pose"][()], np.float64),
            command=np.asarray(a["flange_pose"][()], np.float64), target=np.asarray(a["flange_target"][()], np.float64),
            gripper_command=np.asarray(a["gripper"][()], np.float64), engaged=a["engaged"][()].astype(bool),
            sent=a["sent"][()].astype(bool), lead_limited=a["lead_limited"][()].astype(bool),
            segment=np.asarray(a["segment"][()], np.int64), gripper_t=ns(g["t"][()]),
            gripper_position=np.asarray(g["position"][()], np.float64),
            gripper_segment=np.asarray(g["segment"][()], np.int64), cameras=cams,
            segments={k: (ns(f["segments"][k][()]) if k in ("t_start", "t_end") else np.asarray(f["segments"][k][()]))
                      for k in f["segments"]},
            teleop=dict(cfg["teleop"]), T_flange_tcp=np.asarray(f.attrs["T_flange_tcp"], np.float64),
            status=str(f.attrs.get("status")), events=events)


def camera_frame_indices(raw: RawEpisode, serial: str, times, segment) -> np.ndarray:
    """Store -> raw colour indices, matched EXACTLY on recorder ``cameras/cam_<serial>/t`` rounded to ns.

    The panda adapter retains precisely ``stereo/color_index`` (its own validator is used at read time). Refuse
    other omissions, duplicate/nonmonotonic times and segment disagreement. Old array fixtures without pairing
    metadata retain only the previously supported raw prefix; they cannot justify interior gaps.
    """
    if serial not in raw.cameras:
        raise ValueError(f"{raw.path}/{serial}: store camera is missing in raw recording")
    cam = raw.cameras[serial]
    st, rt = np.asarray(times), np.asarray(cam["t"])
    for label, t in (("store", st), ("raw", rt)):
        if t.ndim != 1 or t.dtype.kind not in "iu" or (t[1:] <= t[:-1]).any():
            raise ValueError(f"{raw.path}/{serial}: {label} timestamps must be integer ns, strictly increasing "
                             "(no duplicates or out-of-order frames)")
    indices = np.searchsorted(rt, st)
    if (indices >= len(rt)).any() or not np.array_equal(rt[indices], st):
        raise ValueError(f"{raw.path}/{serial}: store frame has no exact raw timestamp match")
    expected = cam.get("paired_colour")
    if expected is None:
        expected = np.arange(len(st))  # legacy fixtures: bounded prefix only, no unexplained gaps
    if not np.array_equal(indices, expected):
        raise ValueError(f"{raw.path}/{serial}: store omissions are not explained by stereo/color_index "
                         "(only a raw prefix is allowed without pairing metadata)")
    if not np.array_equal(np.asarray(segment), cam["segment"][indices]):
        raise ValueError(f"{raw.path}/{serial}: store segments differ from mapped raw camera frames")
    return indices.astype(np.int64)


def check_contract_preconditions(raw: RawEpisode) -> None:
    """Refuse recordings the contract does not cover (instead of silently converting them)."""
    if not np.allclose(raw.T_flange_tcp, np.eye(4), atol=1e-9):
        raise ValueError(f"{raw.path}: T_flange_tcp is not the identity; the TCP-vs-flange conversion is not implemented")
    if raw.status != "complete":
        raise ValueError(f"{raw.path}: recording status {raw.status!r} != 'complete'")
    if not np.all(np.diff(raw.robot_t) > 0):
        raise ValueError(f"{raw.path}: robot/t is not strictly increasing")


def latest_at_or_before(times: np.ndarray, query: np.ndarray, strict: bool = False) -> np.ndarray:
    """Index of the last ``times`` entry <= query (< query if ``strict``); -1 when none."""
    return np.searchsorted(times, query, side="left" if strict else "right") - 1


@dataclass
class StepTable:
    """Per observation k on one camera timeline (``times[k]`` = frame time tau_k)."""

    times: np.ndarray  # (K,) ns
    segment: np.ndarray  # (K,) camera segment id
    state_ok: np.ndarray  # (K,) bool: a fresh measured state exists at tau_k in the same segment
    proprio: np.ndarray  # (K, 10) float32 (NaN where not state_ok)
    measured_pose: np.ndarray  # (K, 4, 4) measured flange pose at tau_k (reference of a_k)
    transition_ok: np.ndarray  # (K,) bool: k -> k+1 is a valid policy step (last entry always False)
    action_phys: np.ndarray  # (K, 7) [dp, r, c] (NaN where not transition_ok)
    command_pose: np.ndarray  # (K, 4, 4) the aggregated sent command of step k (goal)
    reasons: dict  # invalid-transition reason -> count


def step_table(raw: RawEpisode, times: np.ndarray, segment: np.ndarray, *, nominal_period_s: float,
               max_step_factor: float = 1.5, max_state_age_ns: int = 15_000_000,
               max_gripper_age_ns: int = MAX_GRIPPER_AGE_NS) -> StepTable:
    """Observations at ``times`` (a camera's frame times, ns, with that camera's segment ids) -> proprio, actions and
    validity, following the module contract. A step k -> k+1 is valid iff both frames are in one recording segment,
    tau_{k+1} - tau_k <= max_step_factor * nominal period, a measured state at most ``max_state_age_ns`` old exists at
    tau_k in that segment (and a gripper register sample at most ``max_gripper_age_ns`` old in that segment), at
    least one command row lies in (tau_k, tau_{k+1}), and every command row in [tau_k, tau_{k+1}) has
    sent == engaged == 1 in the same segment."""
    times, segment = np.asarray(times, np.int64), np.asarray(segment, np.int64)
    K = len(times)
    T_meas = pose7_to_matrix(raw.measured)
    T_cmd = pose7_to_matrix(raw.command)
    i_state = latest_at_or_before(raw.robot_t, times)
    state_ok = (i_state >= 0)
    safe = np.clip(i_state, 0, None)
    state_ok &= (times - raw.robot_t[safe] <= max_state_age_ns) & (raw.segment[safe] == segment)
    gi = latest_at_or_before(raw.gripper_t, times)
    gsafe = np.clip(gi, 0, None)
    grip_meas = closure_from_register(raw.gripper_position[gsafe])
    grip_ok = (gi >= 0) & (times - raw.gripper_t[gsafe] <= max_gripper_age_ns) & (raw.gripper_segment[gsafe] == segment)
    measured = T_meas[safe]
    proprio = np.concatenate([measured[:, :3, 3], rotation_6d(measured[:, :3, :3]), grip_meas[:, None]], 1)
    proprio = np.where((state_ok & grip_ok)[:, None], proprio, np.nan).astype(np.float32)
    state_ok &= grip_ok

    reasons = {"segment_break": 0, "long_step": 0, "no_state": 0, "no_command": 0, "not_sent_or_engaged": 0}
    ok = np.zeros(K, bool)
    act = np.full((K, ACTION_DIM), np.nan)
    goal = np.tile(np.eye(4), (K, 1, 1))
    max_step = max_step_factor * nominal_period_s * 1e9
    for k in range(K - 1):
        if segment[k] != segment[k + 1]:
            reasons["segment_break"] += 1
            continue
        if times[k + 1] - times[k] > max_step:
            reasons["long_step"] += 1
            continue
        if not state_ok[k]:
            reasons["no_state"] += 1
            continue
        lo = np.searchsorted(raw.robot_t, times[k], side="left")
        hi = np.searchsorted(raw.robot_t, times[k + 1], side="left")  # rows with t < tau_{k+1}
        rows = np.arange(lo, hi)
        if len(rows) == 0 or raw.robot_t[hi - 1] <= times[k]:
            reasons["no_command"] += 1
            continue
        if not (raw.sent[rows].all() and raw.engaged[rows].all() and (raw.segment[rows] == segment[k]).all()):
            reasons["not_sent_or_engaged"] += 1
            continue
        last = hi - 1
        goal[k] = T_cmd[last]
        act[k, :6] = pose_delta(measured[k], T_cmd[last])
        act[k, 6] = raw.gripper_command[last]
        ok[k] = True
    return StepTable(times=times, segment=segment, state_ok=state_ok, proprio=proprio, measured_pose=measured,
                     transition_ok=ok, action_phys=act, command_pose=goal, reasons=reasons)


def nominal_period_s(raw: RawEpisode, serial: str) -> float:
    t = raw.cameras[serial]["t"]
    seg = raw.cameras[serial]["segment"]
    d = np.diff(t)[np.diff(seg) == 0]
    return float(np.median(d)) / 1e9


def contract_for(raw: RawEpisode) -> ActionContract:
    """The contract (nominal 15 Hz period) after checking every camera's measured median period against it."""
    for serial in raw.serials:
        measured = nominal_period_s(raw, serial)
        if abs(measured - NOMINAL_PERIOD_S) > PERIOD_TOLERANCE * NOMINAL_PERIOD_S:
            raise ValueError(f"{raw.path}: camera {serial} frame period {measured:.5f} s is not the nominal "
                             f"{NOMINAL_PERIOD_S:.5f} s")
    return ActionContract.from_teleop(raw.teleop)


def to_robot_command(measured_pose7: np.ndarray, action_norm: np.ndarray, contract: ActionContract, *,
                     binary_gripper: bool = False) -> dict:
    """Policy output -> the command in the recorder's convention (NO robot I/O): the flange goal pose in the base frame
    ``[x y z qx qy qz qw]`` (what ``action/flange_pose`` records; the EXACT inverse of the dataset conversion), the
    gripper closure and its Robotiq register, and ``execution_flange_pose``: that goal passed through the recorder's
    radial lead envelope (``limit_lead``; the per-dimension normalisation box is wider than the radial lead, e.g. an
    all-ones translation has norm sqrt(3) x pos_scale). Speed limits act over time and belong to the robot-side
    executor, which must keep the recorder's safety layer.

    ``measured_pose7`` is the measured flange pose at the observation time (``robot/flange_pose`` convention)."""
    phys = contract.denormalise(action_norm)
    ref = pose7_to_matrix(np.asarray(measured_pose7, np.float64))
    goal = apply_delta(ref, phys[..., :6])
    c = phys[..., 6]
    if binary_gripper:
        c = (c >= 0.5).astype(np.float64)
    out = {"flange_pose": matrix_to_pose7(goal), "gripper_closure": c, "gripper_register": register_from_closure(c),
           "delta_phys": phys[..., :6], "frame": contract.frame, "point": contract.point}
    if contract.lead_m is not None:  # the recorder's radial lead envelope (the normalisation box is wider)
        out["execution_flange_pose"] = limit_lead(measured_pose7, out["flange_pose"],
                                                  {"max_position_lead": contract.lead_m,
                                                   "max_rotation_lead_deg": contract.lead_rot_deg})
    return out


def limit_lead(measured_pose7: np.ndarray, goal_pose7: np.ndarray, teleop: dict) -> np.ndarray:
    """The recorder's lead limit as inferred from the data: translation of the goal clamped to
    ``max_position_lead`` around the measured position, rotation to ``max_rotation_lead_deg`` (geodesic slerp)."""
    ref, goal = pose7_to_matrix(measured_pose7), pose7_to_matrix(goal_pose7)
    d = pose_delta(ref, goal)
    n = np.linalg.norm(d[..., :3], axis=-1, keepdims=True)
    lim = float(teleop["max_position_lead"])
    d[..., :3] = np.where(n > lim, d[..., :3] / np.maximum(n, 1e-12) * lim, d[..., :3])
    a = np.linalg.norm(d[..., 3:6], axis=-1, keepdims=True)
    alim = np.deg2rad(float(teleop["max_rotation_lead_deg"]))
    d[..., 3:6] = np.where(a > alim, d[..., 3:6] / np.maximum(a, 1e-12) * alim, d[..., 3:6])
    return matrix_to_pose7(apply_delta(ref, d))


def contract_json(contract: ActionContract) -> str:
    return json.dumps(contract.to_json(), sort_keys=True)


# ----------------------------------------------------------------------------------------------- fixtures
_ARRAY_FIELDS = ("robot_t", "measured", "command", "target", "gripper_command", "engaged", "sent", "lead_limited",
                 "segment", "gripper_t", "gripper_position", "gripper_segment", "T_flange_tcp")


def raw_to_arrays(raw: RawEpisode) -> dict:
    """The action-relevant arrays of a recording (no images), e.g. for a compact test fixture."""
    out = {k: getattr(raw, k) for k in _ARRAY_FIELDS}
    for serial, cam in raw.cameras.items():
        out[f"camera/{serial}/t"], out[f"camera/{serial}/segment"] = cam["t"], cam["segment"]
        if "paired_colour" in cam:
            out[f"camera/{serial}/paired_colour"] = cam["paired_colour"]
    out["segments/t_start"], out["segments/t_end"] = raw.segments["t_start"], raw.segments["t_end"]
    out["meta"] = np.frombuffer(json.dumps({"path": raw.path, "task_name": raw.task_name, "status": raw.status,
                                            "episode_index": raw.episode_index, "teleop": raw.teleop}).encode(),
                                np.uint8)
    return out


def raw_from_arrays(z) -> RawEpisode:
    meta = json.loads(bytes(np.asarray(z["meta"])).decode())
    cams = {}
    for k in z.keys() if hasattr(z, "keys") else z.files:
        if k.startswith("camera/") and k.endswith("/t"):
            serial = k.split("/")[1]
            cams[serial] = {"t": np.asarray(z[k]), "segment": np.asarray(z[f"camera/{serial}/segment"])}
            if f"camera/{serial}/paired_colour" in z:
                cams[serial]["paired_colour"] = np.asarray(z[f"camera/{serial}/paired_colour"])
    return RawEpisode(path=meta["path"], task_name=meta["task_name"], episode_index=int(meta["episode_index"]),
                      cameras=cams, segments={"t_start": np.asarray(z["segments/t_start"]),
                                              "t_end": np.asarray(z["segments/t_end"])},
                      teleop=meta["teleop"], status=meta["status"],
                      **{k: np.asarray(z[k]) for k in _ARRAY_FIELDS})


def load_raw(path) -> RawEpisode:
    """A recording (``.h5``, read-only) or a compact ``raw_to_arrays`` fixture (``.npz``)."""
    if str(path).endswith(".npz"):
        return raw_from_arrays(np.load(path))
    return read_raw_episode(path)
