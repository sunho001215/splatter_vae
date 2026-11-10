"""Live M5 evidence for multi-view deployment (robot-PC side, CPU, no torch): every synchronised camera set's
robot-surface displacement computed with the TRAINING rule of the compact training store
(``s4d.realworld.train_store``):

* sets = ``train_store.mutual_sync`` (mutual nearest frame of every camera to the reference = first training camera,
  no time cap), emitted as soon as no later frame can change them (``LiveSets``);
* the robot state of every member frame = ``pipeline.robot_state_at`` at the frame's timestamp over the recorded
  robot rows, with the gripper register first interpolated onto the robot row times exactly as the
  ``panda_teleop_h5`` adapter does (``StateTimeline``);
* the displacement = ``train_store.set_surface_motion`` (largest displacement of any URDF surface point between two
  members; inf when a member has no valid state) over the training store's legacy M5 surface:
  ``pipeline.RobotGeometry(meta)`` = the composed URDF parsed by ``URDFModel``, ``RobotSurface.build`` (300 points per
  link, seed 0) and the legacy gripper map (``SurfaceMotion``).

The surface points are sampled ONCE where the robot meshes exist (``export_surface``, development host) and shipped as
a small asset next to the UNMODIFIED URDF; the robot PC poses them with the authoritative ``URDFModel.fk``
(``RobotSurface.points``), so no mesh and no re-implemented kinematics are needed there. Depends on numpy, scipy and
h5py (through the authoritative modules); never on torch. A set is accepted iff its displacement is <= the trained
threshold (``train_store.MAX_SET_MOTION_M``, 5 mm); the deploy history enforces it (``ObservationWindow.validate``).
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s4d.rwlearn.store import MAX_SET_MOTION_M

FORMAT = "s4d.rwlearn.deploy.m5_surface/1"


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_surface(robot, out_path) -> dict:
    """Development host only (needs the URDF's meshes): sample the training store's M5 surface with
    ``pipeline.RobotGeometry(SimpleNamespace(robot=robot))`` exactly as ``train_store.write_episode`` builds its legacy
    geometry, and write one ``.npz`` (link-frame points per link + a JSON header with the URDF sha256 and the gripper
    mapping). ``robot``: the recording's ``RobotDescription`` (the adapter's ``robot`` config). Returns the header."""
    from s4d.realworld.pipeline import RobotGeometry

    geom = RobotGeometry(SimpleNamespace(robot=robot))
    links = sorted(geom.surface.link_points)
    r = geom.desc
    header = {"format": FORMAT, "urdf_sha256": _sha256(r.urdf_path), "urdf_name": Path(r.urdf_path).name,
              "urdf_parse": "URDFModel.from_file (use='visual', as RobotGeometry)",
              "surface": geom.surface_kind, "gripper_map": geom.gripper_map, "closed_value": geom.closed_value,
              "points_per_link": 300, "seed": 0, "links": links,
              "points": int(sum(len(geom.surface.link_points[k]) for k in links)),
              "joint_names": list(r.joint_names), "gripper_joint": r.gripper_joint, "gripper_model": r.gripper_model,
              "gripper_state_semantics": r.gripper_state_semantics,
              "rule": "train_store.write_episode legacy RobotGeometry(meta): M5 set motion"}
    arrays = {"header": np.frombuffer(json.dumps(header, sort_keys=True).encode(), np.uint8),
              **{f"link{i}": np.asarray(geom.surface.link_points[k], np.float64) for i, k in enumerate(links)}}
    _write_npz(out_path, arrays)
    return header


def _write_npz(path, arrays: dict) -> None:
    """``np.load``-compatible .npz with fixed member timestamps (byte-reproducible asset, unlike ``np.savez``)."""
    import io
    import zipfile

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(a), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


class SurfaceMotion:
    """The M5 robot-surface model posed with the authoritative URDF FK (``RobotSurface.points``)."""

    def __init__(self, asset_path, urdf_path):
        from s4d.realworld.robotics.model import RobotSurface, URDFModel

        with np.load(asset_path) as z:
            self.header = json.loads(bytes(z["header"]).decode())
            if self.header.get("format") != FORMAT:
                raise ValueError(f"{asset_path}: not a {FORMAT} asset")
            points = {k: np.array(z[f"link{i}"]) for i, k in enumerate(self.header["links"])}
        if _sha256(urdf_path) != self.header["urdf_sha256"]:
            raise ValueError(f"{urdf_path}: URDF sha256 differs from the one the M5 surface was sampled on")
        self.asset_sha256, self.urdf_path = _sha256(asset_path), str(urdf_path)
        self.surface = RobotSurface(URDFModel.from_file(urdf_path), points)

    def q(self, q7, gripper_state) -> dict:
        """Joint values of a recorded state: ``RobotGeometry.q`` with the asset's (legacy) gripper map."""
        from s4d.realworld.robotics.model import gripper_joint_value, joint_dict

        h = self.header
        value = gripper_joint_value(gripper_state, h["gripper_state_semantics"], h["gripper_model"],
                                    closed_value=h["closed_value"], mapping=h["gripper_map"])
        return joint_dict(h["joint_names"], q7, h["gripper_joint"], value)

    def points(self, state) -> np.ndarray:
        """(P, 3) base-frame surface points of ``state`` = (q7, gripper state)."""
        return self.surface.points(self.q(state[0], state[1]))[0]

    def displacement(self, states: list, ok: list) -> float:
        """``train_store.set_surface_motion`` of ONE set: member states (q7, gripper) and their validity."""
        from s4d.realworld.train_store import set_surface_motion

        frames = np.zeros((1, len(states)), np.int64)
        return float(set_surface_motion(self.points, frames, [[s] for s in states], [[bool(o)] for o in ok])[0])


class StateTimeline:
    """Robot rows (time, 7 joint positions) and Robotiq register samples on the robot PC's monotonic clock (ns).

    State at frame times = ``pipeline.robot_state_at`` over the rows (joints and gripper interpolated; valid iff inside
    the rows and the nearest row is <= ``max_sync_ns`` away, the training store's ``robot_sync_max_ns``), with the
    gripper register on the rows = ``np.interp(row_t_s, gripper_t_s, register)`` as the adapter computes it. ``ready``
    tells whether that interpolation is already final (rows after the frame, register samples after those rows),
    i.e. equal to what the recorder's complete arrays give."""

    def __init__(self, max_sync_ns: int, capacity: int = 4096):
        self.max_sync_ns = int(max_sync_ns)
        self._rows = collections.deque(maxlen=int(capacity))  # (t_ns, q7 tuple)
        self._grip = collections.deque(maxlen=int(capacity))  # (t_ns, register)

    def add_robot(self, t_ns: int, q7) -> None:
        if self._rows and int(t_ns) <= self._rows[-1][0]:
            raise ValueError("robot rows must arrive in increasing time")
        q = np.asarray(q7, np.float64).reshape(7)
        self._rows.append((int(t_ns), q))

    def add_gripper(self, t_ns: int, register: float) -> None:
        if self._grip and int(t_ns) <= self._grip[-1][0]:
            raise ValueError("gripper samples must arrive in increasing time")
        self._grip.append((int(t_ns), float(register)))

    def ready(self, t_ns: int) -> bool:
        if not self._rows or not self._grip or self._rows[-1][0] < t_ns:
            return False
        after = next(t for t, _ in self._rows if t >= t_ns)  # the row closing the frame's interpolation interval
        return self._grip[-1][0] >= after

    def states(self, stamps_ns) -> tuple[list, np.ndarray]:
        from s4d.realworld.pipeline import robot_state_at

        stamps = np.asarray(stamps_ns, np.int64)
        if not self._rows or not self._grip:
            return [None] * len(stamps), np.zeros(len(stamps), bool)
        t = np.array([r[0] for r in self._rows], np.int64)
        q = np.stack([r[1] for r in self._rows])
        gt = np.array([g[0] for g in self._grip], np.float64) / 1e9
        gp = np.array([g[1] for g in self._grip], np.float64)
        trajectory = SimpleNamespace(timestamps_ns=t, joint_positions=q, gripper_state=np.interp(t / 1e9, gt, gp))
        qs, gs, inside, _ = robot_state_at(SimpleNamespace(trajectory=trajectory), stamps, self.max_sync_ns)
        return list(zip(qs, gs)), np.asarray(inside, bool)


class LiveSets:
    """Mutual-nearest camera sets (``train_store.mutual_sync`` over the buffered frames, reference = first camera) of
    free-running cameras, each emitted once and only when no later frame can change it:

    * nearest decided: every other camera holds a frame at or after the reference frame (later frames are farther),
      or its newest frame lies less than h = (1 - tolerance) x period / 2 BEFORE it (the camera's next frame of its
      regular timeline is at least (1 - tolerance) periods later, so it cannot be nearer);
    * back-check decided: a member after the reference frame is either followed by a buffered reference frame, or
      lies less than h after it (same argument on the reference timeline). Ties go to the earlier frame as in
      training (``nearest_timestamps``). Validated against stored ``sync/frames`` (scripts/realworld_m5_asset.py).
    Both shortcuts rely on every camera's frame interval being >= (1 - tolerance) periods (the trained timing, which
    the deploy history enforces per window); the first shorter interval switches them off for good (``early`` False:
    exact waiting for bracketing frames from then on) and is counted in ``irregular_intervals``. Buffers keep the
    last ``capacity`` frames per camera (about one second), far more than a pending decision needs.
    """

    def __init__(self, cameras, period_s: float, tolerance: float = 0.01, capacity: int = 16):
        self.cameras = [str(c) for c in cameras]
        self._half_ns = (1.0 - float(tolerance)) * float(period_s) * 1e9 / 2.0
        self._frames = {c: collections.deque(maxlen=int(capacity)) for c in self.cameras}  # (seq, t_ns)
        self._done = -1  # newest reference seq already emitted (or given up)
        self._min_interval_ns = (1.0 - float(tolerance)) * float(period_s) * 1e9
        self.early, self.irregular_intervals = True, 0

    def add(self, camera_id: str, seq: int, t_ns: int) -> None:
        buf = self._frames[str(camera_id)]
        if buf and (seq <= buf[-1][0] or t_ns <= buf[-1][1]):
            raise ValueError(f"{camera_id}: frames must arrive in increasing seq and time")
        if buf and t_ns - buf[-1][1] < self._min_interval_ns:
            self.irregular_intervals += 1
            self.early = False  # the regular-timeline shortcuts no longer hold
        buf.append((int(seq), int(t_ns)))

    def pop_final(self, *, end_of_stream: bool = False) -> list[dict]:
        """Newly final sets, oldest first: {"frame_ids": {cam: seq}, "times_ns": {cam: t}, "complete": bool}.
        ``end_of_stream``: no frame will follow (a recording's end): every buffered reference frame is final."""
        from s4d.realworld.train_store import mutual_sync

        ref = list(self._frames[self.cameras[0]])
        others = [list(self._frames[c]) for c in self.cameras[1:]]
        if not ref or any(not o for o in others):
            return []
        stamps = [np.array([t for _, t in ref], np.int64)] + [np.array([t for _, t in o], np.int64) for o in others]
        frames, _ = mutual_sync(stamps)
        out = []
        for k, (seq, t_ref) in enumerate(ref):
            if seq <= self._done:
                continue
            if not end_of_stream and any(o[-1][1] < t_ref and (not self.early or t_ref - o[-1][1] >= self._half_ns)
                                         for o in others):
                break  # nearest not decided yet for this and every later reference frame
            row = frames[k]
            members = [others[v][int(j)][1] for v, j in enumerate(row[1:]) if j >= 0]
            if not end_of_stream and any(tm > t_ref and (not self.early or tm - t_ref >= self._half_ns)
                                         and ref[-1][1] <= tm for tm in members):
                break  # back-check needs the next reference frame
            self._done = seq
            ids = {self.cameras[0]: seq}
            times = {self.cameras[0]: t_ref}
            for v, j in enumerate(row[1:]):
                if j >= 0:
                    ids[self.cameras[v + 1]], times[self.cameras[v + 1]] = others[v][int(j)]
            out.append({"frame_ids": ids, "times_ns": times, "complete": len(ids) == len(self.cameras)})
        return out


class LiveM5:
    """Sets + state timeline + surface: ``add_frame`` / ``add_robot`` / ``add_gripper`` as data arrives, ``poll`` returns
    the final complete sets whose member states are final: {"frame_ids", "times_ns", "surface_disp_m",
    "observation_time_ns" (latest member)}. Incomplete sets (a camera without a mutual partner) are counted and
    dropped, as the training loaders drop them."""

    def __init__(self, motion: SurfaceMotion, cameras, period_s: float, max_sync_ns: int, *, tolerance: float = 0.01):
        self.motion = motion
        self.sets = LiveSets(cameras, period_s, tolerance)
        self.timeline = StateTimeline(max_sync_ns)
        self._pending: collections.deque = collections.deque()
        self.counts = collections.Counter()

    def add_frame(self, camera_id: str, seq: int, t_ns: int) -> None:
        self.sets.add(camera_id, seq, t_ns)

    def add_robot(self, t_ns: int, q7) -> None:
        self.timeline.add_robot(t_ns, q7)

    def add_gripper(self, t_ns: int, register: float) -> None:
        self.timeline.add_gripper(t_ns, register)

    def poll(self, *, end_of_stream: bool = False) -> list[dict]:
        """``end_of_stream``: no frame or robot sample will follow (a recording's end): evaluate what is pending with
        the data at hand (frames after the last robot row get no valid state, as in training)."""
        for s in self.sets.pop_final(end_of_stream=end_of_stream):
            if s["complete"]:
                self._pending.append(s)
            else:
                self.counts["incomplete_sets"] += 1
        out = []
        while self._pending and (end_of_stream or all(self.timeline.ready(t)
                                                      for t in self._pending[0]["times_ns"].values())):
            s = self._pending.popleft()
            cams = list(s["times_ns"])
            states, ok = self.timeline.states([s["times_ns"][c] for c in cams])
            disp = self.motion.displacement(states, ok)
            self.counts["sets"] += 1
            self.counts["rejected_motion" if disp > MAX_SET_MOTION_M else "accepted"] += 1
            out.append({**s, "surface_disp_m": disp, "observation_time_ns": max(s["times_ns"].values())})
        return out
