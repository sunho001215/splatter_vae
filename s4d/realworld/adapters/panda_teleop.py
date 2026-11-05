"""Adapter for ``panda_teleop_episode`` HDF5 recordings (format_version 2): Franka arm, Robotiq gripper and Intel
RealSense D400 cameras recording colour (JPEG) + depth (PNG z16) per device.

Everything comes from the file; nothing is invented:
- streams, intrinsics, distortion model, depth scale and the depth->colour extrinsic from each camera's
  ``calibration_json`` (librealsense active stream profiles of the running pipeline);
- timestamps from ``t`` (colour) and ``stereo/t`` (depth): the device hardware/global-time stamp mapped onto host
  CLOCK_MONOTONIC (its frame-to-frame steps equal ``hw_timestamp_ms`` steps; arrival ``t_recv`` is ~65 ms later);
- depth frame k is the frame the recorder paired with colour frame k (``stereo/color_index == k``, offset recorded in
  ``stereo/color_offset_ms``);
- robot joints, flange poses (``panda_link8``, the recorded EE frame) and the Robotiq position register
  (``gripper/position``, raw 0..255, its own ~60 Hz clock -> interpolated onto robot times; recorded as such).
The file holds NO camera-to-robot extrinsics (root attribute ``camera_calibration`` is empty): those come from the
robot-grounded ``calibrate`` stage. Adapter-level guesses are listed in ``EpisodeMeta.assumptions``.

Config::

    episodes: {<episode_id>: /path/episode_000000.h5}
    cameras: [<serial>, ...]          # optional subset, default all ``cameras/cam_<serial>`` groups
    robot: {urdf_path, gripper_model, gripper_joint, base_frame, flange_frame, joint_names, arm_note}
    include_ir: false                 # true: also the device's left/right infrared pair (``stereo/left|right``,
                                      # recorders that store them), as ``<serial>/ir_left|ir_right`` streams + an
                                      # ``IrStereoGeometry`` (device-rectified; extrinsics from calibration_json)
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from s4d.realworld.adapters._robot_manifest import _composed_robot
from s4d.realworld.adapters.base import EpisodeAdapter, register_adapter
from s4d.realworld.contract import (
    CameraStream,
    EpisodeMeta,
    Intrinsics,
    IrStereoGeometry,
    Provenance,
    RgbdGeometry,
    RobotDescription,
    RobotTrajectory,
    SourceFile,
)

FORMAT = "panda_teleop_episode"
DISTORTION = {"brown_conrady": "realsense_brown_conrady", "inverse_brown_conrady": "realsense_inverse_brown_conrady",
              "modified_brown_conrady": "realsense_modified_brown_conrady", "none": "none"}
PAIR_TOLERANCE_NS = 1_000_000  # declared colour/depth pairing bound (recorded offsets are ~0.04 ms)


def _ns(seconds: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(seconds, np.float64) * 1e9).astype(np.int64)


def _intrinsics(d: dict, source: str) -> Intrinsics:
    model = DISTORTION.get(d["distortion_model"])
    if model is None:
        raise ValueError(f"unknown RealSense distortion model {d['distortion_model']!r}")
    coeffs = tuple(float(c) for c in d["distortion_coeffs"])
    if model == "none":
        coeffs = ()
    return Intrinsics(width=int(d["width"]), height=int(d["height"]), fx=float(d["fx"]), fy=float(d["fy"]),
                      cx=float(d["ppx"]), cy=float(d["ppy"]), pixel_convention="integer_centers",
                      distortion_model=model, distortion=coeffs, source=source)


def _extrinsic(cal: dict, name: str, source: str, target: str) -> np.ndarray:
    e = cal["extrinsics"][name]
    if (e["source"], e["target"]) != (source, target):
        raise ValueError(f"unexpected extrinsic direction {e['source']}->{e['target']} for {name}")
    T = np.eye(4)
    T[:3, :3] = np.asarray(e["R_target_from_source"], float)
    T[:3, 3] = np.asarray(e["t_target_from_source_m"], float)
    return T


def _pose_matrix(p: np.ndarray) -> np.ndarray:
    """(T,7) [x y z qx qy qz qw] -> (T,4,4)."""
    from scipy.spatial.transform import Rotation

    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(p[:, 3:7]).as_matrix()
    T[:, :3, 3] = p[:, :3]
    return T


@register_adapter
class PandaTeleopH5Adapter(EpisodeAdapter):
    name = "panda_teleop_h5"
    version = "1"

    def __init__(self, config: dict):
        super().__init__(config)
        self.episodes = {k: Path(v) for k, v in config["episodes"].items()}

    def list_episodes(self) -> list[str]:
        return sorted(self.episodes)

    def source_files(self, episode_id: str) -> list[Path]:
        return [self.episodes[episode_id]]

    def _open(self, episode_id: str):
        import h5py

        f = h5py.File(self.episodes[episode_id], "r")
        if f.attrs.get("format") != FORMAT or int(f.attrs.get("format_version", 0)) != 2:
            f.close()
            raise ValueError(f"{episode_id}: not a {FORMAT} v2 file")
        return f

    def _serials(self, f) -> list[str]:
        found = sorted(k[len("cam_"):] for k in f["cameras"] if k.startswith("cam_"))
        wanted = self.config.get("cameras")
        if wanted:
            missing = sorted(set(map(str, wanted)) - set(found))
            if missing:
                raise ValueError(f"cameras {missing} not in the recording ({found})")
            return [s for s in found if s in set(map(str, wanted))]
        return found

    def read_meta(self, episode_id: str) -> EpisodeMeta:
        with self._open(episode_id) as f:
            streams, rgbd, stamps, notes, ir_pairs = [], [], {}, {}, []
            for serial in self._serials(f):
                g = f["cameras"][f"cam_{serial}"]
                cal = json.loads(g.attrs["calibration_json"])
                if cal["device"]["serial_number"] != serial:
                    raise ValueError(f"cam_{serial}: calibration belongs to {cal['device']['serial_number']}")
                color, depth = cal["streams"]["color"], cal["streams"]["depth"]
                src = f"librealsense {cal['device']['sdk_version']} active profile (calibration_id {cal['calibration_id']})"
                if not (g["t_valid"][()].all() and g["stereo/t_valid"][()].all() and g["stereo/depth_ok"][()].all()):
                    raise ValueError(f"cam_{serial}: invalid timestamps or depth frames flagged by the recorder")
                if not (g["stereo/color_index"][()] == np.arange(len(g["t"]))).all():
                    raise ValueError(f"cam_{serial}: depth frames are not paired 1:1 with colour frames")
                emitter = cal.get("emitter", {})
                common = dict(rig_id=serial, eye="mono", rectified=True, device_serial=serial,
                              timestamp_source="device global-time stamp mapped to host CLOCK_MONOTONIC (recorder t)")
                cid, did = f"{serial}/color", f"{serial}/depth"
                streams.append(CameraStream(
                    stream_id=cid, modality="rgb", channels=3, channel_order="RGB", dtype="uint8",
                    intrinsics=_intrinsics(color["intrinsics"], src), sensor_name=f"{cal['device']['name']} RGB",
                    notes=f"recorded {color['format']} {color['width']}x{color['height']}@{color['fps']} as JPEG; "
                          "decoded to RGB; zero distortion coefficients (pinhole, no rectification needed)", **common))
                streams.append(CameraStream(
                    stream_id=did, modality="depth", channels=1, channel_order="DEPTH", dtype="uint16",
                    intrinsics=_intrinsics(depth["intrinsics"], src), sensor_name=f"{cal['device']['name']} depth",
                    ir_emitter="on" if emitter.get("emitter_enabled") else "off", depth_scale_m=float(cal["depth_scale"]),
                    depth_kind="z", notes=f"{depth['format']} PNG16, laser power {emitter.get('laser_power')}", **common))
                e = cal["extrinsics"]["color_from_depth"]
                if (e["source"], e["target"]) != ("depth", "color"):
                    raise ValueError(f"cam_{serial}: unexpected extrinsic direction {e['source']}->{e['target']}")
                T = np.eye(4)
                T[:3, :3] = np.asarray(e["R_target_from_source"], float)
                T[:3, 3] = np.asarray(e["t_target_from_source_m"], float)
                stamps[cid], stamps[did] = _ns(g["t"][()]), _ns(g["stereo/t"][()])
                offset_ns = np.abs(stamps[cid] - stamps[did])
                rgbd.append(RgbdGeometry(
                    rig_id=serial, color_stream=cid, depth_stream=did, status="metric",
                    T_color_depth=tuple(map(tuple, T)), max_pair_offset_ns=PAIR_TOLERANCE_NS,
                    pairing=f"recorder pairing stereo/color_index == k (max |t_color - t_depth| {int(offset_ns.max())} ns)",
                    source=f"{src}: extrinsics color_from_depth ({cal['extrinsics_convention']})"))
                notes[serial] = {"frames": int(len(g["t"])), "segments": np.bincount(g["segment"][()]).tolist(),
                                 "firmware": cal["device"]["firmware_version"]}
                if self.config.get("include_ir"):
                    ir_pairs.append(self._ir_pair(g, serial, cal, src, common, emitter, stamps))
                    streams += ir_pairs[-1][1]
                    ir_pairs[-1] = ir_pairs[-1][0]
            trajectory = self._trajectory(f)
            rc = self.config.get("robot", {})
            if rc.get("urdf_path"):  # composed arm + gripper URDF (scripts/realworld_compose_robot_urdf.py)
                robot, _ = _composed_robot(rc, gripper_state_semantics="robotiq_position_register")
                trajectory.ee_frame = rc["flange_frame"]
            else:
                robot = RobotDescription(arm_model="franka_panda", gripper_model="unknown", urdf_path=None,
                                         mesh_root=None, gripper_state_semantics="robotiq_position_register")
                trajectory.ee_frame = robot.flange_frame
            assumptions = [
                "colour JPEG (recorded from bgr8, quality 90) decoded as RGB; checked visually on the first episode",
                "RealSense intrinsics use librealsense's integer-centre pixel convention",
                f"arm: recorded frames panda_link0/panda_link8; URDF {rc.get('urdf_path')} - "
                f"{rc.get('arm_note', 'arm generation not recorded in the file')}",
                f"gripper: {robot.gripper_model} (file convention 'Robotiq position 3 + 227 * closure (DROID)'); "
                f"mount: {robot.mount_source}",
                "gripper state: raw Robotiq position register (gripper/position, ~60 Hz) linearly interpolated onto "
                "robot/t; register -> finger joint mapping linear (s4d.realworld.robot.gripper_joint_value)",
                "robot/t is the state time on host CLOCK_MONOTONIC (file conventions_json 'time')",
            ]
            prov = Provenance(
                adapter=self.name, adapter_version=self.version, source_root=str(self.episodes[episode_id].parent),
                source_files=tuple(SourceFile(str(p), "", p.stat().st_size) for p in self.source_files(episode_id)),
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                notes=json.dumps({"format": FORMAT, "software_version": str(f.attrs.get("software_version")),
                                  "task": str(f.attrs.get("task_name")), "hostname": str(f.attrs.get("hostname")),
                                  "status": str(f.attrs.get("status")), "cameras": notes,
                                  "segments": {"t_start": f["segments/t_start"][()].tolist(),
                                               "t_end": f["segments/t_end"][()].tolist()},
                                  "conventions": json.loads(f.attrs["conventions_json"]),
                                  "T_flange_controller_ee": np.asarray(f.attrs["T_flange_controller_ee"]).tolist(),
                                  "T_flange_tcp": np.asarray(f.attrs["T_flange_tcp"]).tolist(),
                                  "camera_calibration_attr": str(f.attrs.get("camera_calibration"))}))
            if ir_pairs:
                assumptions.append("infrared pairs: librealsense Y8 left/right streams of each D4xx, rectified on the "
                                   "device (identical K, zero distortion, R = I); IR frame k = depth frame k (same "
                                   "stereo record, equal hardware timestamps checked); emitter state as recorded")
            return EpisodeMeta(episode_id=episode_id, streams=tuple(streams), stereo=(), robot=robot,
                               trajectory=trajectory, stream_timestamps_ns=stamps, provenance=prov,
                               clock=f"host CLOCK_MONOTONIC of the recorder ({f.attrs.get('hostname')})",
                               assumptions=tuple(assumptions), rgbd=tuple(rgbd), ir_stereo=tuple(ir_pairs))

    @staticmethod
    def _ir_pair(g, serial: str, cal: dict, src: str, common: dict, emitter: dict, stamps: dict):
        """(IrStereoGeometry, [left stream, right stream]) from the recorded stereo group; refuses unverifiable data."""
        st = g["stereo"]
        if "left" not in st or "right" not in st:
            raise ValueError(f"cam_{serial}: include_ir but the recording has no stereo/left|right infrared frames")
        if not (st["pair_ok"][()].all() and st["t_valid"][()].all()):
            raise ValueError(f"cam_{serial}: incomplete or invalid infrared pairs flagged by the recorder")
        hw_l, hw_r, hw_d = (st[k][()] for k in ("hw_timestamp_ms", "right_hw_timestamp_ms", "depth_hw_timestamp_ms"))
        if not (np.array_equal(hw_l, hw_r) and np.array_equal(hw_l, hw_d)):
            raise ValueError(f"cam_{serial}: left/right infrared and depth hardware timestamps differ")
        info = cal["stereo"]
        if (info["left"], info["right"]) != ("infrared_1", "infrared_2"):
            raise ValueError(f"cam_{serial}: unexpected stereo roles {info}")
        T_rl = _extrinsic(cal, "infrared_2_from_infrared_1", "infrared_1", "infrared_2")
        T_cl = _extrinsic(cal, "color_from_infrared_1", "infrared_1", "color")
        T_dl = _extrinsic(cal, "depth_from_infrared_1", "infrared_1", "depth")
        if not np.allclose(T_dl, np.eye(4), atol=1e-9):
            raise ValueError(f"cam_{serial}: depth frame is not the left infrared frame")
        out = []
        for eye, name in (("left", "infrared_1"), ("right", "infrared_2")):
            s = cal["streams"][name]
            sid = f"{serial}/ir_{eye}"
            out.append(CameraStream(
                stream_id=sid, modality="infrared", channels=1, channel_order="GRAY", dtype="uint8",
                intrinsics=_intrinsics(s["intrinsics"], src), sensor_name=f"{cal['device']['name']} {name}",
                ir_emitter="on" if emitter.get("emitter_enabled") else "off",
                notes=f"recorded {s['format']} {s['width']}x{s['height']}@{s['fps']} as PNG; device-rectified "
                      f"{eye} infrared ({name})",
                **{**common, "eye": eye}))
            stamps[sid] = _ns(st["t"][()])
        geom = IrStereoGeometry(
            rig_id=serial, left_stream=f"{serial}/ir_left", right_stream=f"{serial}/ir_right", status="metric",
            T_right_left=tuple(map(tuple, T_rl)), baseline_m=float(info["baseline_m"]),
            T_color_left=tuple(map(tuple, T_cl)),
            rectified_by="device (librealsense Y8 infrared streams: identical K, zero distortion, R = I)",
            pairing="stereo record k: left, right and depth share the hardware timestamp",
            source=f"{src}: extrinsics infrared_2_from_infrared_1, color_from_infrared_1 "
                   f"({cal['extrinsics_convention']})",
            notes=f"emitter {emitter.get('emitter_enabled_description', '?')} power {emitter.get('laser_power')}")
        return geom, out

    @staticmethod
    def _trajectory(f) -> RobotTrajectory:
        t = np.asarray(f["robot/t"][()], np.float64)
        gt, gp = np.asarray(f["gripper/t"][()], np.float64), np.asarray(f["gripper/position"][()], np.float64)
        grip = np.interp(t, gt, gp)  # clamps to the first/last register value outside the gripper samples
        return RobotTrajectory(
            timestamps_ns=_ns(t), joint_positions=np.asarray(f["robot/q"][()], np.float64), gripper_state=grip,
            ee_pose_base=_pose_matrix(np.asarray(f["robot/flange_pose"][()], np.float64)), ee_frame=None,
            extra={"dq": np.asarray(f["robot/dq"][()]), "tau_ext": np.asarray(f["robot/tau_ext"][()]),
                   "joint_frames": np.asarray(f["robot/joint_frames"][()])},
            extra_units={"dq": "rad/s", "tau_ext": "N m",
                         "joint_frames": "per joint [x y z qx qy qz qw] in the base frame (m), as recorded"})

    def iter_frames(self, episode_id: str, stream_id: str, start: int = 0,
                    stop: int | None = None) -> Iterator[tuple[int, int, np.ndarray]]:
        from PIL import Image

        serial, kind = stream_id.split("/")
        with self._open(episode_id) as f:
            g = f["cameras"][f"cam_{serial}"]
            data = g[{"color": "color", "depth": "stereo/depth", "ir_left": "stereo/left",
                      "ir_right": "stereo/right"}[kind]]
            stamps = _ns(g["t"][()] if kind == "color" else g["stereo/t"][()])
            stop = len(stamps) if stop is None else min(stop, len(stamps))
            for i in range(start, stop):
                img = Image.open(io.BytesIO(bytes(data[i])))
                if kind == "color":
                    arr = np.asarray(img.convert("RGB"))
                elif kind in ("ir_left", "ir_right"):
                    if img.mode != "L":
                        raise ValueError(f"{stream_id}[{i}]: infrared PNG mode {img.mode} is not 8-bit gray")
                    arr = np.asarray(img, dtype=np.uint8)
                else:
                    if img.mode not in ("I;16", "I;16B", "I;16L"):
                        raise ValueError(f"{stream_id}[{i}]: depth PNG mode {img.mode} is not 16-bit")
                    arr = np.asarray(img, dtype=np.uint16)
                yield i, int(stamps[i]), arr
