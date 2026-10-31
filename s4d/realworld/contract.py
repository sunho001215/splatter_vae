"""Versioned normalized episode contract for real-world recordings.

The contract is what every ingestion adapter must produce and what every downstream stage consumes. It records
camera identity, stream modality (RGB versus infrared, never blurred), timestamps, intrinsics + distortion, stereo
geometry (or its explicit absence), robot state, robot assets, units, coordinate conventions and provenance.

Image pixels are not held in ``EpisodeMeta``; they are read lazily through the adapter (ingest) or the
normalized frame store (later stages), so episodes of any length stay bounded in memory.

Conventions (``CONVENTIONS`` is written into every normalized episode):

* lengths metres, angles radians, timestamps int64 nanoseconds on one declared clock;
* camera axes OpenCV (+x right, +y down, +z forward);
* ``T_a_b`` maps homogeneous points from frame ``b`` into frame ``a``; the world frame is the robot base
  (``panda_link0``), so a camera extrinsic ``T_cam_base`` equals ``w2c`` of ``s4d.data.contract``;
* pixel coordinates are *continuous* (top-left image corner = (0, 0), centre of pixel row i / column j =
  (j + .5, i + .5)), matching ``s4d.data.contract``; adapters declare the convention of the recorded
  intrinsics and ``Intrinsics.K()`` converts;
* depth is metric camera-z (not ray length), float32, 0 = invalid; disparity is left-reference
  ``x_left - x_right`` in rectified pixels unless a field says otherwise.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "s4d.realworld.episode/1.0.0"

CONVENTIONS = {
    "length": "m",
    "angle": "rad",
    "time": "int64 ns, one clock per episode (declared in EpisodeMeta.clock)",
    "camera_axes": "OpenCV: +x right, +y down, +z forward",
    "transform_naming": "T_a_b maps homogeneous points from frame b into frame a",
    "world_frame": "robot base (panda_link0)",
    "camera_extrinsic": "T_cam_base, identical to w2c in s4d.data.contract",
    "pixel_coordinates": "continuous: image corner (0,0); pixel (row i, col j) centre at (j+.5, i+.5)",
    "depth": "metric camera-z float32, 0 = invalid",
    "disparity": "left-reference x_left - x_right in rectified pixels (positive), unless stated",
}

MODALITIES = ("rgb", "infrared")
EYES = ("left", "right", "mono")
PIXEL_CONVENTIONS = ("integer_centers", "continuous")
# Number of coefficients per distortion model. RealSense SDK names are kept verbatim so that an adapter can
# record exactly what the device reported; stages that need undistortion refuse models they do not implement.
DISTORTION_MODELS = {
    "none": 0,
    "opencv_plumb_bob": 5,  # k1 k2 p1 p2 k3 (OpenCV order)
    "opencv_rational": 8,  # k1 k2 p1 p2 k3 k4 k5 k6
    "realsense_brown_conrady": 5,
    "realsense_inverse_brown_conrady": 5,
    "realsense_modified_brown_conrady": 5,
    "kannala_brandt4": 4,
}
STEREO_STATUS = ("metric", "missing", "unverified")
GRIPPER_STATE_SEMANTICS = (
    "normalized_closure",  # 0 = fully open, 1 = fully closed (DROID convention)
    "finger_joint_rad",  # Robotiq URDF finger_joint angle
    "opening_width_m",
    "robotiq_position_register",  # raw 0..255 register value
)


class EpisodeContractError(ValueError):
    pass


def _array(value, dtype=None) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pixel_convention: str
    distortion_model: str = "none"
    distortion: tuple[float, ...] = ()
    skew: float = 0.0
    source: str = "unspecified"  # e.g. realsense_factory, user_calibration, synthetic

    def K(self, convention: str = "continuous") -> np.ndarray:
        """3x3 matrix in ``convention`` (default continuous, the repository convention)."""
        if convention not in PIXEL_CONVENTIONS or self.pixel_convention not in PIXEL_CONVENTIONS:
            raise EpisodeContractError(f"unknown pixel convention {self.pixel_convention!r} -> {convention!r}")
        shift = 0.0
        if self.pixel_convention == "integer_centers" and convention == "continuous":
            shift = 0.5
        elif self.pixel_convention == "continuous" and convention == "integer_centers":
            shift = -0.5
        return np.array(
            [[self.fx, self.skew, self.cx + shift], [0.0, self.fy, self.cy + shift], [0.0, 0.0, 1.0]], np.float64
        )


@dataclass(frozen=True)
class CameraStream:
    stream_id: str  # unique in the episode, e.g. "rig0/left"
    rig_id: str  # physical device (one RealSense)
    eye: str  # left / right / mono
    modality: str  # rgb / infrared
    channels: int  # 3 for rgb, 1 for infrared (replicated grayscale is NOT rgb)
    channel_order: str  # RGB / BGR / GRAY
    dtype: str  # uint8 / uint16
    intrinsics: Intrinsics
    rectified: bool | None  # None = unknown; stereo depth refuses unknown unless a rectification stage runs
    device_serial: str | None = None
    sensor_name: str | None = None
    timestamp_source: str = "unspecified"  # e.g. device_hardware, host_arrival, global_time
    ir_emitter: str | None = None  # on / off / alternating / unknown (infrared only)
    notes: str = ""


@dataclass(frozen=True)
class StereoGeometry:
    rig_id: str
    left_stream: str
    right_stream: str
    status: str  # metric / missing / unverified
    T_right_left: tuple | None = None  # 4x4 nested tuple, metres; maps left-camera points into right camera
    baseline_m: float | None = None
    source: str = "unspecified"
    notes: str = ""

    def T(self) -> np.ndarray:
        if self.T_right_left is None:
            raise EpisodeContractError(f"rig {self.rig_id}: stereo extrinsic is missing (status={self.status})")
        return np.asarray(self.T_right_left, np.float64)


@dataclass(frozen=True)
class RobotDescription:
    arm_model: str  # franka_panda
    gripper_model: str  # e.g. robotiq_2f_85 / robotiq_2f_140 / unknown (calibration refuses unknown)
    urdf_path: str | None  # composed arm+gripper URDF, or None until supplied
    mesh_root: str | None
    base_frame: str = "panda_link0"
    flange_frame: str = "panda_link8"
    gripper_base_frame: str | None = None
    T_flange_gripper: tuple | None = None  # 4x4 mounting/coupling transform, metres
    mount_source: str = "unspecified"
    joint_names: tuple[str, ...] = tuple(f"panda_joint{i}" for i in range(1, 8))
    gripper_state_semantics: str = "normalized_closure"
    gripper_joint: str | None = None  # URDF joint driven by the gripper state (e.g. finger_joint)
    asset_sha256: dict = field(default_factory=dict)


@dataclass
class RobotTrajectory:
    timestamps_ns: np.ndarray  # (T,) int64
    joint_positions: np.ndarray  # (T, 7) rad
    gripper_state: np.ndarray  # (T,) per RobotDescription.gripper_state_semantics
    ee_pose_base: np.ndarray | None = None  # (T,4,4) recorded EE pose; cross-check only, never used as geometry
    ee_frame: str | None = None  # which frame the recorded EE pose refers to
    extra: dict = field(default_factory=dict)  # name -> (T, ...) arrays (velocities, torques, ...)
    extra_units: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceFile:
    path: str  # relative to Provenance.source_root
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Provenance:
    adapter: str
    adapter_version: str
    source_root: str
    source_files: tuple[SourceFile, ...] = ()
    created_utc: str = ""
    notes: str = ""


@dataclass
class EpisodeMeta:
    episode_id: str
    streams: tuple[CameraStream, ...]
    stereo: tuple[StereoGeometry, ...]
    robot: RobotDescription
    trajectory: RobotTrajectory
    stream_timestamps_ns: dict  # stream_id -> (N,) int64
    provenance: Provenance
    clock: str = "unspecified"
    assumptions: tuple[str, ...] = ()  # every adapter assumption awaiting confirmation on real data
    schema_version: str = SCHEMA_VERSION
    conventions: dict = field(default_factory=lambda: dict(CONVENTIONS))

    def stream(self, stream_id: str) -> CameraStream:
        for item in self.streams:
            if item.stream_id == stream_id:
                return item
        raise KeyError(stream_id)

    def rig_stereo(self, rig_id: str) -> StereoGeometry:
        for item in self.stereo:
            if item.rig_id == rig_id:
                return item
        raise KeyError(rig_id)


# ---------------------------------------------------------------- validation


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise EpisodeContractError("; ".join(self.errors))


def _rigid(T: np.ndarray, tol: float = 1e-5) -> bool:
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return False
    R = T[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol
                and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def validate_episode(meta: EpisodeMeta) -> ValidationReport:
    """Structural and physical checks; errors make the episode unusable, warnings disable specific stages."""
    rep = ValidationReport()
    err, warn = rep.errors.append, rep.warnings.append
    if meta.schema_version != SCHEMA_VERSION:
        err(f"schema {meta.schema_version!r} != {SCHEMA_VERSION!r}")
    if not meta.episode_id or "/" in meta.episode_id:
        err("episode_id must be a non-empty path-safe string")
    ids = [s.stream_id for s in meta.streams]
    if len(set(ids)) != len(ids):
        err("duplicate stream ids")
    for s in meta.streams:
        tag = f"stream {s.stream_id}"
        if s.eye not in EYES:
            err(f"{tag}: eye {s.eye!r} not in {EYES}")
        if s.modality not in MODALITIES:
            err(f"{tag}: modality {s.modality!r} not in {MODALITIES}")
        if s.modality == "infrared" and (s.channels != 1 or s.channel_order != "GRAY"):
            err(f"{tag}: infrared must be stored as 1-channel GRAY; replicated grayscale is not rgb")
        if s.modality == "rgb" and (s.channels != 3 or s.channel_order not in ("RGB", "BGR")):
            err(f"{tag}: rgb streams need 3 channels in RGB or BGR order")
        if s.dtype not in ("uint8", "uint16"):
            err(f"{tag}: dtype {s.dtype!r} unsupported")
        k = s.intrinsics
        if k.pixel_convention not in PIXEL_CONVENTIONS:
            err(f"{tag}: pixel convention {k.pixel_convention!r} not in {PIXEL_CONVENTIONS}")
        if not (k.width > 0 and k.height > 0 and k.fx > 0 and k.fy > 0):
            err(f"{tag}: invalid intrinsics size/focal")
        if not (0 <= k.cx <= k.width and 0 <= k.cy <= k.height):
            warn(f"{tag}: principal point outside the image")
        if k.distortion_model not in DISTORTION_MODELS:
            err(f"{tag}: unknown distortion model {k.distortion_model!r}")
        elif len(k.distortion) != DISTORTION_MODELS[k.distortion_model]:
            err(f"{tag}: {k.distortion_model} needs {DISTORTION_MODELS[k.distortion_model]} coefficients")
        stamps = meta.stream_timestamps_ns.get(s.stream_id)
        if stamps is None:
            err(f"{tag}: no timestamps")
        else:
            stamps = np.asarray(stamps)
            if stamps.ndim != 1 or len(stamps) == 0 or stamps.dtype.kind not in "iu":
                err(f"{tag}: timestamps must be a non-empty integer vector (ns)")
            elif np.any(np.diff(stamps) <= 0):
                err(f"{tag}: timestamps not strictly increasing")
        if s.rectified is None:
            warn(f"{tag}: rectification state unknown")
    for g in meta.stereo:
        tag = f"rig {g.rig_id}"
        if g.status not in STEREO_STATUS:
            err(f"{tag}: stereo status {g.status!r} not in {STEREO_STATUS}")
            continue
        for sid, eye in ((g.left_stream, "left"), (g.right_stream, "right")):
            if sid not in ids:
                err(f"{tag}: stereo stream {sid!r} missing")
            elif meta.stream(sid).eye != eye or meta.stream(sid).rig_id != g.rig_id:
                err(f"{tag}: stream {sid!r} is not the {eye} eye of this rig")
        if g.left_stream in ids and g.right_stream in ids:
            a, b = meta.stream(g.left_stream), meta.stream(g.right_stream)
            if a.modality != b.modality:
                err(f"{tag}: left/right modalities differ ({a.modality} vs {b.modality})")
            if (a.intrinsics.width, a.intrinsics.height) != (b.intrinsics.width, b.intrinsics.height):
                err(f"{tag}: left/right image sizes differ")
        if g.status == "metric":
            if g.T_right_left is None or g.baseline_m is None:
                err(f"{tag}: metric stereo needs T_right_left and baseline_m")
            else:
                T = g.T()
                if not _rigid(T):
                    err(f"{tag}: T_right_left is not a rigid transform")
                elif abs(np.linalg.norm(T[:3, 3]) - g.baseline_m) > 1e-4:
                    err(f"{tag}: baseline_m disagrees with |t| of T_right_left")
                if g.baseline_m <= 0:
                    err(f"{tag}: baseline must be positive")
        else:
            warn(f"{tag}: stereo geometry {g.status}; metric stereo depth is unavailable (never invented)")
    r = meta.robot
    if r.arm_model != "franka_panda":
        err(f"arm {r.arm_model!r} unsupported (franka_panda only)")
    if r.gripper_model in ("", "unknown"):
        warn("gripper model unknown: robot-grounded calibration will refuse to run")
    if r.urdf_path is None:
        warn("no URDF supplied: robot-grounded calibration will refuse to run")
    if r.T_flange_gripper is not None and not _rigid(np.asarray(r.T_flange_gripper, float)):
        err("T_flange_gripper is not rigid")
    if r.gripper_state_semantics not in GRIPPER_STATE_SEMANTICS:
        err(f"gripper state semantics {r.gripper_state_semantics!r} not in {GRIPPER_STATE_SEMANTICS}")
    tr = meta.trajectory
    T = len(np.asarray(tr.timestamps_ns))
    if T == 0 or np.asarray(tr.timestamps_ns).dtype.kind not in "iu" or np.any(np.diff(tr.timestamps_ns) <= 0):
        err("robot timestamps must be a strictly increasing non-empty int64 vector")
    if np.asarray(tr.joint_positions).shape != (T, len(r.joint_names)):
        err(f"joint_positions must be (T={T}, {len(r.joint_names)})")
    elif not np.isfinite(tr.joint_positions).all():
        err("joint_positions contain non-finite values")
    if np.asarray(tr.gripper_state).shape != (T,):
        err("gripper_state must be (T,)")
    if tr.ee_pose_base is not None:
        if np.asarray(tr.ee_pose_base).shape != (T, 4, 4):
            err("ee_pose_base must be (T,4,4)")
        if tr.ee_frame is None:
            err("ee_pose_base given without ee_frame")
    for name, value in tr.extra.items():
        if len(value) != T:
            err(f"extra robot state {name!r} length != T")
        if name not in tr.extra_units:
            warn(f"extra robot state {name!r} has no unit")
    if not meta.provenance.adapter:
        err("provenance.adapter missing")
    return rep


# ---------------------------------------------------------------- serialization (json + npz)


def _jsonable(value: Any):
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _tuple4(value):
    return None if value is None else tuple(tuple(float(x) for x in row) for row in value)


def meta_to_json(meta: EpisodeMeta) -> tuple[dict, dict]:
    """(json document without large arrays, arrays for an npz). Round-trips through ``meta_from_json``."""
    doc = _jsonable(dataclasses.replace(meta, trajectory=None, stream_timestamps_ns={}))
    tr = meta.trajectory
    arrays = {
        "trajectory/timestamps_ns": np.asarray(tr.timestamps_ns, np.int64),
        "trajectory/joint_positions": np.asarray(tr.joint_positions, np.float64),
        "trajectory/gripper_state": np.asarray(tr.gripper_state, np.float64),
    }
    if tr.ee_pose_base is not None:
        arrays["trajectory/ee_pose_base"] = np.asarray(tr.ee_pose_base, np.float64)
    for name, value in tr.extra.items():
        arrays[f"trajectory/extra/{name}"] = np.asarray(value)
    for sid, stamps in meta.stream_timestamps_ns.items():
        arrays[f"timestamps/{sid}"] = np.asarray(stamps, np.int64)
    doc["trajectory"] = {"ee_frame": tr.ee_frame, "extra_units": dict(tr.extra_units)}
    return doc, arrays


def meta_from_json(doc: dict, arrays: dict) -> EpisodeMeta:
    def intr(d):
        return Intrinsics(**{**d, "distortion": tuple(d["distortion"])})

    streams = tuple(CameraStream(**{**s, "intrinsics": intr(s["intrinsics"])}) for s in doc["streams"])
    stereo = tuple(StereoGeometry(**{**g, "T_right_left": _tuple4(g["T_right_left"])}) for g in doc["stereo"])
    rd = doc["robot"]
    robot = RobotDescription(**{**rd, "T_flange_gripper": _tuple4(rd["T_flange_gripper"]),
                                "joint_names": tuple(rd["joint_names"])})
    extra = {k[len("trajectory/extra/"):]: np.asarray(v) for k, v in arrays.items() if k.startswith("trajectory/extra/")}
    tr = RobotTrajectory(
        timestamps_ns=np.asarray(arrays["trajectory/timestamps_ns"]),
        joint_positions=np.asarray(arrays["trajectory/joint_positions"]),
        gripper_state=np.asarray(arrays["trajectory/gripper_state"]),
        ee_pose_base=np.asarray(arrays["trajectory/ee_pose_base"]) if "trajectory/ee_pose_base" in arrays else None,
        ee_frame=doc["trajectory"]["ee_frame"],
        extra=extra,
        extra_units=doc["trajectory"]["extra_units"],
    )
    pv = doc["provenance"]
    prov = Provenance(**{**pv, "source_files": tuple(SourceFile(**f) for f in pv["source_files"])})
    stamps = {k[len("timestamps/"):]: np.asarray(v) for k, v in arrays.items() if k.startswith("timestamps/")}
    return EpisodeMeta(
        episode_id=doc["episode_id"], streams=streams, stereo=stereo, robot=robot, trajectory=tr,
        stream_timestamps_ns=stamps, provenance=prov, clock=doc["clock"], assumptions=tuple(doc["assumptions"]),
        schema_version=doc["schema_version"], conventions=doc["conventions"],
    )


def dumps(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, allow_nan=False)
