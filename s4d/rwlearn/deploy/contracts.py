"""JSON-compatible scientific input contract of a deployed real-world FQL policy. Importing this module never imports
torch, scipy or h5py (the robot-side client reconstructs the contract with ``InputSpec.from_json``).

Every field is derived from the checkpoint meta plus the training code/data (``training_contract``): the trained
camera ids come from the FQL dataset's per-episode store cameras (sorted serials = the multi-view order of
``store.sync_index``), the history rule from ``transitions.history_indices`` / ``store.history`` (3 consecutive camera
frames or synchronised steps; frame-stack reset at a run start), the period and action normalisation from the
checkpoint's action contract, the measured-state binding from ``actions.step_table``'s defaults. Nothing is guessed
per condition: a missing source raises.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from s4d.data.droid.largescale.crop import center_box, crop_intrinsics
from s4d.rwlearn.deploy.errors import ContractViolation
from s4d.rwlearn.observation import CONDITIONS, FRAME_SPACING, FRAME_STACK
from s4d.rwlearn.store import MAX_SET_MOTION_M

SIDECAR_FORMAT = "rwlearn_deploy_contract_v1"
NATIVE_DEPTH = "native_realsense"
NATIVE_DEPTH_STAGE = "sensor_depth"  # the calibration record's depth_scale key of the native device depth
RUN_GAP_FACTOR = 1.5  # store.runs_from(max_gap_factor): a larger frame gap starts a new run
STATE_MAX_AGE_NS = 15_000_000  # actions.step_table(max_state_age_ns) default (asserted equal in tests)
GRIPPER_MAX_AGE_NS = 100_000_000  # actions.MAX_GRIPPER_AGE_NS (asserted equal in tests)
FEATURE_PRECISION = "float16"  # fql_data.encode_timeline stores frozen features as float16; FQL reads .float()
SENSOR_DEPTH_RANGE_M = (0.1, 6.0)  # sensor_depth stage min/max_depth_m (configs + stage defaults; asserted in tests)


def _json(value):
    return json.loads(value) if isinstance(value, (str, bytes)) else value


def training_contract(attrs: dict) -> dict:
    """FQL dataset attrs (no image reads) -> the deployment-relevant training facts, as a JSON-serialisable dict."""
    calibration = _json(attrs.get("calibration") or "{}")
    orders = sorted({tuple(v["cameras"]) for v in calibration.values()})
    if not orders:
        raise ValueError("training camera IDs missing: the FQL dataset carries no per-episode store cameras")
    condition = str(attrs["condition"])
    out = {"format": SIDECAR_FORMAT, "condition": condition, "camera_orders": [list(o) for o in orders],
           "history_length": int(attrs["frame_stack"]), "frame_spacing": FRAME_SPACING,
           "sampling_source": "transitions.history_indices / store.history (run-start padding)",
           "export_sha256": str(attrs.get("export_sha256") or ""), "contract": _json(attrs["contract"]),
           "image_hw": list(_json(attrs["image_hw"])), "sequence_feature": bool(attrs.get("sequence_feature", False)),
           "episodes": len(calibration)}
    if CONDITIONS[condition].depth:
        from s4d.realworld.pipeline import DEPTH_SCALE_RULE, RGBD_LIFT_DEPTH

        # The FQL dataset does not record its store's depth stage: the train_data stage default (lift depth) is the
        # code-derived value; check_realworld_deploy.py verifies it against a released store episode when available.
        out["training_depth"] = str(attrs.get("training_depth") or RGBD_LIFT_DEPTH)
        out["depth_source_evidence"] = (f"train_data depth_stage default pipeline.default_depth_stage(meta, 'lift') = "
                                        f"{RGBD_LIFT_DEPTH}; stored depth = depth x accepted calibrate "
                                        f"depth_scale[{RGBD_LIFT_DEPTH}] ({DEPTH_SCALE_RULE})")
        if out["training_depth"] == NATIVE_DEPTH:
            out["depth_source_evidence"] = (
                "native depth overlay: RecordedEpisode.frame(alignment='rs_align'); native RealSense z16 -> "
                "colour-camera z -> validated rig depth_scale[sensor_depth] -> centre crop + nearest 128 resample; "
                "float32 metres, 0 invalid; identical to deployment client preprocessing")
            if "native_depth_provenance" in attrs:
                out["native_depth_provenance"] = _json(attrs["native_depth_provenance"])
    return out


def read_training_contract(meta: dict, source=None) -> dict:
    """``source``: a training-contract dict, a sidecar JSON path, or the FQL dataset (.h5; default meta['dataset']).
    Every source is checked against the checkpoint (condition, export, action contract, image size, frozen-feature
    standardisation: a sidecar must carry exactly the checkpoint's ``feature_norm`` record, or none for a checkpoint
    without one; the dataset predates the fit, so with it the checkpoint's record is taken as is)."""
    if meta.get("fixture") is True and meta.get("observation_contract") is not None:
        training = meta["observation_contract"]
        if training.get("feature_norm") != meta.get("feature_norm"):
            raise ValueError("training contract feature_norm differs from the checkpoint")
        if (("training_depth" in meta or training.get("training_depth") == NATIVE_DEPTH)
                and training.get("training_depth") != meta.get("training_depth")):
            raise ValueError("training contract training_depth differs from the checkpoint")
        return dict(training)  # labelled FIXTURE checkpoints only
    source = source if source is not None else meta.get("dataset")
    if source is None:
        raise ValueError("checkpoint has no training dataset: pass the deploy-contract sidecar or the dataset")
    if isinstance(source, dict):
        training = dict(source)
    elif str(source).endswith(".json"):
        training = json.loads(Path(source).read_text())
    else:
        import h5py

        with h5py.File(source, "r") as f:
            training = training_contract(dict(f.attrs))
        if meta.get("feature_norm") is not None:
            training["feature_norm"] = dict(meta["feature_norm"])
    if training.get("format") != SIDECAR_FORMAT:
        raise ValueError(f"not a {SIDECAR_FORMAT} training contract")
    checks = {"condition": meta["condition"], "export_sha256": meta.get("export_sha256") or "",
              "contract": meta["contract"], "image_hw": list(meta["image_hw"]),
              "feature_norm": meta.get("feature_norm")}
    if "training_depth" in meta or training.get("training_depth") == NATIVE_DEPTH:
        checks["training_depth"] = meta.get("training_depth")
    for key, expected in checks.items():
        if training.get(key) != expected:
            raise ValueError(f"training contract {key} differs from the checkpoint")
    return training


def _differences(condition: str, depth: bool, cameras: bool, training_depth: str | None = None) -> tuple[str, ...]:
    out = ["rgb: training frames are the recorder's JPEG (quality 90, encoded from bgr8) decoded by PIL; live frames are "
           "uncompressed RealSense colour (recorded replay decodes the same JPEG and is bit-exact)"]
    if depth and training_depth == NATIVE_DEPTH:
        out.append("depth: training and deployment use native RealSense z16 aligned to colour (rs.align; "
                   "recorded replay: validated emulation), 0.1-6 m gate, exact depth-camera -> colour-camera z, "
                   "x validated rig depth_scale[sensor_depth], centre crop + nearest resampling; "
                   "float32 metres, 0 invalid")
    elif depth:
        out.append("depth: training = FoundationStereo IR-stereo depth x calibrate depth_scale[foundation_stereo_depth]; "
                   "deployment = native RealSense z16 aligned to colour (rs.align; replay: validated emulation), "
                   "0.1-6 m gate, exact depth-camera -> colour-camera z, x depth_scale[sensor_depth] of the same "
                   "accepted rig document; no FoundationStereo online; weights/normalisation unchanged; residual "
                   "per DEPTH-COMPAT.md: median |dz| 4-7 mm, 12-20 mm in the edge band, 2-5 % holes")
    if cameras:
        out.append("calibration: training used each episode's accepted calibrate record of the per-task rig release; "
                   "deployment loads the published rig document once per session")
    return tuple(out)


@dataclass(frozen=True)
class InputSpec:
    condition: str
    trained_camera_orders: tuple[tuple[str, ...], ...]
    views: int
    modalities: tuple[str, ...]
    needs_calibration: bool
    image_hw: tuple[int, int]
    history_length: int
    frame_spacing: int
    frame_period_s: float
    action: dict
    training_depth: str | None = None
    deployment_depth: str | None = None
    depth_source_evidence: str | None = None
    fixture: bool = False
    camera_ids: tuple[str, ...] = ()
    native_hw: dict = field(default_factory=dict)
    crop_geometry: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)  # stored-grid continuous K and OpenCV c2w per session camera
    native_depth_correction: dict = field(default_factory=dict)
    session_id: str | None = None
    timestamp_tolerance_fraction: float = 0.01  # actions.PERIOD_TOLERANCE, applied per consecutive frame interval
    sync: str = "single camera"
    run_gap_factor: float = RUN_GAP_FACTOR
    max_set_motion_m: float | None = None
    pose_max_age_ns: int = STATE_MAX_AGE_NS
    gripper_max_age_ns: int = GRIPPER_MAX_AGE_NS
    depth_range_m: tuple[float, float] | None = None
    sequence_feature: bool = False
    feature_precision: str | None = None  # frozen-encoder features enter FQL as float16-rounded values (training)
    train_deploy_differences: tuple[str, ...] = ()

    @classmethod
    def from_meta(cls, meta: dict, training: dict) -> InputSpec:
        condition = meta["condition"]
        fixture = meta.get("fixture") is True
        if condition == "ours":
            if not fixture:
                raise ValueError("ours has no trained FQL checkpoint; only an explicitly labelled fixture is supported")
            # Deferred scientific choices MUST be supplied by the fixture, never presented as a trained contract.
            protocol = training["fixture_protocol"]
            views, depth, cameras = protocol["views"], protocol["depth"], protocol["cameras"]
        else:
            c = CONDITIONS[condition]
            views, depth, cameras = c.views, c.depth, c.cameras
        hw = tuple(int(v) for v in meta["image_hw"])
        orders = tuple(tuple(str(c) for c in o) for o in training["camera_orders"])
        if not orders or any(not o or len(set(o)) != len(o) for o in orders):
            raise ValueError("empty or duplicate training camera IDs")
        if views > 1 and (len(orders) != 1 or len(orders[0]) != views):
            raise ValueError("multi-view policy requires one unambiguous training camera order")
        if len(hw) != 2 or hw[0] != hw[1] or hw[0] <= 0:
            raise ValueError("training images must be square")
        if training["history_length"] != FRAME_STACK or training["frame_spacing"] != FRAME_SPACING:
            raise ValueError("history differs from the implemented FQL training sampling")
        period = float(meta["contract"]["nominal_period_s"])
        if not np.isfinite(period) or period <= 0:
            raise ValueError("invalid trained camera period")
        if depth and not training.get("training_depth"):
            raise ValueError("depth condition without the training depth source")
        frozen = condition != "e2e_cnn"
        return cls(condition, orders, int(views), ("rgb", "depth", "valid") if depth else ("rgb",), bool(cameras),
                   hw, FRAME_STACK, FRAME_SPACING, period, dict(meta["contract"]),
                   training_depth=training["training_depth"] if depth else None,
                   deployment_depth=NATIVE_DEPTH if depth else None,
                   depth_source_evidence=training.get("depth_source_evidence") if depth else None, fixture=fixture,
                   sync=("M5: mutual nearest frames of free-running cameras, robot surface displacement <= "
                         f"{MAX_SET_MOTION_M} m across the set" if views > 1 else "single camera"),
                   max_set_motion_m=float(MAX_SET_MOTION_M) if views > 1 else None,
                   depth_range_m=SENSOR_DEPTH_RANGE_M if depth else None,
                   sequence_feature=bool(meta.get("sequence_feature", False)),
                   feature_precision=FEATURE_PRECISION if frozen else None,
                   train_deploy_differences=_differences(condition, depth, cameras, training.get("training_depth")))

    @property
    def history_spacing_s(self) -> float:
        return self.frame_period_s * self.frame_spacing

    def bind(self, *, session_id: str, native_hw: dict, camera_id: str | None = None,
             calibration: dict | None = None, native_depth_correction: dict | None = None) -> InputSpec:
        """Per-session choices: the camera (single-view: exactly one trained id; multi-view: none, all trained cameras
        in training order), each camera's native colour grid, static calibration (K_native on the continuous colour
        grid + OpenCV c2w in the robot base frame) and the native depth correction. Missing or extra inputs raise."""
        if not isinstance(session_id, str) or not session_id:
            raise ContractViolation("session_id must be a nonempty string")
        if self.views == 1:
            trained = {c for order in self.trained_camera_orders for c in order}
            if camera_id not in trained:
                raise ContractViolation(f"choose exactly one trained camera ID from {sorted(trained)}")
            cams = (camera_id,)
        else:
            if camera_id is not None:
                raise ContractViolation("multi-view sessions require all cameras in training order")
            cams = self.trained_camera_orders[0]
        if not isinstance(native_hw, dict) or set(native_hw) != set(cams):
            raise ContractViolation("native_hw must describe exactly the session cameras")
        if self.needs_calibration:
            if not isinstance(calibration, dict) or set(calibration) != set(cams):
                raise ContractViolation("static calibration (K_native, c2w) required for exactly the session cameras")
        elif calibration is not None:
            raise ContractViolation(f"{self.condition} does not consume camera calibration")
        corrections = native_depth_correction
        if self.deployment_depth:
            if not isinstance(corrections, dict) or set(corrections) != set(cams):
                raise ContractViolation("explicit native depth correction required per camera (identity included)")
        elif corrections is not None:
            raise ContractViolation(f"{self.condition} does not consume depth")
        geometry, shapes, cal = {}, {}, {}
        for c in cams:
            hw = tuple(native_hw[c])
            if len(hw) != 2 or any(int(x) != x or x <= 0 for x in hw):
                raise ContractViolation("invalid native colour grid dimensions")
            hw = tuple(int(x) for x in hw)
            box = center_box(*hw)
            shapes[c] = list(hw)
            geometry[c] = {"box_xy_side": [box.x0, box.y0, box.side], "output_hw": list(self.image_hw),
                           "rgb": "PIL BILINEAR (antialiased)", "depth": "nearest native output-pixel centre",
                           "pixels": "continuous centres (i + 0.5, j + 0.5)", "depth_dtype": "float32 metres; 0 invalid"}
            if self.needs_calibration:
                if set(calibration[c]) != {"K_native", "c2w"}:
                    raise ContractViolation(f"{c}: static K_native (continuous colour grid) and c2w required")
                K, T = np.asarray(calibration[c]["K_native"], float), np.asarray(calibration[c]["c2w"], float)
                validate_calibration(K, T)
                cal[c] = {"K": crop_intrinsics(K, box, self.image_hw[0]).tolist(), "c2w": T.tolist()}
        for correction in (corrections or {}).values():
            validate_native_correction(correction)
        return replace(self, camera_ids=cams, native_hw=shapes, crop_geometry=geometry, calibration=cal,
                       native_depth_correction=json.loads(json.dumps(corrections or {})), session_id=session_id)

    def to_json(self) -> dict:
        out = asdict(self)
        out["history_spacing_s"] = self.history_spacing_s
        out["proprio"] = {"dimensions": 10, "conversion": self.action["proprio"],
                          "pose7": "xyz metres + quaternion xyzw; measured flange (panda_link8) in base",
                          "gripper": "measured Robotiq position register"}
        return json.loads(json.dumps(out, allow_nan=False))

    @classmethod
    def from_json(cls, d: dict) -> InputSpec:
        """Inverse of ``to_json`` (client side; ``from_json(spec.to_json()) == spec``)."""
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"history_spacing_s", "proprio"}
        if unknown:
            raise ValueError(f"unknown InputSpec fields {sorted(unknown)}")
        kw = {k: v for k, v in d.items() if k in names}
        kw["trained_camera_orders"] = tuple(tuple(o) for o in kw["trained_camera_orders"])
        for k in ("modalities", "image_hw", "camera_ids", "train_deploy_differences"):
            if k in kw:
                kw[k] = tuple(kw[k])
        if kw.get("depth_range_m") is not None:
            kw["depth_range_m"] = tuple(kw["depth_range_m"])
        return cls(**kw)


def validate_calibration(K, T):
    if K.shape != (3, 3) or T.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ContractViolation("invalid calibration shapes or nonfinite values")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ContractViolation("invalid pinhole intrinsics")
    if not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6) \
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6):
        raise ContractViolation("c2w must be a rigid OpenCV transform in the robot base frame")


def validate_native_correction(correction: dict):
    """Only the accepted calibration's ``depth_scale['sensor_depth']`` (or an explicit identity) may scale native
    RealSense depth; a FoundationStereo factor never does."""
    if not isinstance(correction, dict) or correction.get("stage") != NATIVE_DEPTH_STAGE:
        raise ContractViolation("only a native sensor_depth correction is allowed, never a FoundationStereo factor")
    factor = correction.get("factor")
    if not isinstance(factor, (int, float)) or not np.isfinite(factor) or not 0.9 <= factor <= 1.1:
        raise ContractViolation("invalid native sensor depth scale")
    if factor != 1.0 and (not correction.get("validated") or not correction.get("evidence")):
        raise ContractViolation("nonidentity native correction requires accepted calibration evidence")
