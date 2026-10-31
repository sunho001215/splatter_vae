"""DROID-3D (``jaibrdhn/droid_3d_extrinsics``) calibration, Franka kinematics and per-frame wrist camera poses.

Released schema (dataset card @ 69db9ccf and the authors' DepthWorld reader @ 6823c13d; the data files are gated and
were *not* read here, see docs/droid_largescale.md):

* ``extrinsics.jsonl`` (one JSON object per line): ``episode_uuid`` (DROID metadata uuid
  ``<LAB>+<user_id>+YYYY-MM-DD-HHh-MMm-SSs``; the join key), ``traj_id`` (DepthWorld-internal ordinal, never a join
  key), ``lab``, ``robot_serial``, ``scene_path``, ``T0_ext1_in_world`` / ``T1_ext2_in_world`` (4x4; despite the names
  **world->camera view matrices of the LEFT ZED camera**, world = robot base ``panda_link0``),
  ``dT_gw_gripper_in_world_correction`` (4x4, per robot), ``dq_joint_correction`` (7, rad, per robot), ``_meta``.
* ``camera_intrinsics.jsonl``: ``episode_uuid``, ``cameras.{ext1,ext2,wrist}.{left,right}.{fx,fy,cx,cy,disto,width,
  height}`` at native 1280x720 (ZED factory calibration), plus ``serial`` and ``role`` per camera.
* wrist: no stored pose. ``view(t) = FLIP @ T_gw_static @ dT_gw @ inv(FK(q(t) + dq))`` with ``FLIP = diag(-1,-1,1,1)``,
  ``FK`` = modified-DH Franka panda_link0 -> panda_link8 (flange), ``T_gw_static`` = per-robot-serial ``mean_mat``
  (wrist <- flange) from DepthWorld ``depth_extras/assets/gripper2wrist_transforms.json``. DROID wrist images are
  rotated 180 degrees in-plane relative to that frustum, so the pose of the *stored* image is
  ``inv(view) @ Rz(180)``; FLIP equals Rz(180), so the two cancel exactly (tested).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from s4d.data.droid.largescale.geometry import check_rigid, invert, rotate180_camera, scale_intrinsics

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])

# Franka Emika Panda / FR3 modified (Craig) DH: (a, d, alpha) per joint i, then the flange (d = 0.107).
# T_i = RotX(alpha_i) @ TransX(a_i) @ RotZ(q_i) @ TransZ(d_i)  (Franka "Robot and interface specifications").
FRANKA_MDH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FLANGE_D = 0.107


def _mdh(a: float, d: float, alpha: float, theta) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 3] = ct, -st, a
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = st * sa, ct * sa, ca, ca * d
    T[..., 3, 3] = 1
    return T


def franka_fk(q) -> np.ndarray:
    """``T_link0_link8`` (flange) for joint angles (..., 7)."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 7 or not np.isfinite(q).all():
        raise ValueError("Franka FK needs finite (..., 7) joint angles")
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i, (a, d, alpha) in enumerate(FRANKA_MDH):
        T = T @ _mdh(a, d, alpha, q[..., i])
    return T @ _mdh(0.0, FLANGE_D, 0.0, np.zeros(q.shape[:-1]))


def _matrix(value, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    return check_rigid(T, name)


def parse_droid3d_extrinsics(record: dict) -> dict:
    """One ``extrinsics.jsonl`` line -> world->camera matrices for ext1/ext2 (left) and the per-robot wrist chain."""
    out = {
        "episode_uuid": str(record["episode_uuid"]),
        "robot_serial": str(record["robot_serial"]),
        "w2c": {
            "ext1": _matrix(record["T0_ext1_in_world"], "T0_ext1_in_world"),
            "ext2": _matrix(record["T1_ext2_in_world"], "T1_ext2_in_world"),
        },
        "dT_gw": _matrix(record["dT_gw_gripper_in_world_correction"], "dT_gw_gripper_in_world_correction"),
        "dq": np.asarray(record["dq_joint_correction"], dtype=np.float64),
        "scene_path": str(record.get("scene_path", "")),
        "meta": record.get("_meta", {}),
    }
    if out["dq"].shape != (7,) or not np.isfinite(out["dq"]).all():
        raise ValueError("dq_joint_correction must be 7 finite radians")
    if "T_w2w_scene_correction" in record:  # absent from the inspected release; refuse rather than ignore it
        raise ValueError("T_w2w_scene_correction present: release changed, conversion not implemented")
    return out


def parse_droid3d_intrinsics(record: dict, role: str, *, eye: str = "left", out_wh=(320, 180)) -> dict:
    """Factory ZED intrinsics of one camera role, rescaled to the RLDS/PointWorld 320x180 grid (continuous centres)."""
    cam = record["cameras"][role][eye]
    w, h = int(cam["width"]), int(cam["height"])
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float64)
    K[0, 2] += 0.5  # OpenCV/ZED integer pixel centres -> continuous
    K[1, 2] += 0.5
    serial = record["cameras"][role].get("serial")
    return {
        "K": scale_intrinsics(K, (w, h), tuple(out_wh)),
        "native_wh": (w, h),
        "serial": None if serial is None else str(serial),
    }


def load_hand_eye(path: str | Path) -> dict[str, np.ndarray]:
    """``gripper2wrist_transforms.json``: robot serial -> ``mean_mat`` (wrist camera <- flange)."""
    data = json.loads(Path(path).read_text())
    return {str(serial): _matrix(entry["mean_mat"], f"mean_mat[{serial}]") for serial, entry in data.items()}


def droid3d_wrist_w2c(q, dq, T_gw_static, dT_gw, *, image_rotation_deg: int = 180) -> np.ndarray:
    """Per-frame world->camera of the *stored* DROID wrist images, (T,4,4).

    The released frustum is ``view = FLIP @ T_gw_static @ dT_gw @ inv(FK(q + dq))``; the stored images are rotated by
    ``image_rotation_deg`` relative to it, i.e. their camera is ``c2w = inv(view) @ Rz(rotation)``.
    """
    q = np.asarray(q, dtype=np.float64)
    view = FLIP @ np.asarray(T_gw_static) @ np.asarray(dT_gw) @ invert(franka_fk(q + np.asarray(dq)))
    c2w = invert(view)
    if image_rotation_deg == 180:
        c2w = c2w @ FLIP  # Rz(180) == diag(-1,-1,1,1)
    elif image_rotation_deg != 0:
        raise ValueError("only 0 or 180 degree wrist image rotations are defined")
    return check_rigid(invert(c2w), "wrist w2c")


def droid_wrist_hand_eye(wrist_cam_extrinsics6, ee_pose6_frame0) -> np.ndarray:
    """Original DROID hand-eye from metadata: ``wrist_cam_extrinsics`` is the frame-0 composed pose
    ``change_pose_frame(gripper_offset, cartesian_position[0])``, so ``T_ee_cam = inv(T_base_ee(0)) @ T_base_cam(0)``."""
    from s4d.data.droid.largescale.geometry import hand_eye_from_pose, pose6_to_matrix

    return hand_eye_from_pose(pose6_to_matrix(wrist_cam_extrinsics6), ee_pose6_frame0)


def stored_image_camera(K_frustum, c2w_frustum, width: int, height: int):
    """Camera of an image stored rotated by 180 degrees relative to a frustum (K and pose both change)."""
    return rotate180_camera(K_frustum, c2w_frustum, width, height)


class JsonlIndex:
    """Byte-offset index of a large JSONL file keyed by one field; lookups read only the matching lines.

    Real DROID-3D ``camera_intrinsics.jsonl`` holds 100 ``episode_uuid`` values twice: DROID raw has the same uuid
    under ``success/`` and ``failure/`` (different ``raw_dir``). ``get`` refuses such keys; ``get_all`` returns every
    record so the join can pick the one whose path suffix matches (``join.droid3d_record``).
    """

    def __init__(self, path: str | Path, key: str = "episode_uuid"):
        self.path = Path(path)
        self.spans: dict[str, list[tuple[int, int]]] = {}
        with self.path.open("rb") as stream:
            offset = 0
            for line in stream:
                if line.strip():
                    value = str(json.loads(line)[key])
                    self.spans.setdefault(value, []).append((offset, len(line)))
                offset += len(line)
        self.duplicates = {value for value, spans in self.spans.items() if len(spans) > 1}

    def __len__(self) -> int:
        return len(self.spans)

    def __contains__(self, value: str) -> bool:
        return value in self.spans

    def get_all(self, value: str) -> list[dict]:
        out = []
        with self.path.open("rb") as stream:
            for offset, length in self.spans.get(value, ()):
                stream.seek(offset)
                out.append(json.loads(stream.read(length)))
        return out

    def get(self, value: str) -> dict | None:
        if value in self.duplicates:
            raise ValueError(f"{self.path.name}: {value!r} appears more than once; refusing an ambiguous join")
        records = self.get_all(value)
        return records[0] if records else None
