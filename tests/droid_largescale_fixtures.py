"""Schema-faithful miniature DROID RLDS / PointWorld-DROID / DROID-3D files with exact synthetic ground truth.

Field names, dtypes, shapes and conventions follow the released schemas recorded in docs/droid_largescale.md
(RLDS ``features.json``; PointWorld flow/depth/camera files; DROID-3D ``extrinsics.jsonl`` /
``camera_intrinsics.jsonl``; DepthWorld ``gripper2wrist_transforms.json``). Geometry: a textured table plane z = 0 in
the robot-base frame; tracked points lie on it (static ones and a sliding patch), so every depth map is the exact
ray/plane intersection and depth tests are meaningful. PointWorld's calibration is a deliberately perturbed copy of
the true one (as a different optimizer would give), and its world tracks are lifted under it, as in the release.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from s4d.data.droid.largescale import calibration as C
from s4d.data.droid.largescale.geometry import invert, pose6_to_matrix, transform_points
from s4d.data.droid.largescale.tfrecord import encode_example, write_records

EPISODE = "AUTOLab+0d4edc83+2023-10-21-19h-07m-04s"
SCENE = "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023/"
SERIALS = {"ext1": "22008760", "ext2": "24400334", "wrist": "18026681"}
ROBOT = "panda-295341-1325237"
W, H = 320, 180
RAW_STEPS = 48
CANON = 24
CLIPS = ("0:11", "5:16", "10:21")
Q_HOME = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.785])


def lookat_c2w(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def small_rigid(rx, ry, rz, tx, ty, tz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
    T[:3, 3] = (tx, ty, tz)
    return T


def plane_depth(K, c2w, h=H, w=W) -> np.ndarray:
    """Exact z-depth of the plane z_world = 0 at every continuous pixel centre (0 where not hit in front)."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_world = rays @ c2w[:3, :3].T
    o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -o[2] / d_world[..., 2]
    depth = np.where((s > 0) & np.isfinite(s) & (s < 4.0), s, 0.0)
    return depth


def jpeg(color, h=H, w=W) -> bytes:
    image = np.zeros((h, w, 3), np.uint8)
    image[:] = color
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def frame_color(step: int, cam: int) -> tuple[int, int, int]:
    """Encodes (raw step, camera) in the frame colour so tests can verify which RLDS frame was read."""
    return (8 * step % 256, 60 + 60 * cam, 200 - 4 * step % 128)


def truth(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q = Q_HOME + 0.08 * np.sin(np.linspace(0, 3, RAW_STEPS))[:, None] * np.array([1, 0.5, 0.3, 0.4, 0.2, 0.3, 0.5])
    T_flange_tool = small_rigid(0, 0, -np.pi / 4, 0, 0, 0.1034)
    fk = C.franka_fk(q)
    ee = fk @ T_flange_tool
    cart = np.concatenate([ee[:, :3, 3], Rotation.from_matrix(ee[:, :3, :3]).as_euler("xyz")], -1)
    grip = np.clip(0.5 + 0.5 * np.sin(np.linspace(0, 2, RAW_STEPS)), 0, 1)[:, None]
    dq = np.array([0.004, -0.003, 0.002, 0.003, -0.002, 0.001, 0.002])
    dT_gw = small_rigid(0.003, -0.002, 0.001, 0.002, -0.001, 0.001)
    T_flange_cam = small_rigid(0, 0, np.pi / 2, 0.06, 0.0, 0.02)  # optical axis along the flange approach (+z)
    T_gw_static = invert(T_flange_cam)  # wrist camera <- flange ("mean_mat")
    c2w_true = {
        "ext1": lookat_c2w((0.45, -0.75, 0.55), (0.45, 0.0, 0.0)),
        "ext2": lookat_c2w((0.5, 0.8, 0.6), (0.45, 0.05, 0.0)),
    }
    delta = {
        "ext1": small_rigid(0.01, -0.008, 0.006, 0.012, -0.007, 0.009),
        "ext2": small_rigid(-0.007, 0.01, -0.005, -0.01, 0.008, -0.006),
    }
    w2c_true = {r: invert(c) for r, c in c2w_true.items()}
    w2c_pw = {r: w2c_true[r] @ delta[r] for r in w2c_true}  # a different (perturbed) calibration
    K_ext = np.array([[131.0, 0, 160.0], [0, 131.0, 90.0], [0, 0, 1]])  # continuous centres at 320x180
    K_wrist = np.array([[95.0, 0, 160.0], [0, 95.0, 90.0], [0, 0, 1]])
    view = C.FLIP @ T_gw_static @ dT_gw @ invert(C.franka_fk(q + dq))
    c2w_wrist = invert(view) @ C.FLIP  # stored-image camera (180 degree rotation relative to the frustum)
    stamps = {"ext1": np.round(np.arange(RAW_STEPS - 1) * 66.7 + rng.integers(0, 3, RAW_STEPS - 1)).astype(np.int64)}
    stamps["ext2"] = stamps["ext1"] + 5
    stamps["wrist"] = stamps["ext1"] + 9
    stamps["wrist"][31:] += 60  # a dropped/late wrist frame: raw step 2i no longer nearest for later canonical frames
    canonical = stamps["ext1"][::2][:CANON]
    return dict(
        q=q,
        ee=ee,
        cart=cart,
        grip=grip,
        dq=dq,
        dT_gw=dT_gw,
        T_gw_static=T_gw_static,
        c2w_true=c2w_true,
        w2c_true=w2c_true,
        w2c_pw=w2c_pw,
        K_ext=K_ext,
        K_wrist=K_wrist,
        c2w_wrist=c2w_wrist,
        stamps=stamps,
        canonical=canonical,
        T_flange_tool=T_flange_tool,
    )


def track_world_true(role: str, start: int, end: int, n: int, seed: int) -> np.ndarray:
    """(T,N,3) true world positions on the table plane; the first third slides +x by 1 cm per canonical step."""
    rng = np.random.default_rng(seed)
    base = np.stack([rng.uniform(0.25, 0.65, n), rng.uniform(-0.25, 0.3, n), np.zeros(n)], -1)
    t = np.arange(start, end)[:, None, None]
    moving = np.zeros((1, n, 1))
    moving[0, : n // 3, 0] = 1
    return base[None] + 0.01 * t * moving * np.array([1.0, 0, 0])


# --------------------------------------------------------------------------------------------------- writers
def write_rlds_shards(root: Path, gt: dict, *, decoy_duplicate: bool = False) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)

    def episode(file_path, rec_path, steps=RAW_STEPS, offset=0):
        feats = {
            "episode_metadata/file_path": ("bytes", [file_path.encode()]),
            "episode_metadata/recording_folderpath": ("bytes", [rec_path.encode()]),
            "steps/action": ("float", np.zeros((steps, 7))),
            "steps/is_first": ("int64", np.eye(1, steps, 0, dtype=np.int64)[0]),
            "steps/language_instruction": ("bytes", [b"put the marker in the cup"] * steps),
            "steps/observation/cartesian_position": ("float", gt["cart"][:steps] + offset),
            "steps/observation/gripper_position": ("float", gt["grip"][:steps]),
            "steps/observation/joint_position": ("float", gt["q"][:steps]),
        }
        for cam, key in enumerate(("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")):
            feats[f"steps/observation/{key}"] = ("bytes", [jpeg(frame_color(s, cam)) for s in range(steps)])
        return encode_example(
            {k: (kind, np.asarray(v).reshape(-1) if kind != "bytes" else v) for k, (kind, v) in feats.items()}
        )

    lab_path = "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023"
    other = "gs://gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_18:00:00_2023"
    shard0 = root / "droid_101-train.tfrecord-00000-of-00002"
    shard1 = root / "droid_101-train.tfrecord-00001-of-00002"
    write_records(shard0, [episode(other + "/trajectory.h5", other + "/recordings/MP4", steps=12, offset=0.3)])
    payloads = [episode(lab_path + "/trajectory.h5", lab_path + "/recordings/MP4")]
    if decoy_duplicate:
        payloads.append(episode(lab_path + "/trajectory.h5", lab_path + "/recordings/MP4", offset=0.1))
    write_records(shard1, payloads)
    return [shard0, shard1]


def write_pointworld(
    root: Path, gt: dict, *, n_tracks=(40, 52), episode: str = EPISODE, wrist_attr: str | None = None
) -> None:
    flows = root / "flows-fs-optimized"
    depth_dir = root / "depth_320x180"
    cams_dir = root / "cameras"
    for d in (flows, depth_dir, cams_dir):
        d.mkdir(parents=True, exist_ok=True)
    K_native = gt["K_ext"].copy()
    K_native[:2, 2] -= 0.5  # PointWorld stores integer-centre intrinsics
    with h5py.File(flows / f"{episode}_flows.h5", "w") as f:
        f.attrs.update(
            uuid=episode,
            scene_path=SCENE,
            trajectory_length=RAW_STEPS,
            domain="droid",
            lab="AUTOLab",
            canonical_timestamps=json.dumps(gt["canonical"].tolist()),
            ext1_cam_serial=SERIALS["ext1"],
            ext2_cam_serial=SERIALS["ext2"],
            robot_serial=ROBOT,
            wrist_cam_serial=wrist_attr or SERIALS["wrist"],
            hdf5_path="success/2023-10-21/Sat_Oct_21_19:07:04_2023/trajectory.h5",
        )
        quat = Rotation.from_euler("xyz", gt["cart"][::2][:CANON, 3:]).as_quat()  # x, y, z, w
        for ci, key in enumerate(CLIPS):
            start, end = map(int, key.split(":"))
            g = f.create_group(key)
            g.attrs["demo_length"] = end - start
            raw = 2 * np.arange(start, end)
            g["gripper_pose"] = np.concatenate([gt["cart"][raw, :3], quat[start:end]], -1).astype(np.float32)
            g["gripper_positions"] = (0.725 * gt["grip"][raw, 0]).astype(np.float32)
            g["gripper_open"] = gt["grip"][raw] < 0.5
            g["joint_positions"] = gt["q"][raw].astype(np.float32)
            for v, role in enumerate(("ext1", "ext2")):
                n = n_tracks[v]
                X_true = track_world_true(role, start, end, n, seed=100 * ci + v)
                # PointWorld lifts under its own calibration: X_pw = c2w_pw @ w2c_true @ X_true
                X_pw = transform_points(invert(gt["w2c_pw"][role]) @ gt["w2c_true"][role], X_true)
                cg = g.create_group(f"camera_{SERIALS[role]}_ext")
                cg["intrinsic"] = K_native.astype(np.float32)
                cg["extrinsic"] = gt["w2c_pw"][role].astype(np.float32)
                cg["scene_flows"] = X_pw.astype(np.float16)
                cg["scene_colors"] = np.full((end - start, n, 3), 128, np.uint8)
                cg["scene_normals"] = np.tile(np.array([0, 0, 127], np.int8), (end - start, n, 1))
                vis = np.ones((end - start, n), bool)
                vis[:, -2:] = False
                cg["scene_visibility"] = vis
                cg["scene_depth_valid_mask"] = np.ones((end - start, n), bool)
                rgb = np.empty((1,), dtype=h5py.vlen_dtype(np.uint8))
                rgb[0] = np.frombuffer(jpeg(frame_color(2 * start, v)), np.uint8)
                cg["initial_rgb"] = rgb
                d0 = plane_depth(gt["K_ext"], gt["c2w_true"][role])
                cg.create_dataset("initial_depth", data=np.round(d0 * 1000).astype(np.uint16))
                cg["initial_depth"].attrs.update(format="uint16_mm", scale="millimeters")
    with h5py.File(depth_dir / f"{episode}_depth.h5", "w") as f:
        for role in ("ext1", "ext2", "wrist"):
            group = f.create_group(f"{SERIALS[role]}+{'wrist' if role == 'wrist' else 'ext'}")
            frames = []
            for s in range(RAW_STEPS - 1):
                if role == "wrist":
                    d = plane_depth(gt["K_wrist"], gt["c2w_wrist"][s])
                else:
                    d = plane_depth(gt["K_ext"], gt["c2w_true"][role])
                frames.append(np.round(d * 1000).astype(np.uint16))
            group.create_dataset("depth", data=np.stack(frames), chunks=(1, H, W))
            group["depth"].attrs.update(units="millimeters", write_complete=True)
            group["timestamps"] = gt["stamps"][role]
    cameras = {
        "uuid": episode,
        "scene_path": SCENE,
        "optimization_success": True,
        "optimization_summary": {"final_loss": 0.05},
    }
    for role in ("ext1", "ext2"):
        cameras[SERIALS[role]] = {"optimized_extrinsics": gt["w2c_pw"][role].astype(np.float32).tolist()}
    (cams_dir / f"{episode}_cameras.json").write_text(json.dumps(cameras))


def write_droid3d(
    root: Path, gt: dict, *, episode: str = EPISODE, wrist_serial: str | None = None
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rec = {
        "episode_uuid": episode,
        "traj_id": 7,
        "lab": "AUTOLab",
        "robot_serial": ROBOT,
        "scene_path": "/data/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023",
        "T0_ext1_in_world": gt["w2c_true"]["ext1"].tolist(),
        "T1_ext2_in_world": gt["w2c_true"]["ext2"].tolist(),
        "dT_gw_gripper_in_world_correction": gt["dT_gw"].tolist(),
        "dq_joint_correction": gt["dq"].tolist(),
        "_meta": {"n_iters": 40, "final_cost": 0.3, "stop_reason": "converged"},
    }
    other = dict(rec, episode_uuid="AUTOLab+0d4edc83+2023-10-21-18h-00m-00s", traj_id=0)
    ext_path = root / "extrinsics.jsonl"
    ext_path.write_text(json.dumps(other) + "\n" + json.dumps(rec) + "\n")

    def cam(K, serial, role):
        fx, fy = 4 * K[0, 0], 4 * K[1, 1]
        cx, cy = 4 * K[0, 2] - 0.5, 4 * K[1, 2] - 0.5  # continuous 320x180 -> integer-centre 1280x720
        left = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "disto": [0.0] * 12,
            "width": 1280,
            "height": 720,
            "fov_h": 100.0,
            "fov_v": 70.0,
        }
        return {
            "left": left,
            "right": dict(left),
            "stereo_baseline_m": 0.12,
            "T_right_from_left": np.eye(4).tolist(),
            "serial": serial,
            "role": role,
            "fps": 15,
            "zed_sdk_version": "4.0",
        }

    intr = {
        "traj_id": 7,
        "split": "train",
        "episode_uuid": episode,
        "lab": "AUTOLab",
        "raw_dir": rec["scene_path"],
        "cameras": {
            "ext1": cam(gt["K_ext"], SERIALS["ext1"], "ext1"),
            "ext2": cam(gt["K_ext"], SERIALS["ext2"], "ext2"),
            "wrist": cam(gt["K_wrist"], wrist_serial or SERIALS["wrist"], "wrist"),
        },
    }
    intr_path = root / "camera_intrinsics.jsonl"
    intr_path.write_text(json.dumps(intr) + "\n")
    hand_eye_path = root / "gripper2wrist_transforms.json"
    hand_eye_path.write_text(json.dumps({ROBOT.upper(): {"mean_mat": gt["T_gw_static"].tolist()}}))
    return ext_path, intr_path, hand_eye_path


def build_all(tmp: Path, *, calibration: str = "droid3d", **kw) -> dict:
    from s4d.data.droid.largescale import join
    from s4d.data.droid.largescale.rlds_index import RldsStepReader, build_index, load_episode_rows

    gt = truth()
    shards = write_rlds_shards(tmp / "rlds", gt, decoy_duplicate=kw.pop("decoy_duplicate", False))
    build_index(shards, tmp / "rlds_index")
    write_pointworld(tmp / "pw", gt, **{k: v for k, v in kw.items() if k in ("n_tracks", "wrist_attr")})
    ext_path, intr_path, he_path = write_droid3d(tmp / "d3", gt, wrist_serial=kw.get("d3_wrist_serial"))
    reader = RldsStepReader(tmp / "rlds_index", tmp / "rlds")
    report = join.build_manifest(
        tmp / "pw",
        [EPISODE],
        load_episode_rows(tmp / "rlds_index"),
        tmp / "manifest.jsonl",
        calibration=calibration,
        droid3d_extrinsics=C.JsonlIndex(ext_path),
        droid3d_intrinsics=C.JsonlIndex(intr_path),
        hand_eye=C.load_hand_eye(he_path),
        rlds_reader=reader,
    )
    reader.close()
    return {"gt": gt, "report": report, "tmp": tmp, "manifest": tmp / "manifest.jsonl"}


def dataset_kwargs(env: dict, **kw) -> dict:
    base = dict(
        pointworld_root=env["tmp"] / "pw",
        rlds_index_dir=env["tmp"] / "rlds_index",
        rlds_data_root=env["tmp"] / "rlds",
        split="all",
        image_height=36,
        image_width=64,
        max_tracks=64,
    )
    base.update(kw)
    return base


__all__ = ["build_all", "dataset_kwargs", "truth", "pose6_to_matrix"]
