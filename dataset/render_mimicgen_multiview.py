#!/usr/bin/env python
"""
Create a new multi-view HDF5 dataset from a MimicGen / robomimic dataset.

What this script does:
  - Loads an existing robomimic-style HDF5 dataset (states, actions, etc.).
  - Reconstructs the simulator state for each timestep using robomimic + robosuite.
  - For each timestep, renders multi-view using a SINGLE underlying MuJoCo camera:
      * 1 center camera   (fixed pose)
      * 4 side cameras    (fixed poses, defined by (r, theta ± dtheta, phi ± dphi))
      * 4 random cameras  (per episode: fixed within that episode, re-sampled across episodes)
  - For each logical camera view, generates:
      * RGB image        -> <logical_view_name>_image      (uint8, H, W, 3)
      * segmentation     -> <logical_view_name>_seg        (int32, H, W)
        (segmentation stores geom IDs; -1 is background)

  - Stores all of this into a NEW HDF5 file (robomimic-style):
      /data/demo_xxxx/...
    including copying ALL original datasets + attributes,
    and adding /data/demo_xxxx/obs/<view>_image and <view>_seg datasets.

Important:
  - Angles (theta, phi, delta_theta, delta_phi) are in DEGREES.
  - Existing *_image / *_depth / *_seg observations in the original "obs" group
    are removed, but all other obs entries (e.g., low-dim states) are preserved.

Camera / segmentation notes:
  - We force the base camera to be attached to the WORLD body (bodyid = 0), so its
    pose is purely controlled by (pos, quat) we set, and does not move with the robot.
  - All camera positions are defined on a sphere of radius r AROUND `target_pos`.
    That is, `target_pos` plays the role of the origin for spherical coordinates.
  - After changing the camera pose, we must call sim.forward() so that MuJoCo
    updates derived camera quantities before rendering.
  - When using segmentation=True, MuJoCo typically returns an array of shape (H, W, 2):
      seg[..., 0] : object ids
      seg[..., 1] : geom  ids
    We store seg[..., 1] (the geom IDs) as our segmentation mask.

Camera intrinsics / extrinsics metadata:
  - For each episode demo_xxxx and each logical view (center, side_i, rand_i), we store:
      /data/demo_xxxx/camera_params/<view_name>/K : (3, 3) intrinsic matrix
      /data/demo_xxxx/camera_params/<view_name>/R : (3, 3) rotation (world -> camera)
      /data/demo_xxxx/camera_params/<view_name>/t : (3,)   translation (world -> camera)
  - K is derived from the MuJoCo camera's vertical field-of-view and image size.
"""

import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm

# Importing MimicGen robosuite envs registers environments into robosuite.
from mimicgen.envs.robosuite import (
    coffee,
    mug_cleanup,
    threading,
    nut_assembly,
    three_piece_assembly,
    stack,
)

from robosuite.utils.transform_utils import mat2quat

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


# ------------------------------------------------------------------------- #
#                         HDF5 / I/O helper functions                       #
# ------------------------------------------------------------------------- #

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def copy_attrs(src_obj, dst_obj):
    """Copy all HDF5 attributes from src to dst."""
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def copy_group_structure(src_group, dst_group, exclude_subgroups=None):
    """
    Recursively copy all datasets and groups from src_group to dst_group,
    except any subgroup names listed in exclude_subgroups.

    This is used at the /data/demo level. We typically exclude the "obs"
    subgroup and handle it manually in order to drop old *_image obs fields
    while preserving low-dim obs.
    """
    if exclude_subgroups is None:
        exclude_subgroups = []

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            if name in exclude_subgroups:
                # Skip copying this group here; it will be handled separately.
                continue
            new_group = dst_group.create_group(name)
            copy_attrs(item, new_group)
            copy_group_structure(item, new_group, exclude_subgroups=exclude_subgroups)
        elif isinstance(item, h5py.Dataset):
            data = item[()]  # load dataset into memory
            dset = dst_group.create_dataset(name, data=data, compression="gzip")
            copy_attrs(item, dset)
        else:
            raise TypeError(f"Unsupported HDF5 item type for {name}: {type(item)}")


def copy_obs_without_images(src_obs_group: h5py.Group, dst_obs_group: h5py.Group):
    """
    Copy the original "obs" group from src_demo to dst_demo,
    BUT drop any existing image-like observations.

    We remove keys such as:
      *_image, *_rgb, *_depth, *_seg, etc. (here we filter by suffixes)

    All low-dimensional observations (e.g., robot states) and obs-level attributes
    are preserved.
    """
    # Copy attributes on the obs group itself
    copy_attrs(src_obs_group, dst_obs_group)

    # Suffixes that indicate image-like data in the original dataset
    drop_suffixes = ("_image", "_rgb", "_depth", "_seg")

    for name, item in src_obs_group.items():
        # Skip image-like obs entries
        if any(name.endswith(suf) for suf in drop_suffixes):
            continue

        if isinstance(item, h5py.Dataset):
            data = item[()]
            dset = dst_obs_group.create_dataset(name, data=data, compression="gzip")
            copy_attrs(item, dset)
        elif isinstance(item, h5py.Group):
            # Nested groups under obs are rare but handled generically
            new_group = dst_obs_group.create_group(name)
            copy_attrs(item, new_group)
            copy_group_structure(item, new_group)
        else:
            raise TypeError(f"Unsupported item type inside obs group: {name} -> {type(item)}")


# ------------------------------------------------------------------------- #
#                          Camera pose / math utilities                     #
# ------------------------------------------------------------------------- #

def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return np.deg2rad(deg)


def spherical_to_cartesian(r: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z),
    with the sphere centered at the origin (0, 0, 0).

    Convention:
      - r >= 0
      - theta: azimuth angle in XY-plane (deg), 0 at +X, positive towards +Y
      - phi  : elevation from XY-plane (deg), 0 = horizontal, +90 = straight up

    Returns:
      np.ndarray of shape (3,) corresponding to (x, y, z).
    """
    theta = deg2rad(theta_deg)
    phi = deg2rad(phi_deg)

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    return np.array([x, y, z], dtype=np.float32)


def camera_pos_around_target(
    r: float,
    theta_deg: float,
    phi_deg: float,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute camera position on a sphere of radius r AROUND the given target point.

    This is exactly:
        pos = target + spherical_to_cartesian(r, theta_deg, phi_deg)

    Args:
        r: radius (meters) measured from the target position.
        theta_deg: azimuth angle in degrees in the XY-plane of the world frame.
        phi_deg: elevation angle in degrees, 0 is horizontal, +90 is straight up.
        target: (3,) array, world coordinates of the target point.

    Returns:
        (3,) array for camera position in world coordinates.
    """
    offset = spherical_to_cartesian(r, theta_deg, phi_deg)
    return target + offset


def _compute_lookat_rotation(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a 3x3 rotation matrix R for a camera looking at `target` from `eye`.

    The convention matches the one used for MuJoCo in this script:

        f = normalized (target - eye)       # forward direction (from eye to target)
        s = normalized (f x up)             # side direction
        u = s x f                           # corrected up direction

    Camera axes (expressed in world frame) are:

        x_cam = s
        y_cam = u
        z_cam = -f

    so that:

        R = [x_cam, y_cam, z_cam]

    where R has shape (3, 3) and its columns are the camera frame axes expressed
    in world coordinates. This is consistent with MuJoCo / robosuite expectations
    when converted to quaternions via mat2quat and re-ordered to (w, x, y, z).
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-8:
        raise ValueError("Eye and target are too close to define a camera direction.")
    f = f / f_norm  # forward

    # Side = forward x up
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-8:
        # Forward is almost parallel to up; choose a different up vector
        up_alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        s = np.cross(f, up_alt)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-8:
            raise ValueError("Unable to compute a valid side vector for the camera.")
    s = s / s_norm

    # Corrected up
    u = np.cross(s, f)

    # Rotation matrix: columns are [s, u, -f]
    R = np.stack([s, u, -f], axis=1)  # shape (3, 3)
    return R.astype(np.float32)


def lookat_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a MuJoCo-compatible quaternion for a camera looking at `target` from `eye`.

    This uses _compute_lookat_rotation to build a rotation matrix R, then converts R
    to a quaternion using robosuite's mat2quat. IMPORTANT: mat2quat returns a quaternion
    in (x, y, z, w) order, whereas MuJoCo expects (w, x, y, z). We therefore reorder.
    """
    R = _compute_lookat_rotation(eye, target, up=up)

    # robosuite.mat2quat returns (x, y, z, w)
    quat_xyzw = mat2quat(R)
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )

    return quat_wxyz


def compute_opengl_extrinsics(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OpenGL-style extrinsic parameters (R, t) from a camera eye and target.

    We reuse the same look-at rotation R_world_from_cam as used for MuJoCo:

        R_world_from_cam = _compute_lookat_rotation(eye, target)

    This matrix maps camera-frame coordinates to world-frame coordinates.

    To obtain OpenGL-style extrinsics (world -> camera), we take:

        R_cam_from_world = R_world_from_cam^T
        t_cam_from_world = - R_cam_from_world @ eye

    such that for a world point X_world (3,):

        X_cam = R_cam_from_world @ X_world + t_cam_from_world

    Args:
        eye:  (3,) camera position in world coordinates.
        target: (3,) target point in world coordinates.
        up: optional up vector in world coordinates.

    Returns:
        R: (3, 3) rotation matrix, world -> camera.
        t: (3,) translation vector, world -> camera.
    """
    R_world_from_cam = _compute_lookat_rotation(eye, target, up=up)
    R_cam_from_world = R_world_from_cam.T
    t_cam_from_world = -R_cam_from_world @ eye
    return R_cam_from_world.astype(np.float32), t_cam_from_world.astype(np.float32)


def set_camera_pose(sim, cam_name: str, pos: np.ndarray, quat: np.ndarray):
    """
    Set MuJoCo camera pose for camera `cam_name` in the sim.

    This directly writes into the low-level model arrays:
      - cam_pos[id]   : 3D position vector (in parent body frame)
      - cam_quat[id]  : orientation quaternion [w, x, y, z] (in parent body frame)
      - cam_bodyid[id]: index of the parent body

    In many robosuite environments, cameras are attached to a robot body. This causes
    the camera to move when the robot moves, and our manually set (pos, quat) end up
    being offsets in the robot frame.

    To make the camera purely world-fixed and controlled ONLY by (pos, quat), we:
      - Force cam_bodyid[id] = 0 (world body)
      - Set cam_pos and cam_quat in WORLD coordinates
      - Call sim.forward() to update derived camera quantities before rendering.
    """
    model = sim.model
    cam_id = model.camera_name2id(cam_name)

    # Attach camera to the world body if it is not already
    if model.cam_bodyid[cam_id] != 0:
        model.cam_bodyid[cam_id] = 0

    # Set pose in the (now) world frame
    model.cam_pos[cam_id] = pos
    model.cam_quat[cam_id] = quat

    # IMPORTANT: must call forward() so that MuJoCo recomputes camera transforms
    sim.forward()


def compute_intrinsics_from_mujoco(
    model,
    cam_id: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Compute an intrinsic matrix K from a MuJoCo camera definition.

    We use the MuJoCo vertical field-of-view (cam_fovy) and the render resolution.

    Let:
       fovy = vertical FOV in radians
       H    = image height (pixels)
       W    = image width  (pixels)

    Then:
       fy = (H / 2) / tan(fovy / 2)
       fx = fy * (W / H)                 # assume square pixels

    Principal point is placed at the image center:
       cx = (W - 1) / 2
       cy = (H - 1) / 2

    Returns:
        K: np.ndarray with shape (3, 3), dtype float32
    """
    fovy_deg = float(model.cam_fovy[cam_id])
    fovy = np.deg2rad(fovy_deg)

    fy = (height / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (width / float(height))

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


# ------------------------------------------------------------------------- #
#                          Env / dataset helper functions                   #
# ------------------------------------------------------------------------- #

def hide_eef_markers(env):
    """
    Hide end-effector visual markers (red dot + axis) by directly modifying
    MuJoCo site RGBA values.

    First tries to match specific site name patterns that are commonly used for
    teleoperation / EEF visualization in robosuite. If none are found, falls back
    to zeroing out all sites with non-zero RGBA.
    """
    sim = get_sim_from_env(env)
    model = sim.model

    candidate_substrings = [
        "gripper0_grip_site",
        "gripper0_grip_site_cylinder",
        "eef_site",
        "eef_cylinder",
    ]

    eef_site_ids = []
    for sid in range(model.nsite):
        name = model.site_id2name(sid)
        if name is None:
            continue
        lower_name = name.lower()
        if any(sub in lower_name for sub in candidate_substrings):
            eef_site_ids.append(sid)

    # If nothing matched, fall back to "any site that has non-zero RGBA"
    if len(eef_site_ids) == 0:
        for sid in range(model.nsite):
            if np.any(model.site_rgba[sid] != 0.0):
                eef_site_ids.append(sid)

    for sid in eef_site_ids:
        model.site_rgba[sid, :] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def get_sim_from_env(env):
    """
    Robustly retrieve the current MuJoCo simulator from a robomimic EnvBase wrapper.

    In robomimic, EnvRobosuite wraps a robosuite environment instance in `env.env`,
    and that robosuite env has the `.sim` attribute.
    """
    # Some envs might expose sim directly
    if hasattr(env, "sim"):
        return env.sim
    # Typical robosuite case: env.env.sim
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env.sim
    raise AttributeError("Could not find a MuJoCo sim on env (tried env.sim and env.env.sim).")


def build_env_for_rendering(dataset_path: str):
    """
    Build a robosuite environment from robomimic env metadata for offscreen rendering.

    Steps:
      - Read env metadata from dataset (FileUtils.get_env_metadata_from_dataset).
      - Force offscreen renderer on (has_offscreen_renderer=True, has_renderer=False).
      - Initialize ObsUtils with a minimal obs spec.
      - Create env with EnvUtils.create_env_from_metadata.

    Return:
      env      : robomimic EnvBase wrapper (EnvRobosuite)
      env_meta : env metadata dict
    """
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_kwargs = env_meta["env_kwargs"]

    # Ensure offscreen rendering is enabled
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["has_renderer"] = False

    # We do NOT require camera-based observations in the obs dict
    env_kwargs["use_camera_obs"] = False

    env_meta["env_kwargs"] = env_kwargs

    # Minimal obs spec (required to initialize ObsUtils)
    obs_spec = {
        "obs": {
            "low_dim": ["robot0_eef_pos"],
            "rgb": [],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)

    # Create environment (env_type is stored in env_meta)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
        use_depth_obs=False,  # we do not need depth
    )

    return env, env_meta


def get_all_demo_keys(hdf_file) -> list:
    """
    Return a sorted list of all demo keys in the /data group.
    Typically: ["demo_0000", "demo_0001", ...]
    """
    demos = list(hdf_file["data"].keys())
    indices = np.argsort([int(d[5:]) for d in demos])  # sort by numeric suffix
    demos = [demos[i] for i in indices]
    return demos


# ------------------------------------------------------------------------- #
#                            Main conversion logic                          #
# ------------------------------------------------------------------------- #

def main(args):
    # Basic checks
    if os.path.exists(args.out_dataset):
        raise RuntimeError(
            f"Output dataset already exists: {args.out_dataset} (refusing to overwrite)."
        )

    # Open source dataset
    f_src = h5py.File(args.dataset, "r")

    # Build env from source dataset metadata
    env, env_meta = build_env_for_rendering(args.dataset)

    # Get initial sim (we will refresh this after every reset_to)
    sim = get_sim_from_env(env)

    # Gather all demos from source
    data_src = f_src["data"]
    demo_keys = get_all_demo_keys(f_src)
    if args.num_episodes is not None:
        demo_keys = demo_keys[:args.num_episodes]

    print(f"[INFO] Found {len(demo_keys)} episodes to convert.")

    # Prepare target HDF5
    f_dst = h5py.File(args.out_dataset, "w")

    # Copy top-level attributes from source to destination
    copy_attrs(f_src, f_dst)

    # Copy all top-level groups EXCEPT "data" (we will recreate /data)
    for top_name, top_item in f_src.items():
        if top_name == "data":
            continue
        if isinstance(top_item, h5py.Group):
            g_new = f_dst.create_group(top_name)
            copy_attrs(top_item, g_new)
            copy_group_structure(top_item, g_new)
        elif isinstance(top_item, h5py.Dataset):
            d_new = f_dst.create_dataset(top_name, data=top_item[()], compression="gzip")
            copy_attrs(top_item, d_new)

    # Create /data group in destination and copy attributes
    data_dst = f_dst.create_group("data")
    copy_attrs(data_src, data_dst)

    # RNG for random cameras
    rng = np.random.default_rng(args.seed)

    # Base spherical parameters and deltas (degrees)
    base_r = args.r
    base_theta = args.theta_deg
    base_phi = args.phi_deg

    delta_r = args.delta_r
    delta_theta = args.delta_theta_deg
    delta_phi = args.delta_phi_deg

    # Target position that cameras look at (and are placed around)
    target_pos = np.array(args.target_pos, dtype=np.float32)

    # Underlying MuJoCo camera name (must exist in MJCF)
    base_camera_name = args.base_camera_name

    # Verify base_camera_name exists on the current sim
    model = sim.model
    available_cams = [model.camera_id2name(i) for i in range(model.ncam)]
    if base_camera_name not in available_cams:
        raise ValueError(
            f'Base camera "{base_camera_name}" does not exist in this model. '
            f"Available cameras = {tuple(available_cams)}"
        )
    cam_id = model.camera_name2id(base_camera_name)

    # Create a single intrinsic matrix K (same for all logical views)
    H = args.height
    W = args.width
    K = compute_intrinsics_from_mujoco(model, cam_id, W, H)

    # Logical view names
    center_view = "center"
    side_views = [f"side_{i}" for i in range(4)]   # side_0, side_1, side_2, side_3
    rand_views = [f"rand_{i}" for i in range(4)]   # rand_0, rand_1, rand_2, rand_3
    logical_views = [center_view] + side_views + rand_views

    # -------------------------------------------------------------------------
    # Compute fixed camera poses (AROUND target_pos) for center + 4 side views
    # -------------------------------------------------------------------------

    # Center camera (fixed)
    center_pos = camera_pos_around_target(base_r, base_theta, base_phi, target_pos)
    center_quat = lookat_quat(center_pos, target_pos)

    # Side cameras: using +/- delta_theta, +/- delta_phi pattern around base angles
    side_offsets = [
        (+delta_theta, +delta_phi),
        (+delta_theta, -delta_phi),
        (-delta_theta, +delta_phi),
        (-delta_theta, -delta_phi),
    ]
    side_positions = []
    side_quats = []
    for dth, dph in side_offsets:
        pos = camera_pos_around_target(base_r, base_theta + dth, base_phi + dph, target_pos)
        quat = lookat_quat(pos, target_pos)
        side_positions.append(pos)
        side_quats.append(quat)

    # Pre-build dictionary for fixed logical view poses
    fixed_view_poses = {center_view: (center_pos, center_quat)}
    for name, pos, quat in zip(side_views, side_positions, side_quats):
        fixed_view_poses[name] = (pos, quat)

    print("[INFO] Fixed camera poses computed. Starting per-episode conversion...")

    # -------------------------------------------------------------------------
    # Per-episode loop
    # -------------------------------------------------------------------------
    for ep_idx, demo_key in enumerate(demo_keys):
        print(f"[EPISODE {ep_idx}] Processing {demo_key}")

        demo_src = data_src[demo_key]

        # Create destination demo group and copy its attributes
        demo_dst = data_dst.create_group(demo_key)
        copy_attrs(demo_src, demo_dst)

        # Copy everything under this demo EXCEPT "obs"
        copy_group_structure(demo_src, demo_dst, exclude_subgroups=["obs"])

        # Create /obs subgroup in the new demo for rendered images + preserved obs
        obs_dst = demo_dst.require_group("obs")

        # If the source has an obs group, copy non-image obs entries
        if "obs" in demo_src:
            copy_obs_without_images(demo_src["obs"], obs_dst)

        # States (T, D)
        states = demo_src["states"][()]
        T = states.shape[0]

        # Create datasets for each logical view + RGB / segmentation
        image_dsets = {}
        seg_dsets = {}

        for view_name in logical_views:
            img_key = f"{view_name}_image"
            seg_key = f"{view_name}_seg"

            image_dsets[view_name] = obs_dst.create_dataset(
                img_key,
                shape=(T, H, W, 3),
                dtype=np.uint8,
                compression="gzip",
            )
            seg_dsets[view_name] = obs_dst.create_dataset(
                seg_key,
                shape=(T, H, W),
                dtype=np.int32,
                compression="gzip",
            )

        # ---------------------------------------------------------------------
        # For this episode, sample random camera poses (AROUND target_pos)
        # ---------------------------------------------------------------------
        random_view_poses = {}
        for view_name in rand_views:
            r_sample = rng.uniform(base_r - delta_r, base_r + delta_r)
            theta_sample = rng.uniform(base_theta - delta_theta, base_theta + delta_theta)
            phi_sample = rng.uniform(base_phi - delta_phi, base_phi + delta_phi)

            pos = camera_pos_around_target(r_sample, theta_sample, phi_sample, target_pos)
            quat = lookat_quat(pos, target_pos)
            random_view_poses[view_name] = (pos, quat)

        # ---------------------------------------------------------------------
        # Store per-view camera intrinsics & extrinsics metadata (OpenGL style)
        # ---------------------------------------------------------------------
        cam_params_grp = demo_dst.create_group("camera_params")
        cam_params_grp.attrs["convention"] = np.string_(
            "OpenGL-style extrinsics:"
            "R maps world -> camera, t is translation in camera frame."
        )

        for view_name in logical_views:
            if view_name in fixed_view_poses:
                pos, _quat = fixed_view_poses[view_name]
            else:
                pos, _quat = random_view_poses[view_name]

            # Compute extrinsics for this camera (world -> camera)
            R_cam, t_cam = compute_opengl_extrinsics(pos, target_pos)

            view_grp = cam_params_grp.create_group(view_name)
            view_grp.create_dataset("K", data=K.astype(np.float32))
            view_grp.create_dataset("R", data=R_cam.astype(np.float32))
            view_grp.create_dataset("t", data=t_cam.astype(np.float32))

        # ---------------------------------------------------------------------
        # Timestep loop
        # ---------------------------------------------------------------------
        for t in tqdm(range(T), desc=f"{demo_key}", leave=False):
            # Restore state at time t
            env.reset_to({"states": states[t]})

            # IMPORTANT:
            #   env.reset_to may affect the underlying robosuite env,
            #   so we MUST refresh the sim reference here to ensure
            #   we are modifying the correct simulator instance.
            sim = get_sim_from_env(env)

            # Hide teleop visualization markers (EEF red dot + axis)
            hide_eef_markers(env)

            # For each logical view, set pose of base camera and render
            for view_name in logical_views:
                if view_name in fixed_view_poses:
                    pos, quat = fixed_view_poses[view_name]
                else:
                    pos, quat = random_view_poses[view_name]

                # Set the pose of the single underlying MuJoCo camera
                set_camera_pose(sim, base_camera_name, pos, quat)

                # 1) Render RGB only (no depth, no segmentation)
                rgb = sim.render(
                    camera_name=base_camera_name,
                    width=W,
                    height=H,
                    depth=False,
                    segmentation=False,
                )
                rgb = np.asarray(rgb, dtype=np.uint8)

                # Handle possible (3, H, W) vs (H, W, 3) layouts
                if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[1] == H:
                    # Layout (3, H, W) -> transpose to (H, W, 3)
                    rgb = np.transpose(rgb, (1, 2, 0))

                # Flip vertically to match OpenCV's top-left origin
                rgb = np.flipud(rgb)

                # 2) Render segmentation mask
                seg = sim.render(
                    camera_name=base_camera_name,
                    width=W,
                    height=H,
                    depth=False,
                    segmentation=True,
                )
                seg = np.asarray(seg)

                # Typical MuJoCo / mujoco-py output for segmentation is (H, W, 2):
                #   seg[..., 0] : object ids
                #   seg[..., 1] : geom  ids
                # We keep the geom IDs as segmentation labels.
                if seg.ndim == 3 and seg.shape[-1] >= 2:
                    geom_ids = seg[..., 1]
                else:
                    # Fallback: if segmentation is single-channel, use it as-is
                    geom_ids = seg

                # Flip vertically for consistency
                geom_ids = np.flipud(geom_ids).astype(np.int32)

                # Store into HDF5 datasets under the logical view name
                image_dsets[view_name][t] = rgb
                seg_dsets[view_name][t] = geom_ids

    # Cleanly close files
    f_src.close()
    f_dst.close()
    print("[INFO] Done. New HDF5 dataset written to:", args.out_dataset)


# ------------------------------------------------------------------------- #
#                                   CLI                                     #
# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render multi-view RGB + segmentation from MimicGen / robomimic dataset into a new HDF5 file."
    )

    # Input / output datasets
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to source MimicGen / robomimic HDF5 dataset (states etc.).",
    )
    parser.add_argument(
        "--out_dataset",
        type=str,
        required=True,
        help="Path to NEW HDF5 dataset to create (must not exist).",
    )

    # How many episodes to process
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=150,
        help="Number of episodes to convert (default: 150).",
    )

    # Camera resolution
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Rendered image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Rendered image width.",
    )

    # Underlying MuJoCo camera name (must exist in MJCF)
    parser.add_argument(
        "--base_camera_name",
        type=str,
        default="agentview_full",
        help="Name of the underlying MuJoCo camera used for all logical views.",
    )

    # Spherical parameters for camera placement (all angles in DEGREES)
    parser.add_argument(
        "--r",
        type=float,
        required=True,
        help="Base radius r for center camera (meters).",
    )
    parser.add_argument(
        "--theta_deg",
        type=float,
        required=True,
        help="Base azimuth angle theta (degrees) for center camera.",
    )
    parser.add_argument(
        "--phi_deg",
        type=float,
        required=True,
        help="Base elevation angle phi (degrees) for center camera.",
    )
    parser.add_argument(
        "--delta_r",
        type=float,
        default=0.0,
        help="Radius variation for random cameras (meters).",
    )
    parser.add_argument(
        "--delta_theta_deg",
        type=float,
        default=0.0,
        help="Angular variation delta_theta (degrees) for side & random cameras.",
    )
    parser.add_argument(
        "--delta_phi_deg",
        type=float,
        default=0.0,
        help="Angular variation delta_phi (degrees) for side & random cameras.",
    )

    # Target point that all cameras look at
    parser.add_argument(
        "--target_pos",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Target point [x, y, z] that all cameras look at (meters).",
    )

    # Random seed for random cameras
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling random camera views.",
    )

    args = parser.parse_args()
    main(args)
