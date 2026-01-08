import numpy as np
from typing import Any


def _to_uint8_image(x: Any) -> np.ndarray:
    """Ensure RGB is uint8 HxWx3."""
    arr = np.asarray(x)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).round().astype(np.uint8)
    return arr.astype(np.uint8)


def _to_int32_mask(x: Any) -> np.ndarray:
    """
    Ensure segmentation is int32 HxW.
    Notes:
      - Some render paths may return HxWx2 (objtype, objid). If so, we keep objid by default.
      - Some may return HxWx1.
    """
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        # Common MuJoCo segmentation buffer: (objtype, objid)
        arr = arr[..., 1]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise RuntimeError(f"Segmentation must be HxW (or HxWx1/HxWx2). Got {arr.shape}")
    return arr.astype(np.int32)


def _to_opencv_coords(img: np.ndarray, *, flip_lr: bool = False) -> np.ndarray:
    """
    Convert MuJoCo offscreen render buffer to OpenCV image coordinates.

    - OpenGL / MuJoCo offscreen buffers are commonly vertically flipped (origin bottom-left).
    - Some stacks (notably older mujoco_py setups) can also be mirrored L/R.

    We always flip vertically; left-right flip is optional.
    """
    arr = np.asarray(img)
    arr = np.flipud(arr)
    if flip_lr:
        arr = np.fliplr(arr)
    return np.ascontiguousarray(arr)

def render_rgb(sim, camera_name: str, H: int, W: int) -> np.ndarray:
    """
    Render RGB via robosuite's MjSim.render wrapper.
    """
    out = sim.render(width=W, height=H, camera_name=camera_name, depth=False, segmentation=False)
    if isinstance(out, tuple):
        out = out[0]
    return _to_opencv_coords(_to_uint8_image(out))


def render_segmentation(env, camera_name: str, H: int, W: int, mode: str = "instance") -> np.ndarray:
    """
    Render segmentation using robosuite's geom→instance/class mappings.

    - For most environments: behaves like the original robosuite implementation
      (instance- or class-level mask).
    - For the Wipe environment: all dirt markers share a single label id that is
      different from the table and all other objects, so they get a distinct color.
    - Additionally: TCP visualization sites (eef marker / axes) are removed from
      the segmentation so they don't show up as fake objects.

    Args:
        env: robosuite environment (RobotEnv / Wipe)
        camera_name: camera name (must exist in env.sim.model)
        H, W: image height and width
        mode: "instance" or "class"
            - "instance": each physical instance (robot0, table0, cubeA, cubeB, ...)
            - "class": semantic type (robot, table, cube, ...)

    Returns:
        HxW int32 mask in OpenCV coordinates, where each pixel is an object / class id.
        0 is reserved for background / unmapped pixels.
    """
    import mujoco  # local import so this file doesn't hard-depend at module import time

    sim = env.sim

    # ---------- Raw segmentation render: (..., 2) = (objtype, objid) ----------
    out = sim.render(
        width=W,
        height=H,
        camera_name=camera_name,
        depth=False,
        segmentation=True,
    )
    # In some setups this might be a tuple; keep the last element (seg buffer)
    if isinstance(out, tuple):
        out = out[-1]

    seg_raw = np.asarray(out)
    if seg_raw.ndim != 3 or seg_raw.shape[-1] != 2:
        raise RuntimeError(f"Expected segmentation buffer HxWx2, got {seg_raw.shape}")

    objtype = seg_raw[..., 0].astype(np.int32)
    objid = seg_raw[..., 1].astype(np.int32)

    # ---------- Mask out TCP visualization sites (eef marker / axes) ----------

    # Collect site ids for grip_site / grip_cylinder, if present
    site_ids_to_ignore = []
    for robot in getattr(env, "robots", []):
        for arm in getattr(robot, "arms", []):
            # eef_site_id / eef_cylinder_id are dicts keyed by arm ("right", "left", etc.)
            if hasattr(robot, "eef_site_id") and arm in robot.eef_site_id:
                site_ids_to_ignore.append(int(robot.eef_site_id[arm]))
            if hasattr(robot, "eef_cylinder_id") and arm in robot.eef_cylinder_id:
                site_ids_to_ignore.append(int(robot.eef_cylinder_id[arm]))

    if site_ids_to_ignore:
        site_ids_to_ignore = np.asarray(site_ids_to_ignore, dtype=np.int32)
        site_type = mujoco.mjtObj.mjOBJ_SITE

        # Pixels whose segmentation corresponds to one of these sites → background
        ignore_mask = (objtype == site_type) & np.isin(objid, site_ids_to_ignore)
        objid = objid.copy()
        objid[ignore_mask] = -1  # treat as background / unknown

    # From here on, we treat objid as "geom id per pixel" (non-geoms will map to background)
    geom_id_img = objid  # HxW
    model = env.model

    # ---------- Base mapping: geom_id -> instance / class label ----------

    if mode == "instance":
        # Each unique instance name gets its own integer id
        name2id = {inst: i for i, inst in enumerate(model.instances_to_ids.keys())}
        geom_to_label = {
            geom_id: name2id[inst_name]
            for geom_id, inst_name in model.geom_ids_to_instances.items()
        }
    elif mode == "class":
        # Each class name (robot, table, cube, ...) gets its own integer id
        name2id = {cls: i for i, cls in enumerate(model.classes_to_ids.keys())}
        geom_to_label = {
            geom_id: name2id[cls_name]
            for geom_id, cls_name in model.geom_ids_to_classes.items()
        }
    else:
        raise ValueError(f"Unknown segmentation mode '{mode}', use 'instance' or 'class'.")

    flat_geom = geom_id_img.reshape(-1)

    # Map geom ids to labels; unseen geoms get -1
    label_flat = np.fromiter(
        (geom_to_label.get(int(g), -1) for g in flat_geom),
        dtype=np.int32,
        count=flat_geom.size,
    )
    # Shift labels by +1 so that background / unknown is 0
    label_flat = label_flat + 1
    label_img = label_flat.reshape(geom_id_img.shape)  # HxW

    # ---------- Wipe special case: collapse all markers into one unique label ----------

    is_wipe_env = getattr(env, "__class__", None).__name__ == "Wipe"
    if is_wipe_env and hasattr(model, "mujoco_arena") and hasattr(model.mujoco_arena, "markers"):
        marker_geom_ids = []
        for marker in model.mujoco_arena.markers:
            # Each marker can have one or more visual geoms
            for g_name in getattr(marker, "visual_geoms", []):
                try:
                    gid = sim.model.geom_name2id(g_name)
                except Exception:
                    continue
                marker_geom_ids.append(gid)

        if marker_geom_ids:
            marker_geom_ids = np.asarray(marker_geom_ids, dtype=np.int32)

            flat_geom = geom_id_img.reshape(-1)
            label_flat = label_img.reshape(-1)

            # Pixels whose geom id belongs to any marker
            markers_mask = np.isin(flat_geom, marker_geom_ids)

            # Choose a new label id for "all markers" that does not collide with others
            marker_label = int(label_flat.max()) + 1
            label_flat[markers_mask] = marker_label

            label_img = label_flat.reshape(label_img.shape)

    # ---------- To OpenCV coords + int32 ----------

    label_img = _to_opencv_coords(_to_int32_mask(label_img))
    return label_img
