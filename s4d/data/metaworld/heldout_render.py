"""Near-view held-out camera sets (review item 1a): ground-truth rendering by replaying stored simulator states.

The sets perturb the six training cameras within the RL trajectory ranges (``cameras.heldout_camera_set``). Only
validation episodes are rendered; no new episodes are simulated: every frame is the stored ``qpos``/``qvel`` of the
collected episode, so the held-out views show exactly the states of the training-camera frames.

Import order matters for GPU isolation: callers must run ``enforce_allowed_gpus`` and import ``mujoco_mig_setup``
before importing this module (it imports mujoco through ``collect``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

from s4d.data.metaworld.cameras import (
    HELDOUT_PER_TRAIN_CAMERA,
    HELDOUT_RANGES,
    HELDOUT_SETS,
    heldout_camera_set,
    mujoco_free_camera,
    orbit_rig,
)
from s4d.data.metaworld.collect import DEPTH_UNIT_M, MetaworldScene, encode_depth
from s4d.data.metaworld.heldout import DATA_ROOT

VERSION = "splatter4d-metaworld-heldout-v1"
SET_NAMES = tuple(HELDOUT_SETS)
# Replay must reproduce the stored training-camera frames up to renderer noise.
MAX_RGB_MISMATCH_FRACTION = 1e-3  # pixels differing by more than 8 intensity levels
MAX_DEPTH_MISMATCH_FRACTION = 1e-3  # pixels differing by more than 1 mm


def all_heldout_cameras() -> list[dict]:
    return [camera for name in SET_NAMES for camera in heldout_camera_set(name)]


def _render(scene: MetaworldScene, cams) -> tuple[np.ndarray, np.ndarray]:
    rgb, depth = [], []
    for cam in cams:
        scene.rgb_r.update_scene(scene.data, camera=cam)
        rgb.append(scene.rgb_r.render().copy())
        scene.depth_r.update_scene(scene.data, camera=cam)
        depth.append(encode_depth(scene.depth_r.render()))
    return np.stack(rgb), np.stack(depth)


def _set_state(scene: MetaworldScene, qpos: np.ndarray, qvel: np.ndarray) -> None:
    scene.data.qpos[:] = qpos
    scene.data.qvel[:] = qvel
    mujoco.mj_forward(scene.model, scene.data)


def _set_goal(scene: MetaworldScene, episode: h5py.Group) -> None:
    """A goal marker attached to the world body is a site whose position is not part of qpos; Meta-World reports it in
    the last three observation entries, constant within an episode. Goal sites on other bodies move with qpos.
    ``replay_check`` verifies the result either way."""
    site = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
    if site < 0 or scene.model.site_bodyid[site] != 0:
        return
    goal = np.asarray(episode["obs"][:, -3:])
    if not np.allclose(goal, goal[0], atol=1e-6):
        raise ValueError("goal position changes within the episode")
    scene.model.site_pos[site] = goal[0]


def replay_check(scene: MetaworldScene, episode: h5py.Group, train_cams: np.ndarray) -> dict:
    """Re-render the training cameras at four times of the episode and compare with the stored frames."""
    length = int(episode.attrs["length"])
    _set_goal(scene, episode)
    rows = []
    for t in sorted({0, length // 3, (2 * length) // 3, length - 1}):
        _set_state(scene, episode["qpos"][t], episode["qvel"][t])
        rgb, depth = _render(scene, [scene.cams[i] for i in train_cams])
        drgb = np.abs(rgb.astype(np.int32) - episode["rgb"][t, train_cams].astype(np.int32)).max(-1)
        ddepth = np.abs(depth.astype(np.int64) - episode["depth"][t, train_cams].astype(np.int64))
        rows.append(
            {
                "t": int(t),
                "rgb_max": int(drgb.max()),
                "rgb_fraction_over_8": float((drgb > 8).mean()),
                "depth_max_m": float(ddepth.max() * DEPTH_UNIT_M),
                "depth_fraction_over_1mm": float((ddepth * DEPTH_UNIT_M > 1e-3).mean()),
            }
        )
    return {
        "rows": rows,
        "passed": all(
            r["rgb_fraction_over_8"] <= MAX_RGB_MISMATCH_FRACTION and r["depth_fraction_over_1mm"] <= MAX_DEPTH_MISMATCH_FRACTION
            for r in rows
        ),
    }


def render_heldout_sets(
    task: str, source: Path, manifest: Path, output: Path, *, max_episodes: int | None = None, log=print
) -> Path:
    """Render RGB and depth of every held-out-set camera for the validation episodes of ``manifest``."""
    output = Path(output)
    if DATA_ROOT.resolve() not in output.resolve().parents:
        raise ValueError(f"held-out sets must be written under {DATA_ROOT}")
    tmp = output.with_name(output.name + ".incomplete")
    if output.exists() or tmp.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    episodes = json.loads(Path(manifest).read_text())["validation"]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    cameras = all_heldout_cameras()
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with h5py.File(source, "r") as src:
        height, width = int(src.attrs["height"]), int(src.attrs["width"])
        if src.attrs["task"] != task:
            raise ValueError(f"{source} holds task {src.attrs['task']}, not {task}")
        scene = MetaworldScene(task, 0, height, width)
        train_cams = np.where(src["cameras"]["is_train"][:].astype(bool))[0]
        if not np.allclose(src["cameras"]["c2w"][:][train_cams], scene.rig["c2w"][train_cams], atol=1e-5):
            raise ValueError("stored training cameras differ from the rig")
        mj_cams = [mujoco_free_camera(c["azimuth"], c["elevation"], distance=c["distance"]) for c in cameras]
        rig = orbit_rig([(c["azimuth"], c["elevation"], c["distance"]) for c in cameras], height, width)
        checks = {}
        try:
            with h5py.File(tmp, "w") as f:
                f.attrs.update(
                    {
                        "version": VERSION,
                        "task": task,
                        "source": str(source),
                        "manifest": str(manifest),
                        "height": height,
                        "width": width,
                        "depth_unit_m": DEPTH_UNIT_M,
                        "sets": json.dumps(list(SET_NAMES)),
                        "set_scales_seeds": json.dumps(HELDOUT_SETS),
                        "ranges": json.dumps(HELDOUT_RANGES),
                        "per_train_camera": HELDOUT_PER_TRAIN_CAMERA,
                        "complete": False,
                    }
                )
                cams = f.create_group("cameras")
                cams.create_dataset("names", data=np.asarray([c["name"] for c in cameras], dtype=h5py.string_dtype()))
                cams.create_dataset("set", data=np.asarray([c["set"] for c in cameras], dtype=h5py.string_dtype()))
                for key in ("base", "azimuth_deg", "elevation_deg", "radius_frac", "lateral_m", "azimuth", "elevation", "distance"):
                    cams.create_dataset(key, data=np.asarray([c[key] for c in cameras]))
                for key in ("K", "c2w", "w2c"):
                    cams.create_dataset(key, data=rig[key])
                group = f.create_group("episodes")
                comp = {"compression": "gzip", "compression_opts": 4}
                for n, ep in enumerate(episodes):
                    stored = src["episodes"][ep]
                    check = replay_check(scene, stored, train_cams)
                    checks[ep] = check
                    if not check["passed"]:
                        raise RuntimeError(f"{task}/{ep}: replay does not reproduce the stored frames: {check['rows']}")
                    length = int(stored.attrs["length"])
                    rgb = np.empty((length, len(cameras), height, width, 3), dtype=np.uint8)
                    depth = np.empty((length, len(cameras), height, width), dtype=np.uint16)
                    for t in range(length):
                        _set_state(scene, stored["qpos"][t], stored["qvel"][t])
                        rgb[t], depth[t] = _render(scene, mj_cams)
                    g = group.create_group(ep)
                    g.attrs["length"] = length
                    g.create_dataset("rgb", data=rgb, chunks=(1, 1, height, width, 3), **comp)
                    g.create_dataset("depth", data=depth, chunks=(1, 1, height, width), shuffle=True, **comp)
                    f.flush()
                    log(f"[{task}] {ep} ({n + 1}/{len(episodes)}) {length} frames x {len(cameras)} cameras, "
                        f"replay max |rgb| {max(r['rgb_max'] for r in check['rows'])}, {time.time() - start:.0f}s")
                f.attrs["replay_check"] = json.dumps(checks)
                f.attrs["complete"] = True
            os.replace(tmp, output)
        finally:
            scene.close()
    return output
